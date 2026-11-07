import os
import os.path as osp
import tempfile

import dgl.function as fn
import torch
import torch.nn.functional as F
from dgl.dataloading import DataLoader, NeighborSampler

from .models import SAGE


def _submodel_neighbor_weights(all_models, nprocs, n_layers, hidden_dim, device):
    weights = []
    for rank in range(nprocs):
        layer_weights = []
        for layer in range(n_layers):
            key = f"layers.{layer}.fc_neigh.weight"
            if key not in all_models[rank]:
                key = f"layers.{layer}.lin_l.weight"
            layer_weights.append(all_models[rank][key].to(device))
        weights.append(layer_weights)
    return weights


def _shared_aggregation_embeddings(graph, all_models, nprocs, n_layers, hidden_dim, batch_size, device, embedding_dir):
    submodel_weights = _submodel_neighbor_weights(all_models, nprocs, n_layers, hidden_dim, device)
    sampler = NeighborSampler([10, 10, 10], prefetch_node_feats=["feat"])
    dataloader = DataLoader(
        graph,
        torch.arange(graph.num_nodes()),
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )

    embeddings = []
    with torch.no_grad():
        for batch_idx, (_, _, blocks) in enumerate(dataloader):
            h = blocks[0].srcdata["feat"]
            for layer in range(n_layers):
                block = blocks[layer]
                with block.local_scope():
                    block.srcdata["h"] = h
                    block.update_all(fn.copy_u("h", "m"), fn.mean("m", "agg"))
                    aggregated = block.dstdata["agg"]

                if layer == 0:
                    branch_outputs = [
                        F.relu(F.linear(aggregated, submodel_weights[k][layer]))
                        if layer < n_layers - 1
                        else F.linear(aggregated, submodel_weights[k][layer])
                        for k in range(nprocs)
                    ]
                else:
                    branch_outputs = []
                    for k in range(nprocs):
                        branch = aggregated[:, k * hidden_dim:(k + 1) * hidden_dim]
                        out = F.linear(branch, submodel_weights[k][layer])
                        if layer < n_layers - 1:
                            out = F.relu(out)
                        branch_outputs.append(out)
                h = torch.cat(branch_outputs, dim=1)

                if layer == n_layers - 1:
                    embeddings.append(h.cpu())
            if batch_idx % 100 == 0:
                print(f"[fusion] batch {batch_idx}")
    return torch.cat(embeddings, dim=0)


def _exact_embeddings(graph, all_models, nprocs, hidden_dim, n_layers, batch_size, device):
    in_size = graph.ndata["feat"].shape[1]
    models = []
    for rank in range(nprocs):
        model = SAGE(in_size, hidden_dim, n_layers).to(device)
        model.load_state_dict(all_models[rank])
        model.eval()
        models.append(model)

    sampler = NeighborSampler([10, 10, 10], prefetch_node_feats=["feat"])
    dataloader = DataLoader(
        graph,
        torch.arange(graph.num_nodes()),
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )

    embeddings = []
    with torch.no_grad():
        for batch_idx, (_, _, blocks) in enumerate(dataloader):
            x = blocks[0].srcdata["feat"]
            h = torch.cat([model(blocks, x) for model in models], dim=1)
            embeddings.append(h.cpu())
            if batch_idx % 100 == 0:
                print(f"[fusion] batch {batch_idx}")
    return torch.cat(embeddings, dim=0)


def build_embedding_cache(graph, all_models, nprocs, hidden_dim, batch_size, device, dataset_name, exact_fusion=False, n_layers=3):
    embedding_dir = osp.abspath(osp.join(tempfile.gettempdir(), f"{dataset_name}_embeddings"))
    os.makedirs(embedding_dir, exist_ok=True)
    embedding_file = osp.join(embedding_dir, "fused_embeddings.bin")
    shape_file = osp.join(embedding_dir, "fused_embeddings.shape")

    if exact_fusion:
        embeddings = _exact_embeddings(graph, all_models, nprocs, hidden_dim, n_layers, batch_size, device)
    else:
        embeddings = _shared_aggregation_embeddings(
            graph, all_models, nprocs, n_layers, hidden_dim, batch_size, device, embedding_dir
        )

    num_nodes, feat_dim = embeddings.shape
    embeddings.numpy().tofile(embedding_file)
    with open(shape_file, "w") as handle:
        handle.write(f"{num_nodes},{feat_dim}")

    del embeddings
    return embedding_dir, embedding_file, shape_file, num_nodes, feat_dim
