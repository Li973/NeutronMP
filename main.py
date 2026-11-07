import argparse
import os.path as osp

import dgl
import torch
import torch.distributed as dist

from neutronmp.datasets import SUPPORTED_DATASETS, load_dataset
from neutronmp.distributed import (
    broadcast_partition,
    broadcast_pruning_config,
    gather_models,
    get_dist_info,
    init_distributed,
    resolve_device,
)
from neutronmp.fusion import build_embedding_cache
from neutronmp.integration import cleanup_embedding_cache, train_full_integration, train_pruned_integration
from neutronmp.partitioning import (
    build_partition_graph,
    compute_partition_pagerank,
    expand_subgraph_nodes,
    map_to_local_core,
    topology_aware_hard_partition,
)
from neutronmp.pruning import (
    assign_nodes_to_families,
    metis_partition_families,
    select_experts_per_family,
)
from neutronmp.storage import FeatureStorage
from neutronmp.submodel import train_submodel


def parse_args():
    parser = argparse.ArgumentParser(description="NeutronMP: multi-model parallelism for distributed GNN training")
    parser.add_argument("--dataset_name", type=str, default="cora", choices=SUPPORTED_DATASETS)
    parser.add_argument("--train_duration", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--meta_epochs", type=int, default=30)
    parser.add_argument("--meta_hidden_dim", type=int, default=128)
    parser.add_argument("--cleanup_embeddings", action="store_true")
    parser.add_argument("--exact_fusion", action="store_true")
    parser.add_argument("--enable_pruning", action="store_true")
    parser.add_argument("--num_families", type=int, default=2)
    parser.add_argument("--num_experts_per_node", type=int, default=2)
    return parser.parse_args()


def run_pruning(graph, node_part, nprocs, args):
    partition_graph = build_partition_graph(graph, node_part, nprocs)
    hotspot_order, pagerank_values = compute_partition_pagerank(partition_graph)
    families, family_assign = metis_partition_families(partition_graph, args.num_families)
    expert_selection = select_experts_per_family(
        families, hotspot_order, pagerank_values, args.num_experts_per_node
    )
    node_family = assign_nodes_to_families(node_part, family_assign)
    return family_assign.cpu(), node_family.cpu(), expert_selection


def build_local_training_view(graph, node_part, rank, n_layers):
    core_nodes = torch.where(node_part == rank)[0]
    expanded_nodes = expand_subgraph_nodes(graph, core_nodes, n_layers)
    subgraph = dgl.node_subgraph(graph, expanded_nodes)
    subgraph.ndata["orig_id"] = expanded_nodes
    return core_nodes, expanded_nodes, subgraph


def main():
    world_size, rank, local_rank = get_dist_info()
    device = resolve_device(local_rank)
    init_distributed(world_size, rank, device)
    args = parse_args()

    nprocs = world_size
    if nprocs == 1:
        args.num_families = 1
        args.num_experts_per_node = 1

    if args.enable_pruning and (args.num_experts_per_node > nprocs or args.num_families > nprocs):
        raise ValueError(f"Pruning parameters are invalid for nprocs={nprocs}")

    root = osp.join(osp.dirname(osp.abspath(__file__)), "dataset")
    barrier = dist.barrier if world_size > 1 else None
    dataset, graph, split = load_dataset(args.dataset_name, root, rank, world_size, barrier)
    num_classes, train_idx, val_idx, test_idx = split
    del dataset

    node_part = topology_aware_hard_partition(graph, train_idx, nprocs) if rank == 0 else None
    node_part = broadcast_partition(node_part, graph.num_nodes(), world_size, rank, device)

    node_family = None
    expert_selection = None
    if args.enable_pruning:
        if world_size > 1:
            dist.barrier()
        if rank == 0:
            _, node_family, expert_selection = run_pruning(graph, node_part, nprocs, args)
            config_list = [node_family, expert_selection]
        else:
            config_list = [None, None]
        if world_size > 1:
            node_family, expert_selection = broadcast_pruning_config(config_list, world_size, rank)

    graph_n_layers = 3
    core_nodes, expanded_nodes, subgraph = build_local_training_view(graph, node_part, rank, graph_n_layers)
    local_train_idx = map_to_local_core(train_idx, expanded_nodes)
    print(f"[Rank {rank}] core={len(core_nodes)} expanded={subgraph.num_nodes()} train={len(local_train_idx)}")

    model, head = train_submodel(
        rank, nprocs, device, args, subgraph, num_classes, local_train_idx, args.train_duration
    )
    local_data = {
        "model": {key: value.cpu() for key, value in model.state_dict().items()},
        "head": {key: value.cpu() for key, value in head.state_dict().items()},
    }

    all_models = gather_models(local_data, world_size, rank)
    if all_models is None:
        return

    embedding_dir, embedding_file, shape_file, num_nodes, feat_dim = build_embedding_cache(
        graph,
        all_models,
        nprocs,
        args.hidden_dim,
        args.batch_size,
        device,
        args.dataset_name,
        exact_fusion=args.exact_fusion,
        n_layers=graph_n_layers,
    )
    print(f"[Rank 0] embeddings=({num_nodes}, {feat_dim})")

    storage = FeatureStorage(embedding_dir, device)
    storage.set_path_and_load(embedding_file, shape_file)
    labels = graph.ndata["label"]
    train_val_idx = torch.cat([train_idx, val_idx])

    if args.enable_pruning and expert_selection is not None:
        test_acc = train_pruned_integration(
            storage, labels, train_val_idx, val_idx, test_idx, num_classes,
            node_family, expert_selection, args, device,
        )
        density = args.num_experts_per_node / nprocs * 100 if nprocs else 0.0
        print(f"pruned_test_acc={test_acc:.4f} experts={args.num_experts_per_node}/{nprocs} "
              f"pruned={100 - density:.1f}% families={args.num_families}")
    else:
        test_acc = train_full_integration(
            storage, labels, train_val_idx, val_idx, test_idx, num_classes, args, device
        )
        print(f"test_acc={test_acc:.4f}")

    if args.cleanup_embeddings:
        cleanup_embedding_cache(embedding_dir)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

