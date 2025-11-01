import argparse
import os
import time
import gc
import os.path as osp
import dgl
import dgl.nn as dglnn
import dgl.function as fn

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset, RedditDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset
from pathlib import Path
import ctypes
import ctypes.util
from collections import OrderedDict
import math
import numpy as np


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


epsilon = 1 - math.log(2)


def modified_cross_entropy(logits, labels):
    if labels.dim() == 2:
        labels = labels.squeeze(1)
    ce = F.cross_entropy(logits, labels, reduction='none')
    return (torch.log(epsilon + ce) - math.log(epsilon)).mean()


# ======================
# MmapPageCacheManager & FeatureStorage
# ======================

libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.madvise.restype = ctypes.c_int


class MmapPageCacheManager:
    def __init__(self, max_cached_bytes=10 * 1024 ** 3):
        self.max_bytes = max_cached_bytes
        self.page_size = 4096
        self.cache = OrderedDict()
        self.total_cached = 0

    def access(self, tensor):
        addr = tensor.data_ptr()
        size = tensor.untyped_storage().nbytes()
        end_addr = addr + size

        start_page = addr // self.page_size * self.page_size
        end_page = (end_addr + self.page_size - 1) // self.page_size * self.page_size

        now = time.time()
        touched_pages = []
        for page_addr in range(start_page, end_page, self.page_size):
            if page_addr in self.cache:
                self.cache.move_to_end(page_addr)
            else:
                self.cache[page_addr] = now
                self.total_cached += self.page_size
            touched_pages.append(page_addr)

        self._evict_if_needed()
        return touched_pages

    def _evict_if_needed(self):
        while self.total_cached > self.max_bytes:
            if len(self.cache) == 0:
                break
            oldest_page, _ = self.cache.popitem(last=False)
            libc.madvise(oldest_page, self.page_size, 4)  # MADV_DONTNEED
            self.total_cached -= self.page_size


class FeatureStorage:
    def __init__(self, storage_dir, device):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.mmap_tensor = None
        self.num_nodes = 0
        self.feat_dim = 0
        self.io_time = 0
        self.page_cache_manager = MmapPageCacheManager(max_cached_bytes=10 * 1024 ** 3)

    def set_path_and_load(self, feature_path, shape_path):
        feature_path = Path(feature_path)
        shape_path = Path(shape_path)
        if not feature_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {feature_path}")
        if not shape_path.exists():
            raise FileNotFoundError(f"Shape file not found: {shape_path}")
        with open(shape_path, 'r') as f:
            N, D = map(int, f.read().strip().split(','))
        self.num_nodes = N
        self.feat_dim = D
        total_size = N * D
        self.mmap_tensor = torch.from_file(
            str(feature_path),
            dtype=torch.float32,
            size=total_size,
            shared=True
        ).reshape(N, D)
        print(f"Loaded {feature_path.name} with shape ({N}, {D})")

    def get_features(self, node_ids):
        if self.mmap_tensor is None:
            raise RuntimeError("Call set_path_and_load() first.")
        t1 = time.time()
        node_ids_cpu = node_ids.cpu()
        features_view = self.mmap_tensor[node_ids_cpu]
        features_gpu = features_view.to(self.device)
        self.page_cache_manager.access(features_view)
        t2 = time.time()
        self.io_time += t2 - t1
        return features_gpu

    def get_time(self):
        temp_time = self.io_time
        self.io_time = 0
        return temp_time


# class MetaLearner(nn.Module):
#     def __init__(self, embedding_dim, num_models, num_classes, meta_hidden_dim=128, n_layers=2):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embedding_dim,
#             nhead=4,
#             dim_feedforward=256,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
#         self.classify = nn.Sequential(
#             nn.Linear(embedding_dim, meta_hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(meta_hidden_dim, num_classes)
#         )
#     def forward(self, all_model_embeddings_per_node):  # [B, M, D]
#         encoded = self.transformer(all_model_embeddings_per_node)
#         agg = encoded.mean(dim=1)
#         return self.classify(agg)
class MetaLearner(nn.Module):
    def __init__(self, embedding_dim, num_models, num_classes, meta_hidden_dim=256, n_layers=3, dropout=0.3):
        super().__init__()
        input_dim = embedding_dim * num_models

        layers = []
        in_features = input_dim

        for i in range(n_layers):
            out_features = meta_hidden_dim if i < n_layers - 1 else meta_hidden_dim // 2
            layers.append(nn.Linear(in_features, out_features))
            if i < n_layers - 1:
                layers.append(nn.LayerNorm(out_features))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            in_features = out_features

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(meta_hidden_dim // 2, num_classes)

        self.use_residual = (input_dim == meta_hidden_dim // 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, all_model_embeddings_per_node):

        x = torch.flatten(all_model_embeddings_per_node, start_dim=1)
        residual = x if self.use_residual else None

        x = self.mlp(x)
        if residual is not None:
            x = x + residual

        logits = self.classifier(x)
        return logits


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.n_layers = n_layers

        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.norms.append(nn.LayerNorm(hid_size))

        for i in range(n_layers - 2):
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
            self.norms.append(nn.LayerNorm(hid_size))

        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.norms.append(nn.LayerNorm(hid_size))

        self.classify_head = nn.Linear(hid_size, out_size)
        self.dropout = nn.Dropout(0.5)

        if in_size != hid_size:
            self.input_proj = nn.Linear(in_size, hid_size)
        else:
            self.input_proj = None

    def forward(self, blocks, x):
        h = x

        for i, (layer, block, norm) in enumerate(zip(self.layers, blocks, self.norms)):
            h_res = h

            if i == 0 and self.input_proj is not None:
                h_res = self.input_proj(h_res)

            h = layer(block, h)

            if h.shape == h_res.shape:
                h = h + h_res

            if i != self.n_layers - 1:
                h = F.relu(h)
                h = norm(h)
                h = self.dropout(h)

        embedding = h
        logits = self.classify_head(embedding)
        return logits, embedding


class GIN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.epsilons = nn.ParameterList()

        for i in range(n_layers):
            if i == 0:
                input_dim = in_size
            else:
                input_dim = hid_size

            mlp = nn.Sequential(
                nn.Linear(input_dim, hid_size),
                nn.BatchNorm1d(hid_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hid_size, hid_size),
                nn.BatchNorm1d(hid_size)
            )

            self.layers.append(dglnn.GINConv(mlp, learn_eps=True))
            self.epsilons.append(nn.Parameter(torch.tensor(0.0)))

        self.classify_head = nn.Linear(hid_size, out_size)
        self.dropout = nn.Dropout(0.5)
        self.n_layers = n_layers

    def forward(self, blocks, x):
        h = x

        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)

            if i != self.n_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)

        embedding = h
        logits = self.classify_head(embedding)
        return logits, embedding


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads=[4, 4, 4], n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.heads = heads[:n_layers]
        self.n_layers = n_layers

        self.layers.append(dglnn.GATConv(in_size, hid_size // heads[0], heads[0], activation=F.elu))
        self.norms.append(nn.LayerNorm(hid_size))

        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GATConv(hid_size, hid_size // heads[i], heads[i], activation=F.elu))
            self.norms.append(nn.LayerNorm(hid_size))

        self.layers.append(dglnn.GATConv(hid_size, hid_size, 1, activation=None))
        self.norms.append(nn.LayerNorm(hid_size))

        self.classify_head = nn.Linear(hid_size, out_size)
        self.dropout = nn.Dropout(0.5)

        if in_size != hid_size:
            self.input_proj = nn.Linear(in_size, hid_size)
        else:
            self.input_proj = None

    def forward(self, blocks, x):
        h = x

        for i, (layer, block, norm) in enumerate(zip(self.layers, blocks, self.norms)):
            h_res = h

            if i == 0 and self.input_proj is not None:
                h_res = self.input_proj(h_res)

            h = layer(block, h)

            if i != self.n_layers - 1:
                h = h.flatten(1)
            else:
                h = h.squeeze(1)

            if h.shape == h_res.shape:
                h = h + h_res

            if i != self.n_layers - 1:
                h = norm(h)
                h = self.dropout(h)

        embedding = h
        logits = self.classify_head(embedding)
        return logits, embedding


# ======================
# Training & Evaluation
# ======================

def evaluate(device, model, g, num_classes, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            blocks = [block.to(device) for block in blocks]
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            logits, _ = model(blocks, x)
            y_hats.append(logits)
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


# ======================
# Core: Collect Embeddings & Train Meta-Learner
# ======================

def collect_and_train_meta_learner(proc_id, world_size, device, num_classes,
                                   train_idx, val_idx, test_idx,
                                   model_list, meta_learner, original_g, use_uva,
                                   args, meta_learner_opt, meta_learner_scheduler):
    model = model_list[proc_id].to(device)
    model.eval()
    sampler = NeighborSampler(
        [10, 10, 10],
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    meta_train_idx = torch.cat([train_idx, val_idx]).to(device)
    test_idx = test_idx.to(device)

    my_dir = osp.dirname(osp.realpath(__file__))
    embedding_dir = osp.abspath(osp.join(my_dir, '..', '..', 'data', f'{args.dataset_name}_embeddings'))
    os.makedirs(embedding_dir, exist_ok=True)

    meta_train_embedding_file = osp.join(embedding_dir, f"model_{proc_id}_meta_train.bin")
    meta_train_shape_file = osp.join(embedding_dir, f"model_{proc_id}_meta_train.shape")
    test_embedding_file = osp.join(embedding_dir, f"model_{proc_id}_test.bin")
    test_shape_file = osp.join(embedding_dir, f"model_{proc_id}_test.shape")

    def infer_and_save(dataloader, bin_path, shape_path):
        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        embeddings_list = []

        for input_nodes, output_nodes, blocks in dataloader:
            with torch.no_grad():
                blocks = [block.to(device) for block in blocks]
                x = blocks[0].srcdata["feat"]
                _, embedding = model(blocks, x)
                embeddings_list.append(embedding.cpu())

        if len(embeddings_list) == 0:
            raise RuntimeError("No embeddings generated!")

        all_embeddings = torch.cat(embeddings_list, dim=0)
        N, D = all_embeddings.shape

        all_embeddings.numpy().tofile(bin_path)
        with open(shape_path, 'w') as f:
            f.write(f"{N},{D}")

        print(f"[Proc {proc_id}] Saved {N} embeddings to {bin_path}")
        return N, D

    torch.manual_seed(42)

    dataloader_meta_train = DataLoader(
        original_g,
        meta_train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_ddp=False,
        use_uva=use_uva,
    )
    N_meta_train, emb_dim = infer_and_save(dataloader_meta_train, meta_train_embedding_file, meta_train_shape_file)

    dataloader_test = DataLoader(
        original_g,
        test_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_ddp=False,
        use_uva=use_uva,
    )
    N_test, _ = infer_and_save(dataloader_test, test_embedding_file, test_shape_file)

    dist.barrier()

    if proc_id == 0:
        print(f"Rank 0: Training MetaLearner using embeddings from SSD...")

        if use_uva:
            meta_train_labels = original_g.ndata["label"][meta_train_idx.cpu()].to(device)
            test_labels = original_g.ndata["label"][test_idx.cpu()].to(device)
        else:
            meta_train_labels = original_g.ndata["label"][meta_train_idx]
            test_labels = original_g.ndata["label"][test_idx]

        meta_train_storages = []
        test_storages = []

        for i in range(world_size):
            storage = FeatureStorage(embedding_dir, device)
            storage.set_path_and_load(
                osp.join(embedding_dir, f"model_{i}_meta_train.bin"),
                osp.join(embedding_dir, f"model_{i}_meta_train.shape")
            )
            meta_train_storages.append(storage)

            storage_test = FeatureStorage(embedding_dir, device)
            storage_test.set_path_and_load(
                osp.join(embedding_dir, f"model_{i}_test.bin"),
                osp.join(embedding_dir, f"model_{i}_test.shape")
            )
            test_storages.append(storage_test)

        meta_learner = MetaLearner(
            embedding_dim=emb_dim,
            num_models=world_size,
            num_classes=num_classes,
            meta_hidden_dim=args.meta_hidden_dim,
            n_layers=2
        ).to(device)
        meta_learner_opt = torch.optim.Adam(meta_learner.parameters(), lr=0.001)
        meta_learner_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_learner_opt, T_max=args.meta_epochs)

        meta_learner.train()
        batch_size = min(args.batch_size, 4096)

        print(f"Training MetaLearner for {args.meta_epochs} epochs...")

        for meta_epoch in range(args.meta_epochs):
            total_loss = 0.0
            num_batches = 0

            for start in range(0, len(meta_train_idx), batch_size):
                end = min(start + batch_size, len(meta_train_idx))

                batch_embeddings = []
                for storage in meta_train_storages:
                    emb = storage.mmap_tensor[start:end].to(device)
                    batch_embeddings.append(emb)

                meta_input = torch.stack(batch_embeddings, dim=1)
                labels = meta_train_labels[start:end]

                meta_learner_opt.zero_grad()
                logits = meta_learner(meta_input)
                loss = F.cross_entropy(logits, labels.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(meta_learner.parameters(), max_norm=1.0)
                meta_learner_opt.step()
                if meta_learner_scheduler:
                    meta_learner_scheduler.step()

                total_loss += loss.item()
                num_batches += 1

            if (meta_epoch + 1) % 10 == 0 or meta_epoch == args.meta_epochs - 1:
                avg_loss = total_loss / num_batches
                print(f"Meta-Epoch {meta_epoch + 1:03d}/{args.meta_epochs} | Avg Loss {avg_loss:.4f}")

        meta_learner.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for start in range(0, len(test_idx), batch_size):
                end = min(start + batch_size, len(test_idx))

                batch_embeddings = []
                for storage in test_storages:
                    emb = storage.mmap_tensor[start:end].to(device)
                    batch_embeddings.append(emb)

                meta_input = torch.stack(batch_embeddings, dim=1)
                logits = meta_learner(meta_input)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds)
                all_labels.append(test_labels[start:end])

        final_preds = torch.cat(all_preds, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        acc = MF.accuracy(final_preds, final_labels, task="multiclass", num_classes=num_classes)
        print(f" Final Meta-Learner Test Accuracy: {acc.item():.4f}")

        if args.cleanup_embeddings:
            print(" Cleaning up temporary embedding files...")
            for i in range(world_size):
                for suffix in ["_meta_train", "_test"]:
                    for ext in [".bin", ".shape"]:
                        f = osp.join(embedding_dir, f"model_{i}{suffix}{ext}")
                        if osp.exists(f):
                            try:
                                os.remove(f)
                                print(f"Removed {f}")
                            except Exception as e:
                                print(f"Failed to remove {f}: {e}")
            print("Cleanup done.")


def train(proc_id, nprocs, device, args, g, num_classes, train_idx, test_idx, model, use_uva, signal_file):
    dist.barrier()
    if proc_id == 0:
        start_time = time.time()
    sampler = NeighborSampler(
        [10, 10, 10],
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        use_ddp=True,
        use_uva=use_uva,
    )
    # opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)
    for epoch in range(args.num_epochs):
        if os.path.exists(signal_file):
            print(f"[Rank {proc_id}] Stop signal detected at epoch {epoch}, breaking.")
            break

        model.train()
        total_loss = 0
        if epoch < 30:
            lr = 0.003 * (epoch + 1) / 30
            for param_group in opt.param_groups:
                param_group['lr'] = lr
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            blocks = [b.to(device) for b in blocks]
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"].long()
            logits, _ = model(blocks, x)
            # loss = F.cross_entropy(logits, y)
            loss = modified_cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # scheduler.step()
            total_loss += loss.item()
        # acc_local = evaluate(device, model, g, num_classes, val_dataloader)

        # print(
        #     f"[Proc {proc_id}] Epoch {epoch:05d} | Loss {total_loss / (it + 1):.4f} | "
        #     f"Local Val Accuracy {acc_local.item():.4f}"
        # )

    if proc_id == 0:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n Total training time: {total_time:.2f} seconds \n")

        val_dataloader = DataLoader(
            g,
            test_idx,
            sampler,
            device=device,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            use_ddp=False,
            use_uva=use_uva,
        )
        acc_local = evaluate(device, model, g, num_classes, val_dataloader)
        with open(signal_file, "w") as f:
            f.write("stop\n")
        print(f"[Proc {proc_id}] Local Val Accuracy: {acc_local.item():.4f}")


def run(proc_id, nprocs, devices, g_original, data, args, signal_file):
    set_seed(42)
    device = devices[proc_id]
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=nprocs,
        rank=proc_id,
    )

    num_classes, train_idx_global, val_idx_global, test_idx_global = data

    g = g_original
    if args.mode == "puregpu":
        g = g.to(device)
    else:
        g = g.to("cpu")

    in_size = g.ndata["feat"].shape[1]
    model = SAGE(in_size, args.hidden_dim, num_classes).to(device)
    # if nprocs == 1:
    #     model = SAGE(in_size, args.hidden_dim, num_classes).to(device)
    # else:
    #     model_choices = [SAGE, GAT,GIN]
    #     ModelClass = model_choices[proc_id % len(model_choices)]
    #     model = ModelClass(in_size, args.hidden_dim, num_classes).to(device)
    #     print(f"[Proc {proc_id}] Using {ModelClass.__name__}")

    train(
        proc_id,
        nprocs,
        device,
        args,
        g,
        num_classes,
        train_idx_global,
        val_idx_global,
        model,
        args.mode == "mixed",
        signal_file
    )

    if args.mode == "puregpu":
        g_original = g_original.to(device)

    model_list = [None] * nprocs
    model_list[proc_id] = model

    test_idx_for_meta = test_idx_global.to(device) if args.mode != "benchmark" else test_idx_global
    dist.barrier()

    if proc_id == 0:
        if os.path.exists(signal_file):
            os.remove(signal_file)
        print("Starting Meta-Learning phase...")
        meta_time_start = time.time()
    collect_and_train_meta_learner(
        proc_id, nprocs, device, num_classes,
        train_idx_global,
        val_idx_global,
        test_idx_for_meta,
        model_list, meta_learner=None, original_g=g_original,
        use_uva=(args.mode == "mixed"),
        args=args,
        meta_learner_opt=None,
        meta_learner_scheduler=None
    )
    if proc_id == 0:
        meta_time = time.time() - meta_time_start
        print(f"\n Meta_train_time:{meta_time:.2f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["mixed", "puregpu", "benchmark"],
        help="Training mode.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0,1,2",
        help="GPU(s) in use.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs for base GNN models train.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ogbn-arxiv",
        help="Dataset name.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Root directory of dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for training and inference.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of SAGE model.",
    )
    parser.add_argument(
        "--meta_epochs",
        type=int,
        default=30,
        help="Number of epochs for MetaLearner train.",
    )
    parser.add_argument(
        "--meta_lr",
        type=float,
        default=0.01,
        help="Learning rate for MetaLearner.",
    )
    parser.add_argument(
        "--meta_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for the MetaLearner's MLP.",
    )
    parser.add_argument(
        "--cleanup_embeddings",
        action="store_true",
        help="Whether to delete temporary embedding files after meta-training.",
    )
    args = parser.parse_args()
    devices = list(map(int, args.gpu.split(",")))
    nprocs = len(devices)
    assert torch.cuda.is_available(), "Must have GPUs."

    print(f"Training in {args.mode} mode using {nprocs} GPU(s)")

    print("Loading data")
    my_dir = osp.dirname(osp.realpath(__file__))
    path = osp.abspath(osp.join(my_dir, '..', '..', 'data'))
    signal_file = osp.join(path, f"{args.dataset_name}_stop.signal")

    if args.dataset_name in ['ogbn-arxiv', 'ogbn-papers100M']:
        dataset = AsNodePredDataset(
            DglNodePropPredDataset(args.dataset_name, root=path)
        )
    elif args.dataset_name == 'ogbn-products':
        dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset_name, root=path), split_ratio=[0.4, 0.1, 0.5])
    elif args.dataset_name in ['cora']:
        dataset = AsNodePredDataset(CoraGraphDataset(raw_dir=path))
    elif args.dataset_name in ['reddit']:
        dataset = AsNodePredDataset(RedditDataset(raw_dir=path))
    elif args.dataset_name == 'coauthor_cs':
        raw_dataset = CoauthorCSDataset(raw_dir=path)
        dataset = AsNodePredDataset(raw_dataset, split_ratio=[0.4, 0.1, 0.5])
    elif args.dataset_name == 'coauthor_physics':
        raw_dataset = CoauthorPhysicsDataset(raw_dir=path)
        dataset = AsNodePredDataset(raw_dataset, split_ratio=[0.4, 0.1, 0.5])

    g = dataset[0]

    g.create_formats_()
    if args.dataset_name == "ogbn-arxiv":
        g = dgl.to_bidirected(g, copy_ndata=True)
        g = dgl.add_self_loop(g)

    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)
    data = (
        dataset.num_classes,
        dataset.train_idx,
        dataset.val_idx,
        dataset.test_idx,
    )

    mp.spawn(
        run,
        args=(nprocs, devices, g, data, args, signal_file),
        nprocs=nprocs,
    )