import argparse
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset, RedditDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset
import os.path as osp
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
    # labels: [N, 1] or [N,]
    if labels.dim() == 2:
        labels = labels.squeeze(1)
    ce = F.cross_entropy(logits, labels, reduction='none')
    return (torch.log(epsilon + ce) - math.log(epsilon)).mean()


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()  # 新增：每层后加 LayerNorm
        self.n_layers = n_layers

        # 第一层
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.norms.append(nn.LayerNorm(hid_size))

        # 中间层
        for i in range(n_layers - 2):
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
            self.norms.append(nn.LayerNorm(hid_size))

        # 最后一层（输出 embedding 维度 = hid_size）
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.norms.append(nn.LayerNorm(hid_size))

        # 分类头
        self.classify_head = nn.Linear(hid_size, out_size)
        self.dropout = nn.Dropout(0.5)  # 只在训练分类头前用

        # 可选：输入投影（如果 in_size != hid_size，加残差用）
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


def train(proc_id, nprocs, device, args, g, num_classes, train_idx, test_idx, model, use_uva):
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

    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)

    for epoch in range(args.num_epochs):
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
            loss = modified_cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

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
        print(f"[Proc {proc_id}] Local Val Accuracy: {acc_local.item():.4f}")

    dist.destroy_process_group()


def run(proc_id, nprocs, devices, g_original, data, args):
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
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device],
                                                      find_unused_parameters=True)
    # if nprocs == 1:
    #     model = SAGE(in_size, args.hidden_dim, num_classes).to(device)
    # else:
    #     model_choices = [SAGE, GAT,GIN]
    #     ModelClass = model_choices[proc_id % len(model_choices)]
    #     model = ModelClass(in_size, args.hidden_dim, num_classes).to(device)
    #     print(f"[Proc {proc_id}] Using {ModelClass.__name__}")
    # if proc_id == 0:
    #     print("Training base GNN models...")

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
    )


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
        args=(nprocs, devices, g, data, args),
        nprocs=nprocs,
    )