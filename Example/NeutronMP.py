import argparse
import os
import time
import os.path as osp
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset, RedditDataset
from pathlib import Path
import numpy as np
import math
import tempfile
import pickle
import warnings
import ctypes
import ctypes.util
from collections import OrderedDict
import networkx as nx
import io

warnings.filterwarnings("ignore", category=FutureWarning)



def get_dist_info():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return world_size, rank, local_rank


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



def serialize_model_state(state_dict):
    buffer = io.BytesIO()
    pickle.dump(state_dict, buffer)
    byte_data = buffer.getvalue()
    byte_tensor = torch.ByteTensor(list(byte_data))
    return byte_tensor


def deserialize_model_state(byte_tensor):
    byte_data = bytes(byte_tensor.tolist())
    buffer = io.BytesIO(byte_data)
    state_dict = pickle.load(buffer)
    return state_dict


def send_model_state(model_state, dst_rank):
    byte_tensor = serialize_model_state(model_state)
    size_tensor = torch.LongTensor([len(byte_tensor)])
    dist.send(size_tensor, dst=dst_rank)
    dist.send(byte_tensor, dst=dst_rank)
    print(f"  Sent {len(byte_tensor)} bytes")


def recv_model_state(src_rank):
    size_tensor = torch.LongTensor([0])
    dist.recv(size_tensor, src=src_rank)
    data_size = size_tensor[0].item()
    byte_tensor = torch.ByteTensor(data_size)
    dist.recv(byte_tensor, src=src_rank)
    model_state = deserialize_model_state(byte_tensor)
    print(f"  Received {data_size} bytes from Rank {src_rank}")
    return model_state



epsilon = 1 - math.log(2)


def modified_cross_entropy(logits, labels):
    if labels.dim() == 2:
        labels = labels.squeeze(1)
    ce = F.cross_entropy(logits, labels, reduction='none')
    ce = torch.clamp(ce, min=1e-9)
    return (torch.log(epsilon + ce) - math.log(epsilon)).mean()


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
        for page_addr in range(start_page, end_page, self.page_size):
            if page_addr in self.cache:
                self.cache.move_to_end(page_addr)
            else:
                self.cache[page_addr] = now
                self.total_cached += self.page_size
        self._evict_if_needed()

    def _evict_if_needed(self):
        while self.total_cached > self.max_bytes and len(self.cache) > 0:
            oldest_page, _ = self.cache.popitem(last=False)
            libc.madvise(oldest_page, self.page_size, 4)
            self.total_cached -= self.page_size


class FeatureStorage:
    def __init__(self, storage_dir, device):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.mmap_tensor = None
        self.num_nodes = 0
        self.feat_dim = 0
        self.page_cache_manager = MmapPageCacheManager(max_cached_bytes=10 * 1024 ** 3)

    def set_path_and_load(self, feature_path, shape_path):
        feature_path = Path(feature_path)
        shape_path = Path(shape_path)
        with open(shape_path, 'r') as f:
            N, D = map(int, f.read().strip().split(','))
        self.num_nodes = N
        self.feat_dim = D
        self.mmap_tensor = torch.from_file(
            str(feature_path), dtype=torch.float32,
            size=N * D, shared=True
        ).reshape(N, D)
        print(f"Loaded {feature_path.name} with shape ({N}, {D})")


class MetaLearner(nn.Module):
    def __init__(self, input_dim, num_classes, meta_hidden_dim=256, n_layers=3, dropout=0.3):
        super().__init__()
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        return self.classifier(x)


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, n_layers=3):
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
        self.dropout = nn.Dropout(0.5)
        self.input_proj = nn.Linear(in_size, hid_size) if in_size != hid_size else None

    def forward(self, blocks, x):
        h = x
        for i, (layer, block, norm) in enumerate(zip(self.layers, blocks, self.norms)):
            h_res = self.input_proj(h) if i == 0 and self.input_proj else h
            h = layer(block, h)
            if h.shape == h_res.shape:
                h = h + h_res
            if i != self.n_layers - 1:
                h = F.relu(h)
            h = norm(h)
            h = self.dropout(h)
        return h


def topology_aware_hard_partition(g, train_idx, world_size):
    if world_size == 1:
        print("✓ Skipping METIS for single-process mode")
        return torch.zeros(g.num_nodes(), dtype=torch.long)
    try:
        from dgl.partition import metis_partition_assignment
        node_part = metis_partition_assignment(g, world_size)
        print(f"✓ METIS partitioning succeeded")
    except Exception as e:
        print(f"METIS failed ({e}), using balanced partitioning")
        return train_aware_balanced_partition(g, train_idx, world_size)

    min_train = min((node_part[train_idx] == p).sum().item() for p in range(world_size))
    if min_train == 0:
        print(f"⚠ Rebalancing empty partitions")
        return train_aware_balanced_partition(g, train_idx, world_size)
    return node_part


def train_aware_balanced_partition(g, train_idx, world_size):
    N = g.num_nodes()
    node_part = torch.full((N,), -1, dtype=torch.long)
    train_nodes = train_idx[torch.randperm(len(train_idx))]
    for i, node in enumerate(train_nodes):
        node_part[node] = i % world_size
    remaining = torch.where(node_part == -1)[0]
    for node in remaining:
        part_sizes = [(node_part == p).sum().item() for p in range(world_size)]
        node_part[node] = min(range(world_size), key=lambda p: part_sizes[p])
    return node_part


def expand_subgraph_nodes(g, core_nodes, k_hops):
    if k_hops == 0:
        return core_nodes

    current_frontier = core_nodes.unique()
    all_nodes_set = set(current_frontier.cpu().numpy())

    print(f"  [Expansion] Starting with {len(all_nodes_set)} core nodes.")

    # 逐层扩展
    for i in range(k_hops):
        if len(current_frontier) == 0:
            break
        # 获取当前前沿节点的入邻居 (因为 GNN 是聚合邻居信息到中心节点)
        # DGL 中 edge (u, v) 表示 u -> v，我们需要 v 的邻居 u，所以取 src (index 0)
        neighbors = g.in_edges(current_frontier)[0]
        new_nodes = neighbors.unique()

        # 过滤掉已经包含在内的节点
        new_nodes_np = new_nodes.cpu().numpy()
        added_nodes = [n for n in new_nodes_np if n not in all_nodes_set]

        if not added_nodes:
            print(f"  [Expansion Layer {i + 1}] No new nodes found. Stopping expansion.")
            break

        all_nodes_set.update(added_nodes)
        current_frontier = torch.tensor(added_nodes, dtype=torch.long, device=core_nodes.device)
        print(f"  [Expansion Layer {i + 1}] Added {len(added_nodes)} new boundary nodes. Total: {len(all_nodes_set)}")

    return torch.tensor(list(all_nodes_set), dtype=torch.long, device=core_nodes.device)


def build_partition_graph(g_original, node_part, nprocs):
    print("Building partition graph...")
    edge_counts = {}
    src, dst = g_original.edges()
    src_part = node_part[src]
    dst_part = node_part[dst]
    for s, d in zip(src_part.cpu().numpy(), dst_part.cpu().numpy()):
        if s != d:
            key = (min(s, d), max(s, d))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    pg = nx.Graph()
    for i in range(nprocs):
        pg.add_node(i)
    for (i, j), weight in edge_counts.items():
        pg.add_edge(i, j, weight=weight)
    print(f"✓ Partition graph: {pg.number_of_nodes()} nodes, {pg.number_of_edges()} edges")
    return pg


def compute_partition_pagerank(partition_graph, damping=0.85, max_iter=100):
    print("Computing PageRank for partition importance...")
    n_nodes = partition_graph.number_of_nodes()
    if n_nodes <= 1:
        node = list(partition_graph.nodes())[0] if n_nodes == 1 else 0
        return [node], {node: 1.0}
    try:
        pagerank = nx.pagerank(partition_graph, weight='weight', max_iter=max_iter, tol=1e-6)
    except Exception as e:
        print(f" PageRank failed ({e}), using uniform distribution")
        pagerank = {i: 1.0 / n_nodes for i in range(n_nodes)}
    sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    hotspot_order = [p[0] for p in sorted_pr]
    pagerank_values = {p[0]: p[1] for p in sorted_pr}
    print(f" Top-5 hotspot partitions: {hotspot_order[:5]}")
    return hotspot_order, pagerank_values


def metis_partition_families(partition_graph, num_families):
    print(f"Clustering {partition_graph.number_of_nodes()} partitions into {num_families} families...")
    n = partition_graph.number_of_nodes()
    if num_families >= n or n == 1:
        family_assign = torch.zeros(n, dtype=torch.long)
        families = {0: list(range(n))}
        print(f"  Family 0: {n} partitions (all, trivial clustering)")
        return families, family_assign
    edges = list(partition_graph.edges())
    if len(edges) > 0:
        src_list = [e[0] for e in edges]
        dst_list = [e[1] for e in edges]
        pg_dgl = dgl.graph((src_list, dst_list), num_nodes=n)
        try:
            from dgl.partition import metis_partition_assignment
            family_assign = metis_partition_assignment(pg_dgl, num_families)
            print(f" METIS family clustering succeeded")
        except Exception as e:
            print(f" METIS failed ({e}), using greedy clustering")
            family_assign = torch.zeros(n, dtype=torch.long)
            for i in range(n):
                family_assign[i] = i % num_families
    else:
        family_assign = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            family_assign[i] = i % num_families
    families = {f: [] for f in range(num_families)}
    for node, family in enumerate(family_assign.cpu().numpy()):
        families[family].append(node)
    for f, members in families.items():
        if len(members) > 0:
            print(f"  Family {f}: {len(members)} partitions {members[:8]}")
    return families, family_assign


def select_experts_per_family(families, hotspot_order, pagerank_values, num_experts_per_node, nprocs):
    print(f"\nSelecting experts per family (target={num_experts_per_node})...")
    expert_selection = {}
    for family_id, members in families.items():
        members_set = set(members)
        num_members = len(members)
        if num_members == 0:
            selected = hotspot_order[:num_experts_per_node] if len(hotspot_order) > 0 else [0]
        elif num_members > num_experts_per_node:
            members_sorted = sorted(members, key=lambda x: pagerank_values.get(x, 0), reverse=True)
            selected = members_sorted[:num_experts_per_node]
        else:
            selected = list(members)
        for p in hotspot_order:
            if p not in members_set and len(selected) < num_experts_per_node:
                selected.append(p)
        if len(selected) == 0:
            selected = [0]
        expert_selection[family_id] = selected
        print(f"  Family {family_id}: {len(selected)} experts {selected}")
    return expert_selection


def assign_nodes_to_families(node_part, family_assign):
    node_family = family_assign[node_part]
    return node_family


def train_submodel(rank, world_size, device, args, subg, num_classes, local_train_idx, train_duration):
    set_seed(42 + rank)
    in_size = subg.ndata["feat"].shape[1]
    model = SAGE(in_size, args.hidden_dim).to(device)
    classify_head = nn.Linear(args.hidden_dim, num_classes).to(device)
    if len(local_train_idx) == 0:
        print(f"[Rank {rank}] WARNING: Empty training partition!")
        return model, classify_head

    opt = torch.optim.AdamW(list(model.parameters()) + list(classify_head.parameters()), lr=0.003, weight_decay=0)
    sampler = NeighborSampler([10, 10, 10], prefetch_node_feats=["feat"], prefetch_labels=["label"])
    dataloader = DataLoader(
        subg, local_train_idx, sampler, device=device, batch_size=args.batch_size,
        shuffle=True, drop_last=False, num_workers=0, use_uva=True
    )
    model.train()
    start_time = time.time()
    epoch = 0
    while time.time() - start_time < train_duration:
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device) for b in blocks]
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"].long()
            emb = model(blocks, x)
            logits = classify_head(emb)
            loss = modified_cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch += 1
    duration = time.time() - start_time
    print(f"[Rank {rank}] Training: {duration:.2f}s ({epoch} epochs) | Train: {len(local_train_idx)}")
    return model, classify_head


def main():
    # 步骤 1: 获取分布式信息
    world_size, rank, local_rank = get_dist_info()

    # 步骤 2: 设置设备
    if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        use_gpu = True
    else:
        device = torch.device('cpu')
        use_gpu = False

    # 步骤 3: 自动选择后端
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    is_multinode = master_addr not in ['127.0.0.1', 'localhost']

    if world_size > 1:
        backend = 'gloo' if is_multinode else 'nccl'
        broadcast_device = device if backend == 'nccl' else torch.device('cpu')
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        print(f"[Rank {rank}] Initialized: world_size={world_size}, device={device}, backend={backend}")
    else:
        broadcast_device = device
        backend = 'gloo'
        print(f"[Rank {rank}] Running in single-process mode")

    # 步骤 4: 解析参数
    parser = argparse.ArgumentParser(description="NeutronEP: GNN Ensemble Parallelism")
    parser.add_argument("--dataset_name", type=str, default="cora", choices=["cora", "reddit", "ogbn-arxiv", "ogbn-products", "ogbn-papers100M"])
    parser.add_argument("--train_duration", type=float, default=300.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--meta_epochs", type=int, default=30)
    parser.add_argument("--meta_hidden_dim", type=int, default=128)
    parser.add_argument("--cleanup_embeddings", action="store_true")
    parser.add_argument("--enable_pruning", action="store_true")
    parser.add_argument("--num_families", type=int, default=4)
    parser.add_argument("--num_experts_per_node", type=int, default=8)
    args = parser.parse_args()

    nprocs = world_size
    # 单进程参数校正
    if nprocs == 1:
        if args.enable_pruning:
            print(" WARNING: Single-process mode. Auto-correcting parameters.")
        args.num_families = 1
        args.num_experts_per_node = 1

    if args.enable_pruning:
        if args.num_experts_per_node > nprocs or args.num_families > nprocs:
            raise ValueError(f"Pruning params invalid for nprocs={nprocs}")
        density = (args.num_experts_per_node / nprocs * 100) if nprocs > 0 else 0.0
        if rank == 0:
            print(f"\n PRUNING ENABLED:")
            print(f"   Families: {args.num_families}")
            print(f"   Experts/node: {args.num_experts_per_node}/{nprocs} ({100 - density:.1f}% pruned)")

    if rank == 0:
        print(f" NeutronEP: {nprocs}-way Ensemble Parallelism | Dataset: {args.dataset_name}")
        print(f" World Size: {world_size} | Device: {device}\n")

    # 步骤 5: 加载数据集
    my_dir = '/home/chenggh/jittor/JittorGeometric/'
    path = osp.join(my_dir, 'data')
    if args.dataset_name in ['ogbn-arxiv', 'ogbn-papers100M', 'ogbn-products']:
        dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset_name, root=path))
    elif args.dataset_name == 'cora':
        dataset = AsNodePredDataset(CoraGraphDataset(raw_dir=path))
    elif args.dataset_name == 'reddit':
        dataset = AsNodePredDataset(RedditDataset(raw_dir=path))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    g = dataset[0]
    g.create_formats_()
    if args.dataset_name == "ogbn-arxiv":
        g = dgl.to_bidirected(g, copy_ndata=True)
        g = dgl.add_self_loop(g)

    data = (dataset.num_classes, dataset.train_idx, dataset.val_idx, dataset.test_idx)
    num_classes, global_train_idx, global_val_idx, global_test_idx = data

    if rank == 0:
        node_part = topology_aware_hard_partition(g, global_train_idx, nprocs)
        node_part_tensor = node_part.clone().to(broadcast_device)
        if world_size > 1:
            dist.broadcast(node_part_tensor, src=0)
    else:
        node_part_tensor = torch.zeros(g.num_nodes(), dtype=torch.long).to(broadcast_device)
        if world_size > 1:
            dist.broadcast(node_part_tensor, src=0)
    node_part = node_part_tensor.cpu()

    node_family = None
    expert_selection = None
    if args.enable_pruning:
        if world_size > 1:
            dist.barrier()
        if rank == 0:
            print("\n" + "=" * 60)
            print(" ENSEMBLE PRUNING (Sec 2.3.1)")
            print("=" * 60)
            partition_graph = build_partition_graph(g, node_part, nprocs)
            hotspot_order, pagerank_values = compute_partition_pagerank(partition_graph)
            families, family_assign = metis_partition_families(partition_graph, args.num_families)
            expert_selection = select_experts_per_family(families, hotspot_order, pagerank_values, args.num_experts_per_node, nprocs)
            node_family = assign_nodes_to_families(node_part, family_assign)
            config_list = [family_assign.cpu(), node_family.cpu(), expert_selection]
            if world_size > 1:
                dist.broadcast_object_list(config_list, src=0)
            density = (args.num_experts_per_node / nprocs * 100) if nprocs > 0 else 0.0
            print(f" Pruning: {args.num_families} families, {args.num_experts_per_node}/{nprocs} experts")
            print(f" Activation density: {density:.1f}%")
            print("=" * 60 + "\n")
        else:
            config_list = [None, None, None]
            if world_size > 1:
                dist.broadcast_object_list(config_list, src=0)
            family_assign, node_family, expert_selection = config_list
        print(f"[Rank {rank}] Pruning config received")
    else:
        if world_size > 1:
            dist.barrier()

    core_part_nodes = torch.where(node_part == rank)[0]
    GN_N_LAYERS = 3

    print(f"[Rank {rank}] Expanding subgraph with {GN_N_LAYERS}-hop neighbors...")
    expanded_part_nodes = expand_subgraph_nodes(g, core_part_nodes, GN_N_LAYERS)

    subg = dgl.node_subgraph(g, expanded_part_nodes)
    subg.ndata['orig_id'] = expanded_part_nodes


    def map_to_local_core(global_idx, local_orig_ids):
        sorted_indices = torch.argsort(local_orig_ids)
        sorted_orig_ids = local_orig_ids[sorted_indices]

        # 查找 global_idx 在 sorted_orig_ids 中的位置
        positions = torch.searchsorted(sorted_orig_ids, global_idx)
        positions = torch.clamp(positions, 0, len(sorted_orig_ids) - 1)
        mask = (sorted_orig_ids[positions] == global_idx)

        local_indices = sorted_indices[positions][mask]
        return local_indices

    local_train_idx = map_to_local_core(global_train_idx, expanded_part_nodes)
    local_val_idx = map_to_local_core(global_val_idx, expanded_part_nodes)

    print(f"[Rank {rank}] Subgraph Stats:")
    print(f"   Core Nodes: {len(core_part_nodes)}")
    print(f"   Expanded Nodes (Total): {subg.num_nodes()} (+{subg.num_nodes() - len(core_part_nodes)})")
    print(f"   Local Train Samples: {len(local_train_idx)}")
    print(f"   Local Val Samples: {len(local_val_idx)}")


    model, classify_head = train_submodel(rank, nprocs, device, args, subg, num_classes, local_train_idx, args.train_duration)

    print(f"[Rank {rank}] Preparing model for synchronization...")
    # 准备要传输的数据
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    head_state = {k: v.cpu() for k, v in classify_head.state_dict().items()}
    local_data = {'model': model_state, 'head': head_state}
    all_models = None

    if world_size > 1:
        # 确保所有进程都完成了训练
        dist.barrier()
        if rank != 0:
            # 发送模型给 Rank 0
            print(f"[Rank {rank}] Sending model to Master (Rank 0)...")
            send_model_state(local_data, dst_rank=0)
            print(f"[Rank {rank}] Model sent successfully. Exiting...")
            dist.destroy_process_group()
            return  # Worker 任务结束，直接退出
        else:
            print(f"\nRank 0: Collecting models from {world_size - 1} workers...")
            all_models_data = [None] * world_size
            all_models_data[0] = local_data  # 自己的模型
            for src in range(1, world_size):
                print(f"  Waiting to receive from Rank {src}...")
                received_data = recv_model_state(src)
                all_models_data[src] = received_data
            # 提取纯模型权重
            all_models = [data['model'] for data in all_models_data]
            print(f"Rank 0: Successfully collected all {world_size} models.")
    else:
        # 单进程模式
        all_models = [local_data['model']]


    print("\n=== Fusion Phase: Shared Aggregation + Full Embedding Cache ===")
    n_layers = 3
    hid_size = args.hidden_dim
    submodel_weights_gpu = []
    for k in range(nprocs):
        weights_per_layer = []
        for l in range(n_layers):
            key = f'layers.{l}.fc_neigh.weight'
            if key not in all_models[k]:
                key = f'layers.{l}.lin_l.weight'
            w = all_models[k][key].to(device)
            weights_per_layer.append(w)
        submodel_weights_gpu.append(weights_per_layer)

    sampler = NeighborSampler([10, 10, 10], prefetch_node_feats=["feat"])
    dataloader = DataLoader(
        g, torch.arange(g.num_nodes()), sampler,
        device=device, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=0, use_uva=True)

    embedding_dir = osp.abspath(osp.join(tempfile.gettempdir(), f'{args.dataset_name}_embeddings'))
    os.makedirs(embedding_dir, exist_ok=True)
    fused_embedding_file = osp.join(embedding_dir, f"fused_embeddings.bin")
    fused_shape_file = osp.join(embedding_dir, f"fused_embeddings.shape")

    embeddings_list = []
    with torch.no_grad():
        for batch_idx, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            h = blocks[0].srcdata["feat"]
            for l in range(n_layers):
                block = blocks[l]
                with block.local_scope():
                    block.srcdata['h'] = h
                    block.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'agg'))
                    z = block.dstdata['agg']

                if l == 0:
                    sub_outputs = [
                        F.relu(F.linear(z, submodel_weights_gpu[k][l])) if l < n_layers - 1
                        else F.linear(z, submodel_weights_gpu[k][l])
                        for k in range(nprocs)
                    ]
                    h = torch.cat(sub_outputs, dim=1)
                else:
                    sub_outputs = []
                    for k in range(nprocs):
                        z_k = z[:, k * hid_size: (k + 1) * hid_size]
                        out = F.linear(z_k, submodel_weights_gpu[k][l])
                        if l < n_layers - 1:
                            out = F.relu(out)
                        sub_outputs.append(out)
                    h = torch.cat(sub_outputs, dim=1)

                if l == n_layers - 1:
                    embeddings_list.append(h.cpu())

            if batch_idx % 100 == 0:
                print(f"  Fusion batch {batch_idx}")

    if len(embeddings_list) == 0:
        raise RuntimeError("No embeddings generated!")

    all_embeddings = torch.cat(embeddings_list, dim=0)
    N, D = all_embeddings.shape
    all_embeddings.numpy().tofile(fused_embedding_file)
    with open(fused_shape_file, 'w') as f:
        f.write(f"{N},{D}")
    print(f" Saved fused embeddings: ({N}, {D})")
    del embeddings_list, all_embeddings
    import gc
    gc.collect()

    # ===== STEP 6: 集成器训练 =====
    storage = FeatureStorage(embedding_dir, device)
    storage.set_path_and_load(fused_embedding_file, fused_shape_file)
    labels = g.ndata["label"]
    train_val_idx = torch.cat([global_train_idx, global_val_idx])

    if args.enable_pruning and expert_selection is not None:
        print(f"\n Training {args.num_families} pruned integration units (GPU)...")
        if node_family is None or len(node_family) != g.num_nodes():
            print(" WARNING: node_family invalid, falling back to full ensemble")
            args.enable_pruning = False
        else:
            family_learners = {}
            family_opts = {}
            for f in range(args.num_families):
                experts_f = expert_selection.get(f, [0])
                input_dim = len(experts_f) * hid_size
                family_learners[f] = MetaLearner(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    meta_hidden_dim=args.meta_hidden_dim
                ).to(device)
                family_opts[f] = torch.optim.Adam(family_learners[f].parameters(), lr=0.001)

            for epoch in range(args.meta_epochs):
                total_loss = 0
                num_batches = 0
                for f in range(args.num_families):
                    family_learners[f].train()
                for start in range(0, len(train_val_idx), args.batch_size):
                    end = min(start + args.batch_size, len(train_val_idx))
                    batch_idx = train_val_idx[start:end]
                    batch_family = node_family[batch_idx]
                    batch_emb_full = storage.mmap_tensor[batch_idx.cpu()].to(device)
                    storage.page_cache_manager.access(storage.mmap_tensor[batch_idx.cpu()])
                    batch_labels = labels[batch_idx].to(device)
                    for f in range(args.num_families):
                        mask = (batch_family == f)
                        if mask.sum() == 0:
                            continue
                        experts_f = expert_selection.get(f, [0])
                        selected_indices = []
                        for e in experts_f:
                            selected_indices.extend(range(e * hid_size, (e + 1) * hid_size))
                        selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)
                        batch_emb = batch_emb_full[:, selected_indices]
                        emb_f = batch_emb[mask]
                        labels_f = batch_labels[mask]
                        family_opts[f].zero_grad()
                        logits = family_learners[f](emb_f)
                        loss = F.cross_entropy(logits, labels_f.long())
                        loss.backward()
                        family_opts[f].step()
                        total_loss += loss.item()
                        num_batches += 1
                if epoch % 10 == 0 or epoch == args.meta_epochs - 1:
                    for f in range(args.num_families):
                        family_learners[f].eval()
                    with torch.no_grad():
                        val_emb_full = storage.mmap_tensor[global_val_idx.cpu()].to(device)
                        val_family = node_family[global_val_idx]
                        all_preds, all_labels = [], []
                        for f in range(args.num_families):
                            mask = (val_family == f)
                            if mask.sum() == 0:
                                continue
                            experts_f = expert_selection.get(f, [0])
                            selected_indices = []
                            for e in experts_f:
                                selected_indices.extend(range(e * hid_size, (e + 1) * hid_size))
                            selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)
                            val_emb = val_emb_full[:, selected_indices][mask]
                            logits = family_learners[f](val_emb)
                            all_preds.append(logits.argmax(dim=1))
                            all_labels.append(labels[global_val_idx][mask].to(device))
                        if len(all_preds) > 0 and len(all_labels) > 0:
                            try:
                                val_acc = MF.accuracy(torch.cat(all_preds), torch.cat(all_labels),
                                                      task="multiclass", num_classes=num_classes)
                            except Exception as e:
                                print(f" Accuracy calculation failed: {e}")
                                val_acc = torch.tensor(0.0, device=device)
                        else:
                            val_acc = torch.tensor(0.0, device=device)
                        avg_loss = total_loss / num_batches if num_batches > 0 else 0
                        print(f"Meta-Epoch {epoch:3d}/{args.meta_epochs} | Loss {avg_loss:.4f} | Val Acc {val_acc:.4f}")

            for f in range(args.num_families):
                family_learners[f].eval()
            with torch.no_grad():
                test_emb_full = storage.mmap_tensor[global_test_idx.cpu()].to(device)
                test_family = node_family[global_test_idx]
                all_preds, all_labels = [], []
                for f in range(args.num_families):
                    mask = (test_family == f)
                    if mask.sum() == 0:
                        continue
                    experts_f = expert_selection.get(f, [0])
                    selected_indices = []
                    for e in experts_f:
                        selected_indices.extend(range(e * hid_size, (e + 1) * hid_size))
                    selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)
                    test_emb = test_emb_full[:, selected_indices][mask]
                    logits = family_learners[f](test_emb)
                    all_preds.append(logits.argmax(dim=1))
                    all_labels.append(labels[global_test_idx][mask].to(device))
                if len(all_preds) > 0 and len(all_labels) > 0:
                    try:
                        test_acc = MF.accuracy(torch.cat(all_preds), torch.cat(all_labels),
                                               task="multiclass", num_classes=num_classes)
                    except:
                        test_acc = torch.tensor(0.0, device=device)
                else:
                    test_acc = torch.tensor(0.0, device=device)

            density = (args.num_experts_per_node / nprocs * 100) if nprocs > 0 else 0.0
            print(f"\n{'=' * 60}")
            print(f" Final Test Accuracy (Pruned): {test_acc:.4f}")
            print(f" Experts: {args.num_experts_per_node}/{nprocs} ({100 - density:.1f}% pruned)")
            print(f" Families: {args.num_families}")
            print(f"{'=' * 60}")
    else:
        print(f"\nTraining integration unit (GPU)...")
        input_dim = nprocs * hid_size
        meta_learner = MetaLearner(
            input_dim=input_dim,
            num_classes=num_classes,
            meta_hidden_dim=args.meta_hidden_dim
        ).to(device)
        meta_opt = torch.optim.Adam(meta_learner.parameters(), lr=0.001)
        for epoch in range(args.meta_epochs):
            meta_learner.train()
            total_loss = 0
            num_batches = 0
            for start in range(0, len(train_val_idx), args.batch_size):
                end = min(start + args.batch_size, len(train_val_idx))
                batch_idx = train_val_idx[start:end]
                batch_emb = storage.mmap_tensor[batch_idx.cpu()].to(device)
                storage.page_cache_manager.access(storage.mmap_tensor[batch_idx.cpu()])
                batch_labels = labels[batch_idx].to(device)
                meta_opt.zero_grad()
                logits = meta_learner(batch_emb)
                loss = F.cross_entropy(logits, batch_labels.long())
                loss.backward()
                meta_opt.step()
                total_loss += loss.item()
                num_batches += 1
            if epoch % 10 == 0 or epoch == args.meta_epochs - 1:
                meta_learner.eval()
                with torch.no_grad():
                    val_emb = storage.mmap_tensor[global_val_idx.cpu()].to(device)
                    val_logits = meta_learner(val_emb)
                    val_acc = MF.accuracy(val_logits, labels[global_val_idx].to(device),
                                          task="multiclass", num_classes=num_classes)
                    avg_loss = total_loss / num_batches if num_batches > 0 else 0
                    print(f"Meta-Epoch {epoch:3d}/{args.meta_epochs} | Loss {avg_loss:.4f} | Val Acc {val_acc:.4f}")

        meta_learner.eval()
        with torch.no_grad():
            test_emb = storage.mmap_tensor[global_test_idx.cpu()].to(device)
            test_logits = meta_learner(test_emb)
            test_acc = MF.accuracy(test_logits, labels[global_test_idx].to(device),
                                   task="multiclass", num_classes=num_classes)

        print(f"\n{'=' * 60}")
        print(f" Final Test Accuracy: {test_acc:.4f}")
        print(f"{'=' * 60}")

    if args.cleanup_embeddings:
        print("Cleaning up temporary embedding files...")
        for ext in [".bin", ".shape"]:
            f = osp.join(embedding_dir, f"fused_embeddings{ext}")
            if osp.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Failed to remove {f}: {e}")
        try:
            os.rmdir(embedding_dir)
        except:
            pass

    if world_size > 1:
        dist.destroy_process_group()
    print(f"[Rank 0] Completed!")


if __name__ == "__main__":
    main()