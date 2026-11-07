import networkx as nx
import torch
import dgl
from dgl.partition import metis_partition_assignment


def train_aware_balanced_partition(graph, train_idx, world_size):
    num_nodes = graph.num_nodes()
    node_part = torch.full((num_nodes,), -1, dtype=torch.long)
    train_nodes = train_idx[torch.randperm(len(train_idx))]
    for i, node in enumerate(train_nodes):
        node_part[node] = i % world_size
    remaining = torch.where(node_part == -1)[0]
    for node in remaining:
        sizes = [(node_part == p).sum().item() for p in range(world_size)]
        node_part[node] = min(range(world_size), key=lambda p: sizes[p])
    return node_part


def topology_aware_hard_partition(graph, train_idx, world_size):
    if world_size == 1:
        return torch.zeros(graph.num_nodes(), dtype=torch.long)
    try:
        node_part = metis_partition_assignment(graph, world_size)
    except Exception as exc:
        print(f"[partition] METIS failed ({exc}); using balanced partition")
        return train_aware_balanced_partition(graph, train_idx, world_size)

    min_train = min((node_part[train_idx] == p).sum().item() for p in range(world_size))
    if min_train == 0:
        print("[partition] empty training partition detected; rebalancing")
        return train_aware_balanced_partition(graph, train_idx, world_size)
    return node_part


def expand_subgraph_nodes(graph, core_nodes, k_hops):
    if k_hops == 0:
        return core_nodes

    frontier = core_nodes.unique()
    all_nodes = set(frontier.cpu().numpy())
    for _ in range(k_hops):
        if len(frontier) == 0:
            break
        neighbors = graph.in_edges(frontier)[0]
        new_nodes = neighbors.unique()
        added = [n for n in new_nodes.cpu().numpy() if n not in all_nodes]
        if not added:
            break
        all_nodes.update(added)
        frontier = torch.tensor(added, dtype=torch.long, device=core_nodes.device)
    return torch.tensor(list(all_nodes), dtype=torch.long, device=core_nodes.device)


def build_partition_graph(graph, node_part, nprocs):
    edge_counts = {}
    src, dst = graph.edges()
    src_part = node_part[src]
    dst_part = node_part[dst]
    for s, d in zip(src_part.cpu().numpy(), dst_part.cpu().numpy()):
        if s != d:
            key = (min(s, d), max(s, d))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    partition_graph = nx.Graph()
    for i in range(nprocs):
        partition_graph.add_node(i)
    for (i, j), weight in edge_counts.items():
        partition_graph.add_edge(i, j, weight=weight)
    return partition_graph


def compute_partition_pagerank(partition_graph, damping=0.85, max_iter=100):
    num_nodes = partition_graph.number_of_nodes()
    if num_nodes <= 1:
        node = list(partition_graph.nodes())[0] if num_nodes == 1 else 0
        return [node], {node: 1.0}
    try:
        pagerank = nx.pagerank(partition_graph, weight="weight", max_iter=max_iter, tol=1e-6)
    except Exception as exc:
        print(f"[pagerank] fallback to uniform ({exc})")
        pagerank = {i: 1.0 / num_nodes for i in range(num_nodes)}
    ordered = sorted(pagerank.items(), key=lambda item: item[1], reverse=True)
    return [item[0] for item in ordered], {item[0]: item[1] for item in ordered}


def map_to_local_core(global_idx, local_orig_ids):
    sorted_indices = torch.argsort(local_orig_ids)
    sorted_ids = local_orig_ids[sorted_indices]
    positions = torch.searchsorted(sorted_ids, global_idx)
    positions = torch.clamp(positions, 0, len(sorted_ids) - 1)
    mask = sorted_ids[positions] == global_idx
    return sorted_indices[positions][mask]
