import dgl
import torch
from dgl.partition import metis_partition_assignment


def metis_partition_families(partition_graph, num_families):
    num_nodes = partition_graph.number_of_nodes()
    if num_families >= num_nodes or num_nodes == 1:
        return {0: list(range(num_nodes))}, torch.zeros(num_nodes, dtype=torch.long)

    edges = list(partition_graph.edges())
    if len(edges) > 0:
        src = [edge[0] for edge in edges]
        dst = [edge[1] for edge in edges]
        family_graph = dgl.graph((src, dst), num_nodes=num_nodes)
        try:
            family_assign = metis_partition_assignment(family_graph, num_families)
        except Exception as exc:
            print(f"[families] METIS failed ({exc}); using greedy assignment")
            family_assign = torch.arange(num_nodes) % num_families
    else:
        family_assign = torch.arange(num_nodes) % num_families

    families = {f: [] for f in range(num_families)}
    for node, family in enumerate(family_assign.cpu().numpy()):
        families[int(family)].append(node)
    return families, family_assign


def select_experts_per_family(families, hotspot_order, pagerank_values, num_experts_per_node):
    expert_selection = {}
    for family_id, members in families.items():
        members_set = set(members)
        if len(members) == 0:
            selected = hotspot_order[:num_experts_per_node] if hotspot_order else [0]
        elif len(members) > num_experts_per_node:
            ordered = sorted(members, key=lambda node: pagerank_values.get(node, 0.0), reverse=True)
            selected = ordered[:num_experts_per_node]
        else:
            selected = list(members)

        for partition in hotspot_order:
            if partition not in members_set and len(selected) < num_experts_per_node:
                selected.append(partition)
        if not selected:
            selected = [0]
        expert_selection[family_id] = selected
    return expert_selection


def assign_nodes_to_families(node_part, family_assign):
    return family_assign[node_part]
