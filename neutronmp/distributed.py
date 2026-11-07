import os

import torch
import torch.distributed as dist


def get_dist_info():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return world_size, rank, local_rank


def resolve_device(local_rank):
    if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def init_distributed(world_size, rank, device):
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    is_multinode = master_addr not in ("127.0.0.1", "localhost")

    if world_size > 1:
        backend = "gloo" if is_multinode else "nccl"
        broadcast_device = device if backend == "nccl" else torch.device("cpu")
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    else:
        backend = "gloo"
        broadcast_device = device

    if rank == 0:
        print(f"[Rank 0] world_size={world_size} device={device} backend={backend}")
    return backend, broadcast_device


def gather_models(local_data, world_size, rank):
    if world_size == 1:
        return [local_data["model"]]

    dist.barrier()
    if rank == 0:
        gathered = [None] * world_size
    else:
        gathered = None
    dist.gather_object(local_data, gathered, dst=0)

    if rank != 0:
        dist.destroy_process_group()
        return None
    return [data["model"] for data in gathered]



def broadcast_partition(node_part, num_nodes, world_size, rank, broadcast_device):
    if world_size == 1:
        return node_part.cpu()

    if rank == 0:
        node_part_tensor = node_part.clone().to(broadcast_device)
    else:
        node_part_tensor = torch.zeros(num_nodes, dtype=torch.long).to(broadcast_device)
    dist.broadcast(node_part_tensor, src=0)
    return node_part_tensor.cpu()


def broadcast_pruning_config(config_list, world_size, rank):
    if world_size == 1:
        return config_list
    dist.broadcast_object_list(config_list, src=0)
    return config_list
