import time

import torch
import torch.nn as nn
from dgl.dataloading import DataLoader, NeighborSampler

from .models import SAGE
from .utils import modified_cross_entropy, set_seed


def train_submodel(rank, world_size, device, args, subg, num_classes, local_train_idx, train_duration):
    set_seed(42 + rank)
    in_size = subg.ndata["feat"].shape[1]
    model = SAGE(in_size, args.hidden_dim).to(device)
    head = nn.Linear(args.hidden_dim, num_classes).to(device)

    if len(local_train_idx) == 0:
        print(f"[Rank {rank}] empty training partition")
        return model, head

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=0.003, weight_decay=0
    )
    sampler = NeighborSampler([10, 10, 10], prefetch_node_feats=["feat"], prefetch_labels=["label"])
    dataloader = DataLoader(
        subg,
        local_train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )

    model.train()
    head.train()
    start_time = time.time()
    steps = 0
    while time.time() - start_time < train_duration:
        for _, _, blocks in dataloader:
            blocks = [block.to(device) for block in blocks]
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"].long()
            logits = head(model(blocks, x))
            loss = modified_cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

    elapsed = time.time() - start_time
    print(f"[Rank {rank}] train={elapsed:.2f}s steps={steps} samples={len(local_train_idx)}")
    return model, head
