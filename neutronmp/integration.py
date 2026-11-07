import os
import os.path as osp

import torch
import torch.nn.functional as F
import torchmetrics.functional as MF

from .models import MetaLearner


def _selected_columns(experts, hidden_dim, device):
    indices = []
    for expert in experts:
        indices.extend(range(expert * hidden_dim, (expert + 1) * hidden_dim))
    return torch.tensor(indices, dtype=torch.long, device=device)


def train_full_integration(storage, labels, train_val_idx, val_idx, test_idx, num_classes, args, device):
    input_dim = storage.feat_dim
    learner = MetaLearner(
        input_dim=input_dim,
        num_classes=num_classes,
        meta_hidden_dim=args.meta_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=0.001)

    for epoch in range(args.meta_epochs):
        learner.train()
        total_loss = 0.0
        num_batches = 0
        for start in range(0, len(train_val_idx), args.batch_size):
            batch_idx = train_val_idx[start:start + args.batch_size]
            batch_emb = storage.mmap_tensor[batch_idx.cpu()].to(device)
            storage.page_cache_manager.access(storage.mmap_tensor[batch_idx.cpu()])
            batch_labels = labels[batch_idx].to(device)
            optimizer.zero_grad()
            logits = learner(batch_emb)
            loss = F.cross_entropy(logits, batch_labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        if epoch % 10 == 0 or epoch == args.meta_epochs - 1:
            learner.eval()
            with torch.no_grad():
                val_emb = storage.mmap_tensor[val_idx.cpu()].to(device)
                val_logits = learner(val_emb)
                val_acc = MF.accuracy(
                    val_logits, labels[val_idx].to(device),
                    task="multiclass", num_classes=num_classes,
                )
            avg_loss = total_loss / num_batches if num_batches else 0.0
            print(f"[meta] epoch={epoch} loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    learner.eval()
    with torch.no_grad():
        test_emb = storage.mmap_tensor[test_idx.cpu()].to(device)
        test_logits = learner(test_emb)
        test_acc = MF.accuracy(
            test_logits, labels[test_idx].to(device),
            task="multiclass", num_classes=num_classes,
        )
    return test_acc



def train_pruned_integration(storage, labels, train_val_idx, val_idx, test_idx, num_classes, node_family, expert_selection, args, device):
    family_learners = {}
    family_opts = {}
    hidden_dim = args.hidden_dim
    for family in range(args.num_families):
        experts = expert_selection.get(family, [0])
        learner = MetaLearner(
            input_dim=len(experts) * hidden_dim,
            num_classes=num_classes,
            meta_hidden_dim=args.meta_hidden_dim,
        ).to(device)
        family_learners[family] = learner
        family_opts[family] = torch.optim.Adam(learner.parameters(), lr=0.001)

    for epoch in range(args.meta_epochs):
        total_loss = 0.0
        num_batches = 0
        for family in range(args.num_families):
            family_learners[family].train()

        for start in range(0, len(train_val_idx), args.batch_size):
            batch_idx = train_val_idx[start:start + args.batch_size]
            batch_family = node_family[batch_idx]
            batch_emb_full = storage.mmap_tensor[batch_idx.cpu()].to(device)
            storage.page_cache_manager.access(storage.mmap_tensor[batch_idx.cpu()])
            batch_labels = labels[batch_idx].to(device)

            for family in range(args.num_families):
                mask = batch_family == family
                if mask.sum() == 0:
                    continue
                experts = expert_selection.get(family, [0])
                columns = _selected_columns(experts, hidden_dim, device)
                batch_emb = batch_emb_full[:, columns][mask]
                labels_f = batch_labels[mask]
                family_opts[family].zero_grad()
                logits = family_learners[family](batch_emb)
                loss = F.cross_entropy(logits, labels_f.long())
                loss.backward()
                family_opts[family].step()
                total_loss += loss.item()
                num_batches += 1

        if epoch % 10 == 0 or epoch == args.meta_epochs - 1:
            for family in range(args.num_families):
                family_learners[family].eval()
            with torch.no_grad():
                val_emb_full = storage.mmap_tensor[val_idx.cpu()].to(device)
                val_family = node_family[val_idx]
                preds, targets = [], []
                for family in range(args.num_families):
                    mask = val_family == family
                    if mask.sum() == 0:
                        continue
                    experts = expert_selection.get(family, [0])
                    columns = _selected_columns(experts, hidden_dim, device)
                    logits = family_learners[family](val_emb_full[:, columns][mask])
                    preds.append(logits.argmax(dim=1))
                    targets.append(labels[val_idx][mask].to(device))
            if preds:
                val_acc = MF.accuracy(
                    torch.cat(preds), torch.cat(targets),
                    task="multiclass", num_classes=num_classes,
                )
            else:
                val_acc = torch.tensor(0.0, device=device)
            avg_loss = total_loss / num_batches if num_batches else 0.0
            print(f"[meta] epoch={epoch} loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    for family in range(args.num_families):
        family_learners[family].eval()
    with torch.no_grad():
        test_emb_full = storage.mmap_tensor[test_idx.cpu()].to(device)
        test_family = node_family[test_idx]
        preds, targets = [], []
        for family in range(args.num_families):
            mask = test_family == family
            if mask.sum() == 0:
                continue
            experts = expert_selection.get(family, [0])
            columns = _selected_columns(experts, hidden_dim, device)
            logits = family_learners[family](test_emb_full[:, columns][mask])
            preds.append(logits.argmax(dim=1))
            targets.append(labels[test_idx][mask].to(device))
    if preds:
        test_acc = MF.accuracy(
            torch.cat(preds), torch.cat(targets),
            task="multiclass", num_classes=num_classes,
        )
    else:
        test_acc = torch.tensor(0.0, device=device)
    return test_acc


def cleanup_embedding_cache(embedding_dir):
    for ext in (".bin", ".shape"):
        path = osp.join(embedding_dir, f"fused_embeddings{ext}")
        if osp.exists(path):
            try:
                os.remove(path)
            except OSError as exc:
                print(f"[cleanup] failed to remove {path}: {exc}")
    try:
        os.rmdir(embedding_dir)
    except OSError:
        pass

