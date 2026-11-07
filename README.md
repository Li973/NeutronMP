# NeutronMP

NeutronMP is a distributed graph neural network training system built on multi-model parallelism. It partitions the input graph, trains one independent submodel per worker without iterative cross-worker synchronization, caches the fused intermediate embeddings in a single inference pass, and trains a lightweight stacking unit that integrates the knowledge of all submodels. Topology-aware submodel pruning reduces the number of active experts during inference.

## Repository Layout

```text
NeutronMP/
├── main.py                  # entry point
├── neutronmp/
│   ├── datasets.py          # dataset loading and DGL download mirror
│   ├── distributed.py       # process group, partitioning and model transfer
│   ├── partitioning.py      # METIS partitioning, subgraph expansion, PageRank
│   ├── pruning.py           # topology-aware families and expert selection
│   ├── models.py            # SAGE submodel and stacking meta learner
│   ├── submodel.py          # per-worker submodel training
│   ├── fusion.py            # shared-aggregation embedding cache
│   ├── storage.py           # mmap embedding storage and page cache manager
│   └── integration.py       # meta learner training and evaluation
├── dataset/                 # auto-populated dataset cache
└── README.md
```

## Requirements

- Linux, Python 3.9+
- PyTorch 2.x with CUDA
- DGL 2.x with CUDA
- OGB, torchmetrics, networkx, numpy

```bash
pip install torch torchmetrics networkx numpy
pip install dgl ogb
```

Datasets are downloaded automatically into `NeutronMP/dataset` on first use. If `data.dgl.ai` is not reachable from your network, NeutronMP redirects DGL downloads through an equivalent S3 mirror. Set `DGL_DATA_MIRROR=""` to use the official host, or set it to another mirror base URL.

Supported datasets: `cora`, `coauthor-cs`, `reddit`, `ogbn-arxiv`, `ogbn-products`, `ogbn-papers100M`.

## Quick Start: Cora

### Single-node multi-GPU (NCCL)

Three processes on three GPUs of one machine:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run \
  --nproc_per_node=3 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  main.py \
  --dataset_name cora \
  --train_duration 1.0 \
  --meta_epochs 30 \
  --batch_size 1024 \
  --hidden_dim 256
```

### Single-node simulation of multi-node (Gloo)

Setting `MASTER_ADDR` to a non-loopback address makes NeutronMP select the Gloo backend and CPU-based model transfer, exercising the multi-node path on one machine:

```bash
HOST_IP=$(hostname -I | awk '{print $1}')
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run \
  --nproc_per_node=3 \
  --master_addr=$HOST_IP \
  --master_port=29500 \
  main.py \
  --dataset_name cora \
  --train_duration 1.0 \
  --meta_epochs 30
```

### Multi-node (one process per node)

On the first node:

```bash
python -m torch.distributed.run \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --master_addr=<NODE0_IP> \
  --master_port=29500 \
  main.py --dataset_name cora --train_duration 1.0 --meta_epochs 30
```

On the second node:

```bash
python -m torch.distributed.run \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=1 \
  --master_addr=<NODE0_IP> \
  --master_port=29500 \
  main.py --dataset_name cora --train_duration 1.0 --meta_epochs 30
```

`<NODE0_IP>` is the reachable IP address of the first node. `--nproc_per_node` can be larger than one when a node has multiple GPUs.

## Topology-Aware Pruning

Pruning activates local experts of each partition family plus globally important hotspot experts. For three workers, retaining two experts per node prunes one third of the submodels:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run \
  --nproc_per_node=3 --master_addr=127.0.0.1 --master_port=29500 \
  main.py --dataset_name cora --train_duration 1.0 --meta_epochs 30 \
  --enable_pruning --num_families 2 --num_experts_per_node 2
```

## Command-Line Options

| Option | Default | Description |
|---|---|---|
| `--dataset_name` | `cora` | Dataset name |
| `--train_duration` | `1.0` | Wall-clock training window per submodel in seconds |
| `--batch_size` | `1024` | Mini-batch size |
| `--hidden_dim` | `256` | Hidden dimension of each submodel |
| `--meta_epochs` | `30` | Training epochs of the integration unit |
| `--meta_hidden_dim` | `128` | Hidden dimension of the integration unit |
| `--exact_fusion` | off | Run each submodel's full forward pass during embedding fusion |
| `--enable_pruning` | off | Enable topology-aware submodel pruning |
| `--num_families` | `2` | Number of partition families |
| `--num_experts_per_node` | `2` | Active experts per node after pruning |
| `--cleanup_embeddings` | off | Remove cached embedding files after training |

## Output

Each run reports the per-worker subgraph statistics, submodel training progress, integration-unit validation accuracy, and the final test accuracy. With pruning enabled, the output also reports the retained expert ratio and the acceleration density.

## Citation

```bibtex
@article{li2026neutronmp,
  title   = {NeutronMP: Multi-Model Parallelism for Breaking Synchronization Barriers in Distributed GNN Training},
  author  = {Li, Chunyang and Ai, Xin and Zhang, Yanfeng},
  year    = {2026}
}
```
