import os

import dgl
import dgl.data as dgl_data
from dgl.data import AsNodePredDataset, CoauthorCSDataset, CoraGraphDataset, RedditDataset
from dgl.data import utils as dgl_utils
from ogb.nodeproppred import DglNodePropPredDataset

DGL_DATA_MIRROR = os.environ.get(
    "DGL_DATA_MIRROR", "https://s3.us-west-2.amazonaws.com/dgl-data"
)

SUPPORTED_DATASETS = (
    "cora",
    "coauthor-cs",
    "reddit",
    "ogbn-arxiv",
    "ogbn-products",
    "ogbn-papers100M",
)


def enable_dgl_data_mirror():
    if not DGL_DATA_MIRROR:
        return

    def _get_dgl_url(file_url):
        return DGL_DATA_MIRROR.rstrip("/") + "/" + file_url

    dgl_utils._get_dgl_url = _get_dgl_url
    for attr in dir(dgl_data):
        module = getattr(dgl_data, attr)
        if hasattr(module, "_get_dgl_url"):
            module._get_dgl_url = _get_dgl_url


def _build_dataset(name, root):
    if name in ("ogbn-arxiv", "ogbn-products", "ogbn-papers100M"):
        return AsNodePredDataset(DglNodePropPredDataset(name, root=root), raw_dir=root)
    if name == "cora":
        return AsNodePredDataset(CoraGraphDataset(raw_dir=root), raw_dir=root)
    if name == "reddit":
        return AsNodePredDataset(RedditDataset(raw_dir=root), raw_dir=root)
    if name == "coauthor-cs":
        return AsNodePredDataset(
            CoauthorCSDataset(raw_dir=root),
            split_ratio=[0.6, 0.2, 0.2],
            raw_dir=root,
        )
    raise ValueError(f"Unknown dataset: {name}")


def load_dataset(name, root, rank, world_size, barrier=None):
    enable_dgl_data_mirror()

    if rank == 0:
        _build_dataset(name, root)
    if barrier is not None:
        barrier()

    dataset = _build_dataset(name, root)
    graph = dataset[0]
    graph.create_formats_()
    if name == "ogbn-arxiv":
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph = dgl.add_self_loop(graph)

    split = (dataset.num_classes, dataset.train_idx, dataset.val_idx, dataset.test_idx)
    return dataset, graph, split
