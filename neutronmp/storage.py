import ctypes
import ctypes.util
import time
from collections import OrderedDict
from pathlib import Path

import torch


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
        start_page = addr // self.page_size * self.page_size
        end_page = (addr + size + self.page_size - 1) // self.page_size * self.page_size
        now = time.time()
        for page_addr in range(start_page, end_page, self.page_size):
            if page_addr in self.cache:
                self.cache.move_to_end(page_addr)
            else:
                self.cache[page_addr] = now
                self.total_cached += self.page_size
        self._evict()

    def _evict(self):
        while self.total_cached > self.max_bytes and self.cache:
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
        with open(shape_path, "r") as handle:
            num_nodes, feat_dim = map(int, handle.read().strip().split(","))
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self.mmap_tensor = torch.from_file(
            str(feature_path),
            dtype=torch.float32,
            size=num_nodes * feat_dim,
            shared=True,
        ).reshape(num_nodes, feat_dim)
