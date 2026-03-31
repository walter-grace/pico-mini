"""
MoE Expert Sniper — Read only active experts from SSD via F_NOCACHE + pread.

NOTE: The LRU cache in this file is superseded by madvise-based prefetching
in the llama.cpp eval callback. madvise achieves 2.4x higher throughput with
5000x less memory overhead (0.57 vs 0.24 tok/s, 1 MB vs 5 GB on 8 GB MacBook).
The LRU cache is retained for research reference and for the MLX path which
uses direct I/O (F_NOCACHE + pread) instead of mmap.

For a 256-expert model with 8 active per token:
  - Each expert: ~1.69 MB (4-bit quantized, moe_intermediate_size=512, hidden_size=2048)
  - Per layer: 8 × 1.69 MB = 13.5 MB
  - Per token (40 layers): ~540 MB
  - At 3-5 GB/s NVMe: ~108-180ms = 5.6-9.3 tok/s theoretical

Uses multi-threaded pread (8 workers) to saturate NVMe queue depth.
"""

import os
import json
import fcntl
import time
import numpy as np
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48
PAGE_SIZE = 16384


class LRUExpertCache:
    """LRU cache for parsed expert data. Skips SSD reads on cache hits."""

    def __init__(self, max_experts=100):
        self.max_experts = max_experts
        self.cache = OrderedDict()  # (layer_idx, expert_id) → parsed expert dict
        self.hits = 0
        self.misses = 0

    def get(self, layer_idx, expert_id):
        key = (layer_idx, expert_id)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, layer_idx, expert_id, data):
        key = (layer_idx, expert_id)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_experts:
                self.cache.popitem(last=False)
            self.cache[key] = data

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self):
        total = self.hits + self.misses
        return (f"cache: {len(self.cache)}/{self.max_experts} entries, "
                f"hit_rate={self.hit_rate():.1%} ({self.hits}/{total})")


class MoEExpertReader:
    """
    Reads specific experts from concatenated layer files via F_NOCACHE + pread.
    Expert offset = data_start + expert_id × expert_block_size
    """

    def __init__(self, expert_dir, num_layers, num_workers=8, cache_size=0):
        self.expert_dir = expert_dir
        self.num_layers = num_layers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # LRU cache (0 = disabled)
        self.lru = LRUExpertCache(max_experts=cache_size) if cache_size > 0 else None

        # Parse all layer headers
        self.headers = {}
        self.fds = {}
        for i in range(num_layers):
            path = f"{expert_dir}/layer_{i:02d}.bin"
            with open(path, "rb") as f:
                raw = f.read(PAGE_SIZE)
            self.headers[i] = json.loads(raw.rstrip(b"\x00"))

        # Precompute layout info
        h0 = self.headers[0]["layout"]
        self.expert_block_size = h0["expert_block_size"]
        self.data_start = h0["data_start"]
        self.tensor_layout = h0["tensors"]

        # Stats
        self.read_time = 0.0
        self.reads = 0
        self.bytes_read = 0
        self.cache_hits = 0

        # Prefetch state
        self.prefetch_futures = {}

    def _get_fd(self, layer_idx):
        if layer_idx not in self.fds:
            path = f"{self.expert_dir}/layer_{layer_idx:02d}.bin"
            fd = os.open(path, os.O_RDONLY)
            fcntl.fcntl(fd, F_NOCACHE, 1)
            self.fds[layer_idx] = fd
        return self.fds[layer_idx]

    def _read_expert(self, layer_idx, expert_id):
        """Read one expert's data via pread. Thread-safe."""
        fd = self._get_fd(layer_idx)
        offset = self.data_start + expert_id * self.expert_block_size

        # Read the full expert block
        data = os.pread(fd, self.expert_block_size, offset)
        return data

    def _parse_expert_data(self, raw_data, expert_id):
        """Parse raw bytes into MLX arrays for one expert."""
        import mlx.core as mx

        # Map dtype strings to MLX dtypes
        MLX_DTYPES = {
            "uint32": mx.uint32, "float16": mx.float16, "float32": mx.float32,
            "bfloat16": mx.bfloat16,
        }

        result = {}
        for name, info in self.tensor_layout.items():
            inner_offset = info["inner_offset"]
            nbytes = info["nbytes"]
            shape = info["shape_per_expert"]
            dtype_str = info["dtype"].replace("mlx.core.", "")
            mlx_dtype = MLX_DTYPES.get(dtype_str, mx.float16)

            arr_bytes = raw_data[inner_offset:inner_offset + nbytes]
            # Create MLX array directly from bytes (handles bfloat16 correctly)
            flat = mx.array(np.frombuffer(arr_bytes, dtype=np.uint8))
            arr = flat.view(mlx_dtype).reshape(shape)
            result[name] = arr

        return result

    def prefetch_experts(self, layer_idx, expert_ids):
        """Launch parallel pread for experts not in cache. Non-blocking."""
        futures = {}
        for eid in expert_ids:
            # Skip prefetch if already cached
            if self.lru and (layer_idx, eid) in self.lru.cache:
                continue
            future = self.executor.submit(self._read_expert, layer_idx, eid)
            futures[eid] = future
        self.prefetch_futures[layer_idx] = futures

    def get_experts(self, layer_idx, expert_ids):
        """
        Get parsed expert data for active experts.
        Checks LRU cache first, then prefetched data, then reads from SSD.

        Returns: dict[expert_id] → dict[tensor_name → mx.array]
        """
        t0 = time.time()

        experts = {}
        futures = self.prefetch_futures.pop(layer_idx, {})

        for eid in expert_ids:
            # 1. Check LRU cache
            if self.lru:
                cached = self.lru.get(layer_idx, eid)
                if cached is not None:
                    experts[eid] = cached
                    self.cache_hits += 1
                    continue

            # 2. Check prefetched data
            if eid in futures:
                raw = futures[eid].result()
            else:
                # 3. Synchronous read
                raw = self._read_expert(layer_idx, eid)

            parsed = self._parse_expert_data(raw, eid)
            experts[eid] = parsed
            self.bytes_read += len(raw)

            # Store in cache
            if self.lru:
                self.lru.put(layer_idx, eid, parsed)

        self.read_time += time.time() - t0
        self.reads += len(expert_ids)
        return experts

    def stats(self):
        if self.reads == 0:
            return "No reads yet"
        ssd_reads = self.reads - self.cache_hits
        avg_ms = self.read_time / self.reads * 1000
        throughput = self.bytes_read / self.read_time / 1e9 if self.read_time > 0 else 0
        s = (f"reads={self.reads}, ssd_reads={ssd_reads}, cache_hits={self.cache_hits}, "
             f"avg={avg_ms:.1f}ms/expert, "
             f"throughput={throughput:.1f} GB/s, "
             f"total_bytes={self.bytes_read/1e9:.2f} GB, "
             f"total_time={self.read_time:.1f}s")
        if self.lru:
            s += f"\n  {self.lru.stats()}"
        return s

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.executor.shutdown(wait=False)
