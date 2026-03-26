#!/usr/bin/env python3
"""
Test harness: loads model once, generates reference KV cache,
and measures compression techniques against it.
"""

import os
import sys
import time
import mlx.core as mx

# Reference prompt — deterministic, covers code + prose + math
REFERENCE_PROMPT = """<|im_start|>user
Explain the following topics in detail:
1. How backpropagation works in neural networks, including the chain rule
2. The Python asyncio event loop and how coroutines are scheduled
3. The integral of x^2 * e^x using integration by parts
4. How TCP congestion control works (slow start, AIMD)
5. The eigenvalue decomposition of a symmetric matrix
<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


class Harness:
    """Loads model once, reuses for all experiments."""

    def __init__(self, model_id="mlx-community/Qwen3.5-9B-MLX-4bit"):
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        print(f"  Loading model: {model_id}", flush=True)
        self.model, self.tokenizer = load(model_id)
        self.model_id = model_id

        # Generate reference KV cache
        print("  Generating reference KV cache...", flush=True)
        tokens = mx.array(self.tokenizer.encode(REFERENCE_PROMPT))
        print(f"  Prompt tokens: {len(tokens.tolist())}", flush=True)
        self.ref_cache = make_prompt_cache(self.model)
        print("  Running forward pass...", flush=True)
        logits = self.model(tokens[None], cache=self.ref_cache)
        mx.eval(logits)
        print("  Forward pass complete.", flush=True)

        self.ref_states = [c.state for c in self.ref_cache]
        self.ref_tokens = len(tokens.tolist())
        self.ref_bytes = sum(c.nbytes for c in self.ref_cache)

        print(f"  Reference: {self.ref_tokens} tokens, {self.ref_bytes / (1024*1024):.1f} MB")

    def test_technique(self, compress_fn, decompress_fn, **params):
        """
        Test a compression technique against the reference KV cache.
        Returns metrics dict.
        """
        results = {
            "cosine_sim": 0,
            "mse": 0,
            "compression_ratio": 0,
            "original_mb": self.ref_bytes / (1024 * 1024),
            "compressed_mb": 0,
            "compress_time": 0,
            "decompress_time": 0,
            "error": None,
        }

        try:
            total_cosine = 0
            total_mse = 0
            total_compressed_bytes = 0
            count = 0

            t_compress = 0
            t_decompress = 0

            for layer_state in self.ref_states:
                if isinstance(layer_state, list):
                    tensors = layer_state
                else:
                    tensors = [layer_state]

                for tensor in tensors:
                    if not hasattr(tensor, 'shape'):
                        continue

                    # Compress
                    t0 = time.time()
                    compressed = compress_fn(tensor, **params)
                    mx.eval(compressed.data)
                    t_compress += time.time() - t0

                    # Track compressed size
                    total_compressed_bytes += compressed.data.nbytes
                    if hasattr(compressed.metadata, '__iter__'):
                        for v in compressed.metadata.values():
                            if hasattr(v, 'nbytes'):
                                total_compressed_bytes += v.nbytes

                    # Decompress
                    t0 = time.time()
                    restored = decompress_fn(compressed)
                    mx.eval(restored)
                    t_decompress += time.time() - t0

                    # Quality metrics
                    o = tensor.astype(mx.float32).reshape(-1)
                    r = restored.astype(mx.float32).reshape(-1)

                    # MSE
                    mse = float(mx.mean((o - r) ** 2))
                    total_mse += mse

                    # Cosine similarity
                    dot = float(mx.sum(o * r))
                    norm_o = float(mx.sqrt(mx.sum(o ** 2)))
                    norm_r = float(mx.sqrt(mx.sum(r ** 2)))
                    cosine = dot / (norm_o * norm_r + 1e-8)
                    total_cosine += cosine

                    count += 1

            if count > 0:
                results["cosine_sim"] = total_cosine / count
                results["mse"] = total_mse / count
                results["compressed_mb"] = total_compressed_bytes / (1024 * 1024)
                results["compression_ratio"] = self.ref_bytes / max(total_compressed_bytes, 1)
                results["compress_time"] = t_compress
                results["decompress_time"] = t_decompress

        except Exception as e:
            results["error"] = str(e)

        return results
