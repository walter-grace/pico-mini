#!/usr/bin/env python3
"""
llama.cpp test harness for kv-lab.
Tests built-in --cache-type-k and --cache-type-v flags on the 35B model.

Unlike the MLX harness (direct tensor manipulation), this harness:
1. Starts llama-server with specific cache-type flags
2. Runs a fixed prompt through the server
3. Measures output quality, speed, and memory usage
4. Compares against f16 baseline

Requires: llama-server installed (brew install llama.cpp)
"""

import json
import os
import signal
import subprocess
import time
import urllib.request

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_9B = os.path.expanduser("~/models/Qwen3.5-9B-Q4_K_M.gguf")
MODEL_35B = os.path.expanduser("~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf")

PORT = 8001  # Use 8001 to avoid conflicting with running MLX/llama on 8000
HOST = "127.0.0.1"
SERVER_URL = f"http://{HOST}:{PORT}"

# Cache type options available in llama.cpp
CACHE_TYPES = [
    "f16",     # Baseline — full precision
    "q8_0",    # 8-bit quantized
    "q5_0",    # 5-bit quantized
    "q5_1",    # 5-bit quantized (variant)
    "q4_0",    # 4-bit quantized (current production)
    "q4_1",    # 4-bit quantized (variant)
    "iq4_nl",  # 4-bit non-linear
    "q3_K",    # 3-bit K-quant
    "q6_K",    # 6-bit K-quant
]

# Reference prompt — same as MLX harness for fair comparison
REFERENCE_PROMPT = """Explain the following topics in detail:
1. How backpropagation works in neural networks, including the chain rule
2. The Python asyncio event loop and how coroutines are scheduled
3. The integral of x^2 * e^x using integration by parts
4. How TCP congestion control works (slow start, AIMD)
5. The eigenvalue decomposition of a symmetric matrix"""

# Shorter quality-check prompt for comparing outputs
QUALITY_PROMPT = """What is the derivative of x^3 * sin(x)? Show your work step by step."""


class LlamaHarness:
    """Start/stop llama-server with different cache flags, measure quality."""

    def __init__(self, model_path=None):
        if model_path is None:
            # Auto-detect: prefer 35B, fall back to 9B
            if os.path.exists(MODEL_35B):
                model_path = MODEL_35B
            elif os.path.exists(MODEL_9B):
                model_path = MODEL_9B
            else:
                raise FileNotFoundError(
                    f"No model found. Expected:\n  {MODEL_35B}\n  {MODEL_9B}"
                )

        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.is_35b = "35B" in self.model_name
        self.server_process = None
        self.baseline_output = None

        # Set context size based on model
        self.ctx_size = 12288 if self.is_35b else 65536

        print(f"  LlamaHarness: {self.model_name}")
        print(f"  Context: {self.ctx_size}, Port: {PORT}")

    def _start_server(self, cache_type_k="f16", cache_type_v="f16"):
        """Start llama-server with specific cache flags."""
        self._stop_server()

        cmd = [
            "llama-server",
            "--model", self.model_path,
            "--port", str(PORT),
            "--host", HOST,
            "--flash-attn", "on",
            "--ctx-size", str(self.ctx_size),
            "--cache-type-k", cache_type_k,
            "--cache-type-v", cache_type_v,
            "--n-gpu-layers", "99",
            "--reasoning", "off",
            "-np", "1",
            "-t", "4",
        ]

        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        for i in range(60):  # 60 second timeout
            try:
                resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2)
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    return True
            except Exception:
                pass
            time.sleep(1)

        print(f"    Server failed to start after 60s")
        self._stop_server()
        return False

    def _stop_server(self):
        """Stop the running llama-server."""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            self.server_process = None
            time.sleep(2)  # Let port release

    def _generate(self, prompt, max_tokens=500):
        """Generate completion from running server."""
        payload = json.dumps({
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,  # Deterministic for comparison
        }).encode()

        req = urllib.request.Request(
            f"{SERVER_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        elapsed = time.time() - t0

        content = data["choices"][0]["message"]["content"]
        timings = data.get("timings", {})

        return {
            "content": content,
            "elapsed": elapsed,
            "tokens_per_sec": timings.get("predicted_per_second", 0),
            "prompt_tokens": timings.get("prompt_n", 0),
            "completion_tokens": timings.get("predicted_n", 0),
        }

    def _get_server_props(self):
        """Get server properties including memory usage."""
        try:
            resp = urllib.request.urlopen(f"{SERVER_URL}/props", timeout=5)
            return json.loads(resp.read())
        except Exception:
            return {}

    def _text_similarity(self, text_a, text_b):
        """Simple word-overlap similarity (no numpy needed)."""
        if not text_a or not text_b:
            return 0.0
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        return 2 * len(intersection) / (len(words_a) + len(words_b))

    def generate_baseline(self):
        """Generate f16 baseline output for quality comparison."""
        print("  Generating f16 baseline...")
        if not self._start_server("f16", "f16"):
            return False

        try:
            result = self._generate(QUALITY_PROMPT, max_tokens=500)
            self.baseline_output = result["content"]
            self.baseline_speed = result["tokens_per_sec"]
            print(f"    Baseline: {len(self.baseline_output)} chars, {self.baseline_speed:.1f} tok/s")
        finally:
            self._stop_server()

        return True

    def test_cache_type(self, cache_type_k, cache_type_v=None):
        """
        Test a specific cache-type configuration against the baseline.
        Returns metrics dict.
        """
        if cache_type_v is None:
            cache_type_v = cache_type_k  # Same type for both K and V

        config_name = f"k={cache_type_k},v={cache_type_v}"
        results = {
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
            "config": config_name,
            "text_similarity": 0,
            "tokens_per_sec": 0,
            "speed_ratio": 0,
            "elapsed": 0,
            "output_length": 0,
            "error": None,
        }

        print(f"    Testing {config_name}...")

        if not self._start_server(cache_type_k, cache_type_v):
            results["error"] = "Server failed to start"
            return results

        try:
            # Generate with same prompt
            result = self._generate(QUALITY_PROMPT, max_tokens=500)

            results["tokens_per_sec"] = result["tokens_per_sec"]
            results["elapsed"] = result["elapsed"]
            results["output_length"] = len(result["content"])

            # Speed comparison
            if self.baseline_speed > 0:
                results["speed_ratio"] = result["tokens_per_sec"] / self.baseline_speed

            # Quality comparison against baseline
            if self.baseline_output:
                results["text_similarity"] = self._text_similarity(
                    self.baseline_output, result["content"]
                )

        except Exception as e:
            results["error"] = str(e)
        finally:
            self._stop_server()

        return results

    def run_all_configs(self):
        """Test all cache-type configurations."""
        print(f"\n  Testing {len(CACHE_TYPES)} cache configurations on {self.model_name}")

        # Generate baseline first
        if not self.generate_baseline():
            print("  ERROR: Could not generate baseline. Is llama-server installed?")
            return []

        all_results = []
        for cache_type in CACHE_TYPES:
            if cache_type == "f16":
                continue  # Already have baseline

            result = self.test_cache_type(cache_type, cache_type)
            all_results.append(result)

            sim = result["text_similarity"]
            speed = result["tokens_per_sec"]
            error = result.get("error")

            if error:
                print(f"    [{cache_type}] ERROR: {error}")
            else:
                print(f"    [{cache_type}] similarity={sim:.3f} speed={speed:.1f} tok/s")

        # Also test mixed configs (different K and V types)
        mixed_configs = [
            ("q4_0", "q8_0"),   # Aggressive K, conservative V
            ("q3_K", "q4_0"),   # Very aggressive K, standard V
            ("q5_0", "q4_0"),   # Medium K, aggressive V
            ("q8_0", "q4_0"),   # Conservative K, aggressive V
        ]

        print(f"\n  Testing {len(mixed_configs)} mixed K/V configurations...")
        for k_type, v_type in mixed_configs:
            result = self.test_cache_type(k_type, v_type)
            all_results.append(result)

            sim = result["text_similarity"]
            speed = result["tokens_per_sec"]
            error = result.get("error")

            if error:
                print(f"    [k={k_type},v={v_type}] ERROR: {error}")
            else:
                print(f"    [k={k_type},v={v_type}] similarity={sim:.3f} speed={speed:.1f} tok/s")

        return all_results

    def cleanup(self):
        """Ensure server is stopped."""
        self._stop_server()


def print_results_table(results, baseline_speed):
    """Print a formatted results table."""
    print(f"\n{'='*70}")
    print(f"  {'Config':<20} {'Similarity':>10} {'Speed':>10} {'vs f16':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")

    # Sort by similarity (quality first)
    for r in sorted(results, key=lambda x: x.get("text_similarity", 0), reverse=True):
        if r.get("error"):
            print(f"  {r['config']:<20} {'ERROR':>10} {'':>10} {'':>10}")
            continue
        sim = r["text_similarity"]
        speed = r["tokens_per_sec"]
        ratio = r.get("speed_ratio", 0)
        print(f"  {r['config']:<20} {sim:>10.3f} {speed:>9.1f} {ratio:>9.2f}x")

    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'f16 (baseline)':<20} {'1.000':>10} {baseline_speed:>9.1f} {'1.00x':>10}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    harness = LlamaHarness()
    try:
        results = harness.run_all_configs()
        if results:
            print_results_table(results, harness.baseline_speed)
    finally:
        harness.cleanup()
