#!/usr/bin/env python3
"""
Test distributed MoE inference.

Run this on the Mac mini AFTER starting expert_worker.py on the MacBook.

Tests:
  1. Ping worker — verify connection
  2. Single-layer remote compute — verify correctness
  3. Compare: local vs remote expert output — should match
  4. Full forward pass timing — measure distributed speedup
"""

import argparse
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from expert_client import RemoteExpertClient

BITS = 4
GROUP_SIZE = 64


def test_ping(client):
    """Test 1: Can we reach the worker?"""
    print("\n[Test 1] Ping worker...")
    t0 = time.time()
    client.connect()
    latency = (time.time() - t0) * 1000
    print(f"  Connected in {latency:.1f}ms")
    return True


def test_single_layer(client, local_expert_dir):
    """Test 2: Single-layer remote compute — does the math match?"""
    print("\n[Test 2] Single-layer remote compute...")

    # Create fake hidden state
    mx.random.seed(42)
    hidden = mx.random.normal((1, 3072))  # single token, hidden_dim=3072

    # Fake expert IDs and weights (8 experts)
    expert_ids = [3, 7, 15, 42, 100, 128, 200, 255]
    routing_weights = [0.15, 0.14, 0.13, 0.12, 0.12, 0.12, 0.11, 0.11]

    # Remote compute
    t0 = time.time()
    result = client.compute(
        layer_idx=24,  # first remote layer
        hidden_state=hidden,
        expert_ids=expert_ids,
        routing_weights=routing_weights,
    )
    mx.eval(result)
    remote_time = time.time() - t0

    print(f"  Remote result: shape={result.shape}, mean={result.mean().item():.6f}")
    print(f"  Time: {remote_time*1000:.1f}ms")

    # If we have local experts for the same layer, compare
    import os
    local_layer = os.path.join(local_expert_dir, "layer_24.safetensors")
    if os.path.exists(local_layer):
        print("  Comparing against local computation...")
        data = mx.load(local_layer)

        local_output = mx.zeros_like(hidden)
        for i, (eid, w) in enumerate(zip(expert_ids, routing_weights)):
            gate = mx.quantized_matmul(
                hidden, data["gate_proj.weight"][eid],
                scales=data["gate_proj.scales"][eid],
                biases=data["gate_proj.biases"][eid],
                transpose=True, group_size=GROUP_SIZE, bits=BITS,
            )
            up = mx.quantized_matmul(
                hidden, data["up_proj.weight"][eid],
                scales=data["up_proj.scales"][eid],
                biases=data["up_proj.biases"][eid],
                transpose=True, group_size=GROUP_SIZE, bits=BITS,
            )
            down = mx.quantized_matmul(
                nn.silu(gate) * up,
                data["down_proj.weight"][eid],
                scales=data["down_proj.scales"][eid],
                biases=data["down_proj.biases"][eid],
                transpose=True, group_size=GROUP_SIZE, bits=BITS,
            )
            local_output = local_output + w * down

        mx.eval(local_output)
        diff = mx.abs(result - local_output).max().item()
        print(f"  Local result: mean={local_output.mean().item():.6f}")
        print(f"  Max diff: {diff:.6f}")
        if diff < 0.01:
            print(f"  >>> MATCH")
        else:
            print(f"  >>> MISMATCH — investigate")
    else:
        print(f"  (No local layer 24 to compare against)")

    return True


def test_latency_sweep(client):
    """Test 3: Measure latency across multiple calls."""
    print("\n[Test 3] Latency sweep (20 calls)...")

    mx.random.seed(42)
    hidden = mx.random.normal((1, 3072))
    expert_ids = [3, 7, 15, 42, 100, 128, 200, 255]
    routing_weights = [0.15, 0.14, 0.13, 0.12, 0.12, 0.12, 0.11, 0.11]

    times = []
    for i in range(20):
        t0 = time.time()
        result = client.compute(
            layer_idx=24 + (i % 16),
            hidden_state=hidden,
            expert_ids=expert_ids,
            routing_weights=routing_weights,
        )
        mx.eval(result)
        elapsed = time.time() - t0
        times.append(elapsed * 1000)

    avg = sum(times) / len(times)
    p50 = sorted(times)[10]
    p99 = sorted(times)[19]
    print(f"  Avg: {avg:.1f}ms  P50: {p50:.1f}ms  P99: {p99:.1f}ms")
    print(f"  Stats: {client.stats()}")

    # Estimate tok/s for 16 remote layers
    remote_layers = 16
    per_token_remote = avg * remote_layers
    print(f"\n  Estimated remote overhead: {per_token_remote:.0f}ms for {remote_layers} layers")
    print(f"  If local layers take ~25ms: total ~{per_token_remote + 25:.0f}ms per token")
    print(f"  Estimated tok/s: {1000 / (per_token_remote + 25):.1f}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default="macbook.local:9000", help="Worker address (host:port)")
    parser.add_argument("--local-expert-dir", default="", help="Local expert dir for comparison test")
    args = parser.parse_args()

    host, port = args.worker.split(":")
    port = int(port)

    print("=" * 60)
    print("  Distributed MoE Expert Sniper — Test Suite")
    print(f"  Worker: {host}:{port}")
    print("=" * 60)

    client = RemoteExpertClient(host, port)

    try:
        test_ping(client)
        test_single_layer(client, args.local_expert_dir)
        test_latency_sweep(client)

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n  TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
