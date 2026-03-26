#!/usr/bin/env python3
"""
Single experiment runner with 5-minute timeout.
Imports technique, runs harness, returns metrics.
"""

import signal
import sys
import traceback

from techniques import TECHNIQUES


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Experiment exceeded 5-minute timeout")


def run_experiment(harness, technique_name, params, timeout_seconds=300):
    """
    Run a single compression experiment.

    Args:
        harness: Harness instance with loaded model + reference KV cache
        technique_name: Key in TECHNIQUES registry
        params: Dict of technique parameters

    Returns:
        dict with metrics + error field (None on success)
    """
    result = {
        "technique": technique_name,
        "params": params,
        "cosine_sim": 0,
        "mse": 0,
        "compression_ratio": 0,
        "original_mb": 0,
        "compressed_mb": 0,
        "compress_time": 0,
        "decompress_time": 0,
        "error": None,
    }

    # Validate technique exists
    if technique_name not in TECHNIQUES:
        result["error"] = f"Unknown technique: {technique_name}"
        return result

    compress_fn, decompress_fn = TECHNIQUES[technique_name]

    # Set timeout (Unix only)
    old_handler = None
    if sys.platform != "win32":
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        metrics = harness.test_technique(compress_fn, decompress_fn, **params)
        result.update(metrics)
    except TimeoutError:
        result["error"] = f"Timeout after {timeout_seconds}s"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        if sys.platform != "win32" and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    return result
