#!/usr/bin/env python3
"""
kv-lab: Autonomous KV Cache Research Agent
Inspired by Karpathy's autoresearch — one file to modify, one metric to optimize,
git-based keep/discard, ~12 experiments/hour.

Usage:
    python3 kv_lab.py --hours 12 --backend mlx
    python3 kv_lab.py --hours 1   # quick test
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# Add parent dir for mlx imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiment import run_experiment
from harness import Harness
from lean_prompts import generate_lean_file
from planner import propose_experiment
from techniques import TECHNIQUES

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.tsv")
TECHNIQUES_FILE = os.path.join(os.path.dirname(__file__), "techniques.py")


def load_results():
    """Load results history from TSV."""
    if not os.path.exists(RESULTS_FILE):
        return []
    with open(RESULTS_FILE) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("timestamp")]


def append_result(exp_id, technique, params, metrics, status, description, planner="grid"):
    """Append one row to results.tsv."""
    write_header = not os.path.exists(RESULTS_FILE) or os.path.getsize(RESULTS_FILE) == 0

    with open(RESULTS_FILE, "a") as f:
        if write_header:
            f.write("timestamp\texp_id\ttechnique\tparams\tbackend\tcosine_sim\tmse\tratio\tcompress_s\tdecompress_s\tstatus\tplanner\tdescription\n")

        params_str = json.dumps(params, sort_keys=True)
        row = (
            f"{datetime.now().isoformat()}\t"
            f"{exp_id}\t"
            f"{technique}\t"
            f"{params_str}\t"
            f"mlx\t"
            f"{metrics.get('cosine_sim', 0):.6f}\t"
            f"{metrics.get('mse', 0):.8f}\t"
            f"{metrics.get('compression_ratio', 0):.2f}\t"
            f"{metrics.get('compress_time', 0):.3f}\t"
            f"{metrics.get('decompress_time', 0):.3f}\t"
            f"{status}\t"
            f"{planner}\t"
            f"{description}\n"
        )
        f.write(row)


def git_commit(message):
    """Git commit current state."""
    try:
        subprocess.run(["git", "add", "-A"], cwd=os.path.dirname(__file__), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=os.path.dirname(__file__),
            capture_output=True,
        )
    except Exception as e:
        print(f"  Git commit failed: {e}")


def git_revert():
    """Revert last commit."""
    try:
        subprocess.run(
            ["git", "revert", "HEAD", "--no-edit"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
        )
    except Exception as e:
        print(f"  Git revert failed: {e}")


def read_techniques_source():
    """Read current techniques.py source."""
    with open(TECHNIQUES_FILE) as f:
        return f.read()


def print_banner():
    print("\n" + "=" * 60)
    print("  kv-lab: Autonomous KV Cache Research Agent")
    print("  Metric: maximize compression_ratio where cosine >= 0.99")
    print("=" * 60)


def print_result(exp_id, technique, metrics, status):
    """Pretty-print one experiment result."""
    cosine = metrics.get("cosine_sim", 0)
    ratio = metrics.get("compression_ratio", 0)
    error = metrics.get("error")

    icon = "KEEP" if status == "KEEP" else "REVERT" if status == "REVERT" else "ERROR"
    color_start = "\033[92m" if status == "KEEP" else "\033[91m" if status == "REVERT" else "\033[93m"
    color_end = "\033[0m"

    print(f"\n  [{color_start}{icon}{color_end}] exp-{exp_id}: {technique}")
    print(f"    cosine={cosine:.4f}  ratio={ratio:.2f}x  ", end="")
    if error:
        print(f"error={error}")
    else:
        compress_t = metrics.get("compress_time", 0)
        decompress_t = metrics.get("decompress_time", 0)
        print(f"compress={compress_t:.2f}s  decompress={decompress_t:.2f}s")


def run_lab(hours=12, quality_floor=0.99, planner_mode="auto"):
    """
    Main experiment loop.
    planner_mode: "auto" (use API if available, else grid), "api" (force API), "grid" (force grid)
    """
    print_banner()
    print(f"  Planner mode: {planner_mode}")

    # Load model once
    print("\n  Initializing harness...")
    harness = Harness()

    # State
    best_ratio = 0.0
    consecutive_reverts = 0
    exp_id = 0
    results_history = load_results()
    start_time = time.time()
    end_time = start_time + hours * 3600

    # Find best ratio from history
    for line in results_history:
        parts = line.split("\t")
        if len(parts) >= 8 and parts[10] == "KEEP":
            try:
                ratio = float(parts[7])
                best_ratio = max(best_ratio, ratio)
            except (ValueError, IndexError):
                pass

    print(f"\n  Starting from best ratio: {best_ratio:.2f}x")
    print(f"  Quality floor: cosine >= {quality_floor}")
    print(f"  Running for {hours} hours ({hours * 12} experiments estimated)")
    print(f"  Available techniques: {list(TECHNIQUES.keys())}")

    keeps = 0
    reverts = 0
    errors = 0

    while time.time() < end_time:
        exp_id += 1
        elapsed_hours = (time.time() - start_time) / 3600
        remaining_hours = (end_time - time.time()) / 3600

        print(f"\n{'─' * 60}")
        print(f"  Experiment {exp_id} | {elapsed_hours:.1f}h elapsed | {remaining_hours:.1f}h remaining")
        print(f"  Best: {best_ratio:.2f}x | Keeps: {keeps} | Reverts: {reverts} | Errors: {errors}")

        # 1. PLAN — ask Kimi 2.5 (or fallback) for next experiment
        print("  Planning...")
        techniques_source = read_techniques_source()
        proposal = propose_experiment(
            results_history, techniques_source, best_ratio, consecutive_reverts,
            planner_mode=planner_mode,
        )

        technique = proposal.get("technique", "baseline_minmax")
        params = proposal.get("params", {})
        description = proposal.get("description", "")
        hypothesis = proposal.get("hypothesis", "")
        which_planner = proposal.get("planner", "grid")

        print(f"  Planner: {which_planner}")
        print(f"  Technique: {technique}")
        print(f"  Params: {json.dumps(params)}")
        print(f"  Hypothesis: {hypothesis}")

        # Validate technique exists
        if technique not in TECHNIQUES:
            print(f"  ERROR: Unknown technique '{technique}', skipping")
            append_result(exp_id, technique, params, {}, "ERROR", f"Unknown technique: {technique}", planner=which_planner)
            results_history.append(f"exp-{exp_id}\t{technique}\tERROR\tUnknown technique")
            errors += 1
            consecutive_reverts += 1
            continue

        # 2. COMMIT — snapshot before experiment
        git_commit(f"exp-{exp_id}: {technique} {json.dumps(params)} — {description}")

        # 3. TEST — run experiment with timeout
        print("  Running experiment (5 min timeout)...")
        metrics = run_experiment(harness, technique, params, timeout_seconds=300)

        if metrics.get("error"):
            print(f"  ERROR: {metrics['error']}")
            append_result(exp_id, technique, params, metrics, "ERROR", metrics["error"], planner=which_planner)
            results_history.append(
                f"exp-{exp_id}\t{technique}\t{json.dumps(params)}\tERROR\t{metrics['error']}"
            )
            git_revert()
            errors += 1
            consecutive_reverts += 1
            print_result(exp_id, technique, metrics, "ERROR")
            continue

        # 4. JUDGE — keep or revert
        cosine = metrics.get("cosine_sim", 0)
        ratio = metrics.get("compression_ratio", 0)

        if cosine >= quality_floor and ratio > best_ratio:
            # KEEP — new best!
            status = "KEEP"
            best_ratio = ratio
            consecutive_reverts = 0
            keeps += 1
            git_commit(f"exp-{exp_id} KEEP: {technique} ratio={ratio:.2f}x cosine={cosine:.4f}")

            # Generate Lean proof prompts for Harmonic AI
            lean_path = generate_lean_file(technique, metrics)
            if lean_path:
                print(f"  Lean proof prompt: {lean_path}")
        else:
            # REVERT — doesn't beat best or quality too low
            status = "REVERT"
            reason = []
            if cosine < quality_floor:
                reason.append(f"cosine {cosine:.4f} < {quality_floor}")
            if ratio <= best_ratio:
                reason.append(f"ratio {ratio:.2f}x <= best {best_ratio:.2f}x")
            git_revert()
            reverts += 1
            consecutive_reverts += 1

        # 5. LOG
        append_result(exp_id, technique, params, metrics, status, description, planner=which_planner)
        results_history.append(
            f"exp-{exp_id}\t{technique}\t{json.dumps(params)}\t"
            f"cosine={cosine:.4f}\tratio={ratio:.2f}x\t{status}"
        )
        print_result(exp_id, technique, metrics, status)

    # Summary
    total_time = (time.time() - start_time) / 3600
    print(f"\n{'=' * 60}")
    print(f"  kv-lab complete!")
    print(f"  Duration: {total_time:.1f} hours")
    print(f"  Experiments: {exp_id}")
    print(f"  Keeps: {keeps} | Reverts: {reverts} | Errors: {errors}")
    print(f"  Best compression ratio: {best_ratio:.2f}x (cosine >= {quality_floor})")
    print(f"  Results: {RESULTS_FILE}")
    print(f"{'=' * 60}\n")


def test_planner():
    """Dry-run: test both planners without loading the model."""
    from planner import get_api_key

    print("\n" + "=" * 60)
    print("  kv-lab planner test (no model loaded)")
    print("=" * 60)

    fake_history = [
        "exp-1\tbaseline_minmax\t{}\tcosine=0.9950\tratio=4.00x\tKEEP",
        "exp-2\tpolar_quant\t{}\tcosine=0.9800\tratio=5.20x\tREVERT",
    ]

    # Test grid planner
    print("\n  --- Grid Search Planner ---")
    proposal = propose_experiment(fake_history, "", 4.0, 0, planner_mode="grid")
    print(f"  Planner: {proposal.get('planner')}")
    print(f"  Technique: {proposal.get('technique')}")
    print(f"  Params: {json.dumps(proposal.get('params', {}))}")
    print(f"  Hypothesis: {proposal.get('hypothesis')}")

    # Test API planner
    print("\n  --- Kimi 2.5 API Planner ---")
    api_key = get_api_key()
    if api_key:
        print(f"  API key found: {api_key[:8]}...")
        proposal = propose_experiment(fake_history, "", 4.0, 0, planner_mode="api")
        print(f"  Planner: {proposal.get('planner')}")
        print(f"  Technique: {proposal.get('technique')}")
        print(f"  Params: {json.dumps(proposal.get('params', {}))}")
        print(f"  Hypothesis: {proposal.get('hypothesis')}")
    else:
        print("  No API key found. Set OPENROUTER_API_KEY or ~/.mac-code/openrouter.json")
        print("  Skipping API test.")

    print(f"\n{'=' * 60}\n")


def compare_planners(hours=1, quality_floor=0.99):
    """
    A/B test: run both planners on alternating experiments.
    Same harness, same techniques, different brains.
    """
    print_banner()
    print("  MODE: A/B comparison — grid vs API on alternating experiments")

    from planner import get_api_key
    if not get_api_key():
        print("\n  ERROR: --compare requires an API key for the API planner.")
        print("  Set OPENROUTER_API_KEY or ~/.mac-code/openrouter.json")
        return

    print("\n  Initializing harness...")
    harness = Harness()

    best_ratio = {"api": 0.0, "grid": 0.0}
    keeps = {"api": 0, "grid": 0}
    reverts = {"api": 0, "grid": 0}
    errors = {"api": 0, "grid": 0}

    results_history = load_results()
    start_time = time.time()
    end_time = start_time + hours * 3600
    exp_id = 0

    while time.time() < end_time:
        exp_id += 1
        # Alternate: odd = grid, even = api
        mode = "grid" if exp_id % 2 == 1 else "api"
        elapsed = (time.time() - start_time) / 3600

        print(f"\n{'─' * 60}")
        print(f"  Experiment {exp_id} [{mode.upper()}] | {elapsed:.1f}h elapsed")
        print(f"  Grid: {keeps['grid']} keeps, {best_ratio['grid']:.2f}x best")
        print(f"  API:  {keeps['api']} keeps, {best_ratio['api']:.2f}x best")

        techniques_source = read_techniques_source()
        proposal = propose_experiment(
            results_history, techniques_source,
            max(best_ratio["api"], best_ratio["grid"]),
            0, planner_mode=mode,
        )

        technique = proposal.get("technique", "baseline_minmax")
        params = proposal.get("params", {})
        description = proposal.get("description", "")
        which_planner = proposal.get("planner", mode)

        print(f"  Planner: {which_planner} | Technique: {technique}")
        print(f"  Params: {json.dumps(params)}")

        if technique not in TECHNIQUES:
            errors[which_planner] += 1
            append_result(exp_id, technique, params, {}, "ERROR", f"Unknown: {technique}", planner=which_planner)
            continue

        print("  Running...")
        metrics = run_experiment(harness, technique, params, timeout_seconds=300)

        if metrics.get("error"):
            errors[which_planner] += 1
            append_result(exp_id, technique, params, metrics, "ERROR", metrics["error"], planner=which_planner)
            print_result(exp_id, technique, metrics, "ERROR")
            continue

        cosine = metrics.get("cosine_sim", 0)
        ratio = metrics.get("compression_ratio", 0)

        if cosine >= quality_floor and ratio > best_ratio[which_planner]:
            status = "KEEP"
            best_ratio[which_planner] = ratio
            keeps[which_planner] += 1
        else:
            status = "REVERT"
            reverts[which_planner] += 1

        append_result(exp_id, technique, params, metrics, status, description, planner=which_planner)
        results_history.append(f"exp-{exp_id}\t{technique}\t{which_planner}\t{status}")
        print_result(exp_id, technique, metrics, status)

    # Final scoreboard
    total_time = (time.time() - start_time) / 3600
    print(f"\n{'=' * 60}")
    print(f"  A/B COMPARISON RESULTS ({total_time:.1f} hours, {exp_id} experiments)")
    print(f"{'=' * 60}")
    print(f"  {'':20} {'GRID':>10} {'API':>10}")
    print(f"  {'Keeps':20} {keeps['grid']:>10} {keeps['api']:>10}")
    print(f"  {'Reverts':20} {reverts['grid']:>10} {reverts['api']:>10}")
    print(f"  {'Errors':20} {errors['grid']:>10} {errors['api']:>10}")
    print(f"  {'Best ratio':20} {best_ratio['grid']:>9.2f}x {best_ratio['api']:>9.2f}x")
    winner = "API (Kimi 2.5)" if best_ratio["api"] > best_ratio["grid"] else "Grid Search" if best_ratio["grid"] > best_ratio["api"] else "Tie"
    print(f"\n  Winner: {winner}")
    print(f"{'=' * 60}\n")


def run_llama_test(model_path=None):
    """Test llama.cpp cache-type flags on 35B (or 9B) GGUF model."""
    from harness_llama import LlamaHarness, CACHE_TYPES, print_results_table

    print_banner()
    print("  MODE: llama.cpp cache-type flag testing")

    harness = LlamaHarness(model_path)
    try:
        results = harness.run_all_configs()
        if results:
            print_results_table(results, harness.baseline_speed)

            # Append to results.tsv
            for r in results:
                metrics = {
                    "cosine_sim": r.get("text_similarity", 0),
                    "mse": 0,
                    "compression_ratio": 0,  # Can't measure directly from server
                    "compress_time": 0,
                    "decompress_time": r.get("elapsed", 0),
                }
                status = "KEEP" if r.get("text_similarity", 0) >= 0.8 else "REVERT"
                append_result(
                    0, f"llama_{r['config']}", {"cache_type_k": r["cache_type_k"], "cache_type_v": r["cache_type_v"]},
                    metrics, status,
                    f"llama.cpp {harness.model_name} {r['config']} {r.get('tokens_per_sec', 0):.1f} tok/s",
                    planner="llama",
                )
    finally:
        harness.cleanup()


def run_both(hours=1, quality_floor=0.99, planner_mode="auto", model_path=None):
    """
    Run BOTH backends:
    1. llama.cpp cache-type flag sweep (one-time, ~20 min)
    2. MLX compression experiments (remaining time)
    """
    print_banner()
    print("  MODE: BOTH — llama.cpp flags + MLX compression")

    start_time = time.time()
    total_seconds = hours * 3600

    # Phase 1: llama.cpp cache flags (~20 min)
    print(f"\n{'='*60}")
    print("  PHASE 1: llama.cpp cache-type flags (35B/9B GGUF)")
    print(f"{'='*60}")

    try:
        from harness_llama import LlamaHarness, print_results_table

        llama_harness = LlamaHarness(model_path)
        results = llama_harness.run_all_configs()
        if results:
            print_results_table(results, llama_harness.baseline_speed)
            for r in results:
                metrics = {
                    "cosine_sim": r.get("text_similarity", 0),
                    "mse": 0,
                    "compression_ratio": 0,
                    "compress_time": 0,
                    "decompress_time": r.get("elapsed", 0),
                }
                status = "KEEP" if r.get("text_similarity", 0) >= 0.8 else "REVERT"
                append_result(
                    0, f"llama_{r['config']}", {"cache_type_k": r["cache_type_k"], "cache_type_v": r["cache_type_v"]},
                    metrics, status,
                    f"llama.cpp {llama_harness.model_name} {r.get('tokens_per_sec', 0):.1f} tok/s",
                    planner="llama",
                )
        llama_harness.cleanup()
    except FileNotFoundError as e:
        print(f"  Skipping llama.cpp: {e}")
    except Exception as e:
        print(f"  llama.cpp error: {e}")

    # Phase 2: MLX compression (remaining time)
    elapsed = time.time() - start_time
    remaining_hours = max(0.1, (total_seconds - elapsed) / 3600)

    print(f"\n{'='*60}")
    print(f"  PHASE 2: MLX compression experiments ({remaining_hours:.1f}h remaining)")
    print(f"{'='*60}")

    run_lab(hours=remaining_hours, quality_floor=quality_floor, planner_mode=planner_mode)


def main():
    parser = argparse.ArgumentParser(description="kv-lab: Autonomous KV Cache Research Agent")
    parser.add_argument("--hours", type=float, default=12, help="Hours to run (default: 12)")
    parser.add_argument("--quality-floor", type=float, default=0.99, help="Min cosine similarity (default: 0.99)")
    parser.add_argument("--planner", choices=["auto", "api", "grid"], default="auto",
                        help="Planner mode: auto (API if key exists), api (force), grid (no API)")
    parser.add_argument("--test-planner", action="store_true",
                        help="Dry-run: test both planners without loading the model")
    parser.add_argument("--compare", action="store_true",
                        help="A/B test: alternate grid vs API planners, compare results")
    parser.add_argument("--llama", action="store_true",
                        help="Test llama.cpp cache-type flags (35B or 9B GGUF)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path for llama.cpp testing (auto-detects 35B/9B)")
    parser.add_argument("--both", action="store_true",
                        help="Run BOTH: MLX compression + llama.cpp cache flags")
    args = parser.parse_args()

    if args.test_planner:
        test_planner()
    elif args.llama:
        run_llama_test(model_path=args.model)
    elif args.both:
        run_both(hours=args.hours, quality_floor=args.quality_floor,
                 planner_mode=args.planner, model_path=args.model)
    elif args.compare:
        compare_planners(hours=args.hours, quality_floor=args.quality_floor)
    else:
        run_lab(hours=args.hours, quality_floor=args.quality_floor, planner_mode=args.planner)


if __name__ == "__main__":
    main()
