# kv-lab

Autonomous KV cache compression research agent. Runs overnight, tests theories, keeps winners, reverts failures.

Inspired by [Karpathy's autoresearch](https://x.com/karpathy) — one file to modify, one metric to optimize, git-based keep/discard.

## Quick Start

```bash
# Terminal 1: Start MLX engine
cd ~/Desktop/pico-mini
python3 mlx/mlx_engine.py

# Terminal 2: Run kv-lab
cd ~/Desktop/pico-mini/research

# 1-hour test run
python3 kv_lab.py --hours 1

# Overnight (12 hours, ~144 experiments)
python3 kv_lab.py --hours 12

# With OpenRouter for smart planning (optional)
export OPENROUTER_API_KEY="your-key"
python3 kv_lab.py --hours 12
```

## How It Works

```
PLAN → TEST → JUDGE → LOG → repeat
```

1. **Plan**: Kimi 2.5 (via OpenRouter) proposes an experiment, or falls back to grid search
2. **Test**: Compress reference KV cache, decompress, measure quality (5-min timeout)
3. **Judge**: cosine >= 0.99 AND ratio > best? KEEP. Otherwise REVERT.
4. **Log**: Append to results.tsv

## Metric

**Maximize `compression_ratio` where `cosine_similarity >= 0.99`**

## Files

| File | What |
|---|---|
| `kv_lab.py` | Main loop — runs forever |
| `experiment.py` | Single experiment with 5-min timeout |
| `harness.py` | Loads model once, measures compression |
| `planner.py` | Kimi 2.5 via OpenRouter + fallback grid search |
| `techniques.py` | 8 compression techniques (the file that evolves) |
| `lean_prompts.py` | Lean 4 proof prompts for Harmonic AI |
| `program.md` | Instructions for the planning LLM |
| `results.tsv` | Experiment log (auto-generated) |

## 8 Seed Techniques

1. **baseline_minmax** — Per-group min-max quantization (current production)
2. **polar_quant** — Polar coordinates, angles get more bits
3. **qjl_1bit** — Random projection + sign bit (1-bit per dim)
4. **hadamard_rotate** — Walsh-Hadamard rotation before quantization
5. **mixed_kv** — Different bits for K vs V cache
6. **per_layer_adaptive** — More bits for first/last layers
7. **residual_correction** — Base quantization + QJL on residual
8. **lloyd_max** — Non-uniform quantizer fit to data

## Monitoring

```bash
# Watch results live
tail -f research/results.tsv

# See git history of experiments
git log --oneline -20

# Check current best
grep KEEP research/results.tsv | sort -t$'\t' -k8 -rn | head -5
```

## Lean Proofs

Generate Lean 4 prompts for Harmonic AI verification:

```bash
python3 lean_prompts.py
```

Covers: QJL unbiased estimator, Hadamard norm preservation, PolarQuant error bounds, min-max error bounds, JL lemma, residual decomposition.
