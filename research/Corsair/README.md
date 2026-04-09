# Expert Sniper on d-Matrix Corsair

Research mapping the Expert Sniper MoE inference system onto d-Matrix Corsair hardware. The core insight: **dense 122B needs 64 Corsair cards; MoE 122B + Expert Sniper = 2 cards.**

## The Problem

MoE models (Mixture of Experts) have 10-100x more parameters than they activate per token. Traditional hardware treats all weights equally — you need enough memory for the full model. A 122B MoE model on dense hardware requires 64 Corsair cards even though only ~10B parameters are active at any time.

## The Solution

Expert Sniper streams only active experts from slow storage (SSD/LPDDR5) into fast memory (RAM/SRAM), using:
- **LRU expert cache** — keep hot experts in fast memory
- **Routing bias** — steer the router toward already-cached experts (97% SRAM hit rate)
- **Dead expert elimination** — 45.2% of experts never fire, skip them entirely
- **Co-activation prefetch** — predict next-layer experts before they're needed
- **Union-of-experts batching** — 58% dedup when verifying batched tokens

## Why Corsair Is the Perfect Hardware

| | Mac (measured) | Corsair (projected) |
|---|---|---|
| Fast memory | 16 GB unified RAM | 2 GB SRAM @ **150 TB/s** |
| Slow storage | NVMe SSD @ 3-5 GB/s | LPDDR5 @ **400 GB/s** |
| Expert load time | 400 us | **theoretical min 0.01 us** (bandwidth-limited) |
| Expert load speedup | baseline | **100-1000x** (depends on DMA pipeline, needs measurement) |
| 35B 4-bit tok/s | 6.41 | **TBD** (expert I/O eliminated; becomes compute-bound) |
| 122B tok/s | 2.65 | **TBD** (compute-bound projection needs Corsair profiling) |

Corsair's memory hierarchy (fast SRAM + large LPDDR5) mirrors exactly what Expert Sniper was designed for (fast RAM + large SSD). The algorithm ports directly. **What changes is that the bottleneck shifts from I/O to compute** — quantifying the actual throughput requires either a Corsair simulator or hardware access.

> **Note:** Earlier projections of "500+ tok/s" were upper-bound estimates assuming zero compute overhead on expert loads. Real throughput depends on Corsair's compute pipeline latency, SRAM banking, and how much SRAM is available after system overhead. We're presenting measured algorithms + architecture fit, not guaranteed throughput numbers.

## Measured Results (Real Hardware)

All measurements on Apple M4 Mac Mini (16 GB) — the same algorithm that maps to Corsair.

| Metric | Value | Hardware |
|---|---|---|
| MLX Sniper tok/s (35B 4-bit) | **6.41** | M4 Mac Mini |
| Expert cache hit rate (bias=1.5) | **97.1%** | M2 Cloud |
| Dead experts (never fire) | **45.2%** | Qwen3.5-35B calibration |
| Union-of-experts dedup (K=8) | **58.1%** | Real inference, Qwen3.5-35B |
| Distributed inference (2-node) | **1.20 tok/s** | 2x M2 Cloud, HTTP transport |
| Live expert data (35B 4-bit) | **9.0 GB** | Calculated from dead expert analysis |

## SRAM Requirements: Dense vs MoE + Expert Sniper

| Model | Dense SRAM Needed | MoE + Sniper Active Set | Cards (Dense) | Cards (Sniper) | Savings |
|---|---|---|---|---|---|
| 35B | 17.5 GB | 1.41 GB | 9 | **1** | **9x** |
| 122B | 61 GB | 3.51 GB | 31 | **2** | **15x** |
| 397B | ~200 GB | 7.0 GB | 100 | **4** | **25x** |

> **Caveat:** Card counts above are for expert SRAM only. Total cards depend on KV cache, attention weights, router, and system overhead. The "2 cards for 122B" claim assumes most non-expert memory fits in LPDDR5. Actual sizing needs d-Matrix input on usable SRAM after firmware/system overhead.

## Academic Validation

Six peer-reviewed papers independently validate components of this approach:

1. **Mixture of Cache-Conditional Experts** (Dec 2024) — proves routing bias toward cached experts works: >50% cache miss reduction, <0.1% accuracy loss
2. **SAIL** (Oct 2025) — 10.7x speedup via SRAM for dense LLMs. Expert Sniper extends to MoE.
3. **Fate** (Feb 2025) — 97.15% prefetch accuracy using cross-layer gate similarity
4. **Cache Management for MoE** (Sep 2025) — formalizes MoE caching, proves shared cache 3x better
5. **"Not All Models Suit Expert Offloading"** (May 2025) — validates cache size = 2x active experts for 90% hit rate
6. **SambaNova SN40L CoE** (May 2024) — custom hardware with three-tier memory hierarchy IS Expert Sniper in silicon

See [RELATED_WORK.md](RELATED_WORK.md) for full citations and competing systems analysis.

## Files in This Folder

| File | Description |
|---|---|
| [SAIL_PAPER_RELEVANCE.md](SAIL_PAPER_RELEVANCE.md) | How SAIL validates the SRAM approach, d-Matrix hardware specs |
| [RELATED_WORK.md](RELATED_WORK.md) | 6 validating papers + 8 competing systems + techniques to adopt |
| [DISTRIBUTED_RESULTS.md](DISTRIBUTED_RESULTS.md) | 2-node distributed results, maps to multi-card Corsair |
| [corsair_pitch_data.csv](corsair_pitch_data.csv) | Measured → projected data for each metric |
| [sram_projections.csv](sram_projections.csv) | Detailed SRAM allocation scenarios per model size |
| [results_verified.csv](results_verified.csv) | 42 rows of verified benchmarks across quantization levels |

## d-Matrix Corsair Hardware (from Nov 2024 whitepaper)

| Spec | Per Card | 8-Card Server |
|---|---|---|
| SRAM (Performance Memory) | 2 GB | 16 GB |
| SRAM Bandwidth | 150 TB/s | 1,200 TB/s |
| LPDDR5X (Capacity Memory) | 256 GB | 2 TB |
| LPDDR5X Bandwidth | ~400 GB/s | 3.2 TB/s |
| Compute (MXINT4) | 9,600 TFLOPS | 76,800 TFLOPS |
| Inter-card Link | DMX Link, 115 ns latency | Full mesh |

## Open Questions for d-Matrix

These are the gaps we can't close without hardware access or d-Matrix engineering input:

1. **Real expert load latency from SRAM** — our 0.01 us estimate is the bandwidth-limited theoretical minimum. What's the actual DMA pipeline latency for a ~1.7 MB expert block?
2. **Usable SRAM for expert caching** — 2 GB per card raw. After firmware, KV cache, attention buffers, router weights — how much is left for expert cache?
3. **LPDDR5 cold-expert path performance** — when SRAM misses (even 3% at bias=1.5), the LPDDR5 path at 400 GB/s is slower than H100 HBM (3.35 TB/s). Is there a prefetch/async path to hide this?
4. **MoE gating in the compute pipeline** — routing bias modifies gate logits before softmax. Can this run in Corsair's dataflow without a host round-trip?
5. **Inter-card expert migration** — DMX Link is 115 ns latency. What's the throughput for moving a 1.7 MB expert block across cards?
6. **Simulator or dev board access** — one real measurement replaces all bandwidth-ratio projections
7. **Quality validation at bias=1.5** — we've validated perplexity at bias=1.0. Eval benchmarks (MMLU, GSM8K) at bias=1.5 are still pending

## What We Bring vs What We Need

| We bring (proven) | We need from d-Matrix |
|---|---|
| Working MoE expert streaming code (MLX + llama.cpp) | Hardware access or simulator |
| Routing bias algorithm (97% hit rate, training-free) | Real SRAM latency numbers |
| Dead expert analysis (45% eliminated) | Usable SRAM sizing after overhead |
| Union-of-experts batching (58% dedup) | MoE gating support in compute pipeline |
| Distributed expert partitioning (proven on 2 Macs) | DMX Link expert migration benchmarks |
| 42-row verified benchmark suite | Quality eval at bias=1.5 (we can do this) |

## External References

- [Expert Sniper code (MLX)](https://huggingface.co/waltgrace/mlx-expert-sniper)
- [Expert Sniper code (llama.cpp)](https://huggingface.co/waltgrace/llama-cpp-expert-sniper)
- [Gimlet Labs: Spec Decode on Corsair](https://gimletlabs.ai/blog/low-latency-spec-decode-corsair) — uses gpt-oss-120b but doesn't address MoE expert loading
- [d-Matrix whitepaper (Nov 2024)](https://d-matrix.ai/) — verified hardware specs
