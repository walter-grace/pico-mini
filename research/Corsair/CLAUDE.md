# Corsair Research — Claude Code Guide

This folder contains research on mapping Expert Sniper (MoE expert streaming) to d-Matrix Corsair hardware.

## What is this?

Expert Sniper runs MoE language models larger than available RAM by streaming only active experts from slow storage (SSD) into fast memory (RAM). Corsair is d-Matrix's AI accelerator with 2 GB SRAM (150 TB/s) + 256 GB LPDDR5 (~400 GB/s) — a memory hierarchy that maps directly to Expert Sniper's design.

## File map

```
corsair/
  README.md                  ← Start here. Full research narrative, tables, open questions
  SAIL_PAPER_RELEVANCE.md    ← SAIL paper validation + d-Matrix hardware specs table
  RELATED_WORK.md            ← 6 validating papers + 8 competing systems
  DISTRIBUTED_RESULTS.md     ← 2-node distributed inference → multi-card Corsair mapping
  corsair_pitch_data.csv     ← Key: measured (Mac) → projected (Corsair) metrics
  sram_projections.csv       ← SRAM allocation per model size (1-card, 8-card scenarios)
  results_verified.csv       ← 42 verified benchmarks, V=verified A=agent-tested
```

## Key numbers to know

- **6.41 tok/s** — MLX Expert Sniper on M4 Mac Mini, Qwen3.5-35B 4-bit (measured)
- **97.1%** — SRAM/cache hit rate with routing bias=1.5 (measured)
- **45.2%** — dead experts that never fire, can be eliminated (measured)
- **58.1%** — union-of-experts dedup at batch size K=8 (measured on real inference)
- **1.20 tok/s** — distributed 2-node inference over HTTP (measured)

## What's measured vs projected

**Measured (real hardware, reproducible):**
- All tok/s numbers in results_verified.csv
- Cache hit rates, dead expert counts, E_union dedup percentages
- Distributed inference latency breakdown

**Projected (bandwidth-ratio math, NOT validated on Corsair):**
- All "Corsair projected" columns in corsair_pitch_data.csv and sram_projections.csv
- The "500+ tok/s" and "2000+ tok/s" numbers are upper bounds assuming zero compute overhead
- Card count estimates assume ALL usable SRAM goes to expert cache (optimistic)

## Common asks

**"What's the core claim?"**
→ MoE models only activate 3-17B of 35-397B parameters per token. Expert Sniper caches the hot experts in SRAM and biases the router to reuse them. Dense 122B needs 31 Corsair cards; MoE + Expert Sniper = 2 cards for expert SRAM.

**"What's the weakest part?"**
→ The throughput projections. We proved the algorithm works on Mac. We showed the memory hierarchy maps. But actual Corsair throughput depends on compute pipeline latency, SRAM banking, DMA depth — none of which we can measure without hardware. See "Open Questions for d-Matrix" in README.md.

**"Where's the code?"**
→ Not in this folder. Code lives in sibling directories:
  - `../mlx-sniper/` — MLX implementation (the one that gets 6.41 tok/s)
  - `../llama-cpp/` — llama.cpp patches
  - `../distributed/` — 2-node distributed inference
  - `../scripts/` — benchmark and experiment scripts
  - HuggingFace: https://huggingface.co/waltgrace/mlx-expert-sniper

**"What papers support this?"**
→ See RELATED_WORK.md. The closest is "Mixture of Cache-Conditional Experts" (Dec 2024) which independently proves routing bias toward cached experts works with <0.1% accuracy loss. SAIL (Oct 2025) proves SRAM accelerates LLM inference 10.7x for dense models — Expert Sniper extends this to MoE.

**"What's the ask for d-Matrix?"**
→ README.md has a full "Open Questions" section + "What We Bring vs What We Need" table. TL;DR: we bring working algorithms + benchmarks, we need hardware access or a simulator to replace projections with real numbers.
