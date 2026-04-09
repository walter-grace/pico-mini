# SAIL Paper — Why It Validates Expert Sniper on Corsair

## Paper
**SAIL: SRAM-Accelerated LLM Inference System with Lookup-Table-based GEMV**
- arxiv: 2509.25853 (October 2025)
- Key result: **10.7x speedup** and **19.9x tokens-per-dollar** vs CPU baseline
- Hardware: C-SRAM (compute-capable SRAM) adjacent to LLC cache slices

## What SAIL Proves

1. **SRAM eliminates the memory wall for LLM inference**
   - 90% of token generation time is data movement, not compute
   - SRAM bandwidth (on-die) removes this bottleneck entirely
   - LUT-based GEMV handles 2/3/4-bit quantized weights natively

2. **Low-precision in-memory compute works**
   - LUT-GEMV: store weight combinations in lookup tables
   - Arbitrary bit precision (2-bit, 3-bit, 4-bit) with minimal overhead
   - Pattern-Aware LUT optimization reduces cycles by 13.8%

3. **Only 2% hardware overhead for a single new instruction**
   - C-SRAM arrays function as both compute and storage
   - Dual functionality: normal cache when idle, compute when active

## What SAIL Does NOT Do

- **No MoE models tested** — only dense (Llama-2-7B, Llama-2-13B)
- **No expert-level caching or routing** — treats all weights equally
- **No selective loading** — must fit entire model in SRAM/cache
- **Simulation only** — gem5 simulator, not real silicon

## How Expert Sniper Extends SAIL

| SAIL (Dense) | Expert Sniper (MoE) |
|-------------|-------------------|
| All weights must be in SRAM | Only active experts (3-17B of 35-397B) need SRAM |
| Model size = SRAM requirement | Active set = SRAM requirement (10-100x smaller) |
| No routing intelligence | Routing bias steers 97% of loads to cached/SRAM experts |
| No dead weight elimination | 45.2% of experts never fire — eliminated from cache |
| No predictive loading | Co-activation prefetch predicts next-layer experts |

## The Combined Pitch

"SAIL proves SRAM accelerates dense LLM inference 10.7x by eliminating the memory wall.
Expert Sniper extends this to MoE models, shrinking the SRAM requirement from full model
size to just the active expert set — enabling 122B models on 2 Corsair cards instead of 64."

### SRAM Requirements: Dense vs MoE + Expert Sniper

| Model | Dense SRAM | MoE Active Set | Cards (Dense) | Cards (MoE+Sniper) | Savings |
|-------|-----------|---------------|--------------|--------------------|---------| 
| 35B model | 17.5 GB | 1.41 GB | 9 cards | **1 card** | **9x** |
| 122B model | 61 GB | 3.51 GB | 31 cards | **2 cards** | **15x** |
| 397B model | ~200 GB | 7.0 GB | 100 cards | **4 cards** | **25x** |

### Why Routing Bias is Critical for Corsair

SAIL doesn't need routing bias because dense models load ALL weights sequentially.
MoE models on Corsair face a unique problem: LPDDR5X capacity memory is only ~400 GB/s — 
**slower than H100 HBM (3.35 TB/s)**. Without routing bias, cold expert loads from LPDDR5X 
would actually be slower than GPU inference.

Expert Sniper's routing bias (+1.5 to cached expert logits before softmax) achieves 97% SRAM 
hit rate, ensuring almost all expert loads use the 150 TB/s SRAM path instead of the slow 
LPDDR5X path.

## d-Matrix Hardware Specs (Verified from Whitepaper Nov 2024)

| Spec | Per Card | 8-Card Server |
|------|----------|--------------|
| SRAM (Performance Memory) | 2 GB | 16 GB |
| SRAM Bandwidth | 150 TB/s | 1,200 TB/s |
| LPDDR5X (Capacity Memory) | 256 GB | 2 TB |
| LPDDR5X Bandwidth | ~400 GB/s (est.) | 3.2 TB/s |
| Compute (MXINT4) | 9,600 TFLOPS | 76,800 TFLOPS |
| Form Factor | FHFL PCIe Gen5 | Rack server |
