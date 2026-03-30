# Two Approaches to Running Giant AI Models on Consumer Hardware

## How MoE Sparsity Is Changing What's Possible With Local AI

There's a quiet revolution happening in local AI inference. Two independent projects — built weeks apart — arrived at the same fundamental insight but took radically different paths to exploit it.

The insight: **Mixture-of-Experts models waste 97% of their weights on every token.** If you only load the 3% that matters, you can run models that shouldn't fit on your hardware.

Here's how two teams attacked this problem differently, and what it means for the future of AI on consumer devices.

---

## The Problem Both Projects Solve

Modern MoE models like Qwen3.5-35B-A3B have 256 experts per layer but only activate 8 per token. That means for any given computation, 248 out of 256 expert weight matrices are completely unused.

Standard inference tools (llama.cpp, vLLM, MLX) load the **entire model into RAM**. If the model exceeds your RAM, they either refuse to run or thrash violently — swapping pages in and out of memory with no intelligence about which pages actually matter.

Both projects replace this blind paging with **expert-aware memory management**: understand which experts the model needs, load those, and keep the rest on disk.

---

## Flash-MoE: The Custom Engine Approach

**Repository:** github.com/danveloper/flash-moe
**Target:** Qwen3.5-397B-A17B (209 GB) on a 48 GB MacBook Pro M3 Max
**Result:** ~4.4 tok/s

Flash-MoE is a from-scratch inference engine written in ~8,000 lines of C and Objective-C with hand-tuned Metal compute shaders. It owns the entire pipeline — tokenizer, attention, routing, expert loading, GPU dispatch — with zero dependencies on any ML framework.

**Key design decisions:**

- **No custom cache.** They tested and rejected an LRU cache, finding that on 48 GB unified memory, macOS's built-in page cache achieves ~71% hit rate and any user-space caching added overhead. The OS already does a reasonable job when you have 35 GB of free memory to work with.

- **Serial pipeline.** They discovered that SSD DMA and GPU compute compete for Apple Silicon's memory controller bandwidth, so they run them sequentially rather than trying to overlap.

- **Single-model optimization.** Every shader, every buffer size, every dispatch group is tuned for one specific model architecture.

**The tradeoff:** It's fast and elegant, but it only works on Apple Silicon with ≥48 GB RAM, and only for one model. If you want to run a different MoE architecture or use non-Apple hardware, you need a different solution.

---

## MoE Sniper: The llama.cpp Integration Approach

**Repository:** github.com/walter-grace/sniper_MoE_Llama_cpp
**Target:** Qwen3.5-35B-A3B (10.6-22 GB) on an 8 GB MacBook Air
**Result:** 0.24 tok/s (8 GB, USB drive), projected 5-10 tok/s (16 GB, NVMe)

MoE Sniper takes the opposite approach: instead of building a custom engine, we added an **expert-aware LRU cache** directly into llama.cpp — the most widely used local inference tool with support for every platform and hundreds of model architectures.

**Key design decisions:**

- **Custom LRU cache is essential.** On 8 GB RAM, the OS page cache has no room to work — macOS needs ~3 GB for itself, leaving almost nothing for intelligent caching. Our user-space cache explicitly tracks which experts are hot (frequently accessed) and keeps them pinned, while letting cold experts get paged out.

- **Eval callback interception.** We register a callback with llama.cpp's backend scheduler that fires before every `ggml_mul_mat_id` operation (the expert computation). This intercepts the exact moment expert weights are needed and ensures they're in our cache.

- **Works with any storage.** Models can live on internal NVMe, external SSD, or even a USB flash drive. The cache compensates for slower storage by keeping hot experts in RAM. We demonstrated a 10.6 GB model running from a USB drive on 8 GB RAM — the model is 1.3x larger than the machine's total memory.

- **Multimodal.** The same model handles text, vision (image understanding), and web search. The vision encoder is small enough to pin alongside the language model's attention weights.

**The tradeoff:** It's slower per token than a hand-tuned custom engine, but it works on $999 hardware, supports any GGUF MoE model, and runs on macOS, Linux, and Windows with CPU, CUDA, or Metal backends.

---

## Side-by-Side Comparison

| | Flash-MoE | MoE Sniper |
|---|---|---|
| **Minimum hardware** | 48 GB M3 Max ($3,500+) | 8 GB M2 Air ($999) |
| **Engine** | Custom C/Metal (8,000 lines) | llama.cpp + 500 lines |
| **Expert caching** | OS page cache (71% hit rate) | Custom LRU (tunable) |
| **Platform** | Apple Silicon only | Any (CPU, CUDA, Metal) |
| **Models** | One (Qwen3.5-397B) | Any GGUF MoE model |
| **Vision** | No | Yes |
| **Storage** | Internal NVMe required | NVMe, SSD, or USB drive |
| **Speed** | ~4.4 tok/s | 0.24 tok/s (8 GB USB) |
| **Upstream potential** | Standalone project | Could be a llama.cpp PR |

---

## What This Means

These projects prove that MoE sparsity is a **real, exploitable property** for consumer AI inference — not just a training optimization. The 97% of unused experts per token isn't theoretical waste; it's a concrete opportunity to run models far larger than your hardware should support.

**The Flash-MoE lesson:** When you control the entire stack and have enough RAM for the OS cache to work, you don't need a custom cache. The OS is already a pretty good LRU. Hand-tuned Metal shaders on unified memory can move data fast enough to be practical.

**The MoE Sniper lesson:** When RAM is scarce and the model exceeds available memory, application-level expert caching is essential. The OS doesn't know that expert 47 in layer 12 is hot — but your router does, and your cache can exploit that. And by building on llama.cpp instead of from scratch, the work benefits every platform and every MoE model.

**The combined lesson:** Expert-aware inference is the future for running large MoE models on consumer hardware. Whether you build it as a custom engine or integrate it into existing tools, the core technique — load only active experts, cache the hot ones, stream the cold ones — works across hardware price points from $999 to $3,500+.

The models keep getting bigger. MoE keeps getting more popular. The machines aren't getting cheaper fast enough. Expert-aware inference closes the gap.

---

*MoE Sniper is open source: github.com/walter-grace/sniper_MoE_Llama_cpp*

*Flash-MoE is open source: github.com/danveloper/flash-moe*
