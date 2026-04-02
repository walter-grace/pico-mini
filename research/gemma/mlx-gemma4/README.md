# mlx-gemma4

Run Google Gemma 4 models on Apple Silicon via MLX. First MLX implementation — mlx-lm doesn't support Gemma 4 yet.

## Install

```bash
git clone https://github.com/walter-grace/mac-code
cd mac-code/research/gemma/mlx-gemma4
pip install .
```

## Usage

```bash
# Single prompt
mlx-gemma4 --model google/gemma-4-E2B-it --prompt "What is the capital of France?" -v

# Chat mode
mlx-gemma4 --model google/gemma-4-E2B-it

# Larger models
mlx-gemma4 --model google/gemma-4-E4B-it --prompt "Write a haiku about AI"
mlx-gemma4 --model google/gemma-4-26B-A4B-it --prompt "Hello"  # MoE, needs 50+ GB RAM
```

## Supported Models

| Model | Params | RAM needed | Type |
|-------|--------|-----------|------|
| google/gemma-4-E2B-it | 2B | ~4 GB | Dense |
| google/gemma-4-E4B-it | 4B | ~8 GB | Dense |
| google/gemma-4-26B-A4B-it | 26B (4B active) | ~50 GB | MoE (128 experts) |
| google/gemma-4-31B-it | 31B | ~62 GB | Dense |

For the 26B MoE model on 16 GB Macs, use [mlx-expert-sniper](https://huggingface.co/waltgrace/mlx-expert-sniper) which streams experts from SSD.

## Python API

```python
from mlx_gemma4 import load_model, generate

model, tokenizer = load_model("google/gemma-4-E2B-it")

for token in generate(model, tokenizer, "What is 2+2?"):
    print(token, end="", flush=True)
```

## Architecture

Gemma 4 uses a hybrid attention architecture:
- Sliding window attention (1024 tokens) on 5 of every 6 layers
- Full attention every 6th layer
- K=V sharing on full attention layers
- gelu_pytorch_tanh activation
- Logit softcapping at 30.0

The MoE variant (26B-A4B) adds 128 experts with top-8 routing per layer, running in parallel with the dense MLP.
