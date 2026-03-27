#!/bin/bash
# MoE Expert Sniper — 122B on RunPod L40
# Paste this entire script into the RunPod web terminal.
# It downloads the model, splits it, and runs the sniper.
set -e

echo "============================================"
echo "  MoE Expert Sniper — 122B Setup"
echo "  NVIDIA L40 (48 GB VRAM)"
echo "============================================"

cd /workspace

# 1. Install deps
echo ""
echo "[1/5] Installing dependencies..."
pip install -q safetensors transformers huggingface_hub

# 2. Download the model (~70 GB, ~10-15 min at 900 Mbps)
echo ""
echo "[2/5] Downloading Qwen3.5-122B-A10B-4bit (69.6 GB)..."
echo "       This will take 10-15 minutes."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'mlx-community/Qwen3.5-122B-A10B-4bit',
    local_dir='/workspace/qwen35-122b-a10b-4bit',
    max_workers=8,
)
print('Download complete.')
"

# 3. Clone our code
echo ""
echo "[3/5] Getting sniper code..."
if [ ! -d "/workspace/mac-code" ]; then
    git clone https://github.com/walter-grace/mac-code.git /workspace/mac-code
else
    cd /workspace/mac-code && git pull && cd /workspace
fi

# 4. Split the model
echo ""
echo "[4/5] Splitting model into pinned + experts..."
python3 /workspace/mac-code/research/flash-streaming/split_122b.py \
    --model-dir /workspace/qwen35-122b-a10b-4bit \
    --output-dir /workspace/qwen35-122b-stream

echo ""
echo "Disk usage after split:"
du -sh /workspace/qwen35-122b-stream/
du -sh /workspace/qwen35-122b-stream/experts/

# 5. Run the sniper
echo ""
echo "[5/5] Running Expert Sniper..."
echo ""
python3 /workspace/mac-code/research/flash-streaming/sniper_122b.py \
    --model-dir /workspace/qwen35-122b-stream \
    --prompt "What is the capital of France?" \
    --max-tokens 20 \
    --device cuda

echo ""
echo "============================================"
echo "  DONE. Check results above."
echo "============================================"
