#!/bin/bash
# MoE Sniper — Docker Entrypoint
# Auto-downloads model and starts llama-server.
#
# Usage:
#   docker run --gpus all -p 8201:8201 moe-sniper
#   docker run --gpus all -p 8201:8201 -e NGL=20 moe-sniper
#   docker run --gpus all moe-sniper --benchmark

set -e

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B-Q4_K_M.gguf}"
NGL="${NGL:-35}"
CTX="${CTX:-2048}"
PORT="${PORT:-8201}"

if [ "$1" = "--benchmark" ]; then
    exec bash /opt/benchmark.sh
fi

if [ "$1" = "--model" ]; then
    MODEL_NAME="$2.gguf"
    shift 2
fi

MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

# Download model if not present
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading $MODEL_NAME..."
    python3 -c "
from huggingface_hub import hf_hub_download
import os
os.environ['HF_HOME'] = '/tmp/hf'
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
                filename='$MODEL_NAME',
                local_dir='$MODEL_DIR')
print('Done!')
"
fi

# Vision projector
MMPROJ_ARGS=""
if [ -f "$MODEL_DIR/mmproj-F16.gguf" ]; then
    MMPROJ_ARGS="--mmproj $MODEL_DIR/mmproj-F16.gguf"
fi

echo ""
echo "============================================"
echo "  MoE Sniper — GPU Inference Server"
echo "============================================"
echo "  Model:   $MODEL_NAME"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  VRAM:    $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Layers:  $NGL on GPU"
echo "  Port:    $PORT"
echo "============================================"
echo ""

exec llama-server \
    -m "$MODEL_PATH" \
    $MMPROJ_ARGS \
    -ngl "$NGL" \
    -c "$CTX" \
    --port "$PORT" \
    --host 0.0.0.0 \
    "$@"
