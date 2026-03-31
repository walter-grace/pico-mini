#!/bin/bash
# MoE Sniper — GPU Benchmark Script
# Reproduces our RTX 3090 results. Run this on any NVIDIA GPU with 24+ GB VRAM.
#
# Usage:
#   # On a RunPod, Lambda, or any GPU server:
#   curl -sL https://raw.githubusercontent.com/walter-grace/sniper_MoE_Llama_cpp/main/scripts/gpu_benchmark.sh | bash
#
# Expected results on RTX 3090 (24 GB):
#   Prompt:    ~120 tok/s
#   Generate:  ~26 tok/s

set -e

echo "============================================"
echo "  MoE Sniper — GPU Benchmark"
echo "  Qwen3.5-35B-A3B Q4_K_M on NVIDIA GPU"
echo "============================================"
echo ""

# Check GPU
echo "[1/4] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "  Free VRAM: ${VRAM_FREE} MiB"
echo ""

# Install deps
echo "[2/4] Installing dependencies..."
export PATH=/usr/local/cuda/bin:$PATH
apt-get update -qq && apt-get install -y -qq cmake > /dev/null 2>&1 || true
pip install -q huggingface-hub 2>/dev/null || true
echo "  Done"
echo ""

# Build llama.cpp
echo "[3/4] Building llama.cpp with CUDA..."
cd /tmp
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git 2>&1 | tail -1
fi
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
cmake --build build -j$(nproc) --target llama-server 2>&1 | tail -1
echo "  Built: $(ls build/bin/llama-server)"
echo ""

# Download model
echo "[4/4] Downloading Qwen3.5-35B-A3B Q4_K_M (21 GB)..."
python3 -c "
from huggingface_hub import hf_hub_download
import os
os.environ['HF_HOME'] = '/tmp/hf'
if not os.path.exists('/tmp/models/Qwen3.5-35B-A3B-Q4_K_M.gguf'):
    print('  Downloading...')
    hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
                    filename='Qwen3.5-35B-A3B-Q4_K_M.gguf',
                    local_dir='/tmp/models')
    print('  Done')
else:
    print('  Already downloaded')
"
echo ""

# Determine GPU layers
echo "============================================"
echo "  Running Benchmark"
echo "============================================"
echo ""

# Try ngl values from high to low
for NGL in 999 40 35 30 25 20; do
    echo "Testing ngl $NGL..."

    # Start server in background
    /tmp/llama.cpp/build/bin/llama-server \
        -m /tmp/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
        -ngl $NGL -c 512 --reasoning off \
        --port 8201 --host 0.0.0.0 --no-warmup \
        > /tmp/server.log 2>&1 &
    SERVER_PID=$!

    # Wait for server
    READY=0
    for i in $(seq 1 60); do
        if curl -s http://localhost:8201/health 2>/dev/null | grep -q ok; then
            READY=1
            break
        fi
        sleep 1
    done

    if [ "$READY" = "0" ]; then
        echo "  ngl $NGL: failed to start (likely OOM)"
        kill -9 $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        sleep 2
        continue
    fi

    echo "  Server ready with ngl $NGL"

    # Run benchmark
    RESULT=$(curl -s -X POST http://localhost:8201/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "system", "content": "Answer in one sentence."},
                {"role": "user", "content": "What is the square root of 69?"}
            ],
            "temperature": 0,
            "max_tokens": 30,
            "stream": false
        }' 2>&1)

    # Extract results
    ANSWER=$(echo "$RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message'].get('content','') or r['choices'][0]['message'].get('reasoning_content',''))" 2>/dev/null)
    PREFILL=$(echo "$RESULT" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin)['timings']['prompt_per_second']:.1f}\")" 2>/dev/null)
    GENERATE=$(echo "$RESULT" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin)['timings']['predicted_per_second']:.1f}\")" 2>/dev/null)

    echo ""
    echo "============================================"
    echo "  RESULTS"
    echo "============================================"
    echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "  Model:      Qwen3.5-35B-A3B Q4_K_M (21 GB)"
    echo "  GPU layers: $NGL"
    echo "  Answer:     $ANSWER"
    echo "  Prefill:    $PREFILL tok/s"
    echo "  Generate:   $GENERATE tok/s"
    echo "============================================"

    # Save results
    echo "{\"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader)\", \"ngl\": $NGL, \"prefill\": $PREFILL, \"generate\": $GENERATE, \"answer\": \"$ANSWER\"}" > /tmp/benchmark_result.json
    cat /tmp/benchmark_result.json

    # Cleanup
    kill -9 $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null

    echo ""
    echo "Benchmark complete. Results saved to /tmp/benchmark_result.json"
    exit 0
done

echo "All ngl values failed — GPU may not have enough VRAM for this model."
exit 1
