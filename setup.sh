#!/bin/bash
# mac code setup — one-command install
set -e

echo ""
echo "  🍎 mac code setup"
echo "  ─────────────────────"
echo ""

# 1. Dependencies
echo "  [1/3] Installing dependencies..."
brew install llama.cpp 2>/dev/null || echo "  llama.cpp already installed"
pip3 install huggingface-hub rich ddgs --break-system-packages -q 2>/dev/null
echo "  Dependencies installed."

# 2. Download model (35B MoE — 30 tok/s on 16 GB)
echo "  [2/3] Downloading Qwen3.5-35B-A3B (10.6 GB)..."
echo "         This may take 5-15 minutes depending on your connection."
mkdir -p ~/models
if [ -f "$HOME/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf" ]; then
    echo "  Model already downloaded, skipping."
else
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
print('  Model downloaded.')
"
fi

# 3. Done
echo "  [3/3] Done!"
echo ""
echo "  ─────────────────────"
echo "  Start the LLM server:"
echo ""
echo "    llama-server \\"
echo "      --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \\"
echo "      --port 8000 --host 127.0.0.1 \\"
echo "      --flash-attn on --ctx-size 12288 \\"
echo "      --cache-type-k q4_0 --cache-type-v q4_0 \\"
echo "      --n-gpu-layers 99 --reasoning off -np 1 -t 4"
echo ""
echo "  Then in another terminal, run the agent:"
echo ""
echo "    python3 agent.py"
echo ""
