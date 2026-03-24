# mac code — Setup Instructions for Claude Code

This file tells Claude Code how to install and run mac code on any Mac with Apple Silicon.

## What This Project Is

mac code is a local AI coding agent (like Claude Code) that runs entirely on your Mac using a local LLM via llama.cpp. No cloud, no API keys, no cost.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4, 16GB+ RAM recommended)
- Homebrew installed (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- Python 3.10+
- Go 1.25+ (`brew install go`)

## Installation Steps

### 1. Install llama.cpp

```bash
brew install llama.cpp
```

### 2. Install Python dependencies

```bash
pip3 install huggingface-hub rich --break-system-packages
```

### 3. Download the 9B model (recommended — works with agent tools)

```bash
mkdir -p ~/models
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-9B-GGUF',
    'Qwen3.5-9B-Q4_K_M.gguf', local_dir='$HOME/models/')
"
```

### 4. Optionally download the 35B MoE model (faster reasoning, but no tool calling)

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
"
```

### 5. Build PicoClaw (agent framework)

```bash
cd <this-repo-directory>
git clone https://github.com/sipeed/picoclaw.git
cd picoclaw && make deps && make build && cd ..
```

### 6. Configure PicoClaw

```bash
mkdir -p ~/.picoclaw/workspace
cp config.example.json ~/.picoclaw/config.json
```

### 7. Start the LLM server

For the 9B model (recommended):
```bash
llama-server \
    --model ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 32768 \
    --n-gpu-layers 99 --reasoning off -t 4
```

For the 35B MoE model:
```bash
llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 8192 \
    --n-gpu-layers 99 --reasoning off -np 1 -t 4
```

### 8. Run the agent

```bash
python3 agent.py
```

## File Overview

- `agent.py` — Main agent TUI with auto-routing, slash commands, web search, tools
- `chat.py` — Lightweight streaming chat (no tools, direct to LLM)
- `dashboard.py` — Real-time server monitor (tok/s, slots, memory)
- `config.example.json` — PicoClaw config with DuckDuckGo search + fetch MCP servers
- `setup.sh` — One-command install script (alternative to manual steps)

## Architecture

The agent auto-routes between two models:
- **9B (Q4_K_M)** — 32K context, reliable tool calling, 15-30 tok/s
- **35B MoE (IQ2_M)** — 8K context, faster reasoning, 30-57 tok/s, no tool calling

Questions needing web search or tools route to 9B. Pure reasoning routes to 35B. The swap takes ~20 seconds.

## Common Issues

- **GPU OOM after long sessions**: Reboot the Mac to clear Metal GPU memory, then restart the server
- **Context overflow errors**: Clear PicoClaw sessions: `rm -rf ~/.picoclaw/workspace/sessions/`
- **PicoClaw not found**: Make sure you built it in step 5 and the binary is at `picoclaw/build/picoclaw-darwin-arm64`
- **Model download fails**: Ensure `huggingface-hub` is installed and you have ~11 GB free disk space

## Key Paths

- Models: `~/models/`
- PicoClaw config: `~/.picoclaw/config.json`
- PicoClaw sessions: `~/.picoclaw/workspace/sessions/`
- PicoClaw binary: `<repo>/picoclaw/build/picoclaw-darwin-arm64`
