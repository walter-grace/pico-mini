# Sniper Router — Remote Inference

Run the agent UI on one Mac, inference on another. No model files needed on the client.

```
Client (any Mac)            Server (Mac with model)
┌──────────────┐            ┌──────────────────┐
│ router.py    │── LAN ───▶│ llama-server     │
│ (agent UI)   │            │ or mlx-sniper    │
│ chat/search  │◀── resp ──│ expert streaming  │
│ vision/shell │            │ MoE model        │
└──────────────┘            └──────────────────┘
```

## Setup

### 1. Start the inference server on your server machine

**Option A: llama.cpp (any MoE GGUF model)**
```bash
llama-server \
  -m /path/to/model.gguf \
  --mmproj /path/to/mmproj-F16.gguf \
  -ngl 1 -c 2048 --expert-cache-size 5000 \
  --reasoning off --port 8201 --host 0.0.0.0
```

**Option B: MLX expert sniper (Apple Silicon, faster)**
```bash
mlx-sniper server /path/to/model --port 8899 --host 0.0.0.0
```

Note: `--host 0.0.0.0` is required to accept connections from other machines.

### 2. Run the agent on your client machine

```bash
# Using llama-server backend:
python3 router.py --server <server-ip>:8201

# Using MLX sniper backend:
python3 router.py --server <server-ip>:8899
```

Replace `<server-ip>` with your server's hostname or IP address.

## Commands

- Just type to chat
- `/search <query>` — web search + remote AI synthesis
- `/image <path>` — send local image to remote model for vision
- `/screenshot` — capture screen + send to remote model
- `/shell <task>` — remote AI generates command, executes locally
- `/ping` — check server connection
- `/stats` — speed stats
- `/quit` — exit

## Why Remote?

- **No model files on client** — the client just sends HTTP requests
- **Use your strongest machine for inference** — more RAM, faster GPU, internal NVMe
- **Client handles the UI** — search, shell commands, image capture all run locally
- **Works over WiFi or Ethernet** — any LAN connection
