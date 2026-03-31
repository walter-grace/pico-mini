# MoE Sniper Docker

Pre-built GPU inference server. No compilation needed.

## Build

Build from the **repo root** (not from `docker/`):

```bash
docker build -f docker/Dockerfile -t moe-sniper .
```

## Run

```bash
# Auto-downloads 21 GB model on first start
docker run --gpus all -p 8201:8201 moe-sniper

# Custom GPU layers
docker run --gpus all -p 8201:8201 -e NGL=20 moe-sniper

# Use local model directory (skip download)
docker run --gpus all -p 8201:8201 -v /path/to/models:/models moe-sniper

# Benchmark
docker run --gpus all moe-sniper --benchmark
```

## Connect

Once the server is running, connect from any machine:

```bash
python3 sniper-router/router.py --server http://<server-ip>:8201
```
