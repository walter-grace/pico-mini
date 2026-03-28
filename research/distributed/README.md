# Distributed MoE Expert Sniper

Two Macs, one model, zero SSD reads.

## The Setup

```
Mac mini M4 (16 GB)          MacBook M2 (8 GB)
├── Backbone (1.42 GB)       ├── Expert weights L24-39 (7 GB)
├── Expert weights L0-23     ├── expert_worker.py (server)
├── KV/SSM cache             │
├── moe_agent.py (modified)  │
│                            │
│  hidden_state (6 KB) ───>  │
│  <─── expert_output (6 KB) │
│       via WiFi or USB-C    │
```

## How to Test

### Step 1: Split expert weights
On the Mac mini, split the 35B MoE stream files:
```bash
python3 split_for_distributed.py --layers 0-23 --output ~/models/local-experts/
python3 split_for_distributed.py --layers 24-39 --output ~/models/remote-experts/
```

### Step 2: Copy remote experts to MacBook
```bash
# Via AirDrop, USB drive, or scp
scp -r ~/models/remote-experts/ macbook:~/models/remote-experts/
```

### Step 3: Start the worker on MacBook
```bash
python3 expert_worker.py --expert-dir ~/models/remote-experts/ --port 9000
```

### Step 4: Run the distributed agent on Mac mini
```bash
python3 distributed_agent.py --local-layers 0-23 --remote macbook.local:9000
```

## Expected Results

| Setup | tok/s | Why |
|-------|-------|-----|
| Mac mini alone (SSD sniper) | 1.54 | I/O bound: 576 MB SSD reads/token |
| Mac mini + MacBook (distributed) | ~15-17 | All experts in RAM, compute bound |
| Improvement | **~10x** | Eliminated SSD bottleneck entirely |
