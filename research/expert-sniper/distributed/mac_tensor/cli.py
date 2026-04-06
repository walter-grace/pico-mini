#!/usr/bin/env python3
"""
mac-tensor CLI — Distributed MoE inference across multiple Macs.

Usage:
    mac-tensor node     Start an expert partition node
    mac-tensor chat     Interactive chat (coordinator)
    mac-tensor health   Check node health
    mac-tensor split    Split a model for distributed inference
    mac-tensor download Download a supported model from HuggingFace
"""

import argparse
import json
import os
import sys
import time


SUPPORTED_MODELS = {
    "qwen35": {
        "name": "Qwen 3.5-35B-A3B",
        "hf_id": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "num_experts": 256,
        "num_layers": 40,
        "node_script": "expert_node_fast.py",
        "coordinator_script": "distributed_interactive.py",
        "split_script": "split_qwen.py",
        "needs_split": True,
        "default_port": 8301,
        "default_model_dir": "~/models/qwen35-4bit",
        "stream_dir": "~/models/qwen35-stream",
    },
    "gemma4": {
        "name": "Gemma 4-26B-A4B",
        "hf_id": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "num_experts": 128,
        "num_layers": 30,
        "node_script": "gemma4_expert_node.py",
        "coordinator_script": "gemma4_distributed.py",
        "split_script": "split_gemma4.py",
        "needs_split": False,
        "default_port": 8401,
        "default_model_dir": "~/models/gemma4-4bit",
        "stream_dir": "~/models/gemma4-4bit",
    },
}

# Where our scripts live (same directory as this package)
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def get_model(name):
    if name not in SUPPORTED_MODELS:
        print(f"Error: Unknown model '{name}'")
        print(f"Supported models: {', '.join(SUPPORTED_MODELS.keys())}")
        sys.exit(1)
    return SUPPORTED_MODELS[name]


def expand(path):
    return os.path.expanduser(path)


# ============================================================
# COMMANDS
# ============================================================


def cmd_download(args):
    """Download a model from HuggingFace."""
    model = get_model(args.model)
    output = expand(args.output or model["default_model_dir"])

    print(f"Downloading {model['name']}...")
    print(f"  From: {model['hf_id']}")
    print(f"  To:   {output}")
    print()

    from huggingface_hub import snapshot_download
    snapshot_download(model["hf_id"], local_dir=output)

    print(f"\nDone! Model saved to {output}")

    if model["needs_split"]:
        print(f"\nNext step: run 'mac-tensor split --model {args.model}' to prepare for distributed inference.")
    else:
        print(f"\nThis model doesn't need splitting. You can start nodes directly.")


def cmd_split(args):
    """Split a model for distributed inference."""
    model = get_model(args.model)

    if not model["needs_split"]:
        print(f"{model['name']} doesn't need splitting — expert nodes load directly from safetensors.")
        print(f"You can start nodes with: mac-tensor node --model {args.model} --partition 0-{model['num_experts']//2 - 1}")
        return

    input_dir = expand(args.input or model["default_model_dir"])
    output_dir = expand(args.output or model["stream_dir"])

    if not os.path.exists(input_dir):
        print(f"Error: Model directory not found: {input_dir}")
        print(f"Run 'mac-tensor download --model {args.model}' first.")
        sys.exit(1)

    script = os.path.join(SCRIPT_DIR, model["split_script"])
    if not os.path.exists(script):
        print(f"Error: Split script not found: {script}")
        sys.exit(1)

    print(f"Splitting {model['name']}...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print()

    # Run the split script
    os.system(f"python3 {script} --input {input_dir} --output {output_dir}")


def cmd_node(args):
    """Start an expert partition node."""
    model = get_model(args.model)
    port = args.port or model["default_port"]
    partition = args.partition
    model_dir = expand(args.model_dir or model["stream_dir"])

    if not partition:
        # Default: first half
        half = model["num_experts"] // 2
        partition = f"0-{half - 1}"
        print(f"No --partition specified, defaulting to {partition}")

    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        if model["needs_split"]:
            print(f"Run 'mac-tensor download --model {args.model}' then 'mac-tensor split --model {args.model}'")
        else:
            print(f"Run 'mac-tensor download --model {args.model}' first.")
        sys.exit(1)

    script = os.path.join(SCRIPT_DIR, model["node_script"])
    if not os.path.exists(script):
        print(f"Error: Node script not found: {script}")
        sys.exit(1)

    print(f"Starting {model['name']} expert node")
    print(f"  Partition: {partition}")
    print(f"  Model:     {model_dir}")
    print(f"  Port:      {port}")
    print()

    mem_limit = args.memory_limit or 14.0
    os.execvp("python3", [
        "python3", script,
        "--partition", partition,
        "--model-dir", model_dir,
        "--port", str(port),
        "--memory-limit-gb", str(mem_limit),
    ])


def cmd_chat(args):
    """Start interactive chat (coordinator mode)."""
    model = get_model(args.model)

    if not args.nodes:
        print("Error: --nodes is required (comma-separated URLs of expert nodes)")
        print(f"Example: mac-tensor chat --model {args.model} --nodes http://mac2:{model['default_port']},http://mac3:{model['default_port']}")
        sys.exit(1)

    script = os.path.join(SCRIPT_DIR, model["coordinator_script"])
    if not os.path.exists(script):
        print(f"Error: Coordinator script not found: {script}")
        sys.exit(1)

    max_tokens = args.max_tokens or 300
    temperature = args.temperature or 0.7

    cmd = [
        "python3", script,
        "--nodes", args.nodes,
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
    ]

    # Qwen coordinator takes --model-dir; Gemma takes it too
    if args.model_dir:
        cmd.extend(["--model-dir", expand(args.model_dir)])

    os.execvp("python3", cmd)


def cmd_health(args):
    """Check health of expert nodes."""
    import urllib.request

    nodes = [n.strip() for n in args.nodes.split(",")]

    print(f"Checking {len(nodes)} node(s)...\n")

    all_ok = True
    for url in nodes:
        try:
            resp = urllib.request.urlopen(f"{url}/health", timeout=5)
            data = json.loads(resp.read())
            status = data.get("status", "unknown")
            partition = data.get("partition", "?")
            experts = data.get("total_experts_loaded", data.get("total_layers_loaded", "?"))
            mem = data.get("memory_gb", "?")
            reqs = data.get("compute_requests", 0)
            avg_ms = data.get("avg_compute_ms", 0)
            model_name = data.get("model", "qwen35")

            icon = "OK" if status == "ok" else "FAIL"
            print(f"  [{icon}] {url}")
            print(f"       Model: {model_name} | Partition: {partition} | Memory: {mem} GB")
            print(f"       Experts loaded: {experts} | Requests: {reqs} | Avg: {avg_ms:.1f}ms")
            print()
        except Exception as e:
            print(f"  [FAIL] {url}")
            print(f"       Error: {e}")
            print()
            all_ok = False

    if all_ok:
        print("All nodes healthy!")
    else:
        print("Some nodes are down.")
        sys.exit(1)


def cmd_info(args):
    """Show info about supported models."""
    print("Supported Models\n")
    for key, m in SUPPORTED_MODELS.items():
        print(f"  {key}")
        print(f"    Name:        {m['name']}")
        print(f"    HuggingFace: {m['hf_id']}")
        print(f"    Experts:     {m['num_experts']} per layer, {m['num_layers']} layers")
        print(f"    Needs split: {'Yes' if m['needs_split'] else 'No (loads from safetensors directly)'}")
        print(f"    Default port: {m['default_port']}")
        print()

    print("Quick Start:")
    print("  mac-tensor download --model qwen35")
    print("  mac-tensor split --model qwen35         # on each Mac")
    print("  mac-tensor node --model qwen35 --partition 0-127   # Mac 2")
    print("  mac-tensor node --model qwen35 --partition 128-255 # Mac 3")
    print("  mac-tensor chat --model qwen35 --nodes http://mac2:8301,http://mac3:8301")


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        prog="mac-tensor",
        description="Distributed MoE inference across multiple Macs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mac-tensor info                                    Show supported models
  mac-tensor download --model qwen35                 Download Qwen 3.5-35B
  mac-tensor split --model qwen35                    Split for distributed use
  mac-tensor node --model qwen35 --partition 0-127   Start expert node
  mac-tensor chat --model qwen35 --nodes http://mac2:8301,http://mac3:8301
  mac-tensor health --nodes http://mac2:8301,http://mac3:8301
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # download
    p_dl = subparsers.add_parser("download", help="Download a model from HuggingFace")
    p_dl.add_argument("--model", required=True, choices=SUPPORTED_MODELS.keys())
    p_dl.add_argument("--output", "-o", help="Output directory")

    # split
    p_sp = subparsers.add_parser("split", help="Split model for distributed inference")
    p_sp.add_argument("--model", required=True, choices=SUPPORTED_MODELS.keys())
    p_sp.add_argument("--input", "-i", help="Input model directory")
    p_sp.add_argument("--output", "-o", help="Output directory")

    # node
    p_nd = subparsers.add_parser("node", help="Start an expert partition node")
    p_nd.add_argument("--model", required=True, choices=SUPPORTED_MODELS.keys())
    p_nd.add_argument("--partition", "-p", help="Expert range, e.g. '0-127'")
    p_nd.add_argument("--port", type=int, help="Port to listen on")
    p_nd.add_argument("--model-dir", help="Model directory (override default)")
    p_nd.add_argument("--memory-limit", type=float, help="Memory limit in GB (default: 14)")

    # chat
    p_ch = subparsers.add_parser("chat", help="Interactive chat (coordinator)")
    p_ch.add_argument("--model", required=True, choices=SUPPORTED_MODELS.keys())
    p_ch.add_argument("--nodes", required=True, help="Comma-separated expert node URLs")
    p_ch.add_argument("--max-tokens", type=int, help="Max tokens to generate (default: 300)")
    p_ch.add_argument("--temperature", type=float, help="Sampling temperature (default: 0.7)")
    p_ch.add_argument("--model-dir", help="Model directory (override default)")

    # health
    p_hl = subparsers.add_parser("health", help="Check expert node health")
    p_hl.add_argument("--nodes", required=True, help="Comma-separated expert node URLs")

    # info
    subparsers.add_parser("info", help="Show supported models and quick start guide")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "download": cmd_download,
        "split": cmd_split,
        "node": cmd_node,
        "chat": cmd_chat,
        "health": cmd_health,
        "info": cmd_info,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
