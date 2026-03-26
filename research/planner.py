#!/usr/bin/env python3
"""
Kimi 2.5 via OpenRouter — the research brain.
Proposes experiments based on results history.
"""

import json
import os
import urllib.request

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "moonshotai/kimi-k2.5"


def get_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        config_path = os.path.expanduser("~/.mac-code/openrouter.json")
        if os.path.exists(config_path):
            key = json.loads(open(config_path).read()).get("api_key")
    return key


def propose_experiment(results_history, techniques_source, best_ratio, consecutive_reverts=0, planner_mode="auto"):
    """
    Ask Kimi 2.5 to propose the next experiment.
    planner_mode: "auto" (API if available), "api" (force API), "grid" (force grid)
    Returns: dict with technique, params, description, hypothesis, planner ("api"|"grid")
    """
    if planner_mode == "grid":
        result = fallback_proposal(results_history, consecutive_reverts)
        result["planner"] = "grid"
        return result

    api_key = get_api_key()
    if not api_key:
        if planner_mode == "api":
            print("  WARNING: --planner=api but no OPENROUTER_API_KEY found")
            print("  Set via: export OPENROUTER_API_KEY='your-key'")
            print("  Or save to: ~/.mac-code/openrouter.json")
        result = fallback_proposal(results_history, consecutive_reverts)
        result["planner"] = "grid"
        return result

    # Build prompt
    history_text = results_history[-20:] if len(results_history) > 20 else results_history
    history_str = "\n".join(history_text) if history_text else "No experiments yet."

    prompt = f"""You are optimizing KV cache compression for LLMs on Apple Silicon.
Goal: maximize compression_ratio while keeping cosine_similarity >= 0.99.

Current best compression ratio: {best_ratio:.2f}x

Available techniques in techniques.py:
1. baseline_minmax(bits=2/3/4, group_size=32/64/128/256)
2. polar_quant(bits=2/3/4, group_size=32/64/128)
3. qjl_1bit(projection_dim=64/128/256/512)
4. hadamard_rotate(bits=3/4, group_size=32/64/128)
5. mixed_kv(k_bits=2/3/4, v_bits=2/3/4, group_size=64)
6. per_layer_adaptive(group_size=32/64/128)
7. residual_correction(base_bits=2/3, projection_dim=64/128/256)
8. lloyd_max(bits=2/3/4, iterations=5/10/20)

Recent experiment results:
{history_str}

{"WARNING: Last 5 experiments were all reverted. Try a COMPLETELY different approach." if consecutive_reverts >= 5 else ""}

Propose ONE experiment. Reply with ONLY valid JSON:
{{
    "technique": "technique_name",
    "params": {{"bits": 3, "group_size": 64}},
    "description": "Brief description",
    "hypothesis": "Why this should work"
}}"""

    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())

        # Handle various response formats
        content = ""
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            content = msg.get("content") or ""
            # Kimi/reasoning models may put useful text in reasoning field
            if not content and msg.get("reasoning"):
                content = msg["reasoning"]
        elif "error" in data:
            print(f"  OpenRouter error: {data['error']}")
        else:
            print(f"  Unexpected response: {json.dumps(data)[:200]}")

        if content:
            # Extract JSON from response — handle markdown fences, multiple objects, etc.
            # Find the outermost { } that contains "technique"
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                # If there are multiple JSON objects, find the one with "technique"
                for attempt_start in range(len(content)):
                    if content[attempt_start] == "{":
                        depth = 0
                        for i in range(attempt_start, len(content)):
                            if content[i] == "{":
                                depth += 1
                            elif content[i] == "}":
                                depth -= 1
                                if depth == 0:
                                    candidate = content[attempt_start:i+1]
                                    try:
                                        parsed = json.loads(candidate)
                                        if "technique" in parsed:
                                            parsed["planner"] = "api"
                                            return parsed
                                    except json.JSONDecodeError:
                                        pass
                                    break
    except Exception as e:
        print(f"  Planner error: {e}")

    result = fallback_proposal(results_history, consecutive_reverts)
    result["planner"] = "grid"
    return result


def fallback_proposal(results_history, consecutive_reverts):
    """Fallback when OpenRouter is unavailable — systematic grid search."""
    import random

    techniques = [
        ("baseline_minmax", {"bits": 4, "group_size": 64}),
        ("baseline_minmax", {"bits": 3, "group_size": 64}),
        ("baseline_minmax", {"bits": 4, "group_size": 32}),
        ("baseline_minmax", {"bits": 3, "group_size": 128}),
        ("polar_quant", {"bits": 3, "group_size": 64}),
        ("polar_quant", {"bits": 4, "group_size": 64}),
        ("qjl_1bit", {"projection_dim": 128}),
        ("qjl_1bit", {"projection_dim": 256}),
        ("hadamard_rotate", {"bits": 4, "group_size": 64}),
        ("hadamard_rotate", {"bits": 3, "group_size": 64}),
        ("residual_correction", {"base_bits": 3, "projection_dim": 128}),
        ("residual_correction", {"base_bits": 2, "projection_dim": 256}),
        ("lloyd_max", {"bits": 3, "group_size": 64, "iterations": 10}),
        ("lloyd_max", {"bits": 4, "iterations": 15}),
        ("mixed_kv", {"k_bits": 3, "v_bits": 4, "group_size": 64}),
        ("per_layer_adaptive", {"group_size": 64}),
    ]

    # Skip already-tested combinations
    tested = set()
    for line in results_history:
        parts = line.split("\t")
        if len(parts) >= 3:
            tested.add(parts[2])  # technique name

    # Pick next untested, or random if all tested
    for tech, params in techniques:
        key = f"{tech}_{json.dumps(params, sort_keys=True)}"
        if key not in tested:
            return {
                "technique": tech,
                "params": params,
                "description": f"Grid search: {tech} {params}",
                "hypothesis": "Systematic exploration",
            }

    # All tested — random variation
    tech, params = random.choice(techniques)
    params = {k: v + random.randint(-1, 1) if isinstance(v, int) else v for k, v in params.items()}
    return {
        "technique": tech,
        "params": params,
        "description": f"Random variation: {tech} {params}",
        "hypothesis": "Exploring parameter space",
    }
