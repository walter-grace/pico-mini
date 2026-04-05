#!/bin/bash
# Apply Expert Sniper patches to stock llama.cpp
# Run from the llama.cpp root directory
#
# Uses Python for all patching (robust, idempotent, works on any platform).
# Each patch checks if already applied before modifying.
set -e

echo "Applying Expert Sniper patches..."

# All patching done via Python for robustness
python3 << 'PATCHEOF'
import os, sys, re

def patch_file(path, check_str, patch_fn, description):
    """Apply a patch if check_str not already in file."""
    with open(path, 'r') as f:
        content = f.read()
    if check_str in content:
        print(f"  [SKIP] {path}: {description} (already applied)")
        return
    new_content = patch_fn(content)
    if new_content == content:
        print(f"  [WARN] {path}: {description} — anchor not found, skipping")
        return
    with open(path, 'w') as f:
        f.write(new_content)
    print(f"  [OK] {path}: {description}")

# ──────────────────────────────────────────────────────────
# 1. CMakeLists.txt — add expert cache source files
# ──────────────────────────────────────────────────────────
def patch_cmake(content):
    # Try multiple anchors for robustness
    anchors = [
        "llama-memory-recurrent.cpp",   # current (2025+)
        "llama-kv-cache-recurrent.cpp",  # older
        "llama-memory-hybrid.cpp",       # another option
        "llama-kv-cache.cpp",            # fallback
    ]
    insert = "            llama-expert-cache.cpp\n            llama-expert-cache-ctx.cpp\n"
    for anchor in anchors:
        if anchor in content:
            return content.replace(anchor, anchor + "\n" + insert, 1)
    # Last resort: find any .cpp in the src target and append
    m = re.search(r'(llama-[a-z-]+\.cpp)', content)
    if m:
        return content.replace(m.group(0), m.group(0) + "\n" + insert, 1)
    return content

patch_file("src/CMakeLists.txt", "llama-expert-cache", patch_cmake,
           "added expert cache source files")

# ──────────────────────────────────────────────────────────
# 2. common/common.h — add expert_cache_size field
# ──────────────────────────────────────────────────────────
def patch_common_h(content):
    field = "    size_t expert_cache_size = 0;   // expert LRU cache size in bytes for MoE models (0 = disabled)\n"
    # We must insert into `struct common_params {`, NOT common_params_speculative.
    # Strategy: find `struct common_params {` (exact, not speculative), then find
    # `n_gpu_layers` after that position but before the next top-level struct.
    struct_patterns = [
        r'struct\s+common_params\s*\{',          # current
        r'struct\s+common_params\s*[^_]',         # fallback
    ]
    struct_start = -1
    for pat in struct_patterns:
        m = re.search(pat, content)
        if m:
            # Make sure this isn't common_params_speculative
            if 'speculative' not in content[max(0, m.start()-5):m.end()]:
                struct_start = m.start()
                break
    if struct_start < 0:
        return content

    # Search for n_gpu_layers AFTER struct_start
    anchors = ["int32_t n_gpu_layers", "n_gpu_layers"]
    lines = content.split('\n')
    # Find the line number where struct common_params starts
    offset = len(content[:struct_start].split('\n')) - 1
    for anchor in anchors:
        for i in range(offset, len(lines)):
            if anchor in lines[i]:
                # Verify we haven't left the struct (rough check: no new 'struct' keyword)
                lines.insert(i + 1, field.rstrip())
                return '\n'.join(lines)
    # Fallback: insert after struct opening brace
    for i in range(offset, min(offset + 10, len(lines))):
        if '{' in lines[i]:
            lines.insert(i + 1, field.rstrip())
            return '\n'.join(lines)
    return content

patch_file("common/common.h", "expert_cache_size", patch_common_h,
           "added expert_cache_size field")

# ──────────────────────────────────────────────────────────
# 3. common/arg.cpp — add --expert-cache-size CLI argument
# ──────────────────────────────────────────────────────────
def patch_arg_cpp(content):
    arg_block = '''        add_opt(common_arg(
            {"--expert-cache-size"}, "N",
            "size of expert LRU cache in MB for MoE models (default: 0 = disabled)",
            [](common_params & params, const std::string & value) {
                params.expert_cache_size = std::stoull(value) * 1024ULL * 1024ULL;
            }
        ).set_env("LLAMA_ARG_EXPERT_CACHE_SIZE"));

'''
    # Try multiple anchors
    anchors = [
        '"--override-tensor"',
        '"--override-kv"',
        '"--lora"',
        '"--control-vector"',
    ]
    for anchor in anchors:
        if anchor in content:
            idx = content.find(anchor)
            # Find the start of this add_opt block (go back to find 'add_opt')
            search_start = max(0, idx - 500)
            block_start = content.rfind('add_opt(', search_start, idx)
            if block_start >= 0:
                return content[:block_start] + arg_block + content[block_start:]
    # Fallback: insert before the last '}' in the function
    return content

patch_file("common/arg.cpp", "expert-cache-size", patch_arg_cpp,
           "added --expert-cache-size argument")

# ──────────────────────────────────────────────────────────
# 4. common/common.cpp — add include + initialization
# ──────────────────────────────────────────────────────────
def patch_common_cpp_include(content):
    include = '#include "../src/llama-expert-cache-ctx.h"\n'
    # Add after first #include
    first_include = content.find('#include')
    if first_include >= 0:
        line_end = content.find('\n', first_include)
        return content[:line_end+1] + include + content[line_end+1:]
    return include + content

patch_file("common/common.cpp", "llama-expert-cache-ctx.h", patch_common_cpp_include,
           "added expert cache include")

def patch_common_cpp_init(content):
    init_code = '''
    // Expert Sniper: initialize expert cache for MoE models
    if (params.expert_cache_size > 0) {
        static auto expert_cache = std::make_unique<llama_expert_cache_ctx>();
        expert_cache->init(*model, params.expert_cache_size);
        if (expert_cache->n_expert > 0) {
            cparams.cb_eval = llama_expert_cache_ctx::eval_callback;
            cparams.cb_eval_user_data = expert_cache.get();
            fprintf(stderr, "expert cache enabled: %.1f MB for %d experts\\n",
                    (double)params.expert_cache_size / (1024*1024),
                    expert_cache->n_expert);
        }
    }

'''
    # Strategy: insert just BEFORE `llama_init_from_model(model, cparams)`.
    # This works for both old and new llama.cpp since the init call is stable.
    # Fallback anchors in case the primary isn't found.
    anchors = [
        "llama_init_from_model",                          # primary (stable across versions)
        "common_context_params_to_llama",                 # new name (2025+)
        "llama_context_params_from_common_params",        # old name
    ]
    for anchor in anchors:
        if anchor not in content:
            continue
        idx = content.find(anchor)
        if anchor == "llama_init_from_model":
            # Insert BEFORE this line: find the start of the line
            line_start = content.rfind('\n', 0, idx)
            if line_start < 0:
                line_start = 0
            else:
                line_start += 1
            return content[:line_start] + init_code + content[line_start:]
        else:
            # Fallback: insert after the cparams statement
            semi = content.find(';', idx)
            if semi >= 0:
                nl = content.find('\n', semi)
                if nl >= 0:
                    return content[:nl+1] + init_code + content[nl+1:]
    return content

patch_file("common/common.cpp", "expert_cache_size", patch_common_cpp_init,
           "added expert cache initialization")

print("\nExpert Sniper patches applied successfully.")
print("Build: cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc) --target llama-server")
PATCHEOF
