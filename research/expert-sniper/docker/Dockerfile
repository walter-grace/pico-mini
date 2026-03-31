# MoE Sniper — Pre-built GPU inference server
# Builds llama.cpp with CUDA for fast MoE inference.
#
# Build from repo root:
#   docker build -f docker/Dockerfile -t moe-sniper .
#
# Run:
#   docker run --gpus all -p 8201:8201 moe-sniper

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install build deps
RUN apt-get update -qq && \
    apt-get install -y -qq cmake && \
    rm -rf /var/lib/apt/lists/*

# Build llama.cpp with CUDA (sm_80=A100, sm_86=RTX 3090, sm_89=RTX 4090)
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="80;86;89" \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc) --target llama-server && \
    rm -rf build/CMakeFiles

# Install Python deps
RUN pip install --no-cache-dir huggingface-hub

# Copy entrypoint and benchmark
COPY docker/entrypoint.sh /opt/entrypoint.sh
COPY scripts/gpu_benchmark.sh /opt/benchmark.sh
RUN chmod +x /opt/entrypoint.sh /opt/benchmark.sh

ENV PATH="/opt/llama.cpp/build/bin:${PATH}"
EXPOSE 8201

ENTRYPOINT ["/opt/entrypoint.sh"]
