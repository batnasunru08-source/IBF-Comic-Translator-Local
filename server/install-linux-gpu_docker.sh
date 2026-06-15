#!/bin/bash
# Установка зависимостей для Docker-образа с NVIDIA GPU (CUDA 13.0)
# Используется uv вместо pip
set -e
cd "$(dirname "$0")"

#echo "[INSTALL] Installing uv (fast pip replacement)..."
#pip install -U uv

echo "[INSTALL] Installing base requirements with uv..."
uv pip install -r requirements-linux-gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu132 \
    --index-strategy unsafe-best-match

CUDA_ARCH="${CUDA_ARCH:-120}"
echo "[INSTALL] Using CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"

echo "[INSTALL] Installing llama-cpp-python with CUDA support (uv)..."
ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
export CUDACXX=/usr/local/cuda/bin/nvcc
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}"
export CMAKE_ARGS="-DGGML_CUDA=on"
uv pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu132 \
    --index-strategy unsafe-best-match

echo "[INSTALL] Installing additional runtime dependencies..."
uv pip install diskcache

echo "[INSTALL] Build complete."