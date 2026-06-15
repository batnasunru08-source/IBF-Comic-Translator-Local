#!/bin/bash
# Установка зависимостей для Linux x86_64 + NVIDIA GPU (CUDA 13.0)
# Используется uv вместо pip
set -e
cd "$(dirname "$0")"

echo "[INSTALL] Installing uv (fast pip replacement)..."
pip install -U uv

echo "[INSTALL] Installing base requirements with uv..."
# Используем unsafe-best-match, чтобы корректно смешивать PyPI и дополнительные индексы
uv pip install -r requirements-linux-gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu132 \
    --index-strategy unsafe-best-match

echo "[INSTALL] Detecting GPU compute capability..."
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
if [ -z "$CUDA_ARCH" ]; then
    echo "[INSTALL] Could not detect GPU arch, falling back to 86"
    CUDA_ARCH=86
fi
echo "[INSTALL] Using CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"

echo "[INSTALL] Installing llama-cpp-python with CUDA support (uv)..."
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
uv pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu132 \
    --index-strategy unsafe-best-match

echo "[INSTALL] Done. Now download the model:"
echo "  bash download-model.sh"