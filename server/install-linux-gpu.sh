#!/bin/bash
# Установка зависимостей для Linux x86_64 + NVIDIA GPU (CUDA 13.0)
set -e
cd "$(dirname "$0")"

echo "[INSTALL] Upgrading pip..."
pip install -U pip setuptools wheel

echo "[INSTALL] Installing base requirements..."
pip install -r requirements.txt

echo "[INSTALL] Installing torch + torchvision with CUDA 13.0..."
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu130

echo "[INSTALL] Detecting GPU compute capability..."
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
if [ -z "$CUDA_ARCH" ]; then
    echo "[INSTALL] Could not detect GPU arch, falling back to 86"
    CUDA_ARCH=86
fi
echo "[INSTALL] Using CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"

echo "[INSTALL] Installing llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH" \
    pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade --no-deps

echo "[INSTALL] Done. Now download the model:"
echo "  bash download-model.sh"
