#!/bin/bash
# Установка зависимостей для Linux x86_64 + NVIDIA GPU (CUDA 13.0)
set -e
cd "$(dirname "$0")"

echo "[INSTALL] Upgrading pip..."
pip install -U pip setuptools wheel

echo "[INSTALL] Installing base requirements..."
pip install -r requirements.txt

echo "[INSTALL] Installing torch with CUDA 13.0..."
pip install torch --extra-index-url https://download.pytorch.org/whl/cu130

echo "[INSTALL] Installing llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir

echo "[INSTALL] Done. Now download the model:"
echo "  bash download-model.sh"
