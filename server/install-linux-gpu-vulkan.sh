#!/bin/bash
# Установка зависимостей для Linux + GPU через Vulkan (NVIDIA / AMD / Intel)
# Не требует CUDA toolkit / ROCm — только Vulkan loader + dev headers.
set -e
cd "$(dirname "$0")"

echo "[INSTALL] Checking Vulkan availability..."
if ! command -v vulkaninfo &> /dev/null; then
    echo "[INSTALL] vulkaninfo not found, installing Vulkan SDK/tools..."
    sudo apt-get update
    sudo apt-get install -y libvulkan-dev vulkan-tools glslc
fi

echo "[INSTALL] Vulkan summary:"
vulkaninfo --summary || echo "[INSTALL] WARNING: vulkaninfo failed — GPU may not be visible to Vulkan."

echo "[INSTALL] Upgrading pip..."
pip install -U pip setuptools wheel

echo "[INSTALL] Installing base requirements (incl. torch CPU for PaddleOCR)..."
pip install -r requirements-linux-gpu-vulkan.txt

echo "[INSTALL] Building llama-cpp-python with Vulkan support..."
CMAKE_ARGS="-DGGML_VULKAN=on" \
    pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade --no-deps

echo "[INSTALL] Done. Now download the model:"
echo "  bash download-model.sh"
