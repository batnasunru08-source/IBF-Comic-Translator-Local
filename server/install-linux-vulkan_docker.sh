#!/bin/bash
# Установка зависимостей для Docker-образа с Vulkan ICD (любой вендор).
# Использует prebuilt vulkan-колесо llama-cpp-python (собирать из исходников
# не нужно: https://abetlen.github.io/llama-cpp-python/whl/vulkan).
set -e
cd "$(dirname "$0")"

echo "[INSTALL] Installing base requirements with uv (CPU torch)..."
uv pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    -r requirements-vulkan.txt

echo "[INSTALL] Installing prebuilt llama-cpp-python (Vulkan)..."
uv pip install \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan \
    --index-strategy unsafe-best-match \
    llama-cpp-python

echo "[INSTALL] Build complete."
