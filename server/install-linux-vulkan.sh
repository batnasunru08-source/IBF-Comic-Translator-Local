#!/bin/bash
# Установка зависимостей для Linux x86_64 + Vulkan ICD (любой вендор).
# Без Docker: используется для запуска `python -m uvicorn` напрямую.
# Требует системные пакеты: libvulkan1, mesa-vulkan-drivers, vulkan-tools,
# python3.12-dev (для wheel).
set -e
cd "$(dirname "$0")"

echo "[INSTALL] Installing uv (fast pip replacement)..."
pip install -U uv

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

echo "[INSTALL] Done. Run with:"
echo "  LLAMA_BACKEND=vulkan python -m uvicorn main:app --host 0.0.0.0 --port 8000"
