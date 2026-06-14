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

CUDA_ARCH="${CUDA_ARCH:-120}"
echo "[INSTALL] Using CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"

echo "[INSTALL] Installing llama-cpp-python with CUDA support..."
# The devel image ships a CUDA driver *stub* (libcuda.so, no .1 suffix) for
# build-time linking. Create the .so.1 symlink the linker expects and force
# -lcuda so the CLI tools link successfully without a real driver present.
ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
export CUDACXX=/usr/local/cuda/bin/nvcc
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}"
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH -DCMAKE_EXE_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs -lcuda' -DCMAKE_SHARED_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs -lcuda'" \
    pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade --no-deps

echo "[INSTALL] Installing additional runtime dependencies skipped by --no-deps..."
pip install diskcache

echo "[INSTALL] Build complete. Note: 'import llama_cpp' will only work at"
echo "runtime, where the NVIDIA driver (libcuda.so.1) is mounted via the"
echo "container runtime — it is not available during 'docker build'."

echo "[INSTALL] Done. Now download the model:"
echo "  bash download-model.sh"
