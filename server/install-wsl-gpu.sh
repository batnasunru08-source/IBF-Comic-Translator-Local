#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-linux-gpu.txt

# Paddle and torch currently disagree on exact cu130 runtime patch
# versions. Install the runtime set expected by torch explicitly, then
# install torch itself without dependency resolution.
python -m pip install \
  --upgrade \
  --force-reinstall \
  --no-deps \
  --extra-index-url https://pypi.nvidia.com \
  cuda-bindings==13.0.3 \
  nvidia-cuda-nvrtc==13.0.88 \
  nvidia-cuda-runtime==13.0.96 \
  nvidia-cuda-cupti==13.0.85 \
  nvidia-cudnn-cu13==9.15.1.9 \
  nvidia-cublas==13.1.0.3 \
  nvidia-cufft==12.0.0.61 \
  nvidia-curand==10.4.0.35 \
  nvidia-cusolver==12.0.4.66 \
  nvidia-cusparse==12.6.3.3 \
  nvidia-cusparselt-cu13==0.8.0 \
  nvidia-nccl-cu13==2.28.9 \
  nvidia-nvshmem-cu13==3.4.5 \
  nvidia-nvtx==13.0.85 \
  nvidia-nvjitlink==13.0.88 \
  nvidia-cufile==1.15.1.6 \
  triton==3.6.0

python -m pip install \
  --upgrade \
  --force-reinstall \
  --no-deps \
  torch==2.10.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130

python -m pip show torch nvidia-nccl-cu13 nvidia-cuda-runtime

python - <<'PY'
import paddle
import torch

print("torch:", torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print("paddle:", paddle.__version__, paddle.device.is_compiled_with_cuda(), paddle.device.cuda.device_count())
PY

echo
echo "Note: 'pip check' may still report a torch/paddle CUDA runtime pin mismatch."
echo "That comes from upstream wheel metadata for cu130."
