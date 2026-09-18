#!/usr/bin/env bash
set -euo pipefail

cd /home/bqwang/project/GPR/model/U-3DGPR-Net

python - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA arch list:", torch.cuda.get_arch_list() if torch.cuda.is_available() else [])
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.is_available(), "CUDA is unavailable in server environment 3dinvnet"
assert "sm_120" in torch.cuda.get_arch_list(), "PyTorch wheel lacks RTX 5090 sm_120 kernels"
PY

export PYTHONPATH="$PWD/src"
python smoke_test.py

