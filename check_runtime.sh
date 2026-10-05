#!/usr/bin/env bash
# Linux 版环境自检（等价于 check_runtime.bat）。不会占用显存，可在其他任务使用 GPU 时运行。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SAMMATTE_VENV:-$ROOT/venv}"
PY="$VENV/bin/python"
MODELS="$ROOT/models"

if [ ! -x "$PY" ]; then
  echo "[ERROR] 未找到虚拟环境 Python：$PY"
  echo "        请先执行：bash setup_linux.sh"
  exit 1
fi

export PYTHONPATH="$ROOT/app:$MODELS/sam3${PYTHONPATH:+:$PYTHONPATH}"

"$PY" - "$MODELS" <<'PYEOF'
import os
import subprocess
import sys

models = sys.argv[1]
head = os.path.join(models, "sam3.1", "sam3.1_multiplex.pt")

import numpy
import cv2
import torch
import torchvision
import transformers
import diffusers
import timm
import accelerate
import scipy
import einops
import psutil
import iopath
import pkg_resources as _pr

print("python          ", sys.version.split()[0])
print("python_exe      ", sys.executable)
print("pkg_resources   ", getattr(_pr, "__file__", "?"))
print("torch           ", torch.__version__, "cuda_runtime", torch.version.cuda,
      "cuda_available", torch.cuda.is_available())
print("torchvision     ", torchvision.__version__)
print("numpy           ", numpy.__version__)
print("cv2             ", cv2.__version__)
print("timm            ", timm.__version__)
print("transformers    ", transformers.__version__)
print("diffusers       ", diffusers.__version__)
print("accelerate      ", accelerate.__version__)
print("scipy           ", scipy.__version__)
print("psutil          ", psutil.__version__)
print("sam3_repo_exists", os.path.isdir(os.path.join(models, "sam3")))
print("sam31_checkpoint_exists", os.path.isfile(head))
print("vitmatte_exists", os.path.isdir(os.path.join(models, "vitmatte-base-composition-1k")))
print("videomama_repo_exists", os.path.isdir(os.path.join(models, "VideoMaMa")))

# 注意：导入 sam3 会触发 CUDA 上下文初始化，显存被其他任务占满时可能失败。
try:
    import sam3.model_builder  # noqa: F401
    print("sam3_import     OK")
except Exception as exc:
    print("sam3_import     FAILED:", type(exc).__name__, exc)
    print("                （若为 CUDA out of memory，说明显存被其他任务占满，先释放后再试）")

sys.path.insert(0, os.path.join(models, "VideoMaMa"))
try:
    import pipeline_svd_mask  # noqa: F401
    print("videomama_pipeline_import OK")
except Exception as exc:
    print("videomama_pipeline_import FAILED:", type(exc).__name__, exc)

from sam31_webapp.service import describe_ffmpeg
print(describe_ffmpeg())
PYEOF

echo
echo "--- GPU（nvidia-smi，不申请显存） ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
  if [ -n "${SAMMATTE_SHOW_GPU_PROCS:-}" ]; then
    echo "PID, 进程, 显存占用"
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || echo "[WARN] 查询显存占用进程失败"
  else
    echo "（提示：export SAMMATTE_SHOW_GPU_PROCS=1 可同时列出占用显存的进程）"
  fi
else
  echo "[WARN] 未找到 nvidia-smi"
fi

echo
echo "环境检查完成。"
