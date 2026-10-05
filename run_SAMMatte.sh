#!/usr/bin/env bash
# SAMMatte Linux 启动脚本（前台运行，等价于 Windows 的 run_SAMMatte.bat）
#
# 可覆盖的环境变量：
#   SAM31_HOST   监听地址，默认 0.0.0.0（允许局域网访问）
#   SAM31_PORT   监听端口，默认 8765（请勿使用 8080 / 8188）
#   SAMMATTE_VENV 虚拟环境目录，默认 <项目>/venv
#   SAM31_FFMPEG   指定 ffmpeg 可执行文件；留空则自动查找
#   SAM31_MAX_INFERENCE_PIXELS  默认 SAM 推理像素上限，例如 1280x720
#   SAM31_VITMATTE_DEVICE       gpu | cpu
#   SAM31_VIDEOMAMA_MAX_RESOLUTION  256..2048
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$ROOT/app"
MODELS="$ROOT/models"
VENV="${SAMMATTE_VENV:-$ROOT/venv}"
PY="$VENV/bin/python"

export SAM31_HOST="${SAM31_HOST:-0.0.0.0}"
export SAM31_PORT="${SAM31_PORT:-8765}"

if [ ! -x "$PY" ]; then
  echo "[ERROR] 未找到虚拟环境 Python：$PY"
  echo "        请先执行：bash setup_linux.sh   （或在 SAMMATTE_VENV 指定环境目录）"
  exit 1
fi

if [ ! -f "$APP/run_sam31_webapp.py" ]; then
  echo "[ERROR] 未找到应用入口：$APP/run_sam31_webapp.py"
  exit 1
fi

if [ ! -d "$MODELS/sam3" ]; then
  echo "[ERROR] 缺少 SAM 3 源码目录：$MODELS/sam3"
  exit 1
fi

if [ ! -f "$MODELS/sam3.1/sam3.1_multiplex.pt" ]; then
  echo "[ERROR] 缺少 SAM 3.1 权重：$MODELS/sam3.1/sam3.1_multiplex.pt"
  exit 1
fi

if [ ! -d "$MODELS/vitmatte-base-composition-1k" ]; then
  echo "[WARN] 未找到 ViTMatte 目录：$MODELS/vitmatte-base-composition-1k"
fi

if [ ! -d "$MODELS/VideoMaMa" ]; then
  echo "[WARN] 未找到 VideoMaMa 目录：$MODELS/VideoMaMa"
fi

export PYTHONPATH="$APP:$MODELS/sam3${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/tools/ffmpeg_linux:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[runtime] python=$PY"
echo "[runtime] host=$SAM31_HOST port=$SAM31_PORT"
echo "[runtime] ffmpeg 搜索路径：\${SAM31_FFMPEG} -> tools/ffmpeg_linux -> tools/ffmpeg -> PATH -> imageio-ffmpeg"

exec "$PY" "$APP/run_sam31_webapp.py" --host "$SAM31_HOST" --port "$SAM31_PORT" "$@"
