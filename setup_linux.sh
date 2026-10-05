#!/usr/bin/env bash
# SAMMatte Linux 环境一键搭建（在服务器上有 root 权限时不需要 sudo）
#   - 在项目目录内创建 venv（Ubuntu 缺少 python3-venv 时自动引导 pip）
#   - 安装与 Windows 已知可用环境一致的依赖（torch 2.7.1+cu128 等）
#   - 下载静态 ffmpeg 到 tools/ffmpeg_linux（无 ffmpeg 时回退到 imageio-ffmpeg）
#
# 可覆盖：
#   PIP_INDEX   PyPI 索引，默认 https://pypi.tuna.tsinghua.edu.cn/simple
#   TORCH_VARIANT cu128 | cu126 | cu130，默认 cu128
#   SAMMATTE_VENV 虚拟环境目录，默认 <项目>/venv
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SAMMATTE_VENV:-$ROOT/venv}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_VARIANT="${TORCH_VARIANT:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"

echo "== SAMMatte Linux 环境搭建 =="
echo "项目目录: $ROOT"
echo "虚拟环境: $VENV"
echo "PyPI 索引: $PIP_INDEX"
echo "torch: $TORCH_VERSION / torchvision: $TORCHVISION_VERSION / $TORCH_VARIANT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "[ERROR] 未找到 $PYTHON_BIN"; exit 1; }

if [ ! -x "$VENV/bin/python" ]; then
  echo "-- 创建 venv --"
  if "$PYTHON_BIN" -c "import ensurepip" >/dev/null 2>&1; then
    "$PYTHON_BIN" -m venv "$VENV"
  else
    echo "   （系统缺少 ensurepip，使用 --without-pip + get-pip.py）"
    "$PYTHON_BIN" -m venv --without-pip "$VENV"
  fi
fi

PY="$VENV/bin/python"
PIP="$VENV/bin/python -m pip"

if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "-- 引导 pip --"
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  "$PY" /tmp/get-pip.py --index-url "$PIP_INDEX"
fi

echo "-- 升级构建工具 --"
$PIP install -U pip setuptools wheel --index-url "$PIP_INDEX"

echo "-- 安装 PyTorch（$TORCH_VARIANT） --"
$PIP install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  --index-url "https://download.pytorch.org/whl/$TORCH_VARIANT"

echo "-- 安装项目依赖 --"
$PIP install -r "$ROOT/requirements.txt" --index-url "$PIP_INDEX"

echo "-- 对齐 Windows 已知可用版本（VideoMaMa 对 diffusers 版本敏感） --"
$PIP install --index-url "$PIP_INDEX" \
  "diffusers==0.35.1" "accelerate==1.14.0" "timm==1.0.27" "ftfy==6.1.1" \
  "opencv-python==4.13.0.92" || true

echo "-- ffmpeg 兜底包（当静态 ffmpeg 不可用时） --"
$PIP install --index-url "$PIP_INDEX" imageio-ffmpeg

echo "-- 下载静态 ffmpeg 到 tools/ffmpeg_linux --"
FFMPEG_DIR="$ROOT/tools/ffmpeg_linux"
mkdir -p "$FFMPEG_DIR"
if [ -x "$FFMPEG_DIR/ffmpeg" ]; then
  echo "   已存在：$FFMPEG_DIR/ffmpeg"
else
  OK=0
  for URL in \
    "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-linux64-gpl.tar.xz" \
    "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"; do
    echo "   尝试 $URL"
    if curl -fL --retry 3 -o /tmp/ffmpeg-static.tar.xz "$URL" && \
       tar -xf /tmp/ffmpeg-static.tar.xz -C /tmp; then
      SRC="$(find /tmp -maxdepth 2 -type f -name ffmpeg | head -1)"
      PROBE="$(find /tmp -maxdepth 2 -type f -name ffprobe | head -1)"
      if [ -n "$SRC" ]; then
        cp "$SRC" "$FFMPEG_DIR/ffmpeg"
        [ -n "$PROBE" ] && cp "$PROBE" "$FFMPEG_DIR/ffprobe"
        chmod +x "$FFMPEG_DIR/ffmpeg" "$FFMPEG_DIR/ffprobe" 2>/dev/null || true
        OK=1
      fi
    fi
    rm -rf /tmp/ffmpeg-static.tar.xz /tmp/ffmpeg-*-static /tmp/ffmpeg-master-latest-linux64-gpl
    [ "$OK" = "1" ] && break
  done
  if [ "$OK" != "1" ]; then
    echo "   [WARN] 静态 ffmpeg 下载失败；应用会回退到 imageio-ffmpeg 或系统 PATH 中的 ffmpeg。"
  fi
fi

echo "-- 脚本可执行权限 --"
chmod +x "$ROOT"/*.sh 2>/dev/null || true
if [ -d "$FFMPEG_DIR" ]; then
  chmod +x "$FFMPEG_DIR"/* 2>/dev/null || true
fi

echo
echo "== 完成，开始自检 =="
bash "$ROOT/check_runtime.sh"
echo
echo "启动（前台）： bash run_SAMMatte.sh"
echo "启动（后台）： bash start_sammatte.sh   停止： bash stop_sammatte.sh"
