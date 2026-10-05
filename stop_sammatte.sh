#!/usr/bin/env bash
# 停止由 start_sammatte.sh 启动的 SAMMatte 服务。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$ROOT/sammatte.pid"

if [ ! -f "$PIDFILE" ]; then
  echo "[stop] 未找到 pid 文件（$PIDFILE），尝试按命令行匹配进程。"
  MATCHED="$(pgrep -f 'run_sam31_webapp.py' || true)"
  if [ -z "$MATCHED" ]; then
    echo "[stop] 没有正在运行的 SAMMatte 服务。"
    exit 0
  fi
  echo "[stop] 结束进程：$MATCHED"
  kill $MATCHED 2>/dev/null || true
  sleep 2
  kill -9 $MATCHED 2>/dev/null || true
  exit 0
fi

PID="$(cat "$PIDFILE" 2>/dev/null || true)"
if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
  echo "[stop] pid 文件存在但进程不在运行，清理 pid 文件。"
  rm -f "$PIDFILE"
  exit 0
fi

echo "[stop] 发送 SIGTERM 到 pid=$PID"
kill "$PID" 2>/dev/null || true
for _ in $(seq 1 15); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[stop] 已停止。"
    rm -f "$PIDFILE"
    exit 0
  fi
  sleep 1
done

echo "[stop] 超时，发送 SIGKILL。"
kill -9 "$PID" 2>/dev/null || true
rm -f "$PIDFILE"
echo "[stop] 已强制停止。"
