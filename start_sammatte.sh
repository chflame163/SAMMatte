#!/usr/bin/env bash
# 后台启动 SAMMatte（日志写入 logs/，pid 写入 sammatte.pid），并等待健康检查通过。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/sammatte.log"
PIDFILE="$ROOT/sammatte.pid"
export SAM31_HOST="${SAM31_HOST:-0.0.0.0}"
export SAM31_PORT="${SAM31_PORT:-8765}"

mkdir -p "$LOG_DIR"

if [ -f "$PIDFILE" ]; then
  OLD_PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[ERROR] SAMMatte 已在运行（pid=$OLD_PID）。请先执行 bash stop_sammatte.sh"
    exit 1
  fi
  rm -f "$PIDFILE"
fi

echo "[start] host=$SAM31_HOST port=$SAM31_PORT log=$LOG_FILE"
setsid nohup bash "$ROOT/run_SAMMatte.sh" >>"$LOG_FILE" 2>&1 </dev/null &
SERVER_PID=$!
echo "$SERVER_PID" >"$PIDFILE"
echo "[start] 已启动，pid=$SERVER_PID"

HEALTH_OK=0
for _ in $(seq 1 60); do
  sleep 1
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[ERROR] 进程已退出，最后 40 行日志："
    tail -n 40 "$LOG_FILE" || true
    rm -f "$PIDFILE"
    exit 1
  fi
  if curl -fsS -m 3 "http://127.0.0.1:$SAM31_PORT/api/health" >/dev/null 2>&1; then
    HEALTH_OK=1
    break
  fi
done

if [ "$HEALTH_OK" != "1" ]; then
  echo "[ERROR] 健康检查超时（/api/health）。最后 40 行日志："
  tail -n 40 "$LOG_FILE" || true
  exit 1
fi

LAN_IPS="$(hostname -I 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]' | paste -sd' ' -)"
echo "[start] 健康检查通过。"
echo "[start] 本机访问： http://127.0.0.1:$SAM31_PORT"
for ip in $LAN_IPS; do
  echo "[start] 局域网访问： http://$ip:$SAM31_PORT"
done
echo "[start] 日志： $LOG_FILE      停止： bash stop_sammatte.sh"
