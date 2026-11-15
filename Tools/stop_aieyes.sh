#!/usr/bin/env bash
# 阶段 8：停止交互服务与 llama-server（先停服务，保证音频/证据正常收尾）。
set -uo pipefail

DEST="${AIEYES_DEST:-/home/mylin/Desktop/Nuo}"
cd "$DEST" 2>/dev/null || true

if pgrep -f "aieyes.main" >/dev/null 2>&1; then
  pkill -TERM -f "aieyes.main" || true
  for _ in $(seq 1 10); do
    pgrep -f "aieyes.main" >/dev/null 2>&1 || break
    sleep 1
  done
  echo "SERVICE_STOPPED"
else
  echo "SERVICE_NOT_RUNNING"
fi

if pgrep -f "[l]lama-server" >/dev/null 2>&1; then
  pkill -f "[l]lama-server" || true
  echo "LLAMA_STOPPED"
else
  echo "LLAMA_NOT_RUNNING"
fi
