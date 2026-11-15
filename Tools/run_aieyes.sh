#!/usr/bin/env bash
# 阶段 8：目标板一键启动（先 llama-server，再交互服务）。
# 依赖：项目已部署到 $DEST（默认 /home/mylin/Desktop/Nuo），.venv 已装依赖，models/ 已就位。
# 可以在 SSH 里执行；进程用 setsid 挂起，SSH 断开不影响运行。
set -uo pipefail

DEST="${AIEYES_DEST:-/home/mylin/Desktop/Nuo}"
cd "$DEST" || { echo "NO_DEST $DEST"; exit 1; }
mkdir -p logs evidence

health() {
  curl -s -m 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"ok"'
}

if health; then
  echo "LLAMA_ALREADY_RUNNING"
else
  echo "LLAMA_STARTING"
  setsid nohup bash Tools/serve_llama.sh >/tmp/llama_server_stdout.log 2>&1 </dev/null &
  for _ in $(seq 1 60); do
    health && break
    sleep 1
  done
  if ! health; then
    echo "LLAMA_START_FAILED 见 /tmp/llama_server_stdout.log"
    exit 2
  fi
  echo "LLAMA_READY"
fi

export PYTHONPATH="$DEST/src"
setsid nohup .venv/bin/python -m aieyes.main --config Config/target.json \
  >logs/service.log 2>&1 </dev/null &
PID=$!
sleep 2
if kill -0 "$PID" 2>/dev/null; then
  echo "SERVICE_STARTED pid=$PID log=$DEST/logs/service.log"
else
  echo "SERVICE_EXITED_EARLY 见 logs/service.log"
  exit 3
fi
