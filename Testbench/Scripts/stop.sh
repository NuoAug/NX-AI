#!/usr/bin/env bash
# 停止测试台 HTTP 服务与其自有工作进程。只结束已核实的测试台进程；不触碰模型服务。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
RUNTIME_DIR="${BENCH_DIR}/.runtime"
PORT="$(python3 - "${BENCH_DIR}/Config/bench.json" <<'PY' 2>/dev/null || true
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        print(int(json.load(fh).get("port", 18780)))
except Exception:
    print(18780)
PY
)"
PORT="${PORT:-18780}"
HOST="127.0.0.1"

pid_start_ticks() {
  local pid="$1"
  [ -r "/proc/${pid}/stat" ] || return 0
  sed 's/.*) //' "/proc/${pid}/stat" 2>/dev/null | awk '{print $20}'
}

owned_process() {
  local pid="$1" ticks="$2" argv="$3"
  [ -n "${pid}" ] || return 1
  [ -n "${ticks}" ] || return 1
  [ -d "/proc/${pid}" ] || return 1
  local now
  now="$(pid_start_ticks "${pid}")"
  [ "${now}" = "${ticks}" ] || return 1
  tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -Fq -- "${argv}" || return 1
  return 0
}

read_pid_file() {
  local file="$1"
  [ -f "${file}" ] || return 1
  python3 - "${file}" <<'PY' 2>/dev/null || return 1
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    data = json.load(fh)
print(int(data["pid"]), int(data.get("start_ticks") or 0))
PY
}

kill_owned() {
  local label="$1" pid="$2" ticks="$3" argv="$4" grace="${5:-10}"
  if ! owned_process "${pid}" "${ticks}" "${argv}"; then
    echo "skip ${label}: pid ${pid} 归属校验未通过（不存在、启动时间变化或命令不匹配）"
    return 0
  fi
  echo "stopping ${label}: pid ${pid}"
  kill -TERM "${pid}" 2>/dev/null || true
  local waited=0
  while [ -d "/proc/${pid}" ] && [ "${waited}" -lt $((grace * 10)) ]; do
    sleep 0.1
    waited=$((waited + 1))
  done
  if [ -d "/proc/${pid}" ]; then
    if owned_process "${pid}" "${ticks}" "${argv}"; then
      echo "force kill ${label}: pid ${pid}"
      kill -KILL "${pid}" 2>/dev/null || true
    else
      echo "warn ${label}: pid ${pid} 已不归属测试台，放弃强制结束"
    fi
  fi
}

# 先停 server：它负责停止 worker
SERVER_PID_FILE="${RUNTIME_DIR}/server.pid"
if read_pid_file "${SERVER_PID_FILE}" >/dev/null 2>&1; then
  read -r SERVER_PID SERVER_TICKS < <(read_pid_file "${SERVER_PID_FILE}")
  kill_owned "server" "${SERVER_PID}" "${SERVER_TICKS}" "Testbench/server.py" 15
  rm -f "${SERVER_PID_FILE}"
else
  echo "server pid 文件不存在：${SERVER_PID_FILE}"
fi

# 再核对残留 worker（仅结束命令匹配 Testbench/worker.py 的进程）
WORKER_PID_FILE="${RUNTIME_DIR}/worker.json"
if read_pid_file "${WORKER_PID_FILE}" >/dev/null 2>&1; then
  read -r WORKER_PID WORKER_TICKS < <(read_pid_file "${WORKER_PID_FILE}")
  kill_owned "worker" "${WORKER_PID}" "${WORKER_TICKS}" "Testbench/worker.py" 10
  rm -f "${WORKER_PID_FILE}"
fi

# 核对端口已释放（仅报告，不杀未知进程）
if python3 - "$HOST" "$PORT" <<'PY' 2>/dev/null
import socket, sys
sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect((sys.argv[1], int(sys.argv[2])))
except OSError:
    raise SystemExit(0)
finally:
    sock.close()
raise SystemExit(1)
PY
then
  echo "stopped: ${HOST}:${PORT} 已释放"
else
  echo "warn: ${HOST}:${PORT} 仍有进程监听；请用 'ss -ltnp' 核对该进程是否属于本测试台" >&2
  exit 1
fi
