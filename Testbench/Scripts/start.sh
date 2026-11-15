#!/usr/bin/env bash
# 启动测试台 HTTP 服务（仅本机回环）。不启动录音、相机、模型或综合测试。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
PROJECT_ROOT="$(cd "${BENCH_DIR}/.." && pwd -P)"

# 核对真实绝对路径：测试台必须在项目根下的 Testbench/ 内
case "${BENCH_DIR}" in
  "${PROJECT_ROOT}/Testbench") ;;
  *)
    echo "error: 测试台目录不在 <项目根>/testbench 内: ${BENCH_DIR}" >&2
    exit 1
    ;;
esac

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
PYTHON="${BENCH_DIR}/../.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="$(command -v python3)"
fi

RUNTIME_DIR="${BENCH_DIR}/.runtime"
LOGS_DIR="${BENCH_DIR}/logs"
for dir in "${BENCH_DIR}/runs" "${BENCH_DIR}/uploads" "${RUNTIME_DIR}" "${LOGS_DIR}"; do
  mkdir -p "${dir}"
done

export PYTHONPATH="${PROJECT_ROOT}/Src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${RUNTIME_DIR}/cache"
export TMPDIR="${RUNTIME_DIR}/tmp"
export CUDA_CACHE_PATH="${RUNTIME_DIR}/cuda"
mkdir -p "${XDG_CACHE_HOME}" "${TMPDIR}" "${CUDA_CACHE_PATH}"

pid_start_ticks() {
  # 读取 /proc/<pid>/stat 的字段 22（启动时间 ticks）；读取失败输出空
  local pid="$1"
  [ -r "/proc/${pid}/stat" ] || return 0
  sed 's/.*) //' "/proc/${pid}/stat" 2>/dev/null | awk '{print $20}'
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

owned_process() {
  local pid="$1" ticks="$2" argv="$3"
  [ -n "${pid}" ] || return 1
  [ -d "/proc/${pid}" ] || return 1
  local now
  now="$(pid_start_ticks "${pid}")"
  [ -n "${now}" ] || return 1
  [ "${now}" = "${ticks}" ] || return 1
  tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -Fq -- "${argv}" || return 1
  return 0
}

port_busy() {
  python3 - "$HOST" "$PORT" <<'PY' 2>/dev/null
import socket, sys
sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect((sys.argv[1], int(sys.argv[2])))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
raise SystemExit(0)
PY
}

SERVER_PID_FILE="${RUNTIME_DIR}/server.pid"

# 已有本测试台服务在运行 → 直接报告地址，不重复启动
if read_pid_file "${SERVER_PID_FILE}" >/dev/null 2>&1; then
  read -r OLD_PID OLD_TICKS < <(read_pid_file "${SERVER_PID_FILE}")
  if owned_process "${OLD_PID}" "${OLD_TICKS}" "Testbench/server.py"; then
    if curl --max-time 2 -fsS "http://${HOST}:${PORT}/api/health" >/dev/null 2>&1; then
      echo "already running: http://${HOST}:${PORT}/ (pid ${OLD_PID})"
      exit 0
    fi
  fi
  rm -f "${SERVER_PID_FILE}"
fi

if port_busy; then
  echo "error: ${HOST}:${PORT} 已被其它进程占用；测试台不杀进程、不改端口。" >&2
  echo "使用 'ss -ltnp' 核对占用进程后由用户处理。" >&2
  exit 1
fi

# 日志轮转：超过 10 MiB 时保留最近 5 份
LOG_FILE="${LOGS_DIR}/server.log"
if [ -f "${LOG_FILE}" ]; then
  SIZE="$(wc -c < "${LOG_FILE}" | tr -d ' ')"
  if [ "${SIZE:-0}" -ge 10485760 ]; then
    for index in 4 3 2 1; do
      [ -f "${LOG_FILE}.${index}" ] && mv -f "${LOG_FILE}.${index}" "${LOG_FILE}.$((index + 1))"
    done
    mv -f "${LOG_FILE}" "${LOG_FILE}.1"
  fi
fi

cd "${BENCH_DIR}"
nohup "${PYTHON}" -B "${BENCH_DIR}/server.py" >>"${LOG_FILE}" 2>&1 &
SERVER_PID=$!
TICKS="$(pid_start_ticks "${SERVER_PID}")"
python3 - "$SERVER_PID_FILE" "$SERVER_PID" "${TICKS:-0}" <<'PY'
import json, sys
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump({"pid": int(sys.argv[2]), "start_ticks": int(sys.argv[3] or 0)}, fh)
PY

for _ in $(seq 1 50); do
  if curl --max-time 1 -fsS "http://${HOST}:${PORT}/api/health" >/dev/null 2>&1; then
    echo "started: http://${HOST}:${PORT}/ (pid ${SERVER_PID})"
    echo "log: ${LOG_FILE}"
    exit 0
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    break
  fi
  sleep 0.2
done

echo "error: 服务未在预期时间内就绪，请查看 ${LOG_FILE}" >&2
exit 1
