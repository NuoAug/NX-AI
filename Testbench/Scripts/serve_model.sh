#!/usr/bin/env bash
# 测试台专用运行参数；前台运行，Ctrl+C关闭，不改正式启动脚本。
set -eu
BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$BENCH_DIR/.runtime"
exec 9>"$BENCH_DIR/.runtime/model.lock"
flock -n 9 || { echo '测试台模型入口已在运行。' >&2; exit 1; }
if pgrep -f "[l]lama-server.*--port ${LLAMA_PORT:-8080}([[:space:]]|$)" >/dev/null || curl --silent --max-time 2 http://127.0.0.1:8080/health >/dev/null; then
    echo '已有模型服务，保留现有进程；请勿重复启动。' >&2
    exit 1
fi
"$BENCH_DIR/../.venv/bin/python" - "$BENCH_DIR" "$$" <<'PY'
import json,sys
from pathlib import Path
b=Path(sys.argv[1]);pid=int(sys.argv[2]);ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
(b/'.runtime/model.pid').write_text(json.dumps({'pid':pid,'start_ticks':ticks}))
(b/'.runtime/approved_model.pid').write_text(str(pid))
PY
export LLAMA_MMPROJ="${LLAMA_MMPROJ:-$BENCH_DIR/data/models/mmproj-Q8_0.gguf}"
export LLAMA_BATCH="${LLAMA_BATCH:-256}"
export LLAMA_UBATCH="${LLAMA_UBATCH:-128}"
export LLAMA_ARG_CACHE_RAM="${LLAMA_ARG_CACHE_RAM:-0}"
export LLAMA_ARG_CTX_CHECKPOINTS="${LLAMA_ARG_CTX_CHECKPOINTS:-0}"
export LLAMA_ARG_IMAGE_MIN_TOKENS="${LLAMA_ARG_IMAGE_MIN_TOKENS:-8}"
exec bash "$BENCH_DIR/../tools/serve_llama.sh"
