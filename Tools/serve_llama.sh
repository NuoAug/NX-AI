#!/usr/bin/env bash
# 启动 llama-server（CUDA 后端 + OpenAI 兼容接口 + 视觉投影），供 interaction 层调用。
# 用法：bash Tools/serve_llama.sh            # 前台
#       setsid nohup bash Tools/serve_llama.sh >/tmp/llama_server_stdout.log 2>&1 </dev/null &
set -u

DEST="${AIEYES_DEST:-/home/mylin/Desktop/Nuo}"
LLAMA="${LLAMA_BIN:-$DEST/third_party/llama.cpp/build/bin/llama-server}"
MODEL="${LLAMA_MODEL:-$DEST/models/vlm/Qwen3.5-2B-Q4_K_M.gguf}"
MMPROJ="${LLAMA_MMPROJ:-$DEST/models/vlm/mmproj-F16.gguf}"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8080}"
CTX="${LLAMA_CTX:-4096}"
NGL="${LLAMA_NGL:-99}"
THREADS="${LLAMA_THREADS:-4}"
PARALLEL="${LLAMA_PARALLEL:-1}"
BATCH="${LLAMA_BATCH:-2048}"
UBATCH="${LLAMA_UBATCH:-512}"
ALIAS="${LLAMA_ALIAS:-qwen35}"

for f in "$LLAMA" "$MODEL" "$MMPROJ"; do
    if [ ! -e "$f" ]; then
        echo "missing: $f" >&2
        exit 2
    fi
done

mkdir -p "$DEST/logs"
echo "start $(date -Is) llama=$LLAMA model=$MODEL mmproj=$MMPROJ port=$PORT ctx=$CTX parallel=$PARALLEL ngl=$NGL"
exec "$LLAMA" \
    --model "$MODEL" \
    --mmproj "$MMPROJ" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$CTX" \
    --n-gpu-layers "$NGL" \
    --threads "$THREADS" \
    --jinja \
    --alias "$ALIAS" \
    --parallel "$PARALLEL" \
    --batch-size "$BATCH" \
    --ubatch-size "$UBATCH" \
    --metrics
