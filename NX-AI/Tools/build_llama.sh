#!/usr/bin/env bash
# 在目标机（Jetson Orin NX）上构建带 CUDA 后端的 llama.cpp。
# 由 Lead 在本机 `source Tools/target_env.sh && tssh` 调用；日志固定写 /tmp/llama_build.log。
set -u

DEST="${AIEYES_DEST:-/home/mylin/Desktop/Nuo}"
SRC="$DEST/third_party/llama.cpp"
LOG="${LLAMA_BUILD_LOG:-/tmp/llama_build.log}"
JOBS="${JOBS:-8}"

cd "$SRC" || { echo "missing $SRC" >&2; exit 1; }

# 非交互 SSH 的 PATH 里没有 CUDA，显式补上（Jetson 的 CUDA 装在 /usr/local/cuda）
if [ -d /usr/local/cuda/bin ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
  export CUDACXX=/usr/local/cuda/bin/nvcc
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

echo "=== $(date -Is) configure ===" >>"$LOG"
echo "cwd=$PWD nvcc=$(command -v nvcc || echo missing) jobs=$JOBS" >>"$LOG"
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release >>"$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then echo "CONFIGURE_EXIT=$rc" >>"$LOG"; exit "$rc"; fi

echo "=== $(date -Is) build -j$JOBS ===" >>"$LOG"
cmake --build build --config Release -j"$JOBS" >>"$LOG" 2>&1
rc=$?
echo "BUILD_EXIT=$rc" >>"$LOG"
echo "=== $(date -Is) done ===" >>"$LOG"
exit "$rc"
