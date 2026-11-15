# 运行器构建规范（目标板 llama.cpp CUDA 版）

本文件记录目标板推理运行器的构建方式与已验证环境。构建产物不随仓库分发，
源码包在 `runtime/llama.cpp-master.tar.gz`（38,847,586 B，sha256 `4b5a5fda135b1248…`）。

## 1. 已验证环境

| 项 | 值 |
|---|---|
| 目标板 | Orin NX 16GB，JetPack 7.2.1 / L4T 39.2.1 / CUDA 13.2.86 |
| 系统 | Ubuntu 24.04.4 LTS，内核 6.8.12-1021-tegra，aarch64，8 核 |
| 编译器 | GCC 13.3.0；CMake 3.x；Ninja（可选） |
| 构建目录 | `/home/mylin/Desktop/Nuo/third_party/llama.cpp`（源码）+ `build/` |
| 产物版本 | `llama-server version 0.9.0-dev (build 0, commit unknown)` |
| 产物校验 | `build/bin/llama-server`、`build/bin/llama-bench`、`libggml-cuda.so.*` |

## 2. 构建命令

```bash
cd /home/mylin/Desktop/Nuo/third_party/llama.cpp
export PATH="/usr/local/cuda/bin:$PATH"
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j8
```

要点：

- `-DGGML_CUDA=ON` 才启用 GPU（本机 25W 下实测 `pp64 405 t/s / tg16 15.8 t/s`）；
- `-DCMAKE_CUDA_ARCHITECTURES=87` 对应 Orin 的 Ampere 8.7；
- 非交互 SSH 的 PATH 不含 `/usr/local/cuda/bin`，必须先 export，否则 configure 报
  `CUDA compiler not found`；
- `-DLLAMA_CURL=OFF` 避免构建期联网下载依赖。

## 3. 验证方法

```bash
# 1) 二进制链接的 CUDA 库
ldd build/bin/llama-server | grep -E 'cudart|cublas|libcuda'

# 2) 确认后端走 CUDA（应输出 compute capability 8.7 且 backend 为 CUDA）
build/bin/llama-bench -m /home/mylin/Desktop/Nuo/models/vlm/Qwen3.5-2B-Q4_K_M.gguf -ngl 99 -p 64 -n 16 -r 1

# 3) 服务健康
curl -s http://127.0.0.1:8080/health
```

启动参数见 `NX-AI/tools/serve_llama.sh`；阶段性验收数字（内存/首声/时延）见
`Docs/报告/阶段7-固定基线与性能优化.md`。
