# NX-AI：模型资产与模型工具

> 本机目录于 2026-11-15 整理为 `Models/`、`Runtime/`、`Tools/`。下文目标机绝对路径保留原部署值，目标机未迁移。部署与实机验证流程仅供操作参考，本次没有执行。清单本次只更新路径，原资产哈希和生成时间保留。

本目录集中保存模型资产、运行器及模型工具；根目录 `Tools/` 保留已有部署工具副本。
部署到目标板时，`NX-AI/Models/` 与目标机 `/home/mylin/Desktop/Nuo/models/` 一一对应。

## 1. 目录职责

| 目录 | 职责 | 维护方式 |
|---|---|---|
| `Models/` | 目标板实际加载的模型文件（GGUF / onnx / 词表 / 配置） | 独立保存，不覆盖原始资产 |
| `Runtime/` | 运行器资产：llama.cpp 源码包、随包下发的 Windows CPU 运行器与 piper 运行器（仅留档） | 独立保存 |
| `Tools/` | 模型相关工具源码：构建、启动、内存/时延探针、夹具生成 | 随工程维护 |
| `manifest.json` | 全部模型/运行器文件的 sha256 清单，用于部署与换机核对 | 随工程维护 |

## 2. 模型资产清单（关键文件）

完整清单见 `manifest.json`（771 个清单条目），下表只列被交互子系统直接加载的项。

| 文件（相对本目录） | 字节 | sha256 前 16 | 来源 / 许可 |
|---|---|---|---|
| `Models/vlm/Qwen3.5-2B-Q4_K_M.gguf` | 1,280,835,840 | `aaf42c8b7c3cab2b` | Unsloth 量化，base `Qwen/Qwen3.5-2B`，apache-2.0；HF revision `90057e31161eb95cc0bc1413c4f53b44de9b49c8` |
| `Models/vlm/mmproj-F16.gguf` | 668,227,264 | `7035e9cb8d7c6a96` | 同上（视觉投影） |
| `Models/asr-streaming/sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx` | 165,462,184 | `81a70226a8934e6e` | sherpa-onnx 官方模型 |
| `Models/asr-streaming/.../decoder.int8.onnx` | 71,664,561 | `f3cca9f77bb9d93c` | 同上 |
| `Models/ocr/pp-ocrv6-small/PP-OCRv6_det_small.onnx` | 9,929,594 | `090f04abcd9d9a74` | PaddleOCR PP-OCRv6 small |
| `Models/ocr/pp-ocrv6-small/PP-OCRv6_rec_small.onnx` | 21,234,383 | `6f327246b50388f3` | 同上（字典在 onnx metadata `character` 中，18708 字符） |
| `Models/tts/piper-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx` | 63,201,423 | `04234ec24907e9ef` | Piper zh_CN-huayan-medium |
| `Models/speaker/campplus-zh-cn-16k/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx` | 28,281,138 | `f682b514c05d947e` | 3D-Speaker CAM++ 中文 16k |
| `Models/vad/silero-vad-v6.2.3/silero_vad.onnx` | 2,327,524 | `1a153a22f4509e29` | Silero VAD |
| `Models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/` | 目录 | — | sherpa-onnx 关键词模型；“小冰”词表由工具按 `tokens.txt` 自动生成，不改模型目录 |
| `Runtime/llama.cpp-master.tar.gz` | 38,847,586 | `4b5a5fda135b1248` | 上游 llama.cpp master 源码（目标板构建 CUDA 版用） |

其余资产：`Models/tts/piper-zh_CN-huayan-medium/espeak-ng-data/`（音素化数据）、`Models/*/test_wavs/`（模型自检素材）、`Models/vlm/source.json`（下载来源记录）。

## 3. 运行器规范

- 目标板运行器**不使用** `Runtime/llama.cpp-windows-cpu/`（Windows CPU 版，仅是对齐原始资产包的留档）。
- 目标板用上游 llama.cpp master 自建 CUDA 版：源码来自 `Runtime/llama.cpp-master.tar.gz`，
  构建命令与已验证版本见 `Runtime/BUILD.md`；构建产物在目标板
  `third_party/llama.cpp/build/bin/`（`llama-server` 等）。
- 启动参数统一收在 `serve_llama.sh`（`--parallel 1 --batch-size 256 --ubatch-size 128` 为 4 GiB 内存上限下的定版参数），
  不要在命令行临时改参数后不落盘。

## 4. 部署与校验规范

1. 目标板模型目录：`/home/mylin/Desktop/Nuo/models/<同名子目录>`；
2. 同步：`tar` 管道上传 `NX-AI/Models/` 内容到目标机 `models/`（大文件单独传输）；
3. 校验：在目标板执行
   ```bash
   cd /home/mylin/Desktop/Nuo && sha256sum models/vlm/Qwen3.5-2B-Q4_K_M.gguf
   ```
   与 `manifest.json` 中记录一致才算部署成功；
4. 新增/替换模型时：记录来源与许可 → 放入对应子目录 → `node NX-AI/Tools/make_manifest.js`
   更新清单 → 按另行确认的部署范围校验 → 在 `Docs/报告/` 对应阶段记录版本与实测数字。

## 5. 工具规范

| 工具 | 用途 |
|---|---|
| `Tools/make_manifest.js` | 重新生成 `manifest.json`（sha256 清单） |
| `Tools/build_llama.sh` | 目标板构建 CUDA 版 llama.cpp（参数见 `Runtime/BUILD.md`） |
| `Tools/serve_llama.sh` | 启动 llama-server（定版参数，环境变量可覆盖端口/模型路径） |
| `Tools/mem_probe.py` | 整栈内存与单次调用时延探针（阶段 7 验收口径） |
| `Tools/chat_latency_probe.py` | 流式/非流式热聊首声对比 |
| `Tools/make_compare_fixtures.py` | 生成双图价格比较夹具 |
| `Tools/make_compare_audio_fixtures.py` | 生成固定问题音频夹具（ASR 自检后落盘） |
| `Tools/run_aieyes.sh` / `Tools/stop_aieyes.sh` | 目标板离机启动/停止（先服务端后交互服务，`setsid nohup`） |
| `Tools/soak_test.py` | 30 分钟持续检查（回放驱动真实栈，`--sink auto\|alsa\|null`） |
| `Tools/scene_smoke.py` | 场景模块真机冒烟（相机/OCR/VLM） |

说明：`Tools/`（仓库根）下的同名脚本是**部署副本**（目标板运行与 `tsync` 走这条路径）；
修改模型工具时先改本目录，再同步到根 `Tools/`，避免两处漂移。

## 6. 变更流程（防止“能跑但不可复核”）

1. 任何模型/运行器/参数变更都要有来源、版本号与许可记录；
2. 变更后必须重跑 `make_manifest.js` 并对比清单差异；
3. 目标板先校验 sha256，再跑 `tools/mem_probe.py` 与 `pytest -q tests/unit tests/integration`（原目标机目录）；
4. 实测数字写进 `Docs/报告/阶段N-*.md`，未测项如实列出，不写“应该可以”。
