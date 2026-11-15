# AI Eyes 测试台

> 本机目录已经整理为 `Testbench/`，操作、方案与进度文档已移至 `Docs/测试台/`。阅读[启动与关闭指南](../Docs/测试台/启动与关闭.md)及[执行方案](../Docs/测试台/执行方案.md)。
> 远程目标机仍使用原有小写路径；下文目标机绝对路径和远程命令保留原值。本地 `Scripts/` 只有已有的启停脚本，不包含菜单引用的 `serve_model.sh`、`stop_model.sh`。本次未验证模型、设备或远程启停。

本目录是 AI Eyes 业务链路的独立测试台，界面沿用已确认的 `index.html` 基线，通过本机浏览器访问目标机回环地址。测试台的全部自有文件（源码、前端、配置、脚本、测试、上传、日志、结果、PID、缓存）都在本目录内，停止服务后可整体移除，不影响正式项目。

- 不改动 `Src/`、`Config/target.json`、模型文件与现有 llama-server。
- 真实录音、拍摄、播报和整轮人工测试由用户手动启动；测试台自身不自动开启设备。
- 不保存任何 SSH 凭据；文档、脚本、日志、配置中均不出现密码。

## 目录

```
Testbench/
├── index.html          已确认界面（布局/配色/导航不变）
├── app.js              原生 JS 前端（无框架、无构建链）
├── server.py           HTTP 服务（仅监听 127.0.0.1）
├── supervisor.py       worker 子进程管理（PID/进程组/启动时间归属校验）
├── worker.py           业务运行时（按需创建，装配真实业务组件）
├── bridge.py           事件观测包装（bus/chat/tts/evidence/scene/provider）
├── media.py            CameraHub 单一采集线程、FrameStore、UploadStore
├── metrics.py          2s 资源采样（/proc、/sys）
├── debug.py            功能调试模块（固定白名单）
├── Config/bench.json   测试台固定配置
├── Scripts/start.sh    启动 HTTP 服务
├── Scripts/stop.sh     停止测试台自有进程
├── Tests/              软件自检（fake 组件/临时目录）
├── runs/<run_id>/      每轮 config.json / events.jsonl / metrics.jsonl / result.json / media/
├── uploads/            上传图片与 index.json
├── logs/               server 日志与 worker 日志（10 MiB 轮转 ×5）
├── .runtime/           server.pid、worker.json、缓存、pytest 缓存
├── README.md
└── 测试台.cmd          Windows 远程启动与关闭菜单
```

## 环境要求

- 目标机：项目根 `/home/mylin/Desktop/Nuo`，业务虚拟环境 `../.venv/bin/python`（Python 3.12），依赖 `PIL`、`numpy`、`sounddevice`、`requests`、`pytest`。
- 目标机正式配置 `config/target.json` 保持不变（本机对应 `Config/target.json`）；测试台只读取它生成每轮配置副本。
- 模型服务（llama-server，`http://127.0.0.1:8080/health`）由用户按项目原入口启动；测试台只做健康检查，不启动、不重启、不改参数。

## 启动

目标机：

```bash
bash /home/mylin/Desktop/Nuo/testbench/scripts/start.sh
```

成功时输出 `started: http://127.0.0.1:18780/ (pid ...)` 与日志路径。启动只创建 HTTP 服务，不加载模型、不开音频/相机、不启动综合测试。

本机通过 SSH 本地转发访问（页面必须使用 `127.0.0.1:18780` 或 `localhost:18780`，服务端 Host/Origin 校验只允许这两个值）：

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18780:127.0.0.1:18780 mylin@192.168.2.72
```

浏览器打开 <http://127.0.0.1:18780/>。若本机 18780 已被占用，请先释放该端口；**不要**改用 18781 等其它本地端口（Host/Origin 校验会拒绝，页面返回 403）。

## 停止

页面先停止正在运行的综合测试/预览，然后目标机：

```bash
bash /home/mylin/Desktop/Nuo/testbench/scripts/stop.sh
```

`stop.sh` 只结束经 PID + `/proc` 启动时间 + 命令行三重校验确认为测试台自有的 server/worker，等待退出并核对 18780 释放；不会触碰 llama-server 或其它进程。关闭 SSH 转发不等于停止 server；浏览器断开超过 15 秒时服务端会停止活动的测试/预览并记录 `client_disconnected`。

## 配置

`Config/bench.json` 固定初值：监听 `127.0.0.1:18780`、项目根 `/home/mylin/Desktop/Nuo`、目标机正式配置 `config/target.json`、Python `../.venv/bin/python`、状态轮询 500 ms、资源采样 2000 ms、预览 ≤5 fps、上传单文件 20 MiB / 像素 ≤2400 万、日志 10 MiB×5、事件缓冲 4000 条、心跳 5 s / 超时 15 s。

每轮运行会把目标机 `config/target.json` 复制到 `runs/<run_id>/config.json` 并重定向证据/日志/媒体/声纹样本路径；输入输出设备覆盖为系统默认（`null`），模型路径保持正式值。正式配置文件不被修改。

## 使用流程

1. 打开页面，查看服务状态卡（模型服务、交互服务、音频、相机）与资源/耗时面板。
2. 点“开启预览”才启动相机采集；右框为实时预览，左框只在综合测试/调试真正取帧后显示模型实际使用的画面。停止预览后若仍有任务使用相机则设备保持打开。
3. 点“开始”进入综合测试（需确认）；自然说话走正式语音链路，右侧显示识别文本，左侧显示真实回复与播放状态；点“停止”结束本轮。
4. “上传”标签的 A/B 两槽用于视觉/OCR/比较调试，不会自动推理，也不改变综合测试的相机输入。
5. “功能调试”页按固定模块表动态渲染输入，与综合测试互斥；缺失能力（如 contract 未启用、声纹未注册）只显示未接入及依据。

每轮结果在 `runs/<run_id>/`：`events.jsonl`（真实事件）、`metrics.jsonl`（2s 资源采样）、`result.json`（起止/状态/时延口径/未测项/人工确认待定）、`media/`（本轮实际模型输入图）。浏览器日志仅保留最近 1000 条，清空只清显示，不删除磁盘证据。

## 软件自检

目标机（fake 组件，不碰设备/模型）：

```bash
cd /home/mylin/Desktop/Nuo/testbench
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/mylin/Desktop/Nuo/src \
  ../.venv/bin/python -B -m pytest tests -q \
  -o cache_dir=/home/mylin/Desktop/Nuo/testbench/.runtime/pytest_cache
```

本机（bundled Python，Windows 下 /proc 相关用例自动 skip）：

先进入项目的 `Testbench/` 目录，再执行：

```bash
python -B -m unittest discover -s Tests -t .
```

覆盖：事件桥接与去重、HTTP/会话/游标/上传/跨站控制、worker 行协议与进程归属、相机中枢与帧存储、上传校验与 EXIF、资源采样、调试模块表与参数校验。

## 界面自检要点

- 1440 宽与窄屏各一次：侧栏折叠（点击/Enter/空格）、相机两框与上传两框同宽同高 16:9、图片 `object-fit: contain`、聊天左右气泡、正文可选、日志过滤/清空、上传/移除。
- 刷新后无假对话/假指标，不自动开相机；断线显示真实“未连接”。
- 不出现“演示/原型/示例”等字样。

## 边界与未实现

- `acoustic_first_audio_ms`（真实声学首声）未测量，界面显示“未测量”，不以模块耗时相加代替。
- 资源面板的 CPU 为所有核归一化的 0–100%；Orin 上 RSS 不等于完整内存占用，模型 RSS 仅在详细记录/tooltip 展示。
- GPU/温度仅在 sysfs 节点可读时显示，否则为 `null`，不使用 sudo、不以 CPU 顶替。
- `contract.enabled=false` 时合同调试显示未接入；声纹未注册时说话人调试显示未启用。调试替身测试必须标记。
- 真实设备联调、人工验收由用户按执行方案 §13C 手动启动。

## 移除

先执行 `Scripts/stop.sh` 并确认无测试台自有进程，需要保留结果时先另存 `runs/`，然后删除整个 `Testbench/` 目录即可。测试台不写入项目其它目录、不安装系统服务、不设置开机自启。
