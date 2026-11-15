# interaction.v0.1 合同层

本目录实现 AI_Eyes 交互侧（interaction）与主控（main_control）之间的
`interaction.v0.1` 合同：消息 schema 与严格校验、TCP 传输/消息关联、合同错误码、
可接入状态机的 handler 协议、测试夹具，以及 C01–C12 合同案例的软件回归。

设备能力（相机/ASR/TTS/模型）不在本层实现；合同层只负责“说清楚、不伪造、可恢复”。

## 模块

| 文件 | 职责 |
| --- | --- |
| `errors.py` | `schema_version`、角色/证据模式常量、13 个错误码、`ProtocolError`、错误载荷构造 |
| `schema.py` | `Message` 数据类、9 组消息构造器与严格校验、响应/错误响应构造、状态机语义辅助函数 |
| `transport.py` | 4 字节大端长度 + UTF-8 JSON 帧、请求-响应关联（`reply_to`/`request_id`）、超时与断线 |
| `server.py` | TCP 服务端（交互侧监听）、入站分发、handler 超时/异常映射、出站请求与事件、重连 epoch |
| `handlers.py` | `ContractHandler` 协议、`NullHandler`/`FakeHandler`、供状态机使用的 `InteractionContract` 门面 |
| `fixture_client.py` | 扮演 main_control 的测试夹具（请求、事件、事件等待、RTT 记录） |
| `cases.py` | C01–C12 合同案例软件回归；`python -m aieyes.contract.cases` 输出表格与 JSON |

## 消息覆盖（16 类）

请求均为 `interaction→main_control`，除 `capabilities.get`（双向）与
`interaction.status.get`、`output.cancel`（main_control→interaction）之外。

| 组 | 请求 | 响应/事件 | 方向 |
| --- | --- | --- | --- |
| capabilities | `capabilities.get` | `capabilities.result` | 双向 |
| frame | `frame.acquire` | `frame.result` | interaction→main_control |
| scene | `scene.query` | `scene.result` | interaction→main_control |
| control | `control.intent` | `control.result` | interaction→main_control |
| target | `target.observation` | `target.result` | interaction→main_control |
| guidance | — | `guidance.event`（事件） | main_control→interaction |
| playback | — | `playback.status`（事件） | interaction→main_control |
| output | `output.cancel` | `output.cancel.result` | main_control→interaction（或交互内部） |
| status | `interaction.status.get` | `interaction.status` | main_control→interaction |

公共字段：`schema_version/type/message_id/reply_to/request_id/session_id/
producer{role,instance_id}/clock_id/timestamp_mono_ms/evidence_mode/data`；
请求另有 `deadline_mono_ms`，响应另有 `status` 与 `error`。
角色：`main_control` / `interaction` / `test_fixture`；证据模式：`live/replay/synthetic`。
`live` 结果必须有 `clock_id`（首轮握手除外，由 `CLOCK_UNVERIFIED` 拒绝）。
错误码：`DEPENDENCY_UNAVAILABLE/NO_FRAME/STALE_FRAME/INVALID_FRAME_REF/
UNSUPPORTED_VERSION/CLOCK_UNVERIFIED/STALE_STATE/INVALID_TARGET/INVALID_PAYLOAD/
BUSY/DEADLINE_EXCEEDED/CANCELLED/OUTPUT_UNAVAILABLE`。

## 传输与帧格式（合同未固定，以下是本实现的选择）

合同只要求“本机请求/响应通道 + 常驻事件通道，端口/路径写入配置”，未规定方向与帧头。
本实现选择：

- 交互侧为 TCP 服务端（`ContractConfig.host:port` 为监听地址，`accept_timeout_s` 即服务端语义），
  主控作为客户端连接；单条双向多路复用连接可同时承载请求、响应与事件。
- 帧格式：`| 4 字节大端无符号长度 N | N 字节 UTF-8 JSON |`，`0 < N <= max_message_bytes`
  （默认 4 MiB）。长度非法、JSON 非法或超过上限时关闭该连接，不影响服务端继续接受新连接。
- 消息关联：响应 `reply_to == 请求.message_id` 且 `request_id` 相同才交付；迟到（超过
  请求 deadline 或已被替换）的响应仅记录诊断，不推进状态。请求缺省超时 T15 = 10 s
  （`ContractConfig.response_timeout_ms`，`TimersConfig.t15_control_ms`）。
- 重连：新连接替换旧连接；`epoch` 自增并触发 `on_connect/on_disconnect` 回调，用于重新握手。
  待处理请求在断线时立即以 `ConnectionClosed` 结束，不会悬挂到超时。
- 来源限制：服务端只接受 `main_control`（测试时可显式 `allow_test_fixture=True`）作为可信角色；
  方向错误的请求返回 `status=rejected, code=INVALID_PAYLOAD`。

## 与状态机对接

`handlers.ContractHandler` 是接入边界（async，入参与返回均为 schema 对象/dict）：

```python
class Handler:
    async def capabilities_get(self, request): ...
    async def frame_acquire(self, request): ...      # frame.result.data
    async def scene_query(self, request): ...        # scene.result.data
    async def control_intent(self, request): ...     # control.result.data
    async def target_observation(self, request): ...
    async def output_cancel(self, request): ...      # output.cancel.result.data
    async def interaction_status_get(self, request): ...
    async def guidance_event(self, event): ...       # 事件，无响应
```

返回 `Mapping`（按 ok 组装响应）或 `handlers.HandlerReply(status/data/error)`；
失败抛 `handlers.HandlerFailure(code, message)`。服务端负责信封字段（`reply_to/request_id/
session_id`）、超时（`DEADLINE_EXCEEDED`）、异常映射（未知异常→`DEPENDENCY_UNAVAILABLE`）
与错误状态映射，处理函数不必手写信封。

状态机出站调用使用门面（自动填充 producer/clock/monotonic 时间戳/证据模式/T15 deadline）：

```python
contract = InteractionContract(server)
caps = await contract.query_capabilities()
frame = await contract.acquire_frame(purpose="vqa")
scene = await contract.query_scene(track_id="track-1")
res = await contract.send_control_intent(action="start_guidance", utterance_id="u1",
                                        final_transcript="开始引导")
await contract.publish_playback_status(state="started", source_event_id="evt-1",
                                       segment_index=0, start_basis="device_write")
```

入站事件（`guidance.event`）由服务端直接调用 handler，不走 `EventBus`，避免双重消费；
如需把事件转发给状态机，在 handler 内调用自己的状态机入口或 `EventBus.post`。

语义辅助（`schema.py`）：`scene_measurement` / `measurement_distance_m`（无效或过期一律
返回 `None`，绝不补 0）、`guidance_is_fresh`、`GuidanceDeduper`（重放去重/过期/被撤销）、
`is_completed_ok`（仅 completed 才算完成）、`SequenceGuard`（序号回退→`STALE_STATE`）、
`handshake_fingerprint`（instance_id/boot_id/clock 变化→重新握手）。

## C01–C12 合同案例（软件回归，2026-11-11 目标板实测）

命令：`. .venv/bin/activate && PYTHONPATH=src python -m aieyes.contract.cases`

| 案例 | 结果 | 合同层检查（摘要） |
| --- | --- | --- |
| C01 主控缺失/能力缺失 | passed | 无连接时返回 `NotConnected` 不伪造；能力查询诚实报告缺失；方向守卫 |
| C02 重启/时钟不一致 | passed | epoch 递增、断线回调、指纹变化、序号回退 `STALE_STATE` |
| C03 同一张照片追问 | passed | 同一 `frame_id/captured_mono_ms/sha256` 可被重复引用 |
| C04 新帧/翻转/A-B | passed | 新帧身份变化、派生帧 `parent_frame_id`+3×3 变换、旧帧仍可寻址 |
| C05 过期/缺失/无效测量 | passed | 无效测量 `distance_m=null`（非 0），助手返回 `None` |
| C06 临时 ASR 倾向 vs 最终意图 | passed | 仅最终意图触发 `control.intent`；动作白名单；未完成不得报完成 |
| C07 重复 control 请求 | passed | 同 `request_id` 只执行一次，重复 `message_id` 不重复执行 |
| C08 夜间紧急事件 | passed | `guidance.event` 接收、`playback.status` queued/started 反馈、去重 |
| C09 打断后旧 generation | passed | `output.cancel` scope/代际校验、结果布尔上报 |
| C10 重连重放历史事件 | passed | 过期事件 `dropped_stale`、重复事件去重、新事件可播 |
| C11 输出设备断连 | passed | `OUTPUT_UNAVAILABLE` 失败上报，绝不伪报 completed |
| C12 模型/OCR 编造 | passed | 交互侧不能伪造 `guidance.event`；无有效测量不得声称米数/通行 |

单测：`. .venv/bin/activate && python -m pytest -q Tests/Unit/contract` →
**55 passed**（目标板 2026-11-11 实测；本地无 Python，全部测试在目标板运行）。

案例中标注 `deferred` 的语义（如“状态机应如何使用这些合同事实”）由交互状态机实现，
合同层只提供并验证合同事实。

## 已知限制与待确认

- **控制结果查询缺口**：合同要求按原 `request_id` 查询 `control.result`，但未固定消息名。
  本层实现业务幂等（同 `request_id` 只执行一次）与 `is_completed_ok` 判定，未新增查询消息；
  状态机接入时需与主控确认只读查询绑定，否则超时只能回复“未确认”。
- **身份鉴权**：单机 TCP 只能依据 `producer.role` 自述。真实主控适配器应绑定本机通道/进程
  身份，`test_fixture` 仅在显式测试入口开放；不能把自述角色当作安全认证。
- 单连接承载全部通道；如后续需要“紧急事件优先”，可在同一语义下增加独立连接。
- 真实设备能力（相机/ASR/TTS/模型）由 audio/vision 模块与状态机接入，本层 `NullHandler`
  在无设备时诚实降级。

## 测试命令（目标板）

```bash
cd /home/mylin/Desktop/Nuo && . .venv/bin/activate
python -m pytest -q Tests/Unit/contract
PYTHONPATH=src python -m aieyes.contract.cases
```
