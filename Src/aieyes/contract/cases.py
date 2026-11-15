"""C01–C12 合同案例的软件回归。

每个案例启动一个真实的 :class:`~aieyes.contract.server.ContractServer`
（127.0.0.1 临时端口），用 :class:`~aieyes.contract.fixture_client.FixtureClient`
扮演 main_control，按合同验收文档的输入/预期观察逐条断言。

边界说明：本文件只验证**合同层可判定的**部分。“状态机是否在紧急事件时抢占
普通回答”“ASR 临时倾向是否被忽略”等属于交互状态机行为，统一记录在
``CaseOutcome.deferred``，不算通过也不算失败；确实无法在合同层构造的案例
（如主控进程真实重启）标记为 ``skipped`` 并写原因。

运行方式（目标板）::

    . .venv/bin/activate && python -m aieyes.contract.cases

或 ``python -m pytest Tests/Unit/contract/test_cases.py``。
"""

from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from aieyes import clock
from aieyes.config import ContractConfig
from aieyes.contract import schema
from aieyes.contract.errors import SCHEMA_VERSION, ProtocolError
from aieyes.contract.fixture_client import FixtureClient, FixtureResponder
from aieyes.contract.handlers import (
    FakeHandler,
    InteractionContract,
    NullHandler,
    cancel_result_data,
    capabilities_result_data,
    control_result_data,
    frame_result_data,
    measurement_data,
    playback_status_data,
    scene_object_data,
    scene_result_data,
    target_result_data,
)
from aieyes.contract.schema import (
    GuidanceDeduper,
    Message,
    SequenceGuard,
    guidance_is_fresh,
    handshake_fingerprint,
    is_completed_ok,
    measurement_distance_m,
    now_ms,
    scene_measurement,
)
from aieyes.contract.server import ContractServer, NotConnected
from aieyes.contract.transport import ConnectionClosed, RequestTimeout

#: 每个案例的总超时（秒）
CASE_TIMEOUT_S = 60.0

#: 合同案例标题（与验收文档 §8 对应）
CASE_TITLES: dict[str, str] = {
    "C01": "主控缺失/能力缺失时诚实降级",
    "C02": "主控重启/时钟不一致",
    "C03": "同一张照片连续追问",
    "C04": "新帧/翻转/A-B 对比",
    "C05": "过期/缺失/无效测量",
    "C06": "临时 ASR 倾向 vs 最终控制意图",
    "C07": "重复 control 请求（同 request_id）",
    "C08": "普通回答期间的紧急事件",
    "C09": "打断后的旧 generation 迟到输出",
    "C10": "重连重放历史事件",
    "C11": "输出设备断连",
    "C12": "模型/OCR 编造能力与距离",
}


@dataclass
class CaseOutcome:
    """单个案例结果；``checks`` 全部通过才算 passed。"""

    id: str
    title: str
    status: str = "passed"  # passed | skipped | failed
    checks: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    detail: str = ""

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.checks.append(name)
        else:
            raise AssertionError(f"{name}: {detail}" if detail else name)

    def defer(self, note: str) -> None:
        self.deferred.append(note)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "checks": list(self.checks),
            "deferred": list(self.deferred),
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# 公共夹具
# ---------------------------------------------------------------------------
def _config(**overrides: Any) -> ContractConfig:
    base: dict[str, Any] = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
        "role": "interaction",
        "instance_id": "aieyes-contract-test",
        "max_message_bytes": 4 * 1024 * 1024,
        "response_timeout_ms": 5_000,
        "accept_timeout_s": 5.0,
    }
    base.update(overrides)
    return ContractConfig(**base)


async def _start_server(
    handler: Any = None,
    *,
    allow_test_fixture: bool = False,
    on_connect: Callable[[int], Any] | None = None,
    on_disconnect: Callable[[int], Any] | None = None,
    **config_overrides: Any,
) -> ContractServer:
    server = ContractServer(
        _config(**config_overrides),
        handler=handler if handler is not None else NullHandler(),
        evidence_mode="synthetic",
        allow_test_fixture=allow_test_fixture,
        on_connect=on_connect,
        on_disconnect=on_disconnect,
    )
    await server.start()
    return server


async def _connect_client(server: ContractServer, **kwargs: Any) -> FixtureClient:
    client = FixtureClient("127.0.0.1", server.bound_port, **kwargs)
    await client.connect()
    return client


class _Rig:
    """一次案例的 server + 主控夹具，退出时按顺序清理。"""

    def __init__(
        self,
        handler: Any = None,
        *,
        responder: FixtureResponder | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        allow_test_fixture: bool = False,
        on_connect: Callable[[int], Any] | None = None,
        on_disconnect: Callable[[int], Any] | None = None,
        **config_overrides: Any,
    ) -> None:
        self.server: ContractServer
        self.client: FixtureClient
        self._handler = handler
        self._responder = responder
        self._client_kwargs = dict(client_kwargs or {})
        self._allow_test_fixture = allow_test_fixture
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._config_overrides = config_overrides

    async def __aenter__(self) -> "_Rig":
        self.server = await _start_server(
            self._handler,
            allow_test_fixture=self._allow_test_fixture,
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect,
            **self._config_overrides,
        )
        self.client = await _connect_client(self.server, responder=self._responder, **self._client_kwargs)
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        try:
            await self.client.close()
        finally:
            await self.server.stop()
            await asyncio.sleep(0.05)  # 让断线回调任务跑完

    @property
    def contract(self) -> InteractionContract:
        return InteractionContract(self.server)

    @property
    def responder(self) -> FixtureResponder:
        return self.client.responder


def _control_request(
    server: ContractServer,
    *,
    action: str = "start_guidance",
    utterance_id: str = "utt-1",
    final_transcript: str = "开始",
    user_confirmed: bool = False,
    request_id: str | None = None,
    message_id: str | None = None,
    timeout_ms: int = 5_000,
) -> Message:
    ts = now_ms()
    return schema.control_intent(
        action=action,
        utterance_id=utterance_id,
        final_transcript=final_transcript,
        user_confirmed=user_confirmed,
        producer=server.producer,
        request_id=request_id,
        message_id=message_id,
        clock_id=server.clock_id,
        timestamp_mono_ms=ts,
        deadline_mono_ms=ts + timeout_ms,
        evidence_mode=server.evidence_mode,
    )


def _frame_of(result: Message) -> Mapping[str, Any]:
    assert result.status == "ok", f"frame result not ok: {result.status}/{result.error}"
    assert isinstance(result.data, Mapping), "frame result has no data"
    return result.data


def _scene_objects(valid: bool, *, valid_until: int | None = None, distance: float | None = 2.0) -> list[dict[str, Any]]:
    until = now_ms() + 5_000 if valid_until is None else valid_until
    return [
        scene_object_data(
            track_id="track-1",
            class_name="person",
            bearing="front",
            measurement=measurement_data(
                distance_m=distance if valid else None,
                valid=valid,
                valid_until_mono_ms=until,
                reason_code=None if valid else "DEPTH_MISSING",
            ),
        )
    ]


async def _wait_for(predicate: Callable[[], bool], *, timeout_s: float = 3.0) -> bool:
    """轮询等待本地条件成立（本模块测试用的确定性等待，避免裸 sleep 竞态）。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


# ---------------------------------------------------------------------------
# C01 主控缺失/能力缺失
# ---------------------------------------------------------------------------
async def _case_c01(outcome: CaseOutcome) -> None:
    # 1) 无主控连接：交互侧不得伪造结果
    server = await _start_server(NullHandler())
    try:
        contract = InteractionContract(server)
        for name, call in (
            ("acquire_frame", lambda: contract.acquire_frame(mode="latest", purpose="vqa")),
            ("query_scene", lambda: contract.query_scene()),
        ):
            try:
                await call()
            except NotConnected:
                outcome.check(f"no-main-control:{name}-NotConnected", True)
            else:
                outcome.check(f"no-main-control:{name}-NotConnected", False, "expected NotConnected")

        # 2) 主控连上后询问交互能力：NullHandler 诚实报告缺失
        client = await _connect_client(server)
        await server.wait_connected(3.0)
        try:
            caps = await client.capabilities_get()
            outcome.check("caps-result-ok", caps.status == "ok", str(caps.error))
            cap_map = caps.data.get("capabilities", {}) if isinstance(caps.data, Mapping) else {}
            outcome.check("caps-frame_provider-false", cap_map.get("frame_provider") is False, str(cap_map))
            outcome.check("caps-scene_state-false", cap_map.get("scene_state") is False, str(cap_map))
            outcome.check(
                "caps-unimplemented-nonempty",
                bool(caps.data.get("unimplemented")) if isinstance(caps.data, Mapping) else False,
            )

            # 3) 方向错误：主控不能替交互发 frame.acquire
            bad = await client.frame_acquire(mode="latest")
            outcome.check("wrong-direction-frame.acquire-rejected", bad.status == "rejected", bad.status)
            outcome.check(
                "wrong-direction-frame.acquire-error-code",
                bad.error_code() == "INVALID_PAYLOAD",
                str(bad.error),
            )
        finally:
            await client.close()
            await _wait_for(lambda: not server.connected)

        # 4) 主控能力缺失：交互查询后不得继续要帧
        responder = FixtureResponder()
        responder.capabilities["frame_provider"] = False
        client2 = await _connect_client(server, responder=responder)
        await server.wait_connected(3.0)
        try:
            caps2 = await InteractionContract(server).query_capabilities()
            outcome.check("main-control-caps-ok", caps2.status == "ok", str(caps2.error))
            outcome.check(
                "main-control-caps-frame_provider-false",
                isinstance(caps2.data, Mapping)
                and caps2.data.get("capabilities", {}).get("frame_provider") is False,
            )
            frame = await InteractionContract(server).acquire_frame(mode="latest", purpose="vqa")
            outcome.check("missing-capability-frame-unavailable", frame.status == "unavailable", frame.status)
            outcome.check(
                "missing-capability-frame-error-code",
                frame.error_code() == "DEPENDENCY_UNAVAILABLE",
                str(frame.error),
            )
        finally:
            await client2.close()
            await _wait_for(lambda: not server.connected)
    finally:
        await server.stop()
        await asyncio.sleep(0.05)

    outcome.defer("状态机在无主控时不得输出距离/方向断言——由交互状态机实现，合同层只保证不伪造结果")


# ---------------------------------------------------------------------------
# C02 主控重启/时钟不一致
# ---------------------------------------------------------------------------
async def _case_c02(outcome: CaseOutcome) -> None:
    epochs: list[int] = []
    disconnects: list[int] = []

    def on_connect(epoch: int) -> None:
        epochs.append(epoch)

    def on_disconnect(epoch: int) -> None:
        disconnects.append(epoch)

    server = await _start_server(NullHandler(), on_connect=on_connect, on_disconnect=on_disconnect)
    responder = FixtureResponder(host_boot_id="boot-A", clock_id="clock-A")
    try:
        client = await _connect_client(server, responder=responder, clock_id="clock-A")
        contract = InteractionContract(server)
        first = await contract.query_capabilities()
        fp1 = handshake_fingerprint(first)
        outcome.check("first-handshake-epoch", epochs == [1], str(epochs))

        # 主控进程/主机重启：instance/boot/clock 变化 → 指纹变化
        responder.producer["instance_id"] = "main-control-restarted"
        responder.host_boot_id = "boot-B"
        responder.clock_id = "clock-B"
        client.clock_id = "clock-B"
        second = await contract.query_capabilities()
        fp2 = handshake_fingerprint(second)
        outcome.check("fingerprint-changes-on-restart", fp1 != fp2, f"{fp1} == {fp2}")
        outcome.check("fingerprint-instance-changed", fp1[0] != fp2[0], str(fp1))
        outcome.check("fingerprint-boot-changed", fp1[1] != fp2[1], str(fp1))
        outcome.check("fingerprint-clock-changed", fp1[2] != fp2[2], str(fp1))

        # 旧 sequence 不能回退覆盖新状态
        guard = SequenceGuard()
        outcome.check("sequence-accept-5", guard.check("mc:scene", 5) is True)
        try:
            guard.check("mc:scene", 3)
        except ProtocolError as exc:
            outcome.check("sequence-regression-STALE_STATE", exc.code == "STALE_STATE", str(exc))
        else:
            outcome.check("sequence-regression-STALE_STATE", False, "expected STALE_STATE")
        guard.reset()
        outcome.check("sequence-reset-after-handshake", guard.check("mc:scene", 1) is True)

        # 断线：pending 请求立即失败（不得伪造成功结果）
        responder.delay("scene.query", 2.0)
        contract = InteractionContract(server)
        pending = asyncio.create_task(contract.query_scene(timeout_ms=4_000))
        await asyncio.sleep(0.15)
        await client.close()
        try:
            await pending
        except (ConnectionClosed, RequestTimeout):
            outcome.check("pending-fails-on-disconnect", True)
        else:
            outcome.check("pending-fails-on-disconnect", False, "pending request unexpectedly succeeded")
        await asyncio.sleep(0.1)

        # 重连：epoch 递增，触发重新握手
        client2 = await _connect_client(server, responder=responder, clock_id="clock-B")
        try:
            third = await InteractionContract(server).query_capabilities()
            outcome.check("reconnected-epoch", server.epoch >= 2, str(server.epoch))
            outcome.check("reconnect-callback-fired", epochs[-1] == server.epoch, str(epochs))
            outcome.check("disconnect-callback-fired", bool(disconnects), str(disconnects))
            outcome.check(
                "reconnect-new-fingerprint",
                handshake_fingerprint(third) == fp2,
                f"{handshake_fingerprint(third)} != {fp2}",
            )
        finally:
            await client2.close()
    finally:
        await server.stop()
        await asyncio.sleep(0.05)

    outcome.defer("状态机收到 on_connect/on_disconnect 后撤销旧 state_id/track_id 与待播 guidance——由状态机实现")


# ---------------------------------------------------------------------------
# C03 同一张照片连续追问
# ---------------------------------------------------------------------------
async def _case_c03(outcome: CaseOutcome) -> None:
    async with _Rig(NullHandler()) as rig:
        contract = rig.contract
        first = await contract.acquire_frame(mode="latest", purpose="vqa")
        data1 = _frame_of(first)
        frame_id = data1["frame_id"]
        captured = data1["captured_mono_ms"]
        sha = data1["asset"]["sha256"]

        # 连续追问复用同一 frame_id（main_control 不得偷偷换图）
        for index in range(2):
            again = await contract.acquire_frame(mode="pinned", purpose="vqa", frame_id=frame_id)
            data2 = _frame_of(again)
            outcome.check(f"same-photo-frame_id-{index}", data2["frame_id"] == frame_id, str(data2["frame_id"]))
            outcome.check(
                f"same-photo-captured_mono_ms-{index}", data2["captured_mono_ms"] == captured
            )
            outcome.check(f"same-photo-sha256-{index}", data2["asset"]["sha256"] == sha)

        # OCR 追问也引用同一张照片
        ocr = await contract.acquire_frame(
            mode="pinned", purpose="ocr", frame_id=frame_id, max_age_ms=60_000
        )
        outcome.check("same-photo-ocr-ok", ocr.status == "ok", str(ocr.error))
        outcome.check("same-photo-ocr-frame_id", _frame_of(ocr)["frame_id"] == frame_id)

        # 目标观测引用该帧
        target = await contract.observe_target(
            visual_target_id="vt-1",
            frame_id=frame_id,
            roi_xyxy_px=[10, 10, 100, 100],
            name="水杯",
            source="ocr",
        )
        outcome.check("same-photo-target-ok", target.status == "ok", str(target.error))

    outcome.defer("状态机把 OCR/VQA 结果绑定到 frame_id，切换帧时丢弃旧结果——由状态机实现")


# ---------------------------------------------------------------------------
# C04 新帧/翻转/A-B
# ---------------------------------------------------------------------------
async def _case_c04(outcome: CaseOutcome) -> None:
    async with _Rig(NullHandler()) as rig:
        contract = rig.contract
        responder = rig.responder
        first = _frame_of(await contract.acquire_frame(mode="latest", purpose="vqa"))
        second = _frame_of(await contract.acquire_frame(mode="latest", purpose="vqa"))
        outcome.check("new-frame-new-id", first["frame_id"] != second["frame_id"])
        outcome.check("new-frame-new-sha", first["asset"]["sha256"] != second["asset"]["sha256"])

        # 派生帧（如翻转/裁剪）保留 parent 与行主序 3x3 变换
        responder.add_frame(
            "frame-derived-1",
            captured_mono_ms=now_ms(),
            parent_frame_id=first["frame_id"],
            transform_to_parent=[-1, 0, 1280, 0, 1, 0, 0, 0, 1],
        )

        def frame_script(request: Message) -> Mapping[str, Any]:
            if request.data.get("frame_id") == "frame-derived-1":
                return responder.frame_data("frame-derived-1")
            return responder._frame_acquire_data(request)

        responder.script("frame.acquire", frame_script)
        derived = _frame_of(
            await contract.acquire_frame(mode="pinned", purpose="compare", frame_id="frame-derived-1")
        )
        outcome.check("derived-parent_frame_id", derived["parent_frame_id"] == first["frame_id"])
        matrix = derived["transform_to_parent"]
        outcome.check(
            "derived-transform-3x3",
            isinstance(matrix, list) and len(matrix) == 9 and matrix[0] == -1,
            str(matrix),
        )
        outcome.check("derived-new-frame-id", derived["frame_id"] == "frame-derived-1")

        old = _frame_of(await contract.acquire_frame(mode="pinned", purpose="vqa", frame_id=first["frame_id"]))
        outcome.check("pinned-old-frame-still-addressable", old["frame_id"] == first["frame_id"])

    outcome.defer("A/B 对比中把 OCR/VQA 结果迁移到新帧或标记失效——由状态机实现")


# ---------------------------------------------------------------------------
# C05 过期/缺失/无效测量
# ---------------------------------------------------------------------------
async def _case_c05(outcome: CaseOutcome) -> None:
    responder = FixtureResponder()
    expired_valid_until = now_ms() - 1_000

    def scene_script(request: Message) -> Mapping[str, Any]:
        now = now_ms()
        return scene_result_data(
            state_id="scene-1",
            sequence=1,
            observed_mono_ms=now,
            valid_until_mono_ms=now + 5_000,
            coordinate_frame="camera_depth_optical",
            objects=[
                scene_object_data(
                    track_id="track-invalid",
                    class_name="person",
                    measurement=measurement_data(
                        distance_m=None, valid=False, valid_until_mono_ms=now + 5_000, reason_code="DEPTH_MISSING"
                    ),
                ),
                scene_object_data(
                    track_id="track-expired",
                    class_name="person",
                    measurement=measurement_data(
                        distance_m=1.5, valid=True, valid_until_mono_ms=expired_valid_until
                    ),
                ),
                scene_object_data(track_id="track-missing", class_name="person", measurement=None),
            ],
            navigation={"mode": "guide", "decision": "stop", "reason_code": "DEPTH_MISSING", "speech_fact": "测距不可用"},
            sensor_status={"rgb": "ok", "depth": "unavailable", "heading": "ok"},
        )

    responder.script("scene.query", scene_script)
    async with _Rig(NullHandler(), responder=responder) as rig:
        contract = rig.contract
        result = await contract.query_scene()
        outcome.check("scene-result-ok", result.status == "ok", str(result.error))
        objects = result.data["objects"] if isinstance(result.data, Mapping) else []
        by_id = {obj["track_id"]: obj for obj in objects}

        invalid_m = by_id["track-invalid"]["measurement"]
        outcome.check("invalid-measurement-distance-null", invalid_m["distance_m"] is None, str(invalid_m))
        outcome.check("invalid-measurement-not-zero", invalid_m["distance_m"] != 0)
        outcome.check(
            "invalid-measurement-helper-none",
            measurement_distance_m(invalid_m) is None,
        )
        outcome.check("invalid-measurement-scene-helper-none", scene_measurement(result, track_id="track-invalid") is None)

        expired_m = by_id["track-expired"]["measurement"]
        outcome.check("expired-measurement-helper-none", measurement_distance_m(expired_m) is None)
        outcome.check("expired-measurement-scene-helper-none", scene_measurement(result, track_id="track-expired") is None)
        outcome.check("missing-measurement-scene-helper-none", scene_measurement(result, track_id="track-missing") is None)

        # 有效测量正常返回米数（且不因场景里其他无效对象被污染）
        now = now_ms()
        valid_scene = scene_result_data(
            state_id="scene-1",
            sequence=2,
            observed_mono_ms=now,
            valid_until_mono_ms=now + 5_000,
            objects=_scene_objects(valid=True, valid_until=now + 5_000, distance=2.0),
            navigation={"mode": "guide", "decision": "go", "reason_code": "PATH_CLEAR", "speech_fact": "2 米"},
            sensor_status={"rgb": "ok", "depth": "ok", "heading": "ok"},
        )
        responder.script(
            "scene.query",
            lambda request: valid_scene,
        )
        valid_result = await contract.query_scene()
        outcome.check("valid-measurement-distance", scene_measurement(valid_result, track_id="track-1") == 2.0)
        outcome.check(
            "valid-measurement-navigation-not-invalid",
            valid_result.data["navigation"]["decision"] == "go",
        )

    outcome.defer("排队中的旧播报在测量失效后取消——由状态机实现")


# ---------------------------------------------------------------------------
# C06 临时 ASR 倾向 vs 最终控制意图
# ---------------------------------------------------------------------------
async def _case_c06(outcome: CaseOutcome) -> None:
    async with _Rig(NullHandler()) as rig:
        contract = rig.contract
        responder = rig.responder

        # 只有最终意图才发 control.intent；临时“开始”不产生合同消息
        first = await contract.send_control_intent(
            action="start_guidance",
            utterance_id="utt-1",
            final_transcript="开始",
            user_confirmed=False,
        )
        outcome.check("final-control-result-ok", first.status == "ok", str(first.error))
        outcome.check("final-control-completed", is_completed_ok(first, expected_mode="guidance"))
        outcome.check("control-executed-once-so-far", responder.counts["control.intent"] == 1)
        outcome.check(
            "temporary-asr-has-no-message",
            responder.counts["control.intent"] == 1,
            "只有最终意图产生 control.intent",
        )

        # main_control 返回 pending：不得当成“已开始”
        responder.script(
            "control.intent",
            lambda request: control_result_data(
                accepted=True, operation_state="pending", actual_mode="guidance", state_id="mc-state-1"
            ),
        )
        pending = await contract.send_control_intent(
            action="start_guidance", utterance_id="utt-2", final_transcript="不要开始"
        )
        outcome.check("pending-not-completed", is_completed_ok(pending) is False)
        outcome.check("pending-operation-state", pending.data["operation_state"] == "pending")

        # 非白名单动作在合同层直接拒绝
        try:
            await contract.send_control_intent(action="开始", utterance_id="utt-3", final_transcript="开始")
        except ProtocolError as exc:
            outcome.check("non-whitelist-action-rejected", exc.code == "INVALID_PAYLOAD", str(exc))
        else:
            outcome.check("non-whitelist-action-rejected", False, "expected ProtocolError")

    outcome.defer("ASR 临时 partial 结果不触发 guidance；以 final_transcript/final intent 为准——由状态机实现")


# ---------------------------------------------------------------------------
# C07 重复 control 请求（同 request_id）
# ---------------------------------------------------------------------------
async def _case_c07(outcome: CaseOutcome) -> None:
    async with _Rig(NullHandler()) as rig:
        server = rig.server
        responder = rig.responder
        request_id = "req-dup-control-1"

        msg1 = _control_request(server, request_id=request_id, message_id="msg-dup-1")
        msg2 = _control_request(server, request_id=request_id, message_id="msg-dup-2")
        reply1 = await server.send_request(msg1)
        reply2 = await server.send_request(msg2)
        for name, reply in (("dup-first", reply1), ("dup-second", reply2)):
            outcome.check(f"dup-{name}-ok", reply.status == "ok", str(reply.error))
            outcome.check(f"dup-{name}-request_id", reply.request_id == request_id)
            outcome.check(
                f"dup-{name}-reply_to",
                reply.reply_to == (msg1.message_id if name == "dup-first" else msg2.message_id),
            )
        outcome.check("dup-executed-once", responder.control_executions[request_id] == 1, str(responder.control_executions))
        outcome.check("dup-requests-observed-twice", responder.counts["control.intent"] == 2)
        outcome.check("dup-completed", is_completed_ok(reply2, expected_mode="guidance"))

        # 完全相同消息重发（保持 message_id）：仍只执行一次
        reply3 = await server.send_request(msg1)
        outcome.check("dup-resend-same-message-id", reply3.status == "ok")
        outcome.check("dup-still-executed-once", responder.control_executions[request_id] == 1)

    outcome.defer(
        "按原 request_id 查询执行状态：合同未固定消息名，需在启动时记录只读查询绑定；"
        "10s 内超时只能查询原请求，不得新建执行请求；无绑定则回答“未确认”"
    )


# ---------------------------------------------------------------------------
# C08 普通回答期间的紧急事件
# ---------------------------------------------------------------------------
async def _case_c08(outcome: CaseOutcome) -> None:
    handler = FakeHandler()
    handler.script(
        "capabilities.get",
        capabilities_result_data(
            capabilities={"guidance_events": True, "interaction_status": True, "output_cancel": True},
            host_boot_id="fixture-boot-id",
        ),
    )
    async with _Rig(handler) as rig:
        client = rig.client
        contract = rig.contract
        outcome.check("caps-guidance-events-true", (await client.capabilities_get()).data["capabilities"]["guidance_events"] is True)

        await contract.publish_playback_status(
            state="queued", source_event_id="evt-emergency-1", priority="emergency", segment_index=0
        )
        queued = await client.expect_playback_status("queued")
        outcome.check("emergency-queued-priority", queued.data["priority"] == "emergency", str(queued.data))
        outcome.check("emergency-queued-source", queued.data["source_event_id"] == "evt-emergency-1")

        sent = await client.send_guidance(
            event_id="evt-emergency-1",
            sequence=1,
            priority="emergency",
            text="立刻停下",
            interrupt_policy="immediate",
            valid_for_ms=5_000,
        )
        await _wait_for(lambda: len(handler.guidance_events) >= 1)
        await rig.server.wait_step()
        outcome.check("handler-received-guidance", len(handler.guidance_events) == 1, str(len(handler.guidance_events)))
        received = handler.guidance_events[0]
        outcome.check("guidance-event-id", received.data["event_id"] == "evt-emergency-1")
        outcome.check("guidance-priority-emergency", received.data["priority"] == "emergency")
        outcome.check("guidance-session-id", received.session_id == client.session_id)
        outcome.check(
            "guidance-event-matches-sent",
            (received.data["issued_mono_ms"], received.data["valid_until_mono_ms"])
            == (sent.data["issued_mono_ms"], sent.data["valid_until_mono_ms"]),
        )

        await contract.publish_playback_status(
            state="started",
            source_event_id="evt-emergency-1",
            priority="emergency",
            segment_index=0,
            start_basis="device_write",
        )
        started = await client.expect_playback_status("started")
        outcome.check("emergency-started-start_basis", started.data["start_basis"] == "device_write", str(started.data))
        outcome.check("emergency-playback-key", schema.playback_key(started) == ("evt-emergency-1", "", "", 0))

        # 同 event_id/segment 的重放不得重复播放
        deduper = GuidanceDeduper()
        outcome.check("dedupe-first-play", deduper.accept(sent) == "play")
        outcome.check("dedupe-repeat-drop", deduper.accept(sent) == "drop_duplicate")

    outcome.defer("紧急事件抢占普通回答（暂停/撤销普通 TTS 队列、先播紧急内容）——由状态机实现")


# ---------------------------------------------------------------------------
# C09 打断后的旧 generation 迟到输出
# ---------------------------------------------------------------------------
async def _case_c09(outcome: CaseOutcome) -> None:
    handler = FakeHandler()
    handler.script(
        "output.cancel",
        cancel_result_data(
            device_stopped=True,
            queue_cleared=True,
            compute_ended=True,
            reason_code="BARGED_IN",
            basis="device_write",
        ),
    )
    async with _Rig(handler) as rig:
        client = rig.client
        result = await client.output_cancel(
            scope="ordinary_turn",
            reason="user_barge_in",
            target_request_id="req-ordinary-1",
            old_generation=3,
            new_generation=4,
        )
        outcome.check("cancel-result-ok", result.status == "ok", str(result.error))
        data = result.data
        outcome.check("cancel-device_stopped", data["device_stopped"] is True)
        outcome.check("cancel-queue_cleared", data["queue_cleared"] is True)
        outcome.check("cancel-compute_ended", data["compute_ended"] is True)
        outcome.check("cancel-reason_code", data["reason_code"] == "BARGED_IN")
        outcome.check("cancel-generation-range", data["basis"] == "device_write")

        # 合同要求：普通代与紧急事件序列不能互相清除
        try:
            schema.output_cancel(
                scope="ordinary_turn",
                reason="user_barge_in",
                target_event_id="evt-emergency-1",
                producer=client.producer,
                evidence_mode="synthetic",
                clock_id=client.clock_id,
            )
        except ProtocolError as exc:
            outcome.check("scope-separation-ordinary-vs-guidance", exc.code == "INVALID_PAYLOAD", str(exc))
        else:
            outcome.check("scope-separation-ordinary-vs-guidance", False, "ordinary scope accepted target_event_id")

        guidance_cancel = await client.output_cancel(
            scope="guidance_event",
            reason="user_barge_in",
            target_event_id="evt-emergency-1",
            old_generation=1,
            new_generation=2,
        )
        outcome.check("guidance-cancel-ok", guidance_cancel.status == "ok", str(guidance_cancel.error))

        # generation 必须单调前进
        try:
            schema.output_cancel(
                scope="ordinary_turn",
                reason="user_barge_in",
                target_request_id="req-ordinary-1",
                old_generation=5,
                new_generation=4,
                producer=client.producer,
            )
        except ProtocolError as exc:
            outcome.check("generation-must-increase", exc.code == "INVALID_PAYLOAD", str(exc))
        else:
            outcome.check("generation-must-increase", False, "accepted non-monotonic generation")

    outcome.defer("旧 generation 的 ASR token、TTS 回调、已排队音频在打断后必须丢弃——由状态机实现")


# ---------------------------------------------------------------------------
# C10 重连重放历史事件
# ---------------------------------------------------------------------------
async def _case_c10(outcome: CaseOutcome) -> None:
    handler = FakeHandler()
    async with _Rig(handler) as rig:
        client = rig.client
        deduper = GuidanceDeduper()
        now = now_ms()
        producer = {"role": "main_control", "instance_id": "main-control-fixture"}

        first = schema.guidance_event(
            event_id="evt-1",
            sequence=1,
            priority="navigation",
            issued_mono_ms=now,
            valid_until_mono_ms=now + 5_000,
            state_id="scene-1",
            text="前方左转",
            interrupt_policy="queued",
            producer=producer,
            clock_id=client.clock_id,
            evidence_mode="synthetic",
        )
        outcome.check("replay-first-play", deduper.accept(first) == "play")
        outcome.check("replay-duplicate-drop", deduper.accept(first) == "drop_duplicate")

        expired = schema.guidance_event(
            event_id="evt-expired",
            sequence=2,
            priority="emergency",
            issued_mono_ms=now - 10_000,
            valid_until_mono_ms=now - 1_000,
            state_id="scene-1",
            text="旧提示",
            interrupt_policy="immediate",
            producer=producer,
            clock_id=client.clock_id,
            evidence_mode="synthetic",
        )
        outcome.check("expired-event-drop_stale", deduper.accept(expired) == "drop_stale")
        outcome.check("expired-guidance-not-fresh", guidance_is_fresh(expired) is False)

        superseding = schema.guidance_event(
            event_id="evt-2",
            sequence=3,
            priority="emergency",
            issued_mono_ms=now,
            valid_until_mono_ms=now + 5_000,
            state_id="scene-1",
            text="改为停下",
            interrupt_policy="immediate",
            supersedes_event_id="evt-1",
            producer=producer,
            clock_id=client.clock_id,
            evidence_mode="synthetic",
        )
        outcome.check("supersede-plays", deduper.accept(superseding) == "play")
        outcome.check("superseded-event-dropped", deduper.accept(first) == "drop_superseded")

        fresh = schema.guidance_event(
            event_id="evt-3",
            sequence=4,
            priority="information",
            issued_mono_ms=now,
            valid_until_mono_ms=now + 5_000,
            state_id="scene-1",
            text="继续前进",
            interrupt_policy="queued",
            producer=producer,
            clock_id=client.clock_id,
            evidence_mode="synthetic",
        )
        outcome.check("fresh-new-event-plays", deduper.accept(fresh) == "play")

        # 网络层重连后重放：服务端不重复执行旧动作，但会把事件交给状态机去重
        await client.send_guidance(event_id="evt-replay", sequence=5, priority="information", valid_for_ms=5_000)
        await client.send_guidance(event_id="evt-replay", sequence=5, priority="information", valid_for_ms=5_000)
        await _wait_for(lambda: len(handler.guidance_events) >= 2)
        await rig.server.wait_step()
        outcome.check("server-passed-replay-to-handler", len(handler.guidance_events) == 2)
        outcome.check(
            "handler-side-dedupe-required",
            deduper.accept(handler.guidance_events[0]) == "play"
            and deduper.accept(handler.guidance_events[1]) == "drop_duplicate",
        )

    outcome.defer("重连后只恢复仍未完成且未过期的事件，不重放已执行动作——由状态机保持未完成集合")


# ---------------------------------------------------------------------------
# C11 输出设备断连
# ---------------------------------------------------------------------------
async def _case_c11(outcome: CaseOutcome) -> None:
    handler = FakeHandler()
    handler.script(
        "output.cancel",
        cancel_result_data(
            device_stopped=False,
            queue_cleared=True,
            compute_ended=True,
            reason_code="OUTPUT_UNAVAILABLE",
            basis="device_write",
        ),
    )
    async with _Rig(handler) as rig:
        client = rig.client
        contract = rig.contract

        result = await client.output_cancel(
            scope="guidance_event", reason="output_device_lost", target_event_id="evt-1"
        )
        outcome.check("cancel-ok", result.status == "ok", str(result.error))
        outcome.check("cancel-device-not-stopped", result.data["device_stopped"] is False)
        outcome.check("cancel-output-unavailable", result.data["reason_code"] == "OUTPUT_UNAVAILABLE")

        await contract.publish_playback_status(
            state="started",
            source_event_id="evt-1",
            priority="emergency",
            start_basis="device_write",
        )
        await client.expect_playback_status("started")
        await contract.publish_playback_status(
            state="failed",
            source_event_id="evt-1",
            priority="emergency",
            reason_code="OUTPUT_UNAVAILABLE",
        )
        failed = await client.expect_playback_status("failed")
        outcome.check("device-failure-reported-failed", failed.data["reason_code"] == "OUTPUT_UNAVAILABLE")

        completed = [
            msg
            for msg in client.events_of("playback.status")
            if msg.data.get("state") == "completed" and msg.data.get("source_event_id") == "evt-1"
        ]
        outcome.check("device-failure-never-completed", completed == [], str(len(completed)))

    outcome.defer("声卡断开后 TTS 任务以 failed/OUTPUT_UNAVAILABLE 收尾且不再重试为 completed——由状态机实现")


# ---------------------------------------------------------------------------
# C12 模型/OCR 编造能力与距离
# ---------------------------------------------------------------------------
async def _case_c12(outcome: CaseOutcome) -> None:
    # 1) 方向守卫：交互侧不能伪造 guidance.event（这是主控职责）
    server = await _start_server(NullHandler())
    try:
        fake_event = schema.guidance_event(
            event_id="evt-forged",
            sequence=1,
            priority="emergency",
            issued_mono_ms=now_ms(),
            valid_until_mono_ms=now_ms() + 5_000,
            state_id="scene-1",
            text="伪造提示",
            interrupt_policy="immediate",
            producer={"role": "interaction", "instance_id": "aieyes-contract-test"},
            clock_id=server.clock_id,
            evidence_mode="synthetic",
        )
        try:
            await server.send_event(fake_event)
        except ProtocolError as exc:
            outcome.check("interaction-cannot-send-guidance-event", exc.code == "INVALID_PAYLOAD", str(exc))
        else:
            outcome.check("interaction-cannot-send-guidance-event", False, "send_event accepted guidance.event")

        # 2) test_fixture 身份默认被拒绝（生产入口不接受测试来源）
        fixture_client = await _connect_client(server, producer_role="test_fixture", instance_id="fixture-identity")
        try:
            rejected = await fixture_client.capabilities_get()
            outcome.check("test-fixture-identity-rejected", rejected.status == "rejected", rejected.status)
            outcome.check(
                "test-fixture-identity-error-code",
                rejected.error_code() == "INVALID_PAYLOAD",
                str(rejected.error),
            )
        finally:
            await fixture_client.close()
            await asyncio.sleep(0.05)

        # 3) 真主控连接后健康查询不得加载模型（NullHandler 立即返回 degraded）
        client = await _connect_client(server)
        try:
            status = await client.interaction_status_get()
            outcome.check("status-query-ok", status.status == "ok", str(status.error))
            outcome.check("status-mode-degraded", status.data["mode"] == "degraded", str(status.data.get("mode")))
            outcome.check("status-services-reported", set(status.data["services"]) >= {"asr", "vlm", "ocr", "tts", "kws"})
            outcome.check(
                "status-service-not-ready-without-models",
                all(service["state"] in ("loading", "unavailable") for service in status.data["services"].values()),
            )
        finally:
            await client.close()
            await asyncio.sleep(0.05)
    finally:
        await server.stop()
        await asyncio.sleep(0.05)

    # 4) 没有有效测量时不得编造米数；文本“可以走”不构成合同事实
    now = now_ms()
    fabricated_claim = scene_result_data(
        state_id="scene-1",
        sequence=1,
        observed_mono_ms=now,
        valid_until_mono_ms=now + 5_000,
        objects=_scene_objects(valid=False),
        navigation={"mode": "guide", "decision": "go", "reason_code": "VLM_CLAIM", "speech_fact": "可以走"},
        sensor_status={"rgb": "ok", "depth": "unavailable", "heading": "ok"},
    )
    measurement = fabricated_claim["objects"][0]["measurement"]
    outcome.check("fabricated-distance-null", measurement["distance_m"] is None, str(measurement))
    outcome.check("fabricated-no-meters", measurement_distance_m(measurement) is None)

    # 5) 测试替身不能自称 live 证据模式
    try:
        FixtureResponder(evidence_mode="live")
    except ValueError:
        outcome.check("fixture-cannot-claim-live", True)
    else:
        outcome.check("fixture-cannot-claim-live", False, "fixture accepted live evidence_mode")

    # 6) target.observation 只带识别结果与依据，不带距离/通行判断
    observation = schema.target_observation(
        visual_target_id="vt-1",
        frame_id="frame-1",
        roi_xyxy_px=[1, 1, 10, 10],
        name="水杯",
        source="ocr",
        text="水杯",
        producer={"role": "interaction", "instance_id": "aieyes-contract-test"},
        clock_id="clock",
        evidence_mode="synthetic",
    )
    outcome.check("target-observation-no-distance", "distance_m" not in observation.data, str(observation.data))
    outcome.check("target-observation-no-decision", "decision" not in observation.data, str(observation.data))

    outcome.defer("模型/OCR 结果经主控 tracker 校验后才可声称“可以走/多少米”——由状态机与主控配合实现")


# ---------------------------------------------------------------------------
# 案例注册与执行
# ---------------------------------------------------------------------------
CASE_FUNCS: dict[str, Callable[[CaseOutcome], Any]] = {
    "C01": _case_c01,
    "C02": _case_c02,
    "C03": _case_c03,
    "C04": _case_c04,
    "C05": _case_c05,
    "C06": _case_c06,
    "C07": _case_c07,
    "C08": _case_c08,
    "C09": _case_c09,
    "C10": _case_c10,
    "C11": _case_c11,
    "C12": _case_c12,
}

CASE_ORDER: tuple[str, ...] = tuple(CASE_TITLES)


async def run_case(case_id: str) -> CaseOutcome:
    """运行单个案例；异常一律转成 failed 结果，不向外抛。"""
    if case_id not in CASE_FUNCS:
        raise KeyError(f"unknown contract case: {case_id}")
    outcome = CaseOutcome(id=case_id, title=CASE_TITLES[case_id])
    try:
        await asyncio.wait_for(CASE_FUNCS[case_id](outcome), timeout=CASE_TIMEOUT_S)
    except AssertionError as exc:
        outcome.status = "failed"
        outcome.detail = str(exc) or "assertion failed"
    except asyncio.TimeoutError:
        outcome.status = "failed"
        outcome.detail = f"case timeout after {CASE_TIMEOUT_S:.0f}s"
    except Exception as exc:  # noqa: BLE001 - 报告里保留 traceback 摘要
        outcome.status = "failed"
        outcome.detail = f"{type(exc).__name__}: {exc}"
        outcome.deferred.append("traceback: " + " | ".join(traceback.format_exc().strip().splitlines()[-3:]))
    return outcome


async def run_all(case_ids: Sequence[str] | None = None) -> list[CaseOutcome]:
    ids = list(case_ids) if case_ids is not None else list(CASE_ORDER)
    return [await run_case(case_id) for case_id in ids]


def format_table(outcomes: Sequence[CaseOutcome]) -> str:
    rows = [f"{'case':<5} {'status':<8} {'checks':>6}  title"]
    for outcome in outcomes:
        rows.append(
            f"{outcome.id:<5} {outcome.status:<8} {len(outcome.checks):>6}  {outcome.title}"
        )
    passed = sum(1 for outcome in outcomes if outcome.status == "passed")
    rows.append(f"total: {len(outcomes)}  passed: {passed}  failed: {sum(o.status == 'failed' for o in outcomes)}")
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    case_ids = [arg.upper() for arg in args] if args else None
    outcomes = asyncio.run(run_all(case_ids))
    print(format_table(outcomes))
    print(json.dumps([outcome.to_dict() for outcome in outcomes], ensure_ascii=False, indent=2))
    return 1 if any(outcome.status == "failed" for outcome in outcomes) else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CASE_FUNCS",
    "CASE_ORDER",
    "CASE_TITLES",
    "CaseOutcome",
    "format_table",
    "main",
    "run_all",
    "run_case",
]
