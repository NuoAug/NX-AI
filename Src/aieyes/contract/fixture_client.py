"""测试夹具：扮演 main_control 的客户端 + 脚本化响应器。

用途：在无真实主控/设备时，按 interaction.v0.1 合同驱动交互侧
（``ContractServer``），用于合同案例 C01–C12 的软件回归。

分工：

* :class:`FixtureResponder` —— “主控大脑”的合同替身：响应
  ``frame.acquire`` / ``scene.query`` / ``control.intent`` /
  ``target.observation`` / ``capabilities.get``，并可脚本化任意结果；
  ``control.intent`` 按 ``request_id`` 只执行一次（幂等）。
* :class:`FixtureClient` —— TCP 客户端 + 消息收发 + 事件等待 + 往返时延
  记录；``connect()`` 后可直接调用 ``acquire_frame()`` 等方法。

夹具只用于 ``synthetic``/``replay`` 证据模式，不能冒充真实 ``live`` 主控；
服务端默认也只接受 ``main_control`` 角色（``allow_test_fixture`` 是显式测试
开关）。
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

from aieyes import clock
from aieyes.contract.errors import EVIDENCE_MODES, SCHEMA_VERSION, ProtocolError, error_payload
from aieyes.contract.handlers import (
    control_result_data,
    frame_result_data,
    measurement_data,
    scene_object_data,
    scene_result_data,
    status_for_error_code,
    target_result_data,
)
from aieyes.contract.schema import (
    Message,
    capabilities_get,
    capabilities_result,
    control_intent,
    frame_acquire,
    guidance_event,
    interaction_status_get,
    now_ms,
    output_cancel,
    response_for,
    scene_query,
    target_observation,
)
from aieyes.contract.transport import Connection, ConnectionClosed

logger = logging.getLogger(__name__)

MAIN_CONTROL_ROLE = "main_control"
FIXTURE_ROLE = "test_fixture"

#: 夹具宣称具备的能力（合同 capabilities 布尔表；可按案例覆盖）
DEFAULT_CAPABILITIES: dict[str, bool] = {
    "frame_provider": True,
    "derived_image": True,
    "scene_state": True,
    "navigation_decision": True,
    "track_id_match": True,
    "guidance_events": True,
    "interaction_status": True,
    "output_cancel": True,
}

#: 夹具默认场景（单位：米/弧度；invalid 测量用 distance_m=None）
DEFAULT_SCENE_OBJECTS = ("track-1",)


class FixtureResponder:
    """扮演 main_control 的合同响应器（供 FixtureClient 附着到连接）。"""

    def __init__(
        self,
        *,
        instance_id: str = "main-control-fixture",
        clock_id: str = "fixture-monotonic-boot",
        evidence_mode: str = "synthetic",
        host_boot_id: str = "fixture-boot-id",
        capabilities: Mapping[str, bool] | None = None,
        unimplemented: Sequence[str] = (),
        service: Mapping[str, Any] | None = None,
        resources: Mapping[str, Any] | None = None,
    ) -> None:
        if evidence_mode not in EVIDENCE_MODES:
            raise ValueError(f"invalid evidence_mode: {evidence_mode!r}")
        if evidence_mode == "live":
            raise ValueError(
                "fixture must not claim live evidence; live requires a real main_control adapter"
            )
        self.producer: dict[str, str] = {"role": MAIN_CONTROL_ROLE, "instance_id": instance_id}
        self.clock_id = clock_id
        self.evidence_mode = evidence_mode
        self.host_boot_id = host_boot_id
        self.capabilities = dict(DEFAULT_CAPABILITIES)
        if capabilities is not None:
            self.capabilities.update({str(k): bool(v) for k, v in capabilities.items()})
        self.unimplemented = list(unimplemented)
        self.service = dict(service) if service is not None else {"name": "main-control-fixture", "version": "0.1"}
        self.resources = dict(resources) if resources is not None else None

        self.requests: list[Message] = []
        self.counts: Counter[str] = Counter()
        self.control_executions: Counter[str] = Counter()
        self.control_results: dict[str, dict[str, Any]] = {}
        self.frames: dict[str, dict[str, Any]] = {}
        self.frame_order: list[str] = []
        self.frame_seq = 0
        self.scene_sequence = 0
        self.target_results: dict[str, dict[str, Any]] = {}

        #: 按消息类型覆盖自动响应；回调返回 data Mapping 或抛 ProtocolError
        self.scripts: dict[str, Callable[[Message], Any]] = {}
        #: 按消息类型注入延迟（秒），用于超时案例
        self.delays: dict[str, float] = {}
        self.closed = False

    # --- 配置 -----------------------------------------------------------
    def script(self, msg_type: str, callback: Callable[[Message], Any] | None) -> "FixtureResponder":
        if callback is None:
            self.scripts.pop(msg_type, None)
        else:
            self.scripts[msg_type] = callback
        return self

    def delay(self, msg_type: str, seconds: float) -> "FixtureResponder":
        self.delays[msg_type] = float(seconds)
        return self

    def add_frame(
        self,
        frame_id: str,
        *,
        captured_mono_ms: int | None = None,
        width_px: int = 1280,
        height_px: int = 720,
        source_kind: str = "camera",
        is_historical: bool = False,
        parent_frame_id: str | None = None,
        transform_to_parent: Sequence[float] | None = None,
        available_until_mono_ms: int | None = None,
    ) -> dict[str, Any]:
        now = now_ms()
        record: dict[str, Any] = {
            "frame_id": frame_id,
            "captured_mono_ms": captured_mono_ms,
            "width_px": width_px,
            "height_px": height_px,
            "source_kind": source_kind,
            "is_historical": is_historical,
            "parent_frame_id": parent_frame_id,
            "transform_to_parent": None if transform_to_parent is None else list(transform_to_parent),
            "available_until_mono_ms": available_until_mono_ms if available_until_mono_ms is not None else now + 60_000,
        }
        self.frames[frame_id] = record
        if frame_id not in self.frame_order:
            self.frame_order.append(frame_id)
        return record

    def frame_data(
        self,
        frame_id: str,
        *,
        max_side_px: int | None = None,
        quality: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        record = self.frames[frame_id]
        width = int(record["width_px"])
        height = int(record["height_px"])
        if isinstance(max_side_px, int) and max_side_px > 0:
            scale = min(1.0, max_side_px / float(max(width, height)))
            width = max(1, int(round(width * scale)))
            height = max(1, int(round(height * scale)))
        sha = hashlib.sha256(frame_id.encode("utf-8")).hexdigest()
        return frame_result_data(
            frame_id=frame_id,
            captured_mono_ms=record["captured_mono_ms"],
            width_px=width,
            height_px=height,
            sha256=sha,
            asset_id=f"asset-{frame_id}",
            path=f"/tmp/main-control/{frame_id}.jpg",
            encoding="jpeg",
            source_kind=record["source_kind"],
            is_historical=record["is_historical"],
            color_space="RGB8",
            parent_frame_id=record["parent_frame_id"],
            transform_to_parent=record["transform_to_parent"],
            quality=dict(quality) if quality is not None else {"focus": "good", "exposure": "good"},
            available_until_mono_ms=record["available_until_mono_ms"],
        )

    def default_scene_data(self, *, sequence: int | None = None) -> dict[str, Any]:
        now = now_ms()
        if sequence is None:
            self.scene_sequence += 1
            sequence = self.scene_sequence
        else:
            self.scene_sequence = max(self.scene_sequence, int(sequence))
        valid_until = now + 2_000
        objects = [
            scene_object_data(
                track_id=track_id,
                class_name="person",
                bearing=-0.1 + 0.2 * index,
                measurement=measurement_data(
                    distance_m=2.0 + 0.5 * index,
                    valid=True,
                    valid_until_mono_ms=valid_until,
                    reason_code=None,
                ),
            )
            for index, track_id in enumerate(DEFAULT_SCENE_OBJECTS)
        ]
        return scene_result_data(
            state_id="scene-1",
            sequence=sequence,
            observed_mono_ms=now,
            valid_until_mono_ms=valid_until,
            coordinate_frame="camera_depth_optical",
            objects=objects,
            navigation={
                "mode": "guide",
                "decision": "go",
                "reason_code": "PATH_CLEAR",
                "speech_fact": "前方道路畅通",
            },
            sensor_status={"rgb": "ok", "depth": "ok", "heading": "ok"},
        )

    # --- 主入口 ---------------------------------------------------------
    async def handle(self, request: Message) -> Message | None:
        """响应一个交互侧请求；返回 ``*.result``（异常映射为合同错误）。"""
        if not request.is_request:
            return None
        self.counts[request.type] += 1
        self.requests.append(request)
        delay = self.delays.get(request.type)
        if delay:
            await asyncio.sleep(delay)
        try:
            if request.type == "capabilities.get":
                return capabilities_result(
                    request,
                    producer=self.producer,
                    supported_versions=(SCHEMA_VERSION,),
                    capabilities=self.capabilities,
                    supported_evidence_modes=[self.evidence_mode],
                    host_boot_id=self.host_boot_id,
                    unimplemented=self.unimplemented,
                    resources=self.resources,
                    service=self.service,
                    clock_id=self.clock_id,
                    evidence_mode=self.evidence_mode,
                    timestamp_mono_ms=now_ms(),
                )
            script = self.scripts.get(request.type)
            if script is not None:
                data = script(request)
                if inspect.isawaitable(data):
                    data = await data
            else:
                data = self._auto_data(request)
            if not isinstance(data, Mapping):
                raise ProtocolError(
                    "INVALID_PAYLOAD",
                    f"fixture script for {request.type} returned {type(data).__name__}, expected mapping",
                )
            return response_for(
                request,
                data=data,
                producer=self.producer,
                clock_id=self.clock_id,
                timestamp_mono_ms=now_ms(),
                evidence_mode=self.evidence_mode,
            )
        except ProtocolError as exc:
            return response_for(
                request,
                data=None,
                producer=self.producer,
                status=status_for_error_code(exc.code),
                error=error_payload(exc.code, exc.message, exc.retryable),
                clock_id=self.clock_id,
                timestamp_mono_ms=now_ms(),
                evidence_mode=self.evidence_mode,
            )
        except Exception as exc:  # pragma: no cover - 夹具 bug 暴露为合同错误
            logger.exception("fixture responder failed on %s", request.type)
            return response_for(
                request,
                data=None,
                producer=self.producer,
                status="error",
                error=error_payload("DEPENDENCY_UNAVAILABLE", f"fixture crashed: {exc!r}"[:300], True),
                clock_id=self.clock_id,
                timestamp_mono_ms=now_ms(),
                evidence_mode=self.evidence_mode,
            )

    # --- 自动响应 -------------------------------------------------------
    def _auto_data(self, request: Message) -> Mapping[str, Any]:
        if request.type == "frame.acquire":
            return self._frame_acquire_data(request)
        if request.type == "scene.query":
            if not self.capabilities.get("scene_state", False):
                raise ProtocolError("DEPENDENCY_UNAVAILABLE", "main control has no scene_state capability")
            return self.default_scene_data()
        if request.type == "control.intent":
            return self._control_intent_data(request)
        if request.type == "target.observation":
            return self._target_observation_data(request)
        raise ProtocolError("INVALID_PAYLOAD", f"fixture cannot answer {request.type}")

    def _frame_acquire_data(self, request: Message) -> Mapping[str, Any]:
        if not self.capabilities.get("frame_provider", False):
            raise ProtocolError("DEPENDENCY_UNAVAILABLE", "main control has no frame_provider capability")
        data = request.data
        mode = data.get("mode")
        now = now_ms()
        if mode == "pinned":
            frame_id = data.get("frame_id")
            record = self.frames.get(frame_id)
            if record is None:
                raise ProtocolError("NO_FRAME", f"main control has no frame {frame_id!r}")
            max_age = data.get("max_age_ms")
            captured = record["captured_mono_ms"]
            if isinstance(max_age, int) and captured is not None and now - captured > max_age:
                raise ProtocolError("STALE_FRAME", f"frame {frame_id!r} is older than max_age_ms={max_age}")
        else:
            self.frame_seq += 1
            frame_id = f"frame-{self.frame_seq}"
            self.add_frame(frame_id, captured_mono_ms=now)
        return self.frame_data(frame_id, max_side_px=data.get("max_side_px"))

    def _control_intent_data(self, request: Message) -> Mapping[str, Any]:
        request_id = request.request_id
        if request_id in self.control_results:
            # 合同：同 request_id 只执行一次；重复请求返回当前结果
            return dict(self.control_results[request_id])
        action = request.data.get("action")
        data = control_result_data(
            accepted=True,
            operation_state="completed",
            actual_mode="guidance" if action != "end_guidance" else "idle",
            reason_code=None,
            state_id="mc-state-1",
        )
        self.control_results[request_id] = data
        self.control_executions[request_id] += 1
        return data

    def _target_observation_data(self, request: Message) -> Mapping[str, Any]:
        visual_target_id = request.data.get("visual_target_id")
        scripted = self.target_results.get(visual_target_id)
        if scripted is not None:
            return dict(scripted)
        if not self.capabilities.get("track_id_match", False):
            return target_result_data(match_status="unmatched", reason_code="TRACK_ID_MATCH_UNSUPPORTED")
        return target_result_data(
            match_status="matched",
            track_id=f"track-{visual_target_id}",
            valid_until_mono_ms=now_ms() + 5_000,
        )


class FixtureClient:
    """扮演 main_control 的 TCP 客户端（测试夹具）。"""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        responder: FixtureResponder | None = None,
        instance_id: str = "main-control-fixture",
        clock_id: str = "fixture-monotonic-boot",
        evidence_mode: str = "synthetic",
        session_id: str | None = "sess-fixture",
        producer_role: str = MAIN_CONTROL_ROLE,
        timeout_ms: int = 5_000,
        connect_timeout_s: float = 5.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.responder = responder if responder is not None else FixtureResponder(
            instance_id=instance_id,
            clock_id=clock_id,
            evidence_mode=evidence_mode,
        )
        self.instance_id = instance_id
        self.clock_id = clock_id
        self.evidence_mode = evidence_mode
        self.session_id = session_id
        self.producer: dict[str, str] = {"role": producer_role, "instance_id": instance_id}
        self.timeout_ms = int(timeout_ms)
        self.connect_timeout_s = float(connect_timeout_s)

        self.connection: Connection | None = None
        self.events: list[Message] = []
        self.received_requests: list[Message] = []
        self.responses: list[Message] = []
        self.requests_sent: list[Message] = []
        self.guidance_sent: list[Message] = []
        self.rtt_samples_ms: list[float] = []
        self.closed = False
        self._event_cursor = 0
        self._event_condition = asyncio.Condition()

    # --- 连接 -----------------------------------------------------------
    async def connect(self) -> "FixtureClient":
        if self.connection is not None:
            raise RuntimeError("FixtureClient already connected")
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port), timeout=self.connect_timeout_s
        )
        self.connection = Connection(
            reader,
            writer,
            on_message=self._on_message,
            on_closed=self._on_closed,
            name="fixture-main_control",
        )
        self.connection.start()
        return self

    async def close(self) -> None:
        self.closed = True
        if self.connection is not None:
            await self.connection.close("fixture closing")
            self.connection = None

    async def __aenter__(self) -> "FixtureClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    def _on_closed(self) -> None:
        self.closed = True

    async def _on_message(self, msg: Message) -> None:
        if msg.is_event:
            await self._record_event(msg)
            return
        if msg.is_request:
            # 交互侧不应向主控发这些请求，但夹具仍按合同应答，便于诊断
            self.received_requests.append(msg)
            reply = await self.responder.handle(msg)
            if reply is not None and self.connection is not None:
                await self.connection.send(reply)

    async def _record_event(self, msg: Message) -> None:
        self.events.append(msg)
        async with self._event_condition:
            self._event_condition.notify_all()

    # --- 请求 -----------------------------------------------------------
    def _request_kwargs(self, timeout_ms: int | None) -> dict[str, Any]:
        ts = now_ms()
        timeout = self.timeout_ms if timeout_ms is None else int(timeout_ms)
        return {
            "clock_id": self.clock_id,
            "timestamp_mono_ms": ts,
            "deadline_mono_ms": ts + max(1, timeout),
            "evidence_mode": self.evidence_mode,
            "session_id": self.session_id,
        }

    async def request(self, msg: Message, *, timeout_ms: int | None = None) -> Message:
        if self.connection is None:
            raise ConnectionClosed("fixture is not connected")
        if not msg.is_request:
            raise ValueError(f"{msg.type} is not a request message")
        self.requests_sent.append(msg)
        started = clock.mono_ms()
        timeout = self.timeout_ms if timeout_ms is None else int(timeout_ms)
        reply = await self.connection.request(msg, timeout_ms=timeout)
        self.rtt_samples_ms.append(max(0, clock.mono_ms() - started))
        self.responses.append(reply)
        return reply

    async def send(self, msg: Message) -> None:
        """发送任意消息（事件/错误帧测试用，不做方向限制）。"""
        if self.connection is None:
            raise ConnectionClosed("fixture is not connected")
        await self.connection.send(msg)

    async def request_unchecked(self, msg: Message, *, timeout_ms: int | None = None) -> Message:
        """发送**未经校验**的消息（非法版本/字段等负向案例用）。"""
        if self.connection is None:
            raise ConnectionClosed("fixture is not connected")
        timeout = self.timeout_ms if timeout_ms is None else int(timeout_ms)
        return await self.connection.request(msg, timeout_ms=timeout)

    async def capabilities_get(
        self, requested: Sequence[str] | None = None, *, timeout_ms: int | None = None
    ) -> Message:
        msg = capabilities_get(
            producer=self.producer, requested=requested, **self._request_kwargs(timeout_ms)
        )
        return await self.request(msg, timeout_ms=timeout_ms)

    async def frame_acquire(
        self,
        *,
        mode: str = "latest",
        purpose: str = "vqa",
        frame_id: str | None = None,
        max_age_ms: int | None = None,
        max_side_px: int | None = None,
        roi_xyxy_px: Sequence[int] | None = None,
        timeout_ms: int | None = None,
    ) -> Message:
        msg = frame_acquire(
            mode=mode,
            purpose=purpose,
            frame_id=frame_id,
            max_age_ms=max_age_ms,
            max_side_px=max_side_px,
            roi_xyxy_px=roi_xyxy_px,
            producer=self.producer,
            **self._request_kwargs(timeout_ms),
        )
        return await self.request(msg, timeout_ms=timeout_ms)

    async def scene_query(
        self,
        *,
        track_id: str | None = None,
        fields: Sequence[str] | None = None,
        timeout_ms: int | None = None,
    ) -> Message:
        msg = scene_query(
            track_id=track_id,
            fields=fields,
            producer=self.producer,
            **self._request_kwargs(timeout_ms),
        )
        return await self.request(msg, timeout_ms=timeout_ms)

    async def control_intent(
        self,
        action: str,
        *,
        utterance_id: str,
        final_transcript: str,
        target_ref: str | Mapping[str, Any] | None = None,
        user_confirmed: bool = False,
        timeout_ms: int | None = None,
    ) -> Message:
        msg = control_intent(
            action=action,
            utterance_id=utterance_id,
            final_transcript=final_transcript,
            target_ref=target_ref,
            user_confirmed=user_confirmed,
            producer=self.producer,
            **self._request_kwargs(timeout_ms),
        )
        return await self.request(msg, timeout_ms=timeout_ms)

    async def target_observation(
        self,
        *,
        visual_target_id: str,
        frame_id: str,
        roi_xyxy_px: Sequence[int],
        name: str,
        source: str,
        text: str | None = None,
        user_selection_basis: str | None = None,
        timeout_ms: int | None = None,
    ) -> Message:
        msg = target_observation(
            visual_target_id=visual_target_id,
            frame_id=frame_id,
            roi_xyxy_px=roi_xyxy_px,
            name=name,
            source=source,
            text=text,
            user_selection_basis=user_selection_basis,
            producer=self.producer,
            **self._request_kwargs(timeout_ms),
        )
        return await self.request(msg, timeout_ms=timeout_ms)

    async def output_cancel(
        self,
        *,
        scope: str,
        reason: str,
        target_request_id: str | None = None,
        target_event_id: str | None = None,
        old_generation: int | None = None,
        new_generation: int | None = None,
        timeout_ms: int | None = None,
    ) -> Message:
        msg = output_cancel(
            scope=scope,
            reason=reason,
            target_request_id=target_request_id,
            target_event_id=target_event_id,
            old_generation=old_generation,
            new_generation=new_generation,
            producer=self.producer,
            **self._request_kwargs(timeout_ms),
        )
        return await self.request(msg, timeout_ms=timeout_ms)

    async def interaction_status_get(
        self, subscribe: bool | None = None, *, timeout_ms: int | None = None
    ) -> Message:
        msg = interaction_status_get(
            producer=self.producer, subscribe=subscribe, **self._request_kwargs(timeout_ms)
        )
        return await self.request(msg, timeout_ms=timeout_ms)

    async def send_guidance(
        self,
        *,
        event_id: str,
        sequence: int = 1,
        priority: str = "information",
        state_id: str = "scene-1",
        text: str = "前方有障碍",
        interrupt_policy: str = "immediate",
        valid_for_ms: int = 5_000,
        supersedes_event_id: str | None = None,
        issued_mono_ms: int | None = None,
    ) -> Message:
        issued = now_ms() if issued_mono_ms is None else int(issued_mono_ms)
        msg = guidance_event(
            event_id=event_id,
            sequence=sequence,
            priority=priority,
            issued_mono_ms=issued,
            valid_until_mono_ms=issued + int(valid_for_ms),
            state_id=state_id,
            text=text,
            interrupt_policy=interrupt_policy,
            supersedes_event_id=supersedes_event_id,
            producer=self.producer,
            clock_id=self.clock_id,
            timestamp_mono_ms=now_ms(),
            evidence_mode=self.evidence_mode,
            session_id=self.session_id,
        )
        self.guidance_sent.append(msg)
        await self.send(msg)
        return msg

    # --- 事件等待 -------------------------------------------------------
    async def expect_event(
        self,
        predicate: Callable[[Message], bool] | None = None,
        *,
        msg_type: str | None = None,
        timeout_s: float = 5.0,
    ) -> Message:
        """等待一条尚未消费的事件；超时抛 ``asyncio.TimeoutError``。"""

        def matches(msg: Message) -> bool:
            if msg_type is not None and msg.type != msg_type:
                return False
            return predicate is None or bool(predicate(msg))

        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(timeout_s)
        async with self._event_condition:
            while True:
                for index in range(self._event_cursor, len(self.events)):
                    if matches(self.events[index]):
                        self._event_cursor = index + 1
                        return self.events[index]
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError(
                        f"no matching event ({msg_type or 'any'}) within {timeout_s} s"
                    )
                try:
                    await asyncio.wait_for(self._event_condition.wait(), remaining)
                except asyncio.TimeoutError:
                    raise asyncio.TimeoutError(
                        f"no matching event ({msg_type or 'any'}) within {timeout_s} s"
                    ) from None

    async def expect_playback_status(self, state: str | None = None, *, timeout_s: float = 5.0) -> Message:
        def predicate(msg: Message) -> bool:
            return state is None or msg.data.get("state") == state

        return await self.expect_event(
            predicate, msg_type="playback.status", timeout_s=timeout_s
        )

    def events_of(self, msg_type: str) -> list[Message]:
        return [msg for msg in self.events if msg.type == msg_type]


__all__ = [
    "DEFAULT_CAPABILITIES",
    "FIXTURE_ROLE",
    "FixtureClient",
    "FixtureResponder",
    "MAIN_CONTROL_ROLE",
]
