"""interaction.v0.1 消息 schema：构造、编解码与严格校验。

合同文本见 ``Docs/MCPT/主控与交互接口合同.md``（权威）。本模块用纯
dataclass + dict + 显式校验函数实现，不引入 pydantic 等新依赖。

约定
----
* 公共字段：``schema_version`` / ``type`` / ``message_id`` / ``reply_to`` /
  ``request_id`` / ``session_id`` / ``producer`` / ``clock_id`` /
  ``timestamp_mono_ms`` / ``evidence_mode`` / ``data``。
* 请求额外带 ``deadline_mono_ms``；响应带 ``status`` 与 ``error``；
  事件两者都不带。
* ``Message`` 是内存表示；``to_dict()`` 产生合同 JSON，``from_dict()`` /
  ``parse_message()`` 校验后还原。``parse_message(..., strict=False)`` 用于
  入站消息：必需字段/枚举/类型仍严格，未知扩展字段容忍（向前兼容）。
* 本层只做“合同可判定”的校验（字段、枚举、关联、有效期、方向性语义），
  不实现设备能力，也不代替会话状态机。

9 组消息：capabilities、frame、scene、control、target、guidance、output.cancel、
playback.status、interaction.status。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from aieyes import clock
from aieyes.contract.errors import (
    EVIDENCE_MODES,
    RESPONSE_STATUSES,
    ROLES,
    SCHEMA_VERSION,
    ProtocolError,
    as_error_payload,
    error_payload,
)

# ---------------------------------------------------------------------------
# 消息类型与方向
# ---------------------------------------------------------------------------
#: 请求类型 → 结果类型
REPLY_TYPE: dict[str, str] = {
    "capabilities.get": "capabilities.result",
    "frame.acquire": "frame.result",
    "scene.query": "scene.result",
    "control.intent": "control.result",
    "target.observation": "target.result",
    "output.cancel": "output.cancel.result",
    "interaction.status.get": "interaction.status",
}

#: 请求消息（带 deadline，期望对应 result）
REQUEST_TYPES = frozenset(REPLY_TYPE)

#: 响应消息（带 reply_to / status / error）
RESPONSE_TYPES = frozenset(REPLY_TYPE.values())

#: 事件消息（无 deadline/status；guidance.event 由主控发，playback.status 由交互发）
EVENT_TYPES = frozenset({"guidance.event", "playback.status"})

ALL_MESSAGE_TYPES = REQUEST_TYPES | RESPONSE_TYPES | EVENT_TYPES

#: 交互内部取消的默认响应超时（T15 = 10 s，来自合同与 config.ContractConfig）
DEFAULT_RESPONSE_TIMEOUT_MS = 10_000

#: control.intent 允许的第一版动作（主控按任务状态判断能否执行）
CONTROL_ACTIONS = (
    "start_guidance",
    "pause_guidance",
    "resume_guidance",
    "end_guidance",
    "repeat_guidance",
)

#: guidance.event / playback.status 的优先级枚举
PRIORITIES = ("emergency", "navigation", "information")

#: playback.status 状态枚举
PLAYBACK_STATES = (
    "queued",
    "started",
    "completed",
    "cancelled",
    "failed",
    "dropped_stale",
)

#: 播放依据：设备写入 / 环回实测 / 调度估计（估计不能报成声卡实测）
START_BASES = ("device_write", "loopback_measurement", "estimated")

#: output.cancel 范围
CANCEL_SCOPES = ("ordinary_turn", "ordinary_session", "guidance_event")

#: interaction.status 的模式
INTERACTION_MODES = ("idle", "listening", "processing", "speaking", "degraded")

#: interaction.status.services 必须列出的服务
SERVICE_NAMES = ("asr", "vlm", "ocr", "tts", "kws")

_SERVICE_STATES = ("ready", "loading", "unavailable")
_OP_STATES = ("pending", "completed", "rejected")


# ---------------------------------------------------------------------------
# 标识与时间
# ---------------------------------------------------------------------------
def new_message_id(prefix: str = "msg") -> str:
    """生成唯一消息 ID（重发同一条消息时必须复用原 ID，由调用方负责）。"""
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def new_request_id(prefix: str = "req") -> str:
    """生成一次业务请求的幂等 ID。"""
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def now_ms() -> int:
    return clock.mono_ms()


# ---------------------------------------------------------------------------
# Message 数据类
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Message:
    """一条 interaction.v0.1 消息。

    ``type`` 决定哪些信封字段有意义：请求有 ``deadline_mono_ms``；响应有
    ``status``/``error``；事件两者都没有。``data`` 是合同定义的消息内容。
    """

    type: str
    message_id: str
    producer: dict[str, str]
    data: Mapping[str, Any] | None
    reply_to: str | None = None
    request_id: str | None = None
    session_id: str | None = None
    clock_id: str | None = None
    timestamp_mono_ms: int = 0
    evidence_mode: str = "synthetic"
    deadline_mono_ms: int | None = None
    status: str | None = None
    error: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION

    # --- 分类 -----------------------------------------------------------
    @property
    def is_request(self) -> bool:
        return self.type in REQUEST_TYPES

    @property
    def is_response(self) -> bool:
        return self.type in RESPONSE_TYPES

    @property
    def is_event(self) -> bool:
        return self.type in EVENT_TYPES

    @property
    def role(self) -> str:
        return self.producer.get("role", "")

    @property
    def instance_id(self) -> str:
        return self.producer.get("instance_id", "")

    @property
    def ok(self) -> bool:
        return self.is_response and self.status == "ok"

    def error_code(self) -> str | None:
        return None if self.error is None else self.error.get("code")

    # --- 编解码 ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """按合同示例的字段顺序输出 JSON 可序列化 dict。"""
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "type": self.type,
            "message_id": self.message_id,
            "reply_to": self.reply_to,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "producer": dict(self.producer),
            "clock_id": self.clock_id,
            "timestamp_mono_ms": self.timestamp_mono_ms,
        }
        if self.is_request:
            out["deadline_mono_ms"] = self.deadline_mono_ms
        out["evidence_mode"] = self.evidence_mode
        if self.is_response:
            out["status"] = self.status
            out["error"] = None if self.error is None else dict(self.error)
        out["data"] = None if self.data is None else _plain(self.data)
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, strict: bool = False) -> "Message":
        """从合同 JSON dict 还原并校验；入站用 ``strict=False``。"""
        if not isinstance(raw, Mapping):
            raise ProtocolError("INVALID_PAYLOAD", "message must be a JSON object")
        version = raw.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ProtocolError(
                "UNSUPPORTED_VERSION",
                f"unsupported schema_version: {version!r} (expected {SCHEMA_VERSION!r})",
            )
        for key in ("type", "message_id", "producer", "timestamp_mono_ms", "evidence_mode"):
            if key not in raw:
                raise ProtocolError("INVALID_PAYLOAD", f"message missing required field {key!r}")
        producer = raw.get("producer")
        if not isinstance(producer, Mapping):
            raise ProtocolError("INVALID_PAYLOAD", "producer must be an object")
        msg = cls(
            type=raw["type"],
            message_id=raw["message_id"],
            producer=dict(producer),
            data=raw.get("data"),
            reply_to=raw.get("reply_to"),
            request_id=raw.get("request_id"),
            session_id=raw.get("session_id"),
            clock_id=raw.get("clock_id"),
            timestamp_mono_ms=raw["timestamp_mono_ms"],
            evidence_mode=raw["evidence_mode"],
            deadline_mono_ms=raw.get("deadline_mono_ms"),
            status=raw.get("status"),
            error=raw.get("error"),
            schema_version=version,
        )
        validate_message(msg, strict=strict)
        if strict:
            allowed = _envelope_keys(msg)
            extra = set(raw) - allowed
            if extra:
                raise ProtocolError(
                    "INVALID_PAYLOAD", f"message has unknown envelope fields: {sorted(extra)}"
                )
        return msg


def _plain(value: Any) -> Any:
    """把 Mapping / tuple 等转成 JSON 友好的普通结构（保持不可变输入不变）。"""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _envelope_keys(msg: Message) -> set[str]:
    keys = {
        "schema_version",
        "type",
        "message_id",
        "reply_to",
        "request_id",
        "session_id",
        "producer",
        "clock_id",
        "timestamp_mono_ms",
        "evidence_mode",
        "data",
    }
    if msg.is_request:
        keys.add("deadline_mono_ms")
    if msg.is_response:
        keys.update({"status", "error"})
    return keys


def peek_envelope(raw: Any) -> dict[str, Any]:
    """尽力提取信封字段，用于给解析失败的消息回合同错误码。"""
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key in ("type", "message_id", "reply_to", "request_id", "session_id"):
        value = raw.get(key)
        if isinstance(value, str):
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# 校验原语
# ---------------------------------------------------------------------------
_MISSING = object()


def _fail(message: str, code: str = "INVALID_PAYLOAD") -> None:
    raise ProtocolError(code, message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _str(value: Any, name: str, *, nullable: bool = False, allow_empty: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str):
        _fail(f"{name} must be a string{' or null' if nullable else ''}")
    if not allow_empty and not value:
        _fail(f"{name} must not be empty")


def _int(value: Any, name: str, *, nullable: bool = False, minimum: int | None = None) -> None:
    if value is None and nullable:
        return
    if not _is_int(value):
        _fail(f"{name} must be an integer{' or null' if nullable else ''}")
    if minimum is not None and value < minimum:
        _fail(f"{name} must be >= {minimum}, got {value}")


def _bool(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        _fail(f"{name} must be a boolean")


def _enum(value: Any, name: str, allowed: Sequence[str], *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if value not in allowed:
        _fail(f"{name} must be one of {list(allowed)}, got {value!r}")


def _mapping(value: Any, name: str, *, nullable: bool = False) -> Mapping[str, Any] | None:
    if value is None and nullable:
        return None
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object{' or null' if nullable else ''}")
    return value


def _list(value: Any, name: str, *, nullable: bool = False) -> Sequence[Any] | None:
    if value is None and nullable:
        return None
    if not isinstance(value, (list, tuple)):
        _fail(f"{name} must be an array{' or null' if nullable else ''}")
    return value


def _fields(
    obj: Mapping[str, Any],
    name: str,
    *,
    required: Sequence[str] = (),
    optional: Sequence[str] = (),
    strict: bool,
) -> None:
    missing = [key for key in required if key not in obj]
    if missing:
        _fail(f"{name} missing required fields: {missing}")
    if strict:
        extra = [key for key in obj if key not in set(required) | set(optional)]
        if extra:
            _fail(f"{name} has unknown fields: {extra}")


def _roi(value: Any, name: str, *, nullable: bool = True) -> None:
    if value is None and nullable:
        return
    box = _list(value, name, nullable=False)
    assert box is not None
    if len(box) != 4:
        _fail(f"{name} must contain 4 integers [x0, y0, x1, y1]")
    for item in box:
        _int(item, f"{name}[i]", minimum=0)
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        _fail(f"{name} must satisfy x1 > x0 and y1 > y0")


def _error_object(value: Any, name: str, *, strict: bool) -> None:
    err = _mapping(value, name)
    assert err is not None
    _fields(err, name, required=("code", "message", "retryable"), strict=strict)
    if err["code"] not in ERROR_CODES_SET:
        _fail(f"{name}.code must be one of interaction.v0.1 error codes, got {err['code']!r}")
    _str(err["message"], f"{name}.message", allow_empty=True)
    _bool(err["retryable"], f"{name}.retryable")


ERROR_CODES_SET = frozenset(
    (
        "DEPENDENCY_UNAVAILABLE",
        "NO_FRAME",
        "STALE_FRAME",
        "INVALID_FRAME_REF",
        "UNSUPPORTED_VERSION",
        "CLOCK_UNVERIFIED",
        "STALE_STATE",
        "INVALID_TARGET",
        "INVALID_PAYLOAD",
        "BUSY",
        "DEADLINE_EXCEEDED",
        "CANCELLED",
        "OUTPUT_UNAVAILABLE",
    )
)


# ---------------------------------------------------------------------------
# 各消息 data 校验
# ---------------------------------------------------------------------------
def _validate_capabilities_get(data: Mapping[str, Any], strict: bool) -> None:
    _fields(data, "capabilities.get.data", optional=("requested",), strict=strict)
    if data.get("requested") is not None:
        requested = _list(data["requested"], "requested")
        assert requested is not None
        for i, item in enumerate(requested):
            _str(item, f"requested[{i}]")


def _validate_capabilities_result(data: Mapping[str, Any], strict: bool) -> None:
    _fields(
        data,
        "capabilities.result.data",
        required=("supported_versions", "capabilities"),
        optional=(
            "supported_evidence_modes",
            "host_boot_id",
            "unimplemented",
            "resources",
            "service",
        ),
        strict=strict,
    )
    versions = _list(data["supported_versions"], "supported_versions")
    assert versions is not None
    if not versions:
        _fail("supported_versions must not be empty")
    for i, item in enumerate(versions):
        _str(item, f"supported_versions[{i}]")
    caps = _mapping(data["capabilities"], "capabilities")
    assert caps is not None
    for key, value in caps.items():
        if not isinstance(key, str) or not key:
            _fail("capabilities keys must be non-empty strings")
        _bool(value, f"capabilities[{key}]")
    if data.get("supported_evidence_modes") is not None:
        modes = _list(data["supported_evidence_modes"], "supported_evidence_modes")
        assert modes is not None
        for i, mode in enumerate(modes):
            _enum(mode, f"supported_evidence_modes[{i}]", EVIDENCE_MODES)
    _str(data.get("host_boot_id"), "host_boot_id", nullable=True)
    if data.get("unimplemented") is not None:
        items = _list(data["unimplemented"], "unimplemented")
        assert items is not None
        for i, item in enumerate(items):
            _str(item, f"unimplemented[{i}]")
    if data.get("resources") is not None:
        _mapping(data["resources"], "resources")
    if data.get("service") is not None:
        _mapping(data["service"], "service")


def _validate_frame_acquire(data: Mapping[str, Any], strict: bool) -> None:
    _fields(
        data,
        "frame.acquire.data",
        required=("mode", "purpose", "frame_id", "max_age_ms", "max_side_px", "roi_xyxy_px"),
        strict=strict,
    )
    _enum(data["mode"], "mode", ("latest", "pinned"))
    _enum(data["purpose"], "purpose", ("vqa", "ocr", "compare"))
    _str(data["frame_id"], "frame_id", nullable=True)
    _int(data["max_age_ms"], "max_age_ms", nullable=True, minimum=0)
    _int(data["max_side_px"], "max_side_px", nullable=True, minimum=1)
    _roi(data["roi_xyxy_px"], "roi_xyxy_px")
    if data["mode"] == "latest" and data["frame_id"] is not None:
        _fail("frame.acquire latest must have frame_id=null")
    if data["mode"] == "pinned" and data["frame_id"] is None:
        _fail("frame.acquire pinned requires frame_id", code="INVALID_FRAME_REF")
    if data["roi_xyxy_px"] is not None and data["frame_id"] is None:
        _fail("roi_xyxy_px requires an explicit basis frame_id", code="INVALID_FRAME_REF")


def _validate_frame_result(data: Mapping[str, Any], strict: bool) -> None:
    _fields(
        data,
        "frame.result.data",
        required=(
            "frame_id",
            "captured_mono_ms",
            "source_kind",
            "is_historical",
            "width_px",
            "height_px",
            "color_space",
            "parent_frame_id",
            "transform_to_parent",
            "quality",
            "asset",
        ),
        strict=strict,
    )
    _str(data["frame_id"], "frame_id")
    _int(data["captured_mono_ms"], "captured_mono_ms", nullable=True)
    _enum(data["source_kind"], "source_kind", ("camera", "upload", "fixture"))
    _bool(data["is_historical"], "is_historical")
    _int(data["width_px"], "width_px", minimum=1)
    _int(data["height_px"], "height_px", minimum=1)
    _str(data["color_space"], "color_space")
    _str(data["parent_frame_id"], "parent_frame_id", nullable=True)
    if data["transform_to_parent"] is not None:
        matrix = _list(data["transform_to_parent"], "transform_to_parent")
        assert matrix is not None
        if len(matrix) != 9:
            _fail("transform_to_parent must be a row-major 3x3 matrix (9 numbers)")
        for i, item in enumerate(matrix):
            if not _is_number(item):
                _fail(f"transform_to_parent[{i}] must be a number")
    quality = _mapping(data["quality"], "quality")
    assert quality is not None
    if not quality:
        _fail("quality must not be empty (unchecked items use 'unknown')")
    for key, value in quality.items():
        _str(value, f"quality[{key}]")
    if data["captured_mono_ms"] is None:
        if not (data["is_historical"] and data["source_kind"] == "upload"):
            _fail(
                "captured_mono_ms=null is only valid for upload images with is_historical=true"
            )
    if data["is_historical"] is False and data["captured_mono_ms"] is None:
        _fail("non-historical frame requires captured_mono_ms")
    asset = _mapping(data["asset"], "asset")
    assert asset is not None
    _fields(
        asset,
        "asset",
        required=(
            "asset_id",
            "transport",
            "path",
            "encoding",
            "sha256",
            "available_until_mono_ms",
        ),
        strict=strict,
    )
    _str(asset["asset_id"], "asset.asset_id")
    _enum(asset["transport"], "asset.transport", ("file",))
    _str(asset["path"], "asset.path")
    _str(asset["encoding"], "asset.encoding")
    sha = asset["sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in sha):
        _fail("asset.sha256 must be a 64-character hex digest of the asset bytes")
    _int(asset["available_until_mono_ms"], "asset.available_until_mono_ms", minimum=0)


def _validate_scene_query(data: Mapping[str, Any], strict: bool) -> None:
    _fields(data, "scene.query.data", optional=("track_id", "fields"), strict=strict)
    _str(data.get("track_id"), "track_id", nullable=True)
    if data.get("fields") is not None:
        fields = _list(data["fields"], "fields")
        assert fields is not None
        for i, item in enumerate(fields):
            _str(item, f"fields[{i}]")


def _validate_measurement(data: Mapping[str, Any], name: str, strict: bool) -> None:
    _fields(
        data,
        name,
        required=("distance_m", "valid", "valid_until_mono_ms", "reason_code"),
        strict=strict,
    )
    _bool(data["valid"], f"{name}.valid")
    _int(data["valid_until_mono_ms"], f"{name}.valid_until_mono_ms", nullable=True, minimum=0)
    _str(data["reason_code"], f"{name}.reason_code", nullable=True)
    distance = data["distance_m"]
    if data["valid"]:
        if not _is_number(distance) or distance < 0:
            _fail(f"{name}.distance_m must be a non-negative number when valid=true")
    elif distance is not None:
        _fail(f"{name}.distance_m must be null when valid=false (0 does not mean unknown)")
    if not data["valid"] and not data["reason_code"]:
        _fail(f"{name}.reason_code must explain why the measurement is invalid")


def _validate_scene_result(data: Mapping[str, Any], strict: bool) -> None:
    _fields(
        data,
        "scene.result.data",
        required=(
            "state_id",
            "sequence",
            "observed_mono_ms",
            "valid_until_mono_ms",
            "coordinate_frame",
            "navigation",
            "objects",
            "sensor_status",
        ),
        optional=("measurement",),
        strict=strict,
    )
    _str(data["state_id"], "state_id")
    _int(data["sequence"], "sequence", minimum=0)
    _int(data["observed_mono_ms"], "observed_mono_ms", minimum=0)
    _int(data["valid_until_mono_ms"], "valid_until_mono_ms", minimum=0)
    if data["valid_until_mono_ms"] < data["observed_mono_ms"]:
        _fail("valid_until_mono_ms must not precede observed_mono_ms")
    _str(data["coordinate_frame"], "coordinate_frame")
    nav = _mapping(data["navigation"], "navigation")
    assert nav is not None
    _fields(
        nav,
        "navigation",
        required=("mode", "decision", "reason_code", "speech_fact"),
        strict=strict,
    )
    _str(nav["mode"], "navigation.mode")
    _str(nav["decision"], "navigation.decision")
    _str(nav["reason_code"], "navigation.reason_code", nullable=True)
    _str(nav["speech_fact"], "navigation.speech_fact", nullable=True)
    objects = _list(data["objects"], "objects")
    assert objects is not None
    for i, entry in enumerate(objects):
        obj = _mapping(entry, f"objects[{i}]")
        assert obj is not None
        _fields(
            obj,
            f"objects[{i}]",
            required=("track_id", "class_name", "bearing", "measurement"),
            strict=strict,
        )
        _str(obj["track_id"], f"objects[{i}].track_id")
        _str(obj["class_name"], f"objects[{i}].class_name")
        _str(obj["bearing"], f"objects[{i}].bearing", nullable=True)
        measurement = _mapping(obj["measurement"], f"objects[{i}].measurement")
        assert measurement is not None
        _validate_measurement(measurement, f"objects[{i}].measurement", strict)
    sensors = _mapping(data["sensor_status"], "sensor_status")
    assert sensors is not None
    _fields(
        sensors, "sensor_status", required=("rgb", "depth", "heading"), strict=strict
    )
    for key in ("rgb", "depth", "heading"):
        _str(sensors[key], f"sensor_status.{key}")
    if data.get("measurement") is not None:
        measurement = _mapping(data["measurement"], "measurement")
        assert measurement is not None
        _validate_measurement(measurement, "measurement", strict)


def _validate_control_intent(data: Mapping[str, Any], strict: bool) -> None:
    _fields(
        data,
        "control.intent.data",
        required=("action", "utterance_id", "final_transcript", "target_ref", "user_confirmed"),
        strict=strict,
    )
    _enum(data["action"], "action", CONTROL_ACTIONS)
    _str(data["utterance_id"], "utterance_id")
    _str(data["final_transcript"], "final_transcript")
    target_ref = data["target_ref"]
    if target_ref is not None and not isinstance(target_ref, (str, Mapping)):
        _fail("target_ref must be null, a string, or an object")
    if isinstance(target_ref, str):
        _str(target_ref, "target_ref")
    _bool(data["user_confirmed"], "user_confirmed")


def _validate_control_result(data: Mapping[str, Any], strict: bool) -> None:
    _fields(
        data,
        "control.result.data",
        required=("accepted", "operation_state", "actual_mode", "reason_code", "state_id"),
        strict=strict,
    )
    _bool(data["accepted"], "accepted")
    _enum(data["operation_state"], "operation_state", _OP_STATES)
    _str(data["actual_mode"], "actual_mode", nullable=True)
    _str(data["reason_code"], "reason_code", nullable=True)
    _str(data["state_id"], "state_id", nullable=True)
    if not data["accepted"] and data["operation_state"] != "rejected":
        _fail("accepted=false requires operation_state='rejected'")
    if data["accepted"] and data["operation_state"] == "rejected":
        _fail("accepted=true cannot pair with operation_state='rejected'")


def _validate_target_observation(data: Mapping[str, Any], strict: bool) -> None:
    _fields(
        data,
        "target.observation.data",
        required=(
            "visual_target_id",
            "frame_id",
            "roi_xyxy_px",
            "name",
            "text",
            "source",
            "user_selection_basis",
        ),
        strict=strict,
    )
    _str(data["visual_target_id"], "visual_target_id")
    _str(data["frame_id"], "frame_id")
    _roi(data["roi_xyxy_px"], "roi_xyxy_px", nullable=False)
    _str(data["name"], "name")
    _str(data["text"], "text", nullable=True)
    _str(data["source"], "source")
    _str(data["user_selection_basis"], "user_selection_basis", nullable=True)


def _validate_target_result(data: Mapping[str, Any], strict: bool) -> None:
    _fields(
        data,
        "target.result.data",
        required=("match_status", "track_id", "valid_until_mono_ms", "reason_code"),
        strict=strict,
    )
    _enum(data["match_status"], "match_status", ("matched", "unmatched"))
    _str(data["track_id"], "track_id", nullable=True)
    _int(data["valid_until_mono_ms"], "valid_until_mono_ms", nullable=True, minimum=0)
    _str(data["reason_code"], "reason_code", nullable=True)
    if data["match_status"] == "matched":
        if data["track_id"] is None or data["valid_until_mono_ms"] is None:
            _fail("matched target requires track_id and valid_until_mono_ms")
    else:
        if data["track_id"] is not None:
            _fail("unmatched target must not expose a main_control track_id")
        if data["valid_until_mono_ms"] is not None:
            _fail("unmatched target must not expose a validity window")


def _validate_guidance_event(data: Mapping[str, Any], strict: bool) -> None:
    _fields(
        data,
        "guidance.event.data",
        required=(
            "event_id",
            "sequence",
            "priority",
            "issued_mono_ms",
            "valid_until_mono_ms",
            "state_id",
            "text",
            "interrupt_policy",
            "supersedes_event_id",
        ),
        strict=strict,
    )
    _str(data["event_id"], "event_id")
    _int(data["sequence"], "sequence", minimum=0)
    _enum(data["priority"], "priority", PRIORITIES)
    _int(data["issued_mono_ms"], "issued_mono_ms", minimum=0)
    _int(data["valid_until_mono_ms"], "valid_until_mono_ms", minimum=0)
    if data["valid_until_mono_ms"] < data["issued_mono_ms"]:
        _fail("guidance valid_until_mono_ms must not precede issued_mono_ms")
    _str(data["state_id"], "state_id")
    _str(data["text"], "text")
    _str(data["interrupt_policy"], "interrupt_policy")
    _str(data["supersedes_event_id"], "supersedes_event_id", nullable=True)


def _validate_playback_status(data: Mapping[str, Any], strict: bool) -> None:
    _fields(
        data,
        "playback.status.data",
        required=(
            "source_event_id",
            "source_request_id",
            "utterance_id",
            "segment_index",
            "generation",
            "priority",
            "state",
            "occurred_mono_ms",
            "start_basis",
            "reason_code",
        ),
        strict=strict,
    )
    _str(data["source_event_id"], "source_event_id", nullable=True)
    _str(data["source_request_id"], "source_request_id", nullable=True)
    _str(data["utterance_id"], "utterance_id", nullable=True)
    _int(data["segment_index"], "segment_index", minimum=0)
    _int(data["generation"], "generation", nullable=True, minimum=0)
    _enum(data["priority"], "priority", PRIORITIES, nullable=True)
    _enum(data["state"], "state", PLAYBACK_STATES)
    _int(data["occurred_mono_ms"], "occurred_mono_ms", minimum=0)
    _enum(data["start_basis"], "start_basis", START_BASES, nullable=True)
    _str(data["reason_code"], "reason_code", nullable=True)
    if data["state"] == "started" and data["start_basis"] is None:
        _fail("playback state='started' requires a start_basis")
    if data["source_event_id"] is None and data["source_request_id"] is None:
        _fail("playback.status must reference a source event or request")
    if data["source_event_id"] is not None and data["priority"] is None:
        _fail("guidance-sourced playback.status requires priority")
    if data["source_event_id"] is None and data["priority"] is not None:
        _fail("ordinary-output playback.status has no guidance priority (use null)")


def _validate_output_cancel(data: Mapping[str, Any], strict: bool) -> None:
    _fields(
        data,
        "output.cancel.data",
        required=(
            "scope",
            "target_request_id",
            "target_event_id",
            "old_generation",
            "new_generation",
            "reason",
        ),
        strict=strict,
    )
    _enum(data["scope"], "scope", CANCEL_SCOPES)
    _str(data["reason"], "reason")
    _str(data["target_request_id"], "target_request_id", nullable=True)
    _str(data["target_event_id"], "target_event_id", nullable=True)
    _int(data["old_generation"], "old_generation", nullable=True, minimum=0)
    _int(data["new_generation"], "new_generation", nullable=True, minimum=0)
    if data["scope"] == "guidance_event":
        if data["target_event_id"] is None:
            _fail("scope='guidance_event' requires target_event_id")
        if data["target_request_id"] is not None:
            _fail("scope='guidance_event' must not carry a target_request_id")
        # 合同只要求 scope 明确；危险事件撤销由主控状态控制，代次字段可为空，
        # 因此这里不再拒绝 guidance scope 上的 generation（避免误拒合法发送方）。
    else:
        if data["target_event_id"] is not None:
            _fail("ordinary cancel must not carry target_event_id (danger events are withdrawn by main_control)")
        if data["scope"] == "ordinary_turn" and data["target_request_id"] is None:
            _fail("scope='ordinary_turn' requires target_request_id")
        old, new = data["old_generation"], data["new_generation"]
        if (old is None) != (new is None):
            _fail("old_generation and new_generation must be provided together")
        if old is not None and new is not None and new <= old:
            _fail("new_generation must be greater than old_generation")


def _validate_output_cancel_result(data: Mapping[str, Any], strict: bool) -> None:
    _fields(
        data,
        "output.cancel.result.data",
        required=("device_stopped", "queue_cleared", "compute_ended", "reason_code", "basis"),
        strict=strict,
    )
    _bool(data["device_stopped"], "device_stopped")
    _bool(data["queue_cleared"], "queue_cleared")
    _bool(data["compute_ended"], "compute_ended")
    _str(data["reason_code"], "reason_code", nullable=True)
    _enum(
        data["basis"],
        "basis",
        ("device_write", "loopback_measurement", "estimated", "not_applicable"),
        nullable=True,
    )


def _validate_interaction_status_get(data: Mapping[str, Any], strict: bool) -> None:
    _fields(data, "interaction.status.get.data", optional=("subscribe",), strict=strict)
    if data.get("subscribe") is not None:
        _bool(data["subscribe"], "subscribe")


def _validate_interaction_status(data: Mapping[str, Any], strict: bool) -> None:
    # 合同只要求“最少包含”这些字段，允许实现追加只读字段。
    _fields(
        data,
        "interaction.status.data",
        required=(
            "mode",
            "active_request_id",
            "audio_input_state",
            "audio_output_state",
            "services",
            "queues",
            "memory",
            "metrics_window",
            "last_error",
        ),
        strict=False,
    )
    _enum(data["mode"], "mode", INTERACTION_MODES)
    _str(data["active_request_id"], "active_request_id", nullable=True)
    for key in ("audio_input_state", "audio_output_state"):
        value = data[key]
        if isinstance(value, str):
            _str(value, key)
        else:
            states = _mapping(value, key)
            assert states is not None
            _fields(states, key, required=("state",), strict=False)
            _str(states["state"], f"{key}.state")
    services = _mapping(data["services"], "services")
    assert services is not None
    for name in SERVICE_NAMES:
        if name not in services:
            _fail(f"services missing required entry {name!r}")
    for name, entry in services.items():
        _str(name, "services key")
        service = _mapping(entry, f"services[{name}]")
        assert service is not None
        _fields(service, f"services[{name}]", required=("version", "state"), strict=False)
        _str(service["version"], f"services[{name}].version", nullable=True)
        _enum(service["state"], f"services[{name}].state", _SERVICE_STATES)
    queues = _mapping(data["queues"], "queues")
    assert queues is not None
    for name, length in queues.items():
        _str(name, "queues key")
        _int(length, f"queues[{name}]", minimum=0)
    memory = _mapping(data["memory"], "memory")
    assert memory is not None
    _fields(
        memory,
        "memory",
        required=("metric", "bytes", "processes", "sampling"),
        strict=False,
    )
    _str(memory["metric"], "memory.metric")
    _int(memory["bytes"], "memory.bytes", nullable=True, minimum=0)
    if memory["processes"] is not None:
        procs = _list(memory["processes"], "memory.processes")
        assert procs is not None
        for i, item in enumerate(procs):
            _str(item, f"memory.processes[{i}]")
    _str(memory["sampling"], "memory.sampling", nullable=True)
    metrics = _mapping(data["metrics_window"], "metrics_window")
    assert metrics is not None
    _fields(metrics, "metrics_window", required=("window_ms", "samples"), strict=False)
    _int(metrics["window_ms"], "metrics_window.window_ms", nullable=True, minimum=0)
    _int(metrics["samples"], "metrics_window.samples", minimum=0)
    if data["last_error"] is not None:
        _error_object(data["last_error"], "last_error", strict=False)


_DATA_VALIDATORS = {
    "capabilities.get": _validate_capabilities_get,
    "capabilities.result": _validate_capabilities_result,
    "frame.acquire": _validate_frame_acquire,
    "frame.result": _validate_frame_result,
    "scene.query": _validate_scene_query,
    "scene.result": _validate_scene_result,
    "control.intent": _validate_control_intent,
    "control.result": _validate_control_result,
    "target.observation": _validate_target_observation,
    "target.result": _validate_target_result,
    "guidance.event": _validate_guidance_event,
    "playback.status": _validate_playback_status,
    "output.cancel": _validate_output_cancel,
    "output.cancel.result": _validate_output_cancel_result,
    "interaction.status.get": _validate_interaction_status_get,
    "interaction.status": _validate_interaction_status,
}


# ---------------------------------------------------------------------------
# 整体校验
# ---------------------------------------------------------------------------
def validate_message(msg: Message, *, strict: bool = True) -> Message:
    """校验完整消息；返回 ``msg`` 便于链式使用。

    入站解码用 ``strict=False``（未知扩展字段容忍）；本层构造的消息用
    ``strict=True``。
    """
    if msg.type not in ALL_MESSAGE_TYPES:
        raise ProtocolError("INVALID_PAYLOAD", f"unknown message type: {msg.type!r}")
    if msg.schema_version != SCHEMA_VERSION:
        raise ProtocolError(
            "UNSUPPORTED_VERSION", f"unsupported schema_version: {msg.schema_version!r}"
        )
    _str(msg.message_id, "message_id")
    _str(msg.reply_to, "reply_to", nullable=True)
    _str(msg.request_id, "request_id", nullable=True)
    _str(msg.session_id, "session_id", nullable=True)
    _str(msg.clock_id, "clock_id", nullable=True)
    producer = _mapping(msg.producer, "producer")
    assert producer is not None
    _fields(producer, "producer", required=("role", "instance_id"), strict=strict)
    _enum(producer["role"], "producer.role", ROLES)
    _str(producer["instance_id"], "producer.instance_id")
    _int(msg.timestamp_mono_ms, "timestamp_mono_ms", minimum=0)
    _enum(msg.evidence_mode, "evidence_mode", EVIDENCE_MODES)

    if msg.is_request:
        if msg.reply_to is not None:
            raise ProtocolError("INVALID_PAYLOAD", "request must have reply_to=null")
        if msg.status is not None or msg.error is not None:
            raise ProtocolError("INVALID_PAYLOAD", "request must not carry status/error")
        _int(msg.deadline_mono_ms, "deadline_mono_ms", minimum=0)
    elif msg.is_response:
        _int(msg.deadline_mono_ms, "deadline_mono_ms", nullable=True)
        if msg.deadline_mono_ms is not None:
            raise ProtocolError("INVALID_PAYLOAD", "response must not carry deadline_mono_ms")
        _str(msg.reply_to, "reply_to")
        _str(msg.request_id, "request_id")
        _enum(msg.status, "status", RESPONSE_STATUSES)
        if msg.status == "ok":
            if msg.error is not None:
                raise ProtocolError("INVALID_PAYLOAD", "status=ok requires error=null")
            if msg.data is None:
                raise ProtocolError("INVALID_PAYLOAD", "status=ok requires data")
        else:
            if msg.error is None:
                raise ProtocolError("INVALID_PAYLOAD", f"status={msg.status} requires error")
            _error_object(msg.error, "error", strict=strict)
    else:  # event
        if msg.reply_to is not None:
            raise ProtocolError("INVALID_PAYLOAD", "event must have reply_to=null")
        if msg.deadline_mono_ms is not None:
            raise ProtocolError("INVALID_PAYLOAD", "event must not carry deadline_mono_ms")
        if msg.status is not None or msg.error is not None:
            raise ProtocolError("INVALID_PAYLOAD", "event must not carry status/error")

    # 实时消息必须带已握手时钟域（首次握手允许 null）
    if (
        msg.evidence_mode == "live"
        and msg.clock_id is None
        and msg.type not in ("capabilities.get", "capabilities.result")
    ):
        raise ProtocolError(
            "CLOCK_UNVERIFIED", "live interaction.v0.1 messages require a verified clock_id"
        )

    if msg.data is None:
        if msg.ok:
            raise ProtocolError("INVALID_PAYLOAD", f"{msg.type} requires data")
    else:
        data = _mapping(msg.data, "data")
        assert data is not None
        _DATA_VALIDATORS[msg.type](data, strict)
    return msg


def parse_message(raw: Any, *, strict: bool = False) -> Message:
    """解码入站 JSON dict → ``Message``；失败抛 :class:`ProtocolError`。"""
    return Message.from_dict(raw, strict=strict)


# ---------------------------------------------------------------------------
# 构造
# ---------------------------------------------------------------------------
def _producer(producer: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(producer, Mapping):
        raise TypeError("producer must be a mapping with role/instance_id")
    return {"role": str(producer.get("role", "")), "instance_id": str(producer.get("instance_id", ""))}


def producer_of(role: str, instance_id: str) -> dict[str, str]:
    """便捷构造 producer 对象。"""
    return {"role": role, "instance_id": instance_id}


def _request(
    msg_type: str,
    data: Mapping[str, Any],
    *,
    producer: Mapping[str, str],
    session_id: str | None = None,
    request_id: str | None = None,
    message_id: str | None = None,
    clock_id: str | None = None,
    timestamp_mono_ms: int | None = None,
    deadline_mono_ms: int | None = None,
    evidence_mode: str = "synthetic",
    strict: bool = True,
) -> Message:
    ts = now_ms() if timestamp_mono_ms is None else int(timestamp_mono_ms)
    msg = Message(
        type=msg_type,
        message_id=new_message_id() if message_id is None else message_id,
        producer=_producer(producer),
        data=_plain(data),
        reply_to=None,
        request_id=new_request_id() if request_id is None else request_id,
        session_id=session_id,
        clock_id=clock_id,
        timestamp_mono_ms=ts,
        evidence_mode=evidence_mode,
        deadline_mono_ms=ts + DEFAULT_RESPONSE_TIMEOUT_MS if deadline_mono_ms is None else int(deadline_mono_ms),
    )
    return validate_message(msg, strict=strict)


def response_for(
    request: Message,
    *,
    data: Mapping[str, Any] | None = None,
    producer: Mapping[str, str],
    status: str = "ok",
    error: Any = None,
    message_id: str | None = None,
    clock_id: str | None = None,
    timestamp_mono_ms: int | None = None,
    evidence_mode: str | None = None,
    session_id: str | None = None,
    strict: bool = True,
) -> Message:
    """按请求构造关联响应：``reply_to=request.message_id``，``request_id`` 相同。"""
    if not request.is_request:
        raise ProtocolError("INVALID_PAYLOAD", f"{request.type} is not a request message")
    err = as_error_payload(error)
    if status == "ok" and err is not None:
        status = "error"
    if status != "ok" and err is None:
        err = error_payload("DEPENDENCY_UNAVAILABLE", f"{request.type} failed without error detail")
    msg = Message(
        type=REPLY_TYPE[request.type],
        message_id=new_message_id() if message_id is None else message_id,
        producer=_producer(producer),
        data=None if data is None else _plain(data),
        reply_to=request.message_id,
        request_id=request.request_id,
        session_id=request.session_id if session_id is None else session_id,
        clock_id=request.clock_id if clock_id is None else clock_id,
        timestamp_mono_ms=now_ms() if timestamp_mono_ms is None else int(timestamp_mono_ms),
        evidence_mode=request.evidence_mode if evidence_mode is None else evidence_mode,
        status=status,
        error=err,
    )
    return validate_message(msg, strict=strict)


def error_response(
    request: Message,
    code: str,
    message: str | None = None,
    *,
    producer: Mapping[str, str],
    retryable: bool | None = None,
    status: str = "error",
    **kwargs: Any,
) -> Message:
    """构造失败响应（``error={code,message,retryable}``）。"""
    err = error_payload(code, message or code, retryable)
    return response_for(request, data=None, producer=producer, status=status, error=err, **kwargs)


def _event(
    msg_type: str,
    data: Mapping[str, Any],
    *,
    producer: Mapping[str, str],
    session_id: str | None = None,
    request_id: str | None = None,
    message_id: str | None = None,
    clock_id: str | None = None,
    timestamp_mono_ms: int | None = None,
    evidence_mode: str = "synthetic",
    strict: bool = True,
) -> Message:
    msg = Message(
        type=msg_type,
        message_id=new_message_id("evt") if message_id is None else message_id,
        producer=_producer(producer),
        data=_plain(data),
        reply_to=None,
        request_id=request_id,
        session_id=session_id,
        clock_id=clock_id,
        timestamp_mono_ms=now_ms() if timestamp_mono_ms is None else int(timestamp_mono_ms),
        evidence_mode=evidence_mode,
    )
    return validate_message(msg, strict=strict)


# --- 1. capabilities ------------------------------------------------------
def capabilities_get(
    *,
    producer: Mapping[str, str],
    requested: Sequence[str] | None = None,
    **kwargs: Any,
) -> Message:
    data: dict[str, Any] = {} if requested is None else {"requested": list(requested)}
    return _request("capabilities.get", data, producer=producer, **kwargs)


def capabilities_result(
    request: Message,
    *,
    producer: Mapping[str, str],
    supported_versions: Sequence[str] = (SCHEMA_VERSION,),
    capabilities: Mapping[str, bool] | None = None,
    supported_evidence_modes: Sequence[str] = EVIDENCE_MODES,
    host_boot_id: str | None = None,
    unimplemented: Sequence[str] = (),
    resources: Mapping[str, Any] | None = None,
    service: Mapping[str, Any] | None = None,
    status: str = "ok",
    error: Any = None,
    **kwargs: Any,
) -> Message:
    if not request.is_request or request.type != "capabilities.get":
        raise ProtocolError("INVALID_PAYLOAD", "capabilities_result requires a capabilities.get request")
    data = None
    if status == "ok":
        data = {
            "supported_versions": list(supported_versions),
            "capabilities": dict(capabilities or {}),
            "supported_evidence_modes": list(supported_evidence_modes),
            "host_boot_id": host_boot_id,
            "unimplemented": list(unimplemented),
            "resources": None if resources is None else dict(resources),
            "service": None if service is None else dict(service),
        }
    return response_for(request, data=data, producer=producer, status=status, error=error, **kwargs)


# --- 2. frame -------------------------------------------------------------
def frame_acquire(
    *,
    mode: str,
    purpose: str,
    producer: Mapping[str, str],
    frame_id: str | None = None,
    max_age_ms: int | None = None,
    max_side_px: int | None = None,
    roi_xyxy_px: Sequence[int] | None = None,
    **kwargs: Any,
) -> Message:
    data = {
        "mode": mode,
        "purpose": purpose,
        "frame_id": frame_id,
        "max_age_ms": max_age_ms,
        "max_side_px": max_side_px,
        "roi_xyxy_px": None if roi_xyxy_px is None else list(roi_xyxy_px),
    }
    return _request("frame.acquire", data, producer=producer, **kwargs)


def frame_result(
    request: Message,
    *,
    producer: Mapping[str, str],
    data: Mapping[str, Any] | None = None,
    status: str = "ok",
    error: Any = None,
    **kwargs: Any,
) -> Message:
    _expect_request(request, "frame.acquire", "frame_result")
    return response_for(request, data=data, producer=producer, status=status, error=error, **kwargs)


# --- 3. scene -------------------------------------------------------------
def scene_query(
    *,
    producer: Mapping[str, str],
    track_id: str | None = None,
    fields: Sequence[str] | None = None,
    **kwargs: Any,
) -> Message:
    data = {"track_id": track_id, "fields": None if fields is None else list(fields)}
    return _request("scene.query", data, producer=producer, **kwargs)


def scene_result(
    request: Message,
    *,
    producer: Mapping[str, str],
    data: Mapping[str, Any] | None = None,
    status: str = "ok",
    error: Any = None,
    **kwargs: Any,
) -> Message:
    _expect_request(request, "scene.query", "scene_result")
    return response_for(request, data=data, producer=producer, status=status, error=error, **kwargs)


# --- 4. control -----------------------------------------------------------
def control_intent(
    *,
    action: str,
    utterance_id: str,
    final_transcript: str,
    producer: Mapping[str, str],
    target_ref: str | Mapping[str, Any] | None = None,
    user_confirmed: bool = False,
    **kwargs: Any,
) -> Message:
    data = {
        "action": action,
        "utterance_id": utterance_id,
        "final_transcript": final_transcript,
        "target_ref": None if target_ref is None else _plain(target_ref) if isinstance(target_ref, Mapping) else target_ref,
        "user_confirmed": user_confirmed,
    }
    return _request("control.intent", data, producer=producer, **kwargs)


def control_result(
    request: Message,
    *,
    producer: Mapping[str, str],
    data: Mapping[str, Any] | None = None,
    status: str = "ok",
    error: Any = None,
    **kwargs: Any,
) -> Message:
    _expect_request(request, "control.intent", "control_result")
    return response_for(request, data=data, producer=producer, status=status, error=error, **kwargs)


# --- 5. target ------------------------------------------------------------
def target_observation(
    *,
    visual_target_id: str,
    frame_id: str,
    roi_xyxy_px: Sequence[int],
    name: str,
    source: str,
    producer: Mapping[str, str],
    text: str | None = None,
    user_selection_basis: str | None = None,
    **kwargs: Any,
) -> Message:
    data = {
        "visual_target_id": visual_target_id,
        "frame_id": frame_id,
        "roi_xyxy_px": list(roi_xyxy_px),
        "name": name,
        "text": text,
        "source": source,
        "user_selection_basis": user_selection_basis,
    }
    return _request("target.observation", data, producer=producer, **kwargs)


def target_result(
    request: Message,
    *,
    producer: Mapping[str, str],
    data: Mapping[str, Any] | None = None,
    status: str = "ok",
    error: Any = None,
    **kwargs: Any,
) -> Message:
    _expect_request(request, "target.observation", "target_result")
    return response_for(request, data=data, producer=producer, status=status, error=error, **kwargs)


# --- 6. guidance / playback ----------------------------------------------
def guidance_event(
    *,
    event_id: str,
    sequence: int,
    priority: str,
    issued_mono_ms: int,
    valid_until_mono_ms: int,
    state_id: str,
    text: str,
    interrupt_policy: str,
    producer: Mapping[str, str],
    supersedes_event_id: str | None = None,
    **kwargs: Any,
) -> Message:
    data = {
        "event_id": event_id,
        "sequence": sequence,
        "priority": priority,
        "issued_mono_ms": issued_mono_ms,
        "valid_until_mono_ms": valid_until_mono_ms,
        "state_id": state_id,
        "text": text,
        "interrupt_policy": interrupt_policy,
        "supersedes_event_id": supersedes_event_id,
    }
    return _event("guidance.event", data, producer=producer, **kwargs)


def playback_status(
    *,
    state: str,
    occurred_mono_ms: int,
    producer: Mapping[str, str],
    source_event_id: str | None = None,
    source_request_id: str | None = None,
    utterance_id: str | None = None,
    segment_index: int = 0,
    generation: int | None = None,
    priority: str | None = None,
    start_basis: str | None = None,
    reason_code: str | None = None,
    **kwargs: Any,
) -> Message:
    data = {
        "source_event_id": source_event_id,
        "source_request_id": source_request_id,
        "utterance_id": utterance_id,
        "segment_index": segment_index,
        "generation": generation,
        "priority": priority,
        "state": state,
        "occurred_mono_ms": occurred_mono_ms,
        "start_basis": start_basis,
        "reason_code": reason_code,
    }
    return _event("playback.status", data, producer=producer, **kwargs)


# --- 7. output.cancel -----------------------------------------------------
def output_cancel(
    *,
    scope: str,
    reason: str,
    producer: Mapping[str, str],
    target_request_id: str | None = None,
    target_event_id: str | None = None,
    old_generation: int | None = None,
    new_generation: int | None = None,
    **kwargs: Any,
) -> Message:
    data = {
        "scope": scope,
        "target_request_id": target_request_id,
        "target_event_id": target_event_id,
        "old_generation": old_generation,
        "new_generation": new_generation,
        "reason": reason,
    }
    return _request("output.cancel", data, producer=producer, **kwargs)


def output_cancel_result(
    request: Message,
    *,
    producer: Mapping[str, str],
    data: Mapping[str, Any] | None = None,
    status: str = "ok",
    error: Any = None,
    **kwargs: Any,
) -> Message:
    _expect_request(request, "output.cancel", "output_cancel_result")
    return response_for(request, data=data, producer=producer, status=status, error=error, **kwargs)


# --- 8. interaction.status ------------------------------------------------
def interaction_status_get(
    *,
    producer: Mapping[str, str],
    subscribe: bool | None = None,
    **kwargs: Any,
) -> Message:
    data: dict[str, Any] = {} if subscribe is None else {"subscribe": subscribe}
    return _request("interaction.status.get", data, producer=producer, **kwargs)


def interaction_status(
    request: Message,
    *,
    producer: Mapping[str, str],
    data: Mapping[str, Any] | None = None,
    status: str = "ok",
    error: Any = None,
    **kwargs: Any,
) -> Message:
    _expect_request(request, "interaction.status.get", "interaction_status")
    return response_for(request, data=data, producer=producer, status=status, error=error, **kwargs)


def _expect_request(request: Message, msg_type: str, what: str) -> None:
    if not request.is_request or request.type != msg_type:
        raise ProtocolError("INVALID_PAYLOAD", f"{what} requires a {msg_type} request")


# ---------------------------------------------------------------------------
# 合同语义辅助（供状态机使用，不代替状态机）
# ---------------------------------------------------------------------------
def measurement_distance_m(measurement: Mapping[str, Any], *, now_ms: int | None = None) -> float | None:
    """只在 ``valid=true`` 且未过期时返回米数；否则 None（不得用 0 代替未知）。"""
    if not isinstance(measurement, Mapping):
        return None
    if measurement.get("valid") is not True:
        return None
    distance = measurement.get("distance_m")
    if not _is_number(distance):
        return None
    valid_until = measurement.get("valid_until_mono_ms")
    if not _is_int(valid_until):
        return None
    now = clock.mono_ms() if now_ms is None else now_ms
    if now > valid_until:
        return None
    return float(distance)


def scene_measurement(
    scene_result_msg: Message,
    *,
    track_id: str | None = None,
    now_ms: int | None = None,
) -> float | None:
    """从 ``scene.result`` 中取可转述距离；无效/过期/不匹配时返回 None。

    同时检查对象测量有效期与场景 ``valid_until_mono_ms``（取更早者）。
    """
    if scene_result_msg.type != "scene.result" or not isinstance(scene_result_msg.data, Mapping):
        return None
    data = scene_result_msg.data
    now = clock.mono_ms() if now_ms is None else now_ms
    scene_valid_until = data.get("valid_until_mono_ms")
    if _is_int(scene_valid_until) and now > scene_valid_until:
        return None
    candidates: list[Mapping[str, Any]] = []
    if data.get("measurement") is not None and isinstance(data.get("measurement"), Mapping):
        candidates.append(data["measurement"])
    for obj in data.get("objects") or ():
        if not isinstance(obj, Mapping):
            continue
        if track_id is not None and obj.get("track_id") != track_id:
            continue
        if isinstance(obj.get("measurement"), Mapping):
            candidates.append(obj["measurement"])
    for measurement in candidates:
        distance = measurement_distance_m(measurement, now_ms=now)
        if distance is not None and _is_int(scene_valid_until):
            measurement_until = measurement.get("valid_until_mono_ms")
            if not _is_int(measurement_until) or now > measurement_until:
                continue
        if distance is not None:
            return distance
    return None


def guidance_is_fresh(event_msg: Message, *, now_ms: int | None = None) -> bool:
    """事件是否仍在有效期内（过期的行动提示不得补播）。"""
    if event_msg.type != "guidance.event" or not isinstance(event_msg.data, Mapping):
        return False
    valid_until = event_msg.data.get("valid_until_mono_ms")
    if not _is_int(valid_until):
        return False
    now = clock.mono_ms() if now_ms is None else now_ms
    return now <= valid_until


def is_completed_ok(control_result_msg: Message, *, expected_mode: str | None = None) -> bool:
    """``control.result`` 是否可支持“已经执行完成”的说法。"""
    msg = control_result_msg
    if msg.type != "control.result" or msg.status != "ok" or not isinstance(msg.data, Mapping):
        return False
    data = msg.data
    if data.get("accepted") is not True or data.get("operation_state") != "completed":
        return False
    if expected_mode is not None and data.get("actual_mode") != expected_mode:
        return False
    return True


def playback_key(status_msg: Message) -> tuple[str, str, str, int]:
    """``playback.status`` 去重键：来源消息 + 片段身份（原请求/事件关联不可被 TTS 子任务替代）。"""
    data = status_msg.data if isinstance(status_msg.data, Mapping) else {}
    return (
        str(data.get("source_event_id") or ""),
        str(data.get("source_request_id") or ""),
        str(data.get("utterance_id") or ""),
        int(data.get("segment_index") or 0),
    )


def handshake_fingerprint(capabilities_msg: Message) -> tuple[str, str | None, str | None]:
    """握手指纹：``(instance_id, host_boot_id, clock_id)``。

    任一变化（进程重启/主机重启/时钟域变化）都要求重新握手并撤销旧实时状态。
    """
    if capabilities_msg.type not in ("capabilities.get", "capabilities.result"):
        raise ProtocolError("INVALID_PAYLOAD", "handshake_fingerprint requires a capabilities message")
    data = capabilities_msg.data if isinstance(capabilities_msg.data, Mapping) else {}
    return (
        capabilities_msg.instance_id,
        data.get("host_boot_id") if isinstance(data.get("host_boot_id"), str) else None,
        capabilities_msg.clock_id,
    )


class SequenceGuard:
    """主控实例内 ``sequence`` 单调性检查（序号倒退 → ``STALE_STATE``）。

    ``stream_id`` 通常取 ``f"{instance_id}:{channel}"``；主控重启后调用
    ``reset()`` 再重新握手。``check`` 返回 False 表示重复/未前进的序号。
    """

    def __init__(self) -> None:
        self._last: dict[str, int] = {}

    def check(self, stream_id: str, sequence: int) -> bool:
        if not _is_int(sequence) or sequence < 0:
            raise ProtocolError("INVALID_PAYLOAD", f"invalid sequence: {sequence!r}")
        last = self._last.get(stream_id)
        if last is not None and sequence < last:
            raise ProtocolError(
                "STALE_STATE",
                f"sequence regression on {stream_id}: {sequence} < {last}",
            )
        self._last[stream_id] = sequence
        return last is None or sequence > last

    def last(self, stream_id: str) -> int | None:
        return self._last.get(stream_id)

    def reset(self, stream_id: str | None = None) -> None:
        if stream_id is None:
            self._last.clear()
        else:
            self._last.pop(stream_id, None)


class GuidanceDeduper:
    """``guidance.event`` 去重/过期/替代判断（状态机在入队前调用）。

    * 重复 ``event_id`` → ``drop_duplicate``（重连回放不补播）；
    * 已过 ``valid_until_mono_ms`` → ``drop_stale``；
    * 已被后续事件 ``supersedes_event_id`` 指定 → ``drop_superseded``；
    * 其余 → ``play`` 并记录。
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._superseded: set[str] = set()

    def accept(self, event_msg: Message, *, now_ms: int | None = None) -> str:
        if event_msg.type != "guidance.event" or not isinstance(event_msg.data, Mapping):
            raise ProtocolError("INVALID_PAYLOAD", "GuidanceDeduper accepts guidance.event only")
        data = event_msg.data
        event_id = data.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ProtocolError("INVALID_PAYLOAD", "guidance event missing event_id")
        if event_id in self._superseded:
            return "drop_superseded"
        if not guidance_is_fresh(event_msg, now_ms=now_ms):
            return "drop_stale"
        if event_id in self._seen:
            return "drop_duplicate"
        self._seen.add(event_id)
        supersedes = data.get("supersedes_event_id")
        if isinstance(supersedes, str) and supersedes:
            self._superseded.add(supersedes)
        return "play"

    def reset(self) -> None:
        self._seen.clear()
        self._superseded.clear()


__all__ = [
    "ALL_MESSAGE_TYPES",
    "CANCEL_SCOPES",
    "CONTROL_ACTIONS",
    "DEFAULT_RESPONSE_TIMEOUT_MS",
    "EVENT_TYPES",
    "EVIDENCE_MODES",
    "GuidanceDeduper",
    "INTERACTION_MODES",
    "Message",
    "PLAYBACK_STATES",
    "PRIORITIES",
    "REPLY_TYPE",
    "REQUEST_TYPES",
    "RESPONSE_STATUSES",
    "RESPONSE_TYPES",
    "SCHEMA_VERSION",
    "START_BASES",
    "SequenceGuard",
    "capabilities_get",
    "capabilities_result",
    "control_intent",
    "control_result",
    "error_payload",
    "error_response",
    "frame_acquire",
    "frame_result",
    "guidance_event",
    "guidance_is_fresh",
    "handshake_fingerprint",
    "interaction_status",
    "interaction_status_get",
    "is_completed_ok",
    "measurement_distance_m",
    "new_message_id",
    "new_request_id",
    "now_ms",
    "output_cancel",
    "output_cancel_result",
    "parse_message",
    "peek_envelope",
    "playback_key",
    "playback_status",
    "producer_of",
    "response_for",
    "scene_measurement",
    "scene_query",
    "scene_result",
    "target_observation",
    "target_result",
    "validate_message",
]
