"""interaction.v0.1 错误码、错误载荷与协议异常。

合同（Docs/MCPT/主控与交互接口合同.md）规定：

* ``schema_version`` 固定为 ``interaction.v0.1``；
* 角色为 ``main_control`` / ``interaction`` / ``test_fixture``；
* 证据模式为 ``live`` / ``replay`` / ``synthetic``；
* 响应状态为 ``ok`` / ``unavailable`` / ``rejected`` / ``error``；
* 错误至少覆盖 13 个错误码，失败时为 ``{code, message, retryable}``。

本模块只放枚举/常量/异常；消息构造与校验在 ``schema.py``。
"""

from __future__ import annotations

from typing import Any, Mapping

from aieyes.errors import AieyesError

#: 本实现对应的合同版本
SCHEMA_VERSION = "interaction.v0.1"

#: 生产者角色
ROLES = ("main_control", "interaction", "test_fixture")

#: 证据模式（替身只允许后两种）
EVIDENCE_MODES = ("live", "replay", "synthetic")

#: 响应 ``status`` 枚举
RESPONSE_STATUSES = ("ok", "unavailable", "rejected", "error")

#: 合同要求的 13 个错误码
ERROR_CODES = (
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

#: 本适配层对 ``retryable`` 的默认判断（合同未固定，调用方可覆盖）。
#: 只把“换个时机/重取一次可能成功”的编码标为可重试；权限/格式/身份类错误不可重试。
RETRYABLE_DEFAULT: dict[str, bool] = {
    "DEPENDENCY_UNAVAILABLE": True,
    "NO_FRAME": True,
    "STALE_FRAME": True,
    "INVALID_FRAME_REF": False,
    "UNSUPPORTED_VERSION": False,
    "CLOCK_UNVERIFIED": False,
    "STALE_STATE": False,
    "INVALID_TARGET": False,
    "INVALID_PAYLOAD": False,
    "BUSY": True,
    "DEADLINE_EXCEEDED": True,
    "CANCELLED": False,
    "OUTPUT_UNAVAILABLE": False,
}


class ProtocolError(AieyesError):
    """合同层可映射为响应 ``error`` 的异常。

    ``code`` 必须是 :data:`ERROR_CODES` 之一；``retryable`` 未指定时按
    :data:`RETRYABLE_DEFAULT` 取默认值。
    """

    def __init__(self, code: str, message: str, retryable: bool | None = None) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown interaction.v0.1 error code: {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = RETRYABLE_DEFAULT[code] if retryable is None else bool(retryable)

    def to_error(self) -> dict[str, Any]:
        return error_payload(self.code, self.message, self.retryable)


def error_payload(
    code: str,
    message: str,
    retryable: bool | None = None,
) -> dict[str, Any]:
    """构造响应的 ``error`` 对象 ``{code, message, retryable}``。"""
    if code not in ERROR_CODES:
        raise ValueError(f"unknown interaction.v0.1 error code: {code!r}")
    if retryable is None:
        retryable = RETRYABLE_DEFAULT[code]
    return {"code": code, "message": str(message), "retryable": bool(retryable)}


def as_error_payload(value: Any) -> dict[str, Any] | None:
    """把多种形式的错误转换为 ``{code, message, retryable}``。

    接受 ``None``、错误码字符串、``ProtocolError``、已有 error dict。
    """
    if value is None:
        return None
    if isinstance(value, ProtocolError):
        return value.to_error()
    if isinstance(value, str):
        return error_payload(value, value)
    if isinstance(value, Mapping):
        code = value.get("code")
        if not isinstance(code, str):
            raise ValueError(f"error payload missing code: {value!r}")
        return error_payload(code, str(value.get("message", "")), value.get("retryable"))
    raise ValueError(f"cannot interpret error value: {value!r}")
