"""单调时钟与运行时间工具。

约定（与 interaction.v0.1 合同一致）：

* 进程内所有相对时间都使用 ``mono_ms()``：从系统启动开始计数的毫秒数，
  与 ``clock_id`` 一起写进消息的 ``timestamp_mono_ms``。
* 需要人类可读时间时使用 ``wall_iso()``，只用于日志，不参与任何判定。
* 计时器一律基于 monotonic 时间，禁止使用 ``time.time()`` 做超时判断。
"""

from __future__ import annotations

import datetime as _dt
import time

#: contract clock identity for this process
CLOCK_ID = "aieyes-monotonic-boot"

#: 进程启动时的 monotonic 值，仅用于把 mono_ms 换算成可读的“运行时长”
_PROCESS_T0 = time.monotonic()


def mono_ms() -> int:
    """毫秒级单调时钟（boot 基准，跨线程/跨进程可比）。"""
    return int(time.monotonic() * 1000)


def process_ms() -> int:
    """进程启动以来的毫秒数（日志用）。"""
    return int((time.monotonic() - _PROCESS_T0) * 1000)


def wall_iso() -> str:
    """本地时区 ISO-8601 时间，毫秒精度。"""
    return _dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def wall_ms() -> int:
    """Unix 墙钟毫秒，仅用于日志。"""
    return int(time.time() * 1000)


class Stopwatch:
    """轻量计时器，``ms()`` 返回自构造/``reset()`` 以来的毫秒数。"""

    __slots__ = ("_t0",)

    def __init__(self) -> None:
        self._t0 = mono_ms()

    def reset(self) -> None:
        self._t0 = mono_ms()

    def ms(self) -> int:
        return mono_ms() - self._t0

    def seconds(self) -> float:
        return self.ms() / 1000.0

    def expired(self, budget_ms: int) -> bool:
        return self.ms() >= budget_ms
