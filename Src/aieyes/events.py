"""进程内事件与事件总线。

采集线程 / 网络线程只通过 ``EventBus.post`` 投递事件；会话状态机在
asyncio 线程 ``await bus.get()`` 消费，保证所有状态迁移单线程串行。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# 事件定义（frozen dataclass，时间均为 clock.mono_ms()）
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WakeDetected:
    keyword: str
    t_ms: int


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    t_ms: int


@dataclass(frozen=True, slots=True)
class PartialText:
    text: str
    t_ms: int


@dataclass(frozen=True, slots=True)
class UtteranceRecognized:
    text: str
    t_audio_start_ms: int
    t_audio_end_ms: int
    t_final_ms: int
    #: VAD 段音频（含 preroll）；阶段 5 身份核验使用，可为 None
    samples: Any = None


@dataclass(frozen=True, slots=True)
class PlaybackStarted:
    piece_id: str
    t_ms: int
    text: str = ""


@dataclass(frozen=True, slots=True)
class PlaybackFinished:
    piece_id: str
    t_ms: int
    played_ms: int
    total_ms: int
    cancelled: bool = False
    text: str = ""
    #: 未播完的原因（``playback_stalled`` / ``user_cancel`` / ``interrupted`` 等）；
    #: 正常播完为 ``""``。引擎与 sink 用同一字符串上报，``cancelled`` 布尔分不出
    #: 「卡死」与「用户主动取消」，需要区分时读这个字段。
    stop_reason: str = ""


@dataclass(frozen=True, slots=True)
class FrameCaptured:
    frame_id: str
    t_ms: int
    width: int = 0
    height: int = 0


@dataclass(frozen=True, slots=True)
class ContractIncoming:
    """主控发来的一条已解码消息（contract 层保证是 dict）。"""

    message: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CancelRequested:
    reason: str
    t_ms: int


@dataclass(frozen=True, slots=True)
class TimerFired:
    timer_id: str
    t_ms: int


@dataclass(frozen=True, slots=True)
class DeviceFault:
    device: str  # audio_in | audio_out | camera | llm | tts
    detail: str
    t_ms: int


@dataclass(frozen=True, slots=True)
class ShutdownRequested:
    reason: str = "signal"


# ---------------------------------------------------------------------------
# 事件总线
# ---------------------------------------------------------------------------
class EventBus:
    """asyncio.Queue 封装；``post`` 可从任意线程调用。"""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_running_loop()
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._closed = False

    def post(self, event: Any) -> None:
        if self._closed:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._queue.put_nowait(event)
        else:
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
            except RuntimeError:  # loop already closed
                pass

    async def get(self) -> Any:
        return await self._queue.get()

    def get_nowait(self) -> Any | None:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def close(self) -> None:
        self._closed = True


@dataclass
class TimerSpec:
    """一次性/重复计时器的声明（由状态机注册，由 TimerService 触发）。"""

    timer_id: str
    interval_ms: int
    repeat: bool = False
    auto_start: bool = True
    payload: dict[str, Any] = field(default_factory=dict)
