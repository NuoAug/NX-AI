"""采集断流上报（``DeviceFault(device="audio_in")``）的定向用例。

本机没有 ``sounddevice``（`test_capture.py` 已被 ``importorskip`` 跳过），
所以这里不打开真实流，而是把 ``start()`` 成功之后的状态显式摆出来
（``_running`` / ``_gen`` / ``_last_frame_ms``），直接驱动看门狗的
单次判定 ``_watchdog_tick()``。这样四条要求都是确定性断言，不依赖计时。
"""

from __future__ import annotations

from aieyes.audio.capture import DEFAULT_FAULT_TIMEOUT_MS, SoundDeviceSource
from aieyes.clock import mono_ms
from aieyes.events import DeviceFault


class FakeBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    def post(self, event) -> None:
        self.events.append(event)

    def faults(self) -> list[DeviceFault]:
        return [e for e in self.events if isinstance(e, DeviceFault)]


def _begin_session(src: SoundDeviceSource, *, silent_ms: int) -> int:
    """复刻 ``start()`` 成功后、距上一帧已过 ``silent_ms`` 的状态。"""
    src._running = True
    src._gen += 1
    src._last_frame_ms = mono_ms() - silent_ms
    src._fault_reported = False
    return src._gen


def _src(**kwargs) -> SoundDeviceSource:
    kwargs.setdefault("bus", FakeBus())
    kwargs.setdefault("fault_timeout_ms", 200)
    return SoundDeviceSource(**kwargs)


def test_default_timeout_is_documented_and_far_below_t14_budget() -> None:
    assert DEFAULT_FAULT_TIMEOUT_MS == 2000
    # 流程说明 T14 的恢复预算是 30 秒：阈值必须留出足够余量
    assert DEFAULT_FAULT_TIMEOUT_MS * 10 < 30_000


def test_stall_posts_exactly_one_audio_in_fault() -> None:
    src = _src()
    gen = _begin_session(src, silent_ms=10_000)

    assert src._watchdog_tick(gen) is True
    faults = src.bus.faults()
    assert len(faults) == 1
    fault = faults[0]
    assert fault.device == "audio_in"
    assert "未收到音频帧 200ms" in fault.detail

    # 同一次断流不刷屏：连续判定仍然是同一条
    assert src._watchdog_tick(gen) is True
    assert len(src.bus.faults()) == 1
    src.stop()


def test_fresh_frames_do_not_report_a_fault() -> None:
    src = _src()
    gen = _begin_session(src, silent_ms=0)
    assert src._watchdog_tick(gen) is True
    assert src.bus.faults() == []
    src.stop()


def test_stop_prevents_fault_for_the_old_session() -> None:
    src = _src()
    gen = _begin_session(src, silent_ms=10_000)
    src.stop()  # 用户主动停止：即使已远超阈值也不得报故障

    assert src._watchdog_tick(gen) is False
    assert src.bus.faults() == []


def test_stale_generation_does_not_report() -> None:
    src = _src()
    old_gen = _begin_session(src, silent_ms=10_000)
    # 新一次 start() 会让代次前进；旧会话的看门狗不得再上报
    src._gen += 1
    src._last_frame_ms = mono_ms() - 10_000

    assert src._watchdog_tick(old_gen) is False
    assert src.bus.faults() == []
    src.stop()


def test_no_repeat_between_two_stalls_until_a_frame_arrives() -> None:
    src = _src()
    gen = _begin_session(src, silent_ms=10_000)
    src._watchdog_tick(gen)
    assert len(src.bus.faults()) == 1

    # 第二段断流（期间没有恢复过）不再上报
    src._last_frame_ms = mono_ms() - 10_000
    src._watchdog_tick(gen)
    assert len(src.bus.faults()) == 1

    # 收到新帧 = 恢复；之后再次断流允许再报一次
    src._note_frame()
    src._last_frame_ms = mono_ms() - 10_000
    src._watchdog_tick(gen)
    assert len(src.bus.faults()) == 2
    src.stop()


def test_zero_timeout_disables_the_reporter() -> None:
    src = _src(fault_timeout_ms=0)
    gen = _begin_session(src, silent_ms=10_000)
    assert src._watchdog_tick(gen) is False
    assert src.bus.faults() == []
    src.stop()


def test_no_bus_means_no_fault_and_no_crash() -> None:
    src = _src(bus=None)
    gen = _begin_session(src, silent_ms=10_000)
    # 没给总线时仍要能安全判定（只是没人收事件）
    assert src._watchdog_tick(gen) is True
    src.stop()

# ---------------------------------------------------------------------------
# stop() -> start() 串扰：旧代次不得让新会话"第二次启动不再报"
# ---------------------------------------------------------------------------
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402


class _FakeStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


class _FakeSoundDevice(types.ModuleType):
    """替身 PortAudio：能开流，但永不回调 ⇒ 天然"断流"，无需真实设备。"""

    def __init__(self) -> None:
        super().__init__("sounddevice")
        self.streams: list[_FakeStream] = []

    def InputStream(self, **kwargs) -> _FakeStream:  # noqa: N802 - 模拟第三方 API
        stream = _FakeStream(**kwargs)
        self.streams.append(stream)
        return stream


def _wait_until(pred, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_stop_then_start_new_session_reports_again(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSoundDevice())
    bus = FakeBus()
    src = SoundDeviceSource(bus=bus, fault_timeout_ms=60)

    src.start(lambda frame: None)
    assert _wait_until(lambda: len(bus.faults()) >= 1), "首次会话没有上报断流"
    assert len(bus.faults()) == 1

    src.stop()
    time.sleep(0.2)  # 远超阈值：旧看门狗不得继续往总线里灌故障
    assert len(bus.faults()) == 1, "stop() 之后旧会话仍在报故障"

    src.start(lambda frame: None)  # 第二次启动 = 新代次
    assert _wait_until(lambda: len(bus.faults()) >= 2), "stop() 后再 start()，新会话没能再报"
    assert len(bus.faults()) == 2
    assert src.bus is bus
    src.stop()
