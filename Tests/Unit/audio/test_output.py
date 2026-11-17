"""AudioSink 单测：用注入的假输出流验证播放/游标/取消/抢占/事件。"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from aieyes.audio.output import (
    STOP_COMPLETED,
    STOP_STALLED,
    STOP_USER_CANCEL,
    AudioSink,
    PlaybackStopped,
)
from aieyes.events import DeviceFault, PlaybackFinished, PlaybackStarted
from aieyes.interfaces import SAMPLE_RATE

FRAME = 320


class FakeStream:
    """按块回调的假 PortAudio OutputStream（比实时快，便于单测）。"""

    def __init__(self, callback, samplerate: int, blocksize: int) -> None:
        self.callback = callback
        self.rate = samplerate
        self.blocksize = blocksize
        self.latency = 0.0
        self.started = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.calls = 0

    def start(self) -> None:
        self.started = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.001):
            out = np.zeros((self.blocksize, 1), dtype=np.float32)
            self.calls += 1
            try:
                self.callback(out, self.blocksize, None, None)
            except Exception:  # pragma: no cover
                break

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)

    def close(self) -> None:
        self.stop()


def _factory(callback, samplerate, blocksize):
    return FakeStream(callback, samplerate, blocksize)


class FakeBus:
    def __init__(self) -> None:
        self.events = []

    def post(self, event) -> None:
        self.events.append(event)


def _tone(ms: int, freq: float = 440.0) -> np.ndarray:
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture()
def sink():
    s = AudioSink(stream_factory=_factory)
    yield s
    s.stop()


def test_play_completes_and_cursor_reaches_total(sink):
    handle = sink.play(_tone(300), piece_id="seg-1", text="你好")
    assert handle.piece_id == "seg-1"
    assert handle.total_ms == 300
    assert sink.is_playing
    assert handle.wait(2.0) is True
    assert sink.is_playing is False
    assert sink.played_ms() == 300
    stats = sink.stats()
    assert stats["pieces_started"] == 1
    assert stats["pieces_finished"] == 1


def test_cursor_monotonic(sink):
    sink.play(_tone(2000))
    seen = []
    for _ in range(20):
        seen.append(sink.played_ms())
        time.sleep(0.002)
    assert all(0 <= ms <= 2000 for ms in seen)
    assert seen == sorted(seen)


def test_cancel_stops_playback(sink):
    handle = sink.play(_tone(2000), piece_id="cancel-me")
    time.sleep(0.02)
    assert sink.cancel() is True
    assert handle.wait(1.0) is True
    assert sink.is_playing is False
    assert sink.stats()["cancelled"] == 1
    assert sink.played_ms() < 2000


def test_cancel_wrong_piece_id_returns_false(sink):
    sink.play(_tone(1000), piece_id="a")
    assert sink.cancel("b") is False
    assert sink.is_playing is True


def test_new_play_preempts_old(sink):
    old = sink.play(_tone(3000), piece_id="old")
    new = sink.play(_tone(200), piece_id="new")
    assert old.wait(1.0) is True
    assert sink.stats()["preempted"] == 1
    assert new.wait(2.0) is True
    assert sink.stats()["pieces_finished"] == 1


def test_reference_frame_is_latest_device_chunk(sink):
    sink.play(_tone(2000))
    ref = None
    deadline = time.monotonic() + 1.0
    while ref is None and time.monotonic() < deadline:
        ref = sink.reference_frame()
        time.sleep(0.002)
    assert ref is not None
    assert ref.shape == (FRAME,)
    assert ref.dtype == np.float32
    assert sink.cancel() is True
    assert sink.reference_frame() is None


def test_empty_piece_finishes_immediately(sink):
    handle = sink.play(np.zeros(0, dtype=np.float32))
    assert handle.wait(0.2) is True
    assert sink.played_ms() == 0
    assert sink.is_playing is False


def test_events_posted_to_bus():
    bus = FakeBus()
    sink = AudioSink(stream_factory=_factory, bus=bus)
    try:
        sink.play(_tone(100), piece_id="evt", text="测试")
        time.sleep(0.05)
        assert any(isinstance(ev, PlaybackStarted) for ev in bus.events)
        solved = None
        for _ in range(200):
            solved = [ev for ev in bus.events if isinstance(ev, PlaybackFinished)]
            if solved:
                break
            time.sleep(0.005)
        assert solved, "未收到 PlaybackFinished"
        ev = solved[0]
        assert ev.piece_id == "evt"
        assert ev.played_ms == 100
        assert ev.total_ms == 100
        assert ev.cancelled is False
    finally:
        sink.stop()



class StallingStream:
    """回调先正常跑，收到 ``stall()`` 后不再回调——模拟设备中途失联。"""

    def __init__(self, callback, samplerate: int, blocksize: int) -> None:
        self.callback = callback
        self.rate = samplerate
        self.blocksize = blocksize
        self.latency = 0.0
        self.started = False
        self.calls = 0
        self._stop = threading.Event()
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.started = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stall(self) -> None:
        self._halt.set()

    def _run(self) -> None:
        while not self._stop.wait(0.002):
            if self._halt.is_set():
                continue
            out = np.zeros((self.blocksize, 1), dtype=np.float32)
            self.calls += 1
            try:
                self.callback(out, self.blocksize, None, None)
            except Exception:  # pragma: no cover
                break

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)

    def close(self) -> None:
        self.stop()


def test_stalled_device_ends_piece_as_interrupted():
    bus = FakeBus()
    holders = []

    def factory(callback, samplerate, blocksize):
        stream = StallingStream(callback, samplerate, blocksize)
        holders.append(stream)
        return stream

    sink = AudioSink(stream_factory=factory, bus=bus, stall_timeout_ms=150)
    try:
        handle = sink.play(_tone(10000), piece_id="stall-me", text="长播报")
        time.sleep(0.05)
        holders[0].stall()
        assert handle.wait(3.0) is False, "设备失联必须按中断结束，而不是永远挂起"
        assert sink.is_playing is False
        assert sink.stats()["stalled"] == 1
        assert sink.stats()["pieces_finished"] == 0
        finished = [e for e in bus.events if isinstance(e, PlaybackFinished)]
        assert finished and finished[-1].cancelled is True
        # 第一层：事件流带独立原因，不能与“用户主动停止”同构
        assert isinstance(finished[-1], PlaybackStopped)
        assert finished[-1].stop_reason == STOP_STALLED
        # 第二层：句柄可直接读到原因
        assert handle.stop_reason == STOP_STALLED
        # 第三层：设备故障走 T14 通道（engine 记 fault.audio_out / 播报后待机）
        faults = [e for e in bus.events if isinstance(e, DeviceFault)]
        assert faults and faults[-1].device == "audio_out"
        assert faults[-1].detail.startswith("playback_stalled")
        # 卡死之后 sink 仍可继续用（下一次播报重新开始）
        holders[0]._halt.clear()
        again = sink.play(_tone(120), piece_id="after-stall")
        assert again.wait(2.0) is True
        assert sink.stats()["pieces_finished"] == 1
    finally:
        sink.stop()


def test_watchdog_can_be_disabled():
    holders = []

    def factory(callback, samplerate, blocksize):
        stream = StallingStream(callback, samplerate, blocksize)
        holders.append(stream)
        return stream

    sink = AudioSink(stream_factory=factory, stall_timeout_ms=0)
    try:
        handle = sink.play(_tone(10000), piece_id="no-watchdog")
        time.sleep(0.05)
        holders[0].stall()
        assert handle.wait(0.3) is False
        assert sink.stats()["stalled"] == 0
        assert sink.is_playing is True
    finally:
        sink.stop()
        assert sink.is_playing is False

def test_tts_rate_input_is_resampled():
    sink = AudioSink(stream_factory=_factory)
    try:
        pcm22k = _tone(400)  # 形状不重要，只验证重采样后时长按 16k 计
        handle = sink.play(pcm22k, sample_rate=22050)
        assert abs(handle.total_ms - int(len(pcm22k) * 1000 / 22050)) <= 1
        assert handle.wait(2.0) is True
    finally:
        sink.stop()


def test_user_cancel_and_stall_are_distinguishable():
    """用户按停止 ≠ 播放卡死：事件原因、句柄、DeviceFault 三处都要不同。"""
    results = {}
    for case in ("stall", "user_cancel"):
        bus = FakeBus()
        holders = []

        def factory(callback, samplerate, blocksize, _holders=holders):
            stream = StallingStream(callback, samplerate, blocksize)
            _holders.append(stream)
            return stream

        sink = AudioSink(stream_factory=factory, bus=bus, stall_timeout_ms=150)
        try:
            handle = sink.play(_tone(10000), piece_id=f"{case}-piece")
            time.sleep(0.05)
            if case == "stall":
                holders[0].stall()
            else:
                assert sink.cancel(handle.piece_id) is True
            ended = handle.wait(3.0)
            stopped = [e for e in bus.events if isinstance(e, PlaybackFinished)][-1]
            results[case] = {
                "wait": ended,
                "stop_reason": stopped.stop_reason,
                "handle_reason": handle.stop_reason,
                "fault": any(isinstance(e, DeviceFault) for e in bus.events),
                "stats": sink.stats(),
            }
        finally:
            sink.stop()

    assert results["stall"]["stop_reason"] == STOP_STALLED
    assert results["user_cancel"]["stop_reason"] == STOP_USER_CANCEL
    assert results["stall"]["handle_reason"] == STOP_STALLED
    assert results["user_cancel"]["handle_reason"] == STOP_USER_CANCEL
    assert results["stall"]["fault"] is True
    assert results["user_cancel"]["fault"] is False
    assert results["user_cancel"]["wait"] is True
    assert results["stall"]["stats"]["stalled"] == 1
    assert results["user_cancel"]["stats"]["stalled"] == 0
    assert results["user_cancel"]["stats"]["cancelled"] == 1


def test_natural_completion_reports_completed():
    bus = FakeBus()

    def factory(callback, samplerate, blocksize):
        return StallingStream(callback, samplerate, blocksize)

    sink = AudioSink(stream_factory=factory, bus=bus)
    try:
        handle = sink.play(_tone(100), piece_id="natural")
        assert handle.wait(2.0) is True
        stopped = [e for e in bus.events if isinstance(e, PlaybackFinished)][-1]
        assert stopped.cancelled is False
        assert stopped.stop_reason == STOP_COMPLETED
        assert handle.stop_reason == STOP_COMPLETED
        assert not [e for e in bus.events if isinstance(e, DeviceFault)]
    finally:
        sink.stop()
