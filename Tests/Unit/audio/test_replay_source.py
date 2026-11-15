"""ReplaySource 单测：帧数、时间戳、间隔、实时节拍、提前停止。"""

from __future__ import annotations

import time

import numpy as np

from audio_helpers import tone

from aieyes.audio.replay_source import ReplaySource, build_replay_playlist
from aieyes.interfaces import SAMPLE_RATE


def test_frame_count_and_timestamps():
    a = tone(100)
    b = tone(40, freq=880.0)
    src = ReplaySource([a, b], realtime=False, gap_ms=100)
    got = []
    src.start(got.append)
    assert src.wait(5.0) is True
    assert src.is_running is False
    assert src.frames_sent == src.total_frames
    assert len(got) == src.total_frames
    # 0.1s + 0.1s gap + 0.04s → 帧数 = 5 + 5 + 2
    assert src.total_frames == 12
    assert [f.seq for f in got] == list(range(12))
    # t_start_ms 与 capture 一致：以 clock.mono_ms 为基准的绝对时间，帧间隔严格 20ms
    t0 = got[0].t_start_ms
    assert t0 > 0
    assert [f.t_start_ms - t0 for f in got] == [i * 20 for i in range(12)]
    assert all(f.samples.shape == (320,) for f in got)


def test_gap_is_silence():
    src = ReplaySource([tone(20), tone(20)], realtime=False, gap_ms=100)
    got = []
    src.start(got.append)
    src.wait(5.0)
    silence_frames = got[1:6]
    for frame in silence_frames:
        assert float(np.max(np.abs(frame.samples))) == 0.0


def test_realtime_pacing():
    src = ReplaySource([tone(200)], realtime=True)
    got = []
    start = time.monotonic()
    src.start(got.append)
    assert src.wait(5.0) is True
    elapsed = time.monotonic() - start
    assert elapsed >= 0.15
    assert len(got) == 10


def test_stop_early():
    src = ReplaySource([tone(2000)], realtime=True)
    src.start(lambda frame: None)
    time.sleep(0.05)
    src.stop()
    src.wait(2.0)
    assert src.frames_sent < src.total_frames


def test_build_replay_playlist():
    a = tone(100)
    b = tone(50)
    joined = build_replay_playlist([a, b], gap_ms=200)
    assert joined.shape == (len(a) + len(b) + int(16000 * 0.2),)
    assert joined.dtype == np.float32
