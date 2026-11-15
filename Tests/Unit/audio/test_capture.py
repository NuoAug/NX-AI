"""SoundDeviceSource 单测：错误路径与（若有设备时）打开/关闭冒烟。"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("sounddevice")

from aieyes.audio.capture import SoundDeviceSource  # noqa: E402
from aieyes.errors import AudioDeviceError  # noqa: E402


def test_stop_before_start_is_noop():
    src = SoundDeviceSource()
    src.stop()  # 不应抛异常
    assert src.is_running is False
    assert src.dropped_frames == 0


def test_invalid_device_raises_audio_device_error():
    src = SoundDeviceSource(device="__aieyes_no_such_device__")
    with pytest.raises(AudioDeviceError):
        src.start(lambda frame: None)
    src.stop()


def test_default_input_device_smoke():
    src = SoundDeviceSource()
    frames = []
    try:
        src.start(frames.append)
    except AudioDeviceError as exc:
        pytest.skip(f"当前目标板无可用的默认采集设备: {exc}")
    try:
        time.sleep(0.3)
    finally:
        src.stop()
    assert src.is_running is False
    # 有设备时应当收到至少一帧合法的 20ms/16k 音频（Tegra APE 无声卡时为静音）
    for frame in frames:
        assert frame.samples.shape == (320,)
        assert frame.t_start_ms >= 0
