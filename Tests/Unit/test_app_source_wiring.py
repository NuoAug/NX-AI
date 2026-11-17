"""`app.py` 数据源接线：``bus`` 必须真的传到采集源（否则 audio_in 故障在生产里发不出）。

只构造对象、不开音频流：`SoundDeviceSource.__init__` 不 import sounddevice。
`build_engine` 用 spy 截住 `_make_source`，避免装配整条业务链。
"""

from __future__ import annotations

import pytest

import aieyes.app as app
from aieyes.config import AppConfig
from aieyes.events import EventBus


def test_make_source_passes_bus_into_the_capture_source() -> None:
    bus = EventBus.__new__(EventBus)  # 只为身份比较，不启动事件循环
    src = app._make_source(AppConfig(), bus=bus)
    assert getattr(src, "bus", "缺字段") is bus
    assert int(getattr(src, "fault_timeout_ms", 0)) > 0


def test_make_source_without_bus_keeps_the_reporter_disabled() -> None:
    src = app._make_source(AppConfig())
    assert getattr(src, "bus", "缺字段") is None


def test_build_engine_wires_the_bus_it_creates(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    bus = EventBus.__new__(EventBus)

    def spy(cfg, *, bus=None):  # noqa: ANN001 - 与 _make_source 同签名
        captured["bus"] = bus
        raise RuntimeError("stop-after-source")

    monkeypatch.setattr(app, "_make_source", spy)
    with pytest.raises(RuntimeError, match="stop-after-source"):
        app.build_engine(
            AppConfig(),
            bus=bus,
            chat=object(),
            scene=object(),
            identity=object(),
            sink=object(),
            tts=object(),
        )
    assert captured["bus"] is bus, "build_engine 必须把自己用的 bus 交给采集源"
