"""阶段 6 合同对接：真实 ContractServer + FixtureClient 驱动 InteractionEngine。

覆盖：
1. 握手：交互侧 ContractLink 出站 capabilities.get，夹具应答后记录 contract.handshake；
2. interaction.status.get 返回模式与队列；
3. 普通播报期间收到 emergency guidance.event → 抢占停声、重播引导、playback.status 事件可见；
4. output.cancel(scope=ordinary_turn) 停声并返回 device_stopped。

全部使用 synthetic 证据模式，不依赖麦克风/扬声器/真实主控。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from aieyes.clock import CLOCK_ID
from aieyes.config import AppConfig
from aieyes.contract.fixture_client import FixtureClient
from aieyes.events import EventBus
from aieyes.interfaces import TtsAudio
from aieyes.interaction.contract_link import ContractLink
from aieyes.interaction.engine import InteractionEngine, InteractionState
from aieyes.logging_setup import JsonlEvidenceLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "Config" / "target.json"

EMERGENCY_ID = "ev-emergency-1"


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _cfg(port: int) -> AppConfig:
    cfg = AppConfig.from_file(CONFIG)
    cfg.contract.enabled = True
    cfg.contract.host = "127.0.0.1"
    cfg.contract.port = port
    cfg.evidence_mode = "synthetic"
    return cfg


def _records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


async def _wait_for(predicate, timeout_s: float = 8.0, interval_s: float = 0.02):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(interval_s)
    raise AssertionError("等待超时")


class FakeHandle:
    def __init__(self, total_ms: int) -> None:
        self.total_ms = int(total_ms)
        self._done = threading.Event()
        self._cancelled = False
        threading.Timer(self.total_ms / 1000.0, self._done.set).start()

    def cancel(self) -> None:
        self._cancelled = True
        self._done.set()

    def wait(self, timeout=None) -> bool:
        self._done.wait(timeout)
        return not self._cancelled


class FakeSink:
    """长度可控的假播放出口；is_playing 为真实状态。"""

    def __init__(self, total_ms: int = 700) -> None:
        self.total_ms = int(total_ms)
        self.playing: FakeHandle | None = None
        self.cancel_calls = 0
        self.started = False

    def start(self) -> None:
        self.started = True

    def play(self, pcm, sample_rate, *, piece_id="", text=""):
        handle = FakeHandle(self.total_ms)
        self.playing = handle
        return handle

    def cancel(self) -> None:
        self.cancel_calls += 1
        if self.playing is not None:
            self.playing.cancel()

    def stop(self) -> None:
        self.playing = None

    def close(self) -> None:
        self.playing = None

    @property
    def is_playing(self) -> bool:
        return self.playing is not None and not self.playing._done.is_set()

    def played_ms(self) -> int:
        return 0

    def reference_frame(self):
        return None


class FakeFrontend:
    def __init__(self) -> None:
        self.started = False
        self.listening = False
        self.wake_enabled = True

    def start(self) -> None:
        self.started = True
        self.listening = True

    def stop(self) -> None:
        self.started = False
        self.listening = False

    def set_listening(self, on: bool) -> None:
        self.listening = bool(on)

    def set_wake_enabled(self, on: bool) -> None:
        self.wake_enabled = bool(on)


class FakeTts:
    def prewarm(self, texts) -> None:
        return None

    def synth(self, text: str) -> TtsAudio:
        duration_ms = max(300.0, len(text) * 45.0)
        samples = int(duration_ms / 1000.0 * 16000)
        return TtsAudio(
            pcm=np.zeros(samples, dtype=np.float32),
            sample_rate=16000,
            text=text,
        )

    def clear_cache(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeChat:
    def __init__(self) -> None:
        self.turns: list = []

    def ask(self, text, **kwargs):  # pragma: no cover - 本用例不触发聊天
        return "好的。"

    def reset(self) -> None:
        self.turns = []


@pytest.mark.slow
def test_contract_link_handshake_status_guidance_and_cancel(tmp_path):
    asyncio.run(_scenario(tmp_path))


async def _scenario(tmp_path):
    port = _free_port()
    cfg = _cfg(port)
    evidence_path = tmp_path / "contract.jsonl"
    evidence = JsonlEvidenceLogger(evidence_path)
    bus = EventBus()
    sink = FakeSink(total_ms=700)
    engine = InteractionEngine(
        cfg=cfg,
        bus=bus,
        frontend=FakeFrontend(),
        sink=sink,
        tts=FakeTts(),
        chat=FakeChat(),
        scene=None,
        identity=None,
        evidence=evidence,
    )
    link = ContractLink(cfg, engine=engine, record=engine._record)
    engine.attach_contract(link)
    client = FixtureClient("127.0.0.1", port, clock_id=CLOCK_ID, evidence_mode="synthetic")

    sink.start()
    try:
        await engine.start()
        await link.start()
        await asyncio.wait_for(client.connect(), timeout=8)

        # 1) 握手：交互侧出站 capabilities.get 由夹具应答。
        await _wait_for(lambda: link.peer is not None and link.peer_clock_ok)
        handshakes = [r for r in _records(evidence_path) if r.get("event") == "handshake"]
        assert handshakes and handshakes[-1]["clock_ok"] is True
        assert "interaction.v0.1" in handshakes[-1]["versions"]
        assert any(r.type == "capabilities.get" for r in client.received_requests)

        # 2) status：空闲态与队列字段。
        status = await client.interaction_status_get()
        assert status.ok, status.error
        assert status.data["mode"] == "idle"
        assert "image_tasks" in status.data["queues"]
        assert status.data["memory"]["method"] == "proc_status"

        # 3) 普通播报期间收到紧急引导：抢占停声后播引导，播放状态事件可见。
        speak_task = asyncio.create_task(
            engine._speak(
                "前方画面里有一张桌子，桌上放着一盏台灯，旁边还有一把椅子，你可以先坐下慢慢看。",
                kind="answer",
            )
        )
        await _wait_for(lambda: engine.state is InteractionState.SPEAKING and sink.is_playing)
        cancel_before = sink.cancel_calls
        await client.send_guidance(
            event_id=EMERGENCY_ID,
            priority="emergency",
            interrupt_policy="preempt",
            text="前方有台阶，请先停下来。",
            valid_for_ms=5000,
        )
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(speak_task, timeout=10)
        await _wait_for(
            lambda: any(
                r.get("category") == "guidance" and r.get("event") == "preempt"
                for r in _records(evidence_path)
            )
        )
        assert sink.cancel_calls > cancel_before
        await _wait_for(
            lambda: any(
                r.get("category") == "guidance" and r.get("event") == "done"
                for r in _records(evidence_path)
            )
        )

        def _states_for(event_id: str) -> set[str]:
            states: set[str] = set()
            for msg in client.events:
                if msg.type != "playback.status":
                    continue
                data = msg.data or {}
                if data.get("source_event_id") == event_id:
                    states.add(str(data.get("state")))
            return states

        await _wait_for(lambda: {"queued", "started", "completed"} <= _states_for(EMERGENCY_ID))

        # 4) output.cancel(ordinary_turn)：停声并返回 device_stopped=True。
        speak_task2 = asyncio.create_task(
            engine._speak("这是一段新的普通回答，用来验证主控可以主动取消播报。", kind="answer")
        )
        await _wait_for(lambda: engine.state is InteractionState.SPEAKING and sink.is_playing)
        cancelled = await client.output_cancel(
            scope="ordinary_turn",
            reason="user_stop",
            target_request_id="req-test-1",
        )
        assert cancelled.ok, cancelled.error
        assert cancelled.data["device_stopped"] is True
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(speak_task2, timeout=10)
        assert any(r.get("event") == "cancel" for r in _records(evidence_path) if r.get("category") == "contract")
    finally:
        await client.close()
        await link.stop()
        await engine.stop()
        sink.close()
