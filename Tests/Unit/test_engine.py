"""InteractionEngine 状态机单测：全部用假设备，验证阶段 1/2 的软件行为。

覆盖：唤醒问候、唤醒首声时延、回答播报、插话停声、停止指令、T01 超时回待机。
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np

from aieyes.chat import ChatSession
from aieyes.clock import mono_ms
from aieyes.config import AppConfig
from aieyes.events import DeviceFault, EventBus
from aieyes.interfaces import SAMPLE_RATE, ImageFrame, PlaybackHandle, TtsAudio, Utterance
from aieyes.interaction import InteractionEngine, InteractionState

TIMEOUT = 5.0


class FakeFrontend:
    def __init__(self) -> None:
        self.on_wake = None
        self.on_speech_start = None
        self.on_partial = None
        self.on_utterance = None
        self.on_level = None
        self.listening = False
        self.wake_enabled = True
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def set_listening(self, enabled: bool) -> None:
        self.listening = enabled

    def set_wake_enabled(self, enabled: bool) -> None:
        self.wake_enabled = enabled

    def flush(self) -> None:
        pass


class FakeHandle(PlaybackHandle):
    def __init__(self, piece_id: str, total_ms: int, text: str, sink: "FakeSink") -> None:
        super().__init__(piece_id=piece_id, total_ms=total_ms, text=text)
        self._sink = sink
        self._done = threading.Event()
        self._cancelled = False
        delay_s = max(total_ms, 1) / 1000.0 / max(sink.speed, 0.001)
        timer = threading.Timer(delay_s, self._finish)
        timer.daemon = True
        sink._timers.append(timer)
        timer.start()

    def _finish(self) -> None:
        self._sink._complete(self)

    def cancel(self) -> None:
        self._cancelled = True
        self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        finished = self._done.wait(None if timeout is None else timeout)
        return bool(finished) and not self._cancelled


class FakeSink:
    """假播放出口：按 speed 倍速完成；记录播放与取消，is_playing 反映真实状态。"""

    def __init__(self, speed: float = 50.0) -> None:
        self.speed = speed
        self.playing: FakeHandle | None = None
        self.played: list[tuple[str, str]] = []
        self.cancel_calls = 0
        self._timers: list[threading.Timer] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        pass

    def play(self, pcm, sample_rate=SAMPLE_RATE, *, piece_id="", text="") -> FakeHandle:
        total_ms = int(round(len(pcm) * 1000 / sample_rate))
        handle = FakeHandle(piece_id, total_ms, text, self)
        with self._lock:
            self.playing = handle
            self.played.append((piece_id, text))
        return handle

    def _complete(self, handle: FakeHandle) -> None:
        with self._lock:
            handle._done.set()
            if self.playing is handle:
                self.playing = None

    def stop(self) -> None:
        if self.playing is not None:
            self.playing.cancel()

    def cancel(self) -> None:
        self.cancel_calls += 1
        with self._lock:
            playing = self.playing
            self.playing = None
        if playing is not None:
            playing.cancel()

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self.playing is not None

    def played_ms(self) -> int:
        return 0

    def reference_frame(self):
        return None

    def close(self) -> None:
        for timer in self._timers:
            timer.cancel()


class FakeTts:
    def __init__(self, audio_ms: int = 200) -> None:
        self.audio_ms = audio_ms
        self.calls: list[str] = []
        self.clear_calls = 0
        self.lock = threading.Lock()

    def synth(self, text: str) -> TtsAudio:
        with self.lock:
            self.calls.append(text)
        samples = int(SAMPLE_RATE * self.audio_ms / 1000)
        return TtsAudio(pcm=np.zeros(samples, dtype=np.float32), sample_rate=SAMPLE_RATE, text=text)

    def clear_cache(self) -> None:
        self.clear_calls += 1


class FakeVlm:
    def __init__(self, reply: str = "好的，我在听。", delay_s: float = 0.0) -> None:
        self.reply = reply
        self.delay_s = delay_s
        self.prompts: list[str] = []

    def complete(self, prompt, *, images=(), max_tokens=512, timeout_s=120.0, temperature=None):
        self.prompts.append(prompt)
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.reply


class FakeScene:
    """阶段 3/4 场景服务替身：暴露引擎需要的 ``answer`` 与 ``compare``。"""

    def __init__(
        self,
        text: str = "价格是 12.50 元。",
        *,
        error: Exception | None = None,
        delay_s: float = 0.0,
        provider: object | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.delay_s = delay_s
        self.questions: list[str] = []
        self.intent_hints: list[str | None] = []
        self.provider = provider
        self.compare_calls: list[tuple[object, object, str]] = []
        self.compare_error: Exception | None = None
        self.compare_delay_s = 0.0
        self.answer_frames: list[object | None] = []

    def answer(self, question: str, *, intent_hint: str | None = None, frame: object | None = None):
        self.questions.append(question)
        self.intent_hints.append(intent_hint)
        self.answer_frames.append(frame)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        from types import SimpleNamespace

        return SimpleNamespace(
            text=self.text,
            intent="price",
            frame_id="frame-1",
            degraded=(),
            source_tags=("ocr",),
        )

    def compare(self, a, b, question: str):
        self.compare_calls.append((a, b, question))
        if self.compare_delay_s:
            time.sleep(self.compare_delay_s)
        if self.compare_error is not None:
            raise self.compare_error
        from types import SimpleNamespace

        return SimpleNamespace(
            text="两边都是 12.50 元。",
            common_items=("price",),
            missing=(),
            a_frame_id=str(getattr(a, "frame_id", "")),
            b_frame_id=str(getattr(b, "frame_id", "")),
        )


class FakeProvider:
    """场景提供者替身：每次 acquire 返回一个新的帧 id。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def acquire(self, purpose: str = "vqa") -> ImageFrame:
        self.calls.append(purpose)
        return ImageFrame(
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            t_capture_ms=mono_ms(),
            frame_id=f"f-{len(self.calls)}",
            source="file",
        )


class FailingProvider(FakeProvider):
    """按第几次 acquire 注入失败，用于验证失败保留与有界重试。"""

    def __init__(self, fail_on: dict[int, Exception]) -> None:
        super().__init__()
        self.fail_on = dict(fail_on)

    def acquire(self, purpose: str = "vqa") -> ImageFrame:
        call = len(self.calls) + 1
        if call in self.fail_on:
            self.calls.append(purpose)
            raise self.fail_on[call]
        return super().acquire(purpose)


def _records(engine: InteractionEngine, name: str) -> list[dict]:
    return [r for r in engine.timings.records if r["name"] == name]


async def _wait_until(predicate, timeout: float = TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met within timeout")


def _make_engine(
    cfg: AppConfig | None = None,
    *,
    speed: float = 50.0,
    vlm: FakeVlm | None = None,
    scene: FakeScene | None = None,
    identity=None,
):
    cfg = cfg or AppConfig()
    bus = EventBus()
    frontend = FakeFrontend()
    sink = FakeSink(speed=speed)
    tts = FakeTts()
    chat = ChatSession(vlm or FakeVlm())
    engine = InteractionEngine(
        cfg=cfg,
        bus=bus,
        frontend=frontend,
        sink=sink,
        tts=tts,
        chat=chat,
        scene=scene,
        identity=identity,
    )
    return engine, frontend, sink, tts, chat


def test_wake_greeting_and_first_audio_latency() -> None:
    async def scenario() -> None:
        engine, frontend, sink, tts, _ = _make_engine()
        await engine.start()
        t_wake = mono_ms()
        frontend.on_wake("小冰", t_wake)
        await _wait_until(lambda: bool(_records(engine, "wake.greeting_done")))
        first = _records(engine, "latency.first_audio")
        assert first, "缺少首声时延记录"
        assert first[0]["speech_kind"] == "greeting"
        assert first[0]["ms"] < 500, f"唤醒首声 {first[0]['ms']}ms 超过 0.5s"
        assert engine.state is InteractionState.LISTENING
        assert frontend.listening is True
        assert sink.played and sink.played[0][1] == "我在"
        assert "我在" in tts.calls  # 预合成缓存过
        await engine.stop()

    asyncio.run(scenario())


def test_chat_turn_and_answer_playback() -> None:
    async def scenario() -> None:
        engine, frontend, sink, _tts, chat = _make_engine(vlm=FakeVlm("今天多云，适合散步。"))
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("今天天气怎么样？", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "answer.spoken")))
        assert chat.turns and chat.turns[-1].assistant == "今天多云，适合散步。"
        assert any("今天多云" in text for _pid, text in sink.played)
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        await engine.stop()

    asyncio.run(scenario())


def test_streaming_chat_speaks_first_phrase_early() -> None:
    """阶段 7：真实 VLM 的流式路径按短语边生成边播，首声不等整段。"""

    class FakeStreamVlm:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, prompt, *, images=(), max_tokens=512, timeout_s=120.0, temperature=None):
            self.prompts.append(prompt)
            return "今天多云，适合散步。"

        def complete_stream(self, prompt, *, images=(), max_tokens=None, timeout_s=None, temperature=None):
            self.prompts.append(prompt)
            yield "今天多云，"
            yield "适合散步。"
            yield "记得带伞。"

    async def scenario() -> None:
        engine, frontend, sink, _tts, chat = _make_engine(vlm=FakeStreamVlm())
        assert engine._stream_chat_available() is True
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("今天天气怎么样？", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "answer.spoken")))
        phrase_latencies = [
            r for r in _records(engine, "latency.first_audio") if r["speech_kind"] == "answer"
        ]
        assert len(phrase_latencies) >= 2, f"应按短语多次开口: {phrase_latencies}"
        assert any("今天多云" in text for _pid, text in sink.played)
        done = _records(engine, "llm.done")
        assert done and done[-1].get("streamed") is True
        assert chat.turns and chat.turns[-1].assistant == "今天多云，适合散步。记得带伞。"
        await engine.stop()

    asyncio.run(scenario())


def test_barge_in_stops_playback() -> None:
    async def scenario() -> None:
        cfg = AppConfig()
        engine, frontend, sink, _tts, _ = _make_engine(cfg, speed=1.0, vlm=FakeVlm("好的。" * 10))
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("讲个长故事", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: engine.state is InteractionState.SPEAKING and sink.is_playing)
        cancels_before = sink.cancel_calls
        t_barge = mono_ms()
        frontend.on_speech_start(t_barge)
        await _wait_until(lambda: bool(_records(engine, "barge_in.silenced")), timeout=2.0)
        silenced = _records(engine, "barge_in.silenced")
        assert silenced[0]["ms"] < 300, f"停声时延={silenced}"
        assert sink.cancel_calls > cancels_before, "插话必须让播放出口真正停声"
        await _wait_until(lambda: not sink.is_playing, timeout=2.0)
        await _wait_until(lambda: engine.state is InteractionState.LISTENING, timeout=2.0)
        await engine.stop()

    asyncio.run(scenario())


def test_guidance_single_stop_is_spoken_as_xiating() -> None:
    """停步提示统一说“停下”：主控只发单字“停”时，播报端补成两字并留证据。"""
    from types import SimpleNamespace

    async def scenario() -> None:
        engine, _frontend, sink, tts, _ = _make_engine()
        await engine.start()
        event = SimpleNamespace(
            data={
                "event_id": "evt-stop",
                "priority": "emergency",
                "text": "停",
                "valid_until_mono_ms": mono_ms() + 5000,
            },
            clock_id=None,
            evidence_mode="synthetic",
        )
        await engine.handle_guidance_event(event)
        await _wait_until(lambda: "停下。" in tts.calls)
        assert any(text == "停下。" for _pid, text in sink.played)
        assert not any(text.strip() == "停" for _pid, text in sink.played)
        rewritten = _records(engine, "guidance.event")
        assert rewritten and rewritten[-1]["text_rewritten"] is True
        await engine.stop()

    asyncio.run(scenario())


def test_stop_intent_silences_and_listens() -> None:
    async def scenario() -> None:
        engine, frontend, sink, _tts, _ = _make_engine(speed=1.0, vlm=FakeVlm("好的。" * 10))
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("讲个故事", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: engine.state is InteractionState.SPEAKING and sink.is_playing)
        frontend.on_utterance(Utterance("别说了", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "stop.silenced")))
        stop = _records(engine, "stop.silenced")[0]
        assert stop["ms"] < 300
        # 取消三时刻都要有原始证据（阶段 2 出口 b/c）
        for key in ("silenced_ms", "queue_ms", "played_ms", "old_task_ms"):
            assert key in stop, f"stop.silenced 缺少 {key}: {stop}"
        await _wait_until(lambda: bool(_records(engine, "t18.listen")))
        assert engine.state is InteractionState.LISTENING
        await engine.stop()

    asyncio.run(scenario())


def test_t01_timeout_returns_to_idle() -> None:
    async def scenario() -> None:
        cfg = AppConfig()
        cfg.timers.t01_remind_at_ms = 30
        cfg.timers.t01_initial_wait_ms = 80
        cfg.timers.t01_extend_ms = 60
        cfg.timers.t01_max_extend = 1
        engine, frontend, _sink, _tts, _ = _make_engine(cfg)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: bool(_records(engine, "t01.idle")), timeout=3.0)
        await _wait_until(lambda: engine.state is InteractionState.IDLE)
        assert frontend.listening is False
        assert frontend.wake_enabled is True
        remind = _records(engine, "t01.remind")
        assert remind, "缺少 7s 提醒记录"
        await engine.stop()

    asyncio.run(scenario())


def test_backchannel_after_barge_in_acks_without_model() -> None:
    async def scenario() -> None:
        vlm = FakeVlm("好的。" * 10)
        engine, frontend, sink, _tts, _ = _make_engine(speed=1.0, vlm=vlm)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("讲个长故事", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: engine.state is InteractionState.SPEAKING and sink.is_playing)
        assert len(vlm.prompts) == 1
        frontend.on_speech_start(mono_ms())  # 打断播报
        await _wait_until(lambda: bool(_records(engine, "barge_in.silenced")))
        frontend.on_utterance(Utterance("嗯", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "ack.spoken")))
        assert any(text == "嗯，我在。" for _pid, text in sink.played)
        assert len(vlm.prompts) == 1, "附和不应提交模型"
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        await engine.stop()

    asyncio.run(scenario())


def test_stale_answer_discarded_after_barge_in() -> None:
    async def scenario() -> None:
        vlm = FakeVlm("这是旧回答。", delay_s=0.5)
        engine, frontend, sink, _tts, _ = _make_engine(vlm=vlm)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("讲个笑话", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: engine.state is InteractionState.THINKING)
        frontend.on_speech_start(mono_ms())  # 生成期间用户插话
        await _wait_until(lambda: bool(_records(engine, "barge_in.silenced")))
        await asyncio.sleep(0.9)  # 等旧生成线程结束
        assert not any("这是旧回答" in text for _pid, text in sink.played), "旧结果必须丢弃"
        await engine.stop()

    asyncio.run(scenario())


def test_llm_failure_enters_t13_recovery_listen() -> None:
    async def scenario() -> None:
        cfg = AppConfig()
        cfg.timers.t13_warn_ms = 60
        cfg.timers.t13_resume_ms = 140
        engine, frontend, _sink, _tts, _ = _make_engine(cfg, vlm=FakeVlm(""))
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("今天天气怎么样", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "t13.listen")))
        await _wait_until(lambda: bool(_records(engine, "t13.remind")))
        await _wait_until(lambda: bool(_records(engine, "t13.timeout")), timeout=3.0)
        await _wait_until(lambda: engine.state is InteractionState.IDLE)
        await engine.stop()

    asyncio.run(scenario())


def test_empty_utterance_retries_then_idles() -> None:
    async def scenario() -> None:
        engine, frontend, sink, _tts, _ = _make_engine()
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        for expected in (1, 2, 3):
            frontend.on_utterance(Utterance("", mono_ms(), mono_ms(), mono_ms()))
            await _wait_until(lambda expected=expected: len(_records(engine, "t02.empty")) >= expected)
            if expected < 3:
                await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        await _wait_until(lambda: engine.state is InteractionState.IDLE)
        assert any("没听清" in text for _pid, text in sink.played)
        await engine.stop()

    asyncio.run(scenario())


def test_mute_intent_disables_microphone() -> None:
    async def scenario() -> None:
        engine, frontend, _sink, _tts, _ = _make_engine()
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("静音", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "mic.off")))
        assert frontend.listening is False
        assert frontend.wake_enabled is False
        assert engine._muted is True
        engine.set_microphone(True)
        assert frontend.listening is True
        assert frontend.wake_enabled is True
        assert engine._muted is False
        await engine.stop()

    asyncio.run(scenario())


def test_device_fault_recovers_frontend() -> None:
    async def scenario() -> None:
        engine, frontend, _sink, _tts, _ = _make_engine()
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        engine.bus.post(DeviceFault("audio_in", "test fault", mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "t14.recover")))
        recover = _records(engine, "t14.recover")[0]
        assert recover["ok"] is True, recover
        assert frontend.started is True, "恢复后前端应已重启"
        await engine.stop()

    asyncio.run(scenario())


def test_scene_intent_routes_to_scene_and_speaks() -> None:
    async def scenario() -> None:
        scene = FakeScene()
        engine, frontend, sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("这个多少钱", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "scene.answer")))
        assert scene.questions, "场景服务应收到问题"
        rec = _records(engine, "scene.answer")[0]
        assert rec["intent"] == "price" and rec["frame_id"] == "frame-1", rec
        assert rec["sources"] == "ocr" and rec["chars"] > 0, rec
        await _wait_until(lambda: any("12.50" in text for _pid, text in sink.played))
        await _wait_until(lambda: bool(_records(engine, "answer.spoken")))
        await engine.stop()

    asyncio.run(scenario())


def test_scene_intent_without_scene_degrades() -> None:
    async def scenario() -> None:
        engine, frontend, sink, _tts, _ = _make_engine()
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("看看前面有什么", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: any("还没接上" in text for _pid, text in sink.played))
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        await engine.stop()

    asyncio.run(scenario())


def test_scene_failure_speaks_fallback_and_recovers() -> None:
    async def scenario() -> None:
        scene = FakeScene(error=RuntimeError("camera unavailable"))
        engine, frontend, sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("看看前面有什么", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "scene.error")))
        await _wait_until(lambda: bool(_records(engine, "scene.failed")))
        await _wait_until(lambda: any("没看清" in text for _pid, text in sink.played))
        await _wait_until(lambda: bool(_records(engine, "t13.listen")))
        await engine.stop()

    asyncio.run(scenario())


def test_scene_result_discarded_after_barge_in() -> None:
    async def scenario() -> None:
        scene = FakeScene(text="价格是 12.50 元。", delay_s=0.4)
        engine, frontend, sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("这个多少钱", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: engine.state is InteractionState.THINKING)
        frontend.on_speech_start(mono_ms())  # 取图/分析期间用户插话
        await _wait_until(lambda: bool(_records(engine, "barge_in.silenced")))
        await asyncio.sleep(0.7)  # 等旧场景线程结束
        assert not any("12.50" in text for _pid, text in sink.played), "旧场景结果必须丢弃"
        assert _records(engine, "scene.stale_discarded"), "应记录 stale_discarded"
        await engine.stop()

    asyncio.run(scenario())


def _frame(frame_id: str) -> ImageFrame:
    return ImageFrame(rgb=np.zeros((4, 4, 3), dtype=np.uint8), t_capture_ms=mono_ms(), frame_id=frame_id, source="file")


def test_compare_intent_runs_compare_and_speaks() -> None:
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("比较一下这两瓶水的价格", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_first")))
        assert provider.calls == [], provider.calls
        assert scene.compare_calls == []
        await _wait_until(lambda: any("第一件" in text for _pid, text in sink.played))
        frontend.on_utterance(Utterance("继续", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_second")))
        assert provider.calls == ["vqa"], provider.calls
        assert scene.compare_calls == []
        await _wait_until(lambda: any("第二件" in text for _pid, text in sink.played))
        frontend.on_utterance(Utterance("继续", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.answer")))
        assert len(scene.compare_calls) == 1
        assert provider.calls == ["vqa", "vqa"], provider.calls
        await _wait_until(lambda: any("12.50" in text for _pid, text in sink.played))
        rec = _records(engine, "compare.answer")[0]
        assert rec["common"] == "price" and rec["a_frame"] == "f-1" and rec["b_frame"] == "f-2", rec
        task = engine.image_tasks.tasks[0]
        assert task.state == "answered" and engine.image_tasks.unfinished() == []
        await engine.stop()

    asyncio.run(scenario())


def test_compare_waiting_accepts_noisy_continue() -> None:
    """双图等待中 ASR 把“继续”识成“嗯继”也要恢复拍摄，不能掉进聊天。"""

    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, _sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("比较一下这两瓶水的价格", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_first")))
        frontend.on_utterance(Utterance("嗯继", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_second")))
        assert provider.calls == ["vqa"], provider.calls
        assert _records(engine, "compare.resume_ready"), _records(engine, "compare.resume_ready")
        assert scene.compare_calls == []
        await engine.stop()

    asyncio.run(scenario())


def test_compare_waiting_accepts_short_yes() -> None:
    """等待拍摄时说“好/好的/行”是最自然的确认，不能掉成聊天或“我在，请说”。"""

    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, _sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("帮我对比物品", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_first")))
        frontend.on_utterance(Utterance("好", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_second")))
        assert provider.calls == ["vqa"], provider.calls
        assert _records(engine, "compare.resume_ready"), _records(engine, "compare.resume_ready")
        assert not _records(engine, "chat"), "短确认不应提交聊天模型"
        await engine.stop()

    asyncio.run(scenario())


def test_compare_ready_and_reshoot_reject_negation() -> None:
    async def scenario() -> None:
        engine, _frontend, _sink, _tts, _ = _make_engine()
        assert engine._looks_like_compare_ready("好了")
        for text in ("好", "好的", "好嘞", "行", "可以", "嗯好"):
            assert engine._looks_like_compare_ready(text), text
        for text in ("不好", "好看", "好的，等一下", "不可以", "还没好"):
            assert not engine._looks_like_compare_ready(text), text
        assert not engine._looks_like_compare_ready("还没准备好")
        assert not engine._looks_like_compare_ready("等一下")
        task = engine.image_tasks.open_compare("比较这两件")
        engine._last_compare_task = task
        assert engine._looks_like_compare_repair("重拍第一件")
        assert not engine._looks_like_compare_repair("别重拍")
        await engine.stop()

    asyncio.run(scenario())


def test_compare_natural_expression_detection() -> None:
    """自然说法按阶段判意：准备完成 / 保持 / 暂停 / 放弃 / 补充 / 对调。"""
    async def scenario() -> None:
        engine, _frontend, _sink, _tts, _ = _make_engine()
        for text in ("换好了", "现在是另一件", "第二个放上去了", "来，看这个", "可以看了", "第二个好了"):
            assert engine._looks_like_compare_ready(text), text
        for text in ("等一下", "我还没换好", "先别拍", "这个还没摆正", "第二个还没好", "不要继续"):
            assert not engine._looks_like_compare_ready(text), text
            assert engine._looks_like_compare_hold(text), text
        assert engine._looks_like_compare_pause("先放着，聊点别的")
        assert not engine._looks_like_compare_abandon("先放着，聊点别的")
        assert not engine._looks_like_compare_hold("先不比价格，看日期")
        assert engine._looks_like_compare_refine("先不比价格，看日期")
        assert engine._looks_like_compare_abandon("算了，不比了")
        assert engine._looks_like_compare_refine("只比价格")
        assert engine._looks_like_compare_refine("容量也算上")
        assert engine._looks_like_compare_refine("看看哪个更划算")
        assert engine._looks_like_compare_refine("不比价格了，看日期")
        assert not engine._looks_like_compare_refine("比较一下这两瓶水的价格")
        assert engine._looks_like_compare_swap("不是第二个，是第一个")
        assert engine._looks_like_compare_swap("刚才说反了，左边那个才是第一件")
        await engine.stop()

    asyncio.run(scenario())


async def _run_compare_to_answer(engine, frontend, scene) -> None:
    frontend.on_utterance(Utterance("比较一下这两瓶水的价格", mono_ms(), mono_ms(), mono_ms()))
    await _wait_until(lambda: bool(_records(engine, "compare.wait_first")))
    frontend.on_utterance(Utterance("好了", mono_ms(), mono_ms(), mono_ms()))
    await _wait_until(lambda: bool(_records(engine, "compare.wait_second")))
    frontend.on_utterance(Utterance("好了", mono_ms(), mono_ms(), mono_ms()))
    await _wait_until(lambda: bool(_records(engine, "compare.answer")))


def test_compare_waiting_hold_never_captures_or_chats() -> None:
    """等待时说“还没换好/先别拍/等一下”只保持等待，不采集、不推进、不聊天。"""
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, _sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("比较一下这两瓶水的价格", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_first")))
        holds = ("等一下", "我还没换好", "先别拍", "这个还没摆正", "先看第二个，等一下，还是先别拍")
        for index, text in enumerate(holds, start=1):
            frontend.on_utterance(Utterance(text, mono_ms(), mono_ms(), mono_ms()))
            await _wait_until(lambda: len(_records(engine, "compare.hold")) == index)
            assert provider.calls == [], f"{text} 触发了取图: {provider.calls}"
            assert not _records(engine, "chat"), f"{text} 掉进了聊天"
            assert engine._last_compare_task.pause_reason == "waiting_first"
        frontend.on_utterance(Utterance("换好了", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_second")))
        assert provider.calls == ["vqa"], provider.calls
        await engine.stop()

    asyncio.run(scenario())


def test_compare_refine_reuses_pair_without_recapture() -> None:
    """答完后说“只比价格”：复用两件旧帧重新比较，不重拍。"""
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        await _run_compare_to_answer(engine, frontend, scene)
        calls_before = list(provider.calls)
        scene.compare_calls.clear()
        frontend.on_utterance(Utterance("只比价格", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: len(scene.compare_calls) == 1)
        assert provider.calls == calls_before, f"补充比较项却重拍了: {provider.calls}"
        frame_a, frame_b, question = scene.compare_calls[0]
        assert (frame_a.frame_id, frame_b.frame_id) == ("f-1", "f-2"), scene.compare_calls
        assert "价格" in question, question
        await _wait_until(lambda: len(_records(engine, "compare.answer")) == 2)
        assert engine.image_tasks.unfinished() == []
        await engine.stop()

    asyncio.run(scenario())


def test_compare_pause_keeps_task_and_ready_resumes() -> None:
    """“先放着”是暂停不是结束；随后“好了”接着拍第二件，不重拍第一件。"""
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, _sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("比较一下这两瓶水的价格", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_first")))
        frontend.on_utterance(Utterance("好了", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_second")))
        frontend.on_utterance(Utterance("先放着，聊点别的", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.pause")))
        task = engine._last_compare_task
        assert task is not None and task.pause_reason == "user_pause"
        assert not task.finished, "暂停不应结束任务"
        assert provider.calls == ["vqa"], provider.calls
        frontend.on_utterance(Utterance("好了", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.answer")))
        assert provider.calls == ["vqa", "vqa"], "恢复时应只补拍第二件"
        await engine.stop()

    asyncio.run(scenario())


def test_compare_abandon_then_continue_does_not_revive() -> None:
    """“算了，不比了”是结束；之后“继续”不能复活旧比较。"""
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("比较一下这两瓶水的价格", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_first")))
        frontend.on_utterance(Utterance("算了，不比了", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.abandon")))
        assert engine._last_compare_task is None
        assert all(task.finished for task in engine.image_tasks.tasks)
        frontend.on_utterance(Utterance("继续", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: any("没有要接着看的图" in text for _pid, text in sink.played))
        assert provider.calls == [], provider.calls
        await engine.stop()

    asyncio.run(scenario())


def test_compare_recall_answers_from_existing_frame_without_reset() -> None:
    """“第一件叫什么”用第一件的图回答，比较任务状态不变。"""
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, _sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        await _run_compare_to_answer(engine, frontend, scene)
        task = engine._last_compare_task
        assert task is not None
        state_before = task.state
        scene.answer_frames.clear()
        frontend.on_utterance(Utterance("第一件叫什么", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.recall")))
        assert scene.answer_frames and scene.answer_frames[-1] is task.frames["A"]
        assert task.state == state_before, "插问不应重置比较任务"
        assert len(scene.compare_calls) == 1, "插问不应重新比较"
        await engine.stop()

    asyncio.run(scenario())


def test_compare_recall_ambiguous_asks_one_question() -> None:
    """“刚才那个多少钱”指向不明时只问一件澄清，随后按位置回答。"""
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        await _run_compare_to_answer(engine, frontend, scene)
        scene.answer_frames.clear()
        frontend.on_utterance(Utterance("刚才那个多少钱", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.recall_clarify")))
        await _wait_until(lambda: any("第一件还是第二件" in text for _pid, text in sink.played))
        assert scene.answer_frames == [], "澄清前不应猜对象去回答"
        frontend.on_utterance(Utterance("第二件", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.recall")))
        task = engine._last_compare_task
        assert task is not None and scene.answer_frames[-1] is task.frames["B"]
        await engine.stop()

    asyncio.run(scenario())


def test_compare_swap_correction_reuses_frames() -> None:
    """“不是第二个，是第一个”：只交换 A/B 归属，不重拍。"""
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, _sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        await _run_compare_to_answer(engine, frontend, scene)
        assert scene.compare_calls[0][0].frame_id == "f-1"
        calls_before = list(provider.calls)
        scene.compare_calls.clear()
        frontend.on_utterance(Utterance("不是第二个，是第一个", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.swap")))
        await _wait_until(lambda: len(scene.compare_calls) == 1)
        assert provider.calls == calls_before, f"纠正不应重拍: {provider.calls}"
        frame_a, frame_b, _question = scene.compare_calls[0]
        assert (frame_a.frame_id, frame_b.frame_id) == ("f-2", "f-1"), scene.compare_calls
        await engine.stop()

    asyncio.run(scenario())


def test_compare_reference_correction_after_recall_does_not_swap() -> None:
    """先问第二件，随后“不是第二个，是第一个”是改指代，不是交换 A/B。"""
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, _sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        await _run_compare_to_answer(engine, frontend, scene)
        task = engine._last_compare_task
        assert task is not None
        scene.answer_frames.clear()
        frontend.on_utterance(Utterance("第二件是什么", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.recall")))
        assert scene.answer_frames[-1] is task.frames["B"]
        scene.answer_frames.clear()
        frontend.on_utterance(Utterance("不是第二个，是第一个", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.reference_corrected")))
        await _wait_until(lambda: len(scene.answer_frames) == 1)
        assert scene.answer_frames[-1] is task.frames["A"], "改指代后应回答第一件"
        assert not _records(engine, "compare.swap"), "不应交换 A/B"
        assert len(scene.compare_calls) == 1, "不应重新比较"
        assert task.frames["A"].frame_id == "f-1" and task.frames["B"].frame_id == "f-2"
        await engine.stop()

    asyncio.run(scenario())


def test_compare_reshoot_pending_accepts_bare_position() -> None:
    """问“重拍哪一件”后，用户只答“第二件”也应执行，不需要固定口令。"""
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, _sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        await _run_compare_to_answer(engine, frontend, scene)
        calls_before = list(provider.calls)
        scene.compare_calls.clear()
        frontend.on_utterance(Utterance("重拍", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.reshoot_clarify")))
        frontend.on_utterance(Utterance("第二件", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.reshoot_pending_answer")))
        await _wait_until(lambda: len(scene.compare_calls) == 1)
        assert provider.calls[len(calls_before):] == ["vqa"], provider.calls
        frame_a, frame_b, _question = scene.compare_calls[0]
        assert frame_a.frame_id == "f-1", "第一件被一起重拍了"
        assert frame_b.frame_id == "f-3", "第二件没有取新帧"
        await engine.stop()

    asyncio.run(scenario())


def test_compare_swap_without_evidence_asks_clarify() -> None:
    """没有上一轮比较归属证据时，不因“不是第二个，是第一个”就交换全部对象。"""
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("比较一下这两瓶水的价格", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_first")))
        before = list(provider.calls)
        frontend.on_utterance(Utterance("不是第二个，是第一个", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.swap_clarify")))
        assert provider.calls == before, "澄清前不应取图"
        assert not _records(engine, "compare.swap")
        await _wait_until(lambda: any("对调" in text for _pid, text in sink.played))
        await engine.stop()

    asyncio.run(scenario())


def test_compare_reshoot_after_answer_regrabs_only_named_position() -> None:
    """“第二张不清楚，重来一次”：复用第一件，只重拍第二件。"""
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, _sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        await _run_compare_to_answer(engine, frontend, scene)
        calls_before = list(provider.calls)
        scene.compare_calls.clear()
        frontend.on_utterance(Utterance("第二张不清楚，重来一次", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.reshoot_reuse_pair")))
        await _wait_until(lambda: len(scene.compare_calls) == 1)
        assert provider.calls[len(calls_before):] == ["vqa"], provider.calls
        frame_a, frame_b, _question = scene.compare_calls[0]
        assert frame_a.frame_id == "f-1", "第一件被一起重拍了"
        assert frame_b.frame_id == "f-3", f"第二件没有取新帧: {frame_b.frame_id}"
        await engine.stop()

    asyncio.run(scenario())



def test_compare_waiting_questions_and_negation_never_capture() -> None:
    """等待拍摄时，疑问/否定/不确定/引用不能被“准备/拍”等词抢先消费。"""
    cases = (
        "第一件是什么",
        "好了，不过先别拍",
        "别继续拍",
        "准备好了没有我不知道",
        "我刚才说“继续”，不是现在让你拍",
        "第一件是什么，放好了",
    )

    async def scenario() -> None:
        for text in cases:
            provider = FakeProvider()
            scene = FakeScene(provider=provider)
            engine, frontend, sink, _tts, _ = _make_engine(scene=scene)
            await engine.start()
            frontend.on_wake("小冰", mono_ms())
            await _wait_until(lambda: engine.state is InteractionState.LISTENING)
            frontend.on_utterance(Utterance("比较一下这两瓶水的价格", mono_ms(), mono_ms(), mono_ms()))
            await _wait_until(lambda: bool(_records(engine, "compare.wait_first")))
            before = list(provider.calls)
            frontend.on_utterance(Utterance(text, mono_ms(), mono_ms(), mono_ms()))
            await asyncio.sleep(0.2)
            assert provider.calls == before, f"{text!r} 被当成准备完成并取图: {provider.calls}"
            assert not _records(engine, "chat"), f"{text!r} 掉进聊天"
            assert sink.played, f"{text!r} 没有任何声音反馈"
            await engine.stop()

    asyncio.run(scenario())


def test_compare_waiting_self_correction_final_intent_wins() -> None:
    """同一句里先否定后确认：最终明确意愿是拍，才推进；先确认后否定则保持等待。"""
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, _sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("比较一下这两瓶水的价格", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_first")))

        frontend.on_utterance(Utterance("先别拍，现在好了，可以拍了", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_second")))
        assert provider.calls == ["vqa"], provider.calls
        await engine.stop()

    asyncio.run(scenario())


def test_compare_waiting_status_question_answers_from_state() -> None:
    """“你在等什么/现在到哪一步了”直接答实际状态，不取图、不交模型。"""
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("比较一下这两瓶水的价格", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_first")))
        before = list(provider.calls)
        frontend.on_utterance(Utterance("你在等什么", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.status")))
        assert provider.calls == before, provider.calls
        await _wait_until(lambda: any("第一件" in text for _pid, text in sink.played))
        assert not _records(engine, "chat")
        await engine.stop()

    asyncio.run(scenario())


def test_compare_second_capture_failure_keeps_first_item() -> None:
    """第二件拍摄失败：第一件保留，给一个调整动作，重试只补第二件。"""
    async def scenario() -> None:
        from aieyes.scene.errors import FrameQualityError
        from aieyes.vision.quality import QualityReport

        report = QualityReport(
            width=4, height=4, blur_score=0.0, brightness=10.0,
            size_ok=True, labels=("dark",),
        )
        provider = FailingProvider({2: FrameQualityError("画面太暗", report=report)})
        scene = FakeScene(provider=provider)
        engine, frontend, sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("比较一下这两瓶水的价格", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_first")))
        frontend.on_utterance(Utterance("好了", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_second")))
        frontend.on_utterance(Utterance("好了", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.capture_failed")))

        task = engine._last_compare_task
        assert task is not None
        assert task.frames.get("A") is not None, "第一件被一起丢掉了"
        assert task.frames.get("B") is None
        assert task.pause_reason == "capture_failed_second", task.pause_reason
        assert not task.finished
        await _wait_until(lambda: any("第一件还保留" in text for _pid, text in sink.played))
        spoken = "".join(text for _pid, text in sink.played)
        assert "画面太暗" in spoken and "亮一点" in spoken, spoken

        provider.fail_on.clear()
        scene.compare_calls.clear()
        frontend.on_utterance(Utterance("好了", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: len(scene.compare_calls) == 1)
        assert provider.calls == ["vqa", "vqa", "vqa"], provider.calls
        assert scene.compare_calls[0][0].frame_id == "f-1"
        assert scene.compare_calls[0][1].frame_id == "f-3", "重试应只补第二件"
        await engine.stop()

    asyncio.run(scenario())


def test_compare_first_capture_failure_can_retry() -> None:
    """第一件拍摄失败：任务保留，明确原因；重试只补第一件再等第二件。"""
    async def scenario() -> None:
        from aieyes.scene.errors import FrameAcquisitionError

        provider = FailingProvider({1: FrameAcquisitionError("没有取得画面")})
        scene = FakeScene(provider=provider)
        engine, frontend, sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("比较一下这两瓶水的价格", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_first")))
        frontend.on_utterance(Utterance("好了", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.capture_failed")))

        task = engine._last_compare_task
        assert task is not None
        assert not task.finished and task.frames.get("A") is None
        assert task.pause_reason == "capture_failed_first", task.pause_reason
        await _wait_until(lambda: any("第一件这次没拍到" in text for _pid, text in sink.played))

        provider.fail_on.clear()
        frontend.on_utterance(Utterance("好了", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_second")))
        assert provider.calls == ["vqa", "vqa"], provider.calls
        assert task.frames.get("A") is not None
        await engine.stop()

    asyncio.run(scenario())


def test_capture_failure_guidance_is_single_action() -> None:
    """质量标签映射成一个动作，不猜左右/距离。"""
    from aieyes.scene.errors import FrameQualityError
    from aieyes.vision.quality import QualityReport

    def guidance(labels):
        report = QualityReport(
            width=4, height=4, blur_score=0.0, brightness=10.0,
            size_ok=True, labels=tuple(labels),
        )
        return InteractionEngine._capture_failure_guidance(FrameQualityError("x", report=report))

    assert "亮一点" in guidance(("dark",))[1]
    assert "强光" in guidance(("bright",))[1]
    assert "放稳" in guidance(("blurry",))[1]
    assert "靠近镜头" in guidance(("too_small",))[1]
    assert "拿远一点" in guidance(("too_large",))[1]
    for _reason, action in (guidance(("dark",)), guidance(("blurry",))):
        assert "左" not in action and "右" not in action and "向前" not in action


def test_compare_waiting_global_controls_not_swallowed() -> None:
    """停止/清除/取消不能被“好了/拍/准备”等宽泛词吞掉。"""
    async def run_one(text: str, marker: str) -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, _sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("比较一下这两瓶水的价格", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.wait_first")))
        before = list(provider.calls)
        frontend.on_utterance(Utterance(text, mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, marker)))
        assert provider.calls == before, f"{text!r} 被当成准备完成: {provider.calls}"
        await engine.stop()

    async def scenario() -> None:
        await run_one("清除资料，好了", "materials.cleared")
        await run_one("停止，好了", "stop.silenced")
        await run_one("取消比较", "compare.abandon")

    asyncio.run(scenario())


def test_garbled_filler_asks_to_repeat() -> None:
    async def scenario() -> None:
        engine, frontend, sink, _tts, _ = _make_engine()
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("这个傻一点千是", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "chat.unclear")))
        await _wait_until(lambda: any("没听清" in text for _pid, text in sink.played))
        await engine.stop()

    asyncio.run(scenario())


def test_compare_without_scene_degrades() -> None:
    async def scenario() -> None:
        engine, frontend, sink, _tts, _ = _make_engine()
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("比较一下这两瓶水", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: any("还没接上" in text for _pid, text in sink.played))
        assert engine.image_tasks.unfinished() == []
        await engine.stop()

    asyncio.run(scenario())


def test_compare_capacity_blocked_after_two_unfinished() -> None:
    async def scenario() -> None:
        scene = FakeScene(provider=FakeProvider())
        engine, frontend, sink, _tts, _ = _make_engine(scene=scene)
        await engine.start()
        engine.image_tasks.open_compare("一")
        engine.image_tasks.open_compare("二")
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("比较一下这两瓶水", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.capacity_blocked")))
        await _wait_until(lambda: any("两个没看完的看图任务" in text for _pid, text in sink.played))
        assert scene.compare_calls == []
        await engine.stop()

    asyncio.run(scenario())


def test_resume_reuses_captured_frames_without_new_grab() -> None:
    async def scenario() -> None:
        provider = FakeProvider()
        scene = FakeScene(provider=provider)
        engine, frontend, sink, _tts, _ = _make_engine(scene=scene)
        task = engine.image_tasks.open_compare("比较一下这两个的价格")
        task.frames["A"] = _frame("a-9")
        task.frames["B"] = _frame("b-9")
        engine.image_tasks.mark(task, "paused", reason="barge_in")
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("继续", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "compare.answer")))
        assert provider.calls == [], "已缓存的 A/B 帧不应重新取图"
        assert _records(engine, "compare.resume"), "应记录 resume"
        assert task.state == "answered"
        await _wait_until(lambda: any("12.50" in text for _pid, text in sink.played))
        await engine.stop()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 阶段 5：身份核验、免唤醒候选与资料清除（假身份替身）
# ---------------------------------------------------------------------------
class FakeIdentity:
    """IdentityManager 替身：可注入核验状态、访问权限与候选处置。"""

    def __init__(
        self,
        status: str = "verified",
        *,
        access: bool = True,
        enrolled: bool = True,
        decisions: list[tuple[str, str]] | None = None,
    ) -> None:
        from aieyes.identity import CandidateDecision, IdentityResult, IdentityStatus

        self._status = IdentityStatus(status)
        self._result_cls = IdentityResult
        self._decision_cls = CandidateDecision
        self.access = access
        self.enrolled = enrolled
        self.decisions: list[tuple[str, str]] = list(decisions or [])
        self.verify_calls: list[object] = []
        self.note_calls = 0
        self.begins = 0
        self.topics_opened = 0
        self.topic_closed = 0
        self.topic = type("Topic", (), {"open": False, "remaining_ms": lambda self, now: 900000})()

    def verify(self, samples, duration_ms=None):
        self.verify_calls.append((samples, duration_ms))
        score = 0.81 if self._status.value == "verified" else 0.21
        return self._result_cls(self._status, score, "fake", 1200)

    def note(self, result, now_ms=None) -> None:
        self.note_calls += 1

    def access_ok(self, now_ms=None) -> bool:
        return self.access

    def open_topic(self, now_ms=None, label="") -> int:
        self.topics_opened += 1
        self.topic.open = True
        return (now_ms or mono_ms()) + 900000

    def close_topic(self) -> None:
        self.topic_closed += 1
        self.topic.open = False

    def begin_candidate(self, onset_ms: int) -> None:
        self.begins += 1

    def decide_candidate(self, result, text, *, directed, onset_ms=0, audio_end_ms=0, now_ms=None):
        self.calls = getattr(self, "calls", [])
        self.calls.append({"text": text, "directed": directed, "status": result.status.value})
        action, reason = self.decisions.pop(0) if self.decisions else ("accept", "verified")
        return self._decision_cls(action, reason, result)


class FakeChat:
    def __init__(self) -> None:
        self.turns: list[object] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1
        self.turns.clear()


def test_clear_denied_when_identity_not_verified() -> None:
    async def scenario() -> None:
        identity = FakeIdentity("mismatch", access=False)
        engine, frontend, sink, tts, chat = _make_engine(identity=identity)
        chat.record("之前的对话", "好的")
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("清空对话", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "materials.clear_denied")))
        # 拒绝话术是异步播出（_speak → _speech_worker → tts.synth 在线程里跑），
        # 记录落库早于 sink.play，必须等它真的进播放器再断言，否则断言的是时序不是行为。
        await _wait_until(lambda: any("得先确认是你本人" in text for _pid, text in sink.played))
        assert any("得先确认是你本人" in text for _pid, text in sink.played)
        assert _records(engine, "materials.cleared") == []
        assert len(chat.turns) == 1, "未确认身份不得清除资料"
        assert tts.clear_calls == 0
        await engine.stop()

    asyncio.run(scenario())


def test_clear_removes_chat_and_image_tasks_when_verified() -> None:
    async def scenario() -> None:
        identity = FakeIdentity("verified", access=True)
        engine, frontend, sink, tts, chat = _make_engine(identity=identity)
        chat.record("之前的对话", "好的")
        engine.image_tasks.open_compare("比较价格")
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("清空对话", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "materials.cleared")))
        cleared = _records(engine, "materials.cleared")[0]
        assert cleared["turns"] == 1 and cleared["tasks"] == 1
        assert chat.turns == [] and engine.image_tasks.tasks == []
        assert tts.clear_calls == 1
        await _wait_until(lambda: any("都清掉了" in text for _pid, text in sink.played))
        assert any("都清掉了" in text for _pid, text in sink.played)
        await engine.stop()

    asyncio.run(scenario())


def test_wakefree_opens_topic_after_answer_and_accepts_next_speech() -> None:
    async def scenario() -> None:
        cfg = AppConfig()
        cfg.timers.t17_post_answer_listen_ms = 60
        identity = FakeIdentity("verified", access=True)
        vlm = FakeVlm("今天多云，适合散步。")
        engine, frontend, sink, _tts, _chat = _make_engine(cfg, vlm=vlm, identity=identity)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(
            Utterance("今天天气怎么样？", mono_ms(), mono_ms(), mono_ms(), samples=np.zeros(1600, dtype=np.float32))
        )
        await _wait_until(lambda: bool(_records(engine, "wakefree.enter")), timeout=3.0)
        assert identity.topics_opened == 1
        assert frontend.wake_enabled is False and frontend.listening is True
        assert engine._wakefree is True
        frontend.on_speech_start(mono_ms())
        frontend.on_utterance(
            Utterance("明天呢", mono_ms(), mono_ms(), mono_ms(), samples=np.zeros(1600, dtype=np.float32))
        )
        await _wait_until(lambda: bool(_records(engine, "wakefree.accept")))
        assert identity.verify_calls, "免唤醒必须做本地声纹核验"
        await _wait_until(lambda: len(vlm.prompts) == 2)
        await engine.stop()

    asyncio.run(scenario())


def test_wakefree_ignores_mismatch_and_keeps_listening() -> None:
    async def scenario() -> None:
        cfg = AppConfig()
        cfg.timers.t17_post_answer_listen_ms = 60
        identity = FakeIdentity(
            "verified", access=True, decisions=[("ignore", "identity_mismatch")]
        )
        vlm = FakeVlm("今天多云。")
        engine, frontend, _sink, _tts, _chat = _make_engine(cfg, vlm=vlm, identity=identity)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("今天天气怎么样？", mono_ms(), mono_ms(), mono_ms(), samples=np.zeros(1600, dtype=np.float32)))
        await _wait_until(lambda: bool(_records(engine, "wakefree.enter")), timeout=3.0)
        frontend.on_utterance(Utterance("旁边有人说话", mono_ms(), mono_ms(), mono_ms(), samples=np.zeros(1600, dtype=np.float32)))
        await _wait_until(lambda: bool(_records(engine, "wakefree.ignored")))
        assert len(vlm.prompts) == 1, "他人声音不得提交模型"
        await engine.stop()

    asyncio.run(scenario())


def test_wakefree_need_explicit_wake_returns_idle() -> None:
    async def scenario() -> None:
        cfg = AppConfig()
        cfg.timers.t17_post_answer_listen_ms = 60
        identity = FakeIdentity(
            "unavailable", access=False, decisions=[("need_explicit_wake", "voiceprint_unavailable")]
        )
        engine, frontend, sink, _tts, _chat = _make_engine(cfg, identity=identity)
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("今天天气怎么样？", mono_ms(), mono_ms(), mono_ms(), samples=np.zeros(1600, dtype=np.float32)))
        await _wait_until(lambda: bool(_records(engine, "wakefree.enter")), timeout=3.0)
        frontend.on_utterance(Utterance("继续说吧", mono_ms(), mono_ms(), mono_ms(), samples=np.zeros(1600, dtype=np.float32)))
        await _wait_until(lambda: bool(_records(engine, "wakefree.need_explicit_wake")))
        await _wait_until(lambda: engine.state is InteractionState.IDLE, timeout=3.0)
        assert any("先叫一声“小冰”" in text for _pid, text in sink.played)
        await engine.stop()

    asyncio.run(scenario())




def test_resume_denied_without_identity_access() -> None:
    async def scenario() -> None:
        identity = FakeIdentity("mismatch", access=False)
        scene = FakeScene(provider=FakeProvider())
        engine, frontend, sink, _tts, _chat = _make_engine(scene=scene, identity=identity)
        task = engine.image_tasks.open_compare("比较两个价格")
        task.frames["A"] = _frame("a-1")
        task.frames["B"] = _frame("b-1")
        engine.image_tasks.mark(task, "paused", reason="barge_in")
        await engine.start()
        frontend.on_wake("小冰", mono_ms())
        await _wait_until(lambda: engine.state is InteractionState.LISTENING)
        frontend.on_utterance(Utterance("继续", mono_ms(), mono_ms(), mono_ms()))
        await _wait_until(lambda: bool(_records(engine, "materials.resume_denied")))
        assert _records(engine, "compare.resume") == []
        assert task.state == "paused"
        # 同上：resume_denied 话术晚一拍进播放器。
        await _wait_until(lambda: any("得先确认是你本人" in text for _pid, text in sink.played))
        assert any("得先确认是你本人" in text for _pid, text in sink.played)
        await engine.stop()

    asyncio.run(scenario())
