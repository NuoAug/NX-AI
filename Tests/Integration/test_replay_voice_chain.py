"""阶段 1 软件级闭环（回放）：唤醒 → 问候 → 真人问题 → 识别 → 回答播报。

链路全部用真实组件：ReplaySource（真实录音 wav）→ SherpaKws → EnergyVad →
StreamingAsr → InteractionEngine → PiperTts → RecordingSink（实时消耗，可取消）。
模型用 FakeVlm 固定回答，避免依赖 llama-server 是否已启动；这验证的是
“语音交互软件链路 + 计时/证据”，不是模型质量。

在目标板上跑：
    source Tools/target_env.sh && trun "python -m pytest -q Tests/Integration/test_replay_voice_chain.py -s"
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from aieyes.audio.replay_source import ReplaySource
from aieyes.audio.vad import EnergyVad
from aieyes.chat import ChatSession
from aieyes.config import AppConfig
from aieyes.events import EventBus
from aieyes.interaction.engine import InteractionEngine
from aieyes.interfaces import SAMPLE_RATE, PlaybackHandle
from aieyes.logging_setup import JsonlEvidenceLogger
from aieyes.speech.asr import StreamingAsr
from aieyes.speech.frontend import SpeechFrontend
from aieyes.speech.kws import SherpaKws
from aieyes.speech.tts import PiperTts

pytestmark = [pytest.mark.target, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "Tests" / "Fixtures" / "audio"
CONFIG = ROOT / "Config" / "target.json"


class FakeVlm:
    """固定回答的 VLM 替身；记录调用次数与 prompt。"""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, prompt, *, images=(), max_tokens=512, timeout_s=120.0, temperature=None):
        self.calls += 1
        self.prompts.append(prompt)
        return self.reply


class RecordingSink:
    """按实时节奏“播放”PCM 的 AudioSink 替身，记录每段播放结果。

    - 播放线程每 20ms 推进 played_ms，因此引擎的插话/停声计时是真实时间；
    - cancel() 会立刻结束当前段，wait() 返回是否完整播完；
    - reference_frame() 返回最近一小段已输出音频（AEC 参考信号）。
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = int(sample_rate)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._playing = False
        self._played_ms = 0
        self._total_ms = 0
        self._ref: np.ndarray | None = None
        self.cancel_calls = 0
        self.records: list[dict] = []

    # -- AudioSink 协议 -------------------------------------------------
    def start(self) -> None:
        return None

    def play(self, pcm, sample_rate: int = SAMPLE_RATE, *, piece_id: str = "", text: str = "") -> PlaybackHandle:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        total_ms = int(round(len(pcm) * 1000 / sample_rate))
        with self._lock:
            self._cancel_current()
            stop = threading.Event()
            rec = {"piece_id": piece_id, "text": text, "total_ms": total_ms, "played_ms": 0, "finished": False}
            self.records.append(rec)
            self._stop_evt = stop
            self._playing = True
            self._played_ms = 0
            self._total_ms = total_ms
            frame = max(1, sample_rate // 50)  # 20ms
            chunks = max(1, total_ms // 20)

            def _run() -> None:
                for i in range(chunks):
                    if stop.wait(0.02):
                        break
                    with self._lock:
                        self._played_ms = min(total_ms, (i + 1) * 20)
                        rec["played_ms"] = self._played_ms
                        start = max(0, (i + 1) * frame - frame)
                        self._ref = pcm[start : (i + 1) * frame].copy()
                else:
                    rec["finished"] = True
                with self._lock:
                    if not stop.is_set():
                        self._played_ms = total_ms
                    self._playing = False
                    self._ref = None

            thread = threading.Thread(target=_run, name="recording-sink", daemon=True)
            self._thread = thread
            thread.start()

        def _cancel() -> None:
            stop.set()

        def _wait(timeout: float | None = None) -> bool:
            thread.join(timeout)
            return bool(rec["finished"])

        return PlaybackHandle(piece_id=piece_id, total_ms=total_ms, text=text, _cancel=_cancel, _wait=_wait)

    def _cancel_current(self) -> None:
        if self._playing:
            self.cancel()

    def cancel(self) -> None:
        with self._lock:
            self.cancel_calls += 1
        self._stop_evt.set()

    def stop(self) -> None:
        self.cancel()

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def played_ms(self) -> int:
        with self._lock:
            return int(self._played_ms)

    def reference_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._ref is None else self._ref.copy()

    def close(self) -> None:
        self.stop()


def _read_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _find(records: list[dict], category: str, event: str | None = None) -> list[dict]:
    return [
        r
        for r in records
        if r.get("kind") == "interaction"
        and r.get("category") == category
        and (event is None or r.get("event") == event)
    ]


async def _wait_until(predicate, timeout_s: float, interval_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval_s)
    return False


def _silence(ms: int) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.float32)


def test_replay_voice_chain(tmp_path: Path) -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = AppConfig.from_file(CONFIG)
    if not Path(cfg.models.root).is_dir():
        pytest.skip(f"目标机模型目录不存在: {cfg.models.root}")
    wake = FIXTURES / "wake" / "tts_xiaobing_nihao.wav"
    questions = sorted((FIXTURES / "questions").glob("*询问前方*"))
    if not wake.is_file() or not questions:
        pytest.skip("缺少回放素材（wake/questions）")
    question = questions[0]

    cfg.audio.backend = "replay"
    cfg.audio.replay_realtime = True
    cfg.audio.replay_dir = str(FIXTURES / "questions")

    # 唤醒词 → 等问候播完再进 listening → 问题音频 → 尾部静音收段
    segments = [wake, _silence(2500), question, _silence(1800)]
    source = ReplaySource(segments, realtime=True)
    kws = SherpaKws(cfg.kws)
    vad = EnergyVad(cfg.vad)
    asr = StreamingAsr(cfg.asr)
    tts = PiperTts(cfg.tts)
    sink = RecordingSink()
    frontend = SpeechFrontend(source, kws=kws, vad=vad, asr=asr)
    vlm = FakeVlm("前方有一张桌子，桌上有一盏台灯。")
    chat = ChatSession(vlm, system_prompt="你是戴在用户眼前的智能眼镜助手，回答简短口语化。", max_tokens=64)
    evidence_path = tmp_path / "evidence.jsonl"
    evidence = JsonlEvidenceLogger(evidence_path)
    engine: InteractionEngine | None = None

    async def _scenario() -> bool:
        nonlocal engine
        bus = EventBus()
        engine = InteractionEngine(
            cfg=cfg, bus=bus, frontend=frontend, sink=sink, tts=tts, chat=chat, evidence=evidence
        )
        sink.start()
        await engine.start()
        # 引擎就绪后再开始回放，避免唤醒事件与启动阶段竞争
        frontend.start()
        try:
            return await _wait_until(
                lambda: bool(_find(_read_records(evidence_path), "answer", "spoken")),
                timeout_s=45.0,
            )
        finally:
            await engine.stop()
            frontend.stop()
            sink.close()

    ok = asyncio.run(_scenario())
    records = _read_records(evidence_path)
    if not ok:
        print("=== engine.state:", engine.state if engine else None)
        print("=== records:")
        for r in records:
            print(json.dumps(r, ensure_ascii=False))
    assert ok, f"45s 内没有完成一轮问答，证据: {[ (r.get('category'), r.get('event')) for r in records ]}"

    wake_recs = _find(records, "wake")
    assert any(r.get("event") == "detected" for r in wake_recs), f"没有唤醒记录: {wake_recs}"
    utt = _find(records, "utterance", "final")
    assert utt and utt[-1].get("text"), f"没有识别结果: {utt}"
    assert len(str(utt[-1]["text"])) >= 2
    answers = _find(records, "answer", "spoken")
    assert answers and answers[-1].get("played") is True, f"回答没有完整播完: {answers}"
    assert vlm.calls >= 1, "模型没有被调用"

    # sink 至少播过问候 + 回答
    assert len(sink.records) >= 2, f"播放记录不足: {[r['piece_id'] for r in sink.records]}"
    # 首声时延：问候（缓存预合成）应显著低于 500ms 目标
    first_audio = _find(records, "latency", "first_audio")
    greet = [r for r in first_audio if r.get("speech_kind") == "greeting"]
    assert greet, f"缺少问候首声时延记录: {first_audio}"
    assert greet[0]["ms"] < 500, f"问候首声 {greet[0]['ms']}ms 超过 500ms 目标"
