"""bridge.py 单元测试：事件副本、turn 归并、去重与取消语义。

只用假组件与临时目录，不触碰音频/相机/模型。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
SRC = BENCH.parent / "Src"
for path in (str(BENCH), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from aieyes.events import (  # noqa: E402
    CancelRequested,
    PartialText,
    PlaybackFinished,
    PlaybackStarted,
    SpeechStarted,
    UtteranceRecognized,
)

import bridge as bridge_mod  # noqa: E402
from bridge import Bridge  # noqa: E402


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str | None, dict]] = []

    def __call__(self, etype: str, turn_id: str | None, payload: dict) -> None:
        self.events.append((etype, turn_id, dict(payload)))

    def typed(self, etype: str) -> list[tuple[str, str | None, dict]]:
        return [item for item in self.events if item[0] == etype]


class BridgeTurnTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rec = Recorder()
        self.bridge = Bridge(self.rec, run_id="run-1")

    def test_partial_then_final_share_one_turn(self) -> None:
        self.bridge.note_speech_start(1000)
        self.bridge.note_partial("帮我看看", 1100)
        self.bridge.note_partial("帮我看看日期", 1200)
        self.bridge.note_utterance(
            "帮我看看日期", 1300, t_audio_end_ms=1000, t_final_ms=1450, utterance_id="u1"
        )
        partials = self.rec.typed("asr.partial")
        finals = self.rec.typed("asr.final")
        self.assertEqual([item[2]["text"] for item in partials], ["帮我看看", "帮我看看日期"])
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0][2]["text"], "帮我看看日期")
        self.assertEqual(len({item[1] for item in partials + finals}), 1)
        timings = [item[2] for item in self.rec.typed("timing")]
        self.assertEqual(timings[0]["asr_ms"], 450)

    def test_chat_delta_and_final(self) -> None:
        self.bridge.note_utterance("你好", 100)
        turn = self.bridge.chat_begin()
        self.bridge.chat_text_ready(turn)
        self.bridge.chat_delta(turn, "你")
        self.bridge.chat_delta(turn, "好呀")
        self.bridge.chat_done(turn, "你好呀", generation_ms=321)
        deltas = [item[2]["text"] for item in self.rec.typed("assistant.delta")]
        self.assertEqual(deltas, ["你", "好呀"])
        finals = self.rec.typed("assistant.final")
        self.assertEqual(finals[-1][2]["text"], "你好呀")
        timing = [item[2] for item in self.rec.typed("timing") if "generation_ms" in item[2]]
        self.assertEqual(timing[0]["generation_ms"], 321)

    def test_chat_playback_does_not_duplicate_text(self) -> None:
        self.bridge.note_utterance("问题", 100)
        turn = self.bridge.chat_begin()
        self.bridge.chat_delta(turn, "回答")
        self.bridge.chat_done(turn, "回答", generation_ms=10)
        self.bridge.note_playback_started(
            PlaybackStarted(piece_id="p1", t_ms=300, text="回答")
        )
        playback = self.rec.typed("playback")
        self.assertEqual(playback[-1][2]["text"], "")
        self.assertEqual(playback[-1][2]["kind"], "answer")

    def test_prompt_playback_builds_assistant_text(self) -> None:
        self.bridge.note_playback_started(
            PlaybackStarted(piece_id="g1", t_ms=500, text="我在")
        )
        self.bridge.note_playback_finished(
            PlaybackFinished(piece_id="g1", t_ms=900, played_ms=400, total_ms=400, text="我在")
        )
        finals = self.rec.typed("assistant.final")
        self.assertEqual(finals[-1][2]["text"], "我在")
        playback = self.rec.typed("playback")
        self.assertEqual([item[2]["state"] for item in playback], ["started", "finished"])

    def test_same_piece_echo_is_deduplicated(self) -> None:
        self.bridge.note_playback_started(PlaybackStarted(piece_id="p9", t_ms=100, text="稍等"))
        self.bridge.note_playback_started(PlaybackStarted(piece_id="p9", t_ms=120, text="稍等"))
        started = [item[2] for item in self.rec.typed("playback") if item[2]["state"] == "started"]
        self.assertEqual([item["text"] for item in started], ["稍等", ""])

    def test_cancelled_turn_ignores_late_delta(self) -> None:
        self.bridge.note_utterance("说点什么", 100)
        turn = self.bridge.chat_begin()
        self.bridge.chat_delta(turn, "第一段")
        self.bridge.note_cancel("barge_in", 200)
        before = len(self.rec.typed("assistant.delta"))
        self.bridge.chat_delta(turn, "迟到内容")
        after = len(self.rec.typed("assistant.delta"))
        self.assertEqual(before, after)
        finals = self.rec.typed("assistant.final")
        self.assertTrue(any(item[2].get("cancelled") for item in finals))

    def test_tts_accumulates_and_sets_playback_wait(self) -> None:
        self.bridge.note_utterance("讲个故事", 100)
        self.bridge.chat_begin()
        self.bridge.note_tts(120)
        self.bridge.note_tts(80)
        self.bridge.note_playback_started(PlaybackStarted(piece_id="p2", t_ms=600, text=""))
        timing = [item[2] for item in self.rec.typed("timing")]
        self.assertEqual(timing[0]["tts_ms"], 120)
        self.assertEqual(timing[1]["tts_ms"], 200)
        wait = [item for item in timing if "playback_wait_ms" in item]
        self.assertEqual(len(wait), 1)
        self.assertGreaterEqual(wait[0]["playback_wait_ms"], 0)

    def test_evidence_mapping_warns_on_errors(self) -> None:
        self.bridge.note_evidence(
            {"kind": "interaction", "category": "chat", "event": "failed", "ms": 12}
        )
        logs = self.rec.typed("log")
        self.assertEqual(logs[-1][2]["level"], "WARN")
        before = len(logs)
        self.bridge.note_evidence({"kind": "interaction", "category": "state", "event": "idle"})
        self.assertEqual(len(self.rec.typed("log")), before + 1)
        self.bridge.note_evidence({"kind": "other"})
        self.assertEqual(len(self.rec.typed("log")), before + 1)

    def test_finish_flushes_open_turn(self) -> None:
        self.bridge.note_utterance("在吗", 100)
        turn = self.bridge.chat_begin()
        self.bridge.chat_delta(turn, "在的")
        self.bridge.finish(cancelled=True)
        snap = self.bridge.snapshot_turns()
        self.assertTrue(snap)
        self.assertEqual(snap[0]["assistant"], "在的")


class BridgeWrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rec = Recorder()
        self.bridge = Bridge(self.rec, run_id="run-2")

    def test_observed_bus_forwards_and_copies(self) -> None:
        class FakeBus:
            def __init__(self) -> None:
                self.posted: list[object] = []

            def post(self, ev):
                self.posted.append(ev)

        bus = FakeBus()
        observed = self.bridge.wrap_bus(bus)
        observed.post(PartialText(text="你好", t_ms=10))
        self.assertEqual(self.rec.typed("asr.partial")[-1][2]["text"], "你好")
        # 原事件仍交给真实总线（观察副本不 replaced 原投递）
        self.assertEqual(len(bus.posted), 1)
        self.assertIsInstance(bus.posted[0], PartialText)

    def test_observed_chat_does_not_record_again(self) -> None:
        class FakeVlm:
            def complete_stream(self, *args, **kwargs):
                return iter(["a"])

        class FakeChat:
            def __init__(self) -> None:
                self.vlm = FakeVlm()
                self.records: list[tuple[str, str]] = []

            def record(self, user_text: str, assistant_text: str, *args, **kwargs) -> None:
                self.records.append((user_text, assistant_text))

            def ask(self, user_text: str, **kwargs):
                self.record(user_text, "回答")
                return SimpleNamespace(text="回答")

        inner = FakeChat()
        wrapped = self.bridge.wrap_chat(inner)
        self.bridge.note_utterance("问题", 10)
        turn = self.bridge.chat_begin()
        wrapped.ask("问题")
        self.bridge.chat_done(turn, "回答", generation_ms=5)
        self.assertEqual(inner.records, [("问题", "回答")])
        self.assertEqual(self.rec.typed("assistant.final")[-1][2]["text"], "回答")

    def test_observed_chat_stream_marks_cancelled(self) -> None:
        inner = SimpleNamespace(vlm=SimpleNamespace(complete_stream=lambda *a, **k: iter(["x"])))

        def _stream(text, **kwargs):
            yield "第一段"
            yield "第二段"

        inner.ask_stream = _stream
        wrapped = self.bridge.wrap_chat(inner)
        self.bridge.note_utterance("说", 10)
        turn = self.bridge.chat_begin()
        iterator = wrapped.ask_stream("说")
        next(iterator)
        iterator.close()
        finals = self.rec.typed("assistant.final")
        self.assertEqual(finals[-1][2]["cancelled"], True)

    def test_observed_provider_saves_frame(self) -> None:
        import numpy as np

        from media import FrameStore

        class FakeFrame:
            frame_id = "f-1"
            t_capture_ms = 123
            source = "fake"

            def __init__(self) -> None:
                self.rgb = np.zeros((8, 8, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            store = FrameStore(Path(tmp))
            bridge = Bridge(self.rec, run_id="run-3", frame_store=store)
            inner = SimpleNamespace(acquire=lambda purpose="vqa": FakeFrame())
            wrapped = bridge.wrap_provider(inner)
            frame = wrapped.acquire("vqa")
            self.assertEqual(frame.frame_id, "f-1")
            saved = [item[2] for item in self.rec.typed("frame")]
            self.assertEqual(saved[-1]["frame_id"], "f-1")
            self.assertEqual(saved[-1]["slot"], "model")
            self.assertTrue((Path(tmp) / saved[-1]["file"]).is_file())

    def test_observed_scene_emits_ab_frames(self) -> None:
        import numpy as np

        class FakeFrame:
            def __init__(self, frame_id: str) -> None:
                self.frame_id = frame_id
                self.t_capture_ms = 1
                self.source = "fake"
                self.rgb = np.zeros((4, 4, 3), dtype=np.uint8)

        class FakeScene:
            def compare(self, a, b, question="q", **kwargs):
                return SimpleNamespace(a_frame_id=a.frame_id, b_frame_id=b.frame_id)

        with tempfile.TemporaryDirectory() as tmp:
            from media import FrameStore

            bridge = Bridge(self.rec, run_id="run-4", frame_store=FrameStore(Path(tmp)))
            bridge.note_utterance("比较", 10)
            observed = bridge.wrap_scene(FakeScene())
            observed.compare(FakeFrame("A1"), FakeFrame("B1"), "比较")
            slots = [item[2]["slot"] for item in self.rec.typed("frame")]
            self.assertEqual(slots[-2:], ["A", "B"])
            vision = [item for item in self.rec.typed("timing") if "vision_ms" in item[2]]
            self.assertEqual(len(vision), 1)


if __name__ == "__main__":
    unittest.main()
