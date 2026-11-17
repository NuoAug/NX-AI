"""testbench/debug.py 的单元测试。

只使用 fake ctx、fake 业务组件和临时目录，不加载真实模型、不访问麦克风/
扬声器/相机/网络。运行方式（仓库根目录、bundled Python）：

    python -B -m unittest discover -s testbench/tests -p "test_debug.py" -v
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "testbench", _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import debug as debug_module  # noqa: E402
from debug import DebugError, DebugRunner, MODULE_SPECS  # noqa: E402

EXPECTED_IDS = [
    "audio",
    "kws_vad",
    "asr",
    "chat",
    "tts",
    "interruption",
    "vision",
    "ocr",
    "compare",
    "speaker",
    "contract",
]

ALLOWED_EVENTS = {
    "asr.partial",
    "asr.final",
    "assistant.final",
    "playback",
    "frame",
    "timing",
    "log",
    "error",
}


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def make_cfg(*, contract_enabled: bool = False, speaker_enabled: bool = False):
    models = SimpleNamespace(
        root=str(Path(tempfile.gettempdir()) / "aieyes-debug-test-models")
    )
    models.resolve = lambda rel: Path(models.root) / str(rel)
    return SimpleNamespace(
        audio=SimpleNamespace(sample_rate=16000, frame_ms=20, backend="sounddevice"),
        models=models,
        llm=SimpleNamespace(),
        vision=SimpleNamespace(),
        ocr=SimpleNamespace(),
        vlm=SimpleNamespace(),
        kws=SimpleNamespace(),
        vad=SimpleNamespace(),
        asr=SimpleNamespace(),
        tts=SimpleNamespace(),
        speaker=SimpleNamespace(
            enabled=speaker_enabled,
            threshold=0.65,
            enroll_dir="speaker/enroll",
            min_speech_ms=800,
            max_samples=5,
        ),
        contract=SimpleNamespace(
            enabled=contract_enabled,
            host="127.0.0.1",
            port=18700,
            role="interaction",
        ),
    )


class FakeFrame:
    def __init__(self, frame_id: str, width: int = 64, height: int = 48) -> None:
        self.frame_id = frame_id
        self.width = width
        self.height = height


class FakeProvider:
    def __init__(self, ctx: "FakeCtx") -> None:
        self.ctx = ctx
        self.closed = 0
        self.calls: list[str] = []

    def acquire(self, purpose: str = "vqa") -> FakeFrame:
        self.calls.append(purpose)
        self.ctx.frame_seq += 1
        return FakeFrame(f"cam-{self.ctx.frame_seq}")

    def close(self) -> None:
        self.closed += 1


class FakeScene:
    def __init__(self, answer_text: str = "这是视觉回答") -> None:
        self.answer_text = answer_text
        self.answer_calls: list[tuple[str, object]] = []
        self.compare_calls: list[tuple[object, object, str]] = []
        self.read_calls: list[object] = []
        self.closed = False

    def answer(self, question: str, *, frame=None, **kwargs) -> SimpleNamespace:
        self.answer_calls.append((question, frame))
        return SimpleNamespace(text=self.answer_text, intent="vqa", degraded=False)

    def read_text_result(self, frame) -> SimpleNamespace:
        self.read_calls.append(frame)
        lines = [SimpleNamespace(text="纯牛奶", score=0.98)]
        return SimpleNamespace(lines=lines, t_ms=12)

    def compare(self, a, b, question: str = "") -> SimpleNamespace:
        self.compare_calls.append((a, b, question))
        items = [SimpleNamespace(name="价格", a="5 元", b="6 元", note="")]
        return SimpleNamespace(
            text="A 比 B 便宜",
            a_frame_id=getattr(a, "frame_id", ""),
            b_frame_id=getattr(b, "frame_id", ""),
            items=items,
            degraded=False,
        )

    def close(self) -> None:
        self.closed = True


class FakeVlm:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeCtx:
    def __init__(
        self,
        run_dir: Path,
        *,
        images: dict | None = None,
        cfg=None,
    ) -> None:
        self.worker = None
        self.run_id = "dbg-test-run"
        self.run_dir = Path(run_dir)
        self.module = ""
        self.input: dict = {}
        self.images = dict(images or {})
        self.cfg = cfg or make_cfg()
        self.bench: dict = {}
        self.stop_event = threading.Event()
        self.events: list[tuple[str, dict, object]] = []
        self.logs: list[tuple[str, str]] = []
        self.finish_calls: list[tuple[str, str]] = []
        self.finished_event = threading.Event()
        self.saved: list[dict] = []
        self.providers: list[FakeProvider] = []
        self.frame_seq = 0
        self.scene = FakeScene()

    def emit(self, etype: str, payload: dict | None = None, turn_id=None) -> None:
        self.events.append((etype, dict(payload or {}), turn_id))

    def log(self, level: str, message: str) -> None:
        self.logs.append((level, str(message)))

    def error(self, code: str, message: str, retryable: bool = False) -> None:
        self.emit(
            "error",
            {"code": code, "message": message, "retryable": retryable},
        )

    def timing(self, **fields) -> None:
        self.emit("timing", {k: v for k, v in fields.items() if v is not None})

    @property
    def stopped(self) -> bool:
        return self.stop_event.is_set()

    def wait(self, seconds: float) -> bool:
        return self.stop_event.wait(min(float(seconds), 0.02))

    def finished(self, status: str, error: str = "") -> None:
        self.finish_calls.append((status, error))
        self.finished_event.set()

    def ensure_hub(self):
        return object()

    def camera_provider(self) -> FakeProvider:
        provider = FakeProvider(self)
        self.providers.append(provider)
        return provider

    def load_image(self, slot: str) -> FakeFrame:
        if slot not in self.images:
            raise DebugError("missing_image", f"缺少图片 {slot}")
        return FakeFrame(f"upload-{slot.lower()}")

    def save_frame(self, frame, slot: str, *, purpose: str = "debug") -> dict:
        meta = {
            "file": f"media/{slot}-{len(self.saved)}.jpg",
            "width": getattr(frame, "width", 0),
            "height": getattr(frame, "height", 0),
        }
        self.saved.append(
            {
                "slot": slot,
                "purpose": purpose,
                "frame_id": getattr(frame, "frame_id", ""),
                "meta": meta,
            }
        )
        self.emit(
            "frame",
            {
                "frame_id": getattr(frame, "frame_id", ""),
                "width": meta["width"],
                "height": meta["height"],
                "purpose": purpose,
                "slot": slot,
            },
        )
        return meta

    def scene_service(self, with_ocr: bool = True):
        return self.scene

    def vlm_client(self) -> FakeVlm:
        return FakeVlm()

    @property
    def text_logs(self) -> str:
        return "\n".join(message for _level, message in self.logs)

    @property
    def event_types(self) -> list[str]:
        return [etype for etype, _payload, _turn in self.events]

    def events_of(self, etype: str) -> list[dict]:
        return [payload for kind, payload, _turn in self.events if kind == etype]


class FakeChat:
    def __init__(self, text: str = "这是测试回答") -> None:
        self.text = text
        self.asked: list[str] = []

    def ask(self, user_text: str, **kwargs) -> SimpleNamespace:
        self.asked.append(user_text)
        return SimpleNamespace(text=self.text)


class FakeHandle:
    total_ms = 100

    def wait(self, timeout: float | None = None) -> bool:
        time.sleep(0.01)
        return True

    def cancel(self) -> None:
        pass


class FakeTts:
    def __init__(self) -> None:
        self.synths: list[str] = []
        self.closed = False

    def synth(self, text: str) -> SimpleNamespace:
        self.synths.append(text)
        return SimpleNamespace(
            pcm=[0.1, 0.2], sample_rate=16000, duration_ms=100, text=text
        )

    def close(self) -> None:
        self.closed = True


class FakeSink:
    def __init__(self, *, close_error: bool = False) -> None:
        self.started = 0
        self.played: list[tuple] = []
        self.cancelled: list[str] = []
        self.closed = False
        self.close_error = close_error

    def start(self) -> None:
        self.started += 1

    def play(self, pcm, sample_rate, *, piece_id="", text=""):
        self.played.append((pcm, sample_rate, piece_id, text))
        return FakeHandle()

    def cancel(self, piece_id=None) -> bool:
        self.cancelled.append(piece_id or "")
        return True

    def close(self) -> None:
        self.closed = True
        if self.close_error:
            raise RuntimeError("sink close boom")


class ChatRunner(DebugRunner):
    def __init__(self, ctx: FakeCtx, chat: FakeChat | None = None) -> None:
        super().__init__(ctx)
        self.chat = chat or FakeChat()
        self.vlm = FakeVlm()

    def _make_vlm(self):
        return self.vlm

    def _make_chat(self, vlm):
        return self.chat


class TtsRunner(DebugRunner):
    def __init__(self, ctx: FakeCtx) -> None:
        super().__init__(ctx)
        self.tts = FakeTts()
        self.sink = FakeSink(close_error=True)

    def _make_tts(self):
        return self.tts

    def _make_sink(self, sample_rate: int):
        return self.sink


class SceneRunner(DebugRunner):
    def __init__(self, ctx: FakeCtx, scene: FakeScene | None = None) -> None:
        super().__init__(ctx)
        self.fake_scene = scene or FakeScene()

    def _make_scene(self):
        return self.fake_scene


class UnavailableSpeakerRunner(DebugRunner):
    def _make_gate(self):
        raise RuntimeError("onnxruntime 未安装")


class ModuleSpecsTest(unittest.TestCase):
    def test_ids_are_fixed_and_unique(self):
        ids = [str(item["id"]) for item in MODULE_SPECS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sorted(ids), sorted(EXPECTED_IDS))

    def test_spec_shape_matches_frontend_contract(self):
        for spec in MODULE_SPECS:
            with self.subTest(module=spec.get("id")):
                self.assertIsInstance(spec, dict)
                for key in ("id", "name", "desc", "fields", "actions"):
                    self.assertIn(key, spec)
                self.assertIsInstance(spec["name"], str)
                self.assertTrue(spec["name"])
                self.assertIsInstance(spec["desc"], str)
                self.assertTrue(spec["desc"])
                self.assertIsInstance(spec["fields"], list)
                self.assertIsInstance(spec["actions"], list)

                seen_fields = set()
                for field in spec["fields"]:
                    self.assertIn("name", field)
                    self.assertIn("label", field)
                    self.assertIn(field["type"], {"text", "select", "textarea", "checkbox"})
                    self.assertNotIn(field["name"], seen_fields)
                    seen_fields.add(field["name"])
                    if field["type"] == "select":
                        options = field.get("options")
                        self.assertIsInstance(options, list)
                        self.assertTrue(options)
                        for option in options:
                            self.assertIsInstance(option, str)
                    if "required" in field:
                        self.assertIsInstance(field["required"], bool)

                for action in spec["actions"]:
                    self.assertIsInstance(action.get("name"), str)
                    self.assertTrue(action["name"])
                    self.assertIsInstance(action.get("label"), str)
                    self.assertTrue(action["label"])


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ctx = FakeCtx(Path(self.tmp.name))
        self.runner = DebugRunner(self.ctx)

    def test_unknown_module(self):
        with self.assertRaises(DebugError) as raised:
            self.runner.start("not-a-module", {})
        self.assertEqual(raised.exception.code, "invalid_module")
        self.assertFalse(self.runner.alive)
        self.assertEqual(self.ctx.finish_calls, [])

    def test_missing_required_chat_text(self):
        with self.assertRaises(DebugError) as raised:
            self.runner.start("chat", {"text": "   "})
        self.assertEqual(raised.exception.code, "invalid_request")
        self.assertIn("问题", raised.exception.message)

    def test_missing_uploaded_image_for_compare(self):
        with self.assertRaises(DebugError) as raised:
            self.runner.start(
                "compare",
                {"question": "比较", "source": "上传图片 A/B"},
            )
        self.assertEqual(raised.exception.code, "missing_image")
        self.assertIn("A", raised.exception.message)

    def test_missing_uploaded_image_for_vision(self):
        with self.assertRaises(DebugError) as raised:
            self.runner.start(
                "vision",
                {"question": "这是什么", "source": "上传图片 A"},
            )
        self.assertEqual(raised.exception.code, "missing_image")

    def test_invalid_select_value(self):
        with self.assertRaises(DebugError) as raised:
            self.runner.start(
                "vision",
                {"question": "这是什么", "source": "手机"},
            )
        self.assertEqual(raised.exception.code, "invalid_request")
        self.assertIn("图片来源", raised.exception.message)

    def test_speaker_requires_consent(self):
        with self.assertRaises(DebugError) as raised:
            self.runner.start("speaker", {"consent": False})
        self.assertEqual(raised.exception.code, "consent_required")
        self.assertIn("同意", raised.exception.message)

    def test_action_without_running_task(self):
        with self.assertRaises(DebugError) as raised:
            self.runner.action("status", {})
        self.assertEqual(raised.exception.code, "invalid_request")


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runners: list[DebugRunner] = []

    def tearDown(self):
        for runner in self.runners:
            if runner.alive:
                runner.stop()

    def new_ctx(self, **kwargs) -> FakeCtx:
        ctx = FakeCtx(Path(self.tmp.name), **kwargs)
        return ctx

    def track(self, runner: DebugRunner) -> DebugRunner:
        self.runners.append(runner)
        return runner

    def wait_finished(self, ctx: FakeCtx, runner: DebugRunner) -> tuple[str, str]:
        finished = wait_until(lambda: bool(ctx.finish_calls) and not runner.alive)
        self.assertTrue(finished, msg=f"调试未结束，日志：{ctx.logs}")
        return ctx.finish_calls[-1]

    def assert_allowed_events(self, ctx: FakeCtx) -> None:
        for etype in ctx.event_types:
            self.assertIn(etype, ALLOWED_EVENTS)

    def test_chat_start_emits_final_and_finishes(self):
        ctx = self.new_ctx()
        runner = self.track(ChatRunner(ctx))
        runner.start("chat", {"text": "介绍一下你自己"})
        status, error = self.wait_finished(ctx, runner)
        self.assertEqual((status, error), ("completed", ""))
        self.assert_allowed_events(ctx)
        self.assertIn("assistant.final", ctx.event_types)
        final = ctx.events_of("assistant.final")[-1]
        self.assertEqual(final["text"], "这是测试回答")
        self.assertEqual(runner.chat.asked, ["介绍一下你自己"])
        self.assertTrue(runner.vlm.closed)
        timing = ctx.events_of("timing")
        self.assertTrue(timing)
        self.assertIn("generation_ms", timing[-1])

    def test_tts_playback_and_cleanup_errors_are_contained(self):
        ctx = self.new_ctx()
        runner = self.track(TtsRunner(ctx))
        runner.start("tts", {"text": "你好，测试"})
        status, _error = self.wait_finished(ctx, runner)
        self.assertEqual(status, "completed")
        self.assert_allowed_events(ctx)
        playback = ctx.events_of("playback")
        self.assertEqual([item["state"] for item in playback], ["started", "finished"])
        self.assertTrue(runner.tts.closed)
        self.assertTrue(runner.sink.closed)
        self.assertEqual(ctx.events_of("error"), [])
        self.assertIn("释放失败", ctx.text_logs)

    def test_stop_early_marks_interrupted(self):
        ctx = self.new_ctx()
        runner = self.track(DebugRunner(ctx))
        runner.start("contract", {})
        self.assertTrue(wait_until(lambda: "合同未接入" in ctx.text_logs))
        self.assertTrue(runner.alive)
        runner.stop()
        status, error = self.wait_finished(ctx, runner)
        self.assertEqual(status, "interrupted")
        self.assertEqual(error, "")
        self.assert_allowed_events(ctx)
        self.assertIn("合同未接入", ctx.text_logs)
        runner.stop()  # 幂等，不能抛异常

    def test_contract_disabled_actions_do_not_send(self):
        ctx = self.new_ctx()
        runner = self.track(DebugRunner(ctx))
        runner.start("contract", {})
        self.assertTrue(wait_until(lambda: runner.alive and "合同未接入" in ctx.text_logs))
        status_payload = runner.action("status", {})
        self.assertFalse(status_payload["enabled"])
        self.assertFalse(status_payload["connected"])
        event_payload = runner.action("test_event", {})
        self.assertFalse(event_payload["sent"])
        self.assertEqual(event_payload["reason"], "contract_disabled")
        runner.stop()
        status, _error = self.wait_finished(ctx, runner)
        self.assertEqual(status, "interrupted")

    def test_contract_enabled_still_never_connects(self):
        ctx = self.new_ctx(cfg=make_cfg(contract_enabled=True))
        runner = self.track(DebugRunner(ctx))
        runner.start("contract", {})
        self.assertTrue(wait_until(lambda: "不连接真实主控" in ctx.text_logs))
        status_payload = runner.action("status", {})
        self.assertTrue(status_payload["enabled"])
        self.assertFalse(status_payload["connected"])
        event_payload = runner.action("test_event", {})
        self.assertFalse(event_payload["sent"])
        self.assertEqual(event_payload["reason"], "debug_offline")
        runner.stop()
        status, _error = self.wait_finished(ctx, runner)
        self.assertEqual(status, "interrupted")

    def test_speaker_disabled_reports_reason(self):
        ctx = self.new_ctx(cfg=make_cfg(speaker_enabled=False))
        runner = self.track(DebugRunner(ctx))
        runner.start("speaker", {"consent": True})
        status, _error = self.wait_finished(ctx, runner)
        self.assertEqual(status, "completed")
        self.assertIn("未启用", ctx.text_logs)
        self.assertIn("speaker.enabled=false", ctx.text_logs)
        self.assertEqual(ctx.events_of("error"), [])

    def test_speaker_component_missing_reports_unavailable(self):
        ctx = self.new_ctx(cfg=make_cfg(speaker_enabled=True))
        runner = self.track(UnavailableSpeakerRunner(ctx))
        runner.start("speaker", {"consent": True})
        status, _error = self.wait_finished(ctx, runner)
        self.assertEqual(status, "completed")
        self.assertIn("声纹组件未接入", ctx.text_logs)
        self.assertEqual(ctx.events_of("error"), [])

    def test_action_after_finished_is_rejected(self):
        ctx = self.new_ctx()
        runner = self.track(ChatRunner(ctx))
        runner.start("chat", {"text": "你好"})
        self.wait_finished(ctx, runner)
        with self.assertRaises(DebugError) as raised:
            runner.action("play", {})
        self.assertEqual(raised.exception.code, "invalid_request")

    def test_vision_uses_frame_and_scene_answer(self):
        ctx = self.new_ctx(images={"A": {"path": "A.jpg", "image_id": "img-a"}})
        runner = self.track(SceneRunner(ctx))
        runner.start("vision", {"question": "图里是什么", "source": "上传图片 A"})
        status, _error = self.wait_finished(ctx, runner)
        self.assertEqual(status, "completed")
        self.assert_allowed_events(ctx)
        self.assertIn("frame", ctx.event_types)
        self.assertEqual(ctx.saved[-1]["slot"], "A")
        self.assertIn("assistant.final", ctx.event_types)
        self.assertEqual(
            ctx.events_of("assistant.final")[-1]["text"],
            "这是视觉回答",
        )
        self.assertEqual(runner.fake_scene.answer_calls[0][0], "图里是什么")
        self.assertIn("vision_ms", ctx.events_of("timing")[-1])

    def test_ocr_shows_raw_lines_and_business_answer(self):
        ctx = self.new_ctx()
        runner = self.track(SceneRunner(ctx))
        runner.start("ocr", {"question": "读取文字", "source": "相机"})
        status, _error = self.wait_finished(ctx, runner)
        self.assertEqual(status, "completed")
        self.assertIn("OCR 原文", ctx.text_logs)
        self.assertIn("纯牛奶", ctx.text_logs)
        self.assertIn("assistant.final", ctx.event_types)
        self.assertEqual(len(runner.fake_scene.read_calls), 1)

    def test_compare_uploaded_images(self):
        ctx = self.new_ctx(
            images={
                "A": {"path": "A.jpg", "image_id": "img-a"},
                "B": {"path": "B.jpg", "image_id": "img-b"},
            }
        )
        runner = self.track(SceneRunner(ctx))
        runner.start(
            "compare",
            {"question": "哪个便宜", "source": "上传图片 A/B"},
        )
        status, _error = self.wait_finished(ctx, runner)
        self.assertEqual(status, "completed")
        self.assertEqual([item["slot"] for item in ctx.saved], ["A", "B"])
        self.assertEqual(len(runner.fake_scene.compare_calls), 1)
        self.assertIn("比较结果", ctx.text_logs)
        self.assertIn("assistant.final", ctx.event_types)

    def test_compare_camera_requires_two_manual_captures(self):
        ctx = self.new_ctx()
        runner = self.track(SceneRunner(ctx))
        runner.start(
            "compare",
            {"question": "比较", "source": "相机先后拍摄"},
        )
        self.assertTrue(wait_until(lambda: "拍摄图片 A" in ctx.text_logs))
        result_a = runner.action("capture_a", {})
        self.assertEqual(result_a["slot"], "A")
        with self.assertRaises(DebugError):
            runner.action("capture_a", {})
        result_b = runner.action("capture_b", {})
        self.assertEqual(result_b["slot"], "B")
        status, _error = self.wait_finished(ctx, runner)
        self.assertEqual(status, "completed")
        self.assertEqual([item["slot"] for item in ctx.saved], ["A", "B"])
        self.assertEqual(len(ctx.providers), 2)
        self.assertEqual(len(runner.fake_scene.compare_calls), 1)
        self.assertIn("assistant.final", ctx.event_types)

    def test_interruption_module_metadata_only(self):
        """interruption 真实链路不在单测中加载，只校验 spec 与事件白名单常量的关系。"""
        spec = next(item for item in MODULE_SPECS if item["id"] == "interruption")
        self.assertIn("不注入取消", spec["desc"])
        self.assertEqual(debug_module._DEBUG_EVENT_TYPES, ALLOWED_EVENTS)


if __name__ == "__main__":
    unittest.main()
