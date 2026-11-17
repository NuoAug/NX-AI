"""M5：图片任务容量/引用释放、历史图预算与同图冒充比较的定向契约。

覆盖清单 14/15/16/21 中可用软件路径复现的部分：

- 一次双图比较完成后，必须放下本次引用；否则同一会话内第二次双图比较
  会因「实际证据组最多 2 个」直接容量失败（真实入口 = `InteractionEngine._do_compare`）。
- `imagetasks` 模块自述「只分析历史图不扣次数、不启动 T08」；历史图取图
  不应消耗位置次数，也不应启动 180 秒采集预算。
- 两张完全相同的画面（内容摘要相同 → 同一 `frame_id`）不能冒充「两样东西」
  的比较结果（清单 21：不能默默回退旧图冒充成功）。

全部用例只走本机软件路径（假 OCR/VLM、文件图源），不涉及相机与模型服务。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aieyes.config import AppConfig
from aieyes.events import EventBus
from aieyes.imagetasks import ImageTaskStore
from aieyes.intent import Intent, IntentKind
from aieyes.interaction.engine import InteractionEngine
from aieyes.scene import SceneService
from aieyes.vision.file_camera import load_frame

FIXTURES = Path(__file__).resolve().parents[2] / "Tests" / "Fixtures" / "scene"
if not FIXTURES.is_dir():
    FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "scene"
PRICE_A = FIXTURES / "compare_price_a.png"
PRICE_B = FIXTURES / "compare_price_b.png"

FALLBACK = "两张图我没能比出来，要不换个问题？"


# ----------------------------------------------------------------------
# 假件：OCR / VLM / 场景 / 引擎外壳
# ----------------------------------------------------------------------
class _Ocr:
    def __init__(self, by_frame: dict[str, list[str]] | None = None) -> None:
        self._by_frame = by_frame or {}
        self.calls: list[str] = []

    def recognize(self, rgb, *, source_frame_id: str = ""):
        from aieyes.interfaces import OcrLine, OcrResult

        self.calls.append(source_frame_id)
        lines = self._by_frame.get(source_frame_id, [])
        return OcrResult(
            lines=tuple(OcrLine(text=text, box=(0, 0, 10, 10), score=0.9) for text in lines),
            t_ms=1,
            source_frame_id=source_frame_id,
            image_size=(int(rgb.shape[1]), int(rgb.shape[0])),
        )


class _Vlm:
    def complete(self, prompt, *, images=(), max_tokens=512, timeout_s=120.0, temperature=None):
        return ""


def _service(by_frame: dict[str, list[str]] | None = None) -> SceneService:
    return SceneService(ocr=_Ocr(by_frame), vlm=_Vlm())


class _Provider:
    """按顺序发帧的取图提供者；用完后重复最后一帧（模拟单文件图源）。"""

    def __init__(self, frames) -> None:
        self._frames = list(frames)
        self.calls = 0

    def acquire(self, purpose: str = "vqa"):
        index = min(self.calls, len(self._frames) - 1)
        self.calls += 1
        return self._frames[index]


class _CompareAnswer:
    def __init__(self, text: str, a: str, b: str) -> None:
        self.text = text
        self.items = ()
        self.common_items = ("价格",)
        self.missing = ()
        self.a_frame_id = a
        self.b_frame_id = b
        self.a_tags = ()
        self.b_tags = ()
        self.degraded = ()
        self.timings = {}
        self.intent = "compare"


class _Scene:
    def __init__(self, frames, *, text: str = "比较结果：B更贵。") -> None:
        self.provider = _Provider(frames)
        self.text = text
        self.compare_calls: list[tuple[str, str, str]] = []

    def compare(self, a, b, question="比较一下这两样东西"):
        self.compare_calls.append((a.frame_id, b.frame_id, question))
        return _CompareAnswer(self.text, a.frame_id, b.frame_id)


class _Frontend:
    def set_listening(self, value: bool) -> None:
        pass

    def set_wake_enabled(self, value: bool) -> None:
        pass


class _Sink:
    is_playing = False

    def cancel(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _Tts:
    def synth(self, text):  # pragma: no cover - 不应触发真实 TTS
        raise AssertionError("测试不应触发真实 TTS")


class _Chat:
    def __init__(self) -> None:
        self.turns: list = []

    def record(self, *args, **kwargs) -> None:
        pass


def _engine(scene: _Scene, loop=None):
    cfg = AppConfig()
    cfg.timers.t11_generate_ms = 5000
    engine = InteractionEngine(
        cfg=cfg,
        bus=EventBus(loop=loop) if loop is not None else EventBus(),
        frontend=_Frontend(),
        sink=_Sink(),
        tts=_Tts(),
        chat=_Chat(),
        scene=scene,
    )
    spoken: list[tuple[str, str]] = []

    async def fake_speak(text: str, *, kind: str, ref_ms=None, meta=None) -> bool:
        spoken.append((kind, text))
        return True

    engine._speak = fake_speak  # type: ignore[method-assign]

    async def no_listen(*args, **kwargs) -> None:
        return None

    engine._after_recovery_listen = no_listen  # type: ignore[method-assign]
    engine._enter_listening = no_listen  # type: ignore[method-assign]
    return engine, spoken


def _compare_intent(text: str = "比较一下这两样东西") -> Intent:
    return Intent(kind=IntentKind.COMPARE, text=text, raw=text)


# ----------------------------------------------------------------------
# 1) 完成一次比较后必须放下引用：同一会话第二次双图不能容量失败
# ----------------------------------------------------------------------
def test_finished_compare_releases_group_capacity() -> None:
    """修复前：第一次比较完成后两组仍占容量，第二次 new_group 抛 CapacityError。"""
    store = ImageTaskStore()
    first = store.open_compare("哪个便宜")
    store.mark(first, "capturing")
    group_a = store.new_group(first, position="A", source="file")
    group_b = store.new_group(first, position="B", source="file")
    store.record_capture(first, group_a, "fa")
    store.record_capture(first, group_b, "fb")
    store.finish(first, answer_text="B更便宜", a_frame_id="fa", b_frame_id="fb")

    second = store.open_compare("这两个哪个贵")
    store.mark(second, "capturing")
    # 不得因上一次已完成的任务占位而失败
    group_a2 = store.new_group(second, position="A", source="file")
    group_b2 = store.new_group(second, position="B", source="file")
    assert group_a2.group_id != group_a.group_id
    assert group_b2.position == "B"


def test_finished_task_images_stay_readable_as_history() -> None:
    """放下引用 ≠ 丢掉证据：已结束组的 image_ids 仍可从 store 取回（历史追问复用）。"""
    store = ImageTaskStore()
    task = store.open_compare("哪个便宜")
    store.mark(task, "capturing")
    group = store.new_group(task, position="A", source="file")
    store.record_capture(task, group, "frame-a1")
    store.finish(task, answer_text="ok", a_frame_id="frame-a1")

    kept = [item for item in store.groups if item.group_id == group.group_id]
    assert kept and kept[0].image_ids == ["frame-a1"]


def test_engine_second_compare_runs_instead_of_capacity_fallback() -> None:
    """真实入口：`_do_compare` 连续两次都不能落到「没能比出来」的降级话术。"""
    frame_a = load_frame(PRICE_A)
    frame_b = load_frame(PRICE_B)
    scene = _Scene([frame_a, frame_b, frame_a, frame_b])

    async def run():
        engine, spoken = _engine(scene, loop=asyncio.get_running_loop())
        for question in ("哪个便宜", "这两个哪个贵"):
            await engine._do_compare(_compare_intent(question), engine._qa_gen)
            await engine._do_resume(engine._qa_gen)  # 第一件
            await engine._do_resume(engine._qa_gen)  # 第二件
        return spoken

    spoken = asyncio.run(run())
    assert len(scene.compare_calls) == 2, f"第二次比较没有真的执行：{scene.compare_calls!r}；播报={spoken!r}"
    assert [kind for kind, _ in spoken if kind == "answer"] == ["answer", "answer"], spoken
    assert FALLBACK not in [text for _, text in spoken]


# ----------------------------------------------------------------------
# 2) 历史图引用不扣次数、不启动 180 秒采集预算
# ----------------------------------------------------------------------
def test_failed_second_position_keeps_first_group_evidence() -> None:
    """第二张失败 → 任务 close：第一张的证据引用必须留在 store 里，不能随任务一起消失。"""
    store = ImageTaskStore()
    task = store.open_compare("这两样哪个贵")
    store.mark(task, "capturing")
    group_a = store.new_group(task, position="A", source="camera")
    stored = store.record_capture(task, group_a, "frame-a1")

    # 位置 B 取图失败，上层按 error 结束任务
    store.close(task, reason="error")

    kept = [item for item in store.groups if item.group_id == stored["group_id"]]
    assert kept and kept[0].image_ids == ["frame-a1"], "第一张证据引用不应丢失"
    assert kept[0].position == "A"
    # 证据保留不代表还能续做：任务已 closed，resume_target 不再返回它（见报告 M3 事项）
    assert store.resume_target() is None


def test_history_reference_does_not_consume_position_budget() -> None:
    store = ImageTaskStore()
    task = store.open_compare("和刚才那张比一下")
    group = store.new_group(task, position="A", source="history")
    snapshot = store.record_capture(task, group, "hist-frame-1")

    assert group.captures == 0, snapshot
    assert snapshot["remaining_captures"] == group.max_captures
    assert task.t08_deadline_ms is None, "只分析历史图不应启动 180 秒采集预算"
    assert group.image_ids == ["hist-frame-1"], "历史帧仍须留下证据引用"


def test_history_reference_leaves_camera_budget_untouched() -> None:
    """同一任务里先引用历史图、再拍一张：拍摄次数仍从 0 开始计。"""
    store = ImageTaskStore()
    task = store.open_compare("先看旧的再拍新的")
    historical = store.new_group(task, position="A", source="history")
    store.record_capture(task, historical, "old-frame")

    camera = store.new_group(task, position="B", source="camera")
    store.record_capture(task, camera, "new-frame")
    assert camera.captures == 1
    assert historical.captures == 0


# ----------------------------------------------------------------------
# 3) 两张相同画面不能冒充两样东西的比较结论
# ----------------------------------------------------------------------
def test_compare_refuses_identical_frames() -> None:
    """同一张图当 A/B（内容摘要相同）时，不能给出「比较结果」式结论。"""
    frame = load_frame(PRICE_A)
    same = load_frame(PRICE_A)
    assert same.frame_id == frame.frame_id, "同内容文件必须得到同一 frame_id"

    service = _service({frame.frame_id: ["价格 12.50 元"]})
    answer = service.compare(frame, same, "哪个更便宜")

    assert "same_frame" in answer.degraded, answer.degraded
    assert "比较结果" not in answer.text, answer.text
    assert answer.items == (), answer.items


def test_compare_distinct_frames_still_reports_result() -> None:
    """不同画面仍必须正常比较（不能把正常双图一起挡掉）。"""
    frame_a = load_frame(PRICE_A)
    frame_b = load_frame(PRICE_B)
    service = _service({frame_a.frame_id: ["价格 12.50 元"], frame_b.frame_id: ["价格 9.90 元"]})
    answer = service.compare(frame_a, frame_b, "哪个更便宜")

    assert "same_frame" not in answer.degraded
    assert answer.common_items == ("价格",)
    assert "比较结果" in answer.text
    assert answer.a_frame_id == frame_a.frame_id and answer.b_frame_id == frame_b.frame_id


def test_compare_identical_frames_keeps_both_frame_ids() -> None:
    """拒绝冒充时仍要如实交代两个位置的来源帧（便于上层提示「换一样东西再拍」）。"""
    frame = load_frame(PRICE_A)
    answer = _service().compare(frame, load_frame(PRICE_A), "哪个贵")
    assert answer.a_frame_id == answer.b_frame_id == frame.frame_id


# ----------------------------------------------------------------------
# 4) 真实上传链路（/api/uploads 的存储实现 media.UploadStore）：
#    同一张照片分别上传到 A/B 两个槽位，仍必须在比较前被认成同一张图。
# ----------------------------------------------------------------------
def test_upload_same_photo_into_both_slots_is_caught_as_same_frame(tmp_path: Path) -> None:
    import media

    store = media.UploadStore(tmp_path / "uploads")
    raw_a = PRICE_A.read_bytes()
    raw_b = PRICE_B.read_bytes()

    slot_a = store.save(raw_a, "A")
    slot_b = store.save(raw_b, "B")
    assert slot_a["image_id"] != slot_b["image_id"], "两次上传应有各自的 image_id"

    frame_a = load_frame(store.get(slot_a["image_id"])[0])
    frame_b = load_frame(store.get(slot_b["image_id"])[0])
    service = _service({frame_a.frame_id: ["价格 12.50 元"], frame_b.frame_id: ["价格 9.90 元"]})

    distinct = service.compare(frame_a, frame_b, "哪个便宜")
    assert "same_frame" not in distinct.degraded and "比较结果" in distinct.text

    # 用户把同一张照片分别传进 A、B 两个槽位
    again = store.save(raw_a, "B")
    frame_b2 = load_frame(store.get(again["image_id"])[0])
    same = service.compare(frame_a, frame_b2, "哪个便宜")
    assert "same_frame" in same.degraded, same.degraded
    assert "比较结果" not in same.text, same.text
