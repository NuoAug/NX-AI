"""M5 第六轮：图片任务位置额度、换物/重拍释放与单目标取图计数的定向契约。

对应功能流程说明（`Docs/需求/视觉语言交互子系统功能流程说明.md`）：

- ``L181``/``L360``：单目标最多 3 次、双图每位置 3 次，「实际发出取图请求即计一次」，
  且每位置 3 次**不因换面/换物重置**。
- ``L209``：明确重拍后旧照片不再作为该位置当前证据；另一位置保留；
  「重拍失败也不能默默回退旧图冒充成功」。
- ``L200``：完成一次实际取图后，确需用户再次调整/翻面，可建**新的** 60 秒准备阶段。
- ``L463``：完成后的新追问引用原证据组 —— 旧图作为**历史引用**仍需可读。

这些用例只走本机软件路径（假 OCR/VLM、文件图源），不涉及相机与模型服务。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aieyes.imagetasks import ImageTaskError, ImageTaskStore
from aieyes.scene import SceneService
from aieyes.vision.file_camera import load_frame

FIXTURES = Path(__file__).resolve().parents[2] / "Tests" / "Fixtures" / "scene"
if not FIXTURES.is_dir():
    FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "scene"
PRICE_A = FIXTURES / "compare_price_a.png"
PRICE_B = FIXTURES / "compare_price_b.png"


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


def _service() -> SceneService:
    return SceneService(ocr=_Ocr(), vlm=_Vlm())


def _store() -> ImageTaskStore:
    return ImageTaskStore()


# ----------------------------------------------------------------------
# 1) 换物后位置额度必须连续：不能因为新建证据组就回到 3 次
# ----------------------------------------------------------------------
def test_replacing_target_keeps_position_capture_budget() -> None:
    store = _store()
    task = store.open_compare("比较一下这两样东西")
    group = store.new_group(task, position="A", source="camera")
    for index in range(3):
        store.record_capture(task, group, f"frame-a{index}")

    # 用户换物：释放 A 位置当前证据，重新建组取图（engine 侧就是这么做的）
    store.release_position(task, "A")
    fresh = store.new_group(task, position="A", source="camera")
    with pytest.raises(ImageTaskError, match="次数已用尽"):
        store.record_capture(task, fresh, "frame-a-replaced")


def test_replacing_target_reports_remaining_captures() -> None:
    store = _store()
    task = store.open_compare("比较一下这两样东西")
    group = store.new_group(task, position="A", source="camera")
    store.record_capture(task, group, "frame-a0")
    store.record_capture(task, group, "frame-a1")
    released = store.release_position(task, "A")
    assert released["remaining_captures"] == 1
    assert released["released_group"] == group.group_id


# ----------------------------------------------------------------------
# 2) 释放只影响被指定的位置：另一位置证据保留（L209）
# ----------------------------------------------------------------------
def test_release_position_drops_current_frame_keeps_other_position() -> None:
    store = _store()
    task = store.open_compare("比较一下这两样东西")
    group_a = store.new_group(task, position="A", source="camera")
    group_b = store.new_group(task, position="B", source="camera")
    store.record_capture(task, group_a, "frame-a")
    store.record_capture(task, group_b, "frame-b")
    task.frames["A"] = object()
    task.frames["B"] = object()

    store.release_position(task, "B")

    assert "B" not in task.frames
    assert "A" in task.frames
    assert task.group_b is None
    assert task.group_a is group_b is False or task.group_a is group_a
    assert group_b.released is True
    assert group_a.released is False


def test_release_position_keeps_released_images_as_history() -> None:
    store = _store()
    task = store.open_compare("比较一下这两样东西")
    group = store.new_group(task, position="A", source="camera")
    store.record_capture(task, group, "frame-a-old")
    store.release_position(task, "A")

    # L463：旧图作为历史引用仍可读，只是不再是该位置当前证据
    assert group.image_ids == ["frame-a-old"]
    assert group in store.groups


# ----------------------------------------------------------------------
# 3) 释放后重新取图要起新的准备窗口（L200），不是沿用旧的 T08 截止
# ----------------------------------------------------------------------
def test_release_position_restarts_preparation_window() -> None:
    store = _store()
    task = store.open_compare("比较一下这两样东西")
    group = store.new_group(task, position="A", source="camera")
    store.record_capture(task, group, "frame-a0")
    first_deadline = task.t08_deadline_ms
    assert first_deadline is not None

    store.release_position(task, "A")
    assert task.t08_deadline_ms is None

    fresh = store.new_group(task, position="A", source="camera")
    store.record_capture(task, fresh, "frame-a1")
    assert task.t08_deadline_ms is not None


# ----------------------------------------------------------------------
# 4) 释放要放掉证据组容量，否则换物一次就卡在「最多两组」
# ----------------------------------------------------------------------
def test_release_position_frees_group_capacity() -> None:
    store = _store()
    task = store.open_compare("比较一下这两样东西")
    store.new_group(task, position="A", source="camera")
    store.new_group(task, position="B", source="camera")

    store.release_position(task, "A")
    store.release_position(task, "B")

    store.new_group(task, position="A", source="camera")
    store.new_group(task, position="B", source="camera")


# ----------------------------------------------------------------------
# 5) 单目标（不带任务）取图也要计数，3 次后拒绝（L181/L360）
# ----------------------------------------------------------------------
def test_single_target_capture_counts_without_task() -> None:
    store = _store()
    store.record_single_capture("single-frame-0")
    store.record_single_capture("single-frame-1")
    assert store.single_target_captures == 2
    assert store.single_target_remaining == 1


def test_single_target_budget_exhausted_after_three() -> None:
    store = _store()
    for index in range(3):
        store.record_single_capture(f"single-frame-{index}")
    assert store.single_target_budget_exhausted is True
    with pytest.raises(ImageTaskError, match="次数已用尽"):
        store.record_single_capture("single-frame-3")


def test_single_target_window_expiry_starts_new_picture_task() -> None:
    """15 分钟图片任务窗口过期后视为新任务，额度重新起算（「拍摄预算」章）。"""
    clock = {"now": 1_000_000}

    def now_ms() -> int:
        return clock["now"]

    store = ImageTaskStore(now_ms=now_ms)
    for index in range(3):
        store.record_single_capture(f"single-frame-{index}")
    assert store.single_target_budget_exhausted is True

    clock["now"] += store.task_deadline_ms + 1
    assert store.single_target_captures == 0
    assert store.single_target_budget_exhausted is False
    store.record_single_capture("single-frame-after-window")
    assert store.single_target_remaining == 2


# ----------------------------------------------------------------------
# 6) 场景服务必须把本轮实际用过的帧交出来（engine 复用同图追问靠它）
# ----------------------------------------------------------------------

def test_service_answer_exposes_last_frame() -> None:
    service = _service()
    frame = load_frame(PRICE_A)
    service.answer("看看这是什么", frame=frame)
    assert service.last_frame is frame


def test_service_answer_without_frame_records_acquired_frame() -> None:
    class _Provider:
        def __init__(self) -> None:
            self.frame = load_frame(PRICE_B)
            self.calls = 0

        def acquire(self, purpose: str = "vqa"):
            self.calls += 1
            return self.frame

    provider = _Provider()
    service = SceneService(ocr=_Ocr(), vlm=_Vlm(), provider=provider)
    service.answer("看看这是什么")
    assert provider.calls == 1
    assert service.last_frame is provider.frame


def test_reset_last_frame_drops_cached_frame() -> None:
    """取消/新一轮开始清缓存帧：否则同图追问会拿上一轮的旧图。"""
    service = _service()
    service.answer("看看这是什么", frame=load_frame(PRICE_A))
    assert service.last_frame is not None
    service.reset_last_frame()
    assert service.last_frame is None


# ----------------------------------------------------------------------
# 7) 只读证据访问器：C6「沿用上次那张图」靠它选证据（帧对象由上层自持）
# ----------------------------------------------------------------------
def test_latest_evidence_returns_newest_group() -> None:
    store = _store()
    task = store.open_compare("比较一下这两样东西")
    group_a = store.new_group(task, position="A", source="camera")
    group_b = store.new_group(task, position="B", source="camera")
    store.record_capture(task, group_a, "frame-a")
    store.record_capture(task, group_b, "frame-b")

    newest = store.latest_evidence()
    assert newest is not None
    assert newest["frame_id"] == "frame-b"
    assert newest["position"] == "B"
    assert newest["task_id"] == task.task_id
    assert newest["released"] is False

    only_a = store.latest_evidence("A")
    assert only_a is not None and only_a["frame_id"] == "frame-a"


def test_latest_evidence_skips_released_group() -> None:
    """位置释放之后旧图不能再当"上次那张"直接用；它只留在 history_index 里。"""
    store = _store()
    task = store.open_compare("比较一下这两样东西")
    group = store.new_group(task, position="A", source="camera")
    store.record_capture(task, group, "frame-a-old")
    store.release_position(task, "A")

    assert store.latest_evidence("A") is None
    rows = store.history_index()
    assert [row["frame_id"] for row in rows] == ["frame-a-old"]
    assert rows[0]["released"] is True
    assert rows[0]["image_ids"] == ["frame-a-old"]


def test_history_index_is_newest_first_and_keeps_released() -> None:
    store = _store()
    task = store.open_compare("比较一下这两样东西")
    group_a = store.new_group(task, position="A", source="camera")
    store.record_capture(task, group_a, "frame-a")
    store.finish(task, answer_text="比较结果：B更贵。", a_frame_id="frame-a", b_frame_id="frame-b")
    second = store.open_compare("再看看别的")
    group_a2 = store.new_group(second, position="A", source="camera")
    store.record_capture(second, group_a2, "frame-a2")

    rows = store.history_index(limit=4)
    assert [row["frame_id"] for row in rows] == ["frame-a2", "frame-a"]
    assert rows[0]["released"] is False   # 当前在用
    assert rows[1]["released"] is True    # 上一个任务放下引用后仍可追溯（L463）


def test_group_lookup_by_id() -> None:
    store = _store()
    task = store.open_compare("比较一下这两样东西")
    group = store.new_group(task, position="A", source="camera")
    assert store.group(group.group_id) is group
    assert store.group("grp-does-not-exist") is None


def test_history_index_respects_limit() -> None:
    store = _store()
    for index in range(3):
        task = store.open_compare(f"问题{index}")
        group = store.new_group(task, position="A", source="camera")
        store.record_capture(task, group, f"frame-{index}")
        store.finish(task, answer_text="比较结果：B更贵。")
    assert len(store.history_index(limit=2)) == 2
    assert len(store.history_index(limit=0)) == 0
