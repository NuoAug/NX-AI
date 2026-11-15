"""阶段 4 图片任务/证据组存储单测：容量、计数、T08、恢复目标与截止。"""

from __future__ import annotations

from aieyes.imagetasks import CapacityError, CompareTask, EvidenceGroup, ImageTaskError, ImageTaskStore


def _store(**kwargs) -> tuple[ImageTaskStore, list[int]]:
    now = [1000]
    store = ImageTaskStore(now_ms=lambda: now[0], **kwargs)
    return store, now


def test_open_compare_basic_fields() -> None:
    store, _now = _store()
    task = store.open_compare("  比较一下这两瓶水  ")
    assert task.task_id == "cmp-001"
    assert task.question == "比较一下这两瓶水"
    assert task.state == "open"
    assert task.created_ms == 1000 and task.deadline_ms == 1000 + 900_000
    assert task.finished is False


def test_open_compare_default_question() -> None:
    store, _now = _store()
    task = store.open_compare("   ")
    assert task.question == "比较一下这两样东西"


def test_capacity_blocks_third_unfinished_task() -> None:
    store, _now = _store()
    store.open_compare("一")
    store.open_compare("二")
    try:
        store.open_compare("三")
    except CapacityError as exc:
        assert set(exc.tasks) == {"cmp-001", "cmp-002"}
    else:  # pragma: no cover - 明确失败路径
        raise AssertionError("第三个未完成任务应被容量挡下")


def test_finish_releases_capacity() -> None:
    store, _now = _store()
    first = store.open_compare("一")
    store.open_compare("二")
    store.finish(first, answer_text="一样贵")
    assert first.state == "answered" and first.answer_text == "一样贵"
    third = store.open_compare("三")
    assert third.task_id == "cmp-003"
    assert len(store.unfinished()) == 2


def test_groups_per_position_and_image_limit() -> None:
    store = ImageTaskStore()
    task = store.open_compare("比较")
    group_a = store.new_group(task, position="A", source="file")
    group_b = store.new_group(task, position="B", source="camera")
    assert task.group_a is group_a and task.group_b is group_b
    assert group_a.position == "A" and group_b.source == "camera"
    # 每组最多 2 张，第三次开组超容量
    try:
        store.new_group(task, position="A", source="file")
    except CapacityError:
        pass
    else:  # pragma: no cover
        raise AssertionError("第三组证据应被容量挡下")


def test_record_capture_counts_and_keeps_latest_two() -> None:
    store, _now = _store()
    task = store.open_compare("比较")
    group = store.new_group(task, position="A", source="file")
    snap1 = store.record_capture(task, group, "f1")
    assert snap1["captures"] == 1 and snap1["remaining_captures"] == 2
    assert task.t08_deadline_ms == 1000 + 180_000
    store.record_capture(task, group, "f2")
    store.record_capture(task, group, "f3")
    assert group.image_ids == ["f2", "f3"], group.image_ids
    assert group.captures == 3 and group.remaining_captures == 0
    try:
        store.record_capture(task, group, "f4")
    except ImageTaskError:
        pass
    else:  # pragma: no cover
        raise AssertionError("同一位置第 4 次取图应失败")


def test_t08_deadline_only_started_once() -> None:
    store, now = _store()
    task = store.open_compare("比较")
    group = store.new_group(task, position="A", source="file")
    store.record_capture(task, group, "f1")
    first = task.t08_deadline_ms
    now[0] = 5000
    store.record_capture(task, group, "f2")
    assert task.t08_deadline_ms == first == 1000 + 180_000


def test_t08_expired_property_uses_deadline() -> None:
    from aieyes.clock import mono_ms

    task = CompareTask(
        task_id="t",
        question="q",
        created_ms=0,
        deadline_ms=mono_ms() + 10_000,
        t08_deadline_ms=mono_ms() - 1,
    )
    assert task.t08_expired is True
    task.t08_deadline_ms = mono_ms() + 60_000
    assert task.t08_expired is False
    task2 = CompareTask(task_id="t2", question="q", created_ms=0, deadline_ms=mono_ms() + 10_000)
    assert task2.t08_expired is False, "未开始采集时 T08 不应过期"


def test_resume_target_prefers_latest_unfinished() -> None:
    store, now = _store()
    first = store.open_compare("一")
    now[0] = 2000
    second = store.open_compare("二")
    store.mark(first, "paused", reason="barge_in")
    store.mark(second, "paused", reason="barge_in")
    target = store.resume_target()
    assert target is second, "应恢复最近创建的任务"
    store.finish(second, answer_text="好了")
    assert store.resume_target() is first
    store.close(first)
    assert store.resume_target() is None


def test_mark_rejects_finished_task() -> None:
    store, _now = _store()
    task = store.open_compare("一")
    store.finish(task, answer_text="x")
    try:
        store.mark(task, "capturing")
    except ImageTaskError:
        pass
    else:  # pragma: no cover
        raise AssertionError("已结束任务不能再改状态")


def test_close_sets_closed_and_pause_reason() -> None:
    store, _now = _store()
    task = store.open_compare("一")
    store.close(task, reason="error")
    assert task.state == "closed" and task.finished is True
    assert task.pause_reason == "error"


def test_snapshot_contains_groups_and_budget() -> None:
    store, _now = _store()
    task = store.open_compare("比较")
    group = store.new_group(task, position="A", source="file")
    store.record_capture(task, group, "f1")
    snap = task.snapshot()
    assert snap["task_id"] == task.task_id and snap["state"] == "capturing" or snap["state"] == "open"
    assert snap["a"]["group_id"] == group.group_id
    assert snap["a"]["captures"] == 1 and snap["a"]["remaining_captures"] == 2
    assert snap["b"] is None
    assert snap["t08_deadline_ms"] == 1000 + 180_000
    assert snap["extra_wait_used"] is False


def test_cleanup_keeps_referenced_groups() -> None:
    from aieyes.clock import mono_ms

    store = ImageTaskStore()
    task = store.open_compare("一")
    group = store.new_group(task, position="A", source="file")
    group.deadline_ms = mono_ms() - 1  # 组已过期
    store.cleanup()
    assert group in store.groups, "未完成任务引用的组不能清掉"
    store.close(task)
    store.cleanup()
    assert group not in store.groups


def test_group_expired_property() -> None:
    group = EvidenceGroup(
        group_id="g", position="A", source="file", created_ms=0, deadline_ms=1
    )
    assert group.expired is True
    live = EvidenceGroup(
        group_id="g2", position="B", source="file", created_ms=0, deadline_ms=mono_huge()
    )
    assert live.expired is False


def mono_huge() -> int:
    from aieyes.clock import mono_ms

    return mono_ms() + 60_000


def test_compare_task_is_exported() -> None:
    assert isinstance(CompareTask.__dataclass_fields__, dict)
