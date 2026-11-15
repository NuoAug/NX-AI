"""阶段 4：图片任务、证据组与双图比较的容量/计数/引用管理。

方案第六节要求（`Docs/方案/DeepSeek-V4.1-Flash1017_…含完整方案.md` L328–376）：

- 未完成图片任务最多 2 个；实际证据组最多 2 个；每组原图最多 2 张（单目标可留前后面）；
- 取图次数按「位置」计，同一位置最多 3 次（T10）；双图累计采集从第一次需要采集
  起算 180 秒（T08），准备/额外等待另有 T06（本模块只记录是否已用）；
- 普通双图比较 A/B 各一组；只分析历史图不扣次数、不启动 T08；
- 比较完成/取消只放下本次引用；任务硬截止 15 分钟（T20 口径由上层计时）。

本模块只做纯粹的状态与预算记录，不做 IO、不播报；上层（状态机）决定问什么、
何时取图，并把这些记录写进证据。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable

from aieyes.clock import mono_ms
from aieyes.errors import AieyesError


class ImageTaskError(AieyesError):
    """图片任务状态/预算错误。"""


class CapacityError(ImageTaskError):
    """容量不足：需要上层询问结束哪个任务或释放哪组引用。"""

    def __init__(self, message: str, *, tasks: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.tasks = tasks


@dataclass
class EvidenceGroup:
    """一组证据：一个位置（A/B/单目标）的原图，最多 2 张。"""

    group_id: str
    position: str
    source: str  # "camera" | "file" | "history"
    created_ms: int
    deadline_ms: int
    image_ids: list[str] = field(default_factory=list)
    captures: int = 0
    max_images: int = 2
    max_captures: int = 3

    @property
    def expired(self) -> bool:
        return mono_ms() >= self.deadline_ms

    @property
    def remaining_captures(self) -> int:
        return max(0, self.max_captures - self.captures)

    def add_frame(self, frame_id: str) -> None:
        if self.captures >= self.max_captures:
            raise ImageTaskError(f"位置 {self.position} 的取图次数已用尽")
        self.captures += 1
        if frame_id not in self.image_ids:
            self.image_ids.append(frame_id)
            if len(self.image_ids) > self.max_images:
                self.image_ids.pop(0)  # 只保留最近 max_images 张（前后面/补拍）


@dataclass
class CompareTask:
    """一个双图比较任务（阶段 4）。"""

    task_id: str
    question: str
    created_ms: int
    deadline_ms: int
    state: str = "open"  # open|capturing|analyzing|paused|answered|failed|closed
    group_a: EvidenceGroup | None = None
    group_b: EvidenceGroup | None = None
    t08_deadline_ms: int | None = None
    extra_wait_used: bool = False
    answer_text: str = ""
    answered_items: tuple[str, ...] = ()
    missing_items: tuple[str, ...] = ()
    a_frame_id: str = ""
    b_frame_id: str = ""
    frames: dict[str, object] = field(default_factory=dict, repr=False)
    pause_reason: str = ""

    @property
    def finished(self) -> bool:
        return self.state in ("answered", "failed", "closed")

    @property
    def t08_expired(self) -> bool:
        return self.t08_deadline_ms is not None and mono_ms() >= self.t08_deadline_ms

    @property
    def expired(self) -> bool:
        return mono_ms() >= self.deadline_ms

    def snapshot(self) -> dict[str, object]:
        """引用快照（写证据用）：来源、帧 id、次数、剩余量、截止。"""
        return {
            "task_id": self.task_id,
            "state": self.state,
            "question": self.question,
            "created_ms": self.created_ms,
            "deadline_ms": self.deadline_ms,
            "t08_deadline_ms": self.t08_deadline_ms,
            "extra_wait_used": self.extra_wait_used,
            "a": _group_snapshot(self.group_a),
            "b": _group_snapshot(self.group_b),
        }


def _group_snapshot(group: EvidenceGroup | None) -> dict[str, object] | None:
    if group is None:
        return None
    return {
        "group_id": group.group_id,
        "position": group.position,
        "source": group.source,
        "image_ids": list(group.image_ids),
        "captures": group.captures,
        "remaining_captures": group.remaining_captures,
        "created_ms": group.created_ms,
        "deadline_ms": group.deadline_ms,
    }


class ImageTaskStore:
    """未完成图片任务与证据组的容量/计数登记（单进程内有效）。"""

    def __init__(
        self,
        *,
        max_unfinished: int = 2,
        max_groups: int = 2,
        max_images_per_group: int = 2,
        max_captures_per_position: int = 3,
        t08_ms: int = 180_000,
        task_deadline_ms: int = 900_000,
        group_deadline_ms: int = 900_000,
        now_ms=mono_ms,
    ) -> None:
        self.max_unfinished = max_unfinished
        self.max_groups = max_groups
        self.max_images_per_group = max_images_per_group
        self.max_captures_per_position = max_captures_per_position
        self.t08_ms = t08_ms
        self.task_deadline_ms = task_deadline_ms
        self.group_deadline_ms = group_deadline_ms
        self._now = now_ms
        self._seq = itertools.count(1)
        self.tasks: list[CompareTask] = []
        self.groups: list[EvidenceGroup] = []

    # -- 任务 ---------------------------------------------------------
    def unfinished(self) -> list[CompareTask]:
        return [task for task in self.tasks if not task.finished]

    def open_compare(self, question: str) -> CompareTask:
        active = self.unfinished()
        if len(active) >= self.max_unfinished:
            raise CapacityError(
                "已经有两个没做完的看图任务，先结束一个才能开新的",
                tasks=tuple(task.task_id for task in active),
            )
        now = self._now()
        task = CompareTask(
            task_id=f"cmp-{next(self._seq):03d}",
            question=question.strip() or "比较一下这两样东西",
            created_ms=now,
            deadline_ms=now + self.task_deadline_ms,
        )
        self.tasks.append(task)
        return task

    def resume_target(self) -> CompareTask | None:
        """最近一个未完成任务（被打断后要恢复的对象）。"""
        active = [task for task in self.unfinished() if task.state in ("paused", "open", "capturing", "analyzing")]
        return max(active, key=lambda t: t.created_ms) if active else None

    def mark(self, task: CompareTask, state: str, *, reason: str = "") -> None:
        if task.finished:
            raise ImageTaskError(f"任务 {task.task_id} 已结束，不能再改状态")
        task.state = state
        if reason:
            task.pause_reason = reason

    def finish(
        self,
        task: CompareTask,
        *,
        answer_text: str,
        answered_items: Iterable[str] = (),
        missing_items: Iterable[str] = (),
        a_frame_id: str = "",
        b_frame_id: str = "",
    ) -> None:
        task.answer_text = answer_text
        task.answered_items = tuple(answered_items)
        task.missing_items = tuple(missing_items)
        task.a_frame_id = a_frame_id
        task.b_frame_id = b_frame_id
        task.state = "answered"
        self._release_groups(task)

    def close(self, task: CompareTask, *, reason: str = "closed") -> None:
        task.state = "closed"
        task.pause_reason = reason
        self._release_groups(task)

    def _release_groups(self, task: CompareTask) -> None:
        """完成/取消时只放下本次引用（组本身留给历史缓存/其他任务）。"""
        for group in (task.group_a, task.group_b):
            if group is not None and group not in [t.group_a for t in self.unfinished()] + [
                t.group_b for t in self.unfinished()
            ]:
                pass  # 组保留在 self.groups 供历史复用；引用由 task 字段体现

    # -- 证据组 -------------------------------------------------------
    def new_group(self, task: CompareTask, *, position: str, source: str) -> EvidenceGroup:
        live = [group for group in self.groups if not group.expired]
        if len(live) >= self.max_groups:
            raise CapacityError(
                "已经有两组图了，要用新图得先放下旧的一组",
                tasks=tuple(t.task_id for t in self.unfinished()),
            )
        now = self._now()
        group = EvidenceGroup(
            group_id=f"grp-{next(self._seq):03d}",
            position=position,
            source=source,
            created_ms=now,
            deadline_ms=now + self.group_deadline_ms,
            max_images=self.max_images_per_group,
            max_captures=self.max_captures_per_position,
        )
        self.groups.append(group)
        if position == "A":
            task.group_a = group
        elif position == "B":
            task.group_b = group
        return group

    def record_capture(self, task: CompareTask, group: EvidenceGroup, frame_id: str) -> dict[str, object]:
        """登记一次成功取图：扣该位置次数、必要时启动 T08。"""
        group.add_frame(frame_id)
        if task.t08_deadline_ms is None:
            task.t08_deadline_ms = self._now() + self.t08_ms
        return {
            "group_id": group.group_id,
            "position": group.position,
            "frame_id": frame_id,
            "captures": group.captures,
            "remaining_captures": group.remaining_captures,
            "t08_deadline_ms": task.t08_deadline_ms,
        }

    def clear_all(self) -> dict[str, int]:
        """清除全部图片任务与证据组（阶段 5 “清除本轮资料”）。"""
        tasks = len(self.tasks)
        groups = len(self.groups)
        frames = sum(len(task.frames) for task in self.tasks)
        self.tasks.clear()
        self.groups.clear()
        return {"tasks": tasks, "groups": groups, "frames": frames}

    def cleanup(self) -> None:
        """清掉已过期且不再被未完成任务引用的组。"""
        referenced = {
            group.group_id
            for task in self.unfinished()
            for group in (task.group_a, task.group_b)
            if group is not None
        }
        self.groups = [
            group for group in self.groups if not group.expired or group.group_id in referenced
        ]
