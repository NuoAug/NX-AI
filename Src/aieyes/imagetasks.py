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
    released: bool = False

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
    #: 位置级取图次数（A/B），**跨证据组**延续：换物/重拍换新组也不回到 0
    #: （功能说明「拍摄预算与有限失败处理」「7.6-5」：每位置总拍摄 3 次不因换面、
    #: 换物或失败原因变化重置）。
    position_captures: dict[str, int] = field(default_factory=dict)
    #: 任务结束（answered/failed/closed）时刻；用于判断「刚才那两件」是否还值得
    #: 直接复用，避免很久以后又把旧比较结果当成当前画面。
    finished_ms: int = 0

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
        # 单目标（不走双图任务）的取图额度：按「图片任务」窗口延续，
        # 换问法/翻面/换物都不重置（功能说明「拍摄预算与有限失败处理」：
        # 单目标采集最多 3 次、不能通过换问法重置原图时间或拍摄次数）。
        self._single_captures = 0
        self._single_started_ms: int | None = None
        self._target_group: EvidenceGroup | None = None

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
        task.finished_ms = self._now()
        self._release_groups(task)

    def close(self, task: CompareTask, *, reason: str = "closed") -> None:
        task.state = "closed"
        task.pause_reason = reason
        task.finished_ms = self._now()
        self._release_groups(task)

    def _release_groups(self, task: CompareTask) -> None:
        """完成/取消时只放下本次引用（组本身留给历史缓存/其他任务）。"""
        held = {
            id(group)
            for other in self.unfinished()
            if other is not task
            for group in (other.group_a, other.group_b)
            if group is not None
        }
        for group in (task.group_a, task.group_b):
            if group is None or id(group) in held:
                continue
            # 放下容量占用，但保留在 self.groups：历史追问仍可引用其 image_ids。
            group.released = True

    # -- 证据组 -------------------------------------------------------
    def new_group(self, task: CompareTask, *, position: str, source: str) -> EvidenceGroup:
        live = [group for group in self.groups if not group.expired and not group.released]
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

    def record_capture(
        self,
        task: CompareTask | None,
        group: EvidenceGroup,
        frame_id: str,
    ) -> dict[str, object]:
        """登记一次成功取图：扣该位置次数、必要时启动 T08（历史图引用两者都不做）。

        ``task`` 为 ``None`` 表示单目标取图（不走双图任务）：额度记在 store 上，
        按「图片任务」窗口延续，换问法不重置。
        """
        if group.source == "history":
            # 模块约定：只分析历史图不扣次数、不启动 T08；仍留下证据引用。
            if frame_id not in group.image_ids:
                group.image_ids.append(frame_id)
                if len(group.image_ids) > group.max_images:
                    group.image_ids.pop(0)
            return {
                "group_id": group.group_id,
                "position": group.position,
                "frame_id": frame_id,
                "captures": group.captures,
                "remaining_captures": group.remaining_captures,
                "t08_deadline_ms": None if task is None else task.t08_deadline_ms,
            }
        if task is None:
            return self.record_single_capture(frame_id, group=group)
        # 位置额度以任务级计数为准：换物/重拍换了新组也不能回到 3 次。
        used = max(task.position_captures.get(group.position, 0), group.captures)
        if used >= self.max_captures_per_position:
            raise ImageTaskError(f"位置 {group.position} 的取图次数已用尽")
        group.add_frame(frame_id)
        task.position_captures[group.position] = used + 1
        if task.t08_deadline_ms is None:
            task.t08_deadline_ms = self._now() + self.t08_ms
        return {
            "group_id": group.group_id,
            "position": group.position,
            "frame_id": frame_id,
            "captures": task.position_captures[group.position],
            "remaining_captures": max(0, self.max_captures_per_position - used - 1),
            "t08_deadline_ms": task.t08_deadline_ms,
        }

    # -- 位置占用释放（重拍 / 换物 / 翻面）-----------------------------
    def release_position(self, task: CompareTask, position: str) -> dict[str, object]:
        """让某位置的**当前证据**退出使用，另一位置保持不动。

        只放下「位置占用」：证据组仍留在 :attr:`groups` 里（历史追问还能引用它的
        ``image_ids``，功能说明 L463），帧对象原样返回给调用方而不是销毁；下次该位置
        需要出图时重新 ``new_group`` + 取图。位置额度（``position_captures``）保留，
        重拍不退回次数（7.6-3/7.6-5）。
        """
        if position not in ("A", "B"):
            raise ImageTaskError(f"未知位置：{position!r}")
        group = task.group_a if position == "A" else task.group_b
        frame = task.frames.pop(position, None)
        if position == "A":
            task.group_a = None
        else:
            task.group_b = None
        if group is not None:
            group.released = True
        # 该位置要重新准备：完成一次取图后确需翻面/换物，可建新的准备阶段（L200）。
        task.t08_deadline_ms = None
        used = task.position_captures.get(position, 0)
        return {
            "task_id": task.task_id,
            "position": position,
            "released_group": "" if group is None else group.group_id,
            "frame": frame,
            "captures": used,
            "remaining_captures": max(0, self.max_captures_per_position - used),
            "t08_deadline_ms": task.t08_deadline_ms,
        }

    # -- 只读访问器：给上层实现"复用上次那张图"（C6 历史图引用）-------
    def group(self, group_id: str) -> EvidenceGroup | None:
        """按 id 取证据组（只读；已释放/已过期的组仍可取，便于追溯）。"""
        for group in self.groups:
            if group.group_id == group_id:
                return group
        if self._target_group is not None and self._target_group.group_id == group_id:
            return self._target_group
        return None

    def _task_id_of(self, group: EvidenceGroup) -> str:
        for task in self.tasks:
            if task.group_a is group or task.group_b is group:
                return task.task_id
        return ""

    def _evidence_row(self, group: EvidenceGroup) -> dict[str, object]:
        return {
            "frame_id": group.image_ids[-1] if group.image_ids else "",
            "group_id": group.group_id,
            "position": group.position,
            "source": group.source,
            "created_ms": group.created_ms,
            "task_id": self._task_id_of(group),
            "captures": group.captures,
            "image_ids": list(group.image_ids),
            "released": group.released,
        }

    def latest_evidence(self, position: str | None = None) -> dict[str, object] | None:
        """最近一张可用证据（按证据组创建顺序取最新、未释放、未过期的那组）。

        ``position`` 传 ``"A"``/``"B"`` 只取该位置；传 ``"S"`` 取单目标；``None`` 不限。
        上层据此判断"上次那张"能不能复用（帧对象仍需调用方自持 `frame_id → frame`）。
        """
        rows = [
            (index, group)
            for index, group in enumerate(self.groups)
            if group.image_ids
            and not group.expired
            and not group.released
            and (position is None or group.position == position)
        ]
        if position in (None, "S") and self._target_group is not None:
            target = self._target_group
            if target.image_ids and not target.expired and not target.released:
                rows.append((len(self.groups), target))
        if not rows:
            return None
        # 同一毫秒内建的组按入列顺序区分先后（时间戳可能相等）。
        _, newest = max(rows, key=lambda item: (item[1].created_ms, item[0]))
        return self._evidence_row(newest)

    def history_index(self, *, limit: int = 8) -> list[dict[str, object]]:
        """证据索引（按创建时间倒序），含已释放组：上层用来找"上一个任务留下的图"。"""
        rows = [
            (index, group)
            for index, group in enumerate(self.groups)
            if group.image_ids and not group.expired
        ]
        if self._target_group is not None and self._target_group.image_ids:
            target = self._target_group
            if not target.expired:
                rows.append((len(self.groups), target))
        rows.sort(key=lambda item: (item[1].created_ms, item[0]), reverse=True)
        return [self._evidence_row(group) for _, group in rows[: max(0, limit)]]

    # -- 单目标（不建双图任务）的取图额度 ------------------------------
    def _single_roll_if_window_over(self) -> None:
        """图片任务窗口（默认 15 分钟）过去后，单目标额度视为新任务重新起算。"""
        if self._single_started_ms is None:
            return
        if self._now() - self._single_started_ms < self.task_deadline_ms:
            return
        self._single_captures = 0
        self._single_started_ms = None
        self._target_group = None

    @property
    def single_target_captures(self) -> int:
        self._single_roll_if_window_over()
        return self._single_captures

    @property
    def single_target_remaining(self) -> int:
        return max(0, self.max_captures_per_position - self.single_target_captures)

    @property
    def single_target_budget_exhausted(self) -> bool:
        return self.single_target_remaining == 0

    def target_budget_exhausted(self, *, source: str = "camera") -> bool:
        """单目标还有没有取图次数（上层在**发出取图请求之前**问这个）。"""
        del source  # 目前 file/camera 共用同一份额度，保留参数以便上层统一传参
        return self.single_target_budget_exhausted

    def current_target_group(self, *, source: str = "camera") -> EvidenceGroup:
        """单目标当前证据组（惰性创建；不占双图的两组容量）。"""
        self._single_roll_if_window_over()
        group = self._target_group
        if group is None or group.expired or group.released:
            now = self._now()
            group = EvidenceGroup(
                group_id=f"grp-{next(self._seq):03d}",
                position="S",
                source=source,
                created_ms=now,
                deadline_ms=now + self.group_deadline_ms,
                max_images=self.max_images_per_group,
                max_captures=self.max_captures_per_position,
            )
            self._target_group = group
        group.captures = self._single_captures
        return group

    def record_single_capture(
        self,
        frame_id: str,
        *,
        group: EvidenceGroup | None = None,
    ) -> dict[str, object]:
        """登记一次单目标实际取图（发出请求即计一次，失败/超时/重拍都算）。"""
        self._single_roll_if_window_over()
        if self._single_captures >= self.max_captures_per_position:
            raise ImageTaskError(
                f"目标 的取图次数已用尽（每目标最多 {self.max_captures_per_position} 次）"
            )
        group = group or self.current_target_group()
        self._single_captures += 1
        if self._single_started_ms is None:
            self._single_started_ms = self._now()
        if frame_id and frame_id not in group.image_ids:
            group.image_ids.append(frame_id)
            if len(group.image_ids) > group.max_images:
                group.image_ids.pop(0)
        group.captures = self._single_captures
        return {
            "group_id": group.group_id,
            "position": group.position,
            "frame_id": frame_id,
            "captures": self._single_captures,
            "remaining_captures": max(0, self.max_captures_per_position - self._single_captures),
            "t08_deadline_ms": None,
        }

    def clear_all(self) -> dict[str, int]:
        """清除全部图片任务与证据组（阶段 5 “清除本轮资料”）。"""
        tasks = len(self.tasks)
        groups = len(self.groups)
        frames = sum(len(task.frames) for task in self.tasks)
        self.tasks.clear()
        self.groups.clear()
        self._single_captures = 0
        self._single_started_ms = None
        self._target_group = None
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
