"""task-28 接线：引擎侧 C1/C2/C3/C4/C7 与 CONTRACT 诚实分支。

口径：真实 ``InteractionEngine`` + 受控 scene / image_tasks / chat 替身，只看
引擎实际发出的调用参数（``scene.answer`` 收到的 ``frame=``）、``timings.records``
证据行、以及实际播报的文本，不使用模型自评。
"""

from __future__ import annotations

import asyncio
from typing import Any

from aieyes.chat import ChatSession
from aieyes.config import AppConfig
from aieyes.events import EventBus
from aieyes.imagetasks import ImageTaskStore
from aieyes.intent import Intent, IntentKind
from aieyes.interaction.engine import InteractionEngine


class _Frontend:
    def __init__(self) -> None:
        self.listening: list[bool] = []
        self.wake: list[bool] = []

    def set_listening(self, value: bool) -> None:
        self.listening.append(value)

    def set_wake_enabled(self, value: bool) -> None:
        self.wake.append(value)


class _Sink:
    is_playing = False

    def cancel(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _Tts:
    def synth(self, text):  # pragma: no cover - 测试不应触发真实 TTS
        raise AssertionError("测试不应触发真实 TTS")


class _Vlm:
    """聊天模型替身：记录 prompt 全文。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt, *, images, max_tokens, timeout_s, temperature=None):
        self.prompts.append(prompt)
        return "聊天回答"


class _Answer:
    def __init__(self, text: str, frame_id: str = "f1") -> None:
        self.text = text
        self.frame_id = frame_id
        self.intent = "look"
        self.degraded: tuple[str, ...] = ()
        self.source_tags: tuple[str, ...] = ()


class _CompareAnswer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.common_items = ("都有把手",)
        self.missing = ()
        self.a_frame_id = "frame-a"
        self.b_frame_id = "frame-b"


class _Provider:
    """取图提供者替身：记录每次 acquire，返回带 frame_id 的帧对象。"""

    def __init__(self) -> None:
        self.acquires: list[str] = []

    def acquire(self, purpose: str):
        self.acquires.append(purpose)
        return _Answer(f"frame-{len(self.acquires)}", frame_id=f"frame-{len(self.acquires)}")


class _Scene:
    """受控 scene：记录每次 answer 收到的 frame，模拟 last_frame / reset_last_frame。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.compares: list[tuple[Any, Any, str]] = []
        self.last_frame: Any | None = None
        self.resets = 0
        self.provider = _Provider()

    def answer(self, question: str, *, intent_hint=None, frame=None):
        self.calls.append({"question": question, "frame": frame, "intent_hint": intent_hint})
        if frame is None:
            self.last_frame = f"frame#{len(self.calls)}"
        else:
            self.last_frame = frame
        return _Answer(f"看到了{question}", frame_id=str(self.last_frame))

    def compare(self, frame_a, frame_b, question: str):
        self.compares.append((frame_a, frame_b, question))
        return _CompareAnswer(f"比较结果：{question}")

    def reset_last_frame(self) -> None:
        self.resets += 1
        self.last_frame = None


def _make(*, with_scene: bool = True, image_tasks: ImageTaskStore | None = None):
    cfg = AppConfig()
    cfg.timers.t11_generate_ms = 5000
    vlm = _Vlm()
    chat = ChatSession(vlm=vlm)
    scene = _Scene() if with_scene else None
    spoken: list[tuple[str, str]] = []

    engine = InteractionEngine(
        cfg=cfg,
        bus=EventBus(),
        frontend=_Frontend(),
        sink=_Sink(),
        tts=_Tts(),
        chat=chat,
        scene=scene,
        image_tasks=image_tasks if image_tasks is not None else ImageTaskStore(),
    )

    async def fake_speak(text: str, *, kind: str, ref_ms=None, meta=None, **kwargs) -> bool:
        spoken.append((kind, text))
        return True

    engine._speak = fake_speak  # type: ignore[method-assign]
    return engine, scene, vlm, spoken


def _names(engine: InteractionEngine) -> list[str]:
    return [str(rec.get("name")) for rec in engine.timings.records]


def _rec(engine: InteractionEngine, name: str) -> dict[str, Any]:
    for rec in engine.timings.records:
        if rec.get("name") == name:
            return rec
    raise AssertionError(f"没有找到证据行 {name}: {_names(engine)}")


# ---------------------------------------------------------------- C1 同图追问


def test_followup_question_reuses_cached_frame() -> None:
    """同图追问复用上一帧（不重拍）：第二问的 frame= 是第一次的帧。"""

    async def run():
        engine, scene, _vlm, _spoken = _make()
        await asyncio.wait_for(engine._do_scene(Intent(IntentKind.LOOK, "这是什么", "这是什么"), engine._qa_gen), timeout=10)
        first_frame = engine._last_scene_frame
        await asyncio.wait_for(engine._do_scene(Intent(IntentKind.LOOK, "什么颜色", "什么颜色"), engine._qa_gen), timeout=10)
        return engine, scene, first_frame

    engine, scene, first_frame = asyncio.run(run())
    assert first_frame is not None, "第一次视觉回答后应缓存帧"
    assert scene.calls[0]["frame"] is None, "第一问应真的取图"
    assert scene.calls[1]["frame"] is first_frame, f"同图追问没复用缓存帧: {scene.calls}"
    reused = [r for r in engine.timings.records if r.get("name") == "scene.answer"]
    assert reused and reused[-1].get("reused") is True, f"复用没有留证据: {reused}"


def test_general_look_question_does_not_reuse_previous_frame() -> None:
    """“眼前有什么/我眼前有什么”每次都要新画面，不能拿上一张图回答。"""

    async def run():
        engine, scene, _vlm, _spoken = _make()
        await asyncio.wait_for(
            engine._do_scene(Intent(IntentKind.LOOK, "眼前有什么", "眼前有什么"), engine._qa_gen),
            timeout=10,
        )
        await asyncio.wait_for(
            engine._do_scene(Intent(IntentKind.LOOK, "我眼前有什么", "我眼前有什么"), engine._qa_gen),
            timeout=10,
        )
        return engine, scene

    engine, scene = asyncio.run(run())
    assert scene.calls[0]["frame"] is None
    assert scene.calls[1]["frame"] is None, f"第二次整景提问又复用了旧帧: {scene.calls}"
    rec = [r for r in engine.timings.records if r.get("name") == "scene.answer"][-1]
    assert rec.get("reused") is False, rec
    assert engine.image_tasks.single_target_captures == 0, "整景观察不应消耗单目标额度"


def test_temporal_followup_captures_fresh_frame() -> None:
    """“现在呢”表示场景可能已变化，必须重新取图。"""

    async def run():
        engine, scene, _vlm, _spoken = _make()
        await asyncio.wait_for(
            engine._do_scene(Intent(IntentKind.LOOK, "这是什么", "这是什么"), engine._qa_gen),
            timeout=10,
        )
        await asyncio.wait_for(
            engine._do_scene(Intent(IntentKind.LOOK, "现在呢", "现在呢"), engine._qa_gen),
            timeout=10,
        )
        return engine, scene

    engine, scene = asyncio.run(run())
    assert scene.calls[1]["frame"] is None, f"“现在呢”复用了旧帧: {scene.calls}"
    rec = [r for r in engine.timings.records if r.get("name") == "scene.answer"][-1]
    assert rec.get("reused") is False, rec
    assert engine.image_tasks.single_target_captures == 2


def test_scene_overview_and_temporal_classification() -> None:
    async def run():
        engine, _scene, _vlm, _spoken = _make()
        assert engine._is_scene_overview("眼前有什么")
        assert engine._is_scene_overview("我眼前有什么")
        assert engine._is_scene_overview("前面有什么")
        assert not engine._is_scene_overview("这是什么")
        assert engine._needs_fresh_scene_frame("现在呢")
        assert engine._needs_fresh_scene_frame("再看看")
        assert not engine._needs_fresh_scene_frame("什么颜色")

    asyncio.run(run())


def test_fresh_capture_request_does_not_reuse_frame() -> None:
    """用户说要重新拍：不得复用旧帧。"""

    async def run():
        engine, scene, _vlm, _spoken = _make()
        await asyncio.wait_for(engine._do_scene(Intent(IntentKind.LOOK, "这是什么", "这是什么"), engine._qa_gen), timeout=10)
        await asyncio.wait_for(
            engine._do_scene(Intent(IntentKind.LOOK, "重新拍一下", "重新拍一下"), engine._qa_gen),
            timeout=10,
        )
        return engine, scene

    engine, scene = asyncio.run(run())
    assert scene.calls[1]["frame"] is None, f"要求重拍却复用了旧帧: {scene.calls}"


# ---------------------------------------------------------------- C7 丢帧


def test_cancel_and_stop_drop_cached_frame() -> None:
    """取消/停止后旧帧必须丢掉，否则新问题会拿旧画面作答。"""

    async def run():
        engine, scene, _vlm, _spoken = _make()
        await asyncio.wait_for(engine._do_scene(Intent(IntentKind.LOOK, "这是什么", "这是什么"), engine._qa_gen), timeout=10)
        assert engine._last_scene_frame is not None
        engine._silence_now()
        after_cancel = engine._last_scene_frame
        await asyncio.wait_for(engine._do_scene(Intent(IntentKind.LOOK, "这是什么", "这是什么"), engine._qa_gen), timeout=10)
        assert engine._last_scene_frame is not None
        await asyncio.wait_for(engine._do_stop(), timeout=10)
        return scene, after_cancel, engine._last_scene_frame

    scene, after_cancel, after_stop = asyncio.run(run())
    assert after_cancel is None, "取消后旧帧没被清掉"
    assert after_stop is None, "停止后旧帧没被清掉"
    assert scene.resets >= 2, f"没有调用 scene.reset_last_frame: resets={scene.resets}"


def test_clear_drops_cached_frame() -> None:
    async def run():
        engine, scene, _vlm, _spoken = _make()
        await asyncio.wait_for(engine._do_scene(Intent(IntentKind.LOOK, "这是什么", "这是什么"), engine._qa_gen), timeout=10)
        await asyncio.wait_for(engine._do_clear(), timeout=10)
        return scene, engine._last_scene_frame

    scene, frame = asyncio.run(run())
    assert frame is None, "清除后旧帧没被清掉"
    assert scene.resets >= 1


# ---------------------------------------------------------------- C2 视觉追问


def test_visual_followup_goes_to_scene_not_chat() -> None:
    """「现在呢」是视觉追问：走场景链路，不该发给文本模型。"""

    async def run():
        engine, scene, vlm, spoken = _make()
        await asyncio.wait_for(engine._do_chat("现在呢", engine._qa_gen), timeout=10)
        return engine, scene, vlm, spoken

    engine, scene, vlm, spoken = asyncio.run(run())
    assert scene is not None and scene.calls, "视觉追问没有走场景链路"
    assert vlm.prompts == [], f"视觉追问被当纯文本发给了模型: {vlm.prompts}"
    assert "chat.visual_followup" in _names(engine), _names(engine)
    assert any(kind == "answer" for kind, _t in spoken)


def test_plain_question_still_goes_to_chat() -> None:
    async def run():
        engine, scene, vlm, _spoken = _make()
        await asyncio.wait_for(engine._do_chat("给我讲个笑话", engine._qa_gen), timeout=10)
        return engine, scene, vlm

    engine, scene, vlm = asyncio.run(run())
    assert len(vlm.prompts) == 1, vlm.prompts
    assert not scene.calls, "普通问题不应触发取图"


# ---------------------------------------------------------------- C3 单目标额度


def test_capture_budget_blocks_fourth_look() -> None:
    """单目标 3 次取图额度用尽：不再取图，直接说明。"""

    async def run():
        engine, scene, _vlm, spoken = _make(image_tasks=ImageTaskStore(max_captures_per_position=3))
        for _ in range(3):
            await asyncio.wait_for(
                engine._do_scene(Intent(IntentKind.LOOK, "重新拍一下", "重新拍一下"), engine._qa_gen),
                timeout=10,
            )
        calls_before = len(scene.calls)
        await asyncio.wait_for(
            engine._do_scene(Intent(IntentKind.LOOK, "重新拍一下", "重新拍一下"), engine._qa_gen),
            timeout=10,
        )
        return engine, scene, spoken, calls_before

    engine, scene, spoken, calls_before = asyncio.run(run())
    assert calls_before == 3, f"前三问应各取一帧: {scene.calls}"
    assert len(scene.calls) == 3, "额度用尽后仍然取图了"
    assert ("capture_budget", "这个目标这一轮已经看了三次，换个东西或者让我从头来。") in spoken, spoken
    assert "scene.budget_blocked" in _names(engine), _names(engine)


def test_capture_is_registered_once_per_fresh_frame() -> None:
    """真正取图才登记额度；复用缓存帧不计数。"""

    async def run():
        store = ImageTaskStore()
        engine, _scene, _vlm, _spoken = _make(image_tasks=store)
        await asyncio.wait_for(engine._do_scene(Intent(IntentKind.LOOK, "这是什么", "这是什么"), engine._qa_gen), timeout=10)
        await asyncio.wait_for(engine._do_scene(Intent(IntentKind.LOOK, "什么颜色", "什么颜色"), engine._qa_gen), timeout=10)
        return engine, store

    engine, store = asyncio.run(run())
    counts = [r for r in engine.timings.records if r.get("name") == "scene.capture_count"]
    assert len(counts) == 1, f"复用缓存帧不该再登记额度: {counts}"
    assert counts[0].get("captures") == 1, counts[0]


# ---------------------------------------------------------------- C4 纠正


def test_correct_with_position_releases_that_position() -> None:
    """纠正句带位置：按位置释放旧帧，留 correct.applied 证据行。"""

    async def run():
        engine, _scene, _vlm, _spoken = _make()
        task = engine.image_tasks.open_compare("比较这两件")
        engine._last_compare_task = task
        group = engine.image_tasks.new_group(task, position="B", source="camera")
        engine.image_tasks.record_capture(task, group, "frame-b")
        await asyncio.wait_for(
            engine._do_correct(Intent(IntentKind.CORRECT, "不是这个，我说的是右边那件", "不是这个，我说的是右边那件")),
            timeout=10,
        )
        return engine, task

    engine, task = asyncio.run(run())
    rec = _rec(engine, "correct.applied")
    assert rec.get("position") == "B", rec
    assert rec.get("released") == "B", rec
    assert task.frames.get("B") is None, f"纠正的位置没被释放: {task.frames}"


def test_compare_resume_reuses_captured_frame_without_new_grab() -> None:
    """双图任务被打断后「继续」：复用同一 CompareTask 已取的图，不重新取图。"""

    async def run():
        engine, scene, _vlm, _spoken = _make()
        task = engine.image_tasks.open_compare("比较这两件")
        group_a = engine.image_tasks.new_group(task, position="A", source="camera")
        engine.image_tasks.record_capture(task, group_a, "frame-a")
        frame_a = _Answer("frame-a", frame_id="frame-a")
        task.frames["A"] = frame_a
        engine.image_tasks.mark(task, "paused")
        assert engine.image_tasks.resume_target() is task, "被打断的图任务应是恢复目标"
        before = list(scene.provider.acquires)
        await asyncio.wait_for(engine._do_resume(engine._qa_gen), timeout=10)
        return engine, scene, task, frame_a, before

    engine, scene, task, frame_a, before = asyncio.run(run())
    grabbed = scene.provider.acquires[len(before) :]
    assert len(grabbed) == 1, f"恢复时只该补取缺的那一张，实际取图 {grabbed}"
    assert scene.compares, "没有走到比较"
    assert scene.compares[0][0] is frame_a, "已取过的 A 图被重新取了"
    assert "compare.resume" in _names(engine), _names(engine)
    resumes = [r for r in engine.timings.records if r.get("name") == "compare.resume"]
    assert resumes[0].get("task") == task.task_id, resumes[0]


def test_compare_resume_without_unfinished_task_says_nothing_to_resume() -> None:
    async def run():
        engine, _scene, _vlm, spoken = _make()
        await asyncio.wait_for(engine._do_resume(engine._qa_gen), timeout=10)
        return spoken

    spoken = asyncio.run(run())
    assert ("resume_none", "现在没有要接着看的图。") in spoken, spoken


def test_cancel_during_compare_marks_task_paused() -> None:
    """比较中途被取消：任务标 paused，成为可恢复目标（否则「继续」没东西可接着看）。"""

    class _HangingScene(_Scene):
        def compare(self, frame_a, frame_b, question: str):
            import time

            time.sleep(1.0)
            return super().compare(frame_a, frame_b, question)

    async def run():
        engine, _scene, _vlm, _spoken = _make()
        engine.scene.compare = _HangingScene().compare
        task = engine.image_tasks.open_compare("比较这两件")
        engine._last_compare_task = task
        future = asyncio.create_task(engine._run_compare(task, engine._qa_gen))
        await asyncio.sleep(0.2)
        future.cancel()
        try:
            await future
        except asyncio.CancelledError:
            pass
        return engine, task

    engine, task = asyncio.run(run())
    assert task.state == "paused", f"被打断的比较任务状态是 {task.state}"
    assert engine.image_tasks.resume_target() is task


def test_correct_without_position_only_acknowledges() -> None:
    async def run():
        engine, _scene, _vlm, _spoken = _make()
        await asyncio.wait_for(engine._do_correct(Intent(IntentKind.CORRECT, "不是这个", "不是这个")), timeout=10)
        return engine

    engine = asyncio.run(run())
    rec = _rec(engine, "correct.ack_only")
    assert rec.get("position") == "", rec


def test_correct_is_reachable_from_classify() -> None:
    """上游分类已把「不是这个，我说的是右边那件」判成 correct（speech-b 已修）。"""

    from aieyes.intent import classify

    assert classify("不是这个，我说的是右边那件").kind is IntentKind.CORRECT


# ---------------------------------------------------------------- CONTRACT


def test_contract_intent_answers_honestly_not_unsupported() -> None:
    """「紧急」走 contract 分支：停播 + 说清设备侧没接，不能说成"能力没接上"。"""

    async def run():
        engine, _scene, _vlm, spoken = _make()
        intent = Intent(IntentKind.CONTRACT, "紧急", "紧急")
        _ = intent
        await asyncio.wait_for(engine._utterance_flow("紧急"), timeout=10)
        return engine, spoken

    engine, spoken = asyncio.run(run())
    kinds = [kind for kind, _text in spoken]
    assert "unsupported" not in kinds, f"合同能力被答成未接入: {spoken}"
    assert "contract_unavailable" in kinds, spoken
    line = [text for kind, text in spoken if kind == "contract_unavailable"][0]
    assert "设备" in line, line
    assert "contract.not_linked" in _names(engine), _names(engine)


def test_emergency_stop_still_takes_stop_path() -> None:
    from aieyes.intent import classify

    assert classify("紧急停止").kind is IntentKind.STOP


# ---------------------------------------------------------------- C5 重拍 / 换物


def _prepared_compare(engine, *, state: str = "paused"):
    """准备一个两个位置都已取图的双图任务（帧对象可从 provider 返回里拿到）。"""

    task = engine.image_tasks.open_compare("比较这两件")
    engine._last_compare_task = task
    frames = {}
    for position in ("A", "B"):
        group = engine.image_tasks.new_group(task, position=position, source="camera")
        frame = _Answer(f"frame-{position}", frame_id=f"frame-{position}")
        engine.image_tasks.record_capture(task, group, f"frame-{position}")
        task.frames[position] = frame
        engine._remember_frame(frame)
        frames[position] = frame
    if state == "paused":
        engine.image_tasks.mark(task, "paused")
    return task, frames


def test_reshoot_position_regrabs_only_that_position() -> None:
    """「重拍第一件」：只重取 A，B 保留原帧，位置次数累加（不回到 0）。"""

    async def run():
        engine, scene, _vlm, _spoken = _make()
        task, frames = _prepared_compare(engine)
        before = list(scene.provider.acquires)
        await asyncio.wait_for(engine._do_chat("重拍第一件", engine._qa_gen), timeout=10)
        return engine, scene, task, frames, before

    engine, scene, task, frames, before = asyncio.run(run())
    grabbed = scene.provider.acquires[len(before) :]
    assert len(grabbed) == 1, f"只该重取被点名的那一件，实际 {grabbed}"
    assert task.frames.get("B") is frames["B"], "另一位置被一起重取了"
    assert task.frames.get("A") is not frames["A"], "被点名位置仍是旧照片（旧图冒充当前证据）"
    assert task.position_captures.get("A") == 2, f"位置次数被重置: {task.position_captures}"
    assert task.position_captures.get("B") == 1, task.position_captures
    assert "compare.reshoot" in _names(engine), _names(engine)
    assert engine._frame_cache.get("frame-A") is frames["A"], "放弃的旧帧应留作历史引用"


def test_reshoot_budget_exhausted_blocks_without_releasing() -> None:
    """额度用尽时不先释放、不白取图，直接说明。"""

    async def run():
        engine, scene, _vlm, spoken = _make()
        task, frames = _prepared_compare(engine)
        task.position_captures["A"] = 3
        before = list(scene.provider.acquires)
        await asyncio.wait_for(engine._do_chat("重拍第一件", engine._qa_gen), timeout=10)
        return engine, scene, task, frames, before, spoken

    engine, scene, task, frames, before, spoken = asyncio.run(run())
    assert ("capture_budget", "这个位置的取图次数已经用完了，换个东西或者让我从头来。") in spoken, spoken
    assert scene.provider.acquires[len(before) :] == [], "额度用尽仍发了取图请求"
    assert task.frames.get("A") is frames["A"], "没重拍却把旧照片释放了"
    rec = _rec(engine, "compare.budget_blocked")
    assert rec.get("position") == "A", rec


def test_reshoot_without_position_asks_which_one() -> None:
    """两件都在、又没说哪一件：先问清位置，不动任何证据。"""

    async def run():
        engine, scene, _vlm, spoken = _make()
        task, frames = _prepared_compare(engine)
        before = list(scene.provider.acquires)
        await asyncio.wait_for(engine._do_chat("重拍一下", engine._qa_gen), timeout=10)
        return engine, scene, task, frames, before, spoken

    engine, scene, task, frames, before, spoken = asyncio.run(run())
    assert ("reshoot_clarify", "你想重拍第一件还是第二件？") in spoken, spoken
    assert scene.provider.acquires[len(before) :] == []
    assert task.frames.get("A") is frames["A"] and task.frames.get("B") is frames["B"]
    assert "compare.reshoot_clarify" in _names(engine), _names(engine)


def test_reshoot_after_finished_compare_reuses_other_position() -> None:
    """上一轮比较已结束，点名重拍第二件：A 沿用旧帧，只重取 B，旧任务不被改写。"""

    async def run():
        engine, scene, _vlm, _spoken = _make()
        task, frames = _prepared_compare(engine)
        engine.image_tasks.finish(task, answer_text="比较完成")
        old_id = task.task_id
        before = list(scene.provider.acquires)
        await asyncio.wait_for(engine._do_chat("重拍第二件", engine._qa_gen), timeout=10)
        return engine, task, frames, old_id, scene, before

    engine, task, frames, old_id, scene, before = asyncio.run(run())
    assert "compare.reshoot_reuse_pair" in _names(engine), _names(engine)
    assert task.task_id == old_id, "旧任务被改写"
    new_task = engine._last_compare_task
    assert new_task is not None and new_task.task_id != old_id, "没有开新一轮任务"
    assert new_task.frames.get("A") is frames["A"], "另一件应沿用旧帧"
    assert new_task.frames.get("B") is not frames["B"], "被点名的一件仍是旧照片"
    assert len(scene.provider.acquires[len(before) :]) == 1, scene.provider.acquires


def test_reshoot_words_without_compare_task_go_to_scene_capture() -> None:
    """没有双图任务时「重拍」不抢场景链路：仍走单图取新帧，不答"没有要重拍的图"。"""

    async def run():
        engine, scene, _vlm, spoken = _make()
        await asyncio.wait_for(
            engine._do_chat("重新拍一下", engine._qa_gen), timeout=10
        )
        return engine, scene, spoken

    engine, scene, spoken = asyncio.run(run())
    kinds = [kind for kind, _text in spoken]
    assert "reshoot_none" not in kinds, spoken
    assert scene.calls, "没有取图"
    assert scene.calls[-1]["frame"] is None, f"要求重拍却复用了帧: {scene.calls}"


# ---------------------------------------------------------------- C6 历史图引用


def test_scene_answer_reuses_history_source_without_budget() -> None:
    """上一轮任务留下的照片：按 history 引用回答，不扣单目标额度。"""

    async def run():
        engine, scene, _vlm, _spoken = _make()
        _task, _frames = _prepared_compare(engine)
        assert engine._last_scene_frame is None, "比较任务不该设置单图缓存帧"
        await asyncio.wait_for(
            engine._do_scene(Intent(IntentKind.LOOK, "这是什么", "这是什么"), engine._qa_gen), timeout=10
        )
        return engine, scene

    engine, scene = asyncio.run(run())
    assert scene.calls and scene.calls[-1]["frame"] is not None, f"历史图未被引用: {scene.calls}"
    rec = [r for r in engine.timings.records if r.get("name") == "scene.answer"][-1]
    assert rec.get("source") == "history", rec
    assert rec.get("reused_from") == "history", rec
    assert "scene.history_ref" in _names(engine), _names(engine)
    assert engine.image_tasks.single_target_captures == 0, "引用历史图却扣了额度"


def test_scene_answer_reports_ab_position_for_followup() -> None:
    """同图追问要能说清用的是 A 还是 B（归属写在证据行里）。"""

    async def run():
        engine, _scene, _vlm, _spoken = _make()
        _task, frames = _prepared_compare(engine)
        engine._last_scene_frame = frames["B"]
        await asyncio.wait_for(
            engine._do_scene(Intent(IntentKind.LOOK, "什么颜色", "什么颜色"), engine._qa_gen), timeout=10
        )
        return engine

    engine = asyncio.run(run())
    rec = [r for r in engine.timings.records if r.get("name") == "scene.answer"][-1]
    assert rec.get("position") == "B", rec
    assert rec.get("reused") is True, rec


def test_fresh_capture_request_skips_history_frame() -> None:
    """明确要求重拍时不得用历史图顶替（否则额度与"当前画面"都会失真）。"""

    async def run():
        engine, scene, _vlm, _spoken = _make()
        _prepared_compare(engine)
        await asyncio.wait_for(
            engine._do_scene(Intent(IntentKind.LOOK, "重新拍一下", "重新拍一下"), engine._qa_gen), timeout=10
        )
        return engine, scene

    engine, scene = asyncio.run(run())
    assert scene.calls[-1]["frame"] is None, f"要求重拍却拿历史图顶替: {scene.calls}"
    rec = [r for r in engine.timings.records if r.get("name") == "scene.answer"][-1]
    assert rec.get("source") in ("camera", "file"), rec

# ---------------------------------------------------------------- 合同引导（control.intent）


class _ControlResult:
    """``control.result`` 替身：只暴露 send_guidance_intent 读取的字段。"""

    def __init__(self, *, accepted: bool = True, state: str = "completed", reason_code: str | None = None) -> None:
        self.type = "control.result"
        self.status = "ok"
        self.error = None
        self.data = {
            "accepted": accepted,
            "operation_state": state,
            "reason_code": reason_code,
        }


class _GuidanceContract:
    """记录每次 control.intent 的参数；不实现其它合同方法（引擎只调这一个）。"""

    def __init__(self, result: _ControlResult | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result or _ControlResult()

    async def send_control_intent(self, **kwargs: Any) -> _ControlResult:
        self.calls.append(dict(kwargs))
        return self.result


def _guidance_rows(engine: InteractionEngine) -> list[dict[str, Any]]:
    return [rec for rec in engine.timings.records if rec.get("name") == "control.intent"]


def test_guidance_words_reach_contract_instead_of_chat() -> None:
    """重复/开始/结束引导：以前落 chat（模型自由回答），现在必须发 control.intent。"""

    async def run():
        engine, _scene, vlm, spoken = _make()
        contract = _GuidanceContract()
        engine.attach_contract(contract)
        cases = (("重复引导", "repeat_guidance"), ("开始引导", "start_guidance"), ("结束引导", "end_guidance"))
        seen_ids = []
        for line, action in cases:
            calls_before = len(contract.calls)
            await engine._utterance_flow(line)
            assert len(contract.calls) == calls_before + 1, (line, contract.calls)
            call = contract.calls[-1]
            assert call["action"] == action, call
            assert call["final_transcript"] == line, call
            assert call["utterance_id"], call
            seen_ids.append(call["utterance_id"])
            rows = _guidance_rows(engine)
            assert rows[-1]["action"] == action, rows[-1]
            assert rows[-1]["accepted"] is True, rows[-1]
            assert rows[-1]["operation_state"] == "completed", rows[-1]
        # 三条引导都没有被当成聊天问题丢给模型
        assert vlm.prompts == [], vlm.prompts
        assert "control" in [kind for kind, _text in spoken], spoken
        # 同回合稳定、跨回合唯一
        assert len(set(seen_ids)) == 3, seen_ids

    asyncio.run(run())


def test_local_control_words_do_not_steal_guidance() -> None:
    """暂停/继续/再说一遍引导：本地 STOP/RESUME/REPEAT 不得抢走。"""

    async def run():
        engine, _scene, vlm, _spoken = _make()
        contract = _GuidanceContract()
        engine.attach_contract(contract)
        cases = (
            ("暂停引导", "pause_guidance"),
            ("继续引导", "resume_guidance"),
            ("再说一遍引导", "repeat_guidance"),
        )
        for line, action in cases:
            await engine._utterance_flow(line)
            assert contract.calls[-1]["action"] == action, (line, contract.calls)
            assert contract.calls[-1]["final_transcript"] == line, contract.calls[-1]
        names = _names(engine)
        assert "control.intent" in names, names
        # 没有落到本地停播/续看/重播的证据行
        for stolen in ("stop.silenced", "stop.cancelled", "repeat.reread", "repeat.resume_unplayed", "compare.resume"):
            assert stolen not in names, (stolen, names)
        assert vlm.prompts == [], vlm.prompts

    asyncio.run(run())


def test_guidance_without_contract_says_not_linked() -> None:
    """未接合同：如实说没连上，不发 intent，也不假装执行。"""

    async def run():
        engine, _scene, _vlm, spoken = _make()
        await engine._utterance_flow("开始引导")
        assert "contract.not_linked" in _names(engine), _names(engine)
        assert _guidance_rows(engine) == [], _guidance_rows(engine)
        kinds = [kind for kind, _text in spoken]
        assert "control_unavailable" in kinds, spoken
        line = [text for kind, text in spoken if kind == "control_unavailable"][0]
        assert "没有接到主控" in line, line

    asyncio.run(run())


def test_negated_guidance_does_not_send_intent() -> None:
    """「别暂停引导」是纠正/否定：classify_guidance_action 返回 None，不许发 intent。"""

    async def run():
        engine, _scene, _vlm, _spoken = _make()
        contract = _GuidanceContract()
        engine.attach_contract(contract)
        await engine._utterance_flow("别暂停引导")
        assert contract.calls == [], contract.calls
        assert _guidance_rows(engine) == [], _guidance_rows(engine)

    asyncio.run(run())


def test_emergency_stop_stays_local_stop() -> None:
    """「紧急停止」仍只做本地静默，不被引导 hook 抢走，也不发空间/移动动作。"""

    async def run():
        engine, _scene, _vlm, _spoken = _make()
        contract = _GuidanceContract()
        engine.attach_contract(contract)
        await engine._utterance_flow("紧急停止")
        assert contract.calls == [], contract.calls
        assert _guidance_rows(engine) == [], _guidance_rows(engine)

    asyncio.run(run())

# ---------------------------------------------------------------- 派发层（分类 → 分支）

# 每个 intent kind 在 _utterance_flow 里应当落到哪个目标函数。
_DISPATCH_TARGETS = {
    "clear": "_do_clear",
    "stop": "_do_stop",
    "mute": "_do_mute",
    "backchannel": "_do_backchannel",
    "correct": "_do_correct",
    "repeat": "_do_repeat",
    "resume": "_do_resume",
    "compare": "_do_compare",
    "look": "_do_scene",
    "price": "_do_scene",
    "date": "_do_scene",
    "find": "_do_scene",
    "chat": "_do_chat",
}

# 被包起来观察的全部分支：命中哪个就记哪个，其它分支一律不得被调用。
_DISPATCH_WATCHED = (
    "_do_clear",
    "_do_stop",
    "_do_mute",
    "_do_backchannel",
    "_do_correct",
    "_do_repeat",
    "_do_resume",
    "_do_compare",
    "_do_scene",
    "_do_chat",
    "_do_guidance",
    "_do_backchannel",
)


def _install_dispatch_recorders(engine: InteractionEngine) -> list[str]:
    """把所有 _do_* 分支换成记录器：只记录被调用，不执行原逻辑。"""
    called: list[str] = []

    def make(name: str):
        async def recorder(*args: Any, **kwargs: Any) -> None:
            called.append(name)
            return None

        return recorder

    for name in _DISPATCH_WATCHED:
        setattr(engine, name, make(name))
    return called


DISPATCH_CASES = (
    ("不是这个，我说的是右边那件", "correct"),
    ("再说一遍", "repeat"),
    ("清空对话", "clear"),
    ("停一下", "stop"),
    ("继续", "resume"),
    ("这是什么", "look"),
    ("紧急", "contract"),
    ("给我讲个笑话", "chat"),
)


def test_intent_kinds_dispatch_to_expected_branches() -> None:
    """分类结果必须真的派发到对应 _do_*：断的是派发，不是 _do_* 自身实现。"""

    from aieyes.intent import classify

    async def run():
        engine, _scene, _vlm, _spoken = _make()
        engine.attach_contract(_GuidanceContract())
        called = _install_dispatch_recorders(engine)
        for line, kind in DISPATCH_CASES:
            assert classify(line).kind.value == kind, (line, classify(line).kind.value)
            called.clear()
            await engine._utterance_flow(line)
            expected = _DISPATCH_TARGETS.get(kind)
            if expected is None:
                # CONTRACT 没有 _do_* 目标：只允许不落到任何其它分支
                assert called == [], (line, kind, called)
                continue
            assert called == [expected], (line, kind, called)
        # 引导必须在 STOP/RESUME/REPEAT 之前命中自己的分支
        called.clear()
        await engine._utterance_flow("暂停引导")
        assert called == ["_do_guidance"], called
        called.clear()
        await engine._utterance_flow("再说一遍引导")
        assert called == ["_do_guidance"], called

    asyncio.run(run())


def test_guidance_branch_receives_resolved_action() -> None:
    """派发到 _do_guidance 时，参数是 classify_guidance_action 解析出的动作名。"""

    from aieyes.contract.guidance_intent import classify_guidance_action

    async def run():
        engine, _scene, _vlm, _spoken = _make()
        contract = _GuidanceContract()
        engine.attach_contract(contract)
        seen: list[tuple[str, str]] = []

        async def recorder(action: str, intent: Any) -> None:
            seen.append((action, intent.raw))

        engine._do_guidance = recorder  # type: ignore[method-assign]
        for line in ("开始引导", "暂停引导", "继续引导", "结束引导", "再说一遍引导"):
            await engine._utterance_flow(line)
            assert seen[-1] == (classify_guidance_action(line), line), (line, seen[-1])
        assert [action for action, _raw in seen] == [
            "start_guidance",
            "pause_guidance",
            "resume_guidance",
            "end_guidance",
            "repeat_guidance",
        ], seen

    asyncio.run(run())
