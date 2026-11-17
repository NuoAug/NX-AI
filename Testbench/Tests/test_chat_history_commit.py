"""B3：非流式问答历史提交契约。

口径：可控阻塞假模型 + 真实 ``InteractionEngine._do_chat``，只看
``ChatSession.turns`` 的条数与内容、以及假模型实际收到的 prompt 全文，
不使用模型自评。

目标缺陷（修复前）：``ChatSession.ask`` 在 ``vlm.complete`` 返回后无条件
``record``，而引擎的非流式路径超时放弃后旧线程仍会跑完 → 被放弃的尝试
写进历史，并被后续问题当作上下文。
"""

from __future__ import annotations

import asyncio
import threading

from aieyes.chat import ChatSession
from aieyes.config import AppConfig
from aieyes.events import EventBus
from aieyes.interaction.engine import InteractionEngine

FALLBACK = "我这边没想好，你再问一次？"


class BlockingVlm:
    """第一次 ``complete`` 阻塞到 ``release.set()``，之后立即返回。"""

    def __init__(self, *, block_first: bool = True) -> None:
        self.block_first = block_first
        self.release = threading.Event()
        self.finished = threading.Event()
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def complete(self, prompt, *, images, max_tokens, timeout_s, temperature=None):
        with self._lock:
            self.calls.append(prompt)
            index = len(self.calls)
        if self.block_first and index == 1:
            try:
                self.release.wait(10)
            finally:
                self.finished.set()
            return "旧答案"
        return "新答案"


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


def _make_engine(*, t11_ms: int = 120, retry_max: int = 0, block_first: bool = True):
    cfg = AppConfig()
    cfg.timers.t11_generate_ms = t11_ms
    cfg.timers.t11_retry_max = retry_max
    cfg.timers.t11_retry_delay_ms = 1
    vlm = BlockingVlm(block_first=block_first)
    chat = ChatSession(vlm=vlm)
    engine = InteractionEngine(
        cfg=cfg,
        bus=EventBus(),
        frontend=_Frontend(),
        sink=_Sink(),
        tts=_Tts(),
        chat=chat,
    )
    spoken: list[str] = []

    async def fake_speak(text: str, *, kind: str, ref_ms=None, meta=None, **kwargs) -> bool:
        spoken.append(text)
        return True

    engine._speak = fake_speak  # type: ignore[method-assign]
    return engine, chat, vlm, spoken


def _pairs(chat: ChatSession) -> list[tuple[str, str]]:
    return [(turn.user, turn.assistant) for turn in chat.turns]


def test_successful_non_stream_answer_is_recorded_exactly_once() -> None:
    """成功路径：正常返回时历史只增 1 条（不能修成 0 条或 2 条）。"""

    async def run():
        engine, chat, vlm, spoken = _make_engine(block_first=False)
        await engine._do_chat("普通问题", engine._qa_gen)
        return chat, vlm, spoken

    chat, vlm, spoken = asyncio.run(run())
    assert len(vlm.calls) == 1
    assert _pairs(chat) == [("普通问题", "新答案")]
    assert spoken == ["新答案"]


def test_abandoned_timeout_attempt_is_not_written_to_history() -> None:
    """超时放弃 + 用户插话换问题：旧尝试 0 条记录，下一问历史干净且只记 1 条。"""

    async def run():
        engine, chat, vlm, spoken = _make_engine(t11_ms=120)
        # 第 1 问：模型卡住 → t11 超时放弃（线程仍在跑）
        await asyncio.wait_for(engine._do_chat("第一个问题", engine._qa_gen), timeout=5)
        assert vlm.calls, "假模型应已被调用"
        assert spoken == [FALLBACK]
        assert _pairs(chat) == []

        # 用户插话/换问题：等效 _silence_now 提升问答代次
        engine._qa_gen += 1
        # 放行被放弃的旧线程，等它真正退出 chat.ask（锁释放即证明）
        vlm.release.set()
        acquired = await asyncio.to_thread(engine._llm_thread_lock.acquire, True, 10)
        assert acquired, "被放弃的旧线程没有释放 _llm_thread_lock"
        engine._llm_thread_lock.release()
        # 旧尝试不得留下任何历史
        assert _pairs(chat) == [], f"被放弃的超时尝试写进了历史: {_pairs(chat)!r}"

        # 第 2 问：历史必须是干净的，且只记 1 条
        await asyncio.wait_for(engine._do_chat("第二个问题", engine._qa_gen), timeout=5)
        return chat, vlm, spoken

    chat, vlm, spoken = asyncio.run(run())
    assert _pairs(chat) == [("第二个问题", "新答案")]
    second_prompt = vlm.calls[-1]
    assert "第一个问题" not in second_prompt, "后续问题的 prompt 里混入了被放弃的旧问题"
    assert "旧答案" not in second_prompt, "后续问题的 prompt 里混入了被放弃的旧答案"
    assert spoken == [FALLBACK, "新答案"]


def test_stale_answer_is_discarded_but_not_recorded() -> None:
    """生成期间代次已变：即使模型成功返回，也只记 stale_discarded，不写历史、不播报。"""

    async def run():
        engine, chat, vlm, spoken = _make_engine(t11_ms=5000)
        task = asyncio.create_task(engine._do_chat("旧问题", engine._qa_gen))
        await asyncio.sleep(0.05)  # 让线程进入 complete()
        engine._qa_gen += 1  # 用户插话
        vlm.release.set()
        await asyncio.wait_for(task, timeout=5)
        return engine, chat, vlm, spoken

    engine, chat, vlm, spoken = asyncio.run(run())
    assert _pairs(chat) == []
    assert spoken == []
    names = [record["name"] for record in engine.timings.records]
    assert "llm.stale_discarded" in names, names


def test_ask_default_records_once_and_record_flag_disables_commit() -> None:
    """``ask()`` 默认仍落历史（Testbench/debug.py 口径），``record=False`` 不落。"""

    plain = BlockingVlm(block_first=False)
    chat = ChatSession(vlm=plain)
    chat.ask("问题一")
    assert _pairs(chat) == [("问题一", "新答案")]
    chat.ask("问题二", record=False)
    assert _pairs(chat) == [("问题一", "新答案")]


class StreamVlm:
    """流式假模型：``complete_stream`` 逐段产出。"""

    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces
        self.prompts: list[str] = []

    def complete(self, prompt, *, images, max_tokens, timeout_s, temperature=None):  # pragma: no cover
        raise AssertionError("流式路径不应调用 complete")

    def complete_stream(self, prompt, *, images, max_tokens, timeout_s, temperature=None):
        self.prompts.append(prompt)
        for piece in self.pieces:
            yield piece


def test_streaming_path_still_records_exactly_once() -> None:
    """流式路径不回归：定稿后由调用方 record 一次。"""

    async def run():
        cfg = AppConfig()
        cfg.timers.t11_generate_ms = 5000
        vlm = StreamVlm(["第一句。", "第二句。"])
        chat = ChatSession(vlm=vlm)
        engine = InteractionEngine(
            cfg=cfg,
            bus=EventBus(),
            frontend=_Frontend(),
            sink=_Sink(),
            tts=_Tts(),
            chat=chat,
        )
        spoken: list[str] = []

        async def fake_speak(text: str, *, kind: str, ref_ms=None, meta=None, **kwargs) -> bool:
            spoken.append(text)
            return True

        engine._speak = fake_speak  # type: ignore[method-assign]
        await asyncio.wait_for(engine._do_chat("流式问题", engine._qa_gen), timeout=10)
        return chat, vlm, spoken

    chat, vlm, spoken = asyncio.run(run())
    assert _pairs(chat) == [("流式问题", "第一句。第二句。")]
    # 播报分段粒度由 _split_first_phrase 决定（非本测试契约），只断言播出的总文本
    assert "".join(spoken) == "第一句。第二句。"
    assert len(vlm.prompts) == 1


GROUNDED = "我还没看到画面，让我看一眼再告诉你，好吗？"


def _make_adapter_engine(*, block_first: bool = False):
    """运行时装配口径：引擎 chat = Testbench 的 ChatAdapter（chat_adapter.make_chat）。"""
    from chat_adapter import make_chat

    cfg = AppConfig()
    cfg.timers.t11_generate_ms = 120
    cfg.timers.t11_retry_max = 0
    cfg.timers.t11_retry_delay_ms = 1
    vlm = BlockingVlm(block_first=block_first)
    chat = make_chat(vlm)
    engine = InteractionEngine(
        cfg=cfg,
        bus=EventBus(),
        frontend=_Frontend(),
        sink=_Sink(),
        tts=_Tts(),
        chat=chat,
    )
    spoken: list[str] = []

    async def fake_speak(text: str, *, kind: str, ref_ms=None, meta=None, **kwargs) -> bool:
        spoken.append(text)
        return True

    engine._speak = fake_speak  # type: ignore[method-assign]
    return engine, chat, vlm, spoken


def test_engine_commits_grounded_answer_once_with_chat_adapter() -> None:
    """ChatAdapter 的固定答句也不能与引擎提交重复落盘（否则同一问两句历史）。"""

    async def run():
        engine, chat, vlm, spoken = _make_adapter_engine()
        await asyncio.wait_for(engine._do_chat("那边有人吗？", engine._qa_gen), timeout=10)
        await asyncio.wait_for(engine._do_chat("给我讲个笑话", engine._qa_gen), timeout=10)
        return chat, vlm, spoken

    chat, vlm, spoken = asyncio.run(run())
    assert _pairs(chat) == [("那边有人吗？", GROUNDED), ("给我讲个笑话", "新答案")]
    assert spoken == [GROUNDED, "新答案"]
    # 固定答句命中的那一轮没有走模型
    assert len(vlm.calls) == 1


def test_chat_adapter_direct_ask_still_records_once() -> None:
    """绕过引擎直调 ChatAdapter.ask（默认口径）仍恰好记 1 条。"""
    from chat_adapter import make_chat

    chat = make_chat(BlockingVlm(block_first=False))
    chat.ask("那边有人吗？")
    assert _pairs(chat) == [("那边有人吗？", GROUNDED)]


def _speak_stub(engine, behaviour):
    """把 _speak 换成受控实现：behaviour(text, kind) -> bool，并收集实际播出文本。"""
    spoken: list[str] = []

    async def fake_speak(text: str, *, kind: str, ref_ms=None, meta=None, **kwargs) -> bool:
        played = behaviour(text, kind)
        if played:
            spoken.append(text)
        return played

    engine._speak = fake_speak  # type: ignore[method-assign]
    return spoken


def test_unplayed_answer_is_not_written_to_history() -> None:
    """一个字都没播出去（_speak 返回 False）：答案不能进历史。"""

    async def run():
        engine, chat, vlm, _ = _make_engine(block_first=False)
        spoken = _speak_stub(engine, lambda text, kind: False)
        await asyncio.wait_for(engine._do_chat("未播出的提问", engine._qa_gen), timeout=10)
        return chat, vlm, spoken

    chat, vlm, spoken = asyncio.run(run())
    assert spoken == []  # 确认确实没有播报
    assert len(vlm.calls) == 1  # 模型确实生成过，所以这不是"没生成"
    assert _pairs(chat) == [], f"未播出的答案进了历史: {_pairs(chat)}"


def test_unplayed_answer_stays_out_of_next_prompt() -> None:
    """可被他人复算的证据：未播出答案的下一次模型 prompt 里不能出现它。"""

    async def run():
        engine, chat, vlm, _ = _make_engine(block_first=False)
        state = {"allow": False}

        def behaviour(text: str, kind: str) -> bool:
            return state["allow"] or kind != "answer"

        _speak_stub(engine, behaviour)
        await asyncio.wait_for(engine._do_chat("未播出的提问", engine._qa_gen), timeout=10)
        state["allow"] = True  # 第二问正常播报
        await asyncio.wait_for(engine._do_chat("第二个问题", engine._qa_gen), timeout=10)
        return chat, vlm

    chat, vlm = asyncio.run(run())
    assert len(vlm.calls) == 2
    second_prompt = vlm.calls[1]
    assert "未播出的提问" not in second_prompt, f"未播出的提问进了下一问 prompt:\n{second_prompt}"
    assert "新答案" not in second_prompt, f"未播出的答案进了下一问 prompt:\n{second_prompt}"
    assert _pairs(chat) == [("第二个问题", "新答案")]


def test_partially_played_answer_is_not_written_to_history() -> None:
    """只播了一部分就失败：按"未完整播报即不提交"处理，并留可观测记录。"""

    async def run():
        engine, chat, vlm, _ = _make_engine(block_first=False)
        spoken: list[str] = []

        async def half_speak(text: str, *, kind: str, ref_ms=None, meta=None, **kwargs) -> bool:
            if kind == "answer":
                spoken.append(text[:1])  # 只播出第一个字
                return False
            return True

        engine._speak = half_speak  # type: ignore[method-assign]
        await asyncio.wait_for(engine._do_chat("半截问题", engine._qa_gen), timeout=10)
        return chat, vlm, spoken, engine

    chat, vlm, spoken, engine = asyncio.run(run())
    assert spoken == ["新"]  # 确实只播了前半段
    assert _pairs(chat) == [], f"只播了一半的答案进了历史: {_pairs(chat)}"
    names = [rec.get("name") for rec in engine.timings.records]
    assert "answer.not_committed" in names, f"部分播报没有留下可观测证据: {names}"


def test_interrupt_during_playback_is_not_written_to_history() -> None:
    """播报中被插话（代次前移）：即使已播出部分内容，也不提交历史。"""

    async def run():
        engine, chat, vlm, _ = _make_engine(block_first=False)
        spoken: list[str] = []

        async def interrupt_speak(text: str, *, kind: str, ref_ms=None, meta=None, **kwargs) -> bool:
            if kind == "answer":
                spoken.append(text)
                engine._qa_gen += 1  # 用户在播报中插话
                return False
            return True

        engine._speak = interrupt_speak  # type: ignore[method-assign]
        await asyncio.wait_for(engine._do_chat("播报中被打断", engine._qa_gen), timeout=10)
        return chat, spoken

    chat, spoken = asyncio.run(run())
    assert spoken == ["新答案"]
    assert _pairs(chat) == [], f"播报中被打断的答案进了历史: {_pairs(chat)}"


def test_cancelled_chat_does_not_write_history() -> None:
    """整轮被取消（cancel）：历史 0 条。"""

    async def run():
        engine, chat, vlm, _ = _make_engine(block_first=False)
        entered = asyncio.Event()

        async def blocking_speak(text: str, *, kind: str, ref_ms=None, meta=None, **kwargs) -> bool:
            entered.set()
            await asyncio.sleep(10)
            return True

        engine._speak = blocking_speak  # type: ignore[method-assign]
        task = asyncio.create_task(engine._do_chat("会被取消的问题", engine._qa_gen))
        await asyncio.wait_for(entered.wait(), timeout=10)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return chat, vlm

    chat, vlm = asyncio.run(run())
    assert len(vlm.calls) == 1
    assert _pairs(chat) == [], f"被取消的一轮进了历史: {_pairs(chat)}"


def test_streaming_interrupted_playback_is_not_written() -> None:
    """流式路径同一规则：播报没播完（_speak 返回 False），就不能把回答写进历史。"""

    async def run():
        cfg = AppConfig()
        cfg.timers.t11_generate_ms = 5000
        vlm = StreamVlm(["第一句。", "第二句。"])
        chat = ChatSession(vlm=vlm)
        engine = InteractionEngine(
            cfg=cfg,
            bus=EventBus(),
            frontend=_Frontend(),
            sink=_Sink(),
            tts=_Tts(),
            chat=chat,
        )
        spoken: list[str] = []

        async def fail_play(text: str, *, kind: str, ref_ms=None, meta=None, **kwargs) -> bool:
            if kind != "answer":
                return True
            spoken.append(text)  # 尝试播报，但整段没播完（TTS 失败/被打断）
            return False

        engine._speak = fail_play  # type: ignore[method-assign]
        await asyncio.wait_for(engine._do_chat("流式半截问题", engine._qa_gen), timeout=10)
        return chat, vlm, spoken, engine

    chat, vlm, spoken, engine = asyncio.run(run())
    assert len(vlm.prompts) == 1
    assert spoken, "确实尝试过播报"
    assert _pairs(chat) == [], f"流式未播完的答案进了历史: {_pairs(chat)}"
    names = [rec.get("name") for rec in engine.timings.records]
    assert "answer.not_committed" in names, f"流式未播完没有留证据: {names}"


def test_streaming_unplayed_answer_stays_out_of_next_prompt() -> None:
    """流式路径：第一问没播完 → 第二问真实 prompt 里不能出现第一问的问答全文。

    只断言历史空是不够的：真正决定串答的是下一问发给模型的 prompt 全文。
    """

    async def run():
        cfg = AppConfig()
        cfg.timers.t11_generate_ms = 5000
        vlm = StreamVlm(["第一句。", "第二句。"])
        chat = ChatSession(vlm=vlm)
        engine = InteractionEngine(
            cfg=cfg,
            bus=EventBus(),
            frontend=_Frontend(),
            sink=_Sink(),
            tts=_Tts(),
            chat=chat,
        )
        fail = {"on": True}

        async def speak(text: str, *, kind: str, ref_ms=None, meta=None, **kwargs) -> bool:
            if kind != "answer":
                return True
            return not fail["on"]  # 第一问播报失败，第二问正常播完

        engine._speak = speak  # type: ignore[method-assign]
        await asyncio.wait_for(engine._do_chat("流式未播出的提问", engine._qa_gen), timeout=10)
        fail["on"] = False
        await asyncio.wait_for(engine._do_chat("第二问", engine._qa_gen), timeout=10)
        return chat, vlm

    chat, vlm = asyncio.run(run())
    assert len(vlm.prompts) == 2, vlm.prompts
    second = vlm.prompts[1]
    assert "流式未播出的提问" not in second, f"被放弃的提问进了下一问 prompt:\n{second}"
    assert "第一句。第二句。" not in second, f"没播出的答案进了下一问 prompt:\n{second}"
    assert "第二问" in second, "下一问本身必须在 prompt 里"


def test_debug_direct_stream_does_not_pollute_next_prompt() -> None:
    """debug 直调口径（``chat.ask_stream`` + 调用方不 record）：第二问 prompt 不带上第一问。

    ``Testbench/debug.py:_run_chat`` 优先走 ``chat.ask_stream``，历史由调用方决定是否提交；
    debug 每次运行新建 chat 且不 record，因此这里断言"不落历史 + 不污染下一问 prompt"。
    """

    from aieyes.chat import ChatSession as _ChatSession

    vlm = StreamVlm(["调试第一句。", "调试第二句。"])
    chat = _ChatSession(vlm=vlm)
    first = "".join(str(x) for x in chat.ask_stream("调试问题一"))
    second = "".join(str(x) for x in chat.ask_stream("调试问题二"))
    assert first == "调试第一句。调试第二句。"
    assert second == "调试第一句。调试第二句。"
    assert len(vlm.prompts) == 2
    assert "调试问题一" not in vlm.prompts[1], f"debug 直调把上一问带进了 prompt:\n{vlm.prompts[1]}"


class _AnswerVlm:
    """前几问短答案、某一问长答案（按标点切多段），并记录 prompt。"""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def complete(self, prompt, *, images, max_tokens, timeout_s, temperature=None):
        self.prompts.append(prompt)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


class _Audio:
    pcm = b"\x00\x00" * 8
    sample_rate = 16000


class _PlayHandle:
    def __init__(self, finished: bool) -> None:
        self.finished = finished
        self.total_ms = 200

    def wait(self, timeout=None) -> bool:
        return self.finished


class _PieceSink:
    """第 fail_from 段起播放失败（模拟播到第二段被中断）。"""

    is_playing = False

    def __init__(self, fail_from: int = 99) -> None:
        self.fail_from = fail_from
        self.texts: list[str] = []

    def play(self, pcm, sample_rate, *, piece_id: str = "", text: str = ""):
        index = len(self.texts)
        self.texts.append(text)
        return _PlayHandle(finished=index < self.fail_from)

    def cancel(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _SynthTts:
    def synth(self, text):
        return _Audio()


LONG_TWO = "第一段答案在这里。第二段答案接着往下说。"


def test_repeat_after_partial_playback_continues_unplayed_part() -> None:
    """九.4：播到一半被打断后说“再说一遍”，要从没播完处继续。

    C4 把“没播完的一轮”判为不提交历史后，``_do_repeat`` 只按
    ``chat.turns[-1]`` 取文本会复读**更早一轮**的答案（内容错位）。
    """

    async def run():
        cfg = AppConfig()
        cfg.timers.t11_generate_ms = 5000
        cfg.timers.t11_retry_max = 0
        cfg.tts.max_segment_chars = 12  # 让 LONG_TWO 切成两段
        vlm = _AnswerVlm(["上一轮的旧答案。", LONG_TWO])
        chat = ChatSession(vlm=vlm)
        sink = _PieceSink(fail_from=1)
        engine = InteractionEngine(
            cfg=cfg,
            bus=EventBus(),
            frontend=_Frontend(),
            sink=sink,
            tts=_SynthTts(),
            chat=chat,
        )
        # 第一轮：完整播完并进历史
        await asyncio.wait_for(engine._do_chat("第一问", engine._qa_gen), timeout=10)
        first_turns = [(t.user, t.assistant) for t in chat.turns]
        # 第二轮：第二段播放失败 → 不提交历史
        await asyncio.wait_for(engine._do_chat("第二问", engine._qa_gen), timeout=10)
        after_partial = [(t.user, t.assistant) for t in chat.turns]
        # 用户说“再说一遍”
        sink.fail_from = 99
        spoken_after: list[str] = []
        real_speak = engine._speak

        async def record_speak(text: str, *, kind: str, ref_ms=None, meta=None, **kwargs) -> bool:
            spoken_after.append(text)
            return await real_speak(text, kind=kind, ref_ms=ref_ms, meta=meta, **kwargs)

        engine._speak = record_speak  # type: ignore[method-assign]
        from aieyes.intent import Intent, IntentKind

        await asyncio.wait_for(
            engine._do_repeat(Intent(kind=IntentKind.REPEAT, text="再说一遍", raw="再说一遍")),
            timeout=10,
        )
        return first_turns, after_partial, spoken_after, sink

    first_turns, after_partial, spoken_after, sink = asyncio.run(run())
    assert first_turns == [("第一问", "上一轮的旧答案。")], first_turns
    assert after_partial == [("第一问", "上一轮的旧答案。")], after_partial
    joined = "".join(spoken_after)
    assert "上一轮的旧答案。" not in joined, f"复读了上一轮答案（内容错位）: {spoken_after}"
    assert "第二段答案接着往下说。" in joined, f"没有从没播完处继续: {spoken_after}"


# ---------------------------------------------------------------------------
# 回归（根 Tests/Unit/test_engine.py::test_streaming_chat_speaks_first_phrase_early
# 抓到）：引入 ``answer_mode="append"`` 后，``_speech_worker`` 的 ``start`` 可能 >0，
# 原先 ``if idx == 0`` 的开口延迟判据会让第 2 句起不再记 ``latency.first_audio``。
# 现判据是「本次调用里实际播出的第一段」= ``idx == start``，并加 ``segment_offset``
# 字段区分首句（0）与续句（>=1）。
# ---------------------------------------------------------------------------


class _LatencySlowStreamVlm:
    """流式假模型：每句之间有到达间隔，保证引擎按句开口（3 次 ``_speak``）。"""

    def __init__(self, pieces: list[str], gap_s: float = 0.05) -> None:
        self.pieces = pieces
        self.gap_s = gap_s
        self.prompts: list[str] = []

    def complete(self, prompt, *, images, max_tokens, timeout_s, temperature=None):  # pragma: no cover
        raise AssertionError("流式路径不应调用 complete")

    def complete_stream(self, prompt, *, images, max_tokens, timeout_s, temperature=None):
        import time

        self.prompts.append(prompt)
        for index, piece in enumerate(self.pieces):
            if index:
                time.sleep(self.gap_s)
            yield piece


class _LatencyPlayHandle:
    """播放句柄替身：立刻播完（``wait`` 返回 True）。"""

    def __init__(self, total_ms: int = 20) -> None:
        self.finished = True
        self.total_ms = total_ms

    def wait(self, timeout=None) -> bool:  # noqa: ANN001 - 与 AudioHandle 签名一致
        return True

    def cancel(self) -> None:
        self.finished = False


class _LatencySink:
    """真实走 ``sink.play`` 的播放出口替身（与根 Tests/Unit 的 FakeSink 同口径）。"""

    is_playing = False

    def __init__(self) -> None:
        self.played: list[tuple[str, str]] = []

    def play(self, pcm, sample_rate=16000, *, piece_id="", text=""):  # noqa: ANN001
        self.played.append((piece_id, text))
        return _LatencyPlayHandle()

    def cancel(self) -> None:  # pragma: no cover - 本用例不取消
        pass

    def stop(self) -> None:  # pragma: no cover - 本用例不打断
        pass

    def start(self) -> None:  # pragma: no cover - 由引擎在启动时调用
        pass


class _LatencyTts:
    """TTS 替身：返回带 pcm/sample_rate 的音频对象，不触发真实合成。"""

    def __init__(self, sample_rate: int = 16000, audio_ms: int = 20) -> None:
        self.sample_rate = sample_rate
        self.samples = int(sample_rate * audio_ms / 1000)
        self.calls: list[str] = []

    def synth(self, text: str):
        self.calls.append(text)
        from types import SimpleNamespace

        return SimpleNamespace(pcm=[0.0] * self.samples, sample_rate=self.sample_rate, text=text)


def _first_audio_records(engine) -> list[dict]:
    return [r for r in engine.timings.records if r["name"] == "latency.first_audio"]


def test_streaming_three_sentences_record_three_first_audio_offsets() -> None:
    """流式三句回答：3 条 ``latency.first_audio``，``segment_offset`` 依次 0/1/2。"""

    async def run():
        cfg = AppConfig()
        cfg.timers.t11_generate_ms = 5000
        vlm = _LatencySlowStreamVlm(["甲段内容一。", "乙段内容二。", "丙段内容三。"])
        chat = ChatSession(vlm=vlm)
        sink = _LatencySink()
        engine = InteractionEngine(
            cfg=cfg,
            bus=EventBus(),
            frontend=_Frontend(),
            sink=sink,
            tts=_LatencyTts(),
            chat=chat,
        )
        engine._utterance_t0 = engine.now_ms() - 10  # 让「距用户说完」是确定的小正数
        await asyncio.wait_for(engine._do_chat("流式三句", engine._qa_gen), timeout=10)
        return engine, sink, chat

    engine, sink, chat = asyncio.run(run())

    records = _first_audio_records(engine)
    offsets = [r.get("segment_offset") for r in records]
    assert len(records) == 3, f"流式三句应产生 3 条开口延迟证据，实际 {len(records)}: {records!r}"
    assert offsets == [0, 1, 2], f"segment_offset 应依次 0/1/2，实际 {offsets!r}"
    assert all(r["speech_kind"] == "answer" for r in records), records
    # 每句都真的开口了（播放出口收到三段）
    assert [text for _pid, text in sink.played] == ["甲段内容一。", "乙段内容二。", "丙段内容三。"]
    assert _pairs(chat) == [("流式三句", "甲段内容一。乙段内容二。丙段内容三。")]


def test_first_audio_is_playback_submit_basis_and_p95_takes_offset_zero_only() -> None:
    """``latency.first_audio`` 是**软件播放提交时刻**，不是声学首声；P95 只取每轮首句。

    判据：同一轮流式三句会记 3 条 ``latency.first_audio``（``segment_offset`` 0/1/2），
    只有 ``segment_offset == 0`` 那条属于"每轮首个实质内容开口"，续句必须排除在
    P95 之外；每条都要带 ``basis="playback_submitted"``，防止被当时延/首声指标用。
    """

    async def run():
        cfg = AppConfig()
        cfg.timers.t11_generate_ms = 5000
        vlm = _LatencySlowStreamVlm(["甲段内容一。", "乙段内容二。", "丙段内容三。"])
        engine = InteractionEngine(
            cfg=cfg,
            bus=EventBus(),
            frontend=_Frontend(),
            sink=_LatencySink(),
            tts=_LatencyTts(),
            chat=ChatSession(vlm=vlm),
        )
        engine._utterance_t0 = engine.now_ms() - 10
        await asyncio.wait_for(engine._do_chat("流式三句", engine._qa_gen), timeout=10)
        return engine

    engine = asyncio.run(run())
    records = _first_audio_records(engine)
    assert len(records) == 3, records
    assert all(r.get("basis") == "playback_submitted" for r in records), (
        f"开口延迟必须标注 basis=playback_submitted（播放提交时刻）: {records!r}"
    )
    initial = [r for r in records if r.get("segment_offset") == 0]
    continuation = [r for r in records if (r.get("segment_offset") or 0) > 0]
    assert len(initial) == 1, f"每轮首个开口只能有一条: {initial!r}"
    assert len(continuation) == 2, continuation
    # P95 口径：只统计 segment_offset == 0 的样本
    p95_samples = [int(r["ms"]) for r in records if r.get("segment_offset") == 0]
    assert p95_samples == [int(initial[0]["ms"])], p95_samples
    assert all(int(r["ms"]) >= 0 for r in records), records


# ---------------------------------------------------------------------------
# 回归（M9 独立复验指出、speech-b2 上报）：流式播报被打断时，**已生成但还没开播**
# 的尾部句子原先从未登记进 answer run，用户说「接着说完」只能续到中断那一刻，
# 尾句永久丢失（非流式一次性 split_text 出全部段，所以能续到末尾）。
# 修法：``_do_chat_streaming`` 在 play_failed 时把 buf + 队列里剩余 delta 登记进 run。
# ---------------------------------------------------------------------------

from aieyes.intent import Intent, IntentKind  # noqa: E402 - 追加块内的模块级 import


class _TailHandle:
    def __init__(self, finished: bool, play_ms: int) -> None:
        self.finished = finished
        self.total_ms = play_ms
        self.play_ms = play_ms

    def wait(self, timeout=None) -> bool:
        if not self.finished:
            import time

            time.sleep(self.play_ms / 1000)
        return self.finished


class _TailSink:
    """第 ``fail_from`` 次播放起「播一会儿后被打断」。"""

    is_playing = False

    def __init__(self, fail_from: int = 1, play_ms: int = 150) -> None:
        self.fail_from = fail_from
        self.play_ms = play_ms
        self.played: list[str] = []

    def play(self, pcm, sample_rate=16000, *, piece_id="", text=""):
        index = len(self.played)
        self.played.append(text)
        failing = index >= self.fail_from
        return _TailHandle(finished=not failing, play_ms=self.play_ms)

    def cancel(self) -> None:  # pragma: no cover
        pass

    def stop(self) -> None:  # pragma: no cover
        pass


class _TailTts:
    def synth(self, text):
        from types import SimpleNamespace

        return SimpleNamespace(pcm=[0.0] * 320, sample_rate=16000, text=text)


class _TailStreamVlm:
    """生成快于播报：每句间隔 20ms，足以在被中断前把后续两句推进队列。"""

    def __init__(self, pieces: list[str], gap_s: float = 0.02) -> None:
        self.pieces = pieces
        self.gap_s = gap_s
        self.prompts: list[str] = []

    def complete(self, prompt, *, images, max_tokens, timeout_s, temperature=None):  # pragma: no cover
        raise AssertionError("流式路径不应调用 complete")

    def complete_stream(self, prompt, *, images, max_tokens, timeout_s, temperature=None):
        import time

        self.prompts.append(prompt)
        for index, piece in enumerate(self.pieces):
            if index:
                time.sleep(self.gap_s)
            yield piece


TAIL_SENTENCES = ["甲段内容一。", "乙段内容二。", "丙段内容三。", "丁段内容四。"]


def test_streaming_interrupt_keeps_unplayed_tail_for_repeat() -> None:
    """流式第 2 句被打断：run 里必须还留着丙/丁，续播能把尾句说完。"""

    async def run():
        cfg = AppConfig()
        cfg.timers.t11_generate_ms = 5000
        vlm = _TailStreamVlm(TAIL_SENTENCES)
        chat = ChatSession(vlm=vlm)
        sink = _TailSink(fail_from=1, play_ms=150)
        engine = InteractionEngine(
            cfg=cfg,
            bus=EventBus(),
            frontend=_Frontend(),
            sink=sink,
            tts=_TailTts(),
            chat=chat,
        )
        engine._utterance_t0 = engine.now_ms() - 10
        await asyncio.wait_for(engine._do_chat("说来听听", engine._qa_gen), timeout=20)
        pieces = list(engine._answer_run_pieces)
        unplayed = engine._unplayed_answer_text()
        played_before = list(sink.played)
        # 用户说「接着说完」：此时播报应恢复正常
        sink.fail_from = 99
        await asyncio.wait_for(
            engine._do_repeat(Intent(kind=IntentKind.CHAT, text="接着说完", raw="接着说完")),
            timeout=20,
        )
        return engine, chat, sink, pieces, unplayed, played_before

    engine, chat, sink, pieces, unplayed, played_before = asyncio.run(run())

    joined = "".join(pieces)
    assert joined == "".join(TAIL_SENTENCES), f"run 里丢了句子: pieces={pieces!r}"
    assert unplayed == "乙段内容二。丙段内容三。丁段内容四。", f"未播部分不对: {unplayed!r}"
    assert unplayed.endswith("丁段内容四。"), "尾句没进 run（流式中断丢尾）"
    # 被打断的一轮不进历史（C4 契约），续播只播不记
    assert _pairs(chat) == [], _pairs(chat)
    spoken_after = sink.played[len(played_before):]
    assert spoken_after and spoken_after[0].startswith("刚才那一句"), spoken_after
    assert unplayed in "".join(spoken_after), f"续播没有把尾句说完: {spoken_after!r}"
    names = [r["name"] for r in engine.timings.records]
    assert "repeat.resume_unplayed" in names, names


# ---------------------------------------------------------------------------
# 必修 B：``cancelled`` 布尔分不出「播放卡死」与「用户主动取消」。
# 引擎原先两者都压成 ``reason_code="interrupted"``，PlaybackFinished 还用基类构造，
# 于是 ``playback_stalled`` 在上报层完全不可见。修法：reason 从 handle.stop_reason
# 取，同一字符串同时进 ``reason_code`` 与 ``PlaybackFinished.stop_reason``。
# ---------------------------------------------------------------------------


class _ReasonHandle:
    def __init__(
        self,
        finished: bool,
        stop_reason: str = "",
        total_ms: int = 400,
        played_ms: int = 380,
    ) -> None:
        self._finished = finished
        self.stop_reason = stop_reason
        self.total_ms = total_ms
        self.played_ms = played_ms

    def wait(self, timeout=None) -> bool:
        return self._finished

    def cancel(self) -> None:  # pragma: no cover
        pass


class _ReasonSink:
    is_playing = False

    def __init__(self, handle: _ReasonHandle) -> None:
        self.handle = handle
        self.texts: list[str] = []

    def play(self, pcm, sample_rate=16000, *, piece_id: str = "", text: str = ""):
        self.texts.append(text)
        return self.handle

    def cancel(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _OneAnswerVlm:
    def __init__(self, answer: str = "好的，我知道了。") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def complete(self, prompt, *, images, max_tokens, timeout_s, temperature=None):
        self.prompts.append(prompt)
        return self.answer


class _RecordingContract:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def publish_playback(self, **payload):
        self.payloads.append(payload)


def _drain_finished(bus) -> list:
    import aieyes.events as events

    out = []
    while True:
        try:
            ev = bus._queue.get_nowait()
        except Exception:
            break
        if isinstance(ev, events.PlaybackFinished):
            out.append(ev)
    return out


def _run_one_playback(*, finished: bool, stop_reason: str, played_ms: int = 380):
    """跑一轮问答，返回 (contract payloads, PlaybackFinished 事件, 播放文本)。"""

    async def run():
        cfg = AppConfig()
        cfg.timers.t11_generate_ms = 5000
        sink = _ReasonSink(_ReasonHandle(finished, stop_reason, played_ms=played_ms))
        bus = EventBus()
        engine = InteractionEngine(
            cfg=cfg,
            bus=bus,
            frontend=_Frontend(),
            sink=sink,
            tts=_SynthTts(),
            chat=ChatSession(vlm=_OneAnswerVlm()),
        )
        contract = _RecordingContract()
        engine.contract = contract
        engine._loop = asyncio.get_running_loop()
        engine._utterance_t0 = engine.now_ms() - 10
        await asyncio.wait_for(engine._do_chat("问一句", engine._qa_gen), timeout=10)
        for _ in range(5):
            await asyncio.sleep(0)
        return contract.payloads, _drain_finished(bus), list(sink.texts)

    return asyncio.run(run())


def test_playback_stalled_and_user_cancel_are_distinguishable_in_evidence() -> None:
    """卡死与用户取消在证据层必须分开（原先两者都是 ``interrupted``）。"""
    stalled_payloads, stalled_events, stalled_texts = _run_one_playback(
        finished=False, stop_reason="playback_stalled"
    )
    cancel_payloads, cancel_events, cancel_texts = _run_one_playback(
        finished=False, stop_reason="user_cancel"
    )

    stalled_codes = [p["reason_code"] for p in stalled_payloads if p.get("state") == "cancelled"]
    cancel_codes = [p["reason_code"] for p in cancel_payloads if p.get("state") == "cancelled"]
    assert stalled_codes == ["playback_stalled"], stalled_payloads
    assert cancel_codes == ["user_cancel"], cancel_payloads
    assert stalled_codes != cancel_codes

    assert stalled_events and cancel_events
    assert stalled_events[-1].stop_reason == "playback_stalled", stalled_events
    assert cancel_events[-1].stop_reason == "user_cancel", cancel_events
    assert stalled_events[-1].cancelled is True
    assert cancel_events[-1].cancelled is True
    # 都只播了被中断的那一段，没有因为「区分原因」多播
    assert len(stalled_texts) == 1 and len(cancel_texts) == 1, (stalled_texts, cancel_texts)
    # 去重后页面只收到 engine 这条终态：「已经播了多少」必须保留句柄真值，不能被抹成 0
    assert stalled_events[-1].played_ms == 380, stalled_events[-1]
    assert cancel_events[-1].played_ms == 380, cancel_events[-1]
    assert stalled_events[-1].played_ms > 0 and cancel_events[-1].played_ms > 0


def test_playback_completed_keeps_reason_code_none_and_empty_stop_reason() -> None:
    """正常播完：``reason_code`` 仍为 None，``stop_reason`` 为 ``completed``（与 sink 侧常量一致）。"""
    # 真实 sink 正常播完时 played_ms == total_ms
    payloads, finished_events, texts = _run_one_playback(
        finished=True, stop_reason="", played_ms=400
    )
    completed = [p for p in payloads if p.get("state") == "completed"]
    assert completed, payloads
    assert [p["reason_code"] for p in completed] == [None], completed
    # 正常播完要能和「未标注的旧事件」区分开：sink 侧的常量就是 "completed"
    assert finished_events and finished_events[-1].stop_reason == "completed", finished_events
    assert finished_events[-1].cancelled is False
    assert finished_events[-1].played_ms == finished_events[-1].total_ms, finished_events
    assert len(texts) == 1, texts


# ---------------------------------------------------------------------------
# 必修 A 的一致性：同一 4 句、同一中断点，流式与非流式的「未播部分」必须一致。
# （中断的那一段算「没播完」，要重播；尾句更不许丢——M9 复现的缺陷就在这里。）
# ---------------------------------------------------------------------------


def test_streaming_and_nonstreaming_report_same_unplayed_tail() -> None:
    join = "".join(TAIL_SENTENCES)

    async def run_stream():
        cfg = AppConfig()
        cfg.timers.t11_generate_ms = 5000
        cfg.tts.max_segment_chars = 6  # 与非流式同粒度：每句一段
        sink = _TailSink(fail_from=1, play_ms=150)
        engine = InteractionEngine(
            cfg=cfg,
            bus=EventBus(),
            frontend=_Frontend(),
            sink=sink,
            tts=_TailTts(),
            chat=ChatSession(vlm=_TailStreamVlm(TAIL_SENTENCES)),
        )
        engine._utterance_t0 = engine.now_ms() - 10
        await asyncio.wait_for(engine._do_chat("说来听听", engine._qa_gen), timeout=20)
        return engine._unplayed_answer_text(), list(sink.played)

    async def run_plain():
        cfg = AppConfig()
        cfg.timers.t11_generate_ms = 5000
        cfg.tts.max_segment_chars = 6  # 与非流式同粒度：每句一段
        sink = _PieceSink(fail_from=1)
        engine = InteractionEngine(
            cfg=cfg,
            bus=EventBus(),
            frontend=_Frontend(),
            sink=sink,
            tts=_SynthTts(),
            chat=ChatSession(vlm=_AnswerVlm([join])),
        )
        engine._utterance_t0 = engine.now_ms() - 10
        await asyncio.wait_for(engine._do_chat("说来听听", engine._qa_gen), timeout=20)
        return engine._unplayed_answer_text(), list(sink.texts)

    stream_unplayed, stream_played = asyncio.run(run_stream())
    plain_unplayed, plain_played = asyncio.run(run_plain())

    assert stream_unplayed == plain_unplayed, (
        f"流式与非流式的未播部分不一致: stream={stream_unplayed!r} plain={plain_unplayed!r}"
    )
    assert stream_unplayed.endswith("丁段内容四。"), f"尾句丢了: {stream_unplayed!r}"
    assert "丙段内容三。" in stream_unplayed, stream_unplayed
    # 中断后没有多播：两条路径都只播了「第一段 + 被打断的那一段」
    assert len(stream_played) == 2, stream_played
    assert len(plain_played) == 2, plain_played
