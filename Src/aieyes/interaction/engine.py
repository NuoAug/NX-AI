"""会话交互状态机（Lead 维护）。

职责：消费 ``EventBus`` 上的语音/合同/计时事件，串行驱动一次会话的状态迁移，
保证「输出唯一收尾」：任何时刻最多一条播报链，取消先于新播报，且会话结束
（回待机/关闭）时不再有遗留音频。

状态
----
``IDLE``      待机：仅监听唤醒词
``GREETING``  已唤醒，正在播报“我在”（允许用户立刻插话）
``LISTENING`` 等待用户说话（T01 提醒/延长）
``THINKING``  已收到用户问题，等待模型回答（T11）
``SPEAKING``  正在播报回答（可被打断 → 停声 → 听新话）
``CLOSED``    已关闭

线程模型：所有状态迁移都发生在唯一的 asyncio 任务里；采集线程只通过
``EventBus.post`` 投递事件（由本模块在 ``frontend`` 回调里完成绑定）。
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from .. import events
from ..chat import ChatSession
from ..clock import CLOCK_ID, mono_ms
from ..config import AppConfig
from ..contract.guidance_intent import (
    classify_guidance_action,
    send_guidance_intent,
)
from ..contract.handlers import (
    cancel_result_data,
    capabilities_result_data,
    interaction_status_data,
)
from ..events import EventBus
from ..imagetasks import CapacityError, ImageTaskStore
from ..identity import CandidateDecision, IdentityManager, IdentityResult
from ..intent import Intent, IntentKind, classify
from ..interfaces import AudioSink, Tts, VoiceFrontend
from ..logging_setup import JsonlEvidenceLogger

log = logging.getLogger("aieyes.interaction")

_PIECE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|\n+")
#: 流式播报：这些标点后可切短语（逗号也算，尽量早开口）
_CLAUSE_ENDS = "。！？!?；;，,、：:\n"
#: 首个可播短语的最小/最大长度（字符）
_CLAUSE_MIN_CHARS = 6
_CLAUSE_MAX_CHARS = 24


class InteractionState(str, Enum):
    IDLE = "idle"
    GREETING = "greeting"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CLOSED = "closed"


@dataclass
class Timings:
    """一次会话的计时证据（毫秒，输出为 JSON 可直接引用）。"""

    records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, ms: float, **extra: Any) -> None:
        record = {"name": name, "ms": round(float(ms), 1), "t_ms": mono_ms(), **extra}
        self.records.append(record)


def split_text(text: str, max_chars: int = 60) -> list[str]:
    """把回答按标点切成适合逐段合成的短句（保留标点）。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for chunk in _PIECE_SPLIT_RE.split(text):
        chunk = chunk.strip()
        if not chunk:
            continue
        while len(chunk) > max_chars:
            pieces.append(chunk[:max_chars])
            chunk = chunk[max_chars:]
        if chunk:
            pieces.append(chunk)
    return pieces or [text]


def _split_first_phrase(buffer: str) -> tuple[str | None, str]:
    """从流式缓冲里切出可播的第一个短语（先按分句标点，过长再硬切）。"""
    for index, char in enumerate(buffer):
        if index + 1 >= _CLAUSE_MIN_CHARS and char in _CLAUSE_ENDS:
            return buffer[: index + 1], buffer[index + 1 :]
    if len(buffer) >= _CLAUSE_MAX_CHARS:
        return buffer[:_CLAUSE_MAX_CHARS], buffer[_CLAUSE_MAX_CHARS :]
    return None, buffer


# 阶段 3：单图场景问答（取图 + OCR/VLM）可承接的意图
#: 引擎意图 → 场景模块意图名（避免 ASR 误字让场景层二次分类跑偏）。
_SCENE_INTENT_HINT: dict[IntentKind, str] = {
    IntentKind.LOOK: "describe",
    IntentKind.FIND: "find",
    IntentKind.PRICE: "price",
    IntentKind.DATE: "date",
}
_SCENE_KINDS: frozenset[IntentKind] = frozenset(
    {IntentKind.LOOK, IntentKind.PRICE, IntentKind.DATE, IntentKind.FIND}
)
# 第九章 4 固定话术：只有可信片段边界，从未播完处接着说的提示
NOTICE_REPEAT_RESUME = "刚才那一句可能没播完整，我从这里接着说。"


class InteractionEngine:
    """语音交互状态机（阶段 1/2 的主链路）。"""

    def __init__(
        self,
        *,
        cfg: AppConfig,
        bus: EventBus,
        frontend: VoiceFrontend,
        sink: AudioSink,
        tts: Tts,
        chat: ChatSession,
        scene: Any | None = None,
        image_tasks: ImageTaskStore | None = None,
        identity: IdentityManager | None = None,
        greeting_text: str = "我在",
        reminder_text: str = "我在，请说",
        idle_text: str = "我先待机了，需要时叫我",
        evidence: JsonlEvidenceLogger | None = None,
        now_ms: Callable[[], int] = mono_ms,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.frontend = frontend
        self.sink = sink
        self.tts = tts
        self.chat = chat
        self.scene = scene  # 阶段 3 起：场景问答服务（可选，None 时视觉意图降级说明）
        self.image_tasks = image_tasks or ImageTaskStore(
            t08_ms=cfg.timers.t08_dual_capture_ms,
            max_captures_per_position=cfg.timers.t10_max_grab_per_position,
            task_deadline_ms=cfg.timers.t20_image_task_ms,
            group_deadline_ms=cfg.timers.t20_evidence_group_ms,
        )
        self.identity = identity  # 阶段 5：本地声纹依据与资料访问门（可选）
        self.greeting_text = greeting_text
        self.reminder_text = reminder_text
        self.idle_text = idle_text
        self.evidence = evidence
        self.now_ms = now_ms

        self.state = InteractionState.IDLE
        self.timings = Timings()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._speech_task: asyncio.Task[bool] | None = None
        self._flow_task: asyncio.Task[Any] | None = None
        self._speech_gen = 0
        self._greeting_audio = None
        self._timers: set[asyncio.TimerHandle] = set()
        self._extends = 0
        self._closed = False
        self._utterance_t0: int | None = None
        self._utterance_end_ms: int | None = None
        # 本回合话语 id：每回合开始递增，回合内稳定（control.intent 与播报证据共用）。
        self._utterance_seq = 0
        self._utterance_id = ""
        self._wake_t0: int | None = None
        self._playback_t0: int | None = None
        self._llm_thread_lock = threading.Lock()
        self._qa_gen = 0  # 问答代次：停声/打断后旧结果作废
        self._barge_pending = False  # 刚发生过插话打断，等待用户实际话语
        self._empty_say_attempts = 0  # T02：连续空识别次数
        self._muted = False
        # 阶段 5：免唤醒（话题窗口）与身份状态
        self._wakefree = False
        self._wakefree_onset_ms: int | None = None
        self._last_identity: IdentityResult | None = None
        # 阶段 6：合同桥（None=未启用，本地流程完全不受影响）
        self.contract: Any | None = None
        self._guidance_seen: dict[str, int] = {}
        self._guidance_pending: dict[str, Any] | None = None
        self._active_request_id: str | None = None
        self._playback_priority: dict[str, Any] | None = None
        # 最近一次「回答」播报的 run 状态（供第九章续播复用，不复制分段逻辑）。
        # 一次回答 = 一个 run：流式分多次 _speak 时按 append 追加片段；
        # token 隔离被取消/换问题后的旧状态，避免旧线程把新 run 的索引写脏。
        self._answer_run_token = 0
        self._answer_run_pieces: list[str] = []
        self._answer_run_index = -1
        self._answer_run_text = ""
        # 最近一次 _ask_model 等待生成锁的耗时（ms，-1 表示还没拿到/未开始）
        self._lock_wait_ms = -1.0
        # 上一次视觉回答实际用过的帧：同图追问时复用，避免每问必重拍。
        # 取消/停止/换问题后必须清掉（见 _drop_scene_frame），否则会拿旧图当新画面。
        self._last_scene_frame: Any | None = None
        # 最近一次比较任务（纠正时按位置释放旧帧用；None 表示没有可纠正的任务）
        self._last_compare_task: Any | None = None
        # “刚才那个”这类插问指向不明时，记下待回答的问题，等用户补充第一件/第二件。
        self._compare_recall_pending: dict[str, str] | None = None
        # 等待拍摄时“先别拍”的短回执计数：第一次带用法，之后只说“我等你”。
        self._compare_hold_acks = 0
        # 上一轮回答的对象与类型：纠正“不是第二个，是第一个”时区分改指代/换归属。
        self._last_answer_kind = ""  # compare | recall | scene
        self._last_answer_position = ""  # A | B | AB | S
        self._last_recall_question = ""
        # “你想重拍第一件还是第二件？”之后等待用户只回答“第一件/第二件”。
        self._reshoot_pending: dict[str, str] | None = None
        # 双图各位置连续失败次数：用于有界重试提示，不阻断用户退出。
        self._compare_capture_failures: dict[tuple[str, str], int] = {}
        # 帧对象缓存（`frame_id -> frame`，最近 8 张）：证据存储只留 id，历史图引用
        # （`source="history"`）要按 id 回取帧对象，重拍放弃的旧帧也在这里留一份。
        self._frame_cache: dict[str, Any] = {}
        self._frame_cache_order: list[str] = []

    # ------------------------------------------------------------------
    def attach_contract(self, contract: Any | None) -> None:
        """接上 interaction.v0.1 合同桥；None 表示合同未启用。"""
        self.contract = contract

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._run_task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._bind_frontend()
        # 先进入待机，再启动事件消费与 TTS 预热：否则前端事件（唤醒）可能在
        # 预热期间被处理，随后 _enter_idle 又把状态覆盖回 IDLE，导致流程卡死。
        await self._enter_idle(initial=True)
        self._run_task = asyncio.create_task(self._consume(), name="aieyes-interaction")
        try:
            self._greeting_audio = await asyncio.to_thread(self.tts.synth, self.greeting_text)
            self._record("startup", "greeting_presynth", 0, tts_ms=self._greeting_audio.duration_ms)
        except Exception as exc:  # noqa: BLE001 - 预热失败不阻塞会话
            log.warning("greeting presynth failed: %r", exc)
            self._greeting_audio = None

    async def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel_timers()
        await self._interrupt_speech()
        if self._run_task is not None:
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self.state = InteractionState.CLOSED
        self._record("shutdown", "closed", 0)

    # ------------------------------------------------------------------
    # 前端回调 -> 事件总线（采集线程安全）
    # ------------------------------------------------------------------
    def _bind_frontend(self) -> None:
        self.frontend.on_wake = lambda keyword, t_ms: self.bus.post(events.WakeDetected(keyword, t_ms))
        self.frontend.on_speech_start = lambda t_ms: self.bus.post(events.SpeechStarted(t_ms))
        self.frontend.on_partial = lambda text, t_ms: self.bus.post(events.PartialText(text, t_ms))
        self.frontend.on_utterance = lambda utt: self.bus.post(
            events.UtteranceRecognized(
                utt.text,
                utt.t_audio_start_ms,
                utt.t_audio_end_ms,
                utt.t_final_ms,
                samples=getattr(utt, "samples", None),
            )
        )
        self.frontend.on_level = lambda level, t_ms: None  # 电平不上行，避免拖慢事件队列

    # ------------------------------------------------------------------
    # 事件循环
    # ------------------------------------------------------------------
    async def _consume(self) -> None:
        while not self._closed:
            event = await self.bus.get()
            try:
                await self._dispatch(event)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - 单个事件失败不能拖垮会话
                log.exception("interaction event failed: %r", event)

    async def _dispatch(self, event: Any) -> None:
        if isinstance(event, events.WakeDetected):
            await self._on_wake(event)
        elif isinstance(event, events.SpeechStarted):
            await self._on_speech_start(event)
        elif isinstance(event, events.UtteranceRecognized):
            await self._on_utterance(event)
        elif isinstance(event, events.CancelRequested):
            await self._on_cancel(event)
        elif isinstance(event, events.DeviceFault):
            await self._on_device_fault(event)
        elif isinstance(event, events.ShutdownRequested):
            await self.stop()
        # PartialText/FrameCaptured/ContractIncoming 等阶段 3+ 再接入

    # ------------------------------------------------------------------
    # 唤醒 / 打招呼
    # ------------------------------------------------------------------
    async def _on_wake(self, ev: events.WakeDetected) -> None:
        if self.state != InteractionState.IDLE:
            self._record("wake", "ignored", 0, state=self.state.value, keyword=ev.keyword)
            return
        self._wake_t0 = ev.t_ms
        self._extends = 0
        self.state = InteractionState.GREETING
        self.frontend.set_listening(True)
        self.frontend.set_wake_enabled(False)
        self._record("wake", "detected", self.now_ms() - ev.t_ms, keyword=ev.keyword)
        self._start_flow(self._greet_flow(ev))

    async def _greet_flow(self, ev: events.WakeDetected) -> None:
        played = await self._speak(self.greeting_text, kind="greeting", ref_ms=ev.t_ms)
        # 用户可能在打招呼时就插话：打断后直接进入倾听
        self._record("wake", "greeting_done", self.now_ms() - ev.t_ms, played=played)
        if self.state in (InteractionState.GREETING, InteractionState.LISTENING):
            await self._enter_listening()

    # ------------------------------------------------------------------
    # 听
    # ------------------------------------------------------------------
    async def _enter_listening(self) -> None:
        self._exit_wakefree("explicit_wake")
        self.state = InteractionState.LISTENING
        self._cancel_timers()
        self.frontend.set_listening(True)
        self.frontend.set_wake_enabled(False)
        timers = self.cfg.timers
        self._after(timers.t01_remind_at_ms, self._on_remind)
        self._after(timers.t01_initial_wait_ms, self._on_wait_timeout)
        self._record("listen", "start", 0, extends=self._extends)

    async def _enter_idle(self, *, initial: bool = False) -> None:
        self._cancel_timers()
        self._exit_wakefree("idle")
        self.state = InteractionState.IDLE
        self.frontend.set_listening(False)
        self.frontend.set_wake_enabled(True)
        self._record("idle", "enter", 0, initial=initial)

    def _exit_wakefree(self, reason: str) -> None:
        """退出免唤醒话题窗口（回待机/显式唤醒/清除资料时调用）。"""
        if not self._wakefree:
            return
        self._wakefree = False
        self._wakefree_onset_ms = None
        if self.identity is not None:
            self.identity.close_topic()
        self._record("wakefree", "exit", 0, reason=reason)

    async def _on_remind(self) -> None:
        if self.state is not InteractionState.LISTENING:
            # 提醒到点时若正在播报（例如上一轮提醒/回答还没播完），顺延重试而不是静默丢计时器
            if not self._closed:
                self._after(500, self._on_remind)
            return
        self._record("t01", "remind", 0)
        self.frontend.set_listening(False)
        await self._speak(self.reminder_text, kind="reminder")
        if self.state is InteractionState.LISTENING:
            self.frontend.set_listening(True)

    async def _on_wait_timeout(self) -> None:
        if self.state is not InteractionState.LISTENING:
            if not self._closed:
                self._after(500, self._on_wait_timeout)
            return
        timers = self.cfg.timers
        self._extends += 1
        if self._extends <= timers.t01_max_extend:
            self._record("t01", "extend", 0, extends=self._extends)
            self._after(timers.t01_extend_ms, self._on_wait_timeout)
            return
        self._record("t01", "idle", 0, extends=self._extends)
        self.state = InteractionState.SPEAKING
        self.frontend.set_listening(False)
        await self._speak(self.idle_text, kind="idle")
        await self._enter_idle()

    # ------------------------------------------------------------------
    # 说 / 插话
    # ------------------------------------------------------------------
    async def _on_speech_start(self, ev: events.SpeechStarted) -> None:
        if self.state in (InteractionState.SPEAKING, InteractionState.GREETING, InteractionState.THINKING):
            # 停声时延从 VAD 播报的语音起点算起（含事件排队时间），对齐 T 计时口径
            t0 = ev.t_ms
            was_thinking = self.state is InteractionState.THINKING
            # 思考中被打断：保留流程任务，让它按 _qa_gen 判定并丢弃旧结果；
            # 播报中被打断：直接取消流程，避免旧回答继续排队。
            metrics = await self._interrupt_speech(
                cancel_flow=not was_thinking, reason="barge_in"
            )
            self._record("barge_in", "silenced", self.now_ms() - t0, **metrics)
            self._barge_pending = True
            await self._enter_listening()
        self._utterance_t0 = ev.t_ms
        if self._wakefree:
            self._wakefree_onset_ms = ev.t_ms
            if self.identity is not None:
                self.identity.begin_candidate(ev.t_ms)
        self._record("listen", "speech_start", 0, state=self.state.value)

    def _silence_now(self) -> None:
        """同步停声：失效播报/问答代次、取消播放任务并通知播放出口取消。"""
        self._speech_gen += 1
        self._qa_gen += 1
        self._drop_scene_frame()
        task = self._speech_task
        self._speech_task = None
        if task is not None and not task.done():
            task.cancel()
        try:
            self.sink.cancel()
        except Exception:  # noqa: BLE001 - AudioSink 未实现 cancel 时退回 stop
            try:
                self.sink.stop()
            except Exception:  # noqa: BLE001
                pass

    def _start_flow(self, coro: Any) -> asyncio.Task[Any] | None:
        """启动新的交互流程任务；旧流程与正在播报的内容一并让位。"""
        if self._loop is None or self._closed:
            return None
        old = self._flow_task
        if old is not None and not old.done():
            old.cancel()
            self._silence_now()
        elif self.sink.is_playing:
            self._silence_now()
        task = self._loop.create_task(coro)
        task.add_done_callback(_log_task_error)
        self._flow_task = task
        return task

    async def _interrupt_speech(self, *, cancel_flow: bool = True, reason: str = "interrupt") -> dict[str, float]:
        """停声并记录取消时刻：先失效播报/问答代次，再取消播放任务/当前流程。"""
        t0 = self.now_ms()
        try:
            played_before = int(self.sink.played_ms())
        except Exception:  # noqa: BLE001 - 播放出口未实现游标时按 0 记
            played_before = 0
        speech_task = self._speech_task
        flow_task = self._flow_task if cancel_flow else None
        if flow_task is not None and flow_task is asyncio.current_task():
            flow_task = None
        self._silence_now()
        t_silenced = self.now_ms()
        for task in (speech_task, flow_task):
            if task is not None and not task.done() and task is not asyncio.current_task():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        metrics = {
            "silenced_ms": t_silenced - t0,
            "queue_ms": t_silenced - t0,
            "old_task_ms": self.now_ms() - t0,
            "played_ms": played_before,
        }
        self._record("cancel", "path", metrics["old_task_ms"], reason=reason, **metrics)
        return metrics

    async def _speak(
        self,
        text: str,
        *,
        kind: str,
        ref_ms: int | None = None,
        meta: dict[str, Any] | None = None,
        answer_mode: str = "new",
        answer_pieces: list[str] | None = None,
        answer_offset: int = 0,
    ) -> bool:
        """播报一段文本；返回是否完整播完（被打断/失败返回 False）。

        ``kind="answer"`` 时按 run 记续播状态（返回值契约不变，仍是 bool）：

        - ``answer_mode="new"``：新的一次回答，重置 run（pieces/索引/全文）；
        - ``"append"``：流式同一回答的后续句，把 ``text`` 追加到 run 全文；
        - ``"continue"``：续播已存在 run 的未播部分（``answer_pieces`` +
          ``answer_offset`` 指定从第几段起），全文不变。
        """
        if not (text or "").strip():
            return True
        self._speech_gen += 1
        gen = self._speech_gen
        task = asyncio.create_task(
            self._speech_worker(
                text,
                gen,
                kind,
                ref_ms,
                meta,
                answer_mode=answer_mode,
                answer_pieces=answer_pieces,
                answer_offset=answer_offset,
            )
        )
        self._speech_task = task
        try:
            return await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                # 当前流程自身被取消（插话/停止/新流程让位）：让取消继续向上传播
                raise
            return False
        except Exception:  # noqa: BLE001 - TTS/播放失败按“未播完”处理
            log.exception("speak failed kind=%s", kind)
            self._record("speech", "error", 0, kind=kind)
            self._publish_playback("failed", kind=kind, generation=gen, reason_code="tts_error", meta=meta)
            return False
        finally:
            if self._speech_task is task:
                self._speech_task = None

    async def _speech_worker(
        self,
        text: str,
        gen: int,
        kind: str,
        ref_ms: int | None,
        meta: dict[str, Any] | None = None,
        *,
        answer_mode: str = "new",
        answer_pieces: list[str] | None = None,
        answer_offset: int = 0,
    ) -> bool:
        if answer_pieces is not None:
            # 续播/追加：分段来自同一份 run 列表，不重复 split_text
            pieces = list(answer_pieces)
        else:
            pieces = split_text(text, self.cfg.tts.max_segment_chars)
        run_token = 0
        start = 0
        if kind == "answer":
            # 第九章续播要用「可信片段位置」：run 内记分段与已播完到第几段
            run_token = self._begin_answer_run(text, answer_mode, pieces, answer_offset)
            if answer_mode != "new":
                start = max(0, min(answer_offset, len(pieces)))
        prev_state = self.state
        self.state = InteractionState.SPEAKING if prev_state is not InteractionState.GREETING else prev_state
        self.frontend.set_wake_enabled(False)
        self._record("speech", "worker_start", self.now_ms() - (ref_ms or self.now_ms()), speech_kind=kind)
        played_all = True
        try:
            for idx in range(start, len(pieces)):
                piece = pieces[idx]
                if gen != self._speech_gen or self._closed:
                    played_all = False
                    break
                if kind == "greeting" and idx == 0 and self._greeting_audio is not None:
                    audio = self._greeting_audio
                else:
                    audio = await asyncio.to_thread(self.tts.synth, piece)
                if gen != self._speech_gen or self._closed:
                    played_all = False
                    break
                piece_id = f"{kind}-{gen}-{idx}"
                self._publish_playback("queued", kind=kind, segment_index=idx, generation=gen, meta=meta)
                handle = self.sink.play(audio.pcm, audio.sample_rate, piece_id=piece_id, text=piece)
                t_play = self.now_ms()
                self._publish_playback(
                    "started",
                    kind=kind,
                    segment_index=idx,
                    generation=gen,
                    start_basis=("device_write" if type(self.sink).__name__ == "AudioSink" else "estimated"),
                    meta=meta,
                )
                if idx == start:
                    # 「本次调用里实际播出的第一段」＝开口。answer_mode="new" 时 start==0，与改前一致；
                    # append/continue 时记的是这一次续播的开口延迟（base 仍是距用户说完的累计量）。
                    #
                    # `basis` 说明这是什么时刻：**软件播放提交时刻**（`sink.play()`
                    # 返回、音频交给设备之后），不是声学首声——本机没有麦克风/扬声器
                    # 回路采样，任何措辞或统计都不得把它当"用户听到第一声"的延迟。
                    # 续句（`segment_offset > 0`）是同一轮回答的中途续播，与"每轮首个
                    # 实质内容开口"不是同一个统计对象：算 P95 时只取 `segment_offset == 0`。
                    base = ref_ms if ref_ms is not None else (self._utterance_t0 or t_play)
                    self._record(
                        "latency",
                        "first_audio",
                        t_play - base,
                        speech_kind=kind,
                        segment_offset=start,
                        basis="playback_submitted",
                    )
                self.bus.post(events.PlaybackStarted(piece_id, t_play, piece))
                finished = await asyncio.to_thread(handle.wait, None)
                # `wait()` 的布尔只说明「是否结束」，不说明为什么结束：用户取消时
                # 返回 True、播放卡死时返回 False。原因由 handle 的 `stop_reason`
                # 带出来（``playback_stalled`` / ``user_cancel`` …），证据行与
                # `PlaybackFinished.stop_reason` 用同一个字符串，别再把两者都压成
                # ``interrupted``——那样「卡死」和「用户主动打断」在数据上不可分。
                reason = getattr(handle, "stop_reason", "") or ("completed" if finished else "interrupted")
                # 已经播了多少：优先取句柄的真值（sink 会按实际播出的毫秒记
                # ``played_ms``），取不到才回落到旧口径。被去重后页面只收到 engine
                # 这一条终态，「已经播了多少」不能在这里被抹成 0。
                played_ms = int(getattr(handle, "played_ms", 0) or (handle.total_ms if finished else 0))
                self._publish_playback(
                    "completed" if finished else "cancelled",
                    kind=kind,
                    segment_index=idx,
                    generation=gen,
                    reason_code=None if finished else reason,
                    meta=meta,
                )
                self.bus.post(
                    events.PlaybackFinished(
                        piece_id,
                        self.now_ms(),
                        played_ms,
                        handle.total_ms,
                        cancelled=not finished,
                        text=piece,
                        stop_reason=reason,
                    )
                )
                if not finished:
                    played_all = False
                    break
                if kind == "answer":
                    self._commit_answer_run_index(run_token, idx)
        finally:
            if gen == self._speech_gen and not self._closed:
                self.frontend.set_wake_enabled(True)
                if prev_state is not InteractionState.GREETING:
                    self.state = prev_state if prev_state in (InteractionState.LISTENING, InteractionState.SPEAKING) else InteractionState.LISTENING
            self._flush_pending_guidance()
        return played_all

    # ------------------------------------------------------------------
    # 说完了：意图路由与回答
    # ------------------------------------------------------------------
    async def _on_utterance(self, ev: events.UtteranceRecognized) -> None:
        if self.state is InteractionState.IDLE:
            return
        self._cancel_timers()
        self._utterance_end_ms = ev.t_audio_end_ms
        text = (ev.text or "").strip()
        self._record("utterance", "final", 0, chars=len(text), text=text[:80])
        # ASR 收尾耗时单列：t_final 是 finalize 返回时刻，t_audio_end 是 VAD 段尾。
        # 端点等待（静音判定）不计入这一项，避免把“等静音”和“解码收尾”混成一个数。
        self._record(
            "asr", "final",
            max(0, int(ev.t_final_ms) - int(ev.t_audio_end_ms)),
            audio_ms=max(0, int(ev.t_audio_end_ms) - int(ev.t_audio_start_ms)),
            chars=len(text),
        )
        samples = getattr(ev, "samples", None)
        if samples is not None and len(samples):
            import numpy as np
            data = np.asarray(samples, dtype=np.float32).reshape(-1)
            if data.size:
                peak = float(np.max(np.abs(data)))
                rms = float(np.sqrt(np.mean(np.square(data))))
                self._record(
                    "utterance", "audio", 0,
                    duration_ms=int(round(data.size * 1000 / 16000)),
                    peak_dbfs=round(20 * float(np.log10(max(peak, 1e-9))), 1),
                    rms_dbfs=round(20 * float(np.log10(max(rms, 1e-9))), 1),
                )
        if not text:
            await self._handle_empty_utterance()
            return
        self._empty_say_attempts = 0
        # 流程任务里跑完整问答（可能含数秒生成+播报），事件循环继续处理插话/停止
        self._start_flow(self._utterance_flow(text, samples=getattr(ev, "samples", None)))

    async def _handle_empty_utterance(self) -> None:
        """T02：没识别到文字时提示重说，最多重试 t02_max_retry_say 次。"""
        timers = self.cfg.timers
        self._empty_say_attempts += 1
        attempt = self._empty_say_attempts
        self._record("t02", "empty", 0, attempt=attempt)
        if attempt <= timers.t02_max_retry_say:
            self.state = InteractionState.SPEAKING
            await self._speak("没听清，你再说一遍？", kind="t02_retry")
            await self._enter_listening()
            return
        self._empty_say_attempts = 0
        await self._enter_idle()

    async def _utterance_flow(self, text: str, *, samples: Any = None) -> None:
        gen = self._qa_gen
        # 一回合一个 id：本回合内所有引用（control.intent、证据行）都用它，跨回合唯一。
        self._utterance_seq += 1
        self._utterance_id = f"u{self._utterance_seq}"
        intent = classify(text)
        self._record("intent", intent.kind.value, 0, text=intent.text[:80])
        identity = await self._verify_identity(samples)
        # 阶段 5：免唤醒（话题窗口内）先过声纹+发话意图准入，再交给正常意图路由
        if self._wakefree and self.identity is not None and identity is not None:
            decision = self.identity.decide_candidate(
                identity,
                text,
                directed=self._looks_directed(text),
                onset_ms=self._wakefree_onset_ms or (self._utterance_t0 or self.now_ms()),
                audio_end_ms=self._utterance_end_ms or self.now_ms(),
            )
            self._record(
                "wakefree", "decision", 0, action=decision.action, reason=decision.reason,
                status=identity.status.value, score=round(identity.score, 4),
            )
            if decision.action == "ignore":
                self._record("wakefree", "ignored", 0, reason=decision.reason)
                return
            if decision.action == "release":
                self._record("wakefree", "released", 0, reason=decision.reason)
                return
            if decision.action == "reentry":
                self._record("wakefree", "reentry", 0, reason=decision.reason)
                await self._speak_wakefree_prompt(
                    decision.prompt or "请再叫一次“小冰”，我确认一下是不是你。"
                )
                return
            if decision.action == "need_explicit_wake":
                self._record("wakefree", "need_explicit_wake", 0, reason=decision.reason)
                self.state = InteractionState.SPEAKING
                await self._speak("先叫一声“小冰”，我再接着听。", kind="wakefree_reject")
                await self._enter_idle()
                return
            self._record("wakefree", "accept", 0, reason=decision.reason)
        self._wakefree_onset_ms = None
        # 合同引导（control.intent）：必须先于 STOP/RESUME/REPEAT 判定，否则
        # “暂停引导/继续引导/再说一遍引导”会被本地控制词吃掉，永远到不了主控。
        # 否定/纠正由 classify_guidance_action 自己返回 None，这里不绕过它。
        guidance_action = classify_guidance_action(intent.text or intent.raw)
        if guidance_action is not None:
            await self._do_guidance(guidance_action, intent)
            return
        # 全局控制先于任务层：停止/清除/开关麦/紧急不能被“好了/拍/第一件”等词吞掉。
        if intent.kind is IntentKind.CLEAR:
            await self._do_clear()
            return
        if intent.kind is IntentKind.STOP and not self._looks_like_compare_abandon(text):
            await self._do_stop()
            return
        if intent.kind is IntentKind.MUTE:
            await self._do_mute()
            return
        if intent.kind is IntentKind.UNMUTE:
            self.set_microphone(True)
            self.state = InteractionState.LISTENING
            await self._enter_listening()
            return
        if intent.kind is IntentKind.CONTRACT:
            # 合同/紧急（"紧急" 词表命中，但 "紧急停止" 仍走 STOP）：
            # 设备侧动作只能由主控经 guidance 下发，本地能做到的是停播 + 说清边界。
            if self.contract is not None:
                self._record("contract", "no_local_action", 0, reason="device_side_only")
            else:
                self._record("contract", "not_linked", 0, reason="contract_disabled")
            await self._interrupt_speech(cancel_flow=False, reason="contract")
            self.state = InteractionState.SPEAKING
            await self._speak(
                "紧急停止我这边没有接到设备侧，现在只能停下播报。",
                kind="contract_unavailable",
            )
            await self._enter_listening()
            return

        task = self._last_compare_task
        if (task is not None and not getattr(task, "finished", False)
                and any(task.frames.get(p) is not None for p in ("A", "B"))
                and self._looks_like_compare_repair(text)):
            await self._do_reshoot(text, gen)
            return
        if self._reshoot_pending is not None and await self._handle_reshoot_pending(text, gen):
            return
        if self._compare_waiting():
            # 整句意愿仲裁：疑问/不确定/否定/引用优先于“好了/拍/准备”等词。
            action = self._compare_waiting_action(text)
            if action == "status":
                await self._do_compare_status()
                return
            if action == "clarify":
                await self._clarify_compare_waiting(text)
                return
            if action == "hold":
                self._record("compare", "hold", 0, task=task.task_id, text=text[:40])
                await self._ack_compare_hold()
                return
            if action == "recall":
                if await self._handle_compare_followup(text, gen, intent):
                    return
                self._record("compare", "recall_unavailable", 0, task=task.task_id)
                self.state = InteractionState.SPEAKING
                await self._speak(
                    "这一问我还没有可用的画面，等拍好再问我。",
                    kind="compare_recall_none",
                )
                await self._enter_listening()
                return
            if action == "ready":
                self._record("compare", "resume_ready", 0, task=task.task_id, text=text[:40])
                await self._do_resume(gen)
                return
        if self._last_compare_task is not None and await self._handle_compare_followup(text, gen, intent):
            return
        if self._looks_like_status_question(text):
            # 没有比较任务/比较已结束时，也直接按实际状态回答，不交模型猜。
            await self._do_compare_status()
            return
        if intent.kind is IntentKind.BACKCHANNEL:
            await self._do_backchannel()
            return
        if intent.kind is IntentKind.CORRECT:
            await self._do_correct(intent)
            return
        if intent.kind is IntentKind.REPEAT:
            await self._do_repeat(intent)
            return
        if intent.kind is IntentKind.RESUME:
            await self._do_resume(gen)
            return
        if intent.kind is IntentKind.COMPARE:
            if self.scene is None:
                self.state = InteractionState.SPEAKING
                await self._speak("这个能力我还没接上，先聊点别的吧。", kind="unsupported")
                await self._enter_listening()
                return
            await self._do_compare(intent, gen)
            return
        if intent.kind in _SCENE_KINDS:
            if self.scene is None:
                self.state = InteractionState.SPEAKING
                await self._speak("这个能力我还没接上，先聊点别的吧。", kind="unsupported")
                await self._enter_listening()
                return
            await self._do_scene(intent, gen)
            return
        if intent.kind not in (IntentKind.CHAT,):
            # 其余未接入的分类：诚实说明，不编造画面
            self.state = InteractionState.SPEAKING
            await self._speak("这个能力我还没接上，先聊点别的吧。", kind="unsupported")
            await self._enter_listening()
            return
        if self._looks_like_garbled_utterance(text):
            self._record("chat", "unclear", 0, text=text[:40])
            self.state = InteractionState.SPEAKING
            await self._speak("我没听清，能再说一遍吗？", kind="t02_retry")
            await self._enter_listening()
            return
        await self._do_chat(text, gen)

    async def _do_guidance(self, action: str, intent: Intent) -> None:
        """引导控制意图：交给主控（合同 §ControlIntent），本地只播回执。

        回执措辞与超时口径由 contract.guidance_intent 统一给出，这里不重写；
        未接合同时如实说明没连上，不假装已经执行；不新增任何移动/通行/空间动作。
        """
        if self.contract is None:
            self._record(
                "contract", "not_linked", 0,
                reason="contract_disabled", action=action, utterance_id=self._utterance_id,
            )
            self.state = InteractionState.SPEAKING
            await self._speak(
                "引导功能我这边没有接到主控，现在没法开始或暂停。",
                kind="control_unavailable",
            )
            await self._enter_listening()
            return
        outcome = await send_guidance_intent(
            self.contract,
            action=action,
            utterance_id=self._utterance_id,
            final_transcript=intent.raw,
        )
        self._record(
            "control", "intent", 0,
            action=action, accepted=outcome.accepted,
            operation_state=outcome.operation_state, reason_code=outcome.reason_code,
            utterance_id=self._utterance_id,
        )
        self.state = InteractionState.SPEAKING
        await self._speak(outcome.speech, kind="control")
        await self._enter_listening()

    async def _do_chat(self, text: str, gen: int) -> None:
        # 新问题开始：清掉上一轮答案的 run，旧残片不得串进本轮（第九章）
        self._reset_answer_run()
        if self.image_tasks is not None and self._looks_like_compare_repair(text):
            # 「重拍第二件 / 换一个再看」这类是**对已有双图任务的返工**：必须进重拍
            # 路径，不能掉进文本问答、也不能被"已有帧就不重拍"的复用逻辑绕过
            # （功能说明 L171/L267：明确要求重拍优先进入重拍路径）。
            await self._do_reshoot(text, gen)
            return
        if self.scene is not None and self._looks_like_fresh_capture_request(text):
            # 明确要求重拍/换新画面、但没有双图任务在跟：走单图取新图，
            # 不能被文本问答吞掉（上游词表里没有"重拍"，这里兜住）。
            self._record("chat", "fresh_capture_request", 0, text=text[:40])
            await self._do_scene(Intent(kind=IntentKind.LOOK, text=text, raw=text), gen)
            return
        if self.scene is not None and self._looks_like_visual_followup(text):
            # 「现在呢 / 再看看」是视觉追问：走场景链路（复用上一帧），不进纯文本问答
            self._record("chat", "visual_followup", 0, text=text[:40])
            await self._do_scene(Intent(kind=IntentKind.LOOK, text=text, raw=text), gen)
            return
        if self._stream_chat_available():
            await self._do_chat_streaming(text, gen)
            return
        self.state = InteractionState.THINKING
        self.frontend.set_listening(True)  # 允许用户此时改口/取消
        timers = self.cfg.timers
        t_start = self.now_ms()
        answer = None
        attempts = 0
        while attempts <= timers.t11_retry_max:
            if gen != self._qa_gen:
                # 代次已失效（用户插话/换问题）：不再为过时的一轮发起重试
                self._record("llm", "retry_abandoned", self.now_ms() - t_start, attempt=attempts)
                return
            attempts += 1
            self._lock_wait_ms = -1.0
            try:
                answer = await asyncio.wait_for(
                    asyncio.to_thread(self._ask_model, text, record=False),
                    timeout=timers.t11_generate_ms / 1000,
                )
                self._record("llm", "lock_wait", self._lock_wait_ms, attempt=attempts)
                break
            except asyncio.TimeoutError:
                self._record(
                    "t11",
                    "timeout",
                    self.now_ms() - t_start,
                    attempt=attempts,
                    lock_wait=self._lock_wait_ms,
                )
                if gen != self._qa_gen:
                    self._record("llm", "retry_abandoned", self.now_ms() - t_start, attempt=attempts)
                    return
                if attempts <= timers.t11_retry_max:
                    await asyncio.sleep(timers.t11_retry_delay_ms / 1000)
            except Exception as exc:  # noqa: BLE001 - 模型异常降级到固定话术
                log.exception("chat failed")
                self._record("llm", "error", self.now_ms() - t_start, error=repr(exc)[:200])
                break
        llm_ms = self.now_ms() - t_start
        if gen != self._qa_gen:
            # 生成期间用户插话/停止/换问题：旧结果必须丢弃，不能播报
            self._record(
                "llm",
                "stale_discarded",
                llm_ms,
                chars=len(answer.text) if answer is not None else 0,
            )
            return
        if answer is None or not (answer.text or "").strip():
            self._record("t11", "failed", llm_ms, attempt=attempts)
            self.state = InteractionState.SPEAKING
            await self._speak("我这边没想好，你再问一次？", kind="llm_fallback")
            await self._after_recovery_listen()
            return
        self._record("llm", "done", llm_ms, chars=len(answer.text))
        self.state = InteractionState.SPEAKING
        played = await self._speak(answer.text, kind="answer", ref_ms=ev_reference(self, t_start))
        self._record(
            "answer",
            "spoken",
            self.now_ms() - t_start,
            played=played,
            text=(answer.text or "")[:120],
        )
        if played and gen == self._qa_gen:
            # 历史只记"用户真正听到的回答"：提交晚于播报，且本轮播报完整、未被插话。
            # 生成完成即提交会让 TTS 失败/播报中被打断/整轮取消的答案也留在历史里，
            # 用户从未听全却成了下一问的上下文（超时被放弃那条路径由 record=False 挡住）。
            self.chat.record(
                text, answer.text, images=answer.images, t_ms=answer.t_end_ms
            )
        else:
            self._record(
                "answer",
                "not_committed",
                0,
                played=played,
                stale=gen != self._qa_gen,
                chars=len(answer.text or ""),
            )
        await self._after_answer_listen()

    def _stream_chat_available(self) -> bool:
        """真实 VLM 客户端支持流式时才走增量播报（FakeVlm 仍走整段路径）。"""
        return callable(getattr(self.chat, "ask_stream", None)) and callable(
            getattr(getattr(self.chat, "vlm", None), "complete_stream", None)
        )

    async def _do_chat_streaming(self, text: str, gen: int) -> None:
        """流式问答：第一句生成完就开播，压低热聊首声时延（阶段 7）。"""
        self.state = InteractionState.THINKING
        self.frontend.set_listening(True)  # 允许用户此时改口/取消
        timers = self.cfg.timers
        t_start = self.now_ms()
        ref_ms = ev_reference(self, t_start)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        stop = threading.Event()

        def _producer() -> None:
            try:
                for delta in self.chat.ask_stream(
                    text, timeout_s=timers.t11_generate_ms / 1000
                ):
                    if stop.is_set():
                        break
                    if delta:
                        loop.call_soon_threadsafe(queue.put_nowait, ("delta", str(delta)))
            except Exception as exc:  # noqa: BLE001 - 交主循环统一降级
                log.exception("stream chat failed")
                loop.call_soon_threadsafe(queue.put_nowait, ("error", repr(exc)[:200]))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("end", ""))

        threading.Thread(
            target=_producer, name="aieyes-chat-stream", daemon=True
        ).start()
        parts: list[str] = []
        buf = ""
        spoken = False
        played_all = True
        play_failed = False
        stream_pieces: list[str] = []

        async def _speak_phrase(phrase: str) -> bool:
            """流式逐句播报；同一回答的多句累积到同一个 answer run（第九章续播）。"""
            seg = split_text(phrase, self.cfg.tts.max_segment_chars)
            if not seg:
                return True
            offset = len(stream_pieces)
            first = offset == 0
            stream_pieces.extend(seg)
            self.state = InteractionState.SPEAKING
            return await self._speak(
                phrase,
                kind="answer",
                ref_ms=ref_ms if not spoken else None,
                answer_mode="new" if first else "append",
                answer_pieces=stream_pieces,
                answer_offset=offset,
            )

        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(
                        queue.get(), timeout=timers.t11_generate_ms / 1000
                    )
                except asyncio.TimeoutError:
                    self._record(
                        "t11", "timeout", self.now_ms() - t_start, scope="stream"
                    )
                    break
                if kind == "end":
                    break
                if kind == "error":
                    self._record(
                        "llm", "error", self.now_ms() - t_start, error=payload
                    )
                    break
                parts.append(payload)
                buf += payload
                while True:
                    piece, buf = _split_first_phrase(buf)
                    if piece is None:
                        break
                    if gen != self._qa_gen:
                        self._record(
                            "llm",
                            "stale_discarded",
                            self.now_ms() - t_start,
                            chars=len(piece),
                        )
                        return
                    self.state = InteractionState.SPEAKING
                    ok = await _speak_phrase(piece)
                    played_all = played_all and bool(ok)
                    spoken = True
                    if not ok:
                        # 片段失败即停播后续句子（与非流式 worker 内 break 对齐）
                        play_failed = True
                        stop.set()
                        break
                if play_failed:
                    break
            if play_failed and gen == self._qa_gen:
                # 播报中断时，「已生成但还没交给 _speak」的尾部也要登记进 run：
                # 流式一次性只把已开播的句子放进 run，用户之后说「接着说完」就取不到尾句，
                # 与非流式（一次性 split_text 出全部段）不一致。这里补登记，不改 index。
                tail_parts = [buf]
                buf = ""
                while True:
                    try:
                        tail_kind, tail_payload = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if tail_kind == "delta" and tail_payload:
                        tail_parts.append(str(tail_payload))
                self._extend_answer_run_tail("".join(tail_parts))
            if buf.strip() and gen == self._qa_gen and not play_failed:
                ok = await _speak_phrase(buf)
                played_all = played_all and bool(ok)
                spoken = True
        except asyncio.CancelledError:
            stop.set()
            raise
        finally:
            stop.set()
        llm_ms = self.now_ms() - t_start
        full = "".join(parts).strip()
        if gen != self._qa_gen:
            self._record("llm", "stale_discarded", llm_ms, chars=len(full))
            return
        if not full:
            self._record("t11", "failed", llm_ms)
            self.state = InteractionState.SPEAKING
            await self._speak("我这边没想好，你再问一次？", kind="llm_fallback")
            await self._after_recovery_listen()
            return
        self._record("llm", "done", llm_ms, chars=len(full), streamed=True)
        if played_all and gen == self._qa_gen:
            # 与非流式同一规则：只有每一段都播完、本轮未被插话，才把完整回答写进历史
            self.chat.record(text, full)
        else:
            self._record(
                "answer",
                "not_committed",
                0,
                played=played_all,
                stale=gen != self._qa_gen,
                chars=len(full),
                streamed=True,
            )
        self._record(
            "answer",
            "spoken",
            self.now_ms() - t_start,
            played=played_all,
            text=full[:120],
        )
        await self._after_answer_listen()

    async def _do_scene(self, intent: Intent, gen: int) -> None:
        """阶段 3：单图场景问答（取图 + OCR/VLM），答案以语音播报。"""
        scene = self.scene
        if scene is None:  # pragma: no cover - 调用点已保证非空
            return
        timers = self.cfg.timers
        self.state = InteractionState.THINKING
        self.frontend.set_listening(True)  # 允许用户此时改口/取消
        t_start = self.now_ms()
        question = intent.text or intent.raw
        hint = _SCENE_INTENT_HINT.get(intent.kind)
        source = self._scene_source()
        overview = self._is_scene_overview(question)
        reuse = self._last_scene_frame if self._should_reuse_scene_frame(question) else None
        reused_from = "scene" if reuse is not None else ""
        if reuse is None:
            # 上一轮/已释放位置留下的帧：按**历史图引用**回答（功能说明 L171/L267：
            # 已有可访问且适用照片时复用；`record_capture` 对 `source="history"`
            # 不扣次数、不启动 T08）。没有历史帧才老实取新图。
            history_frame = self._history_scene_frame(question)
            if history_frame is not None:
                reuse = history_frame
                reused_from = "history"
                source = self._scene_source(history=True)
        if reuse is None and not overview and self.image_tasks.target_budget_exhausted(source=source):
            # 单目标 3 次额度用尽：不再发取图请求，直接说明（功能说明 L360）
            self._record("scene", "budget_blocked", 0, source=source)
            self.state = InteractionState.SPEAKING
            await self._speak(
                "这个目标这一轮已经看了三次，换个东西或者让我从头来。",
                kind="capture_budget",
            )
            await self._enter_listening()
            return
        answer: Any = None
        try:
            kwargs: dict[str, Any] = {"intent_hint": hint}
            if reuse is not None:
                # 只有确实要复用时才传 frame=，避免给不支持的 scene 实现多塞参数
                kwargs["frame"] = reuse
            answer = await asyncio.wait_for(
                asyncio.to_thread(scene.answer, question, **kwargs),
                timeout=timers.t11_generate_ms / 1000,
            )
        except asyncio.TimeoutError:
            self._record("t11", "timeout", self.now_ms() - t_start, scope="scene")
        except Exception as exc:  # noqa: BLE001 - 视觉失败降级到固定话术
            log.exception("scene failed")
            self._record("scene", "error", self.now_ms() - t_start, error=repr(exc)[:200])
        scene_ms = self.now_ms() - t_start
        if gen != self._qa_gen:
            # 生成/取图期间用户插话或换问题：旧结果作废，不播报
            self._record("scene", "stale_discarded", scene_ms)
            return
        text = (getattr(answer, "text", "") or "").strip() if answer is not None else ""
        if not text:
            self._record("scene", "failed", scene_ms)
            self.state = InteractionState.SPEAKING
            await self._speak("我没看清画面，你再让我看一眼？", kind="scene_fallback")
            await self._after_recovery_listen()
            return
        self._last_scene_frame = getattr(self.scene, "last_frame", None)
        if reuse is None and not overview:
            # 真正发过取图请求才登记额度；复用缓存帧与整景观察不计数
            group = self.image_tasks.current_target_group(source=source)
            snapshot = self.image_tasks.record_capture(
                None, group, str(getattr(answer, "frame_id", "") or "")
            )
            self._record("scene", "capture_count", 0, **snapshot)
        elif reused_from == "history":
            # 历史图引用：仍留下证据引用，但不扣次数、不起 T08
            group = self.image_tasks.current_target_group(source="history")
            snapshot = self.image_tasks.record_capture(
                None, group, str(getattr(answer, "frame_id", "") or "")
            )
            self._record("scene", "history_ref", 0, **snapshot)
        self._record(
            "scene",
            "answer",
            scene_ms,
            intent=str(getattr(answer, "intent", "") or ""),
            frame_id=str(getattr(answer, "frame_id", "") or ""),
            reused=reuse is not None,
            reused_from=reused_from,
            source=source,
            position=self._evidence_position(reuse),
            degraded=",".join(getattr(answer, "degraded", ()) or ()),
            sources=",".join(str(getattr(tag, "kind", tag)) for tag in (getattr(answer, "source_tags", ()) or ())),
            chars=len(text),
        )
        self.state = InteractionState.SPEAKING
        self._last_answer_kind = "scene"
        self._last_answer_position = self._evidence_position(reuse)
        self._last_recall_question = question
        played = await self._speak(text, kind="answer", ref_ms=ev_reference(self, t_start))
        self._record(
            "answer",
            "spoken",
            self.now_ms() - t_start,
            played=played,
            text=text[:120],
        )
        await self._after_answer_listen()

    # ------------------------------------------------------------------
    # 阶段 4：双图比较与被打断任务的恢复
    # ------------------------------------------------------------------
    def _scene_source(self, *, history: bool = False) -> str:
        """取图来源标签。`history=True` 表示"用已有照片回答"，不占取图额度。"""
        if history:
            return "history"
        provider = getattr(self.scene, "provider", None)
        name = type(provider).__name__.lower() if provider is not None else ""
        return "file" if "file" in name else "camera"

    # 用户明确要求「重新看一张」时不得复用上一帧
    _SCENE_FRESH_WORDS = ("拍一下", "拍照", "新的", "重新看", "再来一张", "换一个", "换个")
    # 整景观察类提问：每次都要新画面，且不占单目标 3 次额度
    _FRESH_LOOK_RE = re.compile(
        r"(眼前|前面|周围|身边|四周|画面里|视野里|看到(?:什么|啥)|看见(?:什么|啥)|看看(?:有|是|前面|周围|眼前))"
    )
    # 时间性视觉追问：物体/场景可能已经变化，必须重新取图
    _TEMPORAL_LOOK_WORDS = ("现在呢", "再看看", "变了没")
    # 「现在呢 / 再看看」这类是视觉追问，不该掉进纯文本聊天
    _VISUAL_FOLLOWUP_WORDS = ("现在呢", "再看看", "还有呢", "变了没", "现在多少")

    def _is_scene_overview(self, question: str) -> bool:
        """整景观察（眼前/前面/周围有什么）：取新图且不占单目标额度。"""
        text = question or ""
        if any(word in text for word in self._SCENE_FRESH_WORDS):
            return False
        return bool(self._FRESH_LOOK_RE.search(text))

    def _needs_fresh_scene_frame(self, question: str) -> bool:
        text = question or ""
        return self._is_scene_overview(text) or any(
            word in text for word in self._TEMPORAL_LOOK_WORDS
        )

    def _should_reuse_scene_frame(self, question: str) -> bool:
        """同图追问：只有明确指向当前画面的细节问题才复用上一帧。"""
        if self._last_scene_frame is None:
            return False
        if self._needs_fresh_scene_frame(question):
            return False
        return not any(word in (question or "") for word in self._SCENE_FRESH_WORDS)

    def _looks_like_visual_followup(self, text: str) -> bool:
        return any(word in (text or "") for word in self._VISUAL_FOLLOWUP_WORDS)

    _COMPARE_READY_RE = re.compile(
        r"(继续|接着|接续|继|好了|放好|摆好|换好|换过|另一件|另一个|第二个放|"
        r"放上去|可以看|来看|看这个|看第二|准备|可以|拍|开始|第一件|第二件|第[一二]张)"
    )
    #: 等待拍摄时用户最自然的短确认。只在整句就是“好/好的/行/可以”时命中，
    #: 避免“好看”“不好”这类句子被当成准备好了。
    _COMPARE_CONFIRM_RE = re.compile(
        r"^(?:嗯+|啊+|哦+|噢+)?\s*[，,]?\s*(?:好|好的|好呀|好嘞|行|可以|没问题|ok|OK)"
        r"[吧了呀啊\s。！!，,]*$"
    )
    #: 明确表示“还没准备好/先别拍”。等待阶段命中后保持等待，不取图、不推进。
    _COMPARE_HOLD_RE = re.compile(
        r"(等一下|等一等|稍等|先等|还没|没好|没换好|没摆好|没放好|"
        r"先别|别拍|不要拍|不用拍|先不|还没摆正|先别急|不要继续|别继续)"
    )
    _COMPARE_PAUSE_RE = re.compile(
        r"(先放着|先放一放|先不比了|待会儿再比|一会儿再比|稍后再比|"
        r"先停一下|聊点别的|说点别的)"
    )
    _COMPARE_ABANDON_RE = re.compile(
        r"(算了|不比了|不比较了|别比了|取消(?:这个)?比较|结束(?:这个)?比较|"
        r"不比这个了|不用比了)"
    )
    _COMPARE_REFINE_RE = re.compile(
        r"(?:只|就|也|再|还|单独|重新).{0,4}(?:比|比较|看|算|加|考虑)"
        r"|(?:价格|容量|大小|颜色|日期|保质期|优惠|重量|尺寸).{0,4}(?:也|都|一起|算|加|比)"
        r"|(?:哪个更|哪个比较|更划算|更便宜|更贵)"
        r"|(?:它们|刚才那两).{0,8}(?:区别|差异|不一样|哪个|怎么样)"
    )
    _COMPARE_RECALL_RE = re.compile(
        r"(?:刚才|第一件|第二件|第一个|第二个|那两个|它们).{0,10}"
        r"(?:多少钱|价格|是什么|叫什么|什么样子|哪个)"
    )
    #: “不是第二个，是第一个 / 刚才说反了，左边才是第一件”：只改 A/B 归属，
    #: 不重拍、不丢其他证据。
    _COMPARE_SWAP_RE = re.compile(
        r"(?:说反|反了|搞反|弄反|左右反|对调|换过来|换个位置)"
        r"|(?:不是|并不是)(?:第一个|第二个|第一件|第二件|左边|右边|左面|右面)"
        r"[^。！？]{0,6}(?:是|应该是|才是)(?:第一个|第二个|第一件|第二件|左边|右边|左面|右面)"
    )
    #: 点到具体是哪一件（插问 / 澄清回答共用）。
    _POSITION_A_RE = re.compile(r"第一件|第一个|第一张|左边|左面|前者")
    _POSITION_B_RE = re.compile(r"第二件|第二个|第二张|右边|右面|后者")

    def _compare_waiting(self) -> bool:
        task = self._last_compare_task
        return (task is not None and not getattr(task, "finished", False)
                and getattr(task, "pause_reason", "") in (
                    "waiting_first", "waiting_second", "user_pause",
                    "capture_failed_first", "capture_failed_second",
                ))

    def _looks_like_compare_ready(self, text: str) -> bool:
        t = text or ""
        if re.search(
            r"(还没|没好|等一下|稍等|先等|别拍|不要拍|不用拍|先不|不好|不行|不拍|不可以|不能|"
            r"不要继续|别继续|不用继续|先别)",
            t,
        ) and not re.search(r"(好了|放好了|可以了|继续拍|开始拍|换好了|摆好了)", t):
            return False
        return bool(self._COMPARE_CONFIRM_RE.match(t) or self._COMPARE_READY_RE.search(t))

    def _looks_like_compare_hold(self, text: str) -> bool:
        t = text or ""
        # “先不比价格，看日期”是改比较项，不是“先别拍”这种保持等待。
        if re.search(r"(?:不比|不看|不算|去掉|删掉)(?:价格|日期|颜色|容量|大小|重量)", t):
            return False
        return bool(self._COMPARE_HOLD_RE.search(t))

    def _looks_like_compare_pause(self, text: str) -> bool:
        return bool(self._COMPARE_PAUSE_RE.search(text or ""))

    def _looks_like_compare_abandon(self, text: str) -> bool:
        t = text or ""
        if re.search(r"(不比价格|不比颜色|不比日期|不比大小|不比容量|不比重)", t):
            return False
        return bool(self._COMPARE_ABANDON_RE.search(t))

    def _looks_like_compare_refine(self, text: str) -> bool:
        t = text or ""
        if self._looks_like_compare_abandon(t):
            return False
        if re.search(r"(?:不比|不看|别比|不要比|不用比|不算|去掉|删掉)(?:价格|日期|颜色|容量|大小|重量)", t):
            # “不比价格了，看日期”是改比较项，不是放弃整个任务。
            return True
        if re.search(r"(不比|别比|不要比|先不比|不用比)", t):
            return False
        return bool(self._COMPARE_REFINE_RE.search(t))

    def _looks_like_compare_recall(self, text: str) -> bool:
        return bool(self._COMPARE_RECALL_RE.search(text or ""))

    def _looks_like_compare_swap(self, text: str) -> bool:
        return bool(self._COMPARE_SWAP_RE.search(text or ""))

    # -- 等待拍摄时的整句意愿仲裁 -------------------------------------
    # 不靠继续加白名单：先去掉引用内容，再按分句取“最后有效意愿”，
    # 疑问/不确定/过去完成态优先于“好了/拍/准备”等词。
    _QUOTED_RE = re.compile(r"[“\"『「][^”\"』」]{0,40}[”\"』」]")
    _STATUS_QUESTION_RE = re.compile(
        r"(现在到哪|到哪一步|进行到哪|你在等什么|在等什么|还要等多久|还要多久|"
        r"现在什么状态|什么进度|进度怎么样|现在在做什么|你在做什么)"
    )
    _UNCERTAIN_WAITING_RE = re.compile(
        r"(不知道|不确定|没想好|不清楚|说不准|准备好了没有|好了没有|"
        r"可以了吗|好了吗|行了吗|能拍了吗|可以拍了吗)"
    )
    _PAST_CAPTURE_RE = re.compile(r"(?:已经|刚才|刚刚).{0,4}(?:拍|拍好|拍了)")
    _READY_CLAUSE_RE = re.compile(
        r"(?:好了|好呀|好嘞|可以了|放好了|摆好了|换好了|准备好了|放上去|"
        r"现在是另一件|来看|看这个|可以看|可以拍|拍吧|开始吧|继续|接着)"
    )
    _HOLD_CLAUSE_RE = re.compile(
        r"(等一下|等一等|稍等|先等|还没|没好|没换好|没摆好|没放好|先别|"
        r"别拍|不要拍|不用拍|不可以拍|不能拍|不好|不行|不可以|不能|先不|先别急|不要继续|别继续|"
        r"不用继续|不是现在|不是让你|没有让你|没让你|没说(?:过)?(?:继续|拍)|我是说再等)"
    )
    _NOISY_READY_RE = re.compile(r"^(?:嗯+|啊+)?继")
    _PREMATURE_COMPARE_RE = re.compile(r"(?:哪个更|哪个贵|哪个便宜|有什么区别|差在哪|哪个划算)")
    _SHORT_READY_RE = re.compile(
        r"^(?:嗯+|啊+|哦+)?\s*(?:好|好的|好呀|好嘞|行|可以|没问题|ok|OK)"
        r"[吧了呀啊\s。！!]*$"
    )

    def _without_quoted(self, text: str) -> str:
        return self._QUOTED_RE.sub(" ", text or "")

    def _correction_target_position(self, text: str) -> str | None:
        """“是第一个/才是第一件/改成第二个”指向的目标位置。"""
        t = self._without_quoted(text)
        m = re.search(
            r"(?:是|应该是|才是|其实是|改成|换成)\s*(?:第一个|第一件|第一张|左边|左面)",
            t,
        )
        if m:
            return "A"
        m = re.search(
            r"(?:是|应该是|才是|其实是|改成|换成)\s*(?:第二个|第二件|第二张|右边|右面)",
            t,
        )
        if m:
            return "B"
        return None

    def _looks_like_status_question(self, text: str) -> bool:
        return bool(self._STATUS_QUESTION_RE.search(self._without_quoted(text)))

    def _compare_clause_kind(self, clause: str) -> str:
        c = (clause or "").strip()
        if not c:
            return "other"
        if self._HOLD_CLAUSE_RE.search(c):
            return "hold"
        if c.endswith(("?", "？")) or re.search(r"(?:吗|呢|么)\s*$", c):
            return "question"
        if self._SHORT_READY_RE.match(c) or self._READY_CLAUSE_RE.search(c) or self._NOISY_READY_RE.match(c):
            return "ready"
        return "other"

    def _compare_waiting_action(self, text: str) -> str:
        """等待拍摄时这句话最终要做什么：status/recall/clarify/hold/ready/none。"""
        t = self._without_quoted(text).strip()
        if not t:
            return "none"
        if self._looks_like_status_question(t):
            return "status"
        if self._looks_like_compare_recall(t):
            return "recall"
        if self._PREMATURE_COMPARE_RE.search(t):
            # “这两个哪个贵”是在问阶段，不是准备完成；先说明还在等哪一件。
            return "status"
        if self._PAST_CAPTURE_RE.search(t):
            return "clarify"
        if self._UNCERTAIN_WAITING_RE.search(t):
            return "clarify"
        clauses = [c for c in re.split(r"[，。！？!?；;、…~\n]+", t) if c.strip()]
        kinds = [self._compare_clause_kind(c) for c in clauses]
        relevant = [kind for kind in kinds if kind in ("ready", "hold", "question")]
        if not relevant:
            return "none"
        last = relevant[-1]
        return "clarify" if last == "question" else last

    async def _ack_compare_hold(self) -> None:
        """“先别拍”的短回执：第一次带用法，之后只说“我等你”，避免重复长篇提示。"""
        prompt = "好，我等你。准备好了说“好了”。" if self._compare_hold_acks == 0 else "好，我等你。"
        self._compare_hold_acks += 1
        self.state = InteractionState.SPEAKING
        await self._speak(prompt, kind="compare_hold")
        await self._enter_listening()

    async def _clarify_compare_waiting(self, text: str) -> None:
        """不确定时只问一个必要问题，不猜、不取图。"""
        task = self._last_compare_task
        reason = getattr(task, "pause_reason", "") if task is not None else ""
        self._record(
            "compare", "clarify", 0,
            task=getattr(task, "task_id", ""), reason=reason, text=text[:40],
        )
        self.state = InteractionState.SPEAKING
        if reason == "waiting_second":
            prompt = "你是说第二件已经放好了吗？放好了说“好了”，还没好就说“等一下”。"
        elif reason == "user_pause":
            prompt = "你是说可以继续了吗？可以就说“继续”或“好了”。"
        else:
            prompt = "你是说第一件已经放好了吗？放好了说“好了”，还没好就说“等一下”。"
        await self._speak(prompt, kind="compare_clarify")
        await self._enter_listening()

    async def _do_compare_status(self) -> None:
        """用实际任务状态回答“你在等什么/到哪一步了”，不交模型猜。"""
        task = self._last_compare_task
        if task is None:
            self._record("status", "query", 0, scope="listen")
            self.state = InteractionState.SPEAKING
            await self._speak("我在听你说。要比较两件东西，就说“对比这两个”。", kind="status")
            await self._enter_listening()
            return
        reason = str(getattr(task, "pause_reason", "") or "")
        if getattr(task, "finished", False):
            prompt = "刚才的比较已经结束。你可以说“继续”看之前的图，或者重新说“对比这两个”。"
        elif reason == "waiting_first":
            prompt = "我在等第一件。把第一件放到镜头前，放好说“好了”。"
        elif reason == "waiting_second":
            prompt = "第一件已经拍到了，我在等第二件。放好说“好了”。"
        elif reason == "user_pause":
            prompt = "比较先暂停着。已经拍到的还保留，说“继续”或“好了”接着来。"
        elif reason == "capture_failed_second":
            prompt = "第二件上次没拍到，第一件还保留着。准备好后说“好了”，或者“重拍第二件”。"
        elif reason == "capture_failed_first":
            prompt = "第一件上次没拍到。准备好后说“好了”再试。"
        else:
            prompt = "我正在看画面，请稍等。"
        self._record(
            "compare", "status", 0,
            task=task.task_id, reason=reason or str(getattr(task, "state", "")),
        )
        self.state = InteractionState.SPEAKING
        await self._speak(prompt, kind="compare_status")
        await self._enter_listening()

    def _compare_recent(self, task: Any, window_ms: int = 180_000) -> bool:
        """未完成任务随时可继续；已完成任务只在短时间内还当作“刚才那两件”。"""
        if not getattr(task, "finished", False):
            return True
        finished = int(getattr(task, "finished_ms", 0) or 0)
        return finished > 0 and (self.now_ms() - finished) <= window_ms

    def _merge_compare_question(self, task: Any, text: str) -> str:
        """把补充要求并进比较问题；只追加，不覆盖已有点到的维度。"""
        base = str(getattr(task, "question", "") or "").strip()
        addition = (text or "").strip()
        if not addition or addition in base:
            return base or "比较一下这两样东西"
        merged = f"{base}；{addition}" if base else addition
        return merged[:160]

    def _compare_focus_text(self, text: str) -> str:
        """补充比较项后的短回执：只说确实点到的东西，不复述整句。"""
        t = text or ""
        parts: list[str] = []
        if re.search(r"价格|价钱|多少钱|贵|便宜|划算", t):
            parts.append("价格")
        if re.search(r"日期|保质期|到期|新鲜", t):
            parts.append("日期")
        if re.search(r"颜色|什么色", t):
            parts.append("颜色")
        if re.search(r"容量|大小|重量|尺寸|规格", t):
            parts.append("规格")
        if parts:
            return "好，这次重点看" + "、".join(parts) + "。"
        return "好，我按你说的补充来比。"

    def _looks_like_garbled_utterance(self, text: str) -> bool:
        """明显夹杂口头填充/重复、又没有明确意图的短句，先请用户重说。"""
        t = (text or "").strip()
        if not t:
            return False
        if re.search(
            r"(有什么|是什么|多少钱|价格|保质期|日期|在哪|哪里|看看|读|念|找|"
            r"对比|比较|重拍|停止|继续|清空|静音|关麦|开麦)", t
        ):
            return False
        if re.search(r"对不对", t) and not re.search(r"不是|搞错|弄错|我说", t):
            return True
        return len(t) <= 8 and bool(re.search(r"(嗯|呃|啊|哦|那个|这个)", t))

    def _looks_like_fresh_capture_request(self, text: str) -> bool:
        """用户是否明确要一张**新画面**（重拍/换一个），而不是追问当前这帧。"""
        question = text or ""
        return any(word in question for word in self._RESHOOT_WORDS + self._SWAP_WORDS)

    def _drop_scene_frame(self) -> None:
        """取消/停止/换问题后丢掉缓存的画面，避免拿旧帧回答新问题。"""
        self._last_scene_frame = None
        self._frame_cache.clear()
        self._frame_cache_order.clear()
        reset = getattr(self.scene, "reset_last_frame", None)
        if callable(reset):
            reset()

    # 明确要求重拍：必须自重拍路径进入取图，不能被"已有帧"复用绕过（L209/L267）
    _RESHOOT_WORDS = (
        "重拍", "重新拍", "再拍一张", "再拍一次", "补拍", "重新照", "重新看那张",
        "重来一次", "重来一遍", "重新来一次", "重新来一遍",
    )
    # 换物：双图两件都换掉（L213：只替换某一件时走 _correct_position 的单位置释放）
    _SWAP_WORDS = ("换一个", "换个", "换一件", "换一物", "换新的")

    def _remember_frame(self, frame: Any) -> None:
        """记住帧对象（证据存储只留 id，历史图引用与重拍放弃的旧帧都要按 id 回取）。"""
        if frame is None:
            return
        frame_id = str(getattr(frame, "frame_id", "") or "")
        if not frame_id:
            return
        self._frame_cache[frame_id] = frame
        if frame_id in self._frame_cache_order:
            self._frame_cache_order.remove(frame_id)
        self._frame_cache_order.append(frame_id)
        while len(self._frame_cache_order) > 8:
            self._frame_cache.pop(self._frame_cache_order.pop(0), None)

    def _history_scene_frame(self, question: str) -> Any:
        """上一次任务/已释放位置留下的帧：拿来做历史图引用（不扣额度，L171/L267）。

        `latest_evidence()` 只返回**未释放**的当前证据；已释放位置（重拍/换物/任务
        结束时留下的旧照片）只在 `history_index()` 里，行上带 `released`。
        """
        if self._needs_fresh_scene_frame(question):
            return None
        if any(word in (question or "") for word in self._SCENE_FRESH_WORDS):
            return None
        latest = getattr(self.image_tasks, "latest_evidence", None)
        if callable(latest):
            try:
                evidence = latest(None)
            except TypeError:  # pragma: no cover - 兼容无参实现
                evidence = latest()
            if isinstance(evidence, dict):
                frame = self._frame_cache.get(str(evidence.get("frame_id") or ""))
                if frame is not None:
                    return frame
        history = getattr(self.image_tasks, "history_index", None)
        if callable(history):
            try:
                rows = history(limit=8)
            except TypeError:  # pragma: no cover - 兼容无参实现
                rows = history()
            for row in rows or ():
                if not isinstance(row, dict) or not row.get("released"):
                    continue
                frame = self._frame_cache.get(str(row.get("frame_id") or ""))
                if frame is not None:
                    return frame
        # 回落：用最近登记过的帧（取消/停止已清空缓存，不会拿被丢弃的旧图）
        for frame_id in reversed(self._frame_cache_order):
            frame = self._frame_cache.get(frame_id)
            if frame is not None:
                return frame
        return None

    def _evidence_position(self, frame: Any) -> str:
        """这一问用的是 A 还是 B（双图归属）；单目标记 S。"""
        if frame is None:
            return "S"
        task = self._last_compare_task
        if task is not None:
            for position in ("A", "B"):
                if task.frames.get(position) is frame:
                    return position
        frame_id = str(getattr(frame, "frame_id", "") or "")
        latest = getattr(self.image_tasks, "latest_evidence", None)
        if frame_id and callable(latest):
            try:
                evidence = latest(None)
            except TypeError:  # pragma: no cover - 兼容无参实现
                evidence = None
            if isinstance(evidence, dict) and str(evidence.get("frame_id") or "") == frame_id:
                return str(evidence.get("position") or "S")
        return "S"

    def _looks_like_compare_repair(self, text: str) -> bool:
        """这句话是不是对**已有双图任务**的返工请求（重拍 / 换物）。"""
        question = text or ""
        if re.search(r"(?:不要|别|不用|没有让你|没让你).{0,6}(?:重拍|重新拍|再拍|补拍|重来|换一个|换个|换一物)", question):
            return False
        wants_reshoot = any(word in question for word in self._RESHOOT_WORDS)
        wants_swap = any(word in question for word in self._SWAP_WORDS)
        if not (wants_reshoot or wants_swap):
            return False
        return self._open_compare_task() is not None or self._last_compare_task is not None

    def _open_compare_task(self) -> Any:
        task = self._last_compare_task
        if task is not None and not getattr(task, "finished", False):
            return task
        resume = getattr(self.image_tasks, "resume_target", None)
        if callable(resume):
            return resume()
        return None

    def _max_position_captures(self) -> int:
        return int(getattr(self.image_tasks, "max_captures_per_position", 3) or 3)

    async def _do_reshoot(self, text: str, gen: int) -> None:
        """重拍/换物：先让旧照片退出该位置，再按**原额度**重取，另一位置不动。

        功能说明 L209 / L213：明确重拍后旧照片不再作为该位置的当前证据，重拍失败
        也不能默默回退旧图；另一位置保留；位置次数不因重拍/换组重置（额度记在
        `CompareTask.position_captures`，`release_position` 不动它）。
        """
        question = text or ""
        task = self._open_compare_task()
        finished_task = task is None and self._last_compare_task is not None
        if task is None and self._last_compare_task is None:
            self._record("compare", "reshoot_none", 0, text=question[:40])
            self.state = InteractionState.SPEAKING
            await self._speak("现在没有要重拍的图，先让我看一眼再说。", kind="reshoot_none")
            await self._enter_listening()
            return
        if task is not None and not finished_task and self._looks_like_compare_refine(question):
            # “重拍第一件，然后只比较价格”：返工和补充要求在同一句里，一起生效。
            task.question = self._merge_compare_question(task, question)
            self._record("compare", "refine_in_repair", 0, task=task.task_id, question=task.question[:80])
        swap = any(word in question for word in self._SWAP_WORDS) and not any(
            word in question for word in self._RESHOOT_WORDS
        )
        if finished_task:
            old = self._last_compare_task
            reusable = (
                not swap
                and old is not None
                and any(old.frames.get(p) is not None for p in ("A", "B"))
            )
            if reusable:
                # 已结束的一轮：用户明确点到某一件时，只重拍那一件，另一件沿用；
                # 没点到哪件就复用两帧先问清位置（不重拍、不丢证据）。
                task = self._reopen_compare(old, question or old.question)
                if task is None:
                    self._record("compare", "capacity_blocked", 0)
                    self.state = InteractionState.SPEAKING
                    await self._speak(
                        "现在有两个没看完的看图任务，要先结束一个才能比新的。",
                        kind="capacity",
                    )
                    await self._enter_listening()
                    return
                self._record(
                    "compare", "reshoot_reuse_pair", 0,
                    task=task.task_id, old=old.task_id,
                    position=self._correct_position(question) or "",
                )
            else:
                # 换物（两件都换）或旧轮没有可用帧：开新一轮。
                self._record("compare", "reshoot_new_round", 0, task=old.task_id)
                try:
                    task = self.image_tasks.open_compare(question or "重拍")
                except CapacityError:
                    self._record("compare", "capacity_blocked", 0)
                    self.state = InteractionState.SPEAKING
                    await self._speak("现在有两个没看完的看图任务，要先结束一个才能比新的。", kind="capacity")
                    await self._enter_listening()
                    return
                self._last_compare_task = task
                await self._run_compare(task, gen, first_ready=True)
                return
        positions: list[str]
        if swap:
            positions = [position for position in ("A", "B") if task.frames.get(position) is not None]
        else:
            position = self._correct_position(question)
            if position is None:
                held = [p for p in ("A", "B") if task.frames.get(p) is not None]
                position = held[0] if len(held) == 1 else None
            if position is None:
                self._record("compare", "reshoot_clarify", 0, task=task.task_id)
                self._reshoot_pending = {"task_id": task.task_id, "question": question}
                self.state = InteractionState.SPEAKING
                await self._speak("你想重拍第一件还是第二件？", kind="reshoot_clarify")
                await self._enter_listening()
                return
            positions = [position]
        if not positions:
            self._record("compare", "reshoot_none", 0, task=task.task_id)
            self.state = InteractionState.SPEAKING
            await self._speak("现在没有要重拍的图，先让我看一眼再说。", kind="reshoot_none")
            await self._enter_listening()
            return
        limit = self._max_position_captures()
        exhausted = [
            position
            for position in positions
            if int(task.position_captures.get(position, 0) or 0) >= limit
        ]
        if exhausted:
            # 额度用尽不先 release：否则用户会看到"准备重拍→又没了"的白跑流程
            self._record(
                "compare", "budget_blocked", 0,
                task=task.task_id, position=",".join(exhausted), limit=limit,
            )
            self.state = InteractionState.SPEAKING
            await self._speak(
                "这个位置的取图次数已经用完了，换个东西或者让我从头来。",
                kind="capture_budget",
            )
            await self._enter_listening()
            return
        for position in positions:
            previous = task.frames.get(position)
            if previous is not None:
                self._remember_frame(previous)  # 放弃的旧帧留作历史引用，不冒充当前证据
            self.image_tasks.release_position(task, position)
            self._record(
                "compare", "reshoot", 0,
                task=task.task_id, position=position, swap=swap,
                remaining=max(0, limit - int(task.position_captures.get(position, 0) or 0)),
            )
        self._last_compare_task = task
        await self._run_compare(task, gen, first_ready=True)

    async def _acquire_frame(self, purpose: str = "vqa") -> Any:
        """按 T09 预算从场景提供者取一帧（相机/文件）。"""
        provider = getattr(self.scene, "provider", None)
        if provider is None:
            raise RuntimeError("没有取图提供者")
        return await asyncio.wait_for(
            asyncio.to_thread(provider.acquire, purpose),
            timeout=self.cfg.timers.t09_single_grab_ms / 1000,
        )

    async def _do_compare(self, intent: Intent, gen: int) -> None:
        question = intent.text or intent.raw or "比较一下这两样东西"
        self._compare_recall_pending = None
        self._compare_hold_acks = 0
        self._reshoot_pending = None
        self._compare_capture_failures.clear()
        try:
            task = self.image_tasks.open_compare(question)
        except CapacityError:
            self._record("compare", "capacity_blocked", 0)
            self.state = InteractionState.SPEAKING
            await self._speak("现在有两个没看完的看图任务，要先结束一个才能比新的。", kind="capacity")
            await self._enter_listening()
            return
        self._record("compare", "start", 0, task=task.task_id, question=question[:80])
        self._last_compare_task = task
        await self._run_compare(task, gen)

    async def _do_resume(self, gen: int) -> None:
        self._compare_hold_acks = 0
        if self.identity is not None and not self.identity.access_ok():
            self._record("materials", "resume_denied", 0, reason="identity_unverified")
            self.state = InteractionState.SPEAKING
            await self._speak("得先确认是你本人，才能接着看之前的图。", kind="resume_denied")
            await self._enter_listening()
            return
        task = self.image_tasks.resume_target()
        if task is None:
            self.state = InteractionState.SPEAKING
            await self._speak("现在没有要接着看的图。", kind="resume_none")
            await self._enter_listening()
            return
        self._record("compare", "resume", 0, task=task.task_id, state=task.state)
        await self._run_compare(task, gen, first_ready=True)

    async def _run_compare(self, task: Any, gen: int, *, first_ready: bool = False) -> None:
        timers = self.cfg.timers
        self.state = InteractionState.THINKING
        self.frontend.set_listening(True)
        t_start = self.now_ms()
        # 两个位置都已有帧 = 这一轮只用历史图（恢复/翻面追问），不占取图额度
        both_cached = all(task.frames.get(position) is not None for position in ("A", "B"))
        started_without_frames = not any(task.frames.get(p) is not None for p in ("A", "B"))
        if started_without_frames and not first_ready:
            # 新比较任务先让用户准备第一件，不能对着当前场景直接拍一张
            self.image_tasks.mark(task, "paused", reason="waiting_first")
            self._record("compare", "wait_first", 0, task=task.task_id)
            self.state = InteractionState.SPEAKING
            await self._speak(
                "先把第一件东西放到镜头前，准备好了说“好”“继续”或“拍”。",
                kind="compare_wait",
            )
            await self._enter_listening()
            return
        source = self._scene_source(history=both_cached)
        answer: Any = None
        capture_position = ""
        try:
            self.image_tasks.mark(task, "capturing")
            for position in ("A", "B"):
                if task.t08_expired or task.expired:
                    raise asyncio.TimeoutError
                group = task.group_a if position == "A" else task.group_b
                if group is None:
                    group = self.image_tasks.new_group(task, position=position, source=source)
                frame = task.frames.get(position)
                if frame is None:
                    capture_position = position
                    frame = await self._acquire_frame("vqa")
                    capture_position = ""
                    self._remember_frame(frame)
                    frame_id = str(getattr(frame, "frame_id", "") or "")
                    self._record(
                        "compare", "capture", self.now_ms() - t_start,
                        task=task.task_id, position=position, frame_id=frame_id,
                    )
                    snapshot = self.image_tasks.record_capture(task, group, frame_id)
                    self._record("compare", "capture_count", 0, **snapshot)
                    task.frames[position] = frame
                    self._compare_capture_failures.pop((task.task_id, position), None)
                    if (position == "A" and started_without_frames
                            and task.frames.get("B") is None):
                        # 第一件已拍：等用户把第二件放到镜头前并说“继续”，
                        # 不能在同一场景里连拍两张冒充双图比较。
                        self.image_tasks.mark(task, "paused", reason="waiting_second")
                        self._record("compare", "wait_second", self.now_ms() - t_start,
                                     task=task.task_id)
                        self.state = InteractionState.SPEAKING
                        await self._speak(
                            "第一件我看到了。请把第二件放到镜头前，说“好”或“继续”再拍。",
                            kind="compare_wait",
                        )
                        await self._enter_listening()
                        return
            capture_position = ""
            self.image_tasks.mark(task, "analyzing")
            answer = await asyncio.wait_for(
                asyncio.to_thread(self.scene.compare, task.frames["A"], task.frames["B"], task.question),
                timeout=timers.t11_generate_ms / 1000,
            )
        except asyncio.CancelledError:
            # 只把真正被打断的采集中状态标成 barge_in；已经在等用户准备
            # （waiting_first/waiting_second）或已被用户明确暂停的任务保留原阶段，
            # 否则插话后“好了/继续”就接不上原来的等待。
            if not task.finished and task.state in ("open", "capturing", "analyzing"):
                self.image_tasks.mark(task, "paused", reason="barge_in")
            self._record("compare", "paused", self.now_ms() - t_start, task=task.task_id)
            raise
        except asyncio.TimeoutError:
            if capture_position:
                await self._handle_compare_capture_failure(task, capture_position, None)
                return
            self._record("t11", "timeout", self.now_ms() - t_start, scope="compare")
            if not task.finished:
                self.image_tasks.mark(task, "paused", reason="timeout")
            self.state = InteractionState.SPEAKING
            await self._speak("这次比较没看完，你说“继续”我再接着看。", kind="compare_fallback")
            await self._after_recovery_listen()
            return
        except Exception as exc:  # noqa: BLE001 - 视觉异常降级到固定话术
            if capture_position:
                await self._handle_compare_capture_failure(task, capture_position, exc)
                return
            self._record("compare", "error", self.now_ms() - t_start, error=repr(exc)[:200])
            if not task.finished:
                self.image_tasks.close(task, reason="error")
            self.state = InteractionState.SPEAKING
            await self._speak("两张图我没能比出来，要不换个问题？", kind="compare_fallback")
            await self._after_recovery_listen()
            return
        compare_ms = self.now_ms() - t_start
        if gen != self._qa_gen:
            if not task.finished:
                self.image_tasks.mark(task, "paused", reason="stale")
            self._record("compare", "stale_discarded", compare_ms, task=task.task_id)
            return
        text = str(getattr(answer, "text", "") or "").strip()
        if not text:
            self.image_tasks.close(task, reason="empty")
            self._record("compare", "failed", compare_ms, task=task.task_id)
            self.state = InteractionState.SPEAKING
            await self._speak("两张图我没看清，没法比较。", kind="compare_fallback")
            await self._after_recovery_listen()
            return
        self.image_tasks.finish(
            task,
            answer_text=text,
            answered_items=getattr(answer, "common_items", ()) or (),
            missing_items=getattr(answer, "missing", ()) or (),
            a_frame_id=str(getattr(answer, "a_frame_id", "") or ""),
            b_frame_id=str(getattr(answer, "b_frame_id", "") or ""),
        )
        self._record(
            "compare",
            "answer",
            compare_ms,
            task=task.task_id,
            chars=len(text),
            common=",".join(getattr(answer, "common_items", ()) or ()),
            missing=",".join(getattr(answer, "missing", ()) or ()),
            a_frame=str(getattr(answer, "a_frame_id", "") or ""),
            b_frame=str(getattr(answer, "b_frame_id", "") or ""),
            snapshot=task.snapshot(),
        )
        self.state = InteractionState.SPEAKING
        self._last_answer_kind = "compare"
        self._last_answer_position = "AB"
        played = await self._speak(text, kind="answer", ref_ms=ev_reference(self, t_start))
        self._record("answer", "spoken", self.now_ms() - t_start, played=played, text=text[:120])
        await self._after_answer_listen()

    async def _handle_compare_capture_failure(
        self, task: Any, position: str, exc: Exception | None
    ) -> None:
        """拍摄失败：保留另一件，给一个当前证据支持的调整动作，且有界提示。"""
        label = "第一件" if position == "A" else "第二件"
        reason, guidance = self._capture_failure_guidance(exc)
        key = (task.task_id, position)
        count = self._compare_capture_failures.get(key, 0) + 1
        self._compare_capture_failures[key] = count
        if count >= 2:
            next_step = "已经两次没拍到了。你可以换一件、稍后再试，或者说不比了。"
        elif position == "B" and task.frames.get("A") is not None:
            next_step = "第一件还保留着，只需要重新看第二件。准备好后说“好了”或“重拍第二件”。"
        else:
            next_step = "准备好后说“好了”再试。"
        message = f"{label}这次没拍到，{reason}。{guidance}{next_step}"
        reason_code = "capture_failed_first" if position == "A" else "capture_failed_second"
        if not task.finished:
            self.image_tasks.mark(task, "paused", reason=reason_code)
        self._record(
            "compare", "capture_failed", 0,
            task=task.task_id, position=position, reason=reason_code,
            detail=(repr(exc)[:200] if exc is not None else "timeout"), count=count,
        )
        self.state = InteractionState.SPEAKING
        await self._speak(message, kind="compare_capture_failed")
        await self._enter_listening()

    @staticmethod
    def _capture_failure_guidance(exc: Exception | None) -> tuple[str, str]:
        """把质量标签映射成“一个当前能执行的调整动作”，不猜方向/距离。"""
        if exc is None:
            return "相机这次没在预想时间内出图", "先确认镜头没有被挡住，或者把物品放稳。"
        # 延迟导入：scene 包会拉起视觉相机模块，Windows 无 fcntl 时不应影响引擎导入。
        try:
            from ..scene.errors import FrameAcquisitionError, FrameQualityError
        except Exception:  # noqa: BLE001 - 导入失败按一般取图失败处理
            FrameAcquisitionError = FrameQualityError = ()  # type: ignore[assignment]
        if FrameQualityError and isinstance(exc, FrameQualityError):
            report = getattr(exc, "report", None)
            labels = tuple(getattr(report, "labels", ()) or ())
            if "dark" in labels:
                return "画面太暗", "把物品移到亮一点的地方，或让镜头别背光。"
            if "bright" in labels:
                return "画面太亮", "避开强光或反光，稍微换个角度。"
            if "blurry" in labels:
                return "画面有点糊", "把物品放稳，镜头和物品都别晃。"
            if "too_small" in labels:
                return "画面里目标太小", "把物品靠近镜头一些，占满画面中间。"
            if "too_large" in labels:
                return "画面里目标太大", "把物品稍微拿远一点，让整体都在画面里。"
            return "画面还不够清楚", "把物品放到画面中间，放稳一点。"
        if FrameAcquisitionError and isinstance(exc, FrameAcquisitionError):
            return "没有取得画面", "先确认镜头没有被挡住，再试一次。"
        return "这次没取得画面", "先确认镜头没有被挡住，再试一次。"

    # -- 双图任务的自然后续：准备/保持、补充、插问、纠正、暂停与放弃 --------
    async def _handle_compare_followup(
        self, text: str, gen: int, intent: Intent | None = None
    ) -> bool:
        """双图任务的后续话。只有确实属于当前任务时才消费，否则交回正常路由。

        “继续/好了/嗯”由当前阶段决定含义：等待拍摄时才是“准备好了”，
        其它时候仍走普通恢复/附和，不做全局同义词映射。
        """
        task = self._last_compare_task
        if task is None:
            return False
        kind = getattr(intent, "kind", None)
        abandon = self._looks_like_compare_abandon(text)
        if kind in (
            IntentKind.STOP,
            IntentKind.CLEAR,
            IntentKind.MUTE,
            IntentKind.UNMUTE,
            IntentKind.CONTRACT,
        ) and not abandon:
            return False

        # 插问指向不明（“刚才那个”）：上一轮已问“第一件还是第二件”，这里接短回答。
        pending = self._compare_recall_pending
        if pending is not None and pending.get("task_id") == getattr(task, "task_id", ""):
            position = self._spoken_position(text)
            if position is not None and len(text.strip()) <= 8:
                self._compare_recall_pending = None
                return await self._answer_compare_recall(
                    pending.get("question") or text, position, task, gen
                )
            self._compare_recall_pending = None

        if abandon:
            await self._abandon_compare(task, text)
            return True

        if self._compare_waiting() and self._looks_like_compare_hold(text):
            self._record("compare", "hold", 0, task=task.task_id, text=text[:40])
            await self._enter_listening()
            return True

        if self._looks_like_compare_swap(text):
            await self._correct_compare_order(text, gen, task)
            return True

        if self._looks_like_compare_recall(text):
            position = self._spoken_position(text)
            if position is not None:
                return await self._answer_compare_recall(text, position, task, gen)
            held = [p for p in ("A", "B") if task.frames.get(p) is not None]
            if len(held) == 1:
                return await self._answer_compare_recall(text, held[0], task, gen)
            if held:
                self._compare_recall_pending = {"task_id": task.task_id, "question": text}
                self._record("compare", "recall_clarify", 0, task=task.task_id, text=text[:40])
                self.state = InteractionState.SPEAKING
                await self._speak("你是问第一件还是第二件？", kind="compare_recall_clarify")
                await self._enter_listening()
                return True
            return False

        if self._looks_like_compare_pause(text):
            await self._pause_compare(task, text)
            return True

        if self._looks_like_compare_refine(text):
            await self._refine_compare(text, gen, task)
            return True
        return False

    def _spoken_position(self, text: str) -> str | None:
        t = text or ""
        has_a = bool(self._POSITION_A_RE.search(t))
        has_b = bool(self._POSITION_B_RE.search(t))
        if has_a and not has_b:
            return "A"
        if has_b and not has_a:
            return "B"
        return None

    def _recall_intent_hint(self, question: str) -> str:
        if re.search(r"多少钱|价格|价钱|几块|售价|打折", question or ""):
            return "price"
        if re.search(r"保质期|生产日期|到期|有效期|日期", question or ""):
            return "date"
        return "describe"

    async def _answer_compare_recall(
        self, question: str, position: str, task: Any, gen: int
    ) -> bool:
        """中途插问：用已拍到的这一件回答，任务阶段保持不变。"""
        if self.scene is None:
            return False
        label = "第一件" if position == "A" else "第二件"
        frame = task.frames.get(position)
        if frame is None:
            self._record("compare", "recall_no_frame", 0, task=task.task_id, position=position)
            self.state = InteractionState.SPEAKING
            await self._speak(f"{label}还没有图，等拍好我再告诉你。", kind="compare_recall_none")
            await self._enter_listening()
            return True
        timers = self.cfg.timers
        self.state = InteractionState.THINKING
        self.frontend.set_listening(True)
        t_start = self.now_ms()
        answer: Any = None
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(
                    self.scene.answer,
                    question,
                    frame=frame,
                    intent_hint=self._recall_intent_hint(question),
                ),
                timeout=timers.t11_generate_ms / 1000,
            )
        except asyncio.TimeoutError:
            self._record("t11", "timeout", self.now_ms() - t_start, scope="compare_recall")
        except Exception as exc:  # noqa: BLE001 - 插问失败不打断比较任务
            log.warning("compare recall failed: %r", exc)
            self._record(
                "compare", "recall_error", self.now_ms() - t_start, error=repr(exc)[:200]
            )
        elapsed = self.now_ms() - t_start
        if gen != self._qa_gen:
            self._record("compare", "recall_stale", elapsed, task=task.task_id)
            return True
        text = (getattr(answer, "text", "") or "").strip() if answer is not None else ""
        if not text:
            self._record("compare", "recall_failed", elapsed, task=task.task_id)
            self.state = InteractionState.SPEAKING
            await self._speak("这一问我没看清，等下再问我。", kind="compare_recall_fallback")
            await self._enter_listening()
            return True
        self._record(
            "compare", "recall", elapsed,
            task=task.task_id, position=position,
            frame_id=str(getattr(answer, "frame_id", "") or ""),
            chars=len(text), text=text[:80],
        )
        self.state = InteractionState.SPEAKING
        self._last_answer_kind = "recall"
        self._last_answer_position = position
        self._last_recall_question = question
        await self._speak(text, kind="answer", ref_ms=ev_reference(self, t_start))
        await self._after_answer_listen()
        return True

    async def _pause_compare(self, task: Any, text: str) -> None:
        """暂停（不是结束）：保留任务与证据，之后说“继续/好了”还能接上。"""
        if not getattr(task, "finished", False):
            try:
                self.image_tasks.mark(task, "paused", reason="user_pause")
            except Exception:  # noqa: BLE001 - 暂停失败不影响回执
                log.warning("pause compare failed", exc_info=True)
        self._record("compare", "pause", 0, task=task.task_id, text=text[:40])
        self.state = InteractionState.SPEAKING
        await self._speak("好，先放着。想聊别的直接说。", kind="compare_pause")
        await self._enter_listening()

    async def _abandon_compare(self, task: Any, text: str) -> None:
        """结束比较：释放任务，后续“继续”不再复活旧比较结果。"""
        if not getattr(task, "finished", False):
            try:
                self.image_tasks.close(task, reason="user_abandon")
            except Exception:  # noqa: BLE001 - 关闭失败仍要给出明确结束回执
                log.warning("abandon compare failed", exc_info=True)
        self._record("compare", "abandon", 0, task=task.task_id, text=text[:40])
        self._last_compare_task = None
        self._compare_recall_pending = None
        self._reshoot_pending = None
        self._compare_capture_failures.clear()
        self.state = InteractionState.SPEAKING
        await self._speak("好，不比了。", kind="compare_abandon")
        await self._after_answer_listen()

    def _reopen_compare(self, old: Any, question: str) -> Any | None:
        """用一个已完成任务的有效帧开新一轮（复用证据，不刷新位置额度）。"""
        try:
            task = self.image_tasks.open_compare(question)
        except CapacityError:
            return None
        for position in ("A", "B"):
            frame = old.frames.get(position)
            if frame is not None:
                task.frames[position] = frame
        task.position_captures.update(dict(getattr(old, "position_captures", {}) or {}))
        self._last_compare_task = task
        self._record(
            "compare", "reopen", 0,
            old=old.task_id, new=task.task_id, copied=len(task.frames),
        )
        return task

    def _swap_compare_frames(self, task: Any) -> bool:
        """只交换 A/B 归属；帧对象、额度与其他证据都不动。"""
        a = task.frames.get("A")
        b = task.frames.get("B")
        if a is None or b is None:
            return False
        task.frames["A"], task.frames["B"] = b, a
        if task.group_a is not None and task.group_b is not None:
            task.group_a.image_ids, task.group_b.image_ids = (
                task.group_b.image_ids, task.group_a.image_ids
            )
        if getattr(task, "a_frame_id", "") or getattr(task, "b_frame_id", ""):
            task.a_frame_id, task.b_frame_id = task.b_frame_id, task.a_frame_id
        return True

    async def _correct_compare_order(self, text: str, gen: int, task: Any) -> None:
        """纠正：先看上一轮问的是什么，再决定改指代、换归属还是先澄清。"""
        target = self._correction_target_position(text) or self._spoken_position(text)
        explicit_swap = bool(
            re.search(r"(?:说反|反了|搞反|弄反|左右反|对调|换过来|换个位置)", text or "")
        )
        # 上一轮问的是某一件：这是改指代，不是无条件交换 A/B。
        if (self._last_answer_kind == "recall"
                and self._last_answer_position in ("A", "B")
                and target is not None):
            question = self._last_recall_question or "这一件是什么"
            if target == self._last_answer_position:
                self._record(
                    "compare", "reference_same", 0,
                    task=task.task_id, position=target, text=text[:40],
                )
            else:
                self._record(
                    "compare", "reference_corrected", 0,
                    task=task.task_id, from_position=self._last_answer_position,
                    to_position=target, text=text[:40],
                )
            await self._answer_compare_recall(question, target, task, gen)
            return
        if not explicit_swap and self._last_answer_kind != "compare" and target is not None:
            # 没有上一轮比较归属证据：先问一句，不无条件交换。
            self._record("compare", "swap_clarify", 0, task=task.task_id, text=text[:40])
            self.state = InteractionState.SPEAKING
            await self._speak(
                "你是说刚才问的对象说错了，还是要把第一件和第二件对调？",
                kind="compare_swap_clarify",
            )
            await self._enter_listening()
            return
        if not self._compare_recent(task):
            self._record("compare", "swap_stale", 0, task=task.task_id)
            self.state = InteractionState.SPEAKING
            await self._speak("那两张图已经隔了一会儿了，重新拍两件我再比较。", kind="compare_swap_stale")
            await self._enter_listening()
            return
        if not all(task.frames.get(p) is not None for p in ("A", "B")):
            self._record("compare", "swap_incomplete", 0, task=task.task_id, text=text[:40])
            self.state = InteractionState.SPEAKING
            await self._speak(
                "好，我记下了。等两件都拍好，我按你说的把第一件和第二件对调。",
                kind="compare_swap_ack",
            )
            await self._enter_listening()
            return
        if getattr(task, "finished", False):
            task = self._reopen_compare(task, task.question)
            if task is None:
                self._record("compare", "capacity_blocked", 0)
                self.state = InteractionState.SPEAKING
                await self._speak("现在有两个没看完的看图任务，要先结束一个才能比新的。", kind="capacity")
                await self._enter_listening()
                return
        self._swap_compare_frames(task)
        self._record("compare", "swap", 0, task=task.task_id, text=text[:40])
        await self._run_compare(task, gen, first_ready=True)

    async def _handle_reshoot_pending(self, text: str, gen: int) -> bool:
        """答“你想重拍哪一件？”：只接受位置/纠正，不把疑问当成选择。"""
        pending = self._reshoot_pending
        task = self._last_compare_task
        if pending is None or task is None or pending.get("task_id") != task.task_id:
            self._reshoot_pending = None
            return False
        if self._looks_like_compare_abandon(text):
            self._reshoot_pending = None
            return False
        if self._looks_like_compare_recall(text):
            # 用户先插问：先回答，重拍待选保留，下一轮再说哪一件。
            return False
        target = self._correction_target_position(text) or self._spoken_position(text)
        if target is None:
            if self._looks_like_compare_hold(text):
                self._record("compare", "reshoot_pending_hold", 0, task=task.task_id)
                self.state = InteractionState.SPEAKING
                await self._speak("好，你想好了再说第一件还是第二件。", kind="reshoot_clarify_hold")
                await self._enter_listening()
                return True
            self._reshoot_pending = None
            return False
        self._reshoot_pending = None
        label = "第一件" if target == "A" else "第二件"
        self._record(
            "compare", "reshoot_pending_answer", 0,
            task=task.task_id, position=target, text=text[:40],
        )
        await self._do_reshoot(f"重拍{label}", gen)
        return True

    async def _refine_compare(self, text: str, gen: int, task: Any) -> None:
        """补充/调整比较项：复用已拍到的帧，缺第二件时只更新要求并继续等。"""
        question = self._merge_compare_question(task, text)
        finished = bool(getattr(task, "finished", False))
        if finished and not self._compare_recent(task):
            # 旧任务不复活：按新的比较要求处理，先请用户准备第一件。
            self._record("compare", "refine_new_round", 0, task=task.task_id)
            await self._do_compare(Intent(IntentKind.COMPARE, question, text), gen)
            return
        both = all(task.frames.get(p) is not None for p in ("A", "B"))
        if finished:
            task = self._reopen_compare(task, question)
            if task is None:
                self._record("compare", "capacity_blocked", 0)
                self.state = InteractionState.SPEAKING
                await self._speak("现在有两个没看完的看图任务，要先结束一个才能比新的。", kind="capacity")
                await self._enter_listening()
                return
            self._record("compare", "refine", 0, task=task.task_id, question=question[:80])
            await self._run_compare(task, gen, first_ready=True)
            return
        task.question = question
        if both:
            self._record("compare", "refine", 0, task=task.task_id, question=question[:80])
            await self._run_compare(task, gen, first_ready=True)
            return
        self._record("compare", "refine_wait", 0, task=task.task_id, question=question[:80])
        self.state = InteractionState.SPEAKING
        await self._speak(self._compare_focus_text(text), kind="compare_refine_ack")
        await self._enter_listening()

    # ------------------------------------------------------------------
    # 阶段 5：身份核验、免唤醒（话题窗口）与资料清除
    # ------------------------------------------------------------------
    _DIRECTED_HINTS = (
        "你", "我", "帮", "请问", "什么", "多少", "几", "能不能", "可以", "给我",
        "看看", "读", "比较", "是不是", "怎么", "哪", "告诉我", "一下",
    )

    async def _verify_identity(self, samples: Any) -> IdentityResult | None:
        """用本段 VAD 音频做本地声纹核验；无音频/未配置时返回 None（不阻塞主链路）。"""
        if self.identity is None or samples is None:
            return None
        try:
            result = await asyncio.to_thread(self.identity.verify, samples)
        except Exception:  # noqa: BLE001 - 声纹核验失败按不可用处理，不打断对话
            log.warning("speaker verify failed", exc_info=True)
            return None
        self._last_identity = result
        self.identity.note(result)
        self._record(
            "identity", "verify", 0,
            status=result.status.value, score=round(result.score, 4),
            reason=result.reason, profile=result.profile_version,
        )
        return result

    def _looks_directed(self, text: str) -> bool:
        """免唤醒候选是否“明确对系统发话”（问句/请求句/第二人称等）。"""
        cleaned = (text or "").strip()
        if not cleaned:
            return False
        if cleaned.endswith(("?", "？")):
            return True
        return any(hint in cleaned for hint in self._DIRECTED_HINTS)

    def _maybe_enter_wakefree(self) -> bool:
        """T21：正常问答结束后空闲退出 → 开 15 分钟话题窗口并进入免唤醒倾听。"""
        if self.identity is None or not self.identity.enrolled or self._muted or self._closed:
            return False
        now = self.now_ms()
        deadline = self.identity.open_topic(now, label="answer")
        self._wakefree = True
        self._wakefree_onset_ms = None
        self.state = InteractionState.LISTENING
        self._cancel_timers()
        self.frontend.set_listening(True)
        self.frontend.set_wake_enabled(False)
        self._after(max(1000, deadline - now), self._on_topic_expired)
        self._record("topic", "open", 0, window_ms=self.cfg.timers.t21_topic_window_ms)
        self._record("wakefree", "enter", 0)
        return True

    async def _on_topic_expired(self) -> None:
        if not self._wakefree:
            return
        self._record("topic", "expired", 0)
        await self._enter_idle()

    async def _on_stop_listen_timeout(self) -> None:
        """T18 停止后的倾听超时：回待机（不建话题窗口）。"""
        if self.state != InteractionState.LISTENING:
            return
        await self._enter_idle()

    async def _do_clear(self) -> None:
        """清除本轮对话与图片资料；身份未确认时拒绝（阶段 5 资料权限）。"""
        t0 = self.now_ms()
        if self.identity is not None and not self.identity.access_ok():
            self._record("materials", "clear_denied", 0, reason="identity_unverified")
            self.state = InteractionState.SPEAKING
            await self._speak("得先确认是你本人以后，我才能清掉之前的资料。", kind="clear_denied")
            await self._enter_listening()
            return
        turns = len(self.chat.turns)
        self.chat.reset()
        cleared = self.image_tasks.clear_all()
        self._last_compare_task = None
        self._compare_recall_pending = None
        self._reshoot_pending = None
        self._compare_capture_failures.clear()
        self._exit_wakefree("cleared")
        self._silence_now()
        self._drop_scene_frame()
        try:
            self.tts.clear_cache()
        except Exception:  # noqa: BLE001 - 清缓存失败不影响本轮清除结论
            log.warning("tts.clear_cache failed", exc_info=True)
        self._record("materials", "cleared", self.now_ms() - t0, turns=turns, **cleared)
        self.state = InteractionState.SPEAKING
        await self._speak("好，之前的对话和图片资料都清掉了。", kind="cleared")
        await self._enter_listening()

    async def _speak_wakefree_prompt(self, text: str) -> None:
        """身份不确定时的“再叫一次”提示：播完继续免唤醒倾听（15 秒重入窗口由身份层维护）。"""
        self.state = InteractionState.SPEAKING
        self.frontend.set_listening(False)
        await self._speak(text, kind="wakefree_prompt")
        if self._closed:
            return
        self.state = InteractionState.LISTENING
        self.frontend.set_listening(True)
        self.frontend.set_wake_enabled(False)

    # ------------------------------------------------------------------
    # 阶段 6：interaction.v0.1 合同快照与入站处理
    # ------------------------------------------------------------------
    def _publish_playback(
        self,
        state: str,
        *,
        kind: str,
        segment_index: int = 0,
        generation: int | None = None,
        start_basis: str | None = None,
        reason_code: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """按合同回放 playback.status；主控未连接时静默跳过（本地流程优先）。"""
        contract = self.contract
        if contract is None or self._loop is None or self._closed:
            return
        meta = meta or {}
        request_id = meta.get("source_request_id") or self._active_request_id
        if request_id is None and not meta.get("source_event_id"):
            # 本地语音回合没有主控请求号：用可识别的本地回合号作为来源，避免伪造主控请求。
            request_id = f"local-turn-{generation if generation is not None else 0}"
        payload: dict[str, Any] = {
            "state": state,
            "source_event_id": meta.get("source_event_id"),
            "source_request_id": request_id,
            "utterance_id": meta.get("utterance_id"),
            "segment_index": segment_index,
            "generation": generation,
            "priority": meta.get("priority"),
            "start_basis": start_basis,
            "reason_code": reason_code,
            "occurred_mono_ms": mono_ms(),
        }
        task = self._loop.create_task(contract.publish_playback(**payload))
        task.add_done_callback(_log_task_error)

    def _flush_pending_guidance(self) -> None:
        """普通播报结束后，把暂存的高优先级引导提示接着播（合同 12.4 简化实现）。"""
        payload = self._guidance_pending
        if payload is None or self._closed or self._loop is None:
            return
        self._guidance_pending = None
        task = self._loop.create_task(self._play_guidance(payload))
        task.add_done_callback(_log_task_error)

    def contract_capabilities(self) -> dict[str, Any]:
        """capabilities.result.data：如实列出已实现/未实现项与资源口径。"""
        services = {
            "kws": {"state": "ready", "version": "sherpa-onnx-2.0.1", "detail": "streaming zipformer"},
            "asr": {"state": "ready", "version": "sherpa-onnx-2.0.1", "detail": "streaming paraformer"},
            "tts": {"state": "ready" if self._greeting_audio is not None else "unknown", "version": "piper-1.8.0"},
            "vlm": {"state": "unavailable", "version": "llama.cpp-0.9.0-dev", "detail": "调用期确认"},
            "ocr": {"state": "unavailable", "version": "pp-ocrv6-small", "detail": "onnxruntime 1.31.1"},
        }
        capabilities = {
            "voice_wake": True,
            "voice_loop": True,
            "speaker_identity": self.identity is not None,
            "scene_compare": True,
            "ocr": True,
            "vlm": True,
            "contract": True,
            "frame_from_main_control": False,
            "scene_from_main_control": False,
            "control_intent": False,
        }
        unimplemented = [
            "frame.acquire 取主控帧",
            "scene.query 主控场景事实",
            "target.observation 目标关联",
            "control.intent 动作下发",
        ]
        return capabilities_result_data(
            capabilities=capabilities,
            unimplemented=unimplemented,
            resources={"memory": self._memory_snapshot(), "mode": "software"},
            service={
                "instance_id": self.cfg.contract.instance_id,
                "clock_id": CLOCK_ID,
                "backend": "local-models",
            },
        )

    def _memory_snapshot(self) -> dict[str, Any]:
        values: dict[str, int] = {}
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith(("VmRSS:", "VmHWM:")):
                        key, rest = line.split(":", 1)
                        values[key.strip()] = int(rest.strip().split()[0]) * 1024
        except OSError:
            pass
        return {
            "metric": "rss_bytes",
            "method": "proc_status",
            "sampling": "on_demand_proc_status",
            "bytes": values.get("VmRSS"),
            "peak_bytes": values.get("VmHWM"),
            "processes": ["interaction"],
            "sampled_mono_ms": mono_ms(),
        }

    def contract_status(self) -> dict[str, Any]:
        """interaction.status.data：轻量快照，未测项按合同写 null/unknown。"""
        mode_map = {
            InteractionState.IDLE: "idle",
            InteractionState.LISTENING: "listening",
            InteractionState.GREETING: "speaking",
            InteractionState.THINKING: "processing",
            InteractionState.SPEAKING: "speaking",
            InteractionState.CLOSED: "degraded",
        }
        listening = getattr(self.frontend, "listening", None)
        audio_input = {"state": "active" if listening else "idle"} if listening is not None else {"state": "unavailable"}
        audio_output = "speaking" if self.sink.is_playing else "idle"
        queues = {
            "image_tasks": len(self.image_tasks.tasks),
            "guidance_pending": 1 if self._guidance_pending else 0,
        }
        return interaction_status_data(
            mode=mode_map.get(self.state, "degraded"),
            active_request_id=self._active_request_id,
            audio_input_state=audio_input,
            audio_output_state={"state": audio_output},
            services={
                "kws": {"state": "ready", "version": "sherpa-onnx-2.0.1"},
                "asr": {"state": "ready", "version": "sherpa-onnx-2.0.1"},
                "tts": {"state": "ready" if self._greeting_audio is not None else "unknown", "version": "piper-1.8.0"},
                "vlm": {"state": "unavailable", "version": "llama.cpp-0.9.0-dev"},
                "ocr": {"state": "unavailable", "version": "pp-ocrv6-small"},
            },
            queues=queues,
            memory=self._memory_snapshot(),
            metrics_window={"window_ms": None, "samples": 0},
            last_error=None,
        )

    async def contract_cancel(self, request: Any) -> dict[str, Any]:
        """output.cancel：停声/清队列/停计算，分别如实回报。"""
        data = getattr(request, "data", None) or {}
        scope = str(data.get("scope") or "ordinary_turn")
        event_id = data.get("event_id") or data.get("source_event_id")
        reason = data.get("reason_code") or data.get("reason")
        stopped = cleared = ended = False
        basis = "not_applicable"
        if scope == "guidance_event":
            meta = self._playback_priority
            if meta is not None and (event_id is None or meta.get("source_event_id") == event_id):
                await self._interrupt_speech(reason="output_cancel")
                stopped = cleared = ended = True
                basis = "estimated"
        else:
            if self.sink.is_playing or self._speech_task is not None or self._flow_task is not None:
                await self._interrupt_speech(reason="output_cancel")
                stopped = cleared = ended = True
                basis = "estimated"
            if scope == "ordinary_session":
                self._guidance_pending = None
                cleared = True
        self._record(
            "contract",
            "cancel",
            0,
            scope=scope,
            device_stopped=stopped,
            queue_cleared=cleared,
            compute_ended=ended,
            reason=reason,
        )
        return cancel_result_data(
            device_stopped=stopped,
            queue_cleared=cleared,
            compute_ended=ended,
            reason_code=reason,
            basis=basis,
        )

    async def handle_guidance_event(self, event: Any) -> None:
        """guidance.event 入站：核验、去重、过期丢弃，紧急事件抢占普通播报。"""
        data = getattr(event, "data", None) or {}
        event_id = str(data.get("event_id") or "")
        if not event_id:
            self._record("guidance", "invalid", 0, reason="missing_event_id")
            return
        now = mono_ms()
        if event_id in self._guidance_seen:
            self._record("guidance", "duplicate", 0, event_id=event_id)
            return
        valid_until = data.get("valid_until_mono_ms")
        if isinstance(valid_until, int) and valid_until < now:
            self._record("guidance", "expired", 0, event_id=event_id, late_ms=now - valid_until)
            return
        clock_id = getattr(event, "clock_id", None)
        evidence_mode = getattr(event, "evidence_mode", None)
        if clock_id and evidence_mode == "live" and clock_id != CLOCK_ID:
            self._record("guidance", "clock_unverified", 0, event_id=event_id, clock_id=clock_id)
            return
        self._guidance_seen[event_id] = now
        while len(self._guidance_seen) > 64:
            self._guidance_seen.pop(next(iter(self._guidance_seen)), None)
        supersedes = data.get("supersedes_event_id")
        if supersedes:
            self._guidance_seen.pop(str(supersedes), None)
        priority = str(data.get("priority") or "information")
        raw_text = str(data.get("text") or "").strip()
        # 停步提示统一说“停下”：主控只发单字“停”时，播报端补成两字，避免单音节听不清。
        text = "停下。" if re.fullmatch(r"停[。！!，,；;：:\s]*", raw_text) else raw_text
        if not text:
            self._record("guidance", "empty", 0, event_id=event_id)
            return
        payload = {"source_event_id": event_id, "priority": priority, "text": text, "state_id": data.get("state_id")}
        self._record(
            "guidance",
            "event",
            0,
            event_id=event_id,
            priority=priority,
            chars=len(text),
            text_rewritten=bool(raw_text and text != raw_text),
        )
        if priority == "emergency":
            await self._preempt_for_guidance(payload)
        elif self.state in (InteractionState.IDLE, InteractionState.LISTENING) and not self.sink.is_playing:
            await self._play_guidance(payload)
        else:
            self._guidance_pending = payload
            self._record("guidance", "deferred", 0, event_id=event_id, priority=priority)

    async def _play_guidance(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        self._playback_priority = payload
        try:
            await self._speak(
                payload["text"],
                kind="guidance",
                meta={"source_event_id": payload.get("source_event_id"), "priority": payload.get("priority")},
            )
        finally:
            self._playback_priority = None
            self._record("guidance", "done", 0, event_id=payload.get("source_event_id"))

    async def _preempt_for_guidance(self, payload: dict[str, Any]) -> None:
        """真实有效紧急事件：抢占普通播报/生成，只用一个播放器，播完单一恢复出口。"""
        prev_state = self.state
        metrics: dict[str, float] = {}
        if prev_state in (InteractionState.SPEAKING, InteractionState.GREETING, InteractionState.THINKING) or self.sink.is_playing:
            metrics = await self._interrupt_speech(reason="guidance_emergency")
        self._record(
            "guidance",
            "preempt",
            0,
            preempted=prev_state.value,
            event_id=payload.get("source_event_id"),
            priority=payload.get("priority"),
            **metrics,
        )
        await self._play_guidance(payload)
        if self._closed:
            return
        self._record("guidance", "recovered", 0, preempted=prev_state.value)
        await self._enter_listening()

    def _ask_model(self, text: str, *, record: bool = True) -> Any:
        """线程内调用：先在本次生成预算内拿生成锁，拿不到就不再发真请求。

        被放弃的旧线程仍会算完（它拿的是 ``record=False``）；若它还没释放锁，
        本次重试线程只等待一个预算就放弃，避免排队时间超过用户耐心、
        也避免再白算一次模型调用。
        """
        budget = max(0.001, self.cfg.timers.t11_generate_ms / 1000)
        t_lock = mono_ms()
        got = self._llm_thread_lock.acquire(timeout=budget)
        self._lock_wait_ms = mono_ms() - t_lock
        if not got:
            return None
        try:
            return self.chat.ask(text, record=record)
        finally:
            self._llm_thread_lock.release()

    async def _after_answer_listen(self) -> None:
        """T17：答完后继续倾听 10s，无话则回待机。"""
        if self._closed:
            return
        self.state = InteractionState.LISTENING
        self.frontend.set_listening(True)
        self.frontend.set_wake_enabled(False)
        self._cancel_timers()
        self._after(self.cfg.timers.t17_post_answer_listen_ms, self._on_post_answer_timeout)
        self._record("t17", "listen", 0)

    async def _on_post_answer_timeout(self) -> None:
        if self.state != InteractionState.LISTENING:
            return
        # T21：正常答完且空闲退出才建立话题窗口（失败/取消/停止路径不进这里）
        if self._maybe_enter_wakefree():
            return
        await self._enter_idle()

    async def _after_recovery_listen(self) -> None:
        """T13：模型失败后的恢复倾听窗口（120s，90s 提醒），无话则回待机。"""
        if self._closed:
            return
        timers = self.cfg.timers
        self.state = InteractionState.LISTENING
        self.frontend.set_listening(True)
        self.frontend.set_wake_enabled(False)
        self._cancel_timers()
        self._extends = 0
        self._after(timers.t13_warn_ms, self._on_recovery_remind)
        self._after(timers.t13_resume_ms, self._on_recovery_timeout)
        self._record("t13", "listen", 0)

    async def _on_recovery_remind(self) -> None:
        if self.state is not InteractionState.LISTENING:
            return
        self._record("t13", "remind", 0)
        self.frontend.set_listening(False)
        await self._speak("我还在，你慢慢说。", kind="t13_remind")
        if self.state is InteractionState.LISTENING:
            self.frontend.set_listening(True)

    async def _on_recovery_timeout(self) -> None:
        if self.state != InteractionState.LISTENING:
            return
        self._record("t13", "timeout", 0)
        await self._enter_idle()

    async def _do_backchannel(self) -> None:
        """附和（嗯/好/对）：打断播报后只回一句“嗯，我在。”，不提交模型。"""
        if self._barge_pending:
            self._barge_pending = False
            self.state = InteractionState.SPEAKING
            await self._speak("嗯，我在。", kind="ack")
            self._record("ack", "spoken", 0)
            await self._after_answer_listen()
            return
        self._record("utterance", "backchannel", 0)
        await self._enter_listening()

    # 纠正句里的位置词 → 比较任务位置（只认左右/第几，不猜语义）
    _CORRECT_POSITIONS = (
        ("左边", "A"),
        ("左面", "A"),
        ("第一", "A"),
        ("右边", "B"),
        ("右面", "B"),
        ("第二", "B"),
    )

    def _correct_position(self, text: str) -> str | None:
        for word, pos in self._CORRECT_POSITIONS:
            if word in (text or ""):
                return pos
        return None

    async def _do_correct(self, intent: "Intent | None" = None) -> None:
        """纠正（不是这个/我说的是右边那件）：确认收到，并按位置重取那一件。

        归属映射（哪件算 A/B）需要比较任务持有 A/B 标签，引擎当前没有该状态，
        这里只做"按位置释放该位置旧帧，让下一轮重取"，不假装已经改完归属。
        """
        text = (getattr(intent, "text", "") or getattr(intent, "raw", "") or "") if intent else ""
        pos = self._correct_position(text)
        released: str | None = None
        task = getattr(self, "_last_compare_task", None)
        if pos is not None and task is not None:
            group = task.group_a if pos == "A" else task.group_b
            has_evidence = task.frames.get(pos) is not None or group is not None
            if has_evidence:
                try:
                    self.image_tasks.release_position(task, pos)
                    released = pos
                except Exception:  # noqa: BLE001 - 释放失败不影响确认收到
                    log.warning("release_position failed on correct", exc_info=True)
        self._record(
            "correct",
            "applied" if released else "ack_only",
            0,
            position=pos or "",
            released=released or "",
            text=text[:40],
        )
        self.state = InteractionState.SPEAKING
        await self._speak("好的，你说。", kind="correct_ack")
        await self._enter_listening()

    async def _do_mute(self) -> None:
        """用户关麦：停听、停唤醒，等主控/重启再开（阶段 6 接合同后可由主控开麦）。"""
        self.state = InteractionState.SPEAKING
        await self._speak("好，我先不听。", kind="mute_ack")
        self.set_microphone(False)

    def set_microphone(self, enabled: bool) -> None:
        """开/关麦克风：关麦时 VAD 与唤醒词都不工作，只保留紧急取消通道。"""
        self._muted = not enabled
        self.frontend.set_listening(enabled)
        self.frontend.set_wake_enabled(enabled)
        self._record("mic", "on" if enabled else "off", 0)

    async def _do_stop(self) -> None:
        t0 = self.now_ms()
        metrics = await self._interrupt_speech(cancel_flow=False, reason="stop")
        self._drop_scene_frame()
        self._compare_recall_pending = None
        self._reshoot_pending = None
        self._record("stop", "silenced", self.now_ms() - t0, **metrics)
        self.state = InteractionState.LISTENING
        self.frontend.set_listening(True)
        self.frontend.set_wake_enabled(False)
        self._cancel_timers()
        self._after(self.cfg.timers.t18_post_stop_listen_ms, self._on_stop_listen_timeout)
        self._record("t18", "listen", 0)

    def _begin_answer_run(
        self,
        text: str,
        mode: str,
        pieces: list[str],
        offset: int,
    ) -> int:
        """开始/接续一次回答的 run，返回该 run 的 token。

        ``mode``：``new`` 新回答（重置）；``append`` 流式同一回答的后续句
        （追加全文、分段列表由调用方累积）；``continue`` 续播既有 run 的未播部分。
        """
        if mode == "new":
            self._answer_run_token += 1
            self._answer_run_pieces = list(pieces)
            self._answer_run_index = -1
            self._answer_run_text = text
        else:
            self._answer_run_pieces = list(pieces)
            self._answer_run_index = max(-1, min(offset - 1, len(pieces) - 1))
            if mode == "append":
                self._answer_run_text += text
        return self._answer_run_token

    def _commit_answer_run_index(self, token: int, idx: int) -> None:
        """推进「已播完到第几段」；token 不符说明 run 已被新回答/取消替换。"""
        if token != self._answer_run_token:
            return
        self._answer_run_index = idx

    def _extend_answer_run_tail(self, text: str) -> None:
        """把「已生成但还没开播」的尾部登记进当前 run（不推进已播位置）。

        流式播报被打断时，后续句子还没交给 ``_speak``，run 里只有已开播的分段；
        不补登记的话「接着说完」取不到尾句。``index`` 不动，尾部天然是未播部分。
        """
        extra = (text or "").strip()
        if not extra:
            return
        pieces = split_text(extra, self.cfg.tts.max_segment_chars)
        if not pieces:
            return
        self._answer_run_pieces.extend(pieces)
        self._answer_run_text += extra

    def _reset_answer_run(self) -> None:
        """新问题开始前清空 run，避免旧答案残片串到新问题（第九章）。"""
        self._answer_run_token += 1
        self._answer_run_pieces = []
        self._answer_run_index = -1
        self._answer_run_text = ""

    def _unplayed_answer_text(self) -> str:
        """第九章 4：只有可信片段边界时，从没播完的那一段开头继续。"""
        pieces = self._answer_run_pieces
        if not pieces:
            return ""
        rest = pieces[self._answer_run_index + 1 :]
        return "".join(rest)

    async def _do_repeat(self, intent: Intent) -> None:
        remainder = self._unplayed_answer_text()
        self.state = InteractionState.SPEAKING
        if remainder:
            offset = self._answer_run_index + 1
            self._record("repeat", "resume_unplayed", 0, chars=len(remainder), offset=offset)
            played_notice = await self._speak(NOTICE_REPEAT_RESUME, kind="repeat_notice")
            if not played_notice:
                # 提示被取消/打断：用户已经在说话了，不能再接着往下播
                self._record("repeat", "resume_aborted", 0, reason="notice_interrupted")
                await self._after_answer_listen()
                return
            await self._speak(
                remainder,
                kind="answer",
                answer_mode="continue",
                answer_pieces=self._answer_run_pieces,
                answer_offset=offset,
            )
        else:
            # 流式中断的一轮不进历史（本轮之前已修），改用 run 全文复读，
            # 否则「再说一遍」会复读更早一轮的答案（内容错位）。
            last = self._answer_run_text or (self.chat.turns[-1].assistant if self.chat.turns else "")
            self._record("repeat", "reread", 0, chars=len(last), from_run=bool(self._answer_run_text))
            await self._speak(last or "刚才没有可重复的内容。", kind="repeat")
        await self._after_answer_listen()

    async def _on_cancel(self, ev: events.CancelRequested) -> None:
        self._record("cancel", "requested", 0, reason=ev.reason)
        metrics = await self._interrupt_speech(reason=ev.reason)
        self._record("cancel", "silenced", metrics["old_task_ms"], **metrics)
        if self.state in (InteractionState.SPEAKING, InteractionState.THINKING, InteractionState.GREETING):
            await self._enter_listening()

    async def _on_device_fault(self, ev: events.DeviceFault) -> None:
        self._record("fault", ev.device, 0, detail=ev.detail[:200])
        if ev.device not in ("audio_in", "audio_out", "tts", "audio"):
            return
        t0 = self.now_ms()
        ok = await self._recover_audio()
        self._record("t14", "recover", self.now_ms() - t0, ok=ok)
        if ok:
            if self.state in (InteractionState.SPEAKING, InteractionState.GREETING, InteractionState.THINKING):
                self.state = InteractionState.SPEAKING
                await self._speak("设备恢复了。", kind="device_recover")
                await self._after_answer_listen()
            else:
                await self._enter_listening()
            return
        self.state = InteractionState.SPEAKING
        await self._speak("音频设备出问题了，我先待机。", kind="device_fault")
        await self._enter_idle()

    async def _recover_audio(self) -> bool:
        """T14：重开采集/播放前端；超时或失败都返回 False（由调用方播报后待机）。"""
        try:
            stop = getattr(self.frontend, "stop", None)
            if callable(stop):
                stop()
        except Exception:  # noqa: BLE001 - 停旧前端失败不阻塞恢复尝试
            log.warning("frontend stop during recovery failed", exc_info=True)
        try:
            start = getattr(self.frontend, "start", None)
            if not callable(start):
                return False
            await asyncio.wait_for(
                asyncio.to_thread(start),
                timeout=self.cfg.timers.t14_device_recover_ms / 1000,
            )
            return True
        except Exception:  # noqa: BLE001 - 恢复失败按设备故障处理
            log.warning("audio device recovery failed", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # 计时器与证据
    # ------------------------------------------------------------------
    def _after(self, delay_ms: int, coro_fn: Callable[[], Any]) -> None:
        if self._loop is None or self._closed:
            return

        def fire() -> None:
            self._timers.discard(handle)  # type: ignore[name-defined]
            if self._closed:
                return
            task = asyncio.ensure_future(coro_fn())
            task.add_done_callback(_log_task_error)

        handle = self._loop.call_later(delay_ms / 1000, fire)
        self._timers.add(handle)

    def _cancel_timers(self) -> None:
        for handle in list(self._timers):
            handle.cancel()
        self._timers.clear()

    def _record(self, category: str, name: str, ms: float, **extra: Any) -> None:
        self.timings.add(f"{category}.{name}", ms, **extra)
        # 核心字段放最后，避免 extra 里的同名字段（如 speech kind）覆盖记录类型
        record = {**extra, "kind": "interaction", "category": category, "event": name, "ms": round(float(ms), 2)}
        if self.evidence is not None:
            self.evidence.write(record)
        else:
            log.info("interaction %s.%s %s", category, name, extra or "")


def _log_task_error(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error("timer task failed: %r", exc)


def ev_reference(engine: InteractionEngine, t_start: int) -> int:
    """回答首声参考点：用户说完的时间（VAD 语音结束）优先，否则模型开始时间。"""
    if engine._utterance_end_ms is not None:
        return engine._utterance_end_ms
    if engine._utterance_t0 is not None:
        return engine._utterance_t0
    return t_start
