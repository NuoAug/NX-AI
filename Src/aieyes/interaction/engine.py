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
    ) -> bool:
        """播报一段文本；返回是否完整播完（被打断/失败返回 False）。"""
        if not (text or "").strip():
            return True
        self._speech_gen += 1
        gen = self._speech_gen
        task = asyncio.create_task(self._speech_worker(text, gen, kind, ref_ms, meta))
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
    ) -> bool:
        pieces = split_text(text, self.cfg.tts.max_segment_chars)
        prev_state = self.state
        self.state = InteractionState.SPEAKING if prev_state is not InteractionState.GREETING else prev_state
        self.frontend.set_wake_enabled(False)
        self._record("speech", "worker_start", self.now_ms() - (ref_ms or self.now_ms()), speech_kind=kind)
        played_all = True
        try:
            for idx, piece in enumerate(pieces):
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
                if idx == 0:
                    base = ref_ms if ref_ms is not None else (self._utterance_t0 or t_play)
                    self._record("latency", "first_audio", t_play - base, speech_kind=kind)
                self.bus.post(events.PlaybackStarted(piece_id, t_play, piece))
                finished = await asyncio.to_thread(handle.wait, None)
                self._publish_playback(
                    "completed" if finished else "cancelled",
                    kind=kind,
                    segment_index=idx,
                    generation=gen,
                    reason_code=None if finished else "interrupted",
                    meta=meta,
                )
                self.bus.post(
                    events.PlaybackFinished(
                        piece_id,
                        self.now_ms(),
                        int(handle.total_ms if finished else 0),
                        handle.total_ms,
                        cancelled=not finished,
                        text=piece,
                    )
                )
                if not finished:
                    played_all = False
                    break
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
        if intent.kind is IntentKind.CLEAR:
            await self._do_clear()
            return
        if intent.kind is IntentKind.STOP:
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
        if intent.kind is IntentKind.BACKCHANNEL:
            await self._do_backchannel()
            return
        if intent.kind is IntentKind.CORRECT:
            await self._do_correct()
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
            # 双图比较/合同能力尚未接入（阶段 4/6）：诚实说明，不编造画面
            self.state = InteractionState.SPEAKING
            await self._speak("这个能力我还没接上，先聊点别的吧。", kind="unsupported")
            await self._enter_listening()
            return
        await self._do_chat(text, gen)

    async def _do_chat(self, text: str, gen: int) -> None:
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
            attempts += 1
            try:
                answer = await asyncio.wait_for(
                    asyncio.to_thread(self._ask_model, text),
                    timeout=timers.t11_generate_ms / 1000,
                )
                break
            except asyncio.TimeoutError:
                self._record("t11", "timeout", self.now_ms() - t_start, attempt=attempts)
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
                    await self._speak(
                        piece, kind="answer", ref_ms=ref_ms if not spoken else None
                    )
                    spoken = True
            if buf.strip() and gen == self._qa_gen:
                self.state = InteractionState.SPEAKING
                await self._speak(
                    buf, kind="answer", ref_ms=ref_ms if not spoken else None
                )
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
        self.chat.record(text, full)
        self._record(
            "answer",
            "spoken",
            self.now_ms() - t_start,
            played=spoken,
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
        answer: Any = None
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(scene.answer, question, intent_hint=hint),
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
        self._record(
            "scene",
            "answer",
            scene_ms,
            intent=str(getattr(answer, "intent", "") or ""),
            frame_id=str(getattr(answer, "frame_id", "") or ""),
            degraded=",".join(getattr(answer, "degraded", ()) or ()),
            sources=",".join(str(getattr(tag, "kind", tag)) for tag in (getattr(answer, "source_tags", ()) or ())),
            chars=len(text),
        )
        self.state = InteractionState.SPEAKING
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
    def _scene_source(self) -> str:
        provider = getattr(self.scene, "provider", None)
        name = type(provider).__name__.lower() if provider is not None else ""
        return "file" if "file" in name else "camera"

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
        try:
            task = self.image_tasks.open_compare(question)
        except CapacityError:
            self._record("compare", "capacity_blocked", 0)
            self.state = InteractionState.SPEAKING
            await self._speak("现在有两个没看完的看图任务，要先结束一个才能比新的。", kind="capacity")
            await self._enter_listening()
            return
        self._record("compare", "start", 0, task=task.task_id, question=question[:80])
        await self._run_compare(task, gen)

    async def _do_resume(self, gen: int) -> None:
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
        await self._run_compare(task, gen)

    async def _run_compare(self, task: Any, gen: int) -> None:
        timers = self.cfg.timers
        self.state = InteractionState.THINKING
        self.frontend.set_listening(True)
        t_start = self.now_ms()
        source = self._scene_source()
        answer: Any = None
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
                    frame = await self._acquire_frame("vqa")
                    frame_id = str(getattr(frame, "frame_id", "") or "")
                    self._record(
                        "compare", "capture", self.now_ms() - t_start,
                        task=task.task_id, position=position, frame_id=frame_id,
                    )
                    snapshot = self.image_tasks.record_capture(task, group, frame_id)
                    self._record("compare", "capture_count", 0, **snapshot)
                    task.frames[position] = frame
            self.image_tasks.mark(task, "analyzing")
            answer = await asyncio.wait_for(
                asyncio.to_thread(self.scene.compare, task.frames["A"], task.frames["B"], task.question),
                timeout=timers.t11_generate_ms / 1000,
            )
        except asyncio.CancelledError:
            if not task.finished:
                self.image_tasks.mark(task, "paused", reason="barge_in")
            self._record("compare", "paused", self.now_ms() - t_start, task=task.task_id)
            raise
        except asyncio.TimeoutError:
            self._record("t11", "timeout", self.now_ms() - t_start, scope="compare")
            if not task.finished:
                self.image_tasks.mark(task, "paused", reason="timeout")
            self.state = InteractionState.SPEAKING
            await self._speak("这次比较没看完，你说“继续”我再接着看。", kind="compare_fallback")
            await self._after_recovery_listen()
            return
        except Exception as exc:  # noqa: BLE001 - 视觉异常降级到固定话术
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
        played = await self._speak(text, kind="answer", ref_ms=ev_reference(self, t_start))
        self._record("answer", "spoken", self.now_ms() - t_start, played=played, text=text[:120])
        await self._after_answer_listen()

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
        self._exit_wakefree("cleared")
        self._silence_now()
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
        text = str(data.get("text") or "").strip()
        if not text:
            self._record("guidance", "empty", 0, event_id=event_id)
            return
        payload = {"source_event_id": event_id, "priority": priority, "text": text, "state_id": data.get("state_id")}
        self._record("guidance", "event", 0, event_id=event_id, priority=priority, chars=len(text))
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

    def _ask_model(self, text: str) -> Any:
        with self._llm_thread_lock:
            return self.chat.ask(text)

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

    async def _do_correct(self) -> None:
        """纠正（不是这个/我说的是…）：阶段 2 先确认收到并重新倾听，语义纠正阶段 3+ 接入。"""
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
        self._record("stop", "silenced", self.now_ms() - t0, **metrics)
        self.state = InteractionState.LISTENING
        self.frontend.set_listening(True)
        self.frontend.set_wake_enabled(False)
        self._cancel_timers()
        self._after(self.cfg.timers.t18_post_stop_listen_ms, self._on_stop_listen_timeout)
        self._record("t18", "listen", 0)

    async def _do_repeat(self, intent: Intent) -> None:
        last = self.chat.turns[-1].assistant if self.chat.turns else ""
        self.state = InteractionState.SPEAKING
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
