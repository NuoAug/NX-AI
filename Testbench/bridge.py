"""观测包装：把真实 AI Eyes 组件的调用与事件转换为测试台事件流。

设计约束（见 Docs/测试台/执行方案.md §7）：

* ``ObservedBus.post`` 先投递原事件，再同步发出轻量副本；不消费原队列，不
  阻塞采集线程，不把音频 samples / 大图发到网页。
* 每轮对话用一个 ``turn_id`` 串起右气泡（ASR）与左气泡（模型/播报）。
* 生成结束不等于播报结束：``assistant.final`` 只表示文本定稿，播放状态由
  ``playback`` 事件单独表达。
* 取消后保留已显示文字并标记中断，迟到 delta 不再发事件。
* 时延用 ``time.monotonic`` 同一主机时基；真实声学首声始终为 ``None``。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

EVENT_TYPES = (
    "state",
    "asr.partial",
    "asr.final",
    "assistant.delta",
    "assistant.final",
    "playback",
    "frame",
    "timing",
    "log",
    "error",
)

#: ``emit(type, turn_id, payload)``；由 worker 负责补 run_id/seq/ts_mono_ms。
EmitFn = Callable[[str, "str | None", dict], None]


def mono_ms() -> int:
    return int(time.monotonic() * 1000)


@dataclass
class _Turn:
    turn_id: str
    user_text: str = ""
    user_final: bool = False
    assistant_text: str = ""
    assistant_done: bool = False
    cancelled: bool = False
    closed: bool = False
    chat_used: bool = False
    first_text_ms: int | None = None
    tts_ms: float = 0.0
    played_pieces: set[str] = field(default_factory=set)
    timings: dict = field(default_factory=dict)


class Bridge:
    """把真实组件的调用观测成测试台事件；线程安全。"""

    def __init__(
        self,
        emit: EmitFn,
        *,
        run_id: str = "",
        frame_store: Any = None,
        clock: Callable[[], int] = mono_ms,
    ) -> None:
        self._emit = emit
        self.run_id = run_id
        self._store = frame_store
        self._clock = clock
        self._lock = threading.RLock()
        self._turn_seq = 0
        self._turns: dict[str, _Turn] = {}
        self._current: _Turn | None = None

    # ------------------------------------------------------------------ 基础
    def _emit_event(self, etype: str, turn: _Turn | None, **payload: Any) -> None:
        try:
            self._emit(etype, turn.turn_id if turn else None, payload)
        except Exception:  # noqa: BLE001 - 观测失败不能影响业务链路
            pass

    @property
    def current_turn_id(self) -> str | None:
        with self._lock:
            return self._current.turn_id if self._current else None

    def _new_turn(self) -> _Turn:
        self._turn_seq += 1
        turn = _Turn(turn_id=f"t{self._turn_seq}")
        self._turns[turn.turn_id] = turn
        self._current = turn
        return turn

    def _ensure_turn(self) -> _Turn:
        cur = self._current
        if cur is None or cur.closed:
            return self._new_turn()
        # 只有播报、没有用户语音的 turn 不得吞掉下一条识别文本（例如问候语）。
        if cur.assistant_text and not cur.user_text and not cur.user_final:
            self._close(cur)
            return self._new_turn()
        return cur

    def _close(self, turn: _Turn, *, cancelled: bool = False) -> None:
        if turn.closed:
            return
        turn.closed = True
        turn.cancelled = turn.cancelled or cancelled
        if not turn.assistant_done and turn.assistant_text:
            turn.assistant_done = True
            self._emit_event(
                "assistant.final",
                turn,
                text=turn.assistant_text,
                cancelled=turn.cancelled,
                source="turn_close",
            )

    def _close_previous_user_turn(self) -> None:
        cur = self._current
        if cur is not None and not cur.closed and (cur.user_text or cur.user_final):
            self._close(cur)

    # ------------------------------------------------------------------ 事件入口
    def note_speech_start(self, t_ms: int) -> None:
        # 新语音开始说明上一轮用户气泡结束；不主动发事件，等 PartialText 建 turn。
        with self._lock:
            cur = self._current
            if cur is not None and cur.user_text and not cur.closed:
                self._close(cur)

    def note_partial(self, text: str, t_ms: int) -> None:
        with self._lock:
            turn = self._ensure_turn()
            if turn.user_final:
                return
            turn.user_text = text
            self._emit_event("asr.partial", turn, text=text)

    def note_utterance(self, text: str, t_ms: int, *, t_audio_end_ms: int | None = None,
                       t_final_ms: int | None = None, utterance_id: str = "") -> None:
        with self._lock:
            turn = self._ensure_turn()
            turn.user_text = text
            turn.user_final = True
            self._emit_event(
                "asr.final",
                turn,
                text=text,
                utterance_id=utterance_id or turn.turn_id,
            )
            if t_audio_end_ms is not None and t_final_ms is not None:
                asr_ms = max(0, int(t_final_ms) - int(t_audio_end_ms))
                turn.timings["asr_ms"] = asr_ms
                self._emit_event("timing", turn, asr_ms=asr_ms)

    def note_cancel(self, reason: str, t_ms: int) -> None:
        with self._lock:
            turn = self._current
            if turn is None or turn.closed:
                return
            turn.cancelled = True
            self._close(turn, cancelled=True)

    def note_playback_started(self, ev: Any) -> None:
        t_ms = int(getattr(ev, "t_ms", self._clock()))
        piece_id = str(getattr(ev, "piece_id", "") or "")
        text = str(getattr(ev, "text", "") or "")
        with self._lock:
            turn = self._current
            if turn is None or turn.closed:
                turn = self._new_turn()
            kind = "answer" if turn.chat_used else "prompt"
            shown = ""
            if piece_id and piece_id in turn.played_pieces:
                shown = ""
            elif turn.chat_used and turn.assistant_text:
                # 聊天文本已由 assistant.delta/final 发出，播放事件不再重复正文
                shown = ""
            elif text:
                shown = text
                turn.assistant_text += text
            if piece_id:
                turn.played_pieces.add(piece_id)
            if turn.first_text_ms is not None and "playback_wait_ms" not in turn.timings:
                wait = max(0, t_ms - int(turn.first_text_ms))
                turn.timings["playback_wait_ms"] = wait
                self._emit_event("timing", turn, playback_wait_ms=wait)
            self._emit_event(
                "playback",
                turn,
                state="started",
                piece_id=piece_id,
                text=shown,
                kind=kind,
            )

    def note_playback_finished(self, ev: Any) -> None:
        piece_id = str(getattr(ev, "piece_id", "") or "")
        cancelled = bool(getattr(ev, "cancelled", False))
        with self._lock:
            turn = self._current
            if turn is None:
                return
            if cancelled:
                turn.cancelled = True
            if turn.assistant_text:
                turn.assistant_done = True
            self._emit_event(
                "playback",
                turn,
                state="finished",
                piece_id=piece_id,
                played_ms=int(getattr(ev, "played_ms", 0) or 0),
                total_ms=int(getattr(ev, "total_ms", 0) or 0),
                cancelled=cancelled,
            )
            if turn.assistant_text:
                self._emit_event(
                    "assistant.final",
                    turn,
                    text=turn.assistant_text,
                    cancelled=turn.cancelled,
                    source="playback",
                )
            if turn.closed or cancelled:
                self._close(turn, cancelled=cancelled)

    # ------------------------------------------------------------------ 聊天
    def chat_begin(self) -> str:
        with self._lock:
            turn = self._ensure_turn()
            turn.chat_used = True
            return turn.turn_id

    def chat_text_ready(self, turn_id: str) -> None:
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is not None and turn.first_text_ms is None:
                turn.first_text_ms = self._clock()

    def chat_delta(self, turn_id: str, text: str) -> None:
        if not text:
            return
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None or turn.closed:
                return
            turn.assistant_text += text
            turn.first_text_ms = turn.first_text_ms or self._clock()
            self._emit_event("assistant.delta", turn, text=text)

    def chat_done(self, turn_id: str, text: str, *, generation_ms: float,
                  cancelled: bool = False) -> None:
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None:
                return
            if text:
                turn.assistant_text = text
            if turn.first_text_ms is None:
                turn.first_text_ms = self._clock()
            turn.assistant_done = True
            if cancelled:
                turn.cancelled = True
            turn.timings["generation_ms"] = max(0, int(generation_ms))
            self._emit_event(
                "assistant.final",
                turn,
                text=turn.assistant_text,
                cancelled=turn.cancelled,
            )
            self._emit_event(
                "timing", turn, generation_ms=max(0, int(generation_ms))
            )

    # ------------------------------------------------------------------ TTS/视觉
    def note_tts(self, duration_ms: float) -> None:
        with self._lock:
            turn = self._current
            if turn is None or turn.closed:
                return
            turn.tts_ms += max(0.0, float(duration_ms))
            if turn.first_text_ms is None:
                turn.first_text_ms = self._clock()
            turn.timings["tts_ms"] = int(turn.tts_ms)
            self._emit_event("timing", turn, tts_ms=int(turn.tts_ms))

    def note_vision(self, duration_ms: float) -> None:
        with self._lock:
            turn = self._current
            if turn is None:
                return
            turn.timings["vision_ms"] = max(0, int(duration_ms))
            self._emit_event("timing", turn, vision_ms=max(0, int(duration_ms)))

    def note_frame(self, frame: Any, *, purpose: str, slot: str,
                   saved: dict | None = None, emit: bool = True) -> None:
        payload = {
            "frame_id": str(getattr(frame, "frame_id", "") or ""),
            "width": int(getattr(frame, "width", 0) or 0),
            "height": int(getattr(frame, "height", 0) or 0),
            "t_capture_ms": int(getattr(frame, "t_capture_ms", 0) or 0),
            "purpose": str(purpose or ""),
            "slot": slot,
        }
        if saved:
            payload["file"] = saved.get("file")
        with self._lock:
            self._emit_event("frame", self._current, **payload)

    # ------------------------------------------------------------------ 日志/错误
    def log(self, level: str, module: str, message: str) -> None:
        with self._lock:
            self._emit_event(
                "log", self._current, level=str(level).upper(), module=module,
                message=message,
            )

    def note_evidence(self, record: dict) -> None:
        """把引擎 ``_record`` 证据映射为运行日志（不重复写原始证据）。"""
        if not isinstance(record, dict):
            return
        kind = str(record.get("kind") or "")
        category = str(record.get("category") or "")
        event = str(record.get("event") or "")
        if kind and kind not in ("interaction", "error", "state", "cancel"):
            return
        if not event:
            return
        ms = record.get("ms")
        bad = any(
            key in event
            for key in ("error", "failed", "timeout", "fault", "interrupted", "disconnect")
        )
        level = "WARN" if bad else "INFO"
        parts = [f"{category}.{event}" if category else event]
        if isinstance(ms, (int, float)):
            parts.append(f"{int(ms)}ms")
        extras = {
            k: v
            for k, v in record.items()
            if k
            not in ("kind", "category", "event", "ms", "ts_iso", "ts_mono_ms", "config")
            and isinstance(v, (str, int, float, bool))
        }
        for key, value in list(extras.items())[:6]:
            parts.append(f"{key}={value}")
        self.log(level, category or "engine", " ".join(parts))


    def error(self, code: str, message: str, *, retryable: bool = False) -> None:
        with self._lock:
            self._emit_event(
                "error", self._current, code=code, message=message,
                retryable=bool(retryable),
            )

    # ------------------------------------------------------------------ 收尾
    def finish(self, *, cancelled: bool = False) -> None:
        with self._lock:
            self._close(self._current, cancelled=cancelled)
            self._current = None

    def snapshot_turns(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "turn_id": t.turn_id,
                    "user": t.user_text,
                    "assistant": t.assistant_text,
                    "done": t.assistant_done,
                    "cancelled": t.cancelled,
                }
                for t in self._turns.values()
            ]

    # ------------------------------------------------------------------ 包装器
    def wrap_bus(self, bus: Any) -> "ObservedBus":
        return ObservedBus(bus, bridge=self)

    def wrap_chat(self, chat: Any) -> "ObservedChat":
        return ObservedChat(chat, bridge=self)

    def wrap_tts(self, tts: Any) -> "ObservedTts":
        return ObservedTts(tts, bridge=self)

    def wrap_evidence(self, evidence: Any) -> "ObservedEvidence":
        return ObservedEvidence(evidence, bridge=self)

    def wrap_provider(self, provider: Any) -> "ObservedProvider":
        return ObservedProvider(provider, bridge=self)

    def wrap_scene(self, scene: Any) -> "ObservedScene":
        return ObservedScene(scene, bridge=self)


# ---------------------------------------------------------------------------
# 包装器实现
# ---------------------------------------------------------------------------
class _Delegate:
    """除显式覆盖的方法/属性外全部透传。"""

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_inner":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_inner"), name, value)


class ObservedBus(_Delegate):
    def __init__(self, inner: Any, *, bridge: Bridge) -> None:
        super().__init__(inner)
        object.__setattr__(self, "_bridge", bridge)

    def post(self, ev: Any) -> Any:
        # 先投递原事件，保证引擎照常消费；再发轻量观测副本。
        result = self._inner.post(ev)
        try:
            self._dispatch(ev)
        except Exception:  # noqa: BLE001
            pass
        return result

    def _dispatch(self, ev: Any) -> None:
        from aieyes import events as evmod  # noqa: PLC0415 - 目标机运行时依赖

        b = self._bridge
        if isinstance(ev, evmod.PartialText):
            b.note_partial(ev.text, ev.t_ms)
        elif isinstance(ev, evmod.UtteranceRecognized):
            b.note_utterance(
                ev.text, ev.t_final_ms,
                t_audio_end_ms=getattr(ev, "t_audio_end_ms", None),
                t_final_ms=getattr(ev, "t_final_ms", None),
                utterance_id=f"{b.current_turn_id or 'u'}-{getattr(ev, 't_final_ms', 0)}",
            )
        elif isinstance(ev, evmod.SpeechStarted):
            b.note_speech_start(ev.t_ms)
        elif isinstance(ev, evmod.PlaybackStarted):
            b.note_playback_started(ev)
        elif isinstance(ev, evmod.PlaybackFinished):
            b.note_playback_finished(ev)
        elif isinstance(ev, evmod.CancelRequested):
            b.note_cancel(getattr(ev, "reason", ""), ev.t_ms)


class ObservedChat(_Delegate):
    def __init__(self, inner: Any, *, bridge: Bridge) -> None:
        super().__init__(inner)
        object.__setattr__(self, "_bridge", bridge)

    def ask(self, user_text: str, **kwargs: Any) -> Any:
        b = self._bridge
        turn_id = b.chat_begin()
        t0 = time.monotonic()
        try:
            answer = self._inner.ask(user_text, **kwargs)
        except BaseException:
            b.chat_done(turn_id, "", generation_ms=(time.monotonic() - t0) * 1000,
                        cancelled=True)
            raise
        text = str(getattr(answer, "text", "") or "")
        b.chat_done(turn_id, text, generation_ms=(time.monotonic() - t0) * 1000)
        return answer

    def ask_stream(self, user_text: str, **kwargs: Any) -> Iterator[str]:
        b = self._bridge
        turn_id = b.chat_begin()
        t0 = time.monotonic()
        first_ms: float | None = None
        parts: list[str] = []
        completed = False
        try:
            for delta in self._inner.ask_stream(user_text, **kwargs):
                if first_ms is None:
                    first_ms = (time.monotonic() - t0) * 1000
                    b.chat_text_ready(turn_id)
                text = str(delta or "")
                parts.append(text)
                b.chat_delta(turn_id, text)
                yield delta
            completed = True
        finally:
            full = "".join(parts).strip()
            gen_ms = first_ms if first_ms is not None else (time.monotonic() - t0) * 1000
            b.chat_done(turn_id, full, generation_ms=gen_ms, cancelled=not completed)


class ObservedTts(_Delegate):
    def __init__(self, inner: Any, *, bridge: Bridge) -> None:
        super().__init__(inner)
        object.__setattr__(self, "_bridge", bridge)

    def synth(self, text: str, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return self._inner.synth(text, *args, **kwargs)
        finally:
            self._bridge.note_tts((time.monotonic() - t0) * 1000)

    def synth_cached(self, text: str, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return self._inner.synth_cached(text, *args, **kwargs)
        finally:
            self._bridge.note_tts((time.monotonic() - t0) * 1000)


class ObservedEvidence(_Delegate):
    def __init__(self, inner: Any, *, bridge: Bridge) -> None:
        super().__init__(inner)
        object.__setattr__(self, "_bridge", bridge)

    def write(self, record: dict) -> Any:
        result = self._inner.write(record)
        try:
            self._bridge.note_evidence(record)
        except Exception:  # noqa: BLE001
            pass
        return result


class ObservedProvider(_Delegate):
    def __init__(self, inner: Any, *, bridge: Bridge) -> None:
        super().__init__(inner)
        object.__setattr__(self, "_bridge", bridge)

    def acquire(self, purpose: str = "vqa") -> Any:
        frame = self._inner.acquire(purpose)
        saved = None
        store = self._bridge._store
        if store is not None and frame is not None:
            try:
                saved = store.save(frame)
            except Exception as exc:  # noqa: BLE001
                self._bridge.log("WARN", "frame", f"保存输入帧失败: {exc}")
        self._bridge.note_frame(frame, purpose=purpose, slot="model", saved=saved)
        return frame


class ObservedScene(_Delegate):
    def __init__(self, inner: Any, *, bridge: Bridge) -> None:
        super().__init__(inner)
        object.__setattr__(self, "_bridge", bridge)

    def _save_pair(self, frame: Any, slot: str) -> dict | None:
        store = self._bridge._store
        if store is None or frame is None:
            return None
        try:
            saved = store.save(frame)
        except Exception:  # noqa: BLE001
            return None
        return saved

    def answer(self, question: str, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return self._inner.answer(question, **kwargs)
        finally:
            self._bridge.note_vision((time.monotonic() - t0) * 1000)

    def compare(self, a: Any, b: Any, question: str = "比较一下这两样东西", **kwargs: Any) -> Any:
        # 双图任务的 A/B 归属在比较调用时明确标注，避免把预览帧误当模型输入。
        for frame, slot in ((a, "A"), (b, "B")):
            if not hasattr(frame, "rgb"):
                continue
            self._bridge.note_frame(
                frame, purpose="vqa", slot=slot, saved=self._save_pair(frame, slot)
            )
        t0 = time.monotonic()
        try:
            return self._inner.compare(a, b, question, **kwargs)
        finally:
            self._bridge.note_vision((time.monotonic() - t0) * 1000)

    def read_text_result(self, frame: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return self._inner.read_text_result(frame, **kwargs)
        finally:
            self._bridge.note_vision((time.monotonic() - t0) * 1000)

    def read_text(self, frame: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return self._inner.read_text(frame, **kwargs)
        finally:
            self._bridge.note_vision((time.monotonic() - t0) * 1000)
