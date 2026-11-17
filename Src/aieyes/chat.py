"""对话上下文与回答生成（Lead 维护）。

只依赖 ``interfaces.Vlm`` 协议（llama.cpp server 的 OpenAI 兼容接口由
``vision.vlm`` 实现），便于用 mock 单测，不直接在状态机里拼 HTTP。

设计取舍：当前 llama-server 以单条 user 消息工作最稳，历史轮次压成
“对话历史”文本块放进 prompt；图像仅在本次请求用 ``images`` 传入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

from .clock import mono_ms
from .errors import InferenceError
from .interfaces import Vlm

DEFAULT_SYSTEM_PROMPT = (
    "你是 AI_Eyes 视觉语言交互子系统的语音助手。用简洁口语中文回答，"
    "通常 1-3 句话；先给结论再给必要细节。看不到的画面、读不出的文字要直接说"
    "“我看不到/读不出来”，不要编造。"
)


@dataclass
class Turn:
    user: str
    assistant: str = ""
    t_ms: int = 0
    images: int = 0


@dataclass
class Answer:
    text: str
    t_start_ms: int
    t_end_ms: int
    prompt_chars: int = 0
    images: int = 0

    @property
    def latency_ms(self) -> int:
        return self.t_end_ms - self.t_start_ms


@dataclass
class ChatSession:
    vlm: Vlm
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_history_turns: int = 8
    max_tokens: int = 256
    timeout_s: float = 60.0
    turns: list[Turn] = field(default_factory=list)

    # -- 上下文 ---------------------------------------------------------
    def record(self, user: str, assistant: str, *, images: int = 0, t_ms: int | None = None) -> None:
        self.turns.append(Turn(user=user, assistant=assistant, t_ms=t_ms or mono_ms(), images=images))
        if len(self.turns) > self.max_history_turns:
            self.turns = self.turns[-self.max_history_turns :]

    def reset(self) -> None:
        self.turns.clear()

    def build_prompt(self, user_text: str) -> str:
        history = self.turns[-self.max_history_turns :]
        parts = [f"[系统] {self.system_prompt}"]
        if history:
            parts.append("[对话历史]")
            for turn in history:
                parts.append(f"用户：{turn.user}")
                if turn.assistant:
                    parts.append(f"助手：{turn.assistant}")
        parts.append(f"[本轮用户] {user_text}")
        parts.append("[要求] 只输出助手要说的中文口语回答，不要加角色前缀。")
        return "\n".join(parts)

    # -- 生成 -----------------------------------------------------------
    def ask(
        self,
        user_text: str,
        *,
        images: Sequence[np.ndarray] = (),
        timeout_s: float | None = None,
        temperature: float | None = None,
        record: bool = True,
    ) -> Answer:
        """生成一轮回答。

        ``record=False`` 供调用方自行按期次提交历史：超时被放弃的生成线程仍会
        跑完并返回，此时不能替用户留下一条从未播出的问答（见引擎 ``_do_chat``）。
        默认 ``True`` 保持 ``Testbench/debug.py`` 等直调用法的历史行为。
        """
        prompt = self.build_prompt(user_text)
        t0 = mono_ms()
        text = self.vlm.complete(
            prompt,
            images=list(images),
            max_tokens=self.max_tokens,
            timeout_s=self.timeout_s if timeout_s is None else timeout_s,
            temperature=temperature,
        )
        t1 = mono_ms()
        text = (text or "").strip()
        if record:
            self.record(user_text, text, images=len(images), t_ms=t1)
        return Answer(
            text=text,
            t_start_ms=t0,
            t_end_ms=t1,
            prompt_chars=len(prompt),
            images=len(images),
        )

    # -- 流式生成 -------------------------------------------------------
    def ask_stream(
        self,
        user_text: str,
        *,
        images: Sequence[np.ndarray] = (),
        timeout_s: float | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """流式生成：逐段 yield 增量文本（需 ``self.vlm`` 支持 ``complete_stream``）。

        与 :meth:`ask` 不同，本方法**不**记录本轮对话；调用方在流结束后用
        ``record(text, full)`` 落历史，这样被打断时不会留下半截回答。
        """
        stream = getattr(self.vlm, "complete_stream", None)
        if not callable(stream):
            raise InferenceError("当前 VLM 实现不支持流式输出")
        prompt = self.build_prompt(user_text)
        return stream(
            prompt,
            images=list(images),
            max_tokens=self.max_tokens,
            timeout_s=self.timeout_s if timeout_s is None else timeout_s,
            temperature=temperature,
        )

