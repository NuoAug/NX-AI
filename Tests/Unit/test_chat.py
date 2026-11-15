"""ChatSession 单测：用 FakeVlm 验证上下文拼接与记录，不发真实请求。"""

import numpy as np

from aieyes.chat import ChatSession


class FakeVlm:
    def __init__(self, reply: str = "好的。") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, prompt, *, images=(), max_tokens=512, timeout_s=120.0, temperature=None):
        self.calls.append(
            {
                "prompt": prompt,
                "images": len(images),
                "max_tokens": max_tokens,
                "timeout_s": timeout_s,
                "temperature": temperature,
            }
        )
        return self.reply


def test_ask_builds_prompt_and_records_turn() -> None:
    vlm = FakeVlm("今天多云，适合散步。")
    chat = ChatSession(vlm, max_tokens=128)
    answer = chat.ask("今天天气怎么样")
    assert answer.text == "今天多云，适合散步。"
    assert vlm.calls and "今天天气怎么样" in vlm.calls[0]["prompt"]
    assert vlm.calls[0]["max_tokens"] == 128
    assert len(chat.turns) == 1
    assert chat.turns[0].assistant == "今天多云，适合散步。"
    assert answer.prompt_chars > 0


def test_history_is_included_and_trimmed() -> None:
    vlm = FakeVlm("好的。")
    chat = ChatSession(vlm, max_history_turns=2)
    chat.ask("第一问")
    chat.ask("第二问")
    chat.ask("第三问")
    prompt = vlm.calls[-1]["prompt"]
    assert "第一问" in prompt and "第二问" in prompt  # 第三问提问时历史里还有前两轮
    chat.ask("第四问")
    prompt = vlm.calls[-1]["prompt"]
    assert "第一问" not in prompt  # 已被裁掉
    assert "第二问" in prompt and "第三问" in prompt
    assert len(chat.turns) == 2


def test_images_passed_through() -> None:
    vlm = FakeVlm("图里是一张桌子。")
    chat = ChatSession(vlm)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    answer = chat.ask("这是什么", images=[image])
    assert answer.images == 1
    assert vlm.calls[0]["images"] == 1


def test_reset_clears_history() -> None:
    chat = ChatSession(FakeVlm())
    chat.ask("你好")
    chat.reset()
    assert chat.turns == []
