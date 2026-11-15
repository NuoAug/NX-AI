"""热聊首声时延探针：真实 VLM（llama-server）+ 真实 PiperTts。

比较两条链路：
* 非流式：``ChatSession.ask`` 等整段生成完，再整段合成 → 首声 = 全文生成 + TTS；
* 流式：``ChatSession.ask_stream`` 收到第一句就合成 → 首声 = 首句生成 + TTS。

输出一行 JSON（阶段 7 证据），用法::

    python Tools/chat_latency_probe.py --config Config/target.json
"""

from __future__ import annotations

import argparse
import json
import time

from aieyes.chat import ChatSession
from aieyes.config import AppConfig
from aieyes.speech.tts import PiperTts
from aieyes.vision.vlm import VlmClient

_CLAUSE_ENDS = "。！？!?；;，,、：:\n"
_CLAUSE_MIN_CHARS = 6
_CLAUSE_MAX_CHARS = 24


def _first_phrase(buffer: str) -> str | None:
    for index, char in enumerate(buffer):
        if index + 1 >= _CLAUSE_MIN_CHARS and char in _CLAUSE_ENDS:
            return buffer[: index + 1].strip() or None
    if len(buffer) >= _CLAUSE_MAX_CHARS:
        return buffer[:_CLAUSE_MAX_CHARS].strip() or None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="热聊首声时延探针")
    parser.add_argument("--config", default="Config/target.json")
    parser.add_argument("--question", default="请告诉我前面有什么东西。")
    args = parser.parse_args()

    cfg = AppConfig.from_file(args.config)
    vlm = VlmClient(cfg.llm, cfg.vlm)
    chat = ChatSession(vlm)
    tts = PiperTts(cfg.tts)
    result: dict[str, object] = {"question": args.question}

    # 预热 TTS（排除首次加载模型的 4~5 秒）
    tts.prewarm(["我在"])
    tts.synth("我在")
    result["health"] = vlm.health()

    # 1) 非流式：整段生成 + 整段合成
    t0 = time.monotonic()
    answer = chat.ask(args.question)
    text_ms = (time.monotonic() - t0) * 1000
    t1 = time.monotonic()
    tts.synth(answer.text)
    synth_ms = (time.monotonic() - t1) * 1000
    result["non_stream"] = {
        "chars": len(answer.text),
        "text_ms": round(text_ms),
        "tts_ms": round(synth_ms),
        "first_audio_ms": round(text_ms + synth_ms),
        "text": answer.text[:80],
    }

    # 2) 流式：第一句生成完就合成
    t0 = time.monotonic()
    stream = chat.ask_stream(args.question, timeout_s=60)
    buffer = ""
    first_delta_ms = None
    sentence = None
    try:
        for delta in stream:
            if first_delta_ms is None:
                first_delta_ms = (time.monotonic() - t0) * 1000
            buffer += delta
            sentence = _first_phrase(buffer)
            if sentence:
                break
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    sentence_ms = (time.monotonic() - t0) * 1000 if sentence else None
    t1 = time.monotonic()
    tts.synth(sentence or buffer)
    synth_ms = (time.monotonic() - t1) * 1000
    result["stream"] = {
        "first_delta_ms": None if first_delta_ms is None else round(first_delta_ms),
        "first_sentence_ms": None if sentence_ms is None else round(sentence_ms),
        "tts_ms": round(synth_ms),
        "first_audio_ms": (
            None if sentence_ms is None else round(sentence_ms + synth_ms)
        ),
        "sentence": sentence or buffer[:80],
    }
    non = result["non_stream"]
    raw = result["stream"]
    if isinstance(non, dict) and isinstance(raw, dict) and raw.get("first_audio_ms"):
        result["speedup"] = round(
            float(non["first_audio_ms"]) / float(raw["first_audio_ms"]), 2
        )
    vlm.close()
    tts.close()
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
