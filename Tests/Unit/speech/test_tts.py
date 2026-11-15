"""TTS 单测：切段、合成“我在”、预合成缓存与并发线程安全。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from aieyes.interfaces import TtsAudio
from aieyes.speech.tts import split_text


def test_split_text_by_sentence():
    assert split_text("你好。今天天气怎么样？我们去散步吧！") == [
        "你好。",
        "今天天气怎么样？",
        "我们去散步吧！",
    ]


def test_split_text_short_text_single_segment():
    assert split_text("我在") == ["我在"]
    assert split_text("") == []


def test_split_text_long_text_limited():
    text = "，".join(["这是一个很长的子句"] * 20) + "。"
    segments = split_text(text, max_chars=30)
    assert len(segments) > 1
    assert all(len(seg) <= 30 for seg in segments), [len(s) for s in segments]
    assert "".join(segments) == text


def test_split_text_prefers_short_first_segment():
    text = "今天天气非常非常不错，我们一起出去散步吧顺便买点东西。第二句话在这里。"
    segments = split_text(text, max_chars=60)
    assert len(segments) >= 3
    assert segments[0].endswith("，")
    assert "".join(segments) == text


def test_synth_women(tts):
    audio = tts.synth("我在")
    assert isinstance(audio, TtsAudio)
    assert audio.pcm.dtype == np.float32
    assert audio.pcm.ndim == 1 and audio.pcm.size > 0
    assert audio.sample_rate == tts.sample_rate == 22050
    assert 0.0 < float(np.max(np.abs(audio.pcm))) <= 1.0
    assert 400 <= audio.duration_ms <= 1500, audio.duration_ms

    stats = dict(tts.last_stats)
    assert stats["total_ms"] > 0
    assert stats["first_chunk_ms"] <= stats["total_ms"] + 1.0
    rtf = stats["total_ms"] / max(1, audio.duration_ms)
    print(
        f"[tts] 我在 duration_ms={audio.duration_ms} samples={audio.pcm.size} "
        f"first_chunk_ms={stats['first_chunk_ms']} total_ms={stats['total_ms']} RTF={rtf:.2f}"
    )


def test_prewarm_and_concurrent_synth(tts):
    stats = tts.prewarm(("我在",))
    assert "我在" in stats
    assert stats["我在"]["duration_ms"] > 200

    texts = ["你好", "在吗", "你好", "在吗"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        audios = list(pool.map(tts.synth, texts))
    for audio in audios:
        assert audio.pcm.size > 0
        assert audio.sample_rate == tts.sample_rate
        assert audio.text in texts


def test_synth_cached_reuses_object(tts):
    first = tts.synth_cached("缓存测试")
    second = tts.synth_cached("缓存测试")
    assert first is second
    assert first.pcm.size > 0
