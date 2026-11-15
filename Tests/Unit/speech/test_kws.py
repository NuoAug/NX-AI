"""KWS 单测：关键词生成 + 真实模型检测通路 + 合成唤醒音。"""

from __future__ import annotations

import pytest

from speech_helpers import FIXTURES, collect_hits, frames_from_wav

from aieyes.config import KwsConfig
from aieyes.errors import ConfigError
from aieyes.interfaces import AudioFrame
from aieyes.speech.kws import (
    SherpaKws,
    build_keywords_text,
    load_tokens,
    phrase_to_tokens,
)


def _tokens(models):
    return load_tokens(models.resolve(models.kws_dir) / "tokens.txt")


def test_phrase_to_tokens_uses_initial_and_final(models):
    tokens = _tokens(models)
    assert phrase_to_tokens("小冰", tokens) == ["x", "iǎo", "b", "īng"]
    assert phrase_to_tokens("你好军哥", tokens) == ["n", "ǐ", "h", "ǎo", "j", "ūn", "g", "ē"]


def test_phrase_to_tokens_ascii_form(models):
    tokens = _tokens(models)
    assert phrase_to_tokens("x iǎo b īng", tokens) == ["x", "iǎo", "b", "īng"]


def test_phrase_to_tokens_unknown_char_raises(models):
    with pytest.raises(ConfigError):
        phrase_to_tokens("龘", _tokens(models))


def test_build_keywords_text_keeps_valid_existing_lines(models):
    tokens = _tokens(models)
    stock = (models.resolve(models.kws_dir) / "keywords.txt").read_text(encoding="utf-8")
    text = build_keywords_text(tokens, ["小冰"], existing_lines=stock.splitlines())
    assert "x iǎo b īng @小冰" in text
    assert "n ǐ h ǎo j ūn g ē @你好军哥" in text


def test_default_keywords_file_generated(kws):
    # 触发一次加载，keywords 文件此时生成（不写模型目录）
    kws.accept(AudioFrame(samples=__import__("numpy").zeros(320, dtype="float32"), t_start_ms=0, seq=0))
    assert kws.ready is True
    path = kws.keywords_path
    assert path is not None and path.is_file()
    content = path.read_text(encoding="utf-8")
    assert "x iǎo b īng @小冰" in content
    assert str(path).startswith("/tmp/")  # 或系统临时目录
    kws.reset()


def test_real_wake_audio_detected(models):
    """模型自带 test_wavs/5.wav + test_keywords.txt → 应检出“周望军”。"""
    kws_dir = models.resolve(models.kws_dir)
    keywords = kws_dir / "test_wavs" / "test_keywords.txt"
    spotter = SherpaKws(models=models, keywords_file=str(keywords))
    try:
        hits = collect_hits(spotter, frames_from_wav(kws_dir / "test_wavs" / "5.wav"))
    finally:
        spotter.close()
    assert hits, "KWS 未在 5.wav 上检出任何关键词"
    assert any("周望军" in keyword for keyword, _ in hits), hits


def test_synthetic_xiaobing_detected(models):
    wav = FIXTURES / "wake" / "tts_xiaobing.wav"
    if not wav.is_file():
        pytest.skip("缺少合成唤醒音 fixture")
    spotter = SherpaKws(models=models, cfg=KwsConfig(wake_phrases=["小冰"]))
    try:
        hits = collect_hits(spotter, frames_from_wav(wav, pad_ms=800))
    finally:
        spotter.close()
    assert any("小冰" in keyword for keyword, _ in hits), hits
