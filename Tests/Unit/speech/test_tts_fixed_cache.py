"""固定提醒离线缓存与响度对齐的定向用例（不联网、不加载 Piper 模型）。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from aieyes.audio.wav import write_wav
from aieyes.config import TtsConfig
from aieyes.errors import ModelUnavailable
from aieyes.speech.tts import PiperTts, normalize_pcm

REPO = Path(__file__).resolve().parents[3]


def _first_existing(*rels: str) -> Path:
    for rel in rels:
        path = REPO / rel
        if path.is_file():
            return path
    return REPO / rels[0]


CATALOG = _first_existing(
    "NX-AI/Models/tts-fixed/xiaoxiao/fixed_prompts.json",
    "models/tts-fixed/xiaoxiao/fixed_prompts.json",
)


def _tone(ms: int = 300, freq: float = 440.0, amp: float = 0.1) -> np.ndarray:
    n = int(22050 * ms / 1000)
    t = np.arange(n, dtype=np.float64) / 22050
    return (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def _make_cache(root: Path, texts: list[str], *, voice: str = "zh-CN-XiaoxiaoNeural") -> Path:
    entries = {}
    for i, text in enumerate(texts):
        name = f"p{i}.wav"
        path = root / name
        write_wav(path, _tone(), 22050)
        entries[text] = {
            "file": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sample_rate": 22050,
            "duration_ms": 300,
        }
    (root / "index.json").write_text(
        json.dumps({"voice": voice, "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def test_fixed_prompt_is_served_from_offline_cache_without_piper(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path, ["停下。"])
    tts = PiperTts(
        TtsConfig(fixed_cache_dir=str(cache), fixed_voice_id="zh-CN-XiaoxiaoNeural"),
        model_dir=tmp_path / "_no_piper_model_",
    )
    audio = tts.synth("停下。")
    assert audio.text == "停下。"
    assert audio.sample_rate == 22050
    assert audio.duration_ms == 300
    stats = tts.last_stats
    assert stats["source"] == "fixed_cache" and stats["cache_hit"] is True
    assert stats["total_ms"] == 0.0
    assert tts.fixed_cache_info()["entries"] == 1


def test_unknown_text_still_falls_back_to_piper(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path, ["停下。"])
    tts = PiperTts(
        TtsConfig(fixed_cache_dir=str(cache)),
        model_dir=tmp_path / "_no_piper_model_",
    )
    with pytest.raises(ModelUnavailable):
        tts.synth("今天天气怎么样")


def test_missing_cache_is_reported_not_fatal(tmp_path: Path) -> None:
    tts = PiperTts(TtsConfig(fixed_cache_dir=str(tmp_path / "missing")))
    info = tts.fixed_cache_info()
    assert info["enabled"] is True
    assert info["entries"] == 0
    assert "error" in info


def test_voice_mismatch_disables_the_cache(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path, ["停下。"], voice="zh-CN-YunxiNeural")
    tts = PiperTts(
        TtsConfig(fixed_cache_dir=str(cache), fixed_voice_id="zh-CN-XiaoxiaoNeural")
    )
    info = tts.fixed_cache_info()
    assert info["voice_mismatch"] is True
    assert info["entries"] == 0


def test_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path, ["停下。"])
    index = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    index["entries"]["停下。"]["sha256"] = "0" * 64
    (cache / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    tts = PiperTts(TtsConfig(fixed_cache_dir=str(cache)))
    info = tts.fixed_cache_info()
    assert info["bad"] == 1
    assert info["entries"] == 0


def test_normalize_hits_reference_rms_and_respects_peak_ceiling() -> None:
    pcm = _tone(500, amp=0.02)
    out = normalize_pcm(pcm, target_rms_dbfs=-19.0, peak_ceiling_dbfs=-1.0)
    rms = float(np.sqrt(np.mean(out * out)))
    peak = float(np.max(np.abs(out)))
    assert abs(20 * math.log10(rms) - (-19.0)) < 0.2
    assert 20 * math.log10(peak) <= -1.0 + 1e-6


def test_normalize_lowers_gain_when_peak_would_clip() -> None:
    # 高波峰因数：一个脉冲 + 大量低电平，按 RMS 提增益会撞峰值上限
    pcm = np.full(22050, 0.001, dtype=np.float32)
    pcm[0] = 0.99
    out = normalize_pcm(pcm, target_rms_dbfs=-19.0, peak_ceiling_dbfs=-1.0)
    peak = float(np.max(np.abs(out)))
    assert 20 * math.log10(peak) <= -1.0 + 1e-6


def test_catalog_stop_prompts_use_two_syllable_word() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert catalog["voice"] == "zh-CN-XiaoxiaoNeural"
    assert catalog["rate"] == "+0%"
    assert catalog["pitch"] == "+0Hz"
    assert catalog["volume"] == "+0%"
    texts = [str(item["text"]).strip() for item in catalog["prompts"]]
    assert texts, "固定提醒清单为空"
    assert "停" not in texts, "停步提示不能只发单字“停”"
    safety = [
        str(item["text"])
        for item in catalog["prompts"]
        if item.get("category") == "safety"
    ]
    assert safety and all("停下" in text for text in safety)
    ids = [str(item["id"]) for item in catalog["prompts"]]
    assert len(ids) == len(set(ids)), "固定提醒 id 重复"


def test_runtime_tts_does_not_import_edge_tts() -> None:
    source_path = _first_existing("Src/aieyes/speech/tts.py", "src/aieyes/speech/tts.py")
    source = source_path.read_text(encoding="utf-8")
    assert "edge_tts" not in source, "运行时 TTS 不得依赖在线 edge-tts"
