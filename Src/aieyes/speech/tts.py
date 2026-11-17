"""TTS：piper-tts 加载 zh_CN-huayan-medium。

接口要求 ``synth(text) -> TtsAudio``（整段返回）。为了压“唤醒后首声”延迟：

* 长文本按标点切段，**首段优先切短**（先合成短句立即播放，后续段再合成）；
* ``synth`` 可重复调用且线程安全（内部串行化；状态机会在线程池里预合成“我在”）；
* ``last_stats`` 记录首块/首段/整体耗时，供 T12 计时与报告。

piper 1.8 的 ``AudioChunk`` 只给 int16 数组，采样率取 ``voice.config.sample_rate``
（zh_CN-huayan-medium = 22050 Hz），这里统一转 float32 / 32768。
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import threading
import time
import wave
from pathlib import Path

import numpy as np

from aieyes.config import ModelsConfig, TtsConfig
from aieyes.errors import ModelUnavailable
from aieyes.interfaces import TtsAudio

log = logging.getLogger(__name__)

_SENTENCE_END = "。！？!?；;\n"
_COMMA = "，,、：:"
_FIXED_INDEX = "index.json"


def _db_to_gain(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _rms_dbfs(pcm: np.ndarray) -> float | None:
    x = np.asarray(pcm, dtype=np.float32)
    if x.size == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(x))))
    return round(20.0 * float(np.log10(max(1e-9, rms))), 1)


def _peak_dbfs(pcm: np.ndarray) -> float | None:
    x = np.asarray(pcm, dtype=np.float32)
    if x.size == 0:
        return None
    peak = float(np.max(np.abs(x)))
    return round(20.0 * float(np.log10(max(1e-9, peak))), 1)


def normalize_pcm(
    pcm: np.ndarray,
    *,
    target_rms_dbfs: float = -19.0,
    peak_ceiling_dbfs: float = -1.0,
) -> np.ndarray:
    """把音频响度对齐到参考试听，并给峰值留余量（避免削波）。

    先按 RMS 取增益；若峰值会超过上限，再压低增益。静音片段原样返回。
    """
    x = np.asarray(pcm, dtype=np.float32)
    if x.size == 0:
        return x
    rms = float(np.sqrt(np.mean(np.square(x))))
    peak = float(np.max(np.abs(x)))
    if rms <= 1e-9 or peak <= 1e-9:
        return x
    gain = _db_to_gain(target_rms_dbfs) / rms
    ceiling = _db_to_gain(peak_ceiling_dbfs)
    if peak * gain > ceiling:
        gain = ceiling / peak
    return (x * gain).astype(np.float32, copy=False)


def _split_long(segment: str, max_chars: int) -> list[str]:
    """把超长句按逗号切，必要时硬切。"""
    segment = segment.strip()
    if len(segment) <= max_chars:
        return [segment]
    out: list[str] = []
    buf = ""
    for ch in segment:
        buf += ch
        if ch in _COMMA and len(buf) >= max(8, max_chars // 2):
            out.append(buf.strip())
            buf = ""
        elif len(buf) >= max_chars:
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return [s for s in out if s]


def split_text(text: str, max_chars: int = 60) -> list[str]:
    """按句末标点切段；超长句再按逗号切；首段过长时优先在逗号处切短。"""
    text = (text or "").strip()
    if not text:
        return []
    sentences: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _SENTENCE_END:
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())

    segments: list[str] = []
    for sentence in sentences:
        segments.extend(_split_long(sentence, max_chars))

    # 首段短句优先：首段太长且逗号位置合理时，在第一个逗号处拆开
    if len(segments) > 1:
        soft = max(10, max_chars // 3)
        first = segments[0]
        if len(first) > soft:
            idx = -1
            for i, ch in enumerate(first):
                if ch in _COMMA and 4 <= i + 1 <= soft * 2:
                    idx = i
                    break
            if idx >= 0 and idx + 1 < len(first):
                segments = [first[: idx + 1].strip(), first[idx + 1 :].strip(), *segments[1:]]
    return [s for s in segments if s]


class PiperTts:
    """piper 语音合成封装；``synth`` 线程安全、可重复调用。"""

    def __init__(
        self,
        cfg: TtsConfig | None = None,
        models: ModelsConfig | None = None,
        *,
        model_dir: str | Path | None = None,
        voice_path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg or TtsConfig()
        self.models = models or ModelsConfig()
        self.model_dir = Path(model_dir) if model_dir else self.models.resolve(self.models.tts_dir)
        self._voice_path = Path(voice_path) if voice_path else None
        self._voice = None
        self.sample_rate = 0
        self._lock = threading.RLock()
        self._cache: dict[str, TtsAudio] = {}
        self._syn_config = None
        self._fixed: dict[str, TtsAudio] = {}
        self._fixed_loaded = False
        self._warm_started = False
        self._fixed_info: dict = {
            "enabled": bool(self.cfg.fixed_cache_dir),
            "dir": str(self.cfg.fixed_cache_dir or ""),
            "voice": str(self.cfg.fixed_voice_id or ""),
            "entries": 0,
            "missing": 0,
            "bad": 0,
            "voice_mismatch": False,
        }
        self.last_stats: dict = {}

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._voice is not None

    def _find_voice(self) -> Path:
        if self._voice_path is not None:
            if not self._voice_path.is_file():
                raise ModelUnavailable(f"TTS 模型不存在: {self._voice_path}")
            return self._voice_path
        if not self.model_dir.is_dir():
            raise ModelUnavailable(f"TTS 模型目录不存在: {self.model_dir}")
        hits = [p for p in sorted(self.model_dir.glob("*.onnx")) if not p.name.endswith(".onnx.json")]
        if not hits:
            raise ModelUnavailable(f"{self.model_dir} 下找不到 *.onnx")
        return hits[0]

    def _ensure_voice(self) -> None:
        if self._voice is not None:
            return
        try:
            import piper.voice  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover
            raise ModelUnavailable(f"未安装 piper-tts: {exc}") from exc
        model_path = self._find_voice()
        config_path = model_path.with_name(model_path.name + ".json")
        espeak_dir = self.model_dir / "espeak-ng-data"
        kwargs: dict = {}
        if config_path.is_file():
            kwargs["config_path"] = str(config_path)
        if espeak_dir.is_dir():
            kwargs["espeak_data_dir"] = espeak_dir
        self._voice = piper.voice.PiperVoice.load(str(model_path), **kwargs)
        self.sample_rate = int(getattr(self._voice.config, "sample_rate", 22050))
        try:
            from piper.config import SynthesisConfig  # noqa: PLC0415

            self._syn_config = SynthesisConfig(
                speaker_id=int(self.cfg.speaker_id) or None,
                length_scale=float(self.cfg.length_scale),
                noise_scale=float(self.cfg.noise_scale),
                noise_w_scale=float(self.cfg.noise_w),
                normalize_audio=True,
            )
        except Exception:
            self._syn_config = None

    # ------------------------------------------------------------------
    def _load_fixed_cache(self) -> None:
        """加载固定提醒缓存（index.json + WAV）；缺失/损坏只记录，不影响动态合成。"""
        if self._fixed_loaded:
            return
        self._fixed_loaded = True
        if not self.cfg.fixed_cache_dir:
            return
        root = Path(self.cfg.fixed_cache_dir)
        index_path = root / _FIXED_INDEX
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - 缓存不可用不应阻塞启动
            log.warning("固定提醒缓存不可用: %s (%s)", index_path, exc)
            self._fixed_info["error"] = f"{type(exc).__name__}: {exc}"
            return
        voice = str(data.get("voice") or "")
        if voice and self.cfg.fixed_voice_id and voice != self.cfg.fixed_voice_id:
            self._fixed_info["voice_mismatch"] = True
            log.warning(
                "固定提醒音色不匹配: cache=%s expected=%s", voice, self.cfg.fixed_voice_id
            )
            return
        entries = data.get("entries")
        if not isinstance(entries, dict):
            self._fixed_info["error"] = "entries 不是对象"
            return
        for text, item in entries.items():
            key = str(text or "").strip()
            if not key or not isinstance(item, dict):
                self._fixed_info["bad"] += 1
                continue
            name = str(item.get("file") or "")
            path = root / name
            if not name or not path.is_file():
                self._fixed_info["missing"] += 1
                log.warning("固定提醒缓存缺文件: %s", path)
                continue
            try:
                raw = path.read_bytes()
                expect = str(item.get("sha256") or "")
                if expect and hashlib.sha256(raw).hexdigest() != expect:
                    self._fixed_info["bad"] += 1
                    log.warning("固定提醒缓存哈希不符: %s", path)
                    continue
                with wave.open(io.BytesIO(raw), "rb") as wav:
                    if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
                        raise ValueError("仅支持 16-bit 单声道 WAV")
                    rate = int(wav.getframerate())
                    pcm = (
                        np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
                        .astype(np.float32)
                        / 32768.0
                    )
            except Exception as exc:  # noqa: BLE001 - 单条损坏只跳过
                self._fixed_info["bad"] += 1
                log.warning("固定提醒缓存读取失败: %s (%s)", path, exc)
                continue
            pcm = normalize_pcm(
                pcm,
                target_rms_dbfs=float(self.cfg.target_rms_dbfs),
                peak_ceiling_dbfs=float(self.cfg.peak_ceiling_dbfs),
            )
            self._fixed[key] = TtsAudio(pcm=pcm, sample_rate=rate, text=key)
        self._fixed_info["entries"] = len(self._fixed)
        if self._fixed:
            log.info(
                "固定提醒缓存已加载: %d 条, voice=%s, dir=%s",
                len(self._fixed),
                voice or "(未标注)",
                root,
            )

    def fixed_cache_info(self) -> dict:
        """固定提醒缓存状态（供启动日志/验收报告读取）。"""
        with self._lock:
            self._load_fixed_cache()
            return dict(self._fixed_info)

    def _fixed_audio(self, text: str) -> TtsAudio | None:
        self._load_fixed_cache()
        return self._fixed.get(text)

    def _warm_voice_async(self) -> None:
        """固定提醒命中后，后台加载 Piper，避免下一条动态回答再付模型加载时延。

        固定提醒让启动问候不再经过 Piper；若不预热，动态聊天首答会多出数秒
        模型加载时间。加载放在后台线程，不阻塞当前播报。
        """
        if self._voice is not None or self._warm_started:
            return
        self._warm_started = True

        def _run() -> None:
            try:
                with self._lock:
                    self._ensure_voice()
                    # 跑一小段，把 ONNX 首帧/内存分配也预热掉。
                    for _ in self._voice.synthesize("你好", syn_config=self._syn_config):
                        pass
                log.info("TTS 动态音色预热完成: %s", self.sample_rate)
            except Exception as exc:  # noqa: BLE001 - 预热失败不阻塞固定提醒播放
                log.warning("TTS 动态音色预热失败: %s", exc)

        threading.Thread(target=_run, name="tts-warm", daemon=True).start()

    # ------------------------------------------------------------------
    def synth(self, text: str) -> TtsAudio:
        """合成整段文本；固定提醒命中缓存时直接返回，空文本返回空 TtsAudio。"""
        with self._lock:
            text = (text or "").strip()
            fixed = self._fixed_audio(text) if text else None
            if fixed is not None:
                self.last_stats = {
                    "text": text,
                    "source": "fixed_cache",
                    "cache_hit": True,
                    "segments": 1,
                    "first_chunk_ms": 0.0,
                    "first_segment_ms": 0.0,
                    "total_ms": 0.0,
                    "samples": int(fixed.pcm.size),
                    "duration_ms": fixed.duration_ms,
                    "sample_rate": fixed.sample_rate,
                    "rms_dbfs": _rms_dbfs(fixed.pcm),
                    "peak_dbfs": _peak_dbfs(fixed.pcm),
                }
                self._warm_voice_async()
                return fixed
            self._ensure_voice()
            if not text:
                return TtsAudio(pcm=np.zeros(0, dtype=np.float32), sample_rate=self.sample_rate, text="")
            segments = split_text(text, int(self.cfg.max_segment_chars))
            t_all = time.perf_counter()
            first_chunk_ms: float | None = None
            first_segment_ms: float | None = None
            parts: list[np.ndarray] = []
            for i, segment in enumerate(segments):
                t_seg = time.perf_counter()
                for chunk in self._voice.synthesize(segment, syn_config=self._syn_config):
                    arr = getattr(chunk, "audio_int16_array", None)
                    if arr is None:
                        raw = getattr(chunk, "audio_int16_bytes", b"")
                        arr = np.frombuffer(raw, dtype=np.int16)
                    pcm16 = np.asarray(arr, dtype=np.int16).reshape(-1)
                    if first_chunk_ms is None:
                        first_chunk_ms = (time.perf_counter() - t_all) * 1000.0
                    parts.append(pcm16.astype(np.float32) / 32768.0)
                if i == 0:
                    first_segment_ms = (time.perf_counter() - t_seg) * 1000.0
            if parts:
                pcm = np.concatenate(parts).astype(np.float32, copy=False)
            else:
                pcm = np.zeros(0, dtype=np.float32)
            pcm = normalize_pcm(
                pcm,
                target_rms_dbfs=float(self.cfg.target_rms_dbfs),
                peak_ceiling_dbfs=float(self.cfg.peak_ceiling_dbfs),
            )
            total_ms = (time.perf_counter() - t_all) * 1000.0
            self.last_stats = {
                "text": text,
                "source": "piper",
                "cache_hit": False,
                "segments": len(segments),
                "first_segment_chars": len(segments[0]) if segments else 0,
                "first_chunk_ms": round(first_chunk_ms or 0.0, 1),
                "first_segment_ms": round(first_segment_ms or 0.0, 1),
                "total_ms": round(total_ms, 1),
                "samples": int(pcm.size),
                "duration_ms": int(round(pcm.size * 1000 / self.sample_rate)) if self.sample_rate else 0,
                "sample_rate": self.sample_rate,
                "rms_dbfs": _rms_dbfs(pcm),
                "peak_dbfs": _peak_dbfs(pcm),
            }
            return TtsAudio(pcm=pcm, sample_rate=self.sample_rate, text=text)

    def synth_cached(self, text: str) -> TtsAudio:
        """带 LRU 上限的缓存版 synth；重复调用直接复用音频。"""
        key = (text or "").strip()
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        audio = self.synth(text)
        with self._lock:
            if len(self._cache) >= 32:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = audio
        return audio

    def prewarm(self, texts: tuple[str, ...] = ("我在",)) -> dict:
        """启动时预合成唤醒回应，返回各文本的耗时统计。"""
        stats: dict = {}
        for text in texts:
            audio = self.synth_cached(text)
            stats[text] = {"duration_ms": audio.duration_ms, "stats": dict(self.last_stats)}
        return stats

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def close(self) -> None:
        with self._lock:
            self._voice = None
            self._cache.clear()
            self._fixed.clear()
            self._fixed_loaded = False
            self._warm_started = False
            self._syn_config = None


__all__ = ["PiperTts", "normalize_pcm", "split_text"]
