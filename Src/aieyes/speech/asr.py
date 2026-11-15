"""流式 ASR：sherpa-onnx streaming paraformer 封装。

``start_utterance`` / ``accept`` / ``finalize`` 对应一次收音：
VAD 判定开口时前端就开始喂帧，判定结束时 finalize 得到 ``Utterance``，
并带上音频起止时间与出字时间（供 T02/T11 计时）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aieyes.clock import mono_ms
from aieyes.config import AsrConfig, ModelsConfig
from aieyes.errors import ModelUnavailable
from aieyes.interfaces import SAMPLE_RATE, AudioFrame, Utterance


def pick_model_file(model_dir: Path, stem: str) -> Path:
    for pattern in (f"{stem}.int8.onnx", f"{stem}*.int8.onnx", f"{stem}.onnx", f"{stem}*.onnx"):
        hits = sorted(model_dir.glob(pattern))
        if hits:
            return hits[0]
    raise ModelUnavailable(f"{model_dir} 下找不到 {stem}*.onnx")


class StreamingAsr:
    """流式 paraformer 识别器（采集线程内同步调用）。"""

    def __init__(
        self,
        cfg: AsrConfig | None = None,
        models: ModelsConfig | None = None,
        *,
        model_dir: str | Path | None = None,
        num_threads: int = 2,
        enable_endpoint_detection: bool = False,
    ) -> None:
        self.cfg = cfg or AsrConfig()
        self.models = models or ModelsConfig()
        self.model_dir = Path(model_dir) if model_dir else self.models.resolve(self.models.asr_dir)
        self.num_threads = int(num_threads)
        self.enable_endpoint_detection = bool(enable_endpoint_detection)
        self._recognizer = None
        self._stream = None
        self._t_start_ms: int | None = None
        self._t_end_ms: int | None = None
        self._last_partial = ""

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._recognizer is not None

    @property
    def has_utterance(self) -> bool:
        return self._stream is not None

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._recognizer is not None:
            return
        try:
            import sherpa_onnx  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover
            raise ModelUnavailable(f"未安装 sherpa-onnx: {exc}") from exc
        model_dir = self.model_dir
        tokens = model_dir / "tokens.txt"
        if not tokens.is_file():
            raise ModelUnavailable(f"ASR tokens.txt 不存在: {tokens}")
        encoder = pick_model_file(model_dir, "encoder")
        decoder = pick_model_file(model_dir, "decoder")
        kwargs = dict(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            num_threads=self.num_threads,
            sample_rate=int(self.cfg.sample_rate),  # nanobind 要求 int
            feature_dim=int(self.cfg.feature_dim),
            decoding_method=str(self.cfg.decoding_method),
            enable_endpoint_detection=self.enable_endpoint_detection,
            provider="cpu",
        )
        try:
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(**kwargs)
        except TypeError:
            # 旧版本可能没有某些字段：退化为最小参数集
            kwargs.pop("enable_endpoint_detection", None)
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(**kwargs)

    # ------------------------------------------------------------------
    def start_utterance(self) -> None:
        self._ensure_loaded()
        self._stream = self._recognizer.create_stream()
        self._t_start_ms = None
        self._t_end_ms = None
        self._last_partial = ""

    def accept(self, frame: AudioFrame) -> str | None:
        """喂一帧；有新的部分结果时返回文本，否则 ``None``。"""
        if self._stream is None:
            self.start_utterance()
        stream = self._stream
        if self._t_start_ms is None:
            self._t_start_ms = int(frame.t_start_ms)
        sr = float(self.cfg.sample_rate)
        stream.accept_waveform(sr, np.asarray(frame.samples, dtype=np.float32))
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        self._t_end_ms = int(frame.t_start_ms) + int(frame.duration_ms)
        text = str(self._recognizer.get_result(stream)).strip()
        if text and text != self._last_partial:
            self._last_partial = text
            return text
        return None

    def finalize(self) -> Utterance:
        """结束当前片段，返回最终文本与时间戳（无音频时返回空文本）。"""
        now = mono_ms()
        stream = self._stream
        text = ""
        if stream is not None:
            try:
                stream.input_finished()
            except Exception:
                pass
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)
            text = str(self._recognizer.get_result(stream)).strip()
        t_start = self._t_start_ms if self._t_start_ms is not None else now
        t_end = self._t_end_ms if self._t_end_ms is not None else t_start
        utt = Utterance(
            text=text,
            t_audio_start_ms=int(t_start),
            t_audio_end_ms=int(t_end),
            t_final_ms=int(now),
        )
        self.reset()
        return utt

    def reset(self) -> None:
        self._stream = None
        self._t_start_ms = None
        self._t_end_ms = None
        self._last_partial = ""

    def close(self) -> None:
        self.reset()
        self._recognizer = None


__all__ = ["StreamingAsr", "pick_model_file"]
