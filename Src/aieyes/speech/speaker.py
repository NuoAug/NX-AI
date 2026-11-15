"""声纹：sherpa-onnx SpeakerEmbeddingExtractor + campplus onnx。

``embed`` 返回 L2 归一化后的 192 维 float32 向量，``cosine`` 为余弦相似度。
前端把 VAD 段落送进来，会话层用 ``SpeakerConfig.threshold``（默认 0.65）
判断是否已注册说话人。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aieyes.config import ModelsConfig, SpeakerConfig
from aieyes.errors import InferenceError, ModelUnavailable
from aieyes.interfaces import SAMPLE_RATE

from aieyes.audio.wav import read_wav, resample_linear


class SherpaSpeaker:
    """说话人向量提取器（campplus 16k 中文模型）。"""

    def __init__(
        self,
        cfg: SpeakerConfig | None = None,
        models: ModelsConfig | None = None,
        *,
        onnx_path: str | Path | None = None,
        num_threads: int = 2,
    ) -> None:
        self.cfg = cfg or SpeakerConfig()
        self.models = models or ModelsConfig()
        self.onnx_path = (
            Path(onnx_path) if onnx_path else self.models.resolve(self.models.speaker_onnx)
        )
        self.num_threads = int(num_threads)
        self._extractor = None
        self._dim = 0

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._extractor is not None

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        return self._dim

    def _ensure_loaded(self) -> None:
        if self._extractor is not None:
            return
        if not self.onnx_path.is_file():
            raise ModelUnavailable(f"声纹模型不存在: {self.onnx_path}")
        try:
            import sherpa_onnx  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover
            raise ModelUnavailable(f"未安装 sherpa-onnx: {exc}") from exc
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(self.onnx_path),
            num_threads=self.num_threads,
            debug=False,
            provider="cpu",
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self._dim = int(self._extractor.dim)

    # ------------------------------------------------------------------
    def embed(self, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """提取 L2 归一化声纹向量。"""
        self._ensure_loaded()
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE and samples.size:
            samples = resample_linear(samples, int(sample_rate), SAMPLE_RATE)
        if samples.size == 0:
            raise InferenceError("声纹输入音频为空")
        stream = self._extractor.create_stream()
        stream.accept_waveform(float(SAMPLE_RATE), samples)
        try:
            stream.input_finished()
        except Exception:
            pass
        if not self._extractor.is_ready(stream):
            raise InferenceError(f"声纹输入过短: {samples.size / SAMPLE_RATE:.2f}s")
        emb = np.asarray(self._extractor.compute(stream), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb

    def embed_file(self, path: str | Path) -> np.ndarray:
        samples, sr = read_wav(path, target_rate=SAMPLE_RATE)
        return self.embed(samples, sr)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度，范围 [-1, 1]；零向量返回 0。"""
        va = np.asarray(a, dtype=np.float32).reshape(-1)
        vb = np.asarray(b, dtype=np.float32).reshape(-1)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def is_same(self, a: np.ndarray, b: np.ndarray) -> bool:
        return self.cosine(a, b) >= float(self.cfg.threshold)

    def close(self) -> None:
        self._extractor = None


__all__ = ["SherpaSpeaker"]
