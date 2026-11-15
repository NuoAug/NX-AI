"""回声消除：块归一化 LMS（block NLMS）自适应滤波。

用途
----
本机播放（TTS/提示音）会被麦克风再次拾取，产生自唤醒和 ASR 串音。AEC 用
播放端参考信号 ``ref`` 估计声学回声 ``y``，从麦克风信号 ``mic`` 中减掉：

    e[n] = mic[n] - wᵀ · x[n]

``w`` 用块归一化 LMS 更新（每个 20 ms 帧更新一次），实现是 numpy FFT 版：
卷积与互相关都用 ``rfft/irfft`` 完成，滤波器长度 200 ms（3200 taps）时每帧
仍有足够余量，且不受 Python 逐样本循环拖慢。

时序要求
--------
``process(mic, ref)`` 里 ``ref`` 必须与 ``mic`` 同一时间轴（第 i 个参考样本
对应第 i 个麦克风样本）。前端用 ``AudioSink.reference_frame()`` 取最近一帧
播放参考，属于近似对齐；离线测试可精确对齐。

本类不负责双讲检测；双讲时自适应会变慢（残差变大但不会发散）。
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000


class NlmsAec:
    """块 NLMS 回声消除器。"""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        filter_ms: int = 200,
        mu: float = 0.6,
        eps: float = 1e-8,
        block: int | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.filter_len = max(32, int(self.sample_rate * filter_ms / 1000))
        self.mu = float(mu)
        self.eps = float(eps)
        #: 每帧样本数（默认 20ms）；用于预分配 FFT 长度
        self.block = int(block or (self.sample_rate // 50))
        self._w = np.zeros(self.filter_len, dtype=np.float32)
        self._x_hist = np.zeros(self.filter_len - 1, dtype=np.float32)
        self._nfft = self._next_pow2(self.block + 2 * self.filter_len)
        # 统计（用于报告 ERLE）
        self._mic_energy = 0.0
        self._err_energy = 0.0
        self.frames = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _next_pow2(n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p

    def reset(self) -> None:
        self._w.fill(0.0)
        self._x_hist.fill(0.0)
        self._mic_energy = 0.0
        self._err_energy = 0.0
        self.frames = 0

    # ------------------------------------------------------------------
    #: 已处理帧数 / ERLE
    @property
    def erle_db(self) -> float:
        """累计 ERLE（dB）：10·log10(Σ mic² / Σ e²)。"""
        if self._err_energy <= 0.0 or self._mic_energy <= 0.0:
            return 0.0
        return float(10.0 * np.log10((self._mic_energy + 1e-20) / (self._err_energy + 1e-20)))

    @property
    def weights(self) -> np.ndarray:
        return self._w.copy()

    # ------------------------------------------------------------------
    def process(self, mic: np.ndarray, ref: np.ndarray | None) -> np.ndarray:
        """处理一帧，返回去回声后的麦克风信号（与 ``mic`` 等长 float32）。"""
        d = np.asarray(mic, dtype=np.float32).reshape(-1)
        n = d.size
        if n == 0:
            return d.copy()
        if ref is None:
            x = np.zeros(n, dtype=np.float32)
        else:
            x = np.asarray(ref, dtype=np.float32).reshape(-1)
            if x.size < n:
                x = np.pad(x, (0, n - x.size))
            elif x.size > n:
                x = x[-n:]

        if n != self.block:
            # 帧长变化时重算 FFT 长度（正常情况下恒定）
            self.block = n
            self._nfft = self._next_pow2(self.block + 2 * self.filter_len)
            self._x_hist = np.zeros(self.filter_len - 1, dtype=np.float32)

        x_win = np.concatenate([self._x_hist, x])  # 长度 (L-1)+N
        self._x_hist = x_win[-(self.filter_len - 1) :].copy() if self.filter_len > 1 else x_win[:0]

        nfft = self._nfft
        X = np.fft.rfft(x_win, nfft)
        W = np.fft.rfft(self._w, nfft)
        y = np.fft.irfft(X * W, nfft)[self.filter_len - 1 : self.filter_len - 1 + n]
        e = d - y.astype(np.float32)

        # 块 NLMS 更新：grad[k] = Σ_n e[n]·x_win[n + (L-1-k)] → 相关后翻转
        E = np.fft.rfft(e, nfft)
        corr = np.fft.irfft(np.conj(E) * X, nfft)[: self.filter_len]
        grad = corr[::-1].astype(np.float32)
        # 标准 NLMS 归一化：除以参考窗口总能量（逐 tap 归一化会把低能量 tap 的
        # FFT 数值噪声放大，实测会发散；全局能量归一化在 3200 taps 下稳定）。
        denom = float(np.sum(x_win.astype(np.float64) ** 2)) + self.eps
        if denom > self.eps:
            self._w += (self.mu / denom) * grad
            # 轻微泄漏，防止极端情况下权重漂移
            self._w *= 1.0 - 1e-6

        self._mic_energy += float(np.sum(d.astype(np.float64) ** 2))
        self._err_energy += float(np.sum(e.astype(np.float64) ** 2))
        self.frames += 1
        return e

    # ------------------------------------------------------------------
    def process_frames(self, mic_frames: np.ndarray, ref_frames: np.ndarray) -> np.ndarray:
        """对 (frames, N) 形状的二维数组逐帧处理，返回同形状结果。"""
        m = np.asarray(mic_frames, dtype=np.float32)
        r = np.asarray(ref_frames, dtype=np.float32)
        out = np.empty_like(m)
        for i in range(m.shape[0]):
            out[i] = self.process(m[i], r[i])
        return out
