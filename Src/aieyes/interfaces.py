"""各模块之间共享的数据类型与协议（Lead 维护）。

职责边界
--------
``audio`` / ``speech``      实现 ``AudioSource``、``AudioSink``、``WakeWordDetector``、
                            ``Vad``、``Asr``、``Tts``、``VoiceFrontend``
``vision``                  实现 ``Camera``、``Ocr``、``Vlm``
``speech.speaker``          实现 ``SpeakerEmbedder``
``contract``                负责 interaction.v0.1 的编解码与传输，不实现协议
``interaction``             组合以上能力，实现会话状态机

线程约定
--------
采集回调（PortAudio/读线程）与 asyncio 主线程并发；``AudioSink``、``Camera`` 的
公开方法必须可在 asyncio 线程调用，内部自行加锁。事件一律通过
``aieyes.events.EventBus`` 投递到 asyncio 队列，不要在回调里直接驱动状态机。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

#: 统一音频采样率（16 kHz，mono，float32 [-1, 1]）
SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320


# ---------------------------------------------------------------------------
# 音频 / 语音
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AudioFrame:
    """一帧采集音频。``t_start_ms`` 是该帧第一个采样点的单调时钟时间。"""

    samples: np.ndarray  # float32, mono, 16 kHz
    t_start_ms: int
    seq: int = 0

    @property
    def duration_ms(self) -> int:
        return int(round(len(self.samples) * 1000 / SAMPLE_RATE))


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """VAD 判定的一段人声（含前置保护音频）。"""

    samples: np.ndarray
    t_start_ms: int
    t_end_ms: int
    cause: str = "vad_end"  # vad_end | max_len | cancel


@dataclass(frozen=True, slots=True)
class Utterance:
    """一次最终识别结果及其时间戳（用于 T02/T11 计时）。"""

    text: str
    t_audio_start_ms: int
    t_audio_end_ms: int
    t_final_ms: int
    #: VAD 段音频（含 preroll，16 kHz float32）；阶段 5 身份核验使用
    samples: np.ndarray | None = None

    @property
    def duration_ms(self) -> int:
        return self.t_audio_end_ms - self.t_audio_start_ms


@dataclass(frozen=True, slots=True)
class TtsAudio:
    """合成好的音频片段。"""

    pcm: np.ndarray  # float32, mono
    sample_rate: int
    text: str = ""

    @property
    def duration_ms(self) -> int:
        return int(round(len(self.pcm) * 1000 / self.sample_rate))


@dataclass
class PlaybackHandle:
    """一次播放的句柄：支持取消与等待完成。"""

    piece_id: str
    total_ms: int
    text: str = ""
    _cancel: Callable[[], None] | None = field(default=None, repr=False)
    _wait: Callable[[float | None], bool] | None = field(default=None, repr=False)

    def cancel(self) -> None:
        if self._cancel is not None:
            self._cancel()

    def wait(self, timeout: float | None = None) -> bool:
        """等待播放结束（或取消）；返回是否已结束。

        注意：返回值 **不等于** 「播完了」——``cancel()`` 之后返回 True，播放卡死
        （一直没播完）返回 False。需要区分「用户主动取消 / 播放卡死 / 正常播完」时，
        读句柄的 ``stop_reason``（``user_cancel`` / ``playback_stalled`` / ``""``）。
        """
        if self._wait is None:
            return True
        return bool(self._wait(timeout))


# ---------------------------------------------------------------------------
# 图像 / 视觉
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ImageFrame:
    """一帧 RGB 图像（uint8 HxWx3）及其取图时间。"""

    rgb: np.ndarray
    t_capture_ms: int
    frame_id: str = ""
    source: str = "camera"  # camera | file | synthetic

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


@dataclass(frozen=True, slots=True)
class OcrLine:
    text: str
    box: tuple[int, int, int, int]  # x, y, w, h（原图像素）
    score: float
    polygon: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class OcrResult:
    lines: tuple[OcrLine, ...] = ()
    t_ms: int = 0
    source_frame_id: str = ""
    image_size: tuple[int, int] = (0, 0)  # (w, h)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


# ---------------------------------------------------------------------------
# 协议（由 audio/speech/vision 各子包实现）
# ---------------------------------------------------------------------------
class AudioSource(Protocol):
    sample_rate: int

    def start(self, on_frame: Callable[[AudioFrame], None]) -> None: ...
    def stop(self) -> None: ...


class AudioSink(Protocol):
    """唯一播放出口：同一时刻最多一段音频，支持取消与可信游标。"""

    def start(self) -> None: ...
    def play(
        self,
        pcm: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        *,
        piece_id: str = "",
        text: str = "",
    ) -> PlaybackHandle: ...
    def stop(self) -> None: ...
    @property
    def is_playing(self) -> bool: ...
    def played_ms(self) -> int:
        """当前片段已真正输出的毫秒数（可信游标，用于取消/续播判定）。"""

    def reference_frame(self) -> np.ndarray | None:
        """最近一帧已输出的音频（AEC 参考信号）；无播放时为 None。"""

    def close(self) -> None: ...


class WakeWordDetector(Protocol):
    def accept(self, frame: AudioFrame) -> str | None:
        """喂一帧；命中返回唤醒词，否则 None。"""

    def reset(self) -> None: ...


class Vad(Protocol):
    def accept(self, frame: AudioFrame) -> SpeechSegment | None:
        """喂一帧；一段人声结束时返回该段（含前置保护音频）。"""

    def reset(self) -> None: ...

    @property
    def in_speech(self) -> bool: ...


class Asr(Protocol):
    def start_utterance(self) -> None:
        """开始一段新的识别（清空流式状态）。"""

    def accept(self, frame: AudioFrame) -> str | None:
        """喂一帧；有新的中间结果时返回文本。"""

    def finalize(self) -> Utterance:
        """结束本段识别并返回带时间戳的最终结果。"""

    def reset(self) -> None: ...


class Tts(Protocol):
    def synth(self, text: str) -> TtsAudio: ...


class VoiceFrontend(Protocol):
    """唤醒 + VAD + 流式 ASR 粘在同一个 AudioSource 上。

    回调在采集线程触发，实现方负责切到事件总线（``EventBus.post``）。
    """

    on_wake: Callable[[str, int], None] | None
    on_speech_start: Callable[[int], None] | None
    on_partial: Callable[[str, int], None] | None
    on_utterance: Callable[[Utterance], None] | None
    on_level: Callable[[float, int], None] | None

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def set_listening(self, enabled: bool) -> None:
        """开/关 VAD+ASR；关闭时丢弃当前未完成片段。"""

    def set_wake_enabled(self, enabled: bool) -> None:
        """开/关唤醒词检测（播报期间可关闭以抑制自唤醒）。"""

    def flush(self) -> None:
        """丢弃当前未完成的识别片段。"""


class Camera(Protocol):
    def open(self) -> None: ...
    def grab(self, timeout_ms: int = 10_000) -> ImageFrame: ...
    def close(self) -> None: ...


class Ocr(Protocol):
    def recognize(self, rgb: np.ndarray, *, source_frame_id: str = "") -> OcrResult: ...


class Vlm(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        images: Sequence[np.ndarray] = (),
        max_tokens: int = 512,
        timeout_s: float = 120.0,
        temperature: float | None = None,
    ) -> str: ...


class SpeakerEmbedder(Protocol):
    def embed(self, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray: ...
    def cosine(self, a: np.ndarray, b: np.ndarray) -> float: ...
