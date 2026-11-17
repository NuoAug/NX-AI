"""SpeechFrontend 尾音假段保护：低电平短尾段不冒充用户话语。"""

from __future__ import annotations

import numpy as np

from aieyes.interfaces import AudioFrame, SpeechSegment, Utterance
from aieyes.speech.frontend import SpeechFrontend


class _Source:
    def start(self, on_frame) -> None:
        self.on_frame = on_frame

    def stop(self) -> None:
        pass


class _Vad:
    def __init__(self, segments: list[SpeechSegment]) -> None:
        self.in_speech = False
        self._segments = list(segments)
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def accept(self, frame: AudioFrame) -> SpeechSegment | None:
        if self._index >= len(self._segments):
            return None
        segment = self._segments[self._index]
        self._index += 1
        return segment


class _Asr:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self._index = 0

    def reset(self) -> None:
        pass

    def start_utterance(self) -> None:
        pass

    def accept(self, frame: AudioFrame) -> str | None:
        return None

    def finalize(self) -> Utterance:
        text = self._texts[self._index]
        self._index += 1
        return Utterance(text=text, t_audio_start_ms=0, t_audio_end_ms=0, t_final_ms=0)


class _AsrWithPreview(_Asr):
    """带 result() 的 ASR：模拟流式结果在收尾前已经可读。"""

    def __init__(self, previews: list[str], finals: list[str]) -> None:
        super().__init__(finals)
        self._previews = list(previews)
        self._preview_index = 0

    def result(self) -> str:
        if self._preview_index >= len(self._previews):
            return self._previews[-1]
        value = self._previews[self._preview_index]
        self._preview_index += 1
        return value


def _segment(level: float, text: str, start_ms: int, end_ms: int) -> SpeechSegment:
    samples = np.full(320, level, dtype=np.float32)
    return SpeechSegment(samples=samples, t_start_ms=start_ms, t_end_ms=end_ms, cause="vad_end")


def _frontend(segments: list[SpeechSegment], texts: list[str]) -> SpeechFrontend:
    frontend = SpeechFrontend(_Source(), vad=_Vad(segments), asr=_Asr(texts))
    frontend._running = True
    frontend._listening = True
    frontend._in_speech = True
    return frontend


def _feed(frontend: SpeechFrontend, frame: AudioFrame) -> list[tuple]:
    frontend._asr_active = True
    frontend._in_speech = True
    return frontend._process(frame)


def test_low_level_tail_noise_is_not_emitted() -> None:
    """真人录音 07 的尾段（rms 0.009 vs 正段 0.038，1s，文本“不东西”）应丢弃。"""
    frontend = _frontend(
        [
            _segment(0.038, "今天的天气不错我准备出去走一走", 0, 3400),
            _segment(0.0094, "不东西", 3400, 4400),
        ],
        ["今天的天气不错我准备出去走一走", "不东西"],
    )

    first = _feed(frontend, AudioFrame(np.zeros(320, np.float32), 0, 0))
    second = _feed(frontend, AudioFrame(np.zeros(320, np.float32), 20, 1))

    assert [action[0] for action in first].count("utterance") == 1
    assert [action[0] for action in second].count("utterance") == 0, second
    assert frontend.dropped_low_energy == 1


def test_continuation_after_short_pause_is_one_utterance() -> None:
    """句末“然后”后停 400ms 再说：两段合成一次话轮，不拆成两条指令。"""
    frontend = SpeechFrontend(
        _Source(),
        vad=_Vad(
            [
                _segment(0.04, "", 0, 3000),
                _segment(0.04, "", 3400, 5000),
            ]
        ),
        asr=_AsrWithPreview(["重拍第一件，然后"], ["重拍第一件，然后只比较价格"]),
    )
    frontend._running = True
    frontend._listening = True
    frontend._in_speech = True

    first = _feed(frontend, AudioFrame(np.zeros(320, np.float32), 0, 0))
    second = _feed(frontend, AudioFrame(np.zeros(320, np.float32), 20, 1))

    assert [action[0] for action in first].count("utterance") == 0, "未说完不应提前提交"
    assert [action[0] for action in second].count("utterance") == 1, second
    utterance = [action[1] for action in second if action[0] == "utterance"][0]
    assert utterance.text == "重拍第一件，然后只比较价格", utterance.text
    assert utterance.t_audio_start_ms == 0 and utterance.t_audio_end_ms == 5000
    assert len(utterance.samples) == 640, "两段音频应合并"


def test_continuation_waits_only_bounded_time() -> None:
    """用户没有再补充：700ms 有界等待到点必须提交，不能一直等。"""
    frontend = SpeechFrontend(
        _Source(),
        vad=_Vad([_segment(0.04, "", 0, 3000)]),
        asr=_AsrWithPreview(["重拍第一件，然后"], ["重拍第一件，然后"]),
    )
    frontend._running = True
    frontend._listening = True
    frontend._in_speech = True

    first = _feed(frontend, AudioFrame(np.zeros(320, np.float32), 0, 0))
    assert [action[0] for action in first].count("utterance") == 0
    # pending_until = 3000 + 700；3800ms 的静音帧触发提交。
    second = _feed(frontend, AudioFrame(np.zeros(320, np.float32), 3800, 1))
    assert [action[0] for action in second].count("utterance") == 1, second
    utterance = [action[1] for action in second if action[0] == "utterance"][0]
    assert utterance.text == "重拍第一件，然后"
    assert utterance.t_audio_end_ms == 3000


def test_complete_sentence_is_emitted_without_extra_wait() -> None:
    """普通完整句立即提交，不因续接机制多等 700ms。"""
    frontend = SpeechFrontend(
        _Source(),
        vad=_Vad([_segment(0.04, "", 0, 2000)]),
        asr=_AsrWithPreview(["第二张不清楚"], ["第二张不清楚"]),
    )
    frontend._running = True
    frontend._listening = True
    frontend._in_speech = True

    actions = _feed(frontend, AudioFrame(np.zeros(320, np.float32), 0, 0))
    assert [action[0] for action in actions].count("utterance") == 1, actions


def test_short_meaningful_word_is_kept_even_if_quiet() -> None:
    """轻声“继续/停”是有效指令，不能按尾音噪声丢掉。"""
    frontend = _frontend(
        [
            _segment(0.038, "把第二件放好了", 0, 2000),
            _segment(0.009, "继续", 2000, 2600),
        ],
        ["把第二件放好了", "继续"],
    )

    first = _feed(frontend, AudioFrame(np.zeros(320, np.float32), 0, 0))
    second = _feed(frontend, AudioFrame(np.zeros(320, np.float32), 20, 1))

    assert [action[0] for action in first].count("utterance") == 1
    assert [action[0] for action in second].count("utterance") == 1, second
    assert frontend.dropped_low_energy == 0
