"""流式 ASR 单测：模型自带 wav → 中文文本 + 时间戳。"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np

from speech_helpers import frames_from_wav, has_cjk

from aieyes.config import AsrConfig
from aieyes.interfaces import SAMPLE_RATE, AudioFrame
from aieyes.speech.asr import StreamingAsr


def _question_wav(models):
    return models.resolve(models.asr_dir) / "test_wavs" / "0.wav"


def test_transcribe_test_wav(asr, models):
    asr.reset()
    asr.start_utterance()
    frames = frames_from_wav(_question_wav(models))
    partials = []
    for frame in frames:
        partial = asr.accept(frame)
        if partial:
            partials.append(partial)
    utt = asr.finalize()

    assert utt.text.strip(), f"ASR 输出为空（partials={partials[-3:]}）"
    assert has_cjk(utt.text), f"ASR 输出不含中文: {utt.text!r}"
    assert utt.t_audio_start_ms == frames[0].t_start_ms
    assert utt.t_audio_end_ms >= utt.t_audio_start_ms
    assert utt.t_final_ms >= utt.t_audio_end_ms
    assert utt.duration_ms > 1000
    print(f"[asr] text={utt.text!r} partials={len(partials)} duration_ms={utt.duration_ms}")


def test_reset_and_empty_finalize(asr):
    asr.reset()
    assert asr.has_utterance is False
    utt = asr.finalize()
    assert utt.text == ""
    assert utt.t_audio_start_ms >= 0
    assert asr.has_utterance is False


def test_silence_after_speech_is_not_fed_to_streaming_model():
    engine = StreamingAsr(AsrConfig())
    stream = Mock()
    engine._stream = stream
    engine._recognizer = Mock()
    engine._recognizer.is_ready.return_value = False
    engine._recognizer.get_result.return_value = ""
    engine.accept(AudioFrame(np.full(320, 0.05, np.float32), 0))
    engine.accept(AudioFrame(np.zeros(320, np.float32), 20))
    assert stream.accept_waveform.call_count == 1


def test_finalize_pads_tail_silence_after_speech():
    """句后静音被跳过时，finalize 必须补零把最后一个 chunk 喂满，否则吞末字。"""
    engine = StreamingAsr(AsrConfig())
    stream = Mock()
    engine._stream = stream
    engine._recognizer = Mock()
    engine._recognizer.is_ready.return_value = False
    engine._recognizer.get_result.return_value = "这两个哪个贵"
    engine._t_start_ms = 0
    engine._t_end_ms = 1000
    engine._speech_started = True

    utt = engine.finalize()

    assert utt.text == "这两个哪个贵"
    stream.accept_waveform.assert_called_once()
    rate, pad = stream.accept_waveform.call_args.args
    assert rate == 16000
    assert len(pad) == int(16000 * StreamingAsr._TAIL_PAD_MS / 1000)
    stream.input_finished.assert_called_once()


def test_finalize_without_speech_does_not_pad():
    """纯噪声/空片段没有可收尾的语音，补零只会白跑一轮解码。"""
    engine = StreamingAsr(AsrConfig())
    stream = Mock()
    engine._stream = stream
    engine._recognizer = Mock()
    engine._recognizer.is_ready.return_value = False
    engine._recognizer.get_result.return_value = ""
    engine._speech_started = False

    engine.finalize()

    stream.accept_waveform.assert_not_called()
    stream.input_finished.assert_called_once()


def test_accept_starts_utterance_implicitly(asr, models):
    asr.reset()
    frames = frames_from_wav(_question_wav(models))[:20]  # 只喂 0.4s
    out = [asr.accept(frame) for frame in frames]
    assert asr.has_utterance is True
    utt = asr.finalize()
    assert utt.t_audio_end_ms > utt.t_audio_start_ms
    assert all(p is None or isinstance(p, str) for p in out)
