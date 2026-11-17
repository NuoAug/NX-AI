import pytest
import numpy as np
from asr_adapter import StreamingAsr


class Stream:
    def __init__(self):
        self.calls = []
    def accept_waveform(self, rate, samples):
        self.calls.append(('wave', rate, samples))
    def input_finished(self):
        self.calls.append(('finished',))


class Recognizer:
    def is_ready(self, stream):
        return False
    def get_result(self, stream):
        return '读一下牌子上的字'


def test_padding_finishes_decoder_without_extending_captured_audio():
    asr = StreamingAsr()
    stream = Stream()
    asr._stream = stream
    asr._recognizer = Recognizer()
    asr._t_start_ms = 100
    asr._t_end_ms = 1200
    result = asr.finalize()
    assert stream.calls[0][0:2] == ('wave', 16000)
    assert stream.calls[0][2].shape == (4800,)
    assert np.count_nonzero(stream.calls[0][2]) == 0
    assert stream.calls[1] == ('finished',)
    assert (result.t_audio_start_ms, result.t_audio_end_ms) == (100, 1200)
    assert result.text == '读一下牌子上的字'
    assert not asr.has_utterance


def test_empty_stream_does_not_invent_decoder_audio():
    asr = StreamingAsr()
    stream = Stream()
    asr._stream = stream
    asr._recognizer = Recognizer()
    asr.finalize()
    assert stream.calls == [('finished',)]


class EchoRecognizer:
    def __init__(self, text):
        self.text = text
    def is_ready(self, stream):
        return False
    def get_result(self, stream):
        return self.text


@pytest.mark.parametrize('text',['柜子里是神魔？','停直','亭止','小斌小斌','前后 有 空格'])
def test_tail_padding_never_rewrites_the_recognized_text(text):
    # 清单第 1 项红线：收尾补零只影响解码器，识别文本原样返回，不做同音纠偏。
    # 项目侧 asr.py:112 只做首尾 strip（既有行为），词句内容不替换。
    asr = StreamingAsr()
    stream = Stream()
    asr._stream = stream
    asr._recognizer = EchoRecognizer(text)
    asr._t_start_ms = 0
    asr._t_end_ms = 500
    assert asr.finalize().text == text


def test_only_outer_whitespace_is_trimmed_by_the_project_decoder():
    asr = StreamingAsr()
    asr._stream = Stream()
    asr._recognizer = EchoRecognizer('  柜子里是神魔？ ')
    asr._t_start_ms = 0
    asr._t_end_ms = 500
    assert asr.finalize().text == '柜子里是神魔？'
