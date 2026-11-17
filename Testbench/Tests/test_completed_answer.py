from unittest import TestCase
from bridge import Bridge
class CompletedAnswerTests(TestCase):
    def test_shutdown_does_not_cancel_fully_spoken_answer(self):
        b=Bridge(lambda *args:None,run_id='test')
        b.note_utterance('测试',0);turn=b.chat_begin();b.chat_done(turn,'完成回答',generation_ms=10)
        b.note_evidence({'kind':'interaction','category':'answer','event':'spoken','played':True})
        b.note_cancel('shutdown',20)
        self.assertFalse(b.snapshot_turns()[-1]['cancelled'])
    def test_incomplete_answer_still_marked_cancelled(self):
        b=Bridge(lambda *args:None,run_id='test')
        b.note_utterance('测试',0);b.chat_begin();b.note_evidence({'kind':'interaction','category':'answer','event':'spoken','played':False})
        b.note_cancel('interrupt',20)
        self.assertTrue(b.snapshot_turns()[-1]['cancelled'])
