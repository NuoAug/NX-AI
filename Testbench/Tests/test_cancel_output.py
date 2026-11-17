import dataclasses
from types import SimpleNamespace

from aieyes import events as evmod
from debug import DebugRunner

from bridge import Bridge, ObservedBus


def test_cancelled_debug_never_emits_a_successful_answer():
    events=[]
    ctx=SimpleNamespace(stopped=True,emit=lambda *args:events.append(args))
    runner=DebugRunner(ctx)
    runner._emit('assistant.final',{'text':'late answer','cancelled':False})
    assert events==[]
    ctx.stopped=False
    runner._emit('assistant.final',{'text':'current answer','cancelled':False})
    assert len(events)==1 and events[0][1]['text']=='current answer'


# --------------------------------------------------------------- 播放结束原因透传
# speech-b 把播放卡死拆成 PlaybackStopped.stop_reason（completed / user_cancel /
# preempted / playback_stalled），页面上要能分清"用户主动停"和"播放卡死"。
class _Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, etype, turn_id, payload):
        self.events.append((etype, turn_id, dict(payload)))

    def finished(self):
        rows = [p for t, _tid, p in self.events if t == "playback" and p.get("state") == "finished"]
        assert rows, f"没有 playback/finished 事件：{self.events}"
        return rows[-1]


def _bridge_with_turn():
    rec = _Recorder()
    bridge = Bridge(rec, run_id="run-x")
    turn_id = bridge.chat_begin()
    return bridge, rec, turn_id


def _playback_ev(**kw):
    base = dict(piece_id="p-1", cancelled=True, played_ms=120, total_ms=900)
    base.update(kw)
    return SimpleNamespace(**base)


def test_playback_finished_carries_playback_stalled_reason():
    bridge, rec, turn_id = _bridge_with_turn()
    bridge.note_playback_finished(_playback_ev(stop_reason="playback_stalled"))
    row = rec.finished()
    assert row["stop_reason"] == "playback_stalled"
    assert row["cancelled"] is True          # 既有字段语义不变
    assert row["piece_id"] == "p-1"
    assert row["played_ms"] == 120 and row["total_ms"] == 900
    assert row["state"] == "finished"


def test_playback_finished_carries_user_cancel_reason():
    bridge, rec, _turn_id = _bridge_with_turn()
    bridge.note_playback_finished(_playback_ev(stop_reason="user_cancel"))
    assert rec.finished()["stop_reason"] == "user_cancel"


def test_playback_finished_without_reason_is_empty_not_user_cancel():
    # 旧事件（没有 stop_reason 属性）必须透传空串，页面按"未知"处理
    bridge, rec, _turn_id = _bridge_with_turn()
    bridge.note_playback_finished(_playback_ev())
    row = rec.finished()
    assert row["stop_reason"] == ""
    assert row["stop_reason"] != "user_cancel"


def test_playback_finished_none_reason_becomes_empty_string():
    bridge, rec, _turn_id = _bridge_with_turn()
    bridge.note_playback_finished(_playback_ev(stop_reason=None))
    assert rec.finished()["stop_reason"] == ""


# ------------------------------------------------- D4：一次播放只留一条 finished
# 一次播放（卡死/取消/播完）会先后收到两条事件：sink 的 PlaybackStopped（PlaybackFinished
# 子类，带 stop_reason）与 engine 的 PlaybackFinished。两条都派发 ⇒ 页面收到两条 finished。
# 判据：跳过 sink 子类，保留 engine 那条为唯一终态（它带 stop_reason）。
class _Bus:
    """ObservedBus 只用到 post()。"""

    def __init__(self):
        self.seen = []

    def post(self, ev):
        self.seen.append(ev)


def _finished_rows(rec):
    return [p for t, _tid, p in rec.events if t == "playback" and p.get("state") == "finished"]


def _engine_finished(reason="playback_stalled", **kw):
    """engine 侧终态事件；D2 落地后 events.PlaybackFinished 才有 stop_reason。"""
    names = {f.name for f in dataclasses.fields(evmod.PlaybackFinished)}
    assert "stop_reason" in names, (
        "D2 未落地：aieyes.events.PlaybackFinished 还没有 stop_reason 字段；"
        "此时做去重会把原因丢成空串（lead m01378 的顺序前提）"
    )
    base = dict(piece_id="p-1", t_ms=460, played_ms=0, total_ms=900, cancelled=True)
    base.update(kw)
    base["stop_reason"] = reason
    return evmod.PlaybackFinished(**base)


def _sink_stopped(reason="playback_stalled", **kw):
    from aieyes.audio.output import PlaybackStopped

    base = dict(
        piece_id="p-1", t_ms=400, played_ms=380, total_ms=900,
        cancelled=True, text="答案",
    )
    base.update(kw)
    return PlaybackStopped(**base, stop_reason=reason)


def _observed_bridge():
    bridge, rec, _turn_id = _bridge_with_turn()
    return ObservedBus(_Bus(), bridge=bridge), bridge, rec


def test_stalled_playback_leaves_one_finished_event_with_reason():
    bus, _bridge, rec = _observed_bridge()
    bus.post(_sink_stopped("playback_stalled"))
    bus.post(_engine_finished("playback_stalled"))
    rows = _finished_rows(rec)
    assert len(rows) == 1, f"卡死时页面只应收到 1 条 finished，实际 {len(rows)} 条：{rows}"
    assert rows[0]["state"] == "finished"
    assert rows[0]["stop_reason"] == "playback_stalled"


def test_normal_completion_leaves_one_finished_event_with_reason():
    # engine.py:602 播完时显式发 stop_reason=""（cancelled=False 就是"播完了"），
    # 所以这里按真实取值断言，不臆造 "completed"。
    bus, _bridge, rec = _observed_bridge()
    bus.post(_sink_stopped("completed", played_ms=900, cancelled=False))
    bus.post(_engine_finished("", played_ms=900, cancelled=False))
    rows = _finished_rows(rec)
    assert len(rows) == 1, f"正常播完只应收到 1 条 finished，实际 {len(rows)} 条：{rows}"
    assert rows[0]["state"] == "finished"
    assert rows[0]["cancelled"] is False
    assert rows[0]["stop_reason"] not in ("playback_stalled", "user_cancel")


def test_user_cancel_leaves_one_finished_event_with_reason():
    bus, _bridge, rec = _observed_bridge()
    bus.post(_sink_stopped("user_cancel", piece_id="p-3", played_ms=100))
    bus.post(_engine_finished("user_cancel", piece_id="p-3"))
    rows = _finished_rows(rec)
    assert len(rows) == 1, f"用户取消只应收到 1 条 finished，实际 {len(rows)} 条：{rows}"
    assert rows[0]["stop_reason"] == "user_cancel"


def test_engine_finished_without_sink_annotation_is_still_dispatched():
    # 反向保护：没有 sink 侧 PlaybackStopped 时，engine 那条必须照旧派发
    bus, _bridge, rec = _observed_bridge()
    bus.post(_engine_finished("", played_ms=900, cancelled=False))
    rows = _finished_rows(rec)
    assert len(rows) == 1
    assert rows[0]["state"] == "finished" and rows[0]["cancelled"] is False
