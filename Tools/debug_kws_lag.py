"""临时诊断：回放线程里 KWS 的帧时间戳与真实时间差（不改业务代码，用完可删）。"""
import time

from aieyes.clock import mono_ms
from aieyes.audio.replay_source import ReplaySource
from aieyes.config import AppConfig
from aieyes.speech.kws import SherpaKws

cfg = AppConfig.from_file("Config/target.json")
kws = SherpaKws(cfg.kws)
wake = "Tests/Fixtures/audio/wake/tts_xiaobing_nihao.wav"

rows = []
hits = []


def cb(frame):
    now = mono_ms()
    t_call = mono_ms()
    hit = kws.accept(frame)
    dur = mono_ms() - t_call
    rows.append((int(frame.t_start_ms), now, now - int(frame.t_start_ms), hit, dur))
    if hit:
        hits.append((int(frame.t_start_ms), now, dur, hit))


src = ReplaySource([wake], realtime=True)
t_start = mono_ms()
src.start(cb)
time.sleep(3.0)
src.stop()

print("start_mono_ms", t_start)
print("frames", len(rows))
print("first_frame_ts", rows[0][0] if rows else None, "lag", rows[0][2] if rows else None)
print("max_lag", max(r[2] for r in rows) if rows else None)
print("hits", hits)
durs = [r[4] for r in rows]
print("kws_accept_ms min/mean/max", min(durs), round(sum(durs) / len(durs), 2), max(durs))
for r in rows[:3] + rows[-3:]:
    print("row", r)
