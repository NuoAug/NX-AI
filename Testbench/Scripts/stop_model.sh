#!/usr/bin/env bash
set -eu
BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$BENCH_DIR/../.venv/bin/python" - "$BENCH_DIR" "${1:-}" <<'PY'
import json,os,signal,sys,time,urllib.request
from pathlib import Path
b=Path(sys.argv[1]);record=b/'.runtime/model.pid'
if not record.is_file():sys.exit('没有测试台持有的模型记录；请按原启动方式关闭。')
owned=json.loads(record.read_text());pid=int(owned['pid']);proc=Path(f'/proc/{pid}')
if not proc.exists():print('模型已停止。');sys.exit(0)
ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]);args=(proc/'cmdline').read_bytes().split(b'\0')
if ticks!=owned['start_ticks'] or Path(os.fsdecode(args[0])).name!='llama-server':sys.exit('进程身份不匹配，拒绝关闭。')
if sys.argv[2]=='--check':print(f'已核对测试台模型 PID {pid} 与启动标识。');sys.exit(0)
try:
    state=json.load(urllib.request.urlopen('http://127.0.0.1:18780/api/state',timeout=2))
except Exception:state={}
if any((state.get(k) or {}).get('status') in ('starting','running','stopping') for k in ('run','debug')) or (state.get('preview') or {}).get('active'):
    sys.exit('测试台仍有活动任务，请先停止任务或关闭测试台服务。')
def exited():
    try:
        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        return fields[0]=='Z' or int(fields[19])!=owned['start_ticks']
    except FileNotFoundError:return True

os.kill(pid,signal.SIGINT)
for _ in range(200):
    if exited():
        if record.is_file() and json.loads(record.read_text())==owned:record.unlink()
        print('模型已正常停止。');sys.exit(0)
    time.sleep(.1)
sys.exit('模型尚未退出，未强制终止；请检查模型日志。')
PY
