"""Device enumeration and testbench-only selection; never opens capture streams."""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SELECTION = ROOT / 'config' / 'devices.json'
VIDEO_SYSFS = Path('/sys/class/video4linux')
KEYS = ('camera', 'microphone', 'speaker')
# holders() 核实不了占用时的标记：它既不是“有进程占用”，也不能当成空闲。
UNVERIFIED_PROCESS = '占用状态无法核实'

def selection():
    if not SELECTION.exists():
        return {key: '' for key in KEYS}
    data = json.loads(SELECTION.read_text())
    return {key: str(data.get(key) or '') for key in KEYS}

def validate(data, choices):
    if not isinstance(data, dict) or set(data) - set(KEYS):
        raise ValueError('设备设置格式无效')
    current = selection()
    selected = {key: str(data.get(key) or '') for key in KEYS}
    for key in KEYS:
        # 未改动的旧值即使当前离线也保留：三项同时失效时，逐项改回默认不能被其余两项卡住。
        if selected[key] not in {item['id'] for item in choices[key]} and selected[key] != current[key]:
            raise ValueError('所选设备已断开或不可用，请刷新设备列表')
    return selected

def save(data, choices):
    selected = validate(data, choices)
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    temp = SELECTION.with_suffix('.tmp')
    temp.write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    temp.replace(SELECTION)
    return selected

# External-device routing and ownership (overrides the original inventory/apply).
def holders(path):
    try:
        result = subprocess.run(['fuser', path],capture_output=True,text=True,timeout=2)
        owners=[]
        for word in result.stdout.split():
            if word.isdigit() and int(word)!=__import__('os').getpid():
                pid=int(word)
                try:name=Path(f'/proc/{pid}/comm').read_text().strip()
                except OSError:name='unknown'
                owners.append({'pid':pid,'process':name})
        return owners
    except (OSError,subprocess.TimeoutExpired):return [{'pid':None,'process':UNVERIFIED_PROCESS}]

def occupancy_unverifiable(owners):
    """占用核实本身失败（fuser 不可用/超时），不能读成“被占用”，更不能读成空闲。"""
    return any(str(o.get('process'))==UNVERIFIED_PROCESS for o in owners or [])

def occupancy_text(owners,free):
    if not owners:return free
    if occupancy_unverifiable(owners):return '占用无法核实（fuser 不可用或超时）；既不能认定空闲，也不能认定被占用'
    return '被占用：'+', '.join(str(o.get('process'))+' PID '+str(o.get('pid')) for o in owners)

def require_available(path,label='摄像头'):
    """设备可用性前置检查：核实不了占用时不抢设备，并如实说明是核实失败。"""
    owners=holders(path)
    if occupancy_unverifiable(owners):
        raise RuntimeError(label+'占用无法核实（fuser 不可用或超时）；为避免抢占其他程序正在使用的设备，本次不采集')
    if owners:raise RuntimeError(label+'被占用：'+', '.join(str(o.get('process'))+' PID '+str(o.get('pid')) for o in owners))

def parse_pipewire_dump(raw):
    decoder=json.JSONDecoder()
    offset=0
    objects={}
    anonymous=[]
    def merge(previous, update):
        result=dict(previous)
        for key,value in update.items():
            result[key]=merge(result[key],value) if isinstance(result.get(key),dict) and isinstance(value,dict) else value
        return result
    while offset<len(raw):
        while offset<len(raw) and raw[offset].isspace():offset+=1
        if offset==len(raw):break
        document,offset=decoder.raw_decode(raw,offset)
        if not isinstance(document,list):raise ValueError('PipeWire快照必须是数组')
        for item in document:
            if not isinstance(item,dict):raise ValueError('PipeWire对象必须是字典')
            ident=item.get('id')
            if ident is None:
                anonymous.append(item)
            elif 'info' in item and item['info'] is None:
                objects.pop(ident,None)
            else:
                objects[ident]=merge(objects.get(ident,{}),item)
    if offset==0:raise ValueError('PipeWire没有返回快照')
    return list(objects.values())+anonymous


def pipewire_nodes():
    data=parse_pipewire_dump(subprocess.run(['pw-dump'],capture_output=True,text=True,timeout=4,check=True).stdout)
    nodes={x['id']:dict(x.get('info',{}).get('props',{})) for x in data if x.get('type')=='PipeWire:Interface:Node'}
    clients={x['id']:x.get('info',{}).get('props',{}) for x in data if x.get('type')=='PipeWire:Interface:Client'}
    for x in data:
        if x.get('type')!='PipeWire:Interface:Link':continue
        props=x.get('info',{}).get('props',{})
        ids=[int(props.get(k,-1)) for k in ('link.input.node','link.output.node')]
        for endpoint,stream in (ids,ids[::-1]):
            ep=nodes.get(endpoint,{});sp=nodes.get(stream,{})
            cls=ep.get('media.class')
            if cls not in ('Audio/Sink','Audio/Source') or not str(sp.get('media.class','')).startswith('Stream/'):continue
            if cls=='Audio/Sink' and endpoint!=ids[0]:continue
            if cls=='Audio/Source' and endpoint!=ids[1]:continue
            cp=clients.get(int(sp.get('client.id',-1)),{})
            owner={'process':sp.get('application.name',cp.get('application.name','unknown')),'pid':sp.get('application.process.id',cp.get('application.process.id'))}
            if owner not in ep.setdefault('_owners',[]):ep['_owners'].append(owner)
    return list(nodes.items())

def external(props):
    return props.get('device.api')=='bluez5' or str(props.get('node.name','')).startswith(('bluez_','alsa_input.usb-','alsa_output.usb-'))

def inventory():
    result={key:[{'id':'','label':'系统默认（外接设备）','occupancy':'待核实'}] for key in KEYS};errors=[]
    audio_verified=False
    try:
        nodes=pipewire_nodes()
        # 只有 pw-dump 真的返回快照，才算“音频服务可用”，否则不能另外断言设备不存在。
        audio_verified=True
        for ident,props in nodes:
            cls=props.get('media.class');key={'Audio/Source':'microphone','Audio/Sink':'speaker'}.get(cls)
            if key and external(props):
                # 只看见 PipeWire 链接，不代表没有 ALSA 直连进程，措辞不越界。
                result[key].append({'id':'pw:'+props['node.name'],'label':props.get('node.description',props['node.name']),'occupancy':occupancy_text(props.get('_owners',[]),'未发现 PipeWire 应用占用')})
    except Exception as exc:errors.append('音频服务不可用: '+str(exc)[:80])
    camera_probe_missing=False
    for node in sorted(VIDEO_SYSFS.glob('video*')):
        path='/dev/'+node.name
        try:
            name=(node/'name').read_text().strip()
        except OSError:continue
        try:
            proc=subprocess.run(['v4l2-ctl','-d',path,'--list-formats'],capture_output=True,text=True,timeout=2)
        except OSError:
            camera_probe_missing=True
            continue
        except subprocess.TimeoutExpired:continue
        if 'imx708' in name.lower() or (proc.returncode==0 and re.search('YUYV|MJPG',proc.stdout)):
            result['camera'].append({'id':path,'label':name+' · '+node.name,'occupancy':occupancy_text(holders(path),'当前未发现占用')})
    if camera_probe_missing:errors.append('摄像头枚举不完整：v4l2-ctl 不可用，无法核实摄像头数量与占用')
    if audio_verified:
        for key in ('microphone','speaker'):
            if len(result[key])==1:errors.append(('麦克风' if key=='microphone' else '扬声器')+'：未发现在线蓝牙或USB外接设备')
    for key in ('microphone','speaker'):
        try:
            target=audio_target(key)
            match=next((x for x in result[key][1:] if x['id']=='pw:'+str(target)),None)
            result[key][0]['occupancy']=match['occupancy'] if match else '默认设备未指向在线外设'
        except Exception:result[key][0]['occupancy']='无法核实默认音频路由'
    return result,errors

def prepare_env(env):
    selected=selection(); nodes=pipewire_nodes()
    lines=['</usr/share/alsa/alsa.conf>']
    for key in ('microphone','speaker'):
        value=selected[key]
        if value and not value.startswith('pw:'):continue
        target=audio_target(key) or '-1'
        # Device availability is checked only when audio is actually opened.
        lines.append('pcm.bench_'+key+' { type pipewire playback_node '+json.dumps(target)+' capture_node '+json.dumps(target)+' hint { show on description "bench_'+key+'" } }')
    path=ROOT/'.runtime'/'alsa.conf';path.write_text('\n'.join(lines)+'\n');env['ALSA_CONFIG_PATH']=str(path)

def apply(cfg):
    selected=selection()
    for key,attr in (('microphone','input_device'),('speaker','output_device')):
        value=selected[key]
        setattr(cfg.audio,attr,'bench_'+key if not value or value.startswith('pw:') else int(value))
    cfg.vision.device=selected['camera'] or ''
    return selected

def require_audio(key):
    # 没有 PipeWire（Windows/精简 Linux）时，pw-dump/wpctl 会抛 FileNotFoundError。
    # 采集本来就不成立，但报错必须说清“无法核实”，不能把工具缺失冒充成设备问题。
    try:
        audio_owner_check(key)
        choices, errors=inventory()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError('音频服务不可用，无法核实' + ('麦克风' if key=='microphone' else '扬声器') + '设备与占用：' + str(exc)[:80]) from exc
    value=selection()[key]
    if not value and audio_target(key) is None:raise RuntimeError('系统默认音频未指向在线外接设备，请连接蓝牙并设为系统默认，或明确选择外接设备')
    if not any(x['id'] for x in choices[key]):
        raise RuntimeError(('麦克风' if key=='microphone' else '扬声器')+'未连接蓝牙或USB外接设备；不会使用板载空通道')
    if value and not any(x['id']==value for x in choices[key]):raise RuntimeError('所选音频设备已断开')


def audio_owner_check(key):
    target=audio_target(key)
    for _,props in pipewire_nodes():
        if target and target==props.get('node.name'):
            owners=[o for o in props.get('_owners',[]) if str(o.get('pid'))!=str(__import__('os').getpid())]
            if owners:raise RuntimeError('音频设备被其他应用占用: '+str(owners))

def audio_target(key):
    value=selection()[key]
    if value.startswith('pw:'):return value[3:]
    if value:return None
    token='@DEFAULT_AUDIO_SOURCE@' if key=='microphone' else '@DEFAULT_AUDIO_SINK@'
    try:
        proc=subprocess.run(['wpctl','inspect',token],capture_output=True,text=True,timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    match=re.search(r'node.name\s*=\s*"([^"]+)"',proc.stdout)
    name=match.group(1) if match else None
    return name if any(p.get('node.name')==name and external(p) for _,p in pipewire_nodes()) else None
