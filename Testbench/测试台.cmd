@echo off
chcp 65001 >nul
set "NUO_LAUNCHER_FILE=%~f0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$s=[IO.File]::ReadAllText($env:NUO_LAUNCHER_FILE,[Text.Encoding]::UTF8); & ([scriptblock]::Create($s.Substring($s.LastIndexOf('# POWERSHELL_BEGIN'))))"
if errorlevel 1 pause
exit /b
# POWERSHELL_BEGIN
$ErrorActionPreference = 'Stop'
# IP 在连接时手动输入；沿用已核验的原目标机密钥，不关闭主机密钥核验。
$Settings = @{ Host=''; User='mylin'; HostKeyAlias='192.168.2.72'; Bench='/home/mylin/Desktop/Nuo/testbench'; Port=18780 }
$Base = Split-Path -Parent $env:NUO_LAUNCHER_FILE
$Runtime = Join-Path $Base '.runtime\windows-launcher'
$StateFile = Join-Path $Runtime 'session.json'
$Url = 'http://127.0.0.1:' + $Settings.Port + '/'
$Utf8 = New-Object System.Text.UTF8Encoding($false)
$script:Tunnel = $null
$RemotePython = @'
import sys,os,json,time,socket,fcntl,subprocess,urllib.request
from pathlib import Path
action,run,folder=sys.argv[1:4]
b=Path(folder).resolve();rt=b/'.runtime';rt.mkdir(parents=True,exist_ok=True)
record=rt/'windows-launcher.json'
def emit(s):print(s,flush=True)
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def matches(item,filename):
    if not item:return False
    p=Path('/proc')/str(item['pid'])
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        return stat[0]!='Z' and int(stat[19])==item['start_ticks'] and filename in (p/'cmdline').read_text()
    except FileNotFoundError:return False
def stop_owned(state):
    for key,name,script in [('server','testbench/server.py','stop.sh'),('model','llama-server','stop_model.sh')]:
        owned=state.get(key)
        if owned and matches(owned,name):
            current=read(rt/('server.pid' if key=='server' else 'model.pid'))
            if current!=owned:raise RuntimeError('Service ownership changed; refusing to stop '+key)
            subprocess.run(['bash',str(b/'scripts'/script)],check=True)
        elif owned:emit(key+': already stopped; other processes preserved')
def healthy(port,path):
    try:
        with urllib.request.urlopen('http://127.0.0.1:'+str(port)+path,timeout=2) as r:
            return r.status==200
    except Exception:return False
def listening(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1',port))==0
def ready(port,path,seconds):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if healthy(port,path):return
        time.sleep(.5)
    raise RuntimeError('Service readiness timed out on port '+str(port))
if b.name!='testbench':raise RuntimeError('Unexpected bench directory')
if action=='stop':
    state=read(record)
    if state:
        if state['run']!=run:raise RuntimeError('Another launcher owns the remote session')
        stop_owned(state);record.unlink()
    emit('NUO_STOPPED:'+run)
elif action=='start':
    guard=open(rt/'windows-launcher.lock','a')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if record.exists():raise RuntimeError('Previous launch not closed; use the Close menu first')
    for name in ['serve_model.sh','stop_model.sh','start.sh','stop.sh']:
        if not (b/'scripts'/name).is_file():raise RuntimeError('Missing bench entry: '+name)
    config=read(b/'config/bench.json') or {}
    if int(config.get('port',18780))!=18780:raise RuntimeError('Bench port must match launcher port 18780')
    state={'run':run,'model':None,'server':None}
    def save():
        temporary=record.with_suffix('.tmp');temporary.write_text(json.dumps(state));os.replace(temporary,record)
    save()
    try:
        if not listening(8080):
            emit('Starting model...')
            (b/'logs').mkdir(exist_ok=True)
            with (b/'logs/windows-model.log').open('ab') as log:
                model=subprocess.Popen(['bash',str(b/'scripts/serve_model.sh')],cwd=b,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            # serve_model.sh writes its PID/start-ticks record before exec.
            for unused in range(50):
                item=read(rt/'model.pid')
                if item and item['pid']==model.pid:state['model']=item;save();break
                if model.poll() is not None:raise RuntimeError('Model start failed; see logs/windows-model.log')
                time.sleep(.1)
            else:raise RuntimeError('Model ownership record was not created')
        else:emit('Reusing existing model; Close will preserve it')
        ready(8080,'/health',120)
        if not healthy(18780,'/api/health'):
            if listening(18780):raise RuntimeError('Remote port 18780 is occupied by an unhealthy/unknown service')
            subprocess.run(['bash',str(b/'scripts/start.sh')],check=True)
            state['server']=read(rt/'server.pid');save()
            if not state['server']:raise RuntimeError('Missing server ownership record')
        else:emit('Reusing existing bench; Close will preserve it')
        ready(18780,'/api/health',15)
        emit('NUO_READY:'+run)
    except BaseException:
        stop_owned(state)
        record.unlink(missing_ok=True)
        raise
    # Read the SSH input pipe until the local tunnel closes. Services remain
    # recoverable via Close after a disconnect; do not stop teammates on EOF.
    for line in sys.stdin:pass
else:raise RuntimeError('Unknown action')
'@

function Read-State {
    if (Test-Path -LiteralPath $StateFile) { return Get-Content -LiteralPath $StateFile -Raw -Encoding UTF8 | ConvertFrom-Json }
    return $null
}
function Read-TargetIP([string]$previous) {
    $label='请输入目标机IP地址'
    if ($previous) { $label+='（回车沿用 '+$previous+'）' }
    $value=(Read-Host $label).Trim()
    if (-not $value -and $previous) { $value=$previous }
    $parsed=$null
    if (-not [Net.IPAddress]::TryParse($value,[ref]$parsed)) { throw 'IP地址格式不正确，请重新选择操作并输入。' }
    return $parsed.ToString()
}
function Get-OwnedTunnel($state) {
    if (-not $state) { return $null }
    try {
        $p = [Diagnostics.Process]::GetProcessById([int]$state.Pid)
        if ($p.StartTime.ToUniversalTime().Ticks.ToString() -ne [string]$state.StartTicks) { return $null }
        if ($p.MainModule.FileName -ne [string]$state.Executable) { return $null }
        return $p
    } catch { return $null }
}
function Quote-Argument([string]$value) {
    # Windows native argv quoting, including backslashes before embedded quotes.
    $escaped = [regex]::Replace($value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}
function Ssh-Arguments([string]$action,[string]$run,[switch]$Forward) {
    $payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($RemotePython))
    $remote = 'python3 -c "exec(__import__(''base64'').b64decode(''' + $payload + '''))" ' + $action + ' ' + $run + ' ' + $Settings.Bench
    $items = @('-T','-o','ConnectTimeout=8','-o','StrictHostKeyChecking=yes','-o',('HostKeyAlias='+$Settings.HostKeyAlias),'-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3')
    if ($Forward) { $items += @('-o','ExitOnForwardFailure=yes','-L',('127.0.0.1:'+ $Settings.Port + ':127.0.0.1:18780')) }
    return $items + @(($Settings.User+'@'+$Settings.Host),$remote)
}
function Open-Browser {
    # The browser is an interactive app requested by the user.
    Start-Process $Url
}
function Test-Port {
    $c = New-Object Net.Sockets.TcpClient
    try { $task=$c.ConnectAsync('127.0.0.1',[int]$Settings.Port); return ($task.Wait(500) -and $c.Connected) }
    catch { return $false } finally { $c.Dispose() }
}
function Start-Bench {
    $state=Read-State
    if ($state) {
        if (Get-OwnedTunnel $state) { Write-Host ('本入口已启动，无需重复启动或打开页面：'+$Url); return }
        throw '检测到上次会话记录，请先选择 2 关闭旧会话，再重新启动。'
    }
    if (Test-Port) { throw '本机18780端口已被占用。请先核对旧隧道，本脚本不会结束未知进程。' }
    $Settings.Host=Read-TargetIP ''
    $run=[Guid]::NewGuid().ToString('N')
    $ssh=(Get-Command ssh.exe -CommandType Application -ErrorAction Stop).Source
    $info=New-Object Diagnostics.ProcessStartInfo
    $info.FileName=$ssh
    $info.Arguments=((Ssh-Arguments start $run -Forward | ForEach-Object { Quote-Argument $_ }) -join ' ')
    $info.UseShellExecute=$false
    $info.RedirectStandardOutput=$true
    $info.RedirectStandardInput=$true
    # SSH still reads its password from the existing Windows console.
    $p=New-Object Diagnostics.Process
    $p.StartInfo=$info
    Write-Host '正在连接目标机，请在出现 password 时输入SSH密码（输入不显示）。'
    if (-not $p.Start()) { throw 'SSH进程未启动。' }
    $script:Tunnel=$p
    $state=@{ Pid=$p.Id; StartTicks=$p.StartTime.ToUniversalTime().Ticks.ToString(); Executable=$ssh; Run=$run; Host=$Settings.Host }
    [IO.File]::WriteAllText($StateFile,($state | ConvertTo-Json),$Utf8)
    $pending=$p.StandardOutput.ReadLineAsync()
    $deadline=[DateTime]::UtcNow.AddSeconds(180)
    $ready=$false
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($pending.IsCompleted) {
            $line=$pending.GetAwaiter().GetResult()
            if ($null -ne $line) {
                Write-Host $line
                [IO.File]::AppendAllText((Join-Path $Runtime 'launcher.log'),$line+[Environment]::NewLine,$Utf8)
                if ($line -eq ('NUO_READY:'+$run)) { $ready=$true; break }
                $pending=$p.StandardOutput.ReadLineAsync()
            }
        }
        if ($p.HasExited) { break }
        Start-Sleep -Milliseconds 150
    }
    if (-not $ready) { throw '启动未完成。请查看上方信息，再选择2清理本次会话；没有自动开启外设。' }
    $health=Invoke-RestMethod -Uri ($Url+'api/health') -TimeoutSec 5
    if ($health.status -ne 'ok' -or -not $health.ready) { throw '隧道已连接，但测试台健康检查未通过，请选择2清理后重试。' }
    Write-Host ('启动完成：'+$Url) -ForegroundColor Green
    Write-Host '保持本菜单窗口打开以维持隧道。关闭测试台请选择2。'
    Open-Browser
}
function Stop-Bench {
    $state=Read-State
    if (-not $state) { Write-Host '没有本入口持有的会话，未关闭其他服务。'; return }
    $Settings.Host=Read-TargetIP ([string]$state.Host)
    Write-Host '正在关闭本次启动的测试台和模型，请输入SSH密码。'
    $ssh=(Get-Command ssh.exe -CommandType Application -ErrorAction Stop).Source
    $info=New-Object Diagnostics.ProcessStartInfo
    $info.FileName=$ssh
    $info.Arguments=((Ssh-Arguments stop ([string]$state.Run) | ForEach-Object { Quote-Argument $_ }) -join ' ')
    $info.UseShellExecute=$false
    $closer=New-Object Diagnostics.Process
    $closer.StartInfo=$info
    try {
        if (-not $closer.Start()) { throw '关闭用SSH进程未启动。' }
        $closer.WaitForExit()
        if ($closer.ExitCode -ne 0) { throw '远端关闭未完成，会话记录和隧道已保留，请检查连接后重试2。' }
    } finally { $closer.Dispose() }
    $p=Get-OwnedTunnel $state
    if ($p) { $p.Kill(); $p.WaitForExit(5000) | Out-Null }
    if ($script:Tunnel) { $script:Tunnel.Dispose(); $script:Tunnel=$null }
    Remove-Item -LiteralPath $StateFile -Force
    Write-Host '已关闭本次启动的服务和隧道；原先已运行的服务保持运行。' -ForegroundColor Green
}
function Show-Menu {
    New-Item -ItemType Directory -Path $Runtime -Force | Out-Null
    while ($true) {
        Write-Host "`n========== Nuo 测试台 =========="
        Write-Host '1  启动模型、测试台、隧道并打开浏览器'
        Write-Host '2  关闭本次启动的服务和隧道'
        Write-Host '3  退出（有会话时先执行关闭）'
        $choice=Read-Host '请选择 1 / 2 / 3'
        $mutex=New-Object Threading.Mutex($false,'Local\NuoTestbenchLauncher18780')
        $locked=$false
        try {
            try { $locked=$mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $locked=$true }
            if (-not $locked) { throw '另一个启动/关闭操作正在进行，请稍后再试。' }
            switch ($choice.Trim()) {
                '1' { Start-Bench }
                '2' { Stop-Bench }
                '3' { if (Read-State) { Stop-Bench }; return }
                default { Write-Host '请输入1、2或3。' }
            }
        } catch { Write-Host ('操作未完成：'+$_.Exception.Message) -ForegroundColor Red }
        finally { if ($locked) { $mutex.ReleaseMutex() }; $mutex.Dispose() }
    }
}
Show-Menu
