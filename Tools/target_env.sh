#!/usr/bin/env bash
# 目标机访问环境：source 本文件后可用 tssh / tscp / trun / tsync。
#
# 口令从本机 Codex 历史（~/.codex/thread_history_1.sqlite）提取到 SSHPW，
# 只写进 .tmp/askpass.sh 供 OpenSSH 使用，禁止回显、禁止写入仓库文件。
#
# 用法：
#   source Tools/target_env.sh
#   tssh 'uptime'
#   tscp -r Src/aieyes/vision mylin@192.168.2.72:/home/mylin/Desktop/Nuo/src/aieyes/
#   trun "python -m pytest -q tests/unit/vision"
set -u

_te_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIEYES_TARGET="${AIEYES_TARGET:-mylin@192.168.2.72}"
AIEYES_DEST="${AIEYES_DEST:-/home/mylin/Desktop/Nuo}"
export AIEYES_TARGET AIEYES_DEST

_hist="$HOME/.codex/thread_history_1.sqlite"
if [ -z "${SSHPW:-}" ] && [ -f "$_hist" ]; then
  SSHPW="$(grep -ao -E 'SSH 密码：[A-Za-z0-9._-]+' "$_hist" | head -1 | sed 's/^SSH 密码：//')"
fi
if [ -z "${SSHPW:-}" ]; then
  echo "[target_env] 未找到 SSH 口令（~/.codex/thread_history_1.sqlite）" >&2
  return 1 2>/dev/null || exit 1
fi
export SSHPW

mkdir -p "$_te_root/.tmp"
_te_askpass="$_te_root/.tmp/askpass.sh"
printf '#!/bin/sh\nprintf "%%s\\n" "$SSHPW"\n' >"$_te_askpass"
chmod +x "$_te_askpass"
export SSH_ASKPASS="$_te_askpass" SSH_ASKPASS_REQUIRE=force DISPLAY="${DISPLAY:-:0}"

_te_ssh_opts=(
  -o PreferredAuthentications=password
  -o PubkeyAuthentication=no
  -o NumberOfPasswordPrompts=1
  -o ConnectTimeout=10
  -o ServerAliveInterval=30
  -o LogLevel=ERROR
  -o StrictHostKeyChecking=accept-new
)

tssh() { ssh "${_te_ssh_opts[@]}" "$AIEYES_TARGET" "$@"; }
tscp() { scp "${_te_ssh_opts[@]}" "$@"; }

# 在目标机项目目录 + venv 下执行命令（PYTHONPATH 指到 src，供 python -m aieyes.* 用）
trun() { tssh "cd '$AIEYES_DEST' && export PYTHONPATH='$AIEYES_DEST/src' && . .venv/bin/activate && $*"; }

# 把本地代码同步到目标机（Src/Config/Tests/Tools/pyproject.toml）
tsync() {
  ( cd "$_te_root" && tar -czf - Src Config Tests Tools pyproject.toml 2>/dev/null \
      | tssh "mkdir -p '$AIEYES_DEST' && tar -C '$AIEYES_DEST' -xzf -" )
  echo "[tsync] -> $AIEYES_TARGET:$AIEYES_DEST"
}
