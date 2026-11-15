#!/usr/bin/env bash
# 在目标机项目 venv 中运行 pytest（接受任意 pytest 参数）。
#   Tools/remote_pytest.sh -q
#   Tools/remote_pytest.sh Tests/Unit/audio -x
set -euo pipefail
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$_root/Tools/target_env.sh"
cd "$_root"
tsync >/dev/null
tssh "cd '$AIEYES_DEST' && . .venv/bin/activate && python -m pytest $*"
