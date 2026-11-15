"""命令行入口：装配并运行 AI_Eyes 视觉语言交互子系统（阶段 1 语音闭环）。

用法示例：
  python -m aieyes.main --config Config/target.json
  python -m aieyes.main --replay ~/Desktop/Nuo/fixtures/replay --duration 20
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .app import configure_logging, run_app
from .config import AppConfig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="aieyes", description="AI_Eyes 视觉语言交互子系统")
    parser.add_argument("--config", default="Config/target.json", help="JSON 配置文件路径")
    parser.add_argument("--replay", default=None, help="改为文件回放模式：包含 wav 的目录")
    parser.add_argument("--duration", type=float, default=None, help="运行时长（秒），默认一直运行")
    parser.add_argument("--verbose", action="store_true", help="输出 DEBUG 日志")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> None:
    cfg = AppConfig.from_file(args.config)
    if args.replay:
        cfg.audio.backend = "replay"
        cfg.audio.replay_dir = args.replay
    configure_logging(cfg, verbose=args.verbose)
    stop = asyncio.Event()
    if args.duration is not None:
        asyncio.get_running_loop().call_later(args.duration, stop.set)
    await run_app(cfg, stop_event=stop)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:  # pragma: no cover - 交互式退出
        return 130
    except Exception as exc:  # noqa: BLE001 - 顶层入口给出可读错误
        print(f"aieyes 启动失败: {exc!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
