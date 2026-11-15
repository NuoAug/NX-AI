"""统一日志：控制台 + JSONL 证据日志。

所有运行期事件（含时延与状态迁移）都写一份 JSONL 到 evidence 目录，
它是阶段验收报告里“证据”的主要来源，格式保持稳定：

    {"ts_mono_ms": 123456, "ts_iso": "...", "kind": "...", "level": "INFO", ...}

字段由调用方通过 ``extra=`` 传入，未知字段原样保留。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any


class JsonlEvidenceLogger:
    """追加写 JSONL，按天/按文件命名由调用方决定。线程安全。"""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:  # pragma: no cover - best effort
                pass

    def __enter__(self) -> "JsonlEvidenceLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _JsonlHandler(logging.Handler):
    def __init__(self, logger_: JsonlEvidenceLogger) -> None:
        super().__init__()
        self._logger = logger_

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - I/O
        try:
            payload: dict[str, Any] = {
                "ts_mono_ms": record.created and int(record.__dict__.get("ts_mono_ms", 0)),
                "ts_iso": self.format_time(record.created),
                "kind": "log",
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            payload.update(record.__dict__.get("extra_fields") or {})
            self._logger.write(payload)
        except Exception:
            self.handleError(record)

    @staticmethod
    def format_time(created: float) -> str:
        import datetime as _dt

        return _dt.datetime.fromtimestamp(created).astimezone().isoformat(
            timespec="milliseconds"
        )


def setup_logging(
    level: str = "INFO",
    evidence_path: str | os.PathLike[str] | None = None,
    *,
    quiet_console: bool = False,
) -> JsonlEvidenceLogger | None:
    """配置根 logger；返回证据 logger（若给了 evidence_path）。"""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    if not quiet_console:
        root.addHandler(console)

    evidence_logger: JsonlEvidenceLogger | None = None
    if evidence_path is not None:
        evidence_logger = JsonlEvidenceLogger(evidence_path)
        root.addHandler(_JsonlHandler(evidence_logger))
    return evidence_logger
