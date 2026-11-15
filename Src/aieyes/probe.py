"""阶段 0 工程与资产盘点探针。

只读取本机/目标板状态与模型文件，不打开声卡、不启动推理服务。
输出 JSON（默认写 evidence/probe.json 并打印摘要），供阶段 0 报告引用。

用法::

    python -m aieyes.probe --config Config/target.json --out /home/mylin/Desktop/Nuo/evidence/probe.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .clock import mono_ms, wall_iso
from .config import AppConfig
from .logging_setup import setup_logging

MIN_VLM_BYTES = 1_200_000_000
MIN_MMPROJ_BYTES = 600_000_000


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _file_facts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {"path": str(path), "exists": True, "bytes": st.st_size, "mtime": int(st.st_mtime)}


def _gguf_arch(path: Path) -> dict[str, Any]:
    """读取 GGUF 头部 magic/version 与 general.architecture（不加载权重）。"""
    out: dict[str, Any] = {"path": str(path)}
    try:
        with path.open("rb") as fh:
            head = fh.read(64 * 1024)
    except OSError as exc:  # pragma: no cover - 仅目标板文件缺失时
        out["error"] = repr(exc)
        return out
    if len(head) < 8 or head[:4] != b"GGUF":
        out["error"] = "not a GGUF file"
        return out
    out["magic"] = "GGUF"
    out["version"] = int.from_bytes(head[4:8], "little")
    marker = b"general.architecture"
    idx = head.find(marker)
    if idx < 0:
        out["architecture"] = None
        return out
    pos = idx + len(marker)
    # 跳过 value 类型 tag（uint32=4）
    if pos + 4 <= len(head):
        vtype = int.from_bytes(head[pos : pos + 4], "little")
        out["value_type"] = vtype
        pos += 4
    if pos + 8 <= len(head):
        n = int.from_bytes(head[pos : pos + 8], "little")
        pos += 8
        if 0 < n < 128 and pos + n <= len(head):
            out["architecture"] = head[pos : pos + n].decode("utf-8", "replace")
    return out


def _cuda_facts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    nvcc = shutil.which("nvcc")
    out["nvcc"] = nvcc or None
    for cand in ("/usr/local/cuda/version.json", "/usr/local/cuda/bin/nvcc"):
        if os.path.exists(cand):
            out["cuda_path"] = os.path.dirname(cand) if cand.endswith("nvcc") else os.path.dirname(cand)
            break
    try:
        stdout = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        out["nvidia_smi"] = stdout or None
    except (OSError, subprocess.SubprocessError) as exc:
        out["nvidia_smi_error"] = repr(exc)
    return out


def _system_facts(cfg: AppConfig) -> dict[str, Any]:
    mem: dict[str, int] = {}
    for line in (_read_text("/proc/meminfo") or "").splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable", "MemFree", "CmaTotal", "CmaFree"):
            try:
                mem[key] = int(rest.strip().split()[0]) * 1024
            except (IndexError, ValueError):
                pass
    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "distro": _read_text("/etc/os-release"),
        "l4t": _read_text("/etc/nv_tegra_release"),
        "meminfo_bytes": mem,
        "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "cuda": _cuda_facts(),
        "asound_cards": _read_text("/proc/asound/cards"),
        "video_devices": sorted(glob.glob("/dev/video*")),
        "disk_free_bytes": _disk_free(cfg),
    }


def _disk_free(cfg: AppConfig) -> int | None:
    root = Path(cfg.models.root)
    target = root if root.exists() else root.parent
    try:
        return shutil.disk_usage(str(target)).free
    except OSError:
        return None


def _models_facts(cfg: AppConfig) -> dict[str, Any]:
    root = Path(cfg.models.root)
    checks: dict[str, dict[str, Any]] = {}
    for key, rel in (
        ("vlm_gguf", cfg.models.vlm_gguf),
        ("vlm_mmproj", cfg.models.vlm_mmproj),
    ):
        checks[key] = _file_facts(root / rel)
    for key, rel, patterns in (
        ("asr", cfg.models.asr_dir, ("*.onnx", "tokens.txt")),
        ("tts", cfg.models.tts_dir, ("*.onnx", "*.onnx.json")),
        ("kws", cfg.models.kws_dir, ("*.onnx", "tokens.txt")),
        ("ocr", cfg.models.ocr_dir, ("*.onnx",)),
        ("speaker", str(Path(cfg.models.speaker_onnx).parent), ("*.onnx",)),
    ):
        base = root / rel
        found = []
        for pat in patterns:
            found.extend(str(Path(p).relative_to(root)) for p in glob.glob(str(base / pat)))
        checks[key] = {"dir": str(base), "exists": base.exists(), "files": sorted(set(found))}
    vlm_ready = (
        checks["vlm_gguf"].get("bytes", 0) >= MIN_VLM_BYTES
        and checks["vlm_mmproj"].get("bytes", 0) >= MIN_MMPROJ_BYTES
    )
    checks["vlm_ready"] = vlm_ready
    checks["gguf_vlm"] = _gguf_arch(root / cfg.models.vlm_gguf) if checks["vlm_gguf"].get("exists") else {}
    checks["gguf_mmproj"] = _gguf_arch(root / cfg.models.vlm_mmproj) if checks["vlm_mmproj"].get("exists") else {}
    return {"root": str(root), "checks": checks}


def _runtime_facts(cfg: AppConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import onnxruntime as ort  # type: ignore

        out["onnxruntime"] = ort.__version__
        out["onnxruntime_providers"] = ort.get_available_providers()
    except Exception as exc:  # noqa: BLE001 - 探针要如实报告
        out["onnxruntime_error"] = repr(exc)
    try:
        import sherpa_onnx  # type: ignore

        out["sherpa_onnx"] = getattr(sherpa_onnx, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        out["sherpa_onnx_error"] = repr(exc)
    try:
        import numpy as np  # type: ignore

        out["numpy"] = np.__version__
    except Exception as exc:  # noqa: BLE001
        out["numpy_error"] = repr(exc)
    try:
        import requests  # type: ignore

        out["requests"] = requests.__version__
        health = cfg.llm.base_url.rstrip("/") + "/health"
        try:
            resp = requests.get(health, timeout=3)
            out["llama_health"] = {"status": resp.status_code, "body": resp.text[:200]}
        except Exception as exc:  # noqa: BLE001
            out["llama_health"] = {"error": repr(exc)}
        models_url = cfg.llm.base_url.rstrip("/") + "/v1/models"
        try:
            resp = requests.get(models_url, timeout=5)
            out["llama_models"] = {"status": resp.status_code, "body": resp.text[:500]}
        except Exception as exc:  # noqa: BLE001
            out["llama_models"] = {"error": repr(exc)}
    except Exception as exc:  # noqa: BLE001
        out["requests_error"] = repr(exc)
    return out


def build_probe(cfg: AppConfig) -> dict[str, Any]:
    return {
        "kind": "phase0_probe",
        "aieyes_version": __version__,
        "wall_iso": wall_iso(),
        "mono_ms": mono_ms(),
        "system": _system_facts(cfg),
        "models": _models_facts(cfg),
        "runtime": _runtime_facts(cfg),
        "config": cfg.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 0 资产与运行环境探针")
    parser.add_argument("--config", default="Config/target.json")
    parser.add_argument("--out", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = AppConfig.from_file(args.config) if os.path.exists(args.config) else AppConfig()
    setup_logging(level="INFO", quiet_console=args.quiet)
    report = build_probe(cfg)

    out_path = args.out or str(Path(cfg.evidence_dir) / "phase0" / "probe.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    checks = report["models"]["checks"]
    summary = {
        "host": report["system"]["hostname"],
        "kernel": report["system"]["kernel"],
        "l4t": (report["system"]["l4t"] or "").splitlines()[0] if report["system"]["l4t"] else None,
        "mem_total_gib": round(report["system"]["meminfo_bytes"].get("MemTotal", 0) / 2**30, 2),
        "vlm_ready": checks.get("vlm_ready"),
        "vlm_arch": checks.get("gguf_vlm", {}).get("architecture"),
        "files": {k: len(v.get("files", [])) for k, v in checks.items() if isinstance(v, dict)},
        "onnxruntime": report["runtime"].get("onnxruntime"),
        "sherpa_onnx": report["runtime"].get("sherpa_onnx"),
        "llama_health": report["runtime"].get("llama_health"),
        "out": out_path,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
