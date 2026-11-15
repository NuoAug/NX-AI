#!/usr/bin/env python3
"""scene 模块真机冒烟工具：真 OCR / 真 VLM / /dev/video4 取图，打印真实毫秒。

用法::

    trun .venv/bin/python Tools/scene_smoke.py --image Tests/Fixtures/images/text_cn_price.png \\
        --question "这个多少钱"
    trun .venv/bin/python Tools/scene_smoke.py --image Tests/Fixtures/images/text_cn_price.png \\
        --question "我的钥匙在哪" --find 钥匙
    trun .venv/bin/python Tools/scene_smoke.py --compare A.png B.png --question "哪个更贵"
    trun .venv/bin/python Tools/scene_smoke.py --camera-smoke

退出码：0 成功；2 参数/运行失败；3 相机被占用（EBUSY）。
"""

from __future__ import annotations

import argparse
import sys

from aieyes.scene import (
    CameraFrameProvider,
    FileFrameProvider,
    FrameAcquisitionError,
    FrameQualityError,
    SceneService,
)


def _print_answer(answer) -> None:
    print(f"intent      : {answer.intent}")
    print(f"text        : {answer.text}")
    print(f"frame_id    : {answer.frame_id or '(无帧)'}")
    print(f"confidence  : {answer.confidence:.2f}  clear={answer.clear}")
    print(f"degraded    : {', '.join(answer.degraded) or '(无)'}")
    print(f"timings(ms) : {answer.timings}")
    if answer.quality is not None:
        print(f"quality     : labels={answer.quality.labels} detail={answer.quality.detail}")
    print(f"ocr_lines   : {len(answer.ocr_lines)}")
    for index, line in enumerate(answer.ocr_lines[:8]):
        print(f"  [{index}] score={line.score:.2f} box={line.box} text={line.text!r}")
    print(f"source_tags : {len(answer.source_tags)}")
    for tag in answer.source_tags:
        print(f"  - {tag.label}({tag.kind}) field={tag.field or '-'} raw={tag.raw_text!r}")
    if answer.vlm_text:
        print(f"vlm_text    : {answer.vlm_text}")
    if answer.notes:
        print(f"notes       : {answer.notes}")


def _camera_smoke() -> int:
    provider = CameraFrameProvider()
    try:
        frame = provider.acquire("vqa")
    except FrameQualityError as exc:
        if exc.frame is None:
            print(f"相机取图失败：{exc}")
            return 3 if "EBUSY" in exc.cause.upper() else 2
        frame = exc.frame
        print(f"画面质量不达标，已取到最后一帧：{exc}")
    except FrameAcquisitionError as exc:
        print(f"相机不可用：{exc.cause or exc}")
        return 3 if "EBUSY" in exc.cause.upper() else 2

    assert provider.last_acquisition is not None
    acquisition = provider.last_acquisition
    print(
        f"相机冒烟 OK: {frame.width}x{frame.height} frame_id={frame.frame_id}"
        f" 尝试={len(acquisition.attempts)} 耗时={acquisition.elapsed_ms}ms"
    )
    for attempt in acquisition.attempts:
        labels = attempt.quality.labels if attempt.quality is not None else ()
        print(
            f"  - attempt{attempt.index}: {attempt.elapsed_ms}ms"
            f" frame={attempt.frame_id} labels={labels} error={attempt.error or '-'}"
        )
    return 0


def _find(image: str, target: str) -> int:
    frame = FileFrameProvider(image).acquire("vqa")
    service = SceneService()
    candidates = service.find_object(frame, target)
    print(f"target={candidates.target} found={candidates.found} degraded={candidates.degraded}")
    print(f"text   : {candidates.text}")
    for candidate in candidates.candidates:
        print(
            f"  - {candidate.label} pos={candidate.position} box={candidate.box}"
            f" conf={candidate.confidence:.2f} feature={candidate.feature!r}"
        )
    return 0


def _compare(a: str, b: str, question: str) -> int:
    service = SceneService()
    result = service.compare(a, b, question)
    print(f"text        : {result.text}")
    print(f"common      : {result.common_items}  missing: {result.missing}")
    print(f"degraded    : {', '.join(result.degraded) or '(无)'}")
    print(f"timings(ms) : {result.timings}")
    for item in result.items:
        print(
            f"  - {item.name}: A={item.a!r} B={item.b!r}"
            f" comparable={item.comparable} note={item.note!r}"
        )
    return 0


def _answer(image: str, question: str) -> int:
    provider = FileFrameProvider(image)
    service = SceneService(provider=provider)
    answer = service.answer(question)
    _print_answer(answer)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="scene 模块真机冒烟（OCR/VLM/相机）")
    parser.add_argument("--image", type=str, help="图片路径；不给则用相机")
    parser.add_argument("--question", type=str, default="看看画面里有什么", help="问题")
    parser.add_argument("--find", type=str, default="", help="找物目标（配合 --image）")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="双图比较")
    parser.add_argument("--camera-smoke", action="store_true", help="仅 /dev/video4 取图冒烟")
    args = parser.parse_args(argv)

    try:
        if args.camera_smoke:
            return _camera_smoke()
        if args.compare:
            return _compare(args.compare[0], args.compare[1], args.question)
        if args.find:
            if not args.image:
                parser.error("--find 需要同时给 --image")
            return _find(args.image, args.find)
        if args.image:
            return _answer(args.image, args.question)
        print("未给 --image，改为相机取图冒烟。")
        return _camera_smoke()
    except Exception as exc:  # noqa: BLE001 - 冒烟工具，打印完整原因
        print(f"运行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
