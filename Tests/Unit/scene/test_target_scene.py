"""目标板真机测试：真实 OCR/VLM 与 /dev/video4 取图冒烟（pytest -m target）。

无 llama-server 或相机被占用时 skip，不阻塞普通单测。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aieyes.scene import (
    CameraFrameProvider,
    FileFrameProvider,
    FrameAcquisitionError,
    FrameQualityError,
    SceneService,
)
from aieyes.vision import VlmClient, load_frame

pytestmark = pytest.mark.target

ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = ROOT / "Tests" / "Fixtures" / "images"
SCENE_DIR = ROOT / "Tests" / "Fixtures" / "scene"


def _require_vlm() -> None:
    if not VlmClient().health(timeout_s=5.0):
        pytest.skip("llama-server 不可用，跳过真机 VLM 测试")


def test_target_real_ocr_vlm_price():
    _require_vlm()
    path = IMAGE_DIR / "text_cn_price.png"
    provider = FileFrameProvider(path)
    service = SceneService(provider=provider)
    answer = service.answer("这个多少钱")

    assert answer.intent == "price"
    assert "12元5角" in answer.text, answer.text
    assert answer.confidence >= 0.8
    assert "ocr" in answer.timings and "total" in answer.timings
    print(
        "\n[scene] 真机 answer price:"
        f" text={answer.text!r} frame={answer.frame_id}"
        f" timings={answer.timings}ms confidence={answer.confidence:.2f}"
    )


def test_target_real_read_text():
    path = IMAGE_DIR / "text_cn_price.png"
    frame = load_frame(path, source="file")
    service = SceneService()
    lines = service.read_text(frame)

    assert lines
    joined = "".join(line.text for line in lines)
    assert "AI_EYES" in joined or "12.50" in joined, joined
    for line in lines:
        assert line.box[2] > 0 and line.box[3] > 0
    print(f"\n[scene] 真机 read_text: {len(lines)} 行 / {joined!r}")


def test_target_real_date_fixture():
    _require_vlm()
    path = SCENE_DIR / "date_label.png"
    if not path.is_file():
        pytest.skip(f"缺少夹具 {path}")
    service = SceneService(provider=FileFrameProvider(path))
    answer = service.answer("什么时候过期")

    assert answer.intent == "date"
    assert "2026年12月1日" in answer.text, answer.text
    assert any(tag.field == "expiry_date" for tag in answer.source_tags)
    print(f"\n[scene] 真机日期: {answer.text!r} timings={answer.timings}ms")


def test_target_camera_smoke():
    provider = CameraFrameProvider()
    try:
        frame = provider.acquire("vqa")
    except FrameQualityError as exc:  # 取到了但质量差，仍算冒烟成功
        if exc.frame is None:
            provider.close()
            pytest.skip(f"相机画面不可用：{exc}")
        frame = exc.frame
    except FrameAcquisitionError as exc:
        provider.close()
        if "EBUSY" in exc.cause.upper():
            pytest.skip(f"/dev/video4 被占用：{exc.cause}")
        pytest.skip(f"相机不可用：{exc.cause or exc}")
    finally:
        # 及时释放 /dev/video4，避免影响后续相机测试
        provider.close()

    assert frame.rgb.ndim == 3 and frame.rgb.shape[2] == 3
    assert frame.width > 0 and frame.height > 0
    assert provider.last_acquisition is not None
    print(
        f"\n[scene] 相机冒烟: {frame.frame_id} {frame.width}x{frame.height}"
        f" attempts={len(provider.last_acquisition.attempts)}"
        f" quality={provider.last_acquisition.quality.labels if provider.last_acquisition.quality else ()}"
    )
