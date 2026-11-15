"""VlmClient 单测：本地 mock HTTP server 校验请求体/图片编码/超时/重试。

不依赖 llama.cpp server 与模型；真实 server 联调见 test_vlm_target.py。
"""

from __future__ import annotations

import base64
import io
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from aieyes.config import LlmConfig, VlmConfig
from aieyes.errors import InferenceError
from aieyes.vision import VlmClient


def _image(width=2000, height=1000) -> np.ndarray:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    return rgb


def _ok_body(text="图中文字是 12.50", *, usage=None, timings=None):
    return {
        "id": "chatcmpl-mock",
        "model": "qwen35",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage
        or {"prompt_tokens": 100, "completion_tokens": 8, "total_tokens": 108},
        "timings": timings or {"prompt_ms": 12.0, "predicted_ms": 30.0},
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # 静默
        pass

    def _send(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):  # noqa: N802
        self.server.requests.append({"method": "GET", "path": self.path, "body": None})
        if self.path == "/health":
            self._send(200, json.dumps({"status": "ok"}).encode())
        else:
            self._send(404, b"{}")

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        self.server.requests.append(
            {"method": "POST", "path": self.path, "body": body, "raw_len": len(raw)}
        )
        step = self.server.plan.pop(0) if self.server.plan else {"status": 200, "json": _ok_body()}
        if step.get("delay_s"):
            time.sleep(float(step["delay_s"]))
        status = int(step.get("status", 200))
        if "raw" in step:
            payload = step["raw"]
        else:
            payload = json.dumps(step.get("json", {})).encode()
        self._send(status, payload)


@pytest.fixture()
def mock_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    server.requests = []
    server.plan = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _client(server, **kwargs) -> VlmClient:
    config = LlmConfig(base_url=f"http://127.0.0.1:{server.server_address[1]}")
    return VlmClient(config, VlmConfig(), **kwargs)


def _decode_data_url(url: str) -> Image.Image:
    assert url.startswith("data:image/jpeg;base64,"), url[:40]
    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw[:2] == b"\xff\xd8", "不是 JPEG 数据"
    return Image.open(io.BytesIO(raw))


def test_request_body_text_only(mock_server):
    client = _client(mock_server, max_retries=0)
    text = client.complete("介绍一下你自己")
    assert text == "图中文字是 12.50"
    assert len(mock_server.requests) == 1
    req = mock_server.requests[0]
    assert req["method"] == "POST" and req["path"] == "/v1/chat/completions"
    body = req["body"]
    assert body["model"] == "qwen35"
    assert body["stream"] is False
    assert body["messages"][0]["role"] == "user"
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "介绍一下你自己"}
    assert len(content) == 1
    assert body["max_tokens"] == 512
    assert body["temperature"] == pytest.approx(0.3)  # 纯文本用 LlmConfig.temperature
    assert body["top_p"] == pytest.approx(0.9)
    # Qwen3.5 默认思考会吃掉 max_tokens，客户端默认显式关闭
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_request_body_image_uses_vlm_temperature(mock_server):
    client = _client(mock_server, max_retries=0)
    client.complete("这是什么", images=[_image(320, 240)])
    body = mock_server.requests[0]["body"]
    assert body["temperature"] == pytest.approx(VlmConfig().temperature)  # 带图 0.2


def test_request_body_two_images_and_resize(mock_server):
    client = _client(mock_server, max_retries=0)
    client.complete(
        "比较这两张图",
        images=[_image(), _image(400, 300)],
        max_tokens=64,
        temperature=0.0,
    )
    body = mock_server.requests[0]["body"]
    content = body["messages"][0]["content"]
    assert [item["type"] for item in content] == ["text", "image_url", "image_url"]
    assert body["max_tokens"] == 64
    assert body["temperature"] == pytest.approx(0.0)
    for item in content[1:]:
        image = _decode_data_url(item["image_url"]["url"])
        assert max(image.size) <= VlmConfig().image_max_side_px
    first = _decode_data_url(content[1]["image_url"]["url"])
    assert first.size == (1024, 512)  # 2000x1000 等比缩到长边 1024
    second = _decode_data_url(content[2]["image_url"]["url"])
    assert second.size == (400, 300)  # 已足够小，不放大
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_enable_thinking_can_be_overridden(mock_server):
    client = _client(mock_server, max_retries=0, enable_thinking=True)
    assert client.enable_thinking is True
    client.complete("hi")
    assert mock_server.requests[0]["body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }


def test_enable_thinking_from_vlm_config(mock_server):
    config = LlmConfig(base_url=f"http://127.0.0.1:{mock_server.server_address[1]}")
    client = VlmClient(config, VlmConfig(enable_thinking=True), max_retries=0)
    assert client.enable_thinking is True
    client.complete("hi")
    assert mock_server.requests[0]["body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert VlmConfig().enable_thinking is False  # 默认关闭思考


def test_empty_content_with_reasoning_gives_actionable_error(mock_server):
    mock_server.plan = [
        {
            "status": 200,
            "json": {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "用户问颜色，我需要先看图……",
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {"total_tokens": 32},
            },
        }
    ]
    client = _client(mock_server, max_retries=0)
    with pytest.raises(InferenceError) as excinfo:
        client.complete("图中是什么颜色", images=[_image(320, 240)], max_tokens=32)
    message = str(excinfo.value)
    assert "reasoning_content" in message and "enable_thinking=False" in message
    assert "finish_reason='length'" in message
    assert client.last_completion is None


def test_detailed_reports_usage_and_timings(mock_server):
    client = _client(mock_server, max_retries=0)
    completion = client.complete_detailed("图里是什么", images=[_image(320, 240)])
    assert completion.text == "图中文字是 12.50"
    assert completion.attempts == 1
    assert completion.images == 1
    assert completion.usage["total_tokens"] == 108
    assert completion.total_tokens == 108
    assert completion.timings["prompt_ms"] == 12.0
    assert completion.t_ms >= 0
    assert completion.to_dict()["usage"]["total_tokens"] == 108
    assert client.last_completion is completion


def test_health_check(mock_server):
    client = _client(mock_server)
    assert client.health() is True
    assert mock_server.requests[0]["path"] == "/health"


def test_retry_on_500_then_success(mock_server):
    mock_server.plan = [
        {"status": 500, "json": {"error": "boom"}},
        {"status": 200, "json": _ok_body("重试成功")},
    ]
    client = _client(mock_server, max_retries=1, retry_backoff_s=0.0)
    completion = client.complete_detailed("hi")
    assert completion.text == "重试成功"
    assert completion.attempts == 2
    assert len(mock_server.requests) == 2


def test_no_retry_on_400(mock_server):
    mock_server.plan = [{"status": 400, "json": {"error": "bad request"}}]
    client = _client(mock_server, max_retries=2, retry_backoff_s=0.0)
    with pytest.raises(InferenceError) as excinfo:
        client.complete("hi")
    assert "HTTP 400" in str(excinfo.value)
    assert len(mock_server.requests) == 1


def test_timeout_raises_and_retries(mock_server):
    mock_server.plan = [
        {"delay_s": 0.5, "json": _ok_body()},
        {"delay_s": 0.5, "json": _ok_body()},
    ]
    client = _client(mock_server, max_retries=1, retry_backoff_s=0.0)
    t0 = time.monotonic()
    with pytest.raises(InferenceError) as excinfo:
        client.complete("hi", timeout_s=0.12)
    elapsed = time.monotonic() - t0
    assert "timed out" in str(excinfo.value)
    assert elapsed < 3.0
    assert len(mock_server.requests) == 2


def test_unreachable_server_raises():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    client = VlmClient(
        LlmConfig(base_url=f"http://127.0.0.1:{port}", connect_timeout_s=0.5),
        VlmConfig(),
        max_retries=0,
    )
    with pytest.raises(InferenceError) as excinfo:
        client.complete("hi")
    assert "unreachable" in str(excinfo.value)
    assert not client.health(timeout_s=0.2)


def test_invalid_json_response(mock_server):
    mock_server.plan = [{"status": 200, "raw": b"this is not json"}]
    client = _client(mock_server, max_retries=0)
    with pytest.raises(InferenceError):
        client.complete("hi")


def test_empty_content_response(mock_server):
    mock_server.plan = [{"status": 200, "json": _ok_body("   ")}]
    client = _client(mock_server, max_retries=0)
    with pytest.raises(InferenceError) as excinfo:
        client.complete("hi")
    assert "empty content" in str(excinfo.value)


def test_missing_choices_response(mock_server):
    mock_server.plan = [{"status": 200, "json": {"id": "x"}}]
    client = _client(mock_server, max_retries=0)
    with pytest.raises(InferenceError):
        client.complete("hi")
