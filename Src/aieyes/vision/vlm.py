"""llama.cpp server 的 OpenAI 兼容 VLM 客户端（``/v1/chat/completions``）。

* 图像走 ``data:image/jpeg;base64,...``，先把长边缩到
  ``VlmConfig.image_max_side_px`` 再 JPEG 编码，控制请求体大小。
* 支持单图/多图（“双图比较”按顺序放两张）、可配超时与重试；
  失败统一抛 ``errors.InferenceError``，消息里带 HTTP 状态/耗时/尝试次数。
* 每次调用记录耗时、token 用量和 llama.cpp 的 ``timings``，供证据统计。
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import requests

from aieyes.config import LlmConfig, VlmConfig
from aieyes.errors import InferenceError

from . import image_ops

#: 与 VlmConfig 无冲突的默认值
DEFAULT_JPEG_QUALITY = 90
DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_BACKOFF_S = 0.5
#: 错误响应体保留的最大字符数
_ERROR_BODY_CHARS = 300


@dataclass(frozen=True)
class VlmCompletion:
    """一次 VLM 调用的完整结果（``text`` 之外还有证据统计字段）。"""

    text: str
    t_ms: int
    usage: dict[str, int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    model: str = ""
    images: int = 0
    attempts: int = 1
    prompt: str = ""

    @property
    def total_tokens(self) -> int:
        try:
            return int(self.usage.get("total_tokens", 0))
        except (TypeError, ValueError):
            return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "t_ms": self.t_ms,
            "usage": dict(self.usage),
            "timings": dict(self.timings),
            "model": self.model,
            "images": self.images,
            "attempts": self.attempts,
        }


class VlmClient:
    """实现 ``interfaces.Vlm`` 协议（``complete`` 返回文本）。"""

    def __init__(
        self,
        llm_config: LlmConfig | None = None,
        vlm_config: VlmConfig | None = None,
        *,
        session: requests.Session | None = None,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
        sleep: Callable[[float], None] = time.sleep,
        enable_thinking: bool | None = None,
    ) -> None:
        self.llm = llm_config or LlmConfig()
        self.vlm = vlm_config or VlmConfig()
        self._session = session or requests.Session()
        self._owns_session = session is None
        self.jpeg_quality = int(jpeg_quality)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._sleep = sleep
        # Qwen3.5 默认开启思考，32~512 token 会被 reasoning_content 吃光导致 content 为空；
        # 交互要的是直接答案，默认关闭。优先显式参数，其次 VlmConfig.enable_thinking（新字段）。
        if enable_thinking is None:
            enable_thinking = bool(getattr(self.vlm, "enable_thinking", False))
        self.enable_thinking = bool(enable_thinking)
        self.last_completion: VlmCompletion | None = None
        self.last_error: str = ""

    # --- 协议方法 ------------------------------------------------------
    def complete(
        self,
        prompt: str,
        *,
        images: Sequence[np.ndarray] = (),
        max_tokens: int = 512,
        timeout_s: float = 120.0,
        temperature: float | None = None,
    ) -> str:
        return self.complete_detailed(
            prompt,
            images=images,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            temperature=temperature,
        ).text

    def complete_detailed(
        self,
        prompt: str,
        *,
        images: Sequence[np.ndarray] = (),
        max_tokens: int | None = None,
        timeout_s: float | None = None,
        temperature: float | None = None,
    ) -> VlmCompletion:
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt)!r}")
        image_list = list(images)
        payload = self._build_payload(
            prompt,
            image_list,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        timeout = float(timeout_s if timeout_s is not None else self.vlm.timeout_s)
        if timeout <= 0:
            raise InferenceError(f"invalid timeout_s: {timeout}")
        t0 = time.monotonic()
        response, attempts = self._request(payload, timeout)
        elapsed_ms = int(round((time.monotonic() - t0) * 1000))
        try:
            body = response.json()
        except ValueError as exc:
            raise InferenceError(
                f"VLM response is not JSON (HTTP {response.status_code}, "
                f"{len(response.content or b'')} bytes, {elapsed_ms} ms): {exc}"
            ) from exc
        text, usage, timings = self._parse_body(body)
        completion = VlmCompletion(
            text=text,
            t_ms=elapsed_ms,
            usage=usage,
            timings=timings,
            model=self.llm.model,
            images=len(image_list),
            attempts=attempts,
            prompt=prompt,
        )
        self.last_completion = completion
        self.last_error = ""
        return completion

    def complete_stream(
        self,
        prompt: str,
        *,
        images: Sequence[np.ndarray | bytes] = (),
        max_tokens: int | None = None,
        timeout_s: float | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """流式生成：逐段 yield 增量文本（llama.cpp ``stream: true`` SSE）。

        用途：交互层拿到第一句就先播报，压低头声时延；失败统一抛
        ``InferenceError``，调用方可在拿到任何增量前回退到 ``complete``。
        """
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt)!r}")
        image_list = list(images)
        payload = self._build_payload(
            prompt, image_list, max_tokens=max_tokens, temperature=temperature
        )
        payload["stream"] = True
        timeout = float(timeout_s if timeout_s is not None else self.vlm.timeout_s)
        if timeout <= 0:
            raise InferenceError(f"invalid timeout_s: {timeout}")
        url = self._url("/v1/chat/completions")
        http_timeout = (float(self.llm.connect_timeout_s), timeout)
        try:
            resp = self._session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=http_timeout,
                stream=True,
            )
        except requests.Timeout as exc:
            raise InferenceError(f"VLM stream timed out after {timeout}s: {exc}") from exc
        except requests.RequestException as exc:
            raise InferenceError(
                f"VLM server unreachable at {self.llm.base_url}: {exc}"
            ) from exc
        try:
            if resp.status_code >= 400:
                body = (resp.text or "")[:_ERROR_BODY_CHARS]
                raise InferenceError(
                    f"VLM stream HTTP {resp.status_code} (url={url}): {body}"
                )
            for raw_line in resp.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                # llama.cpp 的 SSE 是 UTF-8；requests 在没有 charset 时会按 latin-1
                # 解码，中文会变乱码，这里统一按 UTF-8 解。
                if isinstance(raw_line, (bytes, bytearray)):
                    line = bytes(raw_line).decode("utf-8", "replace")
                else:
                    line = str(raw_line)
                line = line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                else:
                    continue
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    yield str(piece)
        finally:
            resp.close()

    def health(self, timeout_s: float | None = None) -> bool:
        """探测 llama.cpp server 是否就绪（``/health`` 返回 200 且非 loading）。"""
        url = self._url("/health")
        timeout = float(
            timeout_s if timeout_s is not None else self.llm.health_timeout_s
        )
        try:
            resp = self._session.get(
                url, timeout=(self.llm.connect_timeout_s, timeout)
            )
        except requests.RequestException as exc:
            self.last_error = f"health check failed: {exc}"
            return False
        if resp.status_code != 200:
            self.last_error = f"health HTTP {resp.status_code}"
            return False
        try:
            data = resp.json()
        except ValueError:
            return True
        status = str(data.get("status", "")).lower()
        return status in ("", "ok", "ready")

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    # --- 请求构造 ------------------------------------------------------
    def _url(self, path: str) -> str:
        return self.llm.base_url.rstrip("/") + path

    def _encode_image(self, image: np.ndarray | bytes) -> str:
        if isinstance(image, (bytes, bytearray, memoryview)):
            jpeg = bytes(image)
        else:
            rgb = image_ops.resize_max_side(image, self.vlm.image_max_side_px)
            jpeg = image_ops.encode_jpeg(rgb, quality=self.jpeg_quality)
        if not jpeg:
            raise InferenceError("empty image bytes")
        return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")

    def _build_payload(
        self,
        prompt: str,
        images: Sequence[np.ndarray | bytes],
        *,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        if isinstance(images, (str, bytes)):  # 防止把单张图当序列迭代
            raise TypeError("images must be a sequence of numpy images")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append(
                {"type": "image_url", "image_url": {"url": self._encode_image(image)}}
            )
        tokens = int(max_tokens) if max_tokens else int(self.llm.max_tokens)
        if tokens <= 0:
            tokens = int(self.vlm.max_tokens)
        # 纯文本对话用 LlmConfig 的默认温度，带图用 VlmConfig 的（可被参数覆盖）
        default_temp = self.vlm.temperature if images else self.llm.temperature
        temp = default_temp if temperature is None else float(temperature)
        return {
            "model": self.llm.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": tokens,
            "temperature": float(temp),
            "top_p": float(self.llm.top_p),
            "stream": False,
            # llama.cpp + Qwen3.5：关闭思考，否则短 max_tokens 只产出 reasoning_content
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

    # --- HTTP / 重试 / 解码 -------------------------------------------
    def _request(self, payload: dict[str, Any], timeout_s: float) -> tuple[requests.Response, int]:
        url = self._url("/v1/chat/completions")
        timeout = (float(self.llm.connect_timeout_s), float(timeout_s))
        attempts = self.max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                resp = self._session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )
            except requests.Timeout as exc:
                last_exc = InferenceError(
                    f"VLM request timed out after {timeout_s}s "
                    f"(attempt {attempt}/{attempts}, url={url}): {exc}"
                )
            except requests.RequestException as exc:
                last_exc = InferenceError(
                    f"VLM server unreachable at {self.llm.base_url} "
                    f"(attempt {attempt}/{attempts}): {exc}"
                )
            else:
                if resp.status_code < 400:
                    return resp, attempt
                body = (resp.text or "")[:_ERROR_BODY_CHARS]
                last_exc = InferenceError(
                    f"VLM server HTTP {resp.status_code} (attempt {attempt}/{attempts}): "
                    f"{body}"
                )
                # 4xx 是请求/参数问题，重试无意义；5xx/429 可重试
                if resp.status_code < 500 and resp.status_code != 429:
                    raise last_exc
            if attempt < attempts:
                self._sleep(self.retry_backoff_s * attempt)
        assert last_exc is not None
        self.last_error = str(last_exc)
        raise last_exc

    @staticmethod
    def _parse_body(body: Any) -> tuple[str, dict[str, int], dict[str, float]]:
        try:
            choice = body["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content")
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(
                f"unexpected VLM response shape: {str(body)[:_ERROR_BODY_CHARS]}"
            ) from exc
        if isinstance(content, list):  # 少数实现返回分段 content
            text = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type", "text") == "text"
            )
        else:
            text = "" if content is None else str(content)
        if not text.strip():
            reasoning = message.get("reasoning_content")
            finish = choice.get("finish_reason")
            hint = ""
            if isinstance(reasoning, str) and reasoning.strip():
                hint = (
                    f"; model produced {len(reasoning)} chars of reasoning_content and "
                    f"finish_reason={finish!r} — 思考占满了 max_tokens，请启用 "
                    f"enable_thinking=False 或增大 max_tokens"
                )
            raise InferenceError(
                f"VLM returned empty content{hint}: {str(body)[:_ERROR_BODY_CHARS]}"
            )
        usage_raw = body.get("usage") or {}
        usage: dict[str, int] = {}
        if isinstance(usage_raw, dict):
            for key, value in usage_raw.items():
                try:
                    usage[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
        timings_raw = body.get("timings") or {}
        timings: dict[str, float] = {}
        if isinstance(timings_raw, dict):
            for key, value in timings_raw.items():
                if isinstance(value, (int, float)):
                    timings[str(key)] = float(value)
        return text.strip(), usage, timings

    def __enter__(self) -> "VlmClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
