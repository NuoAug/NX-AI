"""阶段 6：interaction.v0.1 合同与交互状态机的桥接层。

本模块把合同层的入站消息（capabilities.get / interaction.status.get /
output.cancel / guidance.event）接到 ``InteractionEngine``，并把状态机的播放
状态与能力/健康快照按合同回给主控。合同层本身（schema/传输/handler 协议）
在 ``aieyes.contract`` 中，本模块只做适配，不重复校验。

诚实性约束：
- 只有合同层已完成握手（对端 clock_id 已知）才发 live 事实；软件回放路径的
  证据模式由配置决定（默认 synthetic）。
- ``start_basis`` 如实标注：真实 ALSA 播放写设备后才标 ``device_write``，
  测试/回放消费按 ``estimated``。
- 未实现的动作（如把本地相机换成主控取图）在 capabilities 的
  ``unimplemented`` 中列出，不伪装可用。
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

from ..clock import CLOCK_ID
from ..config import AppConfig
from ..contract.errors import EVIDENCE_MODES
from ..contract.handlers import (
    ContractHandler,
    HandlerFailure,
    InteractionContract,
    capabilities_result_data,
    cancel_result_data,
    interaction_status_data,
)
from ..contract.server import ContractServer
from ..errors import ContractError
from ..interfaces import ImageFrame
from ..vision import image_ops

__all__ = ["ContractBridge", "ContractLink", "load_contract_frame"]


class ContractBridge(ContractHandler):
    """把合同层入站请求转给状态机。"""

    def __init__(self, link: "ContractLink") -> None:
        self._link = link

    @property
    def engine(self) -> Any:
        return self._link.engine

    async def capabilities_get(self, request: Any) -> Mapping[str, Any]:
        return self.engine.contract_capabilities()

    async def interaction_status_get(self, request: Any) -> Mapping[str, Any]:
        return self.engine.contract_status()

    async def output_cancel(self, request: Any) -> Mapping[str, Any]:
        return await self.engine.contract_cancel(request)

    async def guidance_event(self, event: Any) -> None:
        await self.engine.handle_guidance_event(event)

    # 交互侧不会收到这些请求（服务端已按方向拒绝）；保留诚实报错以符合协议。
    async def frame_acquire(self, request: Any) -> Mapping[str, Any]:
        raise HandlerFailure("INVALID_PAYLOAD", "frame.acquire 是交互→主控方向")

    async def scene_query(self, request: Any) -> Mapping[str, Any]:
        raise HandlerFailure("INVALID_PAYLOAD", "scene.query 是交互→主控方向")

    async def control_intent(self, request: Any) -> Mapping[str, Any]:
        raise HandlerFailure("INVALID_PAYLOAD", "control.intent 是交互→主控方向")

    async def target_observation(self, request: Any) -> Mapping[str, Any]:
        raise HandlerFailure("INVALID_PAYLOAD", "target.observation 是交互→主控方向")


class ContractLink:
    """一个合同连接的生命周期 + 状态机出站调用。"""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        engine: Any,
        record: Callable[..., None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.engine = engine
        self._record = record or (lambda *a, **k: None)
        self.bridge = ContractBridge(self)
        evidence_mode = cfg.evidence_mode if cfg.evidence_mode in EVIDENCE_MODES else "synthetic"
        self.server = ContractServer(
            cfg.contract,
            handler=self.bridge,
            evidence_mode=evidence_mode,
            clock_id=CLOCK_ID,
            allow_test_fixture=True,
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect,
            max_message_bytes=cfg.contract.max_message_bytes,
        )
        self.contract = InteractionContract(self.server, default_timeout_ms=cfg.contract.response_timeout_ms)
        self.peer: dict[str, Any] | None = None
        self.peer_clock_ok = False

    # -- 生命周期 --------------------------------------------------------
    async def start(self) -> None:
        await self.server.start()
        self._record("contract", "listen", 0, host=self.cfg.contract.host, port=self.cfg.contract.port)

    async def stop(self) -> None:
        try:
            await self.server.stop()
        except Exception:  # noqa: BLE001 - 关闭期不阻塞退出
            pass

    async def _on_connect(self, epoch: int) -> None:
        self.peer = None
        self.peer_clock_ok = False
        self._record("contract", "connect", 0, epoch=epoch)
        asyncio.create_task(self._handshake(epoch))

    async def _on_disconnect(self, epoch: int) -> None:
        self._record("contract", "disconnect", 0, epoch=epoch, peer_role=(self.peer or {}).get("role"))
        self.peer = None
        self.peer_clock_ok = False

    async def _handshake(self, epoch: int) -> None:
        try:
            reply = await self.contract.query_capabilities()
        except Exception as exc:  # noqa: BLE001 - 握手失败只记录，不猜对端能力
            self._record("contract", "handshake_failed", 0, epoch=epoch, error=str(exc)[:120])
            return
        if not reply.ok:
            self._record("contract", "handshake_rejected", 0, epoch=epoch, code=(reply.error or {}).get("code"))
            return
        data = reply.data or {}
        peer_clock_id = getattr(reply, "clock_id", None)
        self.peer = dict(data)
        self.peer_clock_ok = bool(peer_clock_id) and peer_clock_id == CLOCK_ID
        self._record(
            "contract",
            "handshake",
            0,
            epoch=epoch,
            versions=list(data.get("supported_versions") or []),
            capabilities=list((data.get("capabilities") or {}).keys()),
            clock_id=peer_clock_id,
            clock_ok=self.peer_clock_ok,
        )
        if not self.peer_clock_ok:
            self._record("contract", "clock_mismatch", 0, peer_clock_id=peer_clock_id, clock_id=CLOCK_ID)

    # -- 出站 ------------------------------------------------------------
    async def publish_playback(self, **kwargs: Any) -> None:
        try:
            await self.contract.publish_playback_status(**kwargs)
        except Exception as exc:  # noqa: BLE001 - 主控未连接时由状态机继续本地流程
            self._record("contract", "publish_skipped", 0, state=kwargs.get("state"), error=str(exc)[:120])

    async def send_control_intent(self, **kwargs: Any):
        try:
            return await self.contract.send_control_intent(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ContractError("DEPENDENCY_UNAVAILABLE", f"control.intent 发送失败: {exc}") from exc

    async def query_scene(self, **kwargs: Any):
        return await self.contract.query_scene(**kwargs)

    async def observe_target(self, **kwargs: Any):
        return await self.contract.observe_target(**kwargs)

    async def acquire_frame(self, **kwargs: Any):
        return await self.contract.acquire_frame(**kwargs)


def load_contract_frame(message: Any) -> ImageFrame:
    """核验 frame.result 并解码资产为 ``ImageFrame``（sha256 一致才读）。"""
    if not getattr(message, "ok", False):
        error = (message.error or {}).get("code") if getattr(message, "error", None) else None
        raise ContractError(error or "NO_FRAME", "主控未返回可用帧")
    data = message.data or {}
    asset = data.get("asset") or {}
    path = asset.get("path")
    if not path:
        raise ContractError("NO_FRAME", "frame.result 缺 asset.path")
    file_path = Path(path)
    if not file_path.is_file():
        raise ContractError("NO_FRAME", f"帧资产不存在: {path}")
    raw = file_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    expected = str(asset.get("sha256") or "")
    if expected and len(expected) == 64 and actual.lower() != expected.lower():
        raise ContractError("INVALID_FRAME_REF", "帧资产 sha256 不匹配")
    rgb = image_ops.decode_image(raw)
    return ImageFrame(
        rgb=rgb,
        t_capture_ms=int(data.get("captured_mono_ms") or 0),
        frame_id=str(data.get("frame_id") or ""),
        source="contract",
    )
