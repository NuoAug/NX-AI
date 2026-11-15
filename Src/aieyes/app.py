"""应用装配：把音频前端、语音组件、状态机、模型客户端装成一个可运行进程。

阶段 1/2 用回放或真实声卡运行；视觉/合同能力在后续阶段按需接入。
"""

from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path
from typing import Any

from .chat import ChatSession
from .clock import wall_iso
from .config import AppConfig
from .events import EventBus
from .interaction.engine import InteractionEngine
from .logging_setup import JsonlEvidenceLogger, setup_logging

log = logging.getLogger("aieyes.app")


def _make_source(cfg: AppConfig) -> Any:
    """按配置创建音频输入源（回放 / 真实声卡）。"""
    if cfg.audio.backend == "replay":
        from .audio.replay_source import ReplaySource, build_replay_playlist

        replay_dir = Path(cfg.audio.replay_dir)
        files = sorted(replay_dir.glob("*.wav")) if replay_dir.is_dir() else []
        if not files:
            raise FileNotFoundError(f"回放目录没有 wav 文件: {replay_dir}")
        playlist = build_replay_playlist(files, gap_ms=600)
        return ReplaySource(
            playlist,
            sample_rate=cfg.audio.sample_rate,
            frame_ms=cfg.audio.frame_ms,
            realtime=cfg.audio.replay_realtime,
        )
    from .audio.capture import SoundDeviceSource

    return SoundDeviceSource(cfg.audio, device=cfg.audio.input_device)


def _make_identity(cfg: AppConfig) -> Any | None:
    """阶段 5：声纹门 + 话题窗口/资料权限；未注册声纹时返回 None（不伪报能力）。"""
    if not cfg.speaker.enabled:
        return None
    from .identity import IdentityManager, SpeakerGate

    gate = SpeakerGate(models_root=cfg.models.root, cfg=cfg.speaker)
    try:
        count = gate.enroll_models_dir()
    except Exception:  # noqa: BLE001 - 声纹模型缺失时不阻塞主链路
        log.warning("speaker enrollment failed", exc_info=True)
        count = 0
    if count <= 0:
        log.warning(
            "未找到已注册声纹（%s）；阶段 5 免唤醒/资料权限不启用",
            cfg.speaker.enroll_dir,
        )
        gate.close()
        return None
    log.info("声纹档案已加载：%d 条", count)
    return IdentityManager(gate, cfg.timers, topic_window_ms=cfg.timers.t21_topic_window_ms)


def _make_scene(cfg: AppConfig) -> tuple[Any, Any | None]:
    """按 ``vision.backend`` 创建场景问答服务；返回 (scene, 需外部关闭的 provider)。"""
    from .scene import FileFrameProvider, SceneService

    backend = (cfg.vision.backend or "camera").strip().lower()
    if backend == "file":
        source = (cfg.vision.file_source or "").strip()
        if not source:
            raise FileNotFoundError("vision.backend=file 但未配置 vision.file_source")
        provider = FileFrameProvider(source)
        # 注入的 provider 不由 SceneService 释放，交给调用方关闭
        return SceneService(provider=provider), provider
    if backend == "camera":
        return SceneService(), None
    raise ValueError(f"未知 vision.backend: {cfg.vision.backend!r}（可用 camera/file）")


def build_engine(
    cfg: AppConfig,
    *,
    evidence: JsonlEvidenceLogger | None = None,
    chat: ChatSession | None = None,
    scene: Any | None = None,
    identity: Any | None = None,
    sink: Any | None = None,
    tts: Any | None = None,
    frontend: Any | None = None,
    bus: EventBus | None = None,
) -> tuple[InteractionEngine, dict[str, Any]]:
    """装配交互引擎及其默认依赖；测试可注入替身。"""
    bus = bus or EventBus()
    scene_provider: Any | None = None
    if chat is None:
        from .vision.vlm import VlmClient

        chat = ChatSession(VlmClient(cfg.llm, cfg.vlm))
    if scene is None:
        scene, scene_provider = _make_scene(cfg)
    if identity is None:
        identity = _make_identity(cfg)
    if tts is None:
        from .speech.tts import PiperTts

        tts = PiperTts(cfg.tts)
    if sink is None:
        from .audio.output import AudioSink

        sink = AudioSink(sample_rate=cfg.audio.sample_rate)
    if frontend is None:
        from .audio.vad import EnergyVad
        from .speech.asr import StreamingAsr
        from .speech.frontend import SpeechFrontend
        from .speech.kws import SherpaKws

        frontend = SpeechFrontend(
            _make_source(cfg),
            kws=SherpaKws(cfg.kws),
            vad=EnergyVad(cfg.vad),
            asr=StreamingAsr(cfg.asr),
        )
    engine = InteractionEngine(
        cfg=cfg,
        bus=bus,
        frontend=frontend,
        sink=sink,
        tts=tts,
        chat=chat,
        scene=scene,
        identity=identity,
        evidence=evidence,
    )
    contract: Any | None = None
    if cfg.contract.enabled:
        from .interaction.contract_link import ContractLink

        contract = ContractLink(cfg, engine=engine, record=engine._record)
        engine.attach_contract(contract)
    deps = {
        "bus": bus,
        "frontend": frontend,
        "sink": sink,
        "tts": tts,
        "chat": chat,
        "scene": scene,
        "scene_provider": scene_provider,
        "identity": identity,
        "contract": contract,
    }
    return engine, deps


async def run_app(cfg: AppConfig, *, stop_event: asyncio.Event | None = None) -> None:
    """启动完整交互循环，直到收到停止信号（默认 Ctrl+C/SIGTERM）。"""
    evidence_path = Path(cfg.evidence_dir) / "interaction.jsonl"
    evidence = JsonlEvidenceLogger(evidence_path)
    evidence.write({"kind": "startup", "ts_iso": wall_iso(), "config": cfg.to_dict()})
    engine, deps = build_engine(cfg, evidence=evidence)
    frontend = deps["frontend"]
    sink = deps["sink"]
    tts = deps["tts"]
    scene = deps["scene"]
    scene_provider = deps["scene_provider"]
    identity = deps["identity"]
    contract = deps.get("contract")

    stop = stop_event or asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - Windows
            pass

    sink.start()
    try:
        await engine.start()
        if contract is not None:
            await contract.start()
        # 引擎就绪（已进入待机、问候音已预合成）后再开始采集，避免唤醒事件
        # 与启动阶段的状态初始化竞争。
        frontend.start()
        log.info("aieyes 交互已启动：模式=%s 证据=%s", cfg.audio.backend, evidence_path)
        await stop.wait()
    finally:
        log.info("aieyes 正在停止")
        if contract is not None:
            await contract.stop()
        await engine.stop()
        try:
            frontend.stop()
        finally:
            try:
                sink.close()
            finally:
                tts.close()
                try:
                    scene.close()
                finally:
                    if scene_provider is not None:
                        scene_provider.close()
                if identity is not None:
                    gate = getattr(identity, "gate", None)
                    close_gate = getattr(gate, "close", None)
                    if callable(close_gate):
                        close_gate()
                evidence.write({"kind": "shutdown", "ts_iso": wall_iso()})
                log.info("aieyes 已停止")


def configure_logging(cfg: AppConfig, *, verbose: bool = False) -> None:
    """初始化控制台 + JSONL 证据日志。"""
    setup_logging(
        level=logging.DEBUG if verbose else logging.INFO,
        evidence_path=Path(cfg.log_dir) / "runtime.jsonl",
        quiet_console=False,
    )
