"""运行配置：JSON → 嵌套 dataclass。

配置文件只描述“部署差异”（模型路径、设备名、端口、计时预算），
不放业务逻辑。所有字段都有可用默认值，缺省路径按目标机
``/home/mylin/Desktop/Nuo`` 部署布局给出。
"""

from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AudioConfig:
    """音频输入输出。

    ``backend="replay"`` 是阶段 0 的离线文件回放入口：不打开声卡，
    由 ``aieyes.replay`` 按时间戳喂 WAV，验证整条软件链路。
    """

    backend: str = "alsa"  # "alsa" | "replay"
    input_device: str | None = None  # 声卡名或 "default"；None=PortAudio 默认
    output_device: str | None = None
    sample_rate: int = 16000
    frame_ms: int = 20
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0
    replay_dir: str = ""  # 回放会话目录：<replay_dir>/*.wav 按文件名排序
    replay_realtime: bool = True
    aec_enabled: bool = False
    aec_filter_ms: int = 200  # 自适应滤波器长度


@dataclass
class ModelsConfig:
    """模型资产根目录与相对路径（由 Tools/upload_models.sh 上传）。"""

    root: str = "/home/mylin/Desktop/Nuo/models"
    vlm_gguf: str = "vlm/Qwen3.5-2B-Q4_K_M.gguf"
    vlm_mmproj: str = "vlm/mmproj-F16.gguf"
    asr_dir: str = "asr-streaming/sherpa-onnx-streaming-paraformer-bilingual-zh-en"
    tts_dir: str = "tts/piper-zh_CN-huayan-medium"
    kws_dir: str = "kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    ocr_dir: str = "ocr/pp-ocrv6-small"
    speaker_onnx: str = (
        "speaker/campplus-zh-cn-16k/"
        "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
    )

    def resolve(self, rel: str) -> Path:
        return Path(self.root) / rel


@dataclass
class LlmConfig:
    """llama.cpp server（OpenAI 兼容接口）的连接与生成参数。"""

    base_url: str = "http://127.0.0.1:8080"
    model: str = "qwen35"
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 512
    connect_timeout_s: float = 5.0
    request_timeout_s: float = 120.0
    health_timeout_s: float = 3.0


@dataclass
class KwsConfig:
    """唤醒词（sherpa-onnx KWS zipformer）。"""

    keywords_file: str = ""  # 空=用模型目录内 keywords.txt
    wake_phrases: list[str] = field(default_factory=lambda: ["小冰"])
    keywords_score: float = 3.0
    keywords_threshold: float = 0.25
    num_trailing_blanks: int = 2
    cooldown_ms: int = 800  # 同一唤醒词去抖


@dataclass
class VadConfig:
    threshold: float = 0.5
    min_speech_ms: int = 200
    min_silence_ms: int = 500
    max_utterance_ms: int = 40000  # T02：单段收音 ≤40s
    preroll_ms: int = 300  # 首字保护：VAD 命中前保留的音频
    initial_wait_ms: int = 10000  # T01：唤醒后初始等待


@dataclass
class AsrConfig:
    sample_rate: int = 16000
    feature_dim: int = 80
    decoding_method: str = "greedy_search"
    model_type: str = "paraformer"
    reset_on_silence_ms: int = 600


@dataclass
class TtsConfig:
    voice: str = "zh_CN-huayan-medium"
    speaker_id: int = 0
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    max_segment_chars: int = 60  # 长文本按标点切段，保证首声时延


@dataclass
class VisionConfig:
    device: str = "/dev/video0"
    width: int = 640
    height: int = 480
    fps: int = 30
    pixel_format: str = "auto"  # "auto" | "yuyv" | "mjpeg"
    warmup_frames: int = 3
    grab_timeout_ms: int = 10000  # T09
    max_side_px: int = 1280
    jpeg_quality: int = 90
    backend: str = "camera"  # "camera"（目标板相机）| "file"（离线文件回放）
    file_source: str = ""  # backend=file 时的图片路径或目录


@dataclass
class OcrConfig:
    det_model: str = "PP-OCRv6_det_small.onnx"
    rec_model: str = "PP-OCRv6_rec_small.onnx"
    cls_model: str = "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
    det_limit_side_len: int = 960
    det_thresh: float = 0.3
    det_box_thresh: float = 0.5
    det_unclip_ratio: float = 1.5
    rec_score_thresh: float = 0.5
    cls_enabled: bool = True
    cls_thresh: float = 0.9
    min_confidence: float = 0.5


@dataclass
class VlmConfig:
    max_tokens: int = 512
    timeout_s: float = 120.0
    image_max_side_px: int = 1024
    temperature: float = 0.2
    #: Qwen3.5 默认开启思考，短 max_tokens 会被 reasoning_content 吃光；交互要直接答案，默认关。
    enable_thinking: bool = False


@dataclass
class SpeakerConfig:
    """声纹（campplus embedding）。"""

    enabled: bool = True
    threshold: float = 0.65  # 余弦相似度阈值，按阶段5实测调整
    enroll_dir: str = "speaker/enroll"
    min_speech_ms: int = 800
    max_samples: int = 5


@dataclass
class ContractConfig:
    """interaction.v0.1 传输端点（主控 ↔ 本交互子系统）。"""

    enabled: bool = False  # 阶段6 之前默认关闭
    host: str = "127.0.0.1"
    port: int = 18700
    role: str = "interaction"
    instance_id: str = "aieyes-01"
    max_message_bytes: int = 4 * 1024 * 1024
    response_timeout_ms: int = 10000  # T15
    accept_timeout_s: float = 30.0


@dataclass
class TimersConfig:
    """T01–T21 计时预算（与方案文档第三节一致，单位 ms/s 见字段名）。"""

    # T01 唤醒后初始等待
    t01_initial_wait_ms: int = 10000
    t01_remind_at_ms: int = 7000
    t01_extend_ms: int = 15000
    t01_max_extend: int = 3
    t01_min_gap_ms: int = 8000
    # T02 单段收音
    t02_max_utterance_ms: int = 40000
    t02_asr_timeout_ms: int = 10000
    t02_asr_retry: int = 1
    t02_silence_fallback_ms: int = 5000
    t02_max_retry_say: int = 2
    # T03 缺项澄清
    t03_clarify_rounds: int = 2
    t03_confirm_rounds: int = 1
    t03_round_ms: int = 15000
    # T05 准备阶段
    t05_prep_ms: int = 60000
    t05_prep_warn_ms: int = 20000
    t05_prep_remind_ms: int = 45000
    # T06 额外等待
    t06_extra_wait_ms: int = 120000
    t06_warn_ms: int = 30000
    # T07 跨来源比较
    t07_cross_source_ms: int = 180000
    # T08 双图采集累计
    t08_dual_capture_ms: int = 180000
    # T09 单次取图
    t09_single_grab_ms: int = 10000
    # T10 取图次数
    t10_max_grab_per_position: int = 3
    # T11 生成/分析
    t11_generate_ms: int = 60000
    t11_retry_delay_ms: int = 2000
    t11_retry_max: int = 1
    # T12 TTS/播放
    t12_tts_segment_ms: int = 15000
    t12_play_extra_ms: int = 5000
    # T13 新问题恢复
    t13_resume_ms: int = 120000
    t13_warn_ms: int = 90000
    # T14 设备恢复
    t14_device_recover_ms: int = 30000
    # T15 主控响应
    t15_control_ms: int = 10000
    # T17/T18 答完/停止后倾听
    t17_post_answer_listen_ms: int = 10000
    t18_post_stop_listen_ms: int = 10000
    # T19 免唤醒候选
    t19_candidate_ms: int = 40000
    t19_extend_ms: int = 21000
    t19_final_ms: int = 10000
    t19_reentry_ms: int = 15000
    t19_prompt_cooldown_ms: int = 30000
    # T20/T21 长窗口
    t20_image_task_ms: int = 900000
    t20_evidence_group_ms: int = 900000
    t21_topic_window_ms: int = 900000


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    kws: KwsConfig = field(default_factory=KwsConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    vlm: VlmConfig = field(default_factory=VlmConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)
    timers: TimersConfig = field(default_factory=TimersConfig)

    evidence_dir: str = "/home/mylin/Desktop/Nuo/evidence"
    log_dir: str = "/home/mylin/Desktop/Nuo/logs"
    evidence_mode: str = "live"  # live | replay | synthetic
    session_id_prefix: str = "aieyes"
    language: str = "zh-CN"

    # --- loading -------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"config root must be an object: {path}")
        return _load_dataclass(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def dump_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def _load_dataclass(cls: type, data: dict[str, Any]) -> Any:
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints.get(f.name), data[f.name])
    return cls(**kwargs)


def _coerce(hint: Any, value: Any) -> Any:
    if value is None or hint is None:
        return value
    if dataclasses.is_dataclass(hint) and isinstance(value, dict):
        return _load_dataclass(hint, value)
    origin = typing.get_origin(hint)
    if origin is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return _coerce(args[0], value)
        return value
    if origin in (list, tuple) and isinstance(value, (list, tuple)):
        return value
    return value
