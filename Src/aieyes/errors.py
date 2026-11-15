"""跨模块共享异常。"""

from __future__ import annotations


class AieyesError(Exception):
    """本系统错误的基类。"""


class ConfigError(AieyesError):
    pass


class ModelUnavailable(AieyesError):
    """模型文件缺失 / 加载失败 / 依赖未安装。"""


class AudioDeviceError(AieyesError):
    pass


class CameraUnavailable(AieyesError):
    pass


class VisionError(AieyesError):
    pass


class InferenceError(AieyesError):
    """LLM / OCR / ASR 推理失败的统一异常。"""


class ContractError(AieyesError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class SpeakerNotEnrolled(AieyesError):
    pass
