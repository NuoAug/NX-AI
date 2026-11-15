"""vision 子包：相机取图、OCR、视觉语言模型客户端、图像质量与几何。

- :class:`V4L2Camera` —— V4L2 彩色相机（fcntl ioctl + mmap）
- :class:`FileCamera` —— 文件图源（阶段 0 离线评测 / 双图与翻面测试）
- :class:`OcrEngine` —— PP-OCRv6 small（DB + 方向分类 + CTC）
- :class:`VlmClient` —— llama.cpp server OpenAI 兼容 VLM 客户端
- :mod:`quality` / :mod:`image_ops` —— 质量检查与图像几何/颜色空间工具
"""

from .camera import V4L2Camera, probe_video_devices
from .file_camera import FileCamera, list_images, load_frame
from .image_ops import (
    bgr_to_rgb,
    crop_roi,
    decode_image,
    encode_jpeg,
    ensure_rgb,
    image_size,
    load_image,
    resize_max_side,
    resize_to,
    rgb_to_bgr,
    save_image,
    to_gray,
    to_gray_float,
    yuyv_to_rgb,
)
from .ocr import OcrEngine
from .quality import QualityReport, check_quality, laplacian_variance
from .vlm import VlmClient, VlmCompletion

__all__ = [
    "FileCamera",
    "OcrEngine",
    "QualityReport",
    "V4L2Camera",
    "VlmClient",
    "VlmCompletion",
    "bgr_to_rgb",
    "check_quality",
    "crop_roi",
    "decode_image",
    "encode_jpeg",
    "ensure_rgb",
    "image_size",
    "laplacian_variance",
    "list_images",
    "load_frame",
    "load_image",
    "probe_video_devices",
    "resize_max_side",
    "resize_to",
    "rgb_to_bgr",
    "save_image",
    "to_gray",
    "to_gray_float",
    "yuyv_to_rgb",
]
