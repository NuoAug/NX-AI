"""scene 层公开数据类型：取图记录、来源标签、答案、找物候选、比较结果。

所有“答案”都带可反查字段：来源帧 ``frame_id``、OCR 行（含原图 box/polygon）、
来源标签（口述/包装/画面/历史）与质量报告，便于上层写入证据链。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aieyes.interfaces import ImageFrame, OcrLine
from aieyes.vision.quality import QualityReport

#: 来源标签的中文说法（播报时“听得出”来自哪里）
SOURCE_LABELS: dict[str, str] = {
    "spoken": "口述",
    "ocr": "包装",
    "vlm": "画面",
    "history": "历史",
    "computed": "计算",
}


@dataclass(frozen=True, slots=True)
class SourceTag:
    """一条答案依据的来源标注。

    ``kind``: ``spoken``（用户口述）/ ``ocr``（画面文字，通常来自包装或标签）/
    ``vlm``（画面观察）/ ``history``（历史事实）/ ``computed``（确定性计算）。
    ``line_index`` 指回 ``SceneAnswer.ocr_lines`` 的下标，``raw_text`` 是原文。
    """

    kind: str
    frame_id: str = ""
    line_index: int | None = None
    raw_text: str = ""
    field: str = ""

    @property
    def label(self) -> str:
        return SOURCE_LABELS.get(self.kind, self.kind)


@dataclass(frozen=True, slots=True)
class FrameAttempt:
    """一次取图尝试的记录（重试与失败原因可追溯）。"""

    index: int
    elapsed_ms: int
    frame_id: str = ""
    quality: QualityReport | None = None
    error: str = ""


@dataclass(frozen=True, slots=True)
class FrameAcquisition:
    """一次完整取图（可能含多次尝试）的结果记录。"""

    frame: ImageFrame
    purpose: str
    attempts: tuple[FrameAttempt, ...] = ()
    quality: QualityReport | None = None
    elapsed_ms: int = 0


@dataclass(frozen=True, slots=True)
class ObjectCandidate:
    """一个可分辨的找物候选；``box`` 是原图像素裁剪框。"""

    label: str
    box: tuple[int, int, int, int]  # x, y, w, h（原图像素）
    position: str  # 画面左上方 / 正中间 / 右侧 …
    confidence: float = 0.0
    feature: str = ""
    frame_id: str = ""

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return x + w // 2, y + h // 2


@dataclass(frozen=True, slots=True)
class ObjectCandidates:
    """找物结果：最多两个可分辨候选，附口语化播报文本。"""

    target: str
    found: bool
    text: str
    candidates: tuple[ObjectCandidate, ...] = ()
    frame_id: str = ""
    vlm_text: str = ""
    degraded: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SceneAnswer:
    """一次场景问答的结果；文本可直接送 TTS 播报。"""

    text: str
    intent: str
    frame_id: str = ""
    ocr_lines: tuple[OcrLine, ...] = ()
    vlm_text: str = ""
    source_tags: tuple[SourceTag, ...] = ()
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    quality: QualityReport | None = None
    candidates: ObjectCandidates | None = None
    degraded: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    timings: dict[str, int] = field(default_factory=dict)

    @property
    def clear(self) -> bool:
        """是否“看清且读到了明确证据”。"""
        return self.confidence >= 0.6 and not self.degraded


@dataclass(frozen=True, slots=True)
class CompareItem:
    """比较结果中的一个项目（A/B 同口径）。"""

    name: str
    unit: str = ""
    a: str | None = None
    b: str | None = None
    comparable: bool = False
    note: str = ""


@dataclass(frozen=True, slots=True)
class CompareAnswer:
    """两件物品的比较结果：仅对两边都可靠支持的项目给结论。"""

    text: str
    items: tuple[CompareItem, ...] = ()
    common_items: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    a_frame_id: str = ""
    b_frame_id: str = ""
    a_tags: tuple[SourceTag, ...] = ()
    b_tags: tuple[SourceTag, ...] = ()
    degraded: tuple[str, ...] = ()
    timings: dict[str, int] = field(default_factory=dict)
