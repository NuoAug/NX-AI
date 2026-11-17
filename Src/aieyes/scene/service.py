"""场景服务：问题 → 取图 → OCR-first（必要时 VLM）→ 带证据的口语化答案。

公开 API（阶段 3 交付面）：

* :class:`SceneService`
  * ``answer(question, *, frame=None) -> SceneAnswer``
  * ``read_text(frame) -> list[OcrLine]``
  * ``find_object(frame, target) -> ObjectCandidates``
  * ``compare(a, b, question) -> CompareAnswer``
  * ``close()``
* :func:`classify_intent` —— 规则意图分类（read/price/date/find/describe/calculate）。

回答纪律：有证据才说；没读到就明确说“没读到/看不清”；不编造。价格与日期
只来自 OCR 原文 + :mod:`aieyes.scene.parse` 的确定性解析，VLM 只用于描述和
定位，不让它报价格/日期数字。
"""

from __future__ import annotations

import colorsys
import datetime as _dt
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Sequence

import numpy as np

from aieyes.clock import mono_ms
from aieyes.errors import AieyesError
from aieyes.interfaces import ImageFrame, OcrLine, OcrResult
from aieyes.vision.image_ops import resize_to
from aieyes.vision.quality import QualityReport, check_quality

from .errors import (
    FrameAcquisitionError,
    FrameQualityError,
    SceneAnalysisError,
)
from .models import (
    CompareAnswer,
    CompareItem,
    ObjectCandidate,
    ObjectCandidates,
    SceneAnswer,
    SourceTag,
)
from .parse import (
    AmountFact,
    DateFact,
    compute_purchase,
    extract_amounts,
    extract_dates,
    format_amount_cn,
    format_date_cn,
)
from .prompts import compare_prompt, describe_prompt, find_object_prompt, read_text_prompt
from .providers import CameraFrameProvider, FramePolicy, FrameProvider

log = logging.getLogger(__name__)

#: 意图 → 取图用途（与合同 frame.acquire.purpose 对齐）
PURPOSE_BY_INTENT: dict[str, str] = {
    "read": "ocr",
    "price": "ocr",
    "date": "ocr",
    "find": "vqa",
    "describe": "vqa",
    "unknown": "vqa",
}

INTENT_NAMES: dict[str, str] = {
    "read": "读字",
    "price": "价格",
    "date": "日期",
    "find": "找物",
    "describe": "描述",
    "calculate": "口述计算",
    "unknown": "未知",
}

# ---------------------------------------------------------------------------
# 关键词规则（故意简单可解释，便于测试与定位误判）
# ---------------------------------------------------------------------------
_CALC_RE = re.compile(r"一共|总共|合计|乘以|×|算一下|算算|要多少|花多少")
_DATE_RE = re.compile(
    r"日期|保质期|过期|到期|生产|有效期|批号|新鲜|什么时候坏|能放多久"
)
_FIND_RE = re.compile(r"在哪|哪儿|哪里|放在哪|找|有没有|位置|躲")
_PRICE_RE = re.compile(r"价格|多少钱|几块|几元|贵|便宜|标价|售价|打折|优惠|钱")
_READ_RE = re.compile(r"读|念|文字|什么字|写的|标签上|上面写|牌子|字")
_DESCRIBE_RE = re.compile(
    r"描述|看看|什么样|是什么|颜色|几个|什么东西|认识|干什么|做什么|安全|看看"
)

_PRICE_FIELD_LABELS: dict[str, str] = {
    "original_price": "原价",
    "promo_price": "现价",
    "member_price": "会员价",
    "unit_price": "单价",
    "total_price": "总价",
    "price": "价格",
}
#: 展示价优先顺序（同一件商品有多个价签时取“实际售价”）
_PRIMARY_PRICE_FIELDS: tuple[str, ...] = (
    "promo_price",
    "member_price",
    "price",
    "unit_price",
    "total_price",
)

_COMPARE_NAMES: dict[str, str] = {
    "price": "价格",
    "date": "日期",
    "color": "颜色",
}

#: 这些质量标签会真正影响可用性（过曝的白纸仍可 OCR，只报告不降级）
GATE_LABELS: tuple[str, ...] = ("blurry", "dark", "too_small", "too_large")

#: 三格位置名（行 × 列）
_POSITION_NAMES = (
    ("左上方", "正上方", "右上方"),
    ("左侧", "正中间", "右侧"),
    ("左下方", "正下方", "右下方"),
)


def classify_intent(question: str) -> str:
    """按关键词把问题分成 ``read/price/date/find/describe/calculate``。"""
    text = (question or "").strip()
    if not text:
        return "describe"
    if _CALC_RE.search(text):
        return "calculate"
    if _DATE_RE.search(text):
        return "date"
    if _FIND_RE.search(text):
        return "find"
    if _PRICE_RE.search(text):
        return "price"
    if _READ_RE.search(text):
        return "read"
    if _DESCRIBE_RE.search(text):
        return "describe"
    return "describe"


def _join_ocr_lines(lines: Sequence[OcrLine], limit: int) -> tuple[str, bool]:
    """按上限拼 OCR 原文；返回（文本，是否截断）。"""
    parts: list[str] = []
    total = 0
    truncated = False
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        if total + len(text) > limit:
            if not parts:
                parts.append(text[:limit])
            truncated = True
            break
        parts.append(text)
        total += len(text)
    return "，".join(parts), truncated


def _avg_score(lines: Sequence[OcrLine]) -> float:
    scores = [float(line.score) for line in lines if line.text.strip()]
    return sum(scores) / len(scores) if scores else 0.0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _gated(report: QualityReport | None) -> bool:
    """质量标签里是否含有真正影响可读性的项（过曝白纸不算）。"""
    if report is None:
        return False
    return any(label in GATE_LABELS for label in report.labels)


@dataclass(slots=True)
class _FrameView:
    """compare 时对一张图的分析结果（OCR/价格/日期/颜色）。"""

    frame: ImageFrame
    quality: QualityReport
    ocr: OcrResult | None = None
    ocr_error: str = ""
    price: Decimal | None = None
    price_note: str = ""
    date: _dt.date | None = None
    date_field: str = ""
    color: str = ""
    tags: list[SourceTag] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _AnswerContext:
    """answer 过程中的共享上下文，``finish`` 统一补齐证据与计时。"""

    frame: ImageFrame
    quality: QualityReport
    ocr: OcrResult | None
    degraded: list[str]
    notes: list[str]
    timings: dict[str, int]
    started: int

    def finish(
        self,
        text: str,
        intent: str,
        *,
        vlm_text: str = "",
        tags: Sequence[SourceTag] = (),
        confidence: float = 0.0,
        evidence: Sequence[str] = (),
        candidates: ObjectCandidates | None = None,
        extra_degraded: Sequence[str] = (),
        extra_notes: Sequence[str] = (),
    ) -> SceneAnswer:
        degraded = list(self.degraded) + list(extra_degraded)
        notes = list(self.notes) + list(extra_notes)
        evidence_list = [
            f"帧 {self.frame.frame_id or '（无 id）'} {self.frame.width}x{self.frame.height}",
        ]
        if self.quality.labels:
            evidence_list.append("质量：" + "、".join(self.quality.labels))
        evidence_list.extend(evidence)
        confidence_value = _clip01(confidence)
        if "low_quality" in degraded and confidence_value > 0:
            confidence_value = min(confidence_value, 0.55)
        timings = dict(self.timings)
        timings["total"] = max(0, mono_ms() - self.started)
        return SceneAnswer(
            text=text.strip(),
            intent=intent,
            frame_id=self.frame.frame_id,
            ocr_lines=tuple(self.ocr.lines) if self.ocr is not None else (),
            vlm_text=vlm_text,
            source_tags=tuple(tags),
            confidence=confidence_value,
            evidence=tuple(evidence_list),
            quality=self.quality,
            candidates=candidates,
            degraded=tuple(degraded),
            notes=tuple(notes),
            timings=timings,
        )


class SceneService:
    """场景问答服务；OCR/VLM/取图通道都可以注入假实现做单测。"""

    def __init__(
        self,
        *,
        vlm=None,
        ocr=None,
        provider: FrameProvider | None = None,
        config: "SceneConfig | None" = None,
    ) -> None:
        self.config = config or SceneConfig()
        self._vlm = vlm
        self._ocr = ocr
        self._provider = provider
        self._owns_vlm = vlm is None
        self._owns_ocr = ocr is None
        self._owns_provider = provider is None
        self._ocr_cache: OrderedDict[str, OcrResult] = OrderedDict()
        #: 最近一次 `answer()` 实际使用过的帧：上层做同图追问/翻面复用时取它
        #: （功能说明 L171「复用已有照片」）。取消或新一轮开始时由上层调
        #: :meth:`reset_last_frame` 清掉，避免把上一轮的旧图当成当前画面。
        self.last_frame: ImageFrame | None = None

    # ------------------------------------------------------------------
    # 依赖的惰性默认实现（避免 import 即加载模型/网络库）
    # ------------------------------------------------------------------
    @property
    def vlm(self):
        if self._vlm is None:
            from aieyes.vision.vlm import VlmClient

            self._vlm = VlmClient()
        return self._vlm

    @property
    def ocr(self):
        if self._ocr is None:
            from aieyes.vision.ocr import OcrEngine

            self._ocr = OcrEngine()
        return self._ocr

    @property
    def provider(self) -> FrameProvider:
        if self._provider is None:
            self._provider = CameraFrameProvider(
                policy=FramePolicy(
                    grab_timeout_ms=self.config.t09_ms,
                    max_attempts=self.config.max_attempts,
                    require_clear=True,
                    retry_delay_ms=self.config.retry_delay_ms,
                )
            )
        return self._provider

    def reset_last_frame(self) -> None:
        """丢掉缓存帧：取消、停止、新一轮开始时调用，防止跨轮复用旧图。"""
        self.last_frame = None

    def close(self) -> None:
        """释放自建的模型/取图资源；注入的对象由调用方管理。"""
        if self._owns_provider and self._provider is not None:
            self._provider.close()
            self._provider = None
        if self._owns_vlm and self._vlm is not None:
            close = getattr(self._vlm, "close", None)
            if callable(close):
                close()
            self._vlm = None
        if self._owns_ocr and self._ocr is not None:
            close = getattr(self._ocr, "close", None)
            if callable(close):
                close()
            self._ocr = None
        self._ocr_cache.clear()

    def __enter__(self) -> "SceneService":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def read_text_result(self, frame: ImageFrame) -> OcrResult:
        """对一帧做 OCR，返回完整原始结果（行/box/polygon/分数/模型耗时）。"""
        return self._run_ocr(frame)

    def read_text(self, frame: ImageFrame) -> list[OcrLine]:
        """读画面文字；失败抛 :class:`SceneAnalysisError`。"""
        return list(self.read_text_result(frame).lines)

    def answer(
        self,
        question: str,
        *,
        frame: ImageFrame | None = None,
        intent_hint: str | None = None,
    ) -> SceneAnswer:
        """回答一个场景问题。

        ``frame`` 为 ``None`` 时通过取图提供者取一张（相机/文件）；传入帧则直接
        使用（上层已完成取图与质量检查）。
        """
        started = mono_ms()
        question = (question or "").strip()
        if not question:
            question = "看看画面里有什么"
        timings: dict[str, int] = {}

        # 1) 口述价格 × 数量：确定性计算，不取图、不调模型
        purchase = compute_purchase(question)
        if purchase is not None:
            return self._answer_purchase(purchase, question, started)

        # 上层（状态机）已完成一次意图分类时直接沿用，避免 ASR 误字导致
        # 场景模块把“价格问题”再判成 describe（阶段3真机测试踩到）。
        intent = intent_hint if intent_hint in PURPOSE_BY_INTENT else classify_intent(question)
        purpose = PURPOSE_BY_INTENT.get(intent, "vqa")
        degraded: list[str] = []
        notes: list[str] = []
        quality: QualityReport | None = None
        acquisition_ms = 0

        # 2) 取图（或校验调用方传入的帧）
        if frame is None:
            provider = self.provider
            t0 = mono_ms()
            try:
                frame = provider.acquire(purpose)
            except FrameQualityError as exc:
                acquisition_ms = mono_ms() - t0
                if self.config.analyze_low_quality and exc.frame is not None:
                    frame = exc.frame
                    quality = exc.report
                    degraded.append("low_quality")
                    notes.append(str(exc))
                else:
                    timings["acquire"] = acquisition_ms
                    return self._no_frame_answer(
                        "现在没看清画面，请把东西拿近一点、对准一点再问一次。",
                        intent,
                        started,
                        timings,
                        degraded=("no_frame",),
                        notes=(str(exc),),
                    )
            except FrameAcquisitionError as exc:
                timings["acquire"] = mono_ms() - t0
                return self._no_frame_answer(
                    "现在我取不到画面，请检查相机有没有被挡住，再试一次。",
                    intent,
                    started,
                    timings,
                    degraded=("acquisition_failed",),
                    notes=(str(exc),),
                )
            else:
                acquisition_ms = mono_ms() - t0
                last = getattr(provider, "last_acquisition", None)
                if last is not None and last.frame.frame_id == frame.frame_id:
                    quality = last.quality
        if acquisition_ms:
            timings["acquire"] = acquisition_ms
        assert frame is not None
        if quality is None:
            t0 = mono_ms()
            quality = check_quality(frame.rgb)
            timings["quality"] = mono_ms() - t0
        if _gated(quality) and "low_quality" not in degraded:
            degraded.append("low_quality")
        # 记下本轮真正用过的帧：同图追问/翻面复用时上层拿它复用，不再重新取图。
        self.last_frame = frame

        # 3) OCR-first（找物走 VLM；其它意图都先拿原文）
        ocr_result: OcrResult | None = None
        if intent != "find":
            t0 = mono_ms()
            try:
                ocr_result = self._run_ocr(frame)
            except SceneAnalysisError as exc:
                degraded.append("ocr_failed")
                notes.append(str(exc))
            finally:
                timings["ocr"] = mono_ms() - t0

        context = _AnswerContext(
            frame=frame,
            quality=quality,
            ocr=ocr_result,
            degraded=degraded,
            notes=notes,
            timings=timings,
            started=started,
        )

        # 4) 按意图生成答案
        if intent == "read":
            return self._answer_read(question, context)
        if intent == "price":
            return self._answer_price(question, context)
        if intent == "date":
            return self._answer_date(question, context)
        if intent == "find":
            return self._answer_find(question, frame, context)
        return self._answer_describe(question, context)

    def find_object(self, frame: ImageFrame, target: str) -> ObjectCandidates:
        """在画面里找 ``target``：最多返回 ``config.max_candidates`` 个候选。"""
        target = (target or "").strip()
        if not target:
            return ObjectCandidates(
                target="",
                found=False,
                text="请告诉我要找什么。",
                frame_id=frame.frame_id,
                degraded=("no_target",),
            )
        prompt = find_object_prompt(target, max_candidates=self.config.max_candidates)
        try:
            raw = self._vlm_complete(prompt, frame)
        except SceneAnalysisError as exc:
            return ObjectCandidates(
                target=target,
                found=False,
                text=f"找东西的画面识别失败了，我没看清{target}。",
                frame_id=frame.frame_id,
                degraded=("vlm_failed",),
            )

        items, parsed = _parse_candidate_json(raw)
        if not parsed:
            return ObjectCandidates(
                target=target,
                found=False,
                text=f"我没看清{target}在哪儿。",
                frame_id=frame.frame_id,
                vlm_text=raw,
                degraded=("vlm_parse",),
            )

        width, height = frame.width, frame.height
        candidates: list[ObjectCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or target).strip()[:20] or target
            box = _normalize_box(item.get("box"), width, height)
            feature = str(item.get("feature") or "").strip()[:24]
            if box is None:
                # 没有可用框时给一个中心近似框，并降低置信度
                box = (width // 4, height // 4, width // 2, height // 2)
                approximate = True
            else:
                approximate = False
            position = str(item.get("position") or "").strip()[:12]
            if not position:
                position = _position_of(box, width, height)
            confidence = _clip01(_to_float(item.get("confidence"), default=0.5))
            if approximate:
                confidence *= 0.5
                feature = (feature + "；位置为近似").strip("；")
            candidates.append(
                ObjectCandidate(
                    label=label,
                    box=box,
                    position=position,
                    confidence=confidence,
                    feature=feature,
                    frame_id=frame.frame_id,
                )
            )
        candidates = _dedupe_candidates(candidates, max_candidates=self.config.max_candidates)
        if not candidates:
            return ObjectCandidates(
                target=target,
                found=False,
                text=f"画面里没看到{target}。",
                frame_id=frame.frame_id,
                vlm_text=raw,
            )
        text = _candidates_text(target, candidates)
        return ObjectCandidates(
            target=target,
            found=True,
            text=text,
            candidates=tuple(candidates),
            frame_id=frame.frame_id,
            vlm_text=raw,
        )

    def compare(
        self,
        a: ImageFrame | str | Path,
        b: ImageFrame | str | Path,
        question: str = "比较一下这两样东西",
    ) -> CompareAnswer:
        """两幅图的同口径比较（阶段 4 会复用；这里先实现价格/日期/颜色）。"""
        started = mono_ms()
        timings: dict[str, int] = {}
        degraded: list[str] = []
        try:
            frame_a = _as_frame(a)
            frame_b = _as_frame(b)
        except (AieyesError, ValueError):
            return CompareAnswer(
                text="有一张图我打不开，没法比较。",
                degraded=("bad_input",),
                timings={"total": max(0, mono_ms() - started)},
            )

        if frame_a.frame_id and frame_a.frame_id == frame_b.frame_id:
            # 清单 21：两个位置是同一张图时不能输出“比较结果”，否则等于旧图冒充成功。
            return CompareAnswer(
                text="这两个位置是同一张图，没法比较。把要比较的两样东西分别放到两个位置，各拍一张再来问。",
                a_frame_id=frame_a.frame_id,
                b_frame_id=frame_b.frame_id,
                degraded=("same_frame",),
                timings={"total": max(0, mono_ms() - started)},
            )

        explicit_fields = _compare_fields(question)
        dropped_fields = _compare_dropped_fields(question)
        wanted = explicit_fields or tuple(
            name for name in ("price", "date", "color") if name not in dropped_fields
        )
        if not wanted:
            # 三项都被明确排除：问清要比哪一项，不擅自猜一个维度。
            return CompareAnswer(
                text="价格、日期、颜色都不比的话，你想让我比哪一项？",
                degraded=("no_compare_field",),
                a_frame_id=frame_a.frame_id,
                b_frame_id=frame_b.frame_id,
                timings={"total": max(0, mono_ms() - started)},
            )
        # 用户只说“对比物品”时，结构化字段往往读不全；允许 VLM 补一段
        # “各是什么、最明显的区别”，避免只回一句“价格/日期/颜色都没读全”。
        # 容量/大小/重量等没有结构化口径的维度也交给 VLM，不能因为旧问题里
        # 带了“价格”就把它静默忽略。
        generic_dim = bool(_GENERIC_DIM_RE.search(question or ""))
        generic_question = (
            (not explicit_fields and not dropped_fields)
            or generic_dim
        )
        need_ocr = any(name in ("price", "date") for name in wanted)

        t0 = mono_ms()
        view_a = self._view_for_compare(frame_a, need_ocr=need_ocr)
        timings["a_ocr"] = mono_ms() - t0
        t0 = mono_ms()
        view_b = self._view_for_compare(frame_b, need_ocr=need_ocr)
        timings["b_ocr"] = mono_ms() - t0
        degraded.extend(view_a.degraded)
        degraded.extend(view_b.degraded)

        items: list[CompareItem] = []
        conclusions: list[str] = []
        missing: list[str] = []
        common: list[str] = []
        for name in wanted:
            item = _compare_item(name, view_a, view_b)
            items.append(item)
            if item.comparable:
                common.append(item.name)
                conclusions.append(_compare_conclusion(item, name, view_a, view_b))
            else:
                missing.append(item.name)

        vlm_text = ""
        # 价格/日期已经是可靠结论时，结构化回答本身有信息量，不再加一轮 VLM
        # （否则准确数字会被长描述埋在后面，还多付几秒）。只有只剩颜色或什么都没
        # 读到时，才用并排图补“各是什么、最明显的差异”。
        strong_conclusion = any(label in ("价格", "日期") for label in common)
        if generic_question and (generic_dim or (missing and not strong_conclusion)):
            t_vlm = mono_ms()
            try:
                vlm_text = self._compare_with_vlm(question, frame_a, frame_b)
            except SceneAnalysisError as exc:
                degraded.append("vlm_compare_failed")
                vlm_text = ""
                log.debug("VLM 比较兜底失败: %s", exc)
            timings["vlm_compare"] = mono_ms() - t_vlm
            if not _useful_compare_text(vlm_text):
                if vlm_text:
                    degraded.append("vlm_compare_unclear")
                vlm_text = ""

        if vlm_text:
            # 已有可靠结构化结论（如价格）时保留它，再补 VLM 的可见差异描述，
            # 不能让补充维度把准确数字冲掉。
            prefix = ("比较结果：" + "；".join(conclusions) + "。") if conclusions else ""
            text = prefix + vlm_text.rstrip("。！？!?；;，, ") + "。"
        elif conclusions:
            text = "比较结果：" + "；".join(conclusions) + "。"
            if missing:
                text += "（" + "、".join(missing) + "没读到，这次没比。"
                text += "）"
        else:
            text = (
                "这两样东西我还没能按同一口径比较："
                + "、".join(missing or list(_COMPARE_NAMES.values()))
                + "都没读全。"
            )
        timings["total"] = max(0, mono_ms() - started)
        vlm_tags_a = (
            (SourceTag(kind="vlm", frame_id=frame_a.frame_id, raw_text=vlm_text),)
            if vlm_text
            else ()
        )
        vlm_tags_b = (
            (SourceTag(kind="vlm", frame_id=frame_b.frame_id, raw_text=vlm_text),)
            if vlm_text
            else ()
        )
        return CompareAnswer(
            text=text,
            items=tuple(items),
            common_items=tuple(common),
            missing=tuple(missing),
            a_frame_id=frame_a.frame_id,
            b_frame_id=frame_b.frame_id,
            a_tags=vlm_tags_a or tuple(view_a.tags),
            b_tags=vlm_tags_b or tuple(view_b.tags),
            degraded=tuple(dict.fromkeys(degraded)),
            timings=timings,
        )

    # ------------------------------------------------------------------
    # answer：各意图
    # ------------------------------------------------------------------
    def _answer_purchase(self, purchase, question: str, started: int) -> SceneAnswer:
        text = (
            f"按你说的，每件{format_amount_cn(purchase.unit_price)}，"
            f"{purchase.quantity}件一共{format_amount_cn(purchase.total)}。"
        )
        tags = (
            SourceTag(kind="spoken", raw_text=question[:60], field="unit_price"),
            SourceTag(kind="computed", raw_text=str(purchase.total), field="total"),
        )
        timings = {"parse": 0, "total": max(0, mono_ms() - started)}
        return SceneAnswer(
            text=text,
            intent="calculate",
            source_tags=tags,
            confidence=0.9,
            evidence=(
                f"口述单价 {purchase.unit_price}",
                f"数量 {purchase.quantity}",
                f"计算 {purchase.unit_price} × {purchase.quantity} = {purchase.total}",
            ),
            timings=timings,
        )

    def _answer_read(self, question: str, ctx: _AnswerContext) -> SceneAnswer:
        lines = list(ctx.ocr.lines) if ctx.ocr is not None else []
        vlm_text = ""
        tags: list[SourceTag] = []
        if not lines:
            # OCR 没读到或失败：退到 VLM 描述可见文字，仍说清来源
            try:
                vlm_text = self._vlm_complete(read_text_prompt(question), ctx.frame)
            except SceneAnalysisError as exc:
                ctx.degraded.append("vlm_failed")
                ctx.notes.append(str(exc))
                vlm_text = ""
            message = vlm_text.strip()
            if message and message not in ("看不清", "看不清。"):
                tags.append(SourceTag(kind="vlm", frame_id=ctx.frame.frame_id, raw_text=message))
                return ctx.finish(
                    f"我没识别出清楚的文字，画面助手说：{_shorten(message)}",
                    "read",
                    vlm_text=message,
                    tags=tags,
                    confidence=0.35,
                    extra_degraded=("no_ocr_text",),
                )
            return ctx.finish(
                "画面里我没读到清楚的字。",
                "read",
                confidence=0.1,
                extra_degraded=("no_text",),
            )

        joined, truncated = _join_ocr_lines(lines, self.config.max_read_chars)
        for index, line in enumerate(lines):
            if line.text.strip():
                tags.append(
                    SourceTag(
                        kind="ocr",
                        frame_id=ctx.frame.frame_id,
                        line_index=index,
                        raw_text=line.text,
                        field="text",
                    )
                )
        text = f"画面上写着：{joined}。"
        if truncated:
            text += "后面还有字，没读完。"
        confidence = min(0.95, max(0.35, 0.4 + 0.55 * _avg_score(lines)))
        return ctx.finish(
            text,
            "read",
            tags=tags,
            confidence=confidence,
            evidence=(f"OCR {len(lines)} 行，平均分 {_avg_score(lines):.2f}",),
            extra_degraded=("text_truncated",) if truncated else (),
        )

    def _answer_price(self, question: str, ctx: _AnswerContext) -> SceneAnswer:
        facts = list(extract_amounts(ctx.ocr.lines)) if ctx.ocr is not None else []
        lines = list(ctx.ocr.lines) if ctx.ocr is not None else []
        tags = [
            SourceTag(
                kind="ocr",
                frame_id=ctx.frame.frame_id,
                line_index=fact.line_index,
                raw_text=fact.raw_text,
                field=fact.field,
            )
            for fact in facts
        ]
        if not facts:
            detail = "画面里有字，但没读到明确的价格。" if lines else "画面里没读到字，也没读到价格。"
            return ctx.finish(
                detail,
                "price",
                confidence=0.1,
                extra_degraded=("no_price",),
                evidence=(f"OCR {len(lines)} 行",) if lines else (),
            )

        values = _dedupe_values(facts)
        if len(values) == 1:
            fact = facts[0]
            label = _PRICE_FIELD_LABELS.get(fact.field, "价格")
            return ctx.finish(
                f"包装上写的{label}是{format_amount_cn(fact.value)}。",
                "price",
                tags=tags,
                confidence=min(0.92, max(0.5, 0.5 + 0.4 * _avg_score(lines))),
                evidence=(f"金额原文：{fact.raw_text}", f"解析值 {fact.value}"),
            )

        original = next((f for f in facts if f.field == "original_price"), None)
        current = next(
            (f for f in facts if f.field in ("promo_price", "member_price", "price")), None
        )
        if original is not None and current is not None and original.value != current.value:
            return ctx.finish(
                f"包装上原价是{format_amount_cn(original.value)}，"
                f"现价是{format_amount_cn(current.value)}。",
                "price",
                tags=tags,
                confidence=0.75,
                evidence=(f"原价原文：{original.raw_text}", f"现价原文：{current.raw_text}"),
            )

        shown = values[:2]
        pairs = "，另一个是".join(format_amount_cn(value) for value in shown)
        return ctx.finish(
            f"我读到两个价格：一个是{pairs}，但分不清分别对应哪个商品，"
            "建议只对准一个价签再看一次。",
            "price",
            tags=tags,
            confidence=0.35,
            extra_degraded=("ambiguous_price",),
            evidence=tuple(f"金额原文：{fact.raw_text}" for fact in facts[:2]),
        )

    def _answer_date(self, question: str, ctx: _AnswerContext) -> SceneAnswer:
        facts = list(extract_dates(ctx.ocr.lines)) if ctx.ocr is not None else []
        if not facts:
            return ctx.finish(
                "画面里没读到日期。" if ctx.ocr is not None else "画面文字没读出来，日期没读到。",
                "date",
                confidence=0.1,
                extra_degraded=("no_date",),
            )

        def _fact_tags(selected: Sequence[DateFact]) -> list[SourceTag]:
            return [
                SourceTag(
                    kind="ocr",
                    frame_id=ctx.frame.frame_id,
                    line_index=fact.line_index,
                    raw_text=fact.raw_text,
                    field=fact.field,
                )
                for fact in selected
            ]

        expiry = next((f for f in facts if f.field == "expiry_date" and f.value), None)
        production = next((f for f in facts if f.field == "production_date" and f.value), None)
        unlabeled = next((f for f in facts if f.field == "unlabeled_date" and f.value), None)
        extras = [f for f in facts if f.field in ("batch_id", "shelf_life")]

        if expiry is not None:
            selected = [expiry] + ([production] if production is not None else [])
            if production is not None:
                text = (
                    f"包装上写的生产日期是{format_date_cn(production.value)}，"
                    f"到期日是{format_date_cn(expiry.value)}。"
                )
            else:
                text = f"包装上写的到期日是{format_date_cn(expiry.value)}。"
            return ctx.finish(
                text,
                "date",
                tags=_fact_tags(selected),
                confidence=0.85,
                evidence=tuple(f"日期原文：{fact.raw_text}" for fact in selected),
            )

        if production is not None:
            return ctx.finish(
                f"包装上写的生产日期是{format_date_cn(production.value)}，但没读到到期日。",
                "date",
                tags=_fact_tags([production]),
                confidence=0.65,
                extra_degraded=("no_expiry",),
                evidence=(f"日期原文：{production.raw_text}",),
            )

        if unlabeled is not None:
            return ctx.finish(
                f"画面里有一个日期 {format_date_cn(unlabeled.value)}，"
                "但看不出是生产日期还是到期日。",
                "date",
                tags=_fact_tags([unlabeled]),
                confidence=0.4,
                extra_degraded=("unlabeled_date",),
                evidence=(f"日期原文：{unlabeled.raw_text}",),
            )

        if extras:
            fields = "、".join("批号" if f.field == "batch_id" else "保质期" for f in extras)
            return ctx.finish(
                f"画面里只读到{fields}，没读到明确的日期。",
                "date",
                tags=_fact_tags(extras),
                confidence=0.2,
                extra_degraded=("no_date",),
            )
        return ctx.finish(
            "画面里没读到日期。",
            "date",
            confidence=0.1,
            extra_degraded=("no_date",),
        )

    def _answer_find(self, question: str, frame: ImageFrame, ctx: _AnswerContext) -> SceneAnswer:
        target = _find_target(question)
        t0 = mono_ms()
        candidates = self.find_object(frame, target)
        ctx.timings["vlm"] = mono_ms() - t0
        tags: list[SourceTag] = []
        if candidates.vlm_text:
            tags.append(
                SourceTag(
                    kind="vlm",
                    frame_id=frame.frame_id,
                    raw_text=candidates.vlm_text[:200],
                    field="object",
                )
            )
        confidence = 0.15
        if candidates.found and candidates.candidates:
            confidence = max(0.3, min(0.9, candidates.candidates[0].confidence))
        return ctx.finish(
            candidates.text,
            "find",
            vlm_text=candidates.vlm_text,
            tags=tags,
            confidence=confidence,
            candidates=candidates,
            extra_degraded=candidates.degraded,
            evidence=(f"找物目标：{target}", f"候选 {len(candidates.candidates)} 个"),
        )

    def _answer_describe(self, question: str, ctx: _AnswerContext) -> SceneAnswer:
        t0 = mono_ms()
        vlm_text = ""
        vlm_failed = False
        try:
            vlm_text = self._vlm_complete(describe_prompt(question), ctx.frame)
        except SceneAnalysisError as exc:
            vlm_failed = True
            ctx.degraded.append("vlm_failed")
            ctx.notes.append(str(exc))
        finally:
            ctx.timings["vlm"] = mono_ms() - t0
        message = _shorten(vlm_text.strip(), limit=120)
        if not message:
            extra = () if vlm_failed else ("vlm_empty",)
            lines = list(ctx.ocr.lines) if ctx.ocr is not None else []
            if lines:
                joined, _ = _join_ocr_lines(lines, self.config.max_read_chars)
                return ctx.finish(
                    f"画面助手这次没答上来，我只读到这些字：{joined}。",
                    "describe",
                    tags=(SourceTag(kind="ocr", frame_id=ctx.frame.frame_id, raw_text=joined),),
                    confidence=0.3,
                    extra_degraded=extra,
                )
            return ctx.finish(
                "我这次没看清画面，请把镜头对准一点再问一次。",
                "describe",
                confidence=0.1,
                extra_degraded=extra,
            )
        lines = list(ctx.ocr.lines) if ctx.ocr is not None else []
        evidence: list[str] = []
        if lines:
            joined, _ = _join_ocr_lines(lines, 80)
            evidence.append(f"OCR 原文：{joined}")
        return ctx.finish(
            message if message.endswith(("。", "！", "？")) else message + "。",
            "describe",
            vlm_text=vlm_text,
            tags=(SourceTag(kind="vlm", frame_id=ctx.frame.frame_id, raw_text=vlm_text),),
            confidence=0.65,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # 内部：OCR / VLM 调用与降级
    # ------------------------------------------------------------------
    def _run_ocr(self, frame: ImageFrame) -> OcrResult:
        frame_id = frame.frame_id
        if frame_id:
            cached = self._ocr_cache.get(frame_id)
            if cached is not None:
                self._ocr_cache.move_to_end(frame_id)
                return cached
        try:
            result = self.ocr.recognize(frame.rgb, source_frame_id=frame_id)
        except Exception as exc:  # noqa: BLE001 - OCR 后端异常种类多，统一降级
            raise SceneAnalysisError(
                f"文字识别失败：{exc}", cause=str(exc), provider="ocr"
            ) from exc
        if frame_id:
            self._ocr_cache[frame_id] = result
            while len(self._ocr_cache) > max(1, self.config.ocr_cache_size):
                self._ocr_cache.popitem(last=False)
        return result

    def _vlm_complete(self, prompt: str, frame: ImageFrame) -> str:
        try:
            return self.vlm.complete(
                prompt,
                images=(frame.rgb,),
                max_tokens=self.config.vlm_max_tokens,
                timeout_s=self.config.vlm_timeout_s,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - 视觉模型后端异常统一降级
            raise SceneAnalysisError(
                f"画面识别失败：{exc}", cause=str(exc), provider="vlm"
            ) from exc

    def _compare_with_vlm(
        self, question: str, frame_a: ImageFrame, frame_b: ImageFrame
    ) -> str:
        """两图并排成一张，交给单图 VLM 说清“各是什么、最明显的区别”。

        不依赖服务端多图能力，避免不同 llama.cpp/视觉投影实现对多图的差异。
        """
        try:
            composite = _side_by_side(frame_a.rgb, frame_b.rgb)
            return self.vlm.complete(
                compare_prompt(question),
                images=(composite,),
                max_tokens=min(int(self.config.vlm_max_tokens), 160),
                timeout_s=self.config.vlm_timeout_s,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            raise SceneAnalysisError(
                f"画面比较失败：{exc}", cause=str(exc), provider="vlm"
            ) from exc

    # ------------------------------------------------------------------
    # compare 内部
    # ------------------------------------------------------------------
    def _view_for_compare(self, frame: ImageFrame, *, need_ocr: bool) -> _FrameView:
        view = _FrameView(frame=frame, quality=check_quality(frame.rgb))
        view.color = _dominant_color(frame.rgb)
        if _gated(view.quality):
            view.degraded.append("low_quality")
        if not need_ocr:
            return view
        try:
            view.ocr = self._run_ocr(frame)
        except SceneAnalysisError as exc:
            view.ocr_error = str(exc)
            view.degraded.append("ocr_failed")
            return view

        facts = list(extract_amounts(view.ocr.lines))
        if facts:
            view.tags.extend(
                SourceTag(
                    kind="ocr",
                    frame_id=frame.frame_id,
                    line_index=fact.line_index,
                    raw_text=fact.raw_text,
                    field=fact.field,
                )
                for fact in facts
            )
            distinct = _dedupe_values(facts)
            if len(distinct) == 1:
                view.price = distinct[0]
            else:
                primary = next(
                    (f for f in facts if f.field in _PRIMARY_PRICE_FIELDS), None
                )
                if primary is not None and _is_unique_primary(facts, primary):
                    view.price = primary.value
                else:
                    view.price_note = "价格不唯一"

        dates = list(extract_dates(view.ocr.lines))
        if dates:
            view.tags.extend(
                SourceTag(
                    kind="ocr",
                    frame_id=frame.frame_id,
                    line_index=fact.line_index,
                    raw_text=fact.raw_text,
                    field=fact.field,
                )
                for fact in dates
            )
            expiry = next((f for f in dates if f.field == "expiry_date" and f.value), None)
            production = next((f for f in dates if f.field == "production_date" and f.value), None)
            picked = expiry or production
            if picked is not None:
                view.date = picked.value
                view.date_field = picked.field
        view.tags = _dedupe_tags(view.tags)
        return view

    # ------------------------------------------------------------------
    # 无帧答案
    # ------------------------------------------------------------------
    def _no_frame_answer(
        self,
        text: str,
        intent: str,
        started: int,
        timings: dict[str, int],
        *,
        degraded: Sequence[str],
        notes: Sequence[str],
    ) -> SceneAnswer:
        timings = dict(timings)
        timings["total"] = max(0, mono_ms() - started)
        return SceneAnswer(
            text=text,
            intent=intent,
            confidence=0.0,
            degraded=tuple(degraded),
            notes=tuple(notes),
            timings=timings,
        )


# ---------------------------------------------------------------------------
# SceneService 配置（模块内默认值；如需全局配置再上提到 config.py）
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SceneConfig:
    """场景服务参数（默认值即目标板实测预算）。"""

    t09_ms: int = 10_000  # 取图 + 质量检查总预算
    max_attempts: int = 3  # 同一位置最多取图次数
    analyze_low_quality: bool = True  # 质量不达标时降级分析最后一帧
    vlm_max_tokens: int = 160
    vlm_timeout_s: float = 120.0
    max_candidates: int = 2
    ocr_cache_size: int = 8
    max_read_chars: int = 200
    retry_delay_ms: int = 120


# ---------------------------------------------------------------------------
# 纯函数工具：VLM JSON 解析 / 候选去重 / 位置 / 颜色 / 比较
# ---------------------------------------------------------------------------
def _dedupe_tags(tags: Sequence[SourceTag]) -> list[SourceTag]:
    seen: set[tuple[str, int | None, str, str]] = set()
    result: list[SourceTag] = []
    for tag in tags:
        key = (tag.kind, tag.line_index, tag.raw_text, tag.field)
        if key in seen:
            continue
        seen.add(key)
        result.append(tag)
    return result


def _to_float(value, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_candidate_json(raw: str) -> tuple[list, bool]:
    """从 VLM 输出里抠出 JSON 数组；容忍代码块与前后废话。"""
    text = (raw or "").strip()
    if not text:
        return [], False
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return [], False
    try:
        data = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return [], False
    if not isinstance(data, list):
        return [], False
    return data, True


def _normalize_box(
    box, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """把 VLM 的 0-1000 归一化 [x1,y1,x2,y2] 转成原图像素 (x,y,w,h)。"""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    values = [_to_float(v, default=-1.0) for v in box]
    if any(v < 0 for v in values):
        return None
    x1, y1, x2, y2 = values
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    scale_x = width / 1000.0
    scale_y = height / 1000.0
    px1 = int(round(max(0.0, min(1000.0, x1)) * scale_x))
    py1 = int(round(max(0.0, min(1000.0, y1)) * scale_y))
    px2 = int(round(max(0.0, min(1000.0, x2)) * scale_x))
    py2 = int(round(max(0.0, min(1000.0, y2)) * scale_y))
    px2 = min(width, max(px1 + 8, px2))
    py2 = min(height, max(py1 + 8, py2))
    if px1 >= px2 or py1 >= py2:
        return None
    return px1, py1, px2 - px1, py2 - py1


def _position_of(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x, y, w, h = box
    cx = (x + w / 2) / max(1, width)
    cy = (y + h / 2) / max(1, height)
    column = 0 if cx < 1 / 3 else (1 if cx < 2 / 3 else 2)
    row = 0 if cy < 1 / 3 else (1 if cy < 2 / 3 else 2)
    return _POSITION_NAMES[row][column]


def _iou(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _dedupe_candidates(
    candidates: Sequence[ObjectCandidate], *, max_candidates: int
) -> list[ObjectCandidate]:
    """按置信度排序，去掉框重叠的重复候选，最多留 ``max_candidates`` 个。"""
    ordered = sorted(candidates, key=lambda c: c.confidence, reverse=True)
    kept: list[ObjectCandidate] = []
    for candidate in ordered:
        if any(_iou(candidate.box, other.box) > 0.5 for other in kept):
            continue
        kept.append(candidate)
        if len(kept) >= max(1, max_candidates):
            break
    return kept


def _candidates_text(target: str, candidates: Sequence[ObjectCandidate]) -> str:
    if len(candidates) == 1:
        candidate = candidates[0]
        text = f"我看到了{candidate.label}，在{candidate.position}"
        if candidate.feature:
            text += f"，{candidate.feature}"
        return text + "。"
    first, second = candidates[0], candidates[1]

    def describe(candidate: ObjectCandidate, prefix: str) -> str:
        base = f"{prefix}{candidate.label}在{candidate.position}"
        if candidate.feature:
            base += f"，{candidate.feature}"
        return base

    return f"有两个可能：{describe(first, '一个是')}；{describe(second, '另一个是')}。"


def _dedupe_values(facts: Sequence[AmountFact]) -> list[Decimal]:
    values: list[Decimal] = []
    for fact in facts:
        if fact.value not in values:
            values.append(fact.value)
    return values


def _is_unique_primary(facts: Sequence[AmountFact], primary: AmountFact) -> bool:
    same_field = [f for f in facts if f.field == primary.field]
    return len(same_field) == 1


def _shorten(text: str, limit: int = 80) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for mark in ("。", "！", "？", "，", "；"):
        index = cut.rfind(mark)
        if index >= limit // 2:
            return cut[: index + 1]
    return cut + "……"


def _find_target(question: str) -> str:
    """从找物问题里提取目标名词；提取不到就用去掉疑问词后的问题。"""
    text = (question or "").strip()
    patterns = (
        # “键盘现在在画面中的哪个地方”：先取“在画面…”之前的物体名
        r"^(.{1,20}?)\s*(?:现在|刚才|目前)?\s*(?:在|位于)\s*"
        r"(?:画面|图里|图中|屏幕|照片|视野|眼前|前面)"
        r"(?:里|中|上)?(?:的)?(?:哪里|哪儿|哪个地方|哪个位置|什么位置|何处)",
        r"(?:帮我找|找一下|找一找|寻找|找)\s*([^\s，。？！?！,]{1,20})",
        r"([^\s，。？！?！,]{1,20})\s*(?:在|放|摆)\s*(?:哪|哪里|哪儿)",
        r"(?:有没有|看见|看到)\s*(?:我的)?\s*([^\s，。？！?！,]{1,20})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is not None:
            target = match.group(1).strip()
            target = re.sub(r"^(?:我的|那个|这个|一只|一个|我的那个)", "", target)
            target = re.sub(r"(?:现在|刚才|目前)$", "", target)
            target = re.sub(r"(?:在哪儿|在哪里|在哪|呢|吗|呀|啊)$", "", target)
            if target:
                return target[:20]
    cleaned = re.sub(r"[帮我请一下找看看在哪哪里哪儿呢吗呀啊？?！!。，,]", "", text)
    return (cleaned or text)[:20]


def _dominant_color(rgb) -> str:
    """中心区域平均色的确定性中文命名（用于两图颜色比较）。"""
    try:
        height, width = rgb.shape[0], rgb.shape[1]
    except (AttributeError, IndexError):
        return ""
    if height < 4 or width < 4:
        return ""
    region = rgb[height // 4 : height * 3 // 4, width // 4 : width * 3 // 4]
    mean = region.reshape(-1, 3).mean(axis=0)
    red, green, blue = (float(mean[0]), float(mean[1]), float(mean[2]))
    r, g, b = red / 255.0, green / 255.0, blue / 255.0
    hue, saturation, value = colorsys.rgb_to_hsv(r, g, b)
    if value < 0.15:
        return "黑色"
    if saturation < 0.12:
        return "白色" if value > 0.85 else "灰色"
    degrees = hue * 360.0
    if degrees < 15 or degrees >= 345:
        name = "红色"
    elif degrees < 45:
        name = "橙色"
    elif degrees < 70:
        name = "黄色"
    elif degrees < 160:
        name = "绿色"
    elif degrees < 200:
        name = "青色"
    elif degrees < 255:
        name = "蓝色"
    elif degrees < 290:
        name = "紫色"
    elif degrees < 330:
        name = "粉色"
    else:
        name = "红色"
    if value > 0.85 and saturation < 0.45:
        return "浅" + name
    return name


def _side_by_side(
    left: np.ndarray, right: np.ndarray, *, height: int = 480, gap: int = 12
) -> np.ndarray:
    """把两张图等比缩到同高、中间加白线并排，供单图 VLM 比较。"""

    def _fit(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
            raise ValueError("invalid frame for compare")
        h, w = arr.shape[:2]
        width = max(1, int(round(w * float(height) / float(h))))
        return resize_to(arr, width, int(height))

    canvas_a = _fit(left)
    canvas_b = _fit(right)
    divider = np.full((int(height), max(0, int(gap)), 3), 255, dtype=np.uint8)
    parts = (canvas_a, divider, canvas_b) if divider.shape[1] else (canvas_a, canvas_b)
    return np.concatenate(parts, axis=1)


#: 没有结构化口径、只能交给 VLM 描述的可见维度。
_GENERIC_DIM_RE = re.compile(r"容量|大小|重量|尺寸|规格|形状|包装|材质|成分|口感")


def _compare_fields(question: str) -> tuple[str, ...]:
    """用户明确点到的比较项；空表示“只说了对比，没有指定维度”。

    “不比价格/不看颜色”是明确删掉某一项：先剔掉这些片段，再提取剩余维度。
    """
    text = (question or "").strip()
    text = re.sub(
        r"(?:不比|不看|别比|不要比|不用比|不算|去掉|删掉|不比较)(?:价格|日期|颜色)",
        " ",
        text,
    )
    text = re.sub(r"(?:价格|日期|颜色)(?:不比|不算|不看|去掉|删掉)", " ", text)
    wanted: list[str] = []
    if re.search(r"价格|贵|便宜|钱", text):
        wanted.append("price")
    if re.search(r"日期|保质期|过期|到期|新鲜", text):
        wanted.append("date")
    if re.search(r"颜色|什么色|色", text):
        wanted.append("color")
    return tuple(wanted)


def _compare_dropped_fields(question: str) -> tuple[str, ...]:
    """用户明确说“不比/不看/去掉”的维度；用于从默认三项里排除。"""
    text = question or ""
    dropped: list[str] = []
    for name, pattern in (
        ("price", r"价格|价钱|多少钱"),
        ("date", r"日期|保质期|到期"),
        ("color", r"颜色"),
    ):
        neg = rf"(?:不比|不看|别比|不要比|不用比|不算|去掉|删掉|不比较)(?:{pattern})"
        post = rf"(?:{pattern})(?:不比|不算|不看|去掉|删掉)"
        if re.search(neg, text) or re.search(post, text):
            dropped.append(name)
    return tuple(dropped)



def _compare_items(question: str) -> tuple[str, ...]:
    wanted = _compare_fields(question)
    return wanted or ("price", "date", "color")


def _useful_compare_text(text: str) -> bool:
    """VLM 兜底是否给出了可用描述；整句拒绝/空话不算。"""
    content = (text or "").strip()
    if len(content) < 12:
        return False
    core = content.strip("。！？!?；;，, ")
    return core not in ("看不清", "无法比较", "没法比较", "看不到", "不知道")


def _compare_item(name: str, view_a: _FrameView, view_b: _FrameView) -> CompareItem:
    label = _COMPARE_NAMES[name]
    a_value: str | None = None
    b_value: str | None = None
    note_parts: list[str] = []
    if name == "price":
        a_value = _price_text(view_a)
        b_value = _price_text(view_b)
    elif name == "date":
        a_ok = view_a.date is not None and view_a.date_field == "expiry_date"
        b_ok = view_b.date is not None and view_b.date_field == "expiry_date"
        if a_ok:
            a_value = format_date_cn(view_a.date)
        if b_ok:
            b_value = format_date_cn(view_b.date)
        if view_a.date is not None and not a_ok:
            note_parts.append("A 没读到到期日")
        if view_b.date is not None and not b_ok:
            note_parts.append("B 没读到到期日")
    elif name == "color":
        a_value = view_a.color or None
        b_value = view_b.color or None
    comparable = a_value is not None and b_value is not None
    if not comparable:
        if view_a.ocr_error or view_b.ocr_error:
            note_parts.append("文字没读出来")
        if view_a.price_note:
            note_parts.append("A " + view_a.price_note)
        if view_b.price_note:
            note_parts.append("B " + view_b.price_note)
    return CompareItem(
        name=label,
        unit="元" if name == "price" else "",
        a=a_value,
        b=b_value,
        comparable=comparable,
        note="；".join(note_parts),
    )


def _price_text(view: _FrameView) -> str | None:
    if view.price is None:
        return None
    return format_amount_cn(view.price)


def _compare_conclusion(
    item: CompareItem, name: str, view_a: _FrameView, view_b: _FrameView
) -> str:
    label = item.name
    if name == "price":
        a_value, b_value = view_a.price, view_b.price
        assert a_value is not None and b_value is not None
        if a_value == b_value:
            return f"{label}一样，都是{format_amount_cn(a_value)}"
        bigger = "A" if a_value > b_value else "B"
        smaller = "B" if bigger == "A" else "A"
        return (
            f"{label}方面，{bigger}更贵（{format_amount_cn(max(a_value, b_value))}），"
            f"{smaller}是{format_amount_cn(min(a_value, b_value))}"
        )
    if name == "date":
        a_date, b_date = view_a.date, view_b.date
        assert a_date is not None and b_date is not None
        if a_date == b_date:
            return f"{label}一样，都是{format_date_cn(a_date)}"
        later = "A" if a_date > b_date else "B"
        earlier = "B" if later == "A" else "A"
        later_date = max(a_date, b_date)
        earlier_date = min(a_date, b_date)
        return (
            f"{label}方面，{later}更晚（{format_date_cn(later_date)}），"
            f"{earlier}是{format_date_cn(earlier_date)}"
        )
    if name == "color":
        if view_a.color == view_b.color:
            return f"{label}一样，都是{view_a.color}"
        return f"{label}不同：A 是{view_a.color}，B 是{view_b.color}"
    return f"{label}：A 是{item.a}，B 是{item.b}"


def _as_frame(value: ImageFrame | str | Path) -> ImageFrame:
    if isinstance(value, ImageFrame):
        return value
    if isinstance(value, (str, Path)):
        from aieyes.vision.file_camera import load_frame

        return load_frame(value, source="file")
    raise ValueError(f"unsupported frame source: {type(value)!r}")


__all__ = [
    "INTENT_NAMES",
    "PURPOSE_BY_INTENT",
    "SceneConfig",
    "SceneService",
    "classify_intent",
]
