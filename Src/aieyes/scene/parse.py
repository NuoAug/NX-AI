"""价格 / 日期确定性解析（纯函数，不经过模型）。

输入是 OCR 原文或用户口述文本；输出带来源字段。规则：

* 金额用 :class:`decimal.Decimal` 计算，人民币最多两位小数；负数、编号/日期
  数字不当作金额；支持 ``12.50元``/``12元5角``/``￥12.50``/``1,234.50元``、
  以及整串就是数字的 ``12.5``。
* 日期只认“年在前、带分隔符”的明确写法（``2026-11-11``/``2026年11月11日``/
  ``2026.11.11``/``2026/11/11``）；裸 8 位数字（``20261111``）默认不解析——
  它更可能是批号，需要显式 ``allow_compact=True`` 才解析。
* :func:`extract_dates` / :func:`extract_amounts` 逐行给字段标签：批号永远不是
  日期，保质期只是时长；相邻价签会产出多条金额（由上层判 ambiguous）。

回答纪律：本模块只从原文提取，不补全、不猜测；无法确定就返回 ``None``。
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Sequence

from aieyes.interfaces import OcrLine

#: 人民币金额量化到分
CENT = Decimal("0.01")

# 数字：允许千分位（半角/全角逗号）；小数最多两位
_NUM = r"(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d{1,2})?"
#: 数字前面的边界：排除负号、数字、小数点/逗号
_NUM_START = r"(?<![\d.,，\-−])"
#: 数字后面的边界：不能紧跟数字（“￥12.567”不算 12.56）
_NUM_END = r"(?!\d)"

_YUAN_JIAO_FEN = re.compile(
    _NUM_START + r"(" + _NUM + r")\s*(?:元|圆|块钱|块)\s*(\d)\s*[角毛]"
    r"(?:\s*(\d)\s*分)?"
)
#: 口语写法 “12块5” / “12元5角”（数字与单位粘连，避免把 “12元 5件” 读成 12.5）
_KUAI_JIAO = re.compile(
    _NUM_START + r"(" + _NUM + r")(?:块|元)(\d)(?:毛|角)?"
    r"(?![\d个件袋瓶盒包斤份支条罐人元块米克升毫])"
)
_CURRENCY_SYMBOL = re.compile(
    r"[￥¥]\s*(" + _NUM + r")" + _NUM_END
)
_UNIT_AFTER = re.compile(
    _NUM_START + r"(" + _NUM + r")\s*(?:元|圆|块钱|块)" + _NUM_END
)
#: 整行就是一个带两位小数的数字（价签常见，如 “12.50”）
_BARE_DECIMAL = re.compile(r"^\s*(\d{1,4}\.\d{1,2})\s*$")
_JIAO_FULL = re.compile(r"^\s*(?:[￥¥]\s*)?(\d)\s*[角毛]\s*$")
_FEN_FULL = re.compile(r"^\s*(\d)\s*分(?:钱)?\s*$")

#: 整串就是金额时允许的前缀（标签 / 币种）
_AMOUNT_FULL = re.compile(
    r"^\s*(?:价格|售价|标价|单价|总价|原价|现价|会员价|优惠价|"
    r"price|金额|合计|小计)?\s*[:：=]?\s*[￥¥]?\s*"
    r"(" + _NUM + r")\s*(?:元|圆|块)?\s*$",
    re.IGNORECASE,
)
#: 行内有这些词时，允许无单位数字按金额解释（“价格 12.5”）
_PRICE_CONTEXT = re.compile(
    r"价格|售价|标价|单价|总价|原价|现价|会员价|优惠价|折后|price|金额|合计",
    re.IGNORECASE,
)
#: 裸数字后面跟这些字符时不是金额（日期/号码/单位）

# 日期：年在前的明确格式
_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<!\d)(\d{4})\s*[-–—/.]\s*(\d{1,2})\s*[-–—/.]\s*(\d{1,2})(?!\d)"),
    re.compile(r"(?<!\d)(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?(?!\d)"),
)
_COMPACT_DATE = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)")

# 日期字段关键词（按优先级：批号 > 到期 > 生产 > 保质期）
_BATCH_KEYS = re.compile(r"批号|批次|生产批号|\bLOT\b|\bBN\b|\bBATCH\b", re.IGNORECASE)
_EXPIRY_KEYS = re.compile(
    r"到期|有效期至|限期使用|此日期前|保质期至|失效日期|"
    r"\bEXP\b|USE\s*BY|BEST\s*BEFORE",
    re.IGNORECASE,
)
_PRODUCTION_KEYS = re.compile(
    r"生产日期|制造日期|生产于|灌装日期|包装日期|\bMFD\b|\bMFG\b|\bPROD\b",
    re.IGNORECASE,
)
_SHELF_KEYS = re.compile(r"保质期|储存期|货架期|保存期")
_SHELF_DURATION = re.compile(r"(\d{1,3})\s*(?:个)?(?:月|天|日|年)")

_PRICE_FIELD_KEYS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"会员价"), "member_price"),
    (re.compile(r"原价"), "original_price"),
    (re.compile(r"现价|促销价|特价|优惠价"), "promo_price"),
    (re.compile(r"单价|每\s*(?:斤|kg|千克|克|袋|件|盒|瓶)|/\s*(?:斤|kg|kg|千克|克)"), "unit_price"),
    (re.compile(r"总价|合计|小计|总计"), "total_price"),
)


# ---------------------------------------------------------------------------
# 金额
# ---------------------------------------------------------------------------
def _to_decimal(raw: str) -> Decimal | None:
    try:
        value = Decimal(raw.replace(",", "").replace("，", ""))
    except (InvalidOperation, ValueError):
        return None
    if not value.is_finite() or value < 0:
        return None
    return value


def _amount_from_match(text: str, match: re.Match[str]) -> Decimal | None:
    """把一次带币种/单位的正则命中转成金额。"""
    return _to_decimal(match.group(1))


def parse_amount(text: str) -> Decimal | None:
    """从一段原文里解析人民币金额；无法确定返回 ``None``。

    支持 ``12.50元`` / ``12元5角`` / ``12元5角3分`` / ``￥12.50`` /
    ``1,234.50元`` / 整串 ``12.5``；负数、日期/批号数字、无单位且无价格
    上下文的数字不认（例如 ``2026-11-11``、``批号 20261111``）。
    """
    if not text or not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None

    if raw.startswith(("-", "−")):
        return None

    match = _YUAN_JIAO_FEN.search(raw)
    if match is not None:
        yuan = _to_decimal(match.group(1))
        if yuan is None:
            return None
        jiao = int(match.group(2))
        fen = int(match.group(3) or 0)
        return (yuan + Decimal(jiao) / 10 + Decimal(fen) / 100).quantize(CENT)

    match = _CURRENCY_SYMBOL.search(raw)
    if match is not None:
        return _amount_from_match(raw, match)

    match = _KUAI_JIAO.search(raw)
    if match is not None:
        yuan = _to_decimal(match.group(1))
        if yuan is not None:
            return (yuan + Decimal(int(match.group(2))) / 10).quantize(CENT)

    match = _UNIT_AFTER.search(raw)
    if match is not None:
        return _amount_from_match(raw, match)

    match = _JIAO_FULL.match(raw)
    if match is not None:
        return (Decimal(int(match.group(1))) / 10).quantize(CENT)

    match = _FEN_FULL.match(raw)
    if match is not None:
        return (Decimal(int(match.group(1))) / 100).quantize(CENT)

    if _PRICE_CONTEXT.search(raw):
        match = _AMOUNT_FULL.match(raw)
        if match is not None:
            return _money_like(match.group(1))

    # 整串就是数字（允许前后标点）："12.5"
    match = _AMOUNT_FULL.match(raw)
    if match is not None and not re.search(r"[年月日/\-]", raw.replace("价格", "")):
        value = _money_like(match.group(1))
        if value is not None:
            return value
    return None


def _money_like(raw: str) -> Decimal | None:
    """无单位数字当金额时的约束：8 位以上整数更像批号/日期，不认。"""
    value = _to_decimal(raw)
    if value is None:
        return None
    if value == value.to_integral_value() and len(raw.replace(",", "").replace("，", "")) > 6:
        return None
    return value.quantize(CENT) if value.as_tuple().exponent < -2 else value


def format_amount_cn(value: Decimal) -> str:
    """金额转口语化中文（TTS 用）：``12.50`` → ``12元5角``。"""
    value = value.quantize(CENT, rounding=ROUND_HALF_UP)
    sign = "-" if value < 0 else ""
    value = abs(value)
    yuan = int(value)
    jiao = int((value * 10) % 10)
    fen = int((value * 100) % 10)
    if jiao == 0 and fen == 0:
        return f"{sign}{yuan}元"
    if jiao and fen:
        return f"{sign}{yuan}元{jiao}角{fen}分"
    if jiao:
        return f"{sign}{yuan}元{jiao}角"
    return f"{sign}{yuan}元零{fen}分"


def format_date_cn(value: _dt.date) -> str:
    return f"{value.year}年{value.month}月{value.day}日"


# ---------------------------------------------------------------------------
# 日期
# ---------------------------------------------------------------------------
def parse_date(text: str, *, allow_compact: bool = False) -> _dt.date | None:
    """从一段原文里解析明确日期；无法确定返回 ``None``。

    ``allow_compact=True`` 时才接受裸 8 位数字（``20261111``），默认拒绝，
    因为包装上的 8 位数字常常是批号而不是日期。
    """
    if not text or not isinstance(text, str):
        return None
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                return _dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                continue
    if allow_compact:
        for match in _COMPACT_DATE.finditer(text):
            try:
                return _dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# 逐行提取（带字段标签）
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AmountFact:
    """一条金额事实：原文、标准值、字段、行号与来源。"""

    raw_text: str
    value: Decimal
    field: str = "price"
    line_index: int = 0
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class DateFact:
    """一条日期事实；``field`` 取值见模块说明。"""

    raw_text: str
    value: _dt.date | None
    field: str = "unlabeled_date"  # expiry_date|production_date|shelf_life|batch_id|unlabeled_date
    line_index: int = 0
    score: float = 0.0


def _lines_of(lines: Sequence[OcrLine | str]) -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    for line in lines:
        if isinstance(line, OcrLine):
            result.append((line.text, float(line.score)))
        else:
            result.append((str(line), 0.0))
    return result


def _price_field_at(text: str, pos: int) -> str:
    """按金额出现位置往前找最近的价签关键词，决定字段名。"""
    best: tuple[int, str] | None = None
    for pattern, field in _PRICE_FIELD_KEYS:
        for match in pattern.finditer(text, 0, pos + 1):
            if best is None or match.start() > best[0]:
                best = (match.start(), field)
    return best[1] if best is not None else "price"


def extract_amounts(lines: Sequence[OcrLine | str]) -> tuple[AmountFact, ...]:
    """逐行提取金额事实；同一行多个金额（如“原价15元，现价12.5元”）都保留。"""
    facts: list[AmountFact] = []
    for index, (text, score) in enumerate(_lines_of(lines)):
        if not text.strip():
            continue
        matches: list[tuple[int, Decimal]] = []
        for match in _YUAN_JIAO_FEN.finditer(text):
            yuan = _to_decimal(match.group(1))
            if yuan is not None:
                jiao = int(match.group(2))
                fen = int(match.group(3) or 0)
                matches.append(
                    (match.start(), (yuan + Decimal(jiao) / 10 + Decimal(fen) / 100).quantize(CENT))
                )
        for match in _CURRENCY_SYMBOL.finditer(text):
            value = _amount_from_match(text, match)
            if value is not None:
                matches.append((match.start(), value))
        for match in _KUAI_JIAO.finditer(text):
            yuan = _to_decimal(match.group(1))
            if yuan is not None:
                matches.append(
                    (match.start(), (yuan + Decimal(int(match.group(2))) / 10).quantize(CENT))
                )
        for match in _UNIT_AFTER.finditer(text):
            value = _amount_from_match(text, match)
            if value is not None:
                matches.append((match.start(), value))

        if not matches and _PRICE_CONTEXT.search(text):
            # 无单位数字 + 价格上下文（“价格 12.5”）
            match = _AMOUNT_FULL.match(text)
            if match is not None:
                value = _money_like(match.group(1))
                if value is not None:
                    matches.append((match.start(1), value))
        if not matches:
            bare = _BARE_DECIMAL.match(text)
            if bare is not None:
                value = _money_like(bare.group(1))
                if value is not None:
                    matches.append((bare.start(1), value))

        # 位置相近的重复命中（“12块5” 同时命中 12块 与 12块5）只留跨度大的
        seen: list[tuple[int, Decimal]] = []
        for pos, value in sorted(matches, key=lambda item: item[0]):
            if any(abs(pos - other) <= 1 for other, _ in seen):
                continue
            seen.append((pos, value))
        for pos, value in seen:
            facts.append(
                AmountFact(
                    raw_text=text,
                    value=value,
                    field=_price_field_at(text, pos),
                    line_index=index,
                    score=score,
                )
            )
    return tuple(facts)


def extract_dates(lines: Sequence[OcrLine | str]) -> tuple[DateFact, ...]:
    """逐行提取日期事实，并区分到期日/生产日期/保质期/批号。

    * 行内有“批号/LOT/BN”字样：``field="batch_id"``，即使写了日期也不当日期；
    * 有明确日期 + 到期词 → ``expiry_date``，生产词 → ``production_date``，否则
      ``unlabeled_date``；
    * 只有“保质期 12 个月”这类时长 → ``shelf_life``（不是日期）；
    * 裸 8 位数字（可能批号）→ ``batch_id`` 保留原文、``value=None``。
    """
    facts: list[DateFact] = []
    for index, (text, score) in enumerate(_lines_of(lines)):
        if not text.strip():
            continue
        upper = text
        if _BATCH_KEYS.search(upper):
            facts.append(
                DateFact(text, None, field="batch_id", line_index=index, score=score)
            )
            continue
        value = parse_date(text)
        if value is not None:
            if _EXPIRY_KEYS.search(upper):
                field = "expiry_date"
            elif _PRODUCTION_KEYS.search(upper):
                field = "production_date"
            else:
                field = "unlabeled_date"
            facts.append(DateFact(text, value, field=field, line_index=index, score=score))
            continue
        if _SHELF_KEYS.search(upper):
            facts.append(
                DateFact(text, None, field="shelf_life", line_index=index, score=score)
            )
            continue
        compact = _COMPACT_DATE.search(text)
        if compact is not None and len(re.sub(r"\D", "", text)) >= 8:
            facts.append(
                DateFact(text, None, field="batch_id", line_index=index, score=score)
            )
    return tuple(facts)


def pick_date(facts: Sequence[DateFact], *, prefer: str = "expiry_date") -> DateFact | None:
    """从日期事实里选一条用于回答：优先到期日，其次明确日期，最后无标签日期。"""
    for fact in facts:
        if fact.field == prefer and fact.value is not None:
            return fact
    for fact in facts:
        if fact.value is not None and fact.field not in ("batch_id", "shelf_life"):
            return fact
    for fact in facts:
        if fact.value is not None:
            return fact
    return None


# ---------------------------------------------------------------------------
# 口述价格 × 数量 → 计算（不取图）
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SpokenPurchase:
    """用户口述的“单价 × 数量”上下文与确定性结果。"""

    unit_price: Decimal
    quantity: int
    total: Decimal
    unit: str = "元"


_QTY = re.compile(
    r"(?:买|要|来|拿|装|有)\s*(\d{1,4})\s*(个|件|袋|瓶|盒|包|斤|份|支|条|罐)"
)
_ASK_TOTAL = re.compile(r"多少钱|一共|总共|合计|总共多少|算一下|算算|要多少|花多少")


def compute_purchase(question: str) -> SpokenPurchase | None:
    """识别“单价 × 数量”的口述计算题；条件不齐返回 ``None``。

    例：``每件12块5，买3件一共多少钱`` → 12.50 × 3 = 37.50。
    只用确定性 Decimal 计算，不取图、不调模型。
    """
    if not question or not isinstance(question, str):
        return None
    if not _ASK_TOTAL.search(question):
        return None
    price = parse_amount(question)
    if price is None:
        return None
    match = _QTY.search(question)
    if match is None:
        return None
    quantity = int(match.group(1))
    if quantity <= 0:
        return None
    total = (price * quantity).quantize(CENT, rounding=ROUND_HALF_UP)
    return SpokenPurchase(unit_price=price, quantity=quantity, total=total)
