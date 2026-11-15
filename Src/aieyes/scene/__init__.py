"""场景层：取图 → 读字/价格/日期 → 找物/描述/比较，答案带证据与来源标签。

对外只需要 :class:`SceneService` 与 :class:`FrameProvider` 的两个实现：

>>> from aieyes.scene import FileFrameProvider, SceneService
>>> provider = FileFrameProvider("Tests/Fixtures/images/text_cn_price.png")
>>> service = SceneService(provider=provider)
>>> answer = service.answer("这个多少钱")
>>> answer.intent, float(answer.confidence)
"""

from .errors import FrameAcquisitionError, FrameQualityError, SceneAnalysisError, SceneError
from .models import (
    SOURCE_LABELS,
    CompareAnswer,
    CompareItem,
    FrameAcquisition,
    FrameAttempt,
    ObjectCandidate,
    ObjectCandidates,
    SceneAnswer,
    SourceTag,
)
from .parse import (
    AmountFact,
    DateFact,
    SpokenPurchase,
    compute_purchase,
    extract_amounts,
    extract_dates,
    format_amount_cn,
    format_date_cn,
    parse_amount,
    parse_date,
    pick_date,
)
from .providers import (
    CameraFrameProvider,
    FileFrameProvider,
    FramePolicy,
    FrameProvider,
)
from .service import (
    INTENT_NAMES,
    PURPOSE_BY_INTENT,
    SceneConfig,
    SceneService,
    classify_intent,
)

__all__ = [
    "INTENT_NAMES",
    "PURPOSE_BY_INTENT",
    "SOURCE_LABELS",
    "AmountFact",
    "CameraFrameProvider",
    "CompareAnswer",
    "CompareItem",
    "DateFact",
    "FileFrameProvider",
    "FrameAcquisition",
    "FrameAcquisitionError",
    "FrameAttempt",
    "FramePolicy",
    "FrameProvider",
    "FrameQualityError",
    "ObjectCandidate",
    "ObjectCandidates",
    "SceneAnalysisError",
    "SceneAnswer",
    "SceneConfig",
    "SceneError",
    "SceneService",
    "SourceTag",
    "SpokenPurchase",
    "classify_intent",
    "compute_purchase",
    "extract_amounts",
    "extract_dates",
    "format_amount_cn",
    "format_date_cn",
    "parse_amount",
    "parse_date",
    "pick_date",
]
