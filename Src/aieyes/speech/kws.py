"""唤醒词检测：sherpa-onnx KeywordSpotter 封装。

模型自带 ``keywords.txt`` 里没有“小冰”这类自定义唤醒词，因此这里按
``tokens.txt`` 的**实际 token 串**生成关键词行（格式
``<token> <token> ... @<唤醒词>``），例如::

    x iǎo b īng @小冰

KWS 的 tokens 是拼音声母/韵母级（带声调符号），所以先探测 ``tokens.txt``
是否存在对应 token，再做“去声调 / 整音节”两种回退；生成结果写入临时目录，
不修改模型资产。多唤醒词共用一个 keywords 文件，检测成功后重置流并做
cooldown 去抖。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from aieyes.clock import mono_ms
from aieyes.config import KwsConfig, ModelsConfig
from aieyes.errors import ConfigError, ModelUnavailable
from aieyes.interfaces import AudioFrame

#: 常用唤醒词汉字的拼音（带声调，声母/韵母拆分）。自定义唤醒词若不在表内，
#: 可把 ``wake_phrases`` 写成空格分隔的 pinyin token（如 ``"x iǎo b īng"``）。
_PINYIN: dict[str, tuple[str, str]] = {
    "小": ("x", "iǎo"),
    "冰": ("b", "īng"),
    "你": ("n", "ǐ"),
    "好": ("h", "ǎo"),
    "军": ("j", "ūn"),
    "哥": ("g", "ē"),
    "天": ("t", "iān"),
    "猫": ("m", "āo"),
    "精": ("j", "īng"),
    "灵": ("l", "íng"),
    "助": ("zh", "ù"),
    "手": ("sh", "ǒu"),
}

_TONE_MARKS = "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ"
_TONE_BASE = "aaaaeeeeiiiioooouuuuüüüü"
_TONE_TABLE = str.maketrans(_TONE_MARKS, _TONE_BASE)


def load_tokens(tokens_file: str | Path) -> set[str]:
    """读取 ``tokens.txt``（每行 ``token id``），返回 token 集合。"""
    path = Path(tokens_file)
    if not path.is_file():
        raise ModelUnavailable(f"KWS tokens.txt 不存在: {path}")
    tokens: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts:
            tokens.add(parts[0])
    return tokens


def _syllable_variants(initial: str, final: str) -> list[str]:
    """声母+韵母的可能写法：带调韵母、去调韵母、整音节。"""
    base = final.translate(_TONE_TABLE)
    out = [final, base, initial + final, initial + base]
    seen: list[str] = []
    for tok in out:
        if tok and tok not in seen:
            seen.append(tok)
    return seen


def _syllable_tokens(initial: str, final: str, tokens: set[str]) -> list[str]:
    """单字 → token 序列。

    模型词表把拼音拆成“声母/带调韵母”两个 token（如 ``n ǐ``），因此优先输出
    ``[声母, 韵母]``；若词表只收录整音节则退化为单个 token。
    """
    if initial in tokens and final in tokens:
        return [initial, final]
    if final in tokens:
        return [final]
    base = final.translate(_TONE_TABLE)
    for tok in (initial + final, initial + base, base):
        if tok in tokens:
            return [tok]
    return []


def phrase_to_tokens(phrase: str, tokens: set[str]) -> list[str]:
    """把一个唤醒词转成 tokens.txt 里的 token 序列。

    - 已是 pinyin 形式（含空格且不含汉字，如 ``x iǎo b īng``）：按空格切分并逐个校验；
    - 汉字：查内置拼音表，按“声母+带调韵母 → 整音节”回退；
    - 表内没有的汉字抛 ``ConfigError``，提示改用 pinyin 写法。
    """
    phrase = phrase.strip()
    if not phrase:
        raise ConfigError("唤醒词不能为空")
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in phrase)
    if " " in phrase and not has_cjk:
        parts = [p for p in phrase.split() if p]
        missing = [p for p in parts if p not in tokens]
        if missing:
            raise ConfigError(
                f"唤醒词 {phrase!r} 中的 token {missing} 不在 tokens.txt；"
                f"请检查模型 token 表或改用其它写法"
            )
        return parts

    result: list[str] = []
    for ch in phrase:
        entry = _PINYIN.get(ch)
        if entry is None:
            raise ConfigError(
                f"无法为唤醒词字符 {ch!r} 生成拼音 token；"
                f"请在 KwsConfig.wake_phrases 里用空格分隔的 pinyin 写法（如 'x iǎo b īng'）"
            )
        picked = _syllable_tokens(*entry, tokens)
        if not picked:
            variants = _syllable_variants(*entry)
            raise ConfigError(
                f"唤醒词字符 {ch!r} 的候选 token {variants} 均不在 tokens.txt；"
                f"模型 token 表可能不是拼音声母/韵母格式"
            )
        result.extend(picked)
    return result


def build_keywords_text(
    tokens: set[str],
    phrases: Sequence[str],
    *,
    existing_lines: Iterable[str] = (),
) -> str:
    """生成 keywords 文件内容（保留 existing 里 token 全部存在的行）。"""
    lines: list[str] = []
    seen_phrases: set[str] = set()
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        body = stripped.split("@", 1)[0].split()
        if body and all(tok in tokens for tok in body) and stripped not in lines:
            lines.append(stripped)
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase or phrase in seen_phrases:
            continue
        seen_phrases.add(phrase)
        toks = phrase_to_tokens(phrase, tokens)
        line = " ".join(toks) + f" @{phrase}"
        if line not in lines:
            lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def _pick_model_file(model_dir: Path, stem: str) -> Path:
    """按 int8 优先挑选 encoder/decoder/joiner onnx。"""
    for pattern in (f"{stem}.int8.onnx", f"{stem}*.int8.onnx", f"{stem}.onnx", f"{stem}*.onnx"):
        hits = sorted(model_dir.glob(pattern))
        if hits:
            return hits[0]
    raise ModelUnavailable(f"{model_dir} 下找不到 {stem}*.onnx")


class SherpaKws:
    """sherpa-onnx 关键词检测器；``accept`` 在采集线程同步调用。"""

    def __init__(
        self,
        cfg: KwsConfig | None = None,
        models: ModelsConfig | None = None,
        *,
        model_dir: str | Path | None = None,
        keywords_file: str | Path | None = None,
        num_threads: int = 2,
    ) -> None:
        self.cfg = cfg or KwsConfig()
        self.models = models or ModelsConfig()
        self.model_dir = Path(model_dir) if model_dir else self.models.resolve(self.models.kws_dir)
        self._keywords_file = Path(keywords_file) if keywords_file else (
            Path(self.cfg.keywords_file) if self.cfg.keywords_file else None
        )
        self.num_threads = int(num_threads)
        self._spotter = None
        self._stream = None
        self._keywords_path: Path | None = None
        self._last_hit_ms = -10**12
        self._last_keyword = ""

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._spotter is not None

    @property
    def keywords_path(self) -> Path | None:
        return self._keywords_path

    # ------------------------------------------------------------------
    def _prepare_keywords(self) -> Path:
        model_dir = self.model_dir
        if not model_dir.is_dir():
            raise ModelUnavailable(f"KWS 模型目录不存在: {model_dir}")
        tokens = load_tokens(model_dir / "tokens.txt")
        existing_lines: list[str] = []
        stock = model_dir / "keywords.txt"
        if stock.is_file():
            existing_lines = stock.read_text(encoding="utf-8").splitlines()
        text = build_keywords_text(tokens, self.cfg.wake_phrases, existing_lines=existing_lines)
        if not text.strip():
            raise ConfigError("唤醒词为空且模型自带 keywords.txt 无可用的行")
        cache_dir = Path(tempfile.gettempdir()) / "aieyes_kws"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"keywords_{abs(hash(text)) & 0xFFFFFFFF:08x}.txt"
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
        return target

    def _ensure_loaded(self) -> None:
        if self._spotter is not None:
            return
        try:
            import sherpa_onnx  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover
            raise ModelUnavailable(f"未安装 sherpa-onnx: {exc}") from exc
        model_dir = self.model_dir
        tokens = model_dir / "tokens.txt"
        if not tokens.is_file():
            raise ModelUnavailable(f"KWS tokens.txt 不存在: {tokens}")
        keywords = self._keywords_file or self._prepare_keywords()
        if not keywords.is_file() or keywords.stat().st_size == 0:
            raise ModelUnavailable(f"KWS keywords 文件无效: {keywords}")
        encoder = _pick_model_file(model_dir, "encoder")
        decoder = _pick_model_file(model_dir, "decoder")
        joiner = _pick_model_file(model_dir, "joiner")
        self._keywords_path = keywords
        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            keywords_file=str(keywords),
            num_threads=self.num_threads,
            sample_rate=16000,  # sherpa-onnx 2.0.1 的 nanobind 参数要求 int，传 float 会 TypeError
            feature_dim=80,
            max_active_paths=4,
            keywords_score=float(self.cfg.keywords_score),
            keywords_threshold=float(self.cfg.keywords_threshold),
            num_trailing_blanks=int(self.cfg.num_trailing_blanks),
            provider="cpu",
        )
        self._stream = self._spotter.create_stream()

    # ------------------------------------------------------------------
    def accept(self, frame: AudioFrame) -> str | None:
        """喂一帧；命中返回唤醒词文本，否则 ``None``（含 cooldown 抑制）。"""
        self._ensure_loaded()
        stream = self._stream
        if stream is None:
            stream = self._stream = self._spotter.create_stream()
        samples = frame.samples
        stream.accept_waveform(16000.0, samples)
        while self._spotter.is_ready(stream):
            self._spotter.decode_stream(stream)
        result = self._spotter.get_result(stream)
        if not result:
            return None
        keyword = str(result).strip()
        if "@" in keyword:  # 兼容部分版本返回整行
            keyword = keyword.rsplit("@", 1)[-1].strip()
        now = mono_ms()
        if now - self._last_hit_ms < int(self.cfg.cooldown_ms):
            self._reset_stream()
            return None
        self._last_hit_ms = now
        self._last_keyword = keyword
        self._reset_stream()
        return keyword

    def flush(self, pad_ms: int = 660) -> str | None:
        """喂尾部静音并结束输入，用于文件回放/单测里逼出尾随 blank 后的命中。

        实时麦克风流不需要它：连续静音会自然补够 trailing blanks。
        """
        self._ensure_loaded()
        stream = self._stream
        if stream is None:
            return None
        stream.accept_waveform(16000.0, np.zeros(int(16000 * pad_ms / 1000), dtype=np.float32))
        try:
            stream.input_finished()
        except Exception:
            pass
        keyword = None
        while self._spotter.is_ready(stream):
            self._spotter.decode_stream(stream)
            result = self._spotter.get_result(stream)
            if result:
                keyword = str(result).strip()
                break
        if keyword is not None:
            self._reset_stream()
            return keyword
        self._reset_stream()
        return None

    def _reset_stream(self) -> None:
        if self._spotter is None:
            self._stream = None
            return
        self._stream = self._spotter.create_stream()

    def reset(self) -> None:
        self._reset_stream()
        self._last_hit_ms = -10**12
        self._last_keyword = ""

    def close(self) -> None:
        self._stream = None
        self._spotter = None


__all__ = [
    "SherpaKws",
    "build_keywords_text",
    "load_tokens",
    "phrase_to_tokens",
]
