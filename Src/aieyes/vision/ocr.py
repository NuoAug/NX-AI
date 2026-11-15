"""PP-OCRv6 small 全流水线（onnxruntime，CPU）：DB 检测 + 方向分类 + CTC 识别。

输出 ``interfaces.OcrResult``：每行 ``OcrLine(text, box=(x,y,w,h) 原图像素,
score, polygon)``，按阅读顺序（上→下、左→右）排序。

实现说明
--------
* **不依赖 OpenCV**：DB 后处理用 numpy + 行程并查集做连通域（8 邻接），
  矩形框按 ``expand_box`` 做 unclip 近似（PP-OCR 的 unclip 是多边形外扩，
  对横排文字用矩形近似足够）。
* **字符字典**：优先 ``OcrConfig`` 同目录下 ``*keys*.txt`` / ``*dict*.txt`` /
  其他 ``.txt``；其次找 onnx 模型 metadata；再次 ``AIEYES_OCR_DICT``
  环境变量指定路径。都找不到时抛 ``ModelUnavailable``，消息里列出目录
  内容与识别模型输出类别数，方便定位缺失文件。
* **输入形状自适应**：模型若导出为固定尺寸则 letterbox 到该尺寸，动态轴
  则按 ``det_limit_side_len`` 等比缩放并补齐到 32 的倍数。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aieyes.clock import mono_ms
from aieyes.config import ModelsConfig, OcrConfig
from aieyes.errors import ModelUnavailable
from aieyes.interfaces import OcrLine, OcrResult

from . import image_ops

#: 检测模型归一化参数（PaddleOCR DB 默认）
DET_MEAN = (0.485, 0.456, 0.406)
DET_STD = (0.229, 0.224, 0.225)
#: 识别/方向分类模型归一化参数（PaddleOCR 默认 0.5/0.5）
REC_MEAN = (0.5, 0.5, 0.5)
REC_STD = (0.5, 0.5, 0.5)
#: 单图最多识别的文本行数（防噪声图产生大量小连通域拖慢流水线）
MAX_BOXES = 128
#: 方向分类输入尺寸（PaddleOCR cls 模型固定 3x48x192）
CLS_H, CLS_W = 48, 192


# ---------------------------------------------------------------------------
# 连通域（行程 + 并查集）
# ---------------------------------------------------------------------------
@dataclass
class _Component:
    x1: int
    y1: int
    x2: int  # exclusive
    y2: int  # exclusive
    area: int
    prob_sum: float

    @property
    def score(self) -> float:
        return self.prob_sum / self.area if self.area else 0.0


def _connected_components(mask: np.ndarray, prob: np.ndarray) -> list[_Component]:
    """8 邻接连通域；返回每个连通域的包围盒与概率均值所需统计量。"""
    height, width = mask.shape
    parent: list[int] = []

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:  # 路径压缩
            parent[a], a = root, parent[a]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    runs_by_row: list[list[tuple[int, int, int]]] = []
    roots = []  # 与 run label 同序（union 后需再压缩）
    stats: list[_Component] = []
    prev_runs: list[tuple[int, int, int]] = []
    rows = mask.astype(np.int8)
    for y in range(height):
        row = rows[y]
        if not row.any():
            runs_by_row.append([])
            prev_runs = []
            continue
        padded = np.empty(width + 2, dtype=np.int8)
        padded[0] = 0
        padded[-1] = 0
        padded[1:-1] = row
        diff = np.diff(padded)
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        runs: list[tuple[int, int, int]] = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            label = len(parent)
            parent.append(label)
            stats.append(
                _Component(
                    x1=int(start),
                    y1=y,
                    x2=int(end),
                    y2=y + 1,
                    area=int(end - start),
                    prob_sum=float(prob[y, start:end].sum()),
                )
            )
            runs.append((int(start), int(end), label))
        # 与上一行行程做 8 邻接合并（覆盖范围各向两侧扩 1 像素）
        i = j = 0
        while i < len(prev_runs) and j < len(runs):
            ps, pe, pl = prev_runs[i]
            cs, ce, cl = runs[j]
            if ps - 1 < ce and cs < pe:
                union(pl, cl)
            if pe < ce:
                i += 1
            else:
                j += 1
        runs_by_row.append(runs)
        prev_runs = runs
    # 合并统计到根
    root_to_id: dict[int, int] = {}
    merged: list[_Component] = []
    for runs in runs_by_row:
        for start, end, label in runs:
            root = find(label)
            cid = root_to_id.get(root)
            comp = stats[label]
            if cid is None:
                cid = len(merged)
                root_to_id[root] = cid
                merged.append(
                    _Component(
                        x1=comp.x1,
                        y1=comp.y1,
                        x2=comp.x2,
                        y2=comp.y2,
                        area=comp.area,
                        prob_sum=comp.prob_sum,
                    )
                )
                continue
            agg = merged[cid]
            agg.x1 = min(agg.x1, comp.x1)
            agg.y1 = min(agg.y1, comp.y1)
            agg.x2 = max(agg.x2, comp.x2)
            agg.y2 = max(agg.y2, comp.y2)
            agg.area += comp.area
            agg.prob_sum += comp.prob_sum
    return merged


# ---------------------------------------------------------------------------
# 字符字典
# ---------------------------------------------------------------------------
_DICT_GLOBS = ("*keys*.txt", "*dict*.txt", "*.txt")


def _parse_dict_lines(text: str) -> list[str]:
    chars: list[str] = []
    for line in text.splitlines():
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].lstrip("-").isdigit():
            chars.append(parts[1])
        else:
            chars.append(line)
    return chars


def _load_char_dict(
    ocr_dir: Path, metadata: dict[str, str], *, explicit: str | Path | None = None
) -> tuple[list[str], str]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("AIEYES_OCR_DICT")
    if env:
        candidates.append(Path(env))
    for pattern in _DICT_GLOBS:
        candidates.extend(sorted(ocr_dir.glob(pattern)))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            chars = _parse_dict_lines(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if len(chars) >= 2:
            return chars, str(path)
    # 模型 metadata 里若带字典（多行文本）
    for key, value in metadata.items():
        if not isinstance(value, str) or "\n" not in value:
            continue
        lines = [line for line in value.splitlines() if line]
        if len(lines) >= 100:  # 字符字典至少上百行，避免误用普通描述
            return lines, f"onnx-metadata:{key}"
    files = sorted(p.name for p in ocr_dir.iterdir()) if ocr_dir.is_dir() else []
    raise ModelUnavailable(
        "OCR 字符字典缺失：目录里没有 dict/keys txt，模型 metadata 也没有。"
        f" ocr_dir={ocr_dir} files={files} metadata_keys={sorted(metadata)}"
    )


# ---------------------------------------------------------------------------
# OCR 引擎
# ---------------------------------------------------------------------------
class OcrEngine:
    """实现 ``interfaces.Ocr`` 协议。首次 ``recognize`` 时惰性加载模型。"""

    def __init__(
        self,
        config: OcrConfig | None = None,
        *,
        ocr_dir: str | Path | None = None,
        char_dict: str | Path | None = None,
        num_threads: int = 4,
    ) -> None:
        self.config = config or OcrConfig()
        if ocr_dir is None:
            models = ModelsConfig()
            ocr_dir = models.resolve(models.ocr_dir)
        self.ocr_dir = Path(ocr_dir)
        self._char_dict_path = char_dict
        self._num_threads = max(1, int(num_threads))
        self._det: Any = None
        self._rec: Any = None
        self._cls: Any = None
        self._loaded = False
        self._chars: list[str] = []
        self.dict_source = ""
        self._rec_classes = 0
        self._det_shape: tuple[Any, ...] = ()
        self._rec_shape: tuple[Any, ...] = ()
        self._det_input = ""
        self._rec_input = ""
        self._cls_input = ""
        self.last_init_ms = 0

    # --- 加载 ----------------------------------------------------------
    def load(self) -> "OcrEngine":
        if self._loaded:
            return self
        t0 = mono_ms()
        import onnxruntime as ort  # 延迟导入：不跑 OCR 的进程不用付加载代价

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = self._num_threads
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        det_path = self._model_path(self.config.det_model)
        rec_path = self._model_path(self.config.rec_model)
        if not det_path.is_file():
            raise ModelUnavailable(
                f"OCR det model missing: {det_path}; ocr_dir files="
                f"{self._dir_listing()}"
            )
        if not rec_path.is_file():
            raise ModelUnavailable(
                f"OCR rec model missing: {rec_path}; ocr_dir files="
                f"{self._dir_listing()}"
            )
        self._det = ort.InferenceSession(
            str(det_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        self._rec = ort.InferenceSession(
            str(rec_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        det_input = self._det.get_inputs()[0]
        rec_input = self._rec.get_inputs()[0]
        self._det_input = det_input.name
        self._rec_input = rec_input.name
        self._det_shape = tuple(det_input.shape)
        self._rec_shape = tuple(rec_input.shape)
        rec_out_shape = tuple(self._rec.get_outputs()[0].shape)
        if len(rec_out_shape) == 3 and isinstance(rec_out_shape[2], int):
            self._rec_classes = int(rec_out_shape[2])
        metadata = dict(self._rec.get_modelmeta().custom_metadata_map or {})
        self._chars, self.dict_source = _load_char_dict(
            self.ocr_dir, metadata, explicit=self._char_dict_path
        )
        # 方向分类可选：缺模型就跳过（不影响主流程，只是 180° 图会识别率下降）
        self._cls = None
        if self.config.cls_enabled:
            cls_path = self._model_path(self.config.cls_model)
            if cls_path.is_file():
                try:
                    self._cls = ort.InferenceSession(
                        str(cls_path),
                        sess_options=sess_opts,
                        providers=["CPUExecutionProvider"],
                    )
                    self._cls_input = self._cls.get_inputs()[0].name
                except Exception:  # cls 是增强项，加载失败不阻塞 OCR
                    self._cls = None
        self._loaded = True
        self.last_init_ms = mono_ms() - t0
        return self

    def _model_path(self, rel: str) -> Path:
        path = Path(rel)
        return path if path.is_absolute() else self.ocr_dir / path

    def _dir_listing(self) -> list[str]:
        if not self.ocr_dir.is_dir():
            return [f"(missing dir {self.ocr_dir})"]
        return sorted(p.name for p in self.ocr_dir.iterdir())

    # --- 识别 ----------------------------------------------------------
    def recognize(
        self, rgb: np.ndarray, *, source_frame_id: str = ""
    ) -> OcrResult:
        t0 = mono_ms()
        if not self._loaded:
            self.load()
        image = image_ops.ensure_rgb(rgb)
        height, width = image.shape[:2]
        det_boxes = self._detect(image)
        lines: list[OcrLine] = []
        if det_boxes:
            crops = [self._crop(image, box) for box in det_boxes]
            for (box_xyxy, det_score), crop in zip(det_boxes, crops):
                if crop is None:
                    continue
                text, rec_score = self._recognize_crop(crop)
                if not text:
                    continue
                score = float(min(det_score, rec_score))
                if score < float(self.config.min_confidence):
                    continue
                x1, y1, x2, y2 = box_xyxy
                lines.append(
                    OcrLine(
                        text=text,
                        box=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                        score=round(score, 4),
                        polygon=(
                            (int(x1), int(y1)),
                            (int(x2), int(y1)),
                            (int(x2), int(y2)),
                            (int(x1), int(y2)),
                        ),
                    )
                )
        lines.sort(key=lambda ln: (ln.box[1], ln.box[0]))
        return OcrResult(
            lines=tuple(lines),
            t_ms=mono_ms() - t0,
            source_frame_id=source_frame_id,
            image_size=(width, height),
        )

    # --- DB 检测 -------------------------------------------------------
    def _detect(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        height, width = image.shape[:2]
        tensor, scale_x, scale_y = self._det_input_tensor(image)
        outputs = self._det.run(None, {self._det_input: tensor})
        prob = np.asarray(outputs[0])
        prob = np.squeeze(prob)
        if prob.ndim != 2:
            raise ModelUnavailable(
                f"OCR det output shape unexpected: {np.asarray(outputs[0]).shape}"
            )
        det_h, det_w = prob.shape
        boxes: list[tuple[tuple[int, int, int, int], float]] = []
        mask = prob > float(self.config.det_thresh)
        if not mask.any():
            return boxes
        comps = _connected_components(mask, prob)
        # 概率图坐标 → 原图坐标：先 1/scale 反变换，再裁剪到原图内
        inv_sx = (1.0 / scale_x) if scale_x > 0 else 1.0
        inv_sy = (1.0 / scale_y) if scale_y > 0 else 1.0
        candidates: list[tuple[float, int, int, int, int, float]] = []
        for comp in comps:
            if comp.area < 3 or comp.score < float(self.config.det_box_thresh):
                continue
            box_w = comp.x2 - comp.x1
            box_h = comp.y2 - comp.y1
            if box_w < 2 or box_h < 2:
                continue
            ex1, ey1, ex2, ey2 = image_ops.expand_box(
                (comp.x1, comp.y1, comp.x2, comp.y2), self.config.det_unclip_ratio
            )
            x1 = int(round(ex1 * inv_sx))
            y1 = int(round(ey1 * inv_sy))
            x2 = int(round(ex2 * inv_sx))
            y2 = int(round(ey2 * inv_sy))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            candidates.append((comp.score, x1, y1, x2, y2, comp.score))
        # 最多保留 MAX_BOXES 个（按检测分数降序），再按阅读顺序输出
        candidates.sort(key=lambda item: item[0], reverse=True)
        kept = candidates[:MAX_BOXES]
        boxes = [((c[1], c[2], c[3], c[4]), c[5]) for c in kept]
        boxes.sort(key=lambda item: (item[0][1], item[0][0]))
        return boxes

    def _det_input_tensor(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        height, width = image.shape[:2]
        fixed_h = _fixed_dim(self._det_shape, 2)
        fixed_w = _fixed_dim(self._det_shape, 3)
        if fixed_h and fixed_w:
            resized, scale_x, scale_y = image_ops.resize_keep_aspect(
                image, target_width=fixed_w, target_height=fixed_h
            )
            if resized.shape[:2] != (fixed_h, fixed_w):
                resized = image_ops.pad_to(resized, fixed_w, fixed_h)
        else:
            limit = int(self.config.det_limit_side_len)
            height, width = image.shape[:2]
            if max(height, width) > limit:
                resized, scale_x, scale_y = image_ops.resize_keep_aspect(
                    image, max_side_px=limit, pad_multiple=32
                )
            else:
                # 小图不放大（与 PaddleOCR resize type=0 一致），只补齐到 32 倍数
                pad_w = int(np.ceil(width / 32.0)) * 32
                pad_h = int(np.ceil(height / 32.0)) * 32
                resized = image_ops.pad_to(image, pad_w, pad_h)
                scale_x = scale_y = 1.0
        norm = resized.astype(np.float32) / 255.0
        norm = (norm - np.asarray(DET_MEAN, dtype=np.float32)) / np.asarray(
            DET_STD, dtype=np.float32
        )
        chw = np.ascontiguousarray(norm.transpose(2, 0, 1)[None])
        return chw, float(scale_x), float(scale_y)

    # --- 裁图 + 方向 + 识别 -------------------------------------------
    @staticmethod
    def _crop(
        image: np.ndarray, box: tuple[tuple[int, int, int, int], float]
    ) -> np.ndarray | None:
        (x1, y1, x2, y2), _ = box
        pad = 2
        height, width = image.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return np.ascontiguousarray(image[y1:y2, x1:x2])

    def _recognize_crop(self, crop: np.ndarray) -> tuple[str, float]:
        if self._cls is not None and self._is_upside_down(crop):
            crop = np.ascontiguousarray(crop[::-1, ::-1])
        tensor = self._rec_input_tensor(crop)
        outputs = self._rec.run(None, {self._rec_input: tensor})
        return self._ctc_decode(np.asarray(outputs[0]))

    def _is_upside_down(self, crop: np.ndarray) -> bool:
        try:
            tensor = self._cls_tensor(crop)
            outputs = self._cls.run(None, {self._cls_input: tensor})
            logits = np.asarray(outputs[0]).reshape(-1)
        except Exception:
            return False
        if logits.size < 2:
            return False
        probs = _probabilities(logits)
        return float(probs[1]) >= float(self.config.cls_thresh)

    def _cls_tensor(self, crop: np.ndarray) -> np.ndarray:
        resized, _, _ = image_ops.resize_keep_aspect(
            crop, target_width=CLS_W, target_height=CLS_H
        )
        if resized.shape[:2] != (CLS_H, CLS_W):
            resized = image_ops.pad_to(resized, CLS_W, CLS_H)
        return _to_chw(resized, REC_MEAN, REC_STD)

    def _rec_input_tensor(self, crop: np.ndarray) -> np.ndarray:
        fixed_h = _fixed_dim(self._rec_shape, 2)
        fixed_w = _fixed_dim(self._rec_shape, 3)
        target_h = fixed_h or 48
        height, width = crop.shape[:2]
        new_w = max(8, int(round(width * target_h / float(height))))
        if fixed_w:
            new_w = min(fixed_w, new_w)
            resized, _, _ = image_ops.resize_keep_aspect(
                crop, target_width=new_w, target_height=target_h
            )
            if resized.shape[:2] != (target_h, fixed_w):
                resized = image_ops.pad_to(resized, fixed_w, target_h)
        else:
            resized, _, _ = image_ops.resize_keep_aspect(
                crop, target_width=new_w, target_height=target_h
            )
            if resized.shape[0] != target_h:
                resized = image_ops.pad_to(resized, resized.shape[1], target_h)
        return _to_chw(resized, REC_MEAN, REC_STD)

    # --- CTC 解码 ------------------------------------------------------
    def _ctc_decode(self, outputs: np.ndarray) -> tuple[str, float]:
        arr = outputs
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise ModelUnavailable(f"OCR rec output shape unexpected: {outputs.shape}")
        classes = int(arr.shape[1])
        num_chars = len(self._chars)
        # PP-OCRv6 rec 实测：输出 18710 类 = blank(0) + 字典 18708 字符 + 末尾
        # 1 个 ASCII 空格类（字典里的 U+3000 在 index 1748，不含 ASCII 空格）。
        if classes == num_chars + 2:
            blank, offset, extra_space = 0, 1, True
        elif classes == num_chars + 1:
            blank, offset, extra_space = 0, 1, False
        elif classes == num_chars:  # 无 blank 的模型（少见）
            blank, offset, extra_space = -1, 0, False
        else:
            raise ModelUnavailable(
                f"OCR rec output classes {classes} 与字典 {num_chars} 字符不匹配；"
                f"dict={self.dict_source}"
            )
        probs = _probabilities(arr)
        indices = probs.argmax(axis=1)
        max_probs = probs.max(axis=1)
        chars: list[str] = []
        scores: list[float] = []
        prev = -2
        for t, index in enumerate(indices.tolist()):
            if index == blank:
                prev = index
                continue
            if index == prev:
                continue
            prev = index
            char_index = index - offset
            if 0 <= char_index < num_chars:
                chars.append(self._chars[char_index])
                scores.append(float(max_probs[t]))
            elif extra_space and char_index == num_chars:
                chars.append(" ")
                scores.append(float(max_probs[t]))
        if not chars:
            return "", 0.0
        return "".join(chars), float(np.mean(scores))

    # --- 便捷方法 ------------------------------------------------------
    def warmup(self) -> int:
        """用纯色小图跑一次完整流水线，返回耗时毫秒（首次 ORT 初始化较慢）。"""
        if not self._loaded:
            self.load()
        dummy = np.full((64, 96, 3), 255, dtype=np.uint8)
        result = self.recognize(dummy)
        return result.t_ms

    def close(self) -> None:
        self._det = self._rec = self._cls = None
        self._loaded = False


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _fixed_dim(shape: Sequence[Any], index: int) -> int:
    if len(shape) <= index:
        return 0
    value = shape[index]
    if isinstance(value, int) and value > 0:
        return int(value)
    return 0


def _to_chw(image: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    norm = image.astype(np.float32) / 255.0
    norm = (norm - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(norm.transpose(2, 0, 1)[None])


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _probabilities(outputs: np.ndarray) -> np.ndarray:
    """模型输出已是概率（PP-OCRv6 rec/cls 实测每行和为 1）就直接用，否则 softmax。"""
    arr = np.asarray(outputs, dtype=np.float32)
    if arr.size and float(arr.min()) >= -1e-4 and float(arr.max()) <= 1.0 + 1e-4:
        row_sums = arr.sum(axis=-1)
        if bool(np.all(np.abs(row_sums - 1.0) <= 1e-2)):
            return np.clip(arr, 0.0, 1.0)
    return _softmax(arr)
