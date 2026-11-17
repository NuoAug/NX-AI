"""测试台OCR沿用项目加载/字典逻辑，仅关闭CPU内存池预留与动态形状内存模板。
加载方法取自本机已核对的项目版本；模型、精度和预处理保持一致。
"""
from __future__ import annotations
from aieyes.vision.ocr import OcrEngine as ProjectOcrEngine, _load_char_dict
from aieyes.clock import mono_ms
from aieyes.errors import ModelUnavailable

class OcrEngine(ProjectOcrEngine):
    def load(self) -> "OcrEngine":
        if self._loaded:
            return self
        t0 = mono_ms()
        import onnxruntime as ort  # 延迟导入：不跑 OCR 的进程不用付加载代价

        sess_opts = ort.SessionOptions()
        sess_opts.enable_cpu_mem_arena = False
        sess_opts.enable_mem_pattern = False
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
