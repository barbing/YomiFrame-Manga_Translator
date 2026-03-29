# -*- coding: utf-8 -*-
"""PaddleOCR text detection wrapper."""
from __future__ import annotations
import os
from typing import List, Tuple
from app.models.resolution import resolve_paddle_local_det_dir, resolve_paddle_system_det_dir

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_mkldnn", "0")

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None


class PaddleTextDetector:
    def __init__(self, use_gpu: bool) -> None:
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR is not installed.")
        try:
            import paddle
            paddle.set_flags({"FLAGS_use_mkldnn": 0, "FLAGS_enable_onednn": 0})
        except Exception:
            pass
        det_model_dir = None if resolve_paddle_system_det_dir() else resolve_paddle_local_det_dir()

        self._detector = PaddleOCR(
            det_model_dir=det_model_dir,
            det=True,
            rec=False,
            cls=False,
            lang="japan",
            use_gpu=use_gpu,
            ir_optim=False,
            use_tensorrt=False,
            show_log=False,
            det_db_thresh=0.2,
            det_db_box_thresh=0.4,
            det_db_unclip_ratio=1.7,
            det_limit_side_len=1280,
        )

    def detect(self, image_path: str) -> List[Tuple[List[List[float]], float]]:
        image = _read_image(image_path)
        if image is None:
            return []
        return self.detect_image(image)

    def detect_image(self, image) -> List[Tuple[List[List[float]], float]]:
        if image is None:
            return []
        dt_boxes, raw_scores = self._detector.text_detector(image)
        if dt_boxes is None:
            return []
        scores = _extract_scores(raw_scores, len(dt_boxes))
        output = []
        for idx, item in enumerate(dt_boxes):
            if hasattr(item, "tolist"):
                polygon = item.tolist()
            else:
                polygon = item
            score = scores[idx] if idx < len(scores) else 0.5
            output.append((polygon, score))
        return output

    def unload(self) -> None:
        """Release resources."""
        if hasattr(self, "_detector"):
            del self._detector
        
        try:
            import paddle
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
                import gc
                gc.collect()
        except Exception:
            pass


def _read_image(image_path: str):
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    image = cv2.imread(image_path)
    if image is None:
        try:
            data = np.fromfile(image_path, dtype=np.uint8)
            if data.size:
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            image = None
    return image


def _extract_scores(raw_scores, expected_len: int) -> List[float]:
    if expected_len <= 0:
        return []
    if raw_scores is None:
        return []
    if hasattr(raw_scores, "tolist"):
        try:
            raw_scores = raw_scores.tolist()
        except Exception:
            raw_scores = None
    if not isinstance(raw_scores, (list, tuple)):
        return []
    if len(raw_scores) != expected_len:
        return []
    out = []
    for value in raw_scores:
        out.append(_coerce_score(value))
    return out


def _coerce_score(value) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.5
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
