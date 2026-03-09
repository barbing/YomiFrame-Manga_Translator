# -*- coding: utf-8 -*-
"""ComicTextDetector wrapper."""
from __future__ import annotations
import os
import sys
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _third_party_root() -> str:
    return os.path.join(_repo_root(), "app", "third_party", "comic-text-detector")


def _default_model_dir() -> str:
    return os.path.join(_repo_root(), "models", "comic-text-detector")


def _select_model_path(model_dir: str, use_gpu: bool) -> str:
    # This model is user-downloaded and resides in the models/ folder.
    # We do NOT check system paths to avoid conflict.
    
    # 1. Portable Model Check
    portable_model_root = None
    local_model_path = os.path.join(os.getcwd(), "models", "comic-text-detector")
    if os.path.exists(os.path.join(local_model_path, "comictextdetector.pt")):
        portable_model_root = local_model_path

    # Determine effective root (Allow override, then portable, then default arg)
    effective_model_root = portable_model_root or model_dir

    override = os.environ.get("MT_COMICTEXT_MODEL_PATH", "").strip()
    if override:
        return override

    onnx_path = os.path.join(effective_model_root, "comictextdetector.pt.onnx")
    pt_path = os.path.join(effective_model_root, "comictextdetector.pt")

    if use_gpu and os.path.isfile(pt_path):
        logger.info(f"Selected GPU model: {pt_path}")
        return pt_path
    if os.path.isfile(onnx_path):
        logger.info(f"Selected ONNX model: {onnx_path}")
        return onnx_path
    if os.path.isfile(pt_path):
        return pt_path
    return onnx_path


def _bbox_to_polygon(xyxy: list) -> List[List[float]]:
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    if x1 < x0 or y1 < y0:
        x1 = x0 + max(1.0, x1)
        y1 = y0 + max(1.0, y1)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _normalize_line(line) -> List[List[float]]:
    if line is None:
        return []
    if len(line) == 8 and not hasattr(line[0], "__len__"):
        return [
            [float(line[0]), float(line[1])],
            [float(line[2]), float(line[3])],
            [float(line[4]), float(line[5])],
            [float(line[6]), float(line[7])],
        ]
    output = []
    for point in line:
        if point is None:
            continue
        if hasattr(point, "__len__") and len(point) >= 2:
            output.append([float(point[0]), float(point[1])])
    return output


class ComicTextDetector:
    def __init__(self, use_gpu: bool, model_dir: str | None = None) -> None:
        repo_root = _third_party_root()
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        _ensure_utils_package(repo_root)
        self.merge_mode = "none"

        try:
            import torch
        except Exception:
            torch = None

        model_root = model_dir or os.environ.get("MT_COMICTEXT_MODEL_DIR", "").strip() or _default_model_dir()
        self._model_path = _select_model_path(model_root, use_gpu)
        if not os.path.isfile(self._model_path):
            raise RuntimeError(
                "ComicTextDetector model not found. Download comictextdetector.pt.onnx (CPU) or "
                "comictextdetector.pt (GPU) from https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1 "
                f"and place it under {model_root}."
            )

        device = "cpu"
        if (
            use_gpu
            and torch is not None
            and torch.cuda.is_available()
            and self._model_path.endswith(".pt")
        ):
            device = "cuda"

        from inference import TextDetector
        from utils.textmask import REFINEMASK_INPAINT

        self._refine_mode = REFINEMASK_INPAINT
        input_size = int(os.environ.get("MT_COMICTEXT_INPUT_SIZE", "640"))
        self._detector = TextDetector(
            model_path=self._model_path,
            input_size=input_size,
            device=device,
            act="leaky",
            conf_thresh=0.5,
            nms_thresh=0.4,
        )
        logger.info(f"ComicTextDetector initialized. GPU={use_gpu}, Device={device}")

    def unload(self) -> None:
        """Unload model and free VRAM."""
        if hasattr(self, "_detector"):
            del self._detector
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        except Exception:
            pass

    def detect(self, image_path: str, input_size: int = 1024) -> List[Tuple[List[List[float]], float]]:
        image = _read_image(image_path)
        if image is None:
            return []
        return self.detect_image(image, input_size)

    def detect_image(self, image, input_size: int = 1024) -> List[Tuple[List[List[float]], float]]:
        if image is None:
            return []
        # Update input size if changed
        if input_size != self._detector.input_size:
             self._detector.input_size = (input_size, input_size)
             
        _, _, blk_list = self._detector(
            image,
            refine_mode=self._refine_mode,
            keep_undetected_mask=False,
        )
        output: List[Tuple[List[List[float]], float]] = []
        for blk in blk_list:
            # Use real probability from detector (default to 1.0 if missing)
            score = getattr(blk, "prob", 1.0)
            
            line_box = _lines_bounds(getattr(blk, "lines", []) or [])
            if line_box:
                output.append((_bbox_to_polygon(line_box), score))
                continue
            xyxy = getattr(blk, "xyxy", None)
            if xyxy:
                output.append((_bbox_to_polygon(xyxy), score))
        return output


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


def _ensure_utils_package(repo_root: str) -> None:
    utils_dir = os.path.join(repo_root, "utils")
    if not os.path.isdir(utils_dir):
        return
    existing = sys.modules.get("utils")
    if existing is None:
        import types
        pkg = types.ModuleType("utils")
        pkg.__path__ = [utils_dir]
        sys.modules["utils"] = pkg
        return
    current_path = getattr(existing, "__path__", None)
    if current_path is None or utils_dir not in list(current_path):
        import types
        pkg = types.ModuleType("utils")
        pkg.__path__ = [utils_dir]
        sys.modules["utils"] = pkg


def _lines_bounds(lines: list) -> list | None:
    if not lines:
        return None
    xs = []
    ys = []
    for line in lines:
        for point in line:
            if point is None or not hasattr(point, "__len__") or len(point) < 2:
                continue
            xs.append(point[0])
            ys.append(point[1])
    if not xs or not ys:
        return None
    x0, y0 = int(min(xs)), int(min(ys))
    x1, y1 = int(max(xs)), int(max(ys))
    return [x0, y0, x1, y1]
