# -*- coding: utf-8 -*-
"""PaddleOCR recognition wrapper."""
from __future__ import annotations
import os
from typing import Optional

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_mkldnn", "0")

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None


class PaddleOcrRecognizer:
    def __init__(self, use_gpu: bool) -> None:
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR is not installed.")
        try:
            import paddle
            paddle.set_flags({"FLAGS_use_mkldnn": 0, "FLAGS_enable_onednn": 0})
        except Exception:
            pass
        requested_lang = os.environ.get("MT_PADDLE_OCR_LANG", "auto").strip().lower()
        if requested_lang not in {"auto", "japan", "ch"}:
            requested_lang = "auto"

        rec_model_dir = None
        cls_model_dir = None
        lang = "japan"

        # 1. System Cache (Priority 1)
        use_system = False
        system_lang = None
        try:
            home = os.path.expanduser("~")
            system_rec_paths = {
                "japan": [
                    os.path.join(home, ".paddleocr", "whl", "rec", "japan", "japan_PP-OCRv4_rec_infer"),
                    os.path.join(home, ".paddleocr", "whl", "rec", "japan", "japan_PP-OCRv3_rec_infer"),
                ],
                "ch": [
                    os.path.join(home, ".paddleocr", "whl", "rec", "ch", "ch_PP-OCRv4_rec_infer"),
                    os.path.join(home, ".paddleocr", "whl", "rec", "ch", "ch_PP-OCRv3_rec_infer"),
                ],
            }
            if requested_lang in {"japan", "ch"}:
                candidates = system_rec_paths[requested_lang]
            else:
                candidates = system_rec_paths["japan"] + system_rec_paths["ch"]
            for path in candidates:
                if os.path.exists(path):
                    use_system = True
                    system_lang = "japan" if "japan" in path else "ch"
                    break
        except Exception:
            pass

        # 2. Local models (Priority 2 if system cache unavailable)
        app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(app_root, "models", "paddleocr")
        local_map = {
            "japan": [
                os.path.join(models_dir, "japan_PP-OCRv4_rec_infer"),
                os.path.join(models_dir, "japan_PP-OCRv3_rec_infer"),
            ],
            "ch": [
                os.path.join(models_dir, "ch_PP-OCRv4_rec_infer"),
                os.path.join(models_dir, "ch_PP-OCRv3_rec_infer"),
            ],
        }
        local_lang = None
        local_rec = None
        if not use_system:
            preferred_order = [requested_lang] if requested_lang in {"japan", "ch"} else ["japan", "ch"]
            for choice in preferred_order:
                for candidate in local_map[choice]:
                    if os.path.exists(candidate):
                        local_lang = choice
                        local_rec = candidate
                        break
                if local_rec:
                    break
            if local_rec:
                rec_model_dir = local_rec
            local_cls = os.path.join(models_dir, "ch_ppocr_mobile_v2.0_cls_infer")
            if os.path.exists(local_cls):
                cls_model_dir = local_cls

        if requested_lang in {"japan", "ch"}:
            lang = requested_lang
        elif use_system and system_lang:
            lang = system_lang
        elif local_lang:
            lang = local_lang

        self._engine = PaddleOCR(
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir,
            det=False,
            rec=True,
            cls=False,
            lang=lang,
            use_gpu=use_gpu,
            ir_optim=False,
            use_tensorrt=False,
            show_log=False,
        )

    def unload(self) -> None:
        """Unload model and free VRAM."""
        if hasattr(self, "_engine"):
            del self._engine
        
        try:
            import paddle
            import gc
            # Paddle doesn't have a clear "unload" but clearing cache helps
            paddle.device.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

    def recognize(self, image) -> str:
        try:
            import numpy as np
        except ImportError:
            np = None
        candidates = []
        for angle in (0, 90, 270):
            img = image
            if angle and hasattr(image, "rotate"):
                img = image.rotate(angle, expand=True)
            if np is not None and not isinstance(img, (str, bytes, list)):
                try:
                    img = np.array(img)
                except Exception:
                    pass
            text = _run_rec(self._engine, img)
            score = _text_score(text)
            candidates.append((score, text))
        candidates.sort(reverse=True)
        best = candidates[0][1] if candidates else ""
        return best.strip()

    def recognize_with_confidence(self, image) -> tuple[str, float]:
        """Recognize text and return confidence score."""
        try:
            import numpy as np
        except ImportError:
            np = None
            
        candidates = []
        for angle in (0, 90, 270):
            img = image
            if angle and hasattr(image, "rotate"):
                img = image.rotate(angle, expand=True)
            if np is not None and not isinstance(img, (str, bytes, list)):
                try:
                    img = np.array(img)
                except Exception:
                    pass
            
            # Run recognition
            # PaddleOCR returns [[('text', score)]] or [None]
            result = self._engine.ocr(img, det=False, rec=True, cls=False)
            if not result:
                continue
                
            first = result[0]
            if isinstance(first, list) and first:
                item = first[0] # ('text', score)
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    text = str(item[0]).strip()
                    score = float(item[1])
                    candidates.append((score, text))
        
        candidates.sort(reverse=True) # Best score first
        if candidates:
            return candidates[0][1], candidates[0][0]
        return "", 0.0


def _run_rec(engine, image) -> str:
    result = engine.ocr(image, det=False, rec=True, cls=False)
    if not result:
        return ""
    first = result[0]
    if isinstance(first, list) and first:
        item = first[0]
        if isinstance(item, (list, tuple)) and item:
            return str(item[0]).strip()
    return ""


def _text_score(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    jp = sum(1 for ch in text if _is_japanese(ch))
    return jp * 2 + total


def _is_japanese(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF  # Hiragana/Katakana
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified
    )
