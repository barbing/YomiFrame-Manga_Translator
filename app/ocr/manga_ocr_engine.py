# -*- coding: utf-8 -*-
"""MangaOCR wrapper."""
from __future__ import annotations
import os
import sys
import ctypes
from pathlib import Path
from app.models.resolution import resolve_manga_ocr_local_dir, resolve_manga_ocr_system_ref


def _add_dll_search_paths() -> None:
    if not hasattr(os, "add_dll_directory"):
        return
    candidates = [
        Path(sys.prefix) / "Library" / "bin",
        Path(sys.prefix) / "DLLs",
        Path(sys.prefix),
        Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
    ]
    for path in candidates:
        if path.exists():
            try:
                os.add_dll_directory(str(path))
            except OSError:
                pass
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.environ["PATH"] = f"{torch_lib};{os.environ.get('PATH','')}"


def _preload_torch_dlls() -> None:
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if not torch_lib.exists():
        return
    for name in ("shm.dll", "torch_cpu.dll", "torch_cuda.dll", "torch.dll"):
        path = torch_lib / name
        if path.exists():
            try:
                ctypes.WinDLL(str(path))
            except OSError:
                pass


def ensure_torch_runtime_ready():
    """Prepare DLL paths and import torch before OCR model init."""
    _add_dll_search_paths()
    _preload_torch_dlls()
    import torch
    return torch


def resolve_manga_ocr_model_ref() -> str | None:
    """Resolve the best MangaOCR model reference: system cache, local path, or None."""
    return resolve_manga_ocr_system_ref() or resolve_manga_ocr_local_dir()


def create_manga_ocr_instance(use_gpu: bool):
    """Create MangaOCR instance using shared model resolution logic."""
    try:
        from manga_ocr import MangaOcr
    except Exception as exc:
        raise RuntimeError(f"Failed to import manga-ocr: {exc}") from exc

    model_ref = resolve_manga_ocr_model_ref()
    if model_ref:
        return MangaOcr(pretrained_model_name_or_path=model_ref, force_cpu=not use_gpu)
    # Final fallback (library default behavior, may download)
    return MangaOcr(force_cpu=not use_gpu)


class MangaOcrEngine:
    def recognize_with_confidence(self, image) -> tuple[str, float]:
        """Recognize text and return confidence score."""
        try:
            import numpy as np
            from PIL import Image
            import cv2
            if isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] == 3:
                     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            # CRITICAL: MangaOCR expects RGB inputs.
            # If input is Grayscale (L) or RGBA, convert to RGB to match standard library behavior.
            if hasattr(image, "convert"):
                image = image.convert("RGB")
        except Exception:
            pass

        try:
            import torch
            import numpy as np
            
            # Access internal components
            if not hasattr(self._engine, "model") or not hasattr(self._engine, "processor"):
                return self._engine(image), 1.0

            model = self._engine.model
            processor = self._engine.processor
            tokenizer = getattr(self._engine, "tokenizer", None)
            if tokenizer is None:
                return self._engine(image), 1.0
            
            # Prepare input - Match manga-ocr behavior exactly
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(model.device)
            if model.device.type == "cuda":
                pixel_values = pixel_values.to(model.dtype)

            # Generate with scores
            # Forces greedy decoding (num_beams=1) to match standard behavior
            # AND include standard args (if any) from the engine wrapper
            gen_args = getattr(self._engine, "args", {})
            
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values, 
                    output_scores=True,
                    return_dict_in_generate=True,
                    **gen_args,
                )
            
            sequences = outputs.sequences
            scores = outputs.scores
            
            text = tokenizer.decode(sequences[0], skip_special_tokens=True)
            # CRITICAL: Clean up artifacts (SentencePiece spaces)
            text = text.replace(" ", "")
            
            # Calculate confidence (mean probability)
            gens = sequences[0]
            if hasattr(model.config, "decoder_start_token_id") and gens[0] == model.config.decoder_start_token_id:
                gens = gens[1:]
            
            probs = []
            for i, score_step in enumerate(scores):
                if i >= len(gens): break
                token_id = gens[i]
                if token_id == tokenizer.eos_token_id:
                    break
                    
                step_probs = torch.softmax(score_step, dim=-1)
                prob = step_probs[0, token_id].item()
                probs.append(prob)
            
            if not probs:
                confidence = 0.0
            else:
                confidence = float(np.mean(probs))
                
            return text, confidence
            
        except Exception as e:
            print(f"[MangaOCR] Confidence extraction failed: {e}")
            return self._engine(image), 1.0

    def __init__(self, use_gpu: bool) -> None:
        try:
            ensure_torch_runtime_ready()
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(f"Failed to load torch: {exc}") from exc
        self._engine = create_manga_ocr_instance(use_gpu)

    def recognize(self, image) -> str:
        try:
            import numpy as np
            from PIL import Image
            import cv2
            if isinstance(image, np.ndarray):
                # Convert BGR (OpenCV) to RGB (PIL)
                if image.ndim == 3 and image.shape[2] == 3:
                     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
        except Exception:
            pass  # Fallback to original input if conversion fails
            
        return self._engine(image)

    def close(self) -> None:
        """Release resources."""
        if hasattr(self, "_engine"):
            del self._engine
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        except Exception:
            pass
