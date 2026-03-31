# -*- coding: utf-8 -*-
"""Persistent MangaOCR worker process for faster OCR."""
from __future__ import annotations
import os
import tempfile
import threading
import uuid
from multiprocessing import get_context
from typing import Optional
from pathlib import Path
import sys
import ctypes

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None


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


def _worker_main(requests, responses, use_gpu: bool) -> None:
    _add_dll_search_paths()
    _preload_torch_dlls()
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TORCH_FORCE_CPU"] = "1"
    from PIL import Image as PILImage
    from app.ocr.manga_ocr_engine import create_manga_ocr_instance

    ocr = create_manga_ocr_instance(use_gpu)
    while True:
        item = requests.get()
        if item is None:
            break
        request_id, image_path = item
        try:
            img = PILImage.open(image_path)
            text = ocr(img)
            responses.put((request_id, text, None))
        except Exception as exc:  # pragma: no cover - runtime dependency
            responses.put((request_id, "", str(exc)))


class MangaOcrWorker:
    def __init__(self, use_gpu: bool = True) -> None:
        if Image is None:
            raise RuntimeError("Pillow is not installed.")
        ctx = get_context("spawn")
        self._requests = ctx.Queue()
        self._responses = ctx.Queue()
        self._proc = ctx.Process(
            target=_worker_main,
            args=(self._requests, self._responses, use_gpu),
            daemon=True,
        )
        self._proc.start()
        self._lock = threading.Lock()

    def recognize(self, image) -> str:
        if Image is None:
            raise RuntimeError("Pillow is not installed.")
        if not hasattr(image, "save"):
            try:
                import numpy as np
                import cv2
                if isinstance(image, np.ndarray):
                    if image.ndim == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
            except Exception:
                pass
        with self._lock:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp_path = tmp.name
            try:
                image.save(tmp_path)
                request_id = uuid.uuid4().hex
                self._requests.put((request_id, tmp_path))
                while True:
                    resp_id, text, error = self._responses.get()
                    if resp_id != request_id:
                        continue
                    if error:
                        raise RuntimeError(error)
                    return str(text).strip()
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def close(self) -> None:
        try:
            self._requests.put(None)
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.join(timeout=2)

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass
