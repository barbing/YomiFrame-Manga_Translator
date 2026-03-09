# -*- coding: utf-8 -*-
"""MangaOCR subprocess wrapper to avoid in-process DLL issues."""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile
from typing import Optional

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None


class MangaOcrSubprocess:
    def __init__(self, python_executable: str | None = None, use_gpu: bool = True) -> None:
        self._python = python_executable or sys.executable
        self._use_gpu = use_gpu

    def recognize(self, image) -> str:
        if Image is None:
            raise RuntimeError("Pillow is not installed.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp_path = tmp.name
        try:
            image.save(tmp_path)
            return _run_ocr(self._python, tmp_path, self._use_gpu)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _run_ocr(python_executable: str, image_path: str, use_gpu: bool) -> str:
    env = os.environ.copy()
    app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = f"{app_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    if not use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["TORCH_FORCE_CPU"] = "1"
    code = (
        "from app.ocr.manga_ocr_engine import create_manga_ocr_instance;"
        "from PIL import Image;"
        "import sys;"
        f"ocr=create_manga_ocr_instance({str(use_gpu)});"
        "img=Image.open(sys.argv[1]);"
        "print(ocr(img))"
    )
    result = subprocess.run(
        [python_executable, "-c", code, image_path],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "MangaOCR subprocess failed.")
    return result.stdout.strip()
