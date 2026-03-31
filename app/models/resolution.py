# -*- coding: utf-8 -*-
"""Shared model resolution helpers for startup checks and runtime loaders."""
from __future__ import annotations

import os
from typing import Iterable, Optional

from app.config.defaults import MANGA_OCR_FILES


def _app_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def models_root() -> str:
    return os.path.join(_app_root(), "models")


def _hf_home() -> str:
    return os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )


def _hf_snapshot_dirs(user: str, repo: str) -> list[str]:
    model_dir = os.path.join(_hf_home(), "hub", f"models--{user}--{repo}", "snapshots")
    if not os.path.isdir(model_dir):
        return []
    dirs: list[str] = []
    for entry in os.listdir(model_dir):
        path = os.path.join(model_dir, entry)
        if os.path.isdir(path):
            dirs.append(path)
    return dirs


def _first_dir_with_files(dirs: Iterable[str], required_files: Iterable[str]) -> Optional[str]:
    required = tuple(required_files)
    for directory in dirs:
        if all(os.path.exists(os.path.join(directory, name)) for name in required):
            return directory
    return None


def resolve_manga_ocr_system_ref() -> Optional[str]:
    snapshot = _first_dir_with_files(
        _hf_snapshot_dirs("kha-white", "manga-ocr-base"),
        MANGA_OCR_FILES,
    )
    if snapshot:
        return snapshot
    return None


def resolve_manga_ocr_local_dir(base_dir: Optional[str] = None) -> Optional[str]:
    target = os.path.join(base_dir or models_root(), "manga-ocr")
    if _first_dir_with_files([target], MANGA_OCR_FILES):
        return target
    return None


def resolve_ner_system_snapshot() -> Optional[str]:
    required = ("config.json",)
    optional_weights = ("model.safetensors", "pytorch_model.bin")
    for snapshot in _hf_snapshot_dirs("jurabi", "bert-ner-japanese"):
        if not all(os.path.exists(os.path.join(snapshot, name)) for name in required):
            continue
        if any(os.path.exists(os.path.join(snapshot, name)) for name in optional_weights):
            return snapshot
    return None


def resolve_ner_local_dir(model_dir: Optional[str] = None) -> Optional[str]:
    base_dir = model_dir or os.path.join(models_root(), "ner")
    candidates = [base_dir]
    nested = os.path.join(base_dir, "models--jurabi--bert-ner-japanese", "snapshots")
    if os.path.isdir(nested):
        for entry in os.listdir(nested):
            path = os.path.join(nested, entry)
            if os.path.isdir(path):
                candidates.append(path)

    required = ("config.json",)
    optional_weights = ("model.safetensors", "pytorch_model.bin")
    for candidate in candidates:
        if not all(os.path.exists(os.path.join(candidate, name)) for name in required):
            continue
        if any(os.path.exists(os.path.join(candidate, name)) for name in optional_weights):
            return candidate
    return None


def _paddle_home() -> str:
    return os.path.join(os.path.expanduser("~"), ".paddleocr", "whl")


def resolve_paddle_system_det_dir() -> Optional[str]:
    candidates = [
        os.path.join(_paddle_home(), "det", "ch", "ch_PP-OCRv4_det_infer"),
        os.path.join(_paddle_home(), "det", "ch", "ch_PP-OCRv3_det_infer"),
        os.path.join(_paddle_home(), "det", "ml", "Multilingual_PP-OCRv3_det_infer"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def resolve_paddle_local_det_dir(base_dir: Optional[str] = None) -> Optional[str]:
    root = os.path.join(base_dir or models_root(), "paddleocr")
    candidates = [
        os.path.join(root, "ch_PP-OCRv4_det_infer"),
        os.path.join(root, "Multilingual_PP-OCRv3_det_infer"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def resolve_paddle_system_rec_dir(requested_lang: str = "auto") -> tuple[Optional[str], Optional[str]]:
    roots = {
        "japan": [
            os.path.join(_paddle_home(), "rec", "japan", "japan_PP-OCRv4_rec_infer"),
            os.path.join(_paddle_home(), "rec", "japan", "japan_PP-OCRv3_rec_infer"),
        ],
        "ch": [
            os.path.join(_paddle_home(), "rec", "ch", "ch_PP-OCRv4_rec_infer"),
            os.path.join(_paddle_home(), "rec", "ch", "ch_PP-OCRv3_rec_infer"),
        ],
    }
    order = [requested_lang] if requested_lang in roots else ["japan", "ch"]
    for lang in order:
        for candidate in roots[lang]:
            if os.path.isdir(candidate):
                return candidate, lang
    return None, None


def resolve_paddle_local_rec_dir(
    requested_lang: str = "auto",
    base_dir: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    root = os.path.join(base_dir or models_root(), "paddleocr")
    options = {
        "japan": [
            os.path.join(root, "japan_PP-OCRv4_rec_infer"),
            os.path.join(root, "japan_PP-OCRv3_rec_infer"),
        ],
        "ch": [
            os.path.join(root, "ch_PP-OCRv4_rec_infer"),
            os.path.join(root, "ch_PP-OCRv3_rec_infer"),
        ],
    }
    order = [requested_lang] if requested_lang in options else ["japan", "ch"]
    for lang in order:
        for candidate in options[lang]:
            if os.path.isdir(candidate):
                cls_dir = os.path.join(root, "ch_ppocr_mobile_v2.0_cls_infer")
                return candidate, lang, cls_dir if os.path.isdir(cls_dir) else None
    return None, None, None


def has_paddle_runtime_models(requested_lang: str = "auto", base_dir: Optional[str] = None) -> bool:
    if resolve_paddle_system_det_dir() and resolve_paddle_system_rec_dir(requested_lang)[0]:
        return True
    return bool(
        resolve_paddle_local_det_dir(base_dir) and resolve_paddle_local_rec_dir(requested_lang, base_dir)[0]
    )
