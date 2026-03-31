# -*- coding: utf-8 -*-
"""Pre-scan module for building glossary before translation.

This module performs a light OCR-first pass on all manga pages to extract
character names and build a complete glossary before translation begins.
The default path stays lightweight: OCR + MeCab + alias graph + glossary
translation with the active translator. Heavier NER remains opt-in.
"""
from __future__ import annotations

import os
import logging
import re
from typing import Callable, Optional, TYPE_CHECKING

from PySide6 import QtCore

if TYPE_CHECKING:
    from app.pipeline.controller import PipelineSettings

logger = logging.getLogger(__name__)


class PrescanWorker(QtCore.QThread):
    """
    Worker thread that performs pre-scanning of manga pages.
    
    Emits:
        progress_changed(int): Percentage complete (0-100)
        message(str): Status messages
        finished_with_glossary(dict): Completed style guide with discovered names
    """
    
    progress_changed = QtCore.Signal(int)
    message = QtCore.Signal(str)
    finished_with_glossary = QtCore.Signal(dict)
    
    def __init__(
        self,
        settings: "PipelineSettings",
        images: list[str],
        style_guide: dict,
        parent=None,
    ):
        super().__init__(parent)
        self._settings = settings
        self._images = images
        self._style_guide = style_guide
        self._stop_requested = False
    
    def request_stop(self):
        self._stop_requested = True
    
    def run(self):
        """Execute the pre-scan process."""
        try:
            result = prescan_for_glossary(
                import_dir=self._settings.import_dir,
                images=self._images,
                style_guide=self._style_guide,
                settings=self._settings,
                progress_callback=self._on_progress,
                message_callback=self._on_message,
                stop_check=lambda: self._stop_requested,
            )
            self.finished_with_glossary.emit(result)
        except Exception as e:
            self.message.emit(f"Pre-scan failed: {e}")
            logger.exception("Pre-scan error")
            self.finished_with_glossary.emit(self._style_guide)
    
    def _on_progress(self, percent: int):
        self.progress_changed.emit(percent)
    
    def _on_message(self, msg: str):
        self.message.emit(msg)


def prescan_for_glossary(
    import_dir: str,
    images: list[str],
    style_guide: dict,
    settings: "PipelineSettings",
    progress_callback: Optional[Callable[[int], None]] = None,
    message_callback: Optional[Callable[[str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    translator=None,
    detector=None,
    ocr_engine=None,
) -> dict:
    """
    Pre-scan all images and build a complete glossary.
    
    This runs OCR + lightweight entity extraction on all pages, then
    translates accepted name candidates using the active translator client.
    
    Args:
        import_dir: Path to manga images folder
        images: List of image filenames (already sorted)
        style_guide: Existing style guide to update
        settings: Pipeline settings
        progress_callback: Called with percentage (0-100)
        message_callback: Called with status messages
        stop_check: Returns True if scan should stop
        translator: Optional LLM translator client
        detector: Optional existing detector instance
        ocr_engine: Optional existing OCR engine instance
    
    Returns:
        Updated style_guide with all discovered names
    """
    if message_callback:
        message_callback(f"Pre-scanning {len(images)} pages for character names...")
    
    # Initialize components
    from app.detect.paddle_detector import PaddleTextDetector
    
    # Track if we created them locally (to unload them later)
    _local_detector = False
    _local_ocr = False

    # Use detector based on settings
    if detector is None:
        _local_detector = True
        if settings.detector_engine == "ComicTextDetector":
            try:
                from app.detect.comic_text_detector import ComicTextDetector
                detector = ComicTextDetector(settings.use_gpu)
            except Exception as e:
                if message_callback:
                    message_callback(f"ComicTextDetector unavailable: {e}. Using PaddleOCR.")
        
        if detector is None:
            detector = PaddleTextDetector(settings.use_gpu)
    
    # Initialize OCR
    if ocr_engine is None:
        _local_ocr = True
        if settings.ocr_engine == "MangaOCR":
            try:
                from app.ocr.manga_ocr_engine import MangaOcrEngine
                ocr_engine = MangaOcrEngine(settings.use_gpu)
            except Exception:
                pass
        
        if ocr_engine is None:
            from app.ocr.paddle_ocr_recognizer import PaddleOcrRecognizer
            ocr_engine = PaddleOcrRecognizer(settings.use_gpu)
    
    # Initialize NLP components
    from app.nlp.character_graph import CharacterGraph
    from app.nlp.mecab_extractor import MeCabExtractor
    from collections import defaultdict

    graph = CharacterGraph()
    context_buffer = defaultdict(list)  # temporary storage for context: name -> [sentences]
    all_extracted_names = []
    candidate_stats = {}
    preserved_canonicals = set()

    # Common false positives from NER/OCR (interjections, common nouns)
    NOISE_BLOCKLIST = {
        "しょうゆ", "醤油",
        "すまん", "スマン", "ごめん",
        "助平", "スケベ",
        "民草",
        "日本", "勇者",
        "イイ", "いい",
        "うわーこ", "うわあ",
        "クリス緩ー",
        "先生", "先輩", "後輩",
        "親父", "お袋",
        "ソレ", "それ", "アレ", "あれ",
        "ヤツ", "やつ", "アイツ", "あいつ",
        "自分", "じぶん",
        "アンタ", "あんた", "お前",
        "ボク", "ぼく", "俺", "おれ", "私", "わたし",
        "誰", "だれ",
        "皆", "みんな",
        "クッキー",
    }
    
    try:
        mecab = MeCabExtractor()
        ner = None
        use_ner = bool(
            getattr(settings, "prescan_use_ner", False)
            or os.getenv("MT_PRESCAN_USE_NER") == "1"
        )
        if use_ner:
            from app.nlp.ner_extractor import NERExtractor

            ner = NERExtractor()
            if message_callback:
                message_callback("Pre-Scan mode: MeCab + NER enhancement enabled.")
        elif message_callback:
            message_callback("Pre-Scan mode: lightweight MeCab discovery.")
    except Exception as e:
        if message_callback:
            message_callback(f"NLP initialization failed: {e}. Pre-scan skipped.")
        return style_guide
    
    # Process images
    total = len(images)
    
    for idx, image_name in enumerate(images):
        if stop_check and stop_check():
            if message_callback:
                message_callback("Pre-scan stopped.")
            break
        
        image_path = os.path.join(import_dir, image_name)
        
        try:
            # Detect text regions
            from app.pipeline.controller import _detect_regions, _get_image_size

            detections = _detect_regions(
                detector,
                image_path,
                _get_image_size(image_path),
                input_size=min(768, int(getattr(settings, "detector_input_size", 1024) or 1024)),
                use_gpu=settings.use_gpu,
                message_callback=message_callback,
            )
            page_image = _load_image_cv2(image_path)
            if page_image is None:
                continue
            
            # OCR each region
            for det in detections:
                # Crop and recognize
                crop = None
                
                # Normalize detection format
                # ComicTextDetector returns (polygon, score) or [polygon, score]
                if isinstance(det, (tuple, list)) and len(det) >= 1:
                    bbox = _polygon_to_rect(det[0])
                # Paddle returns dict with 'bbox' or 'box'
                elif isinstance(det, dict):
                    bbox = det.get("bbox")
                    if not bbox and "box" in det:
                        bbox = _polygon_to_rect(det["box"])
                else:
                    bbox = None

                if not bbox:
                    continue
                
                crop = _crop_image(page_image, bbox)
                if crop is None:
                    continue
                
                if hasattr(ocr_engine, "recognize_with_confidence"):
                    text, score = ocr_engine.recognize_with_confidence(crop)
                else:
                    text = ocr_engine.recognize(crop)
                    score = 1.0  # Dummy score
                
                if text and text.strip():
                    text = text.strip()
                    
                    # Extract names from this text block
                    names = []
                    ner_conf_map = {}
                    if ner and ner.is_available():
                        entities = ner.extract_names(text)
                        names = ner.to_extracted_names(entities)
                        for entity in entities:
                            entity_surface = _normalize_name(getattr(entity, "text", ""))
                            if not entity_surface:
                                continue
                            conf = _coerce_confidence(getattr(entity, "confidence", 0.0))
                            if conf > ner_conf_map.get(entity_surface, 0.0):
                                ner_conf_map[entity_surface] = conf

                        # Backfill readings using MeCab if available (NER usually lacks reading)
                        if mecab.is_available() and names:
                            for name in names:
                                if not name.reading or name.reading == name.surface:
                                    name.reading = mecab.get_reading(name.surface)
                    elif mecab.is_available():
                        names = mecab.extract_proper_nouns(text)

                    # Store names and context
                    if names:
                        for name in names:
                            # Normalize the surface to remove spurious spaces
                            normalized_surface = _normalize_name(name.surface)
                            
                            # Filter noise
                            if not normalized_surface or normalized_surface in NOISE_BLOCKLIST:
                                logger.debug(f"[Pre-scan] Filtered out: '{name.surface}' (noise or invalid)")
                                continue

                            # Update the name object's surface with the normalized version
                            # (ExtractedName is a dataclass, so we can modify in place)
                            name.surface = normalized_surface
                            
                            # Also normalize the reading if present
                            if name.reading:
                                name.reading = _normalize_name(name.reading)

                            all_extracted_names.append(name)
                            if len(context_buffer[normalized_surface]) < 8:
                                context_buffer[normalized_surface].append(text)
                            _update_candidate_stats(
                                candidate_stats,
                                normalized_surface,
                                idx,
                                _coerce_confidence(score),
                                ner_conf_map.get(normalized_surface, 0.0),
                                name.reading or "",
                            )
        
        except Exception as e:
            logger.warning(f"Pre-scan error on {image_name}: {e}")
            continue
        
        # Update progress
        if progress_callback:
            progress_callback(int((idx + 1) / total * 100))
    
    if message_callback:
        message_callback(f"Analyzed {total} pages. Building character graph...")
    
    # Build character graph from extracted names.
    for char_data in style_guide.get("characters", []) or []:
        if not isinstance(char_data, dict):
            continue
        try:
            canonical = _pick_character_canonical(char_data)
            if not canonical:
                continue
            preserved_canonicals.add(canonical)

            graph.add_character(
                canonical=canonical,
                reading=str(char_data.get("reading", "")).strip(),
                translation=str(char_data.get("translation", "")).strip() or None,
                gender=str(char_data.get("gender", "")).strip(),
                info=str(char_data.get("info", "")).strip(),
            )

            for alias in char_data.get("aliases", []) or []:
                alias_source = _alias_to_source(alias)
                if alias_source and alias_source != canonical:
                    graph.add_alias(alias_source, canonical)

            for ctx in char_data.get("context", []) or []:
                if isinstance(ctx, str) and ctx.strip():
                    graph.add_context_sentence(canonical, ctx.strip())
        except Exception as e:
            logger.warning(f"Pre-scan: Skipping malformed character entry: {e}")

    # Import existing glossary (for manual glossary-only entries).
    existing_glossary = style_guide.get("glossary", []) or []
    try:
        graph.merge_from_glossary(existing_glossary)
    except Exception as e:
        logger.warning(f"Pre-scan: Existing glossary import failed: {e}")

    for canonical, node in graph._nodes.items():
        if getattr(node, "translation", None):
            preserved_canonicals.add(canonical)

    try:
        graph.auto_link_aliases(all_extracted_names)
    except Exception as e:
        logger.warning(f"Pre-scan: Auto-link failed: {e}")

    for surface, contexts in context_buffer.items():
        canonical = graph.find_canonical(surface)
        if canonical:
            for ctx in contexts[:5]:
                graph.add_context_sentence(canonical, ctx)

    candidate_count = len(candidate_stats)
    graph, accepted_surfaces = _filter_graph_by_confidence(graph, candidate_stats, preserved_canonicals)
    accepted_count = len(accepted_surfaces)
    filtered_count = max(0, candidate_count - accepted_count)

    if message_callback:
        message_callback(
            f"Pre-Scan candidates: {candidate_count} | accepted: {accepted_count} | filtered: {filtered_count}."
        )

    untranslated_before = sum(1 for node in graph._nodes.values() if not getattr(node, "translation", None))
    translated_now = _translate_graph_nodes_with_active_client(
        graph,
        translator,
        settings,
        message_callback=message_callback,
    )
    unresolved_count = sum(1 for node in graph._nodes.values() if not getattr(node, "translation", None))

    if message_callback:
        message_callback(
            f"Pre-Scan character set: {len(graph._nodes)} canonical | translated now: {translated_now} | unresolved: {unresolved_count}."
        )
        if untranslated_before > 0 and translated_now == 0:
            message_callback(
                "Warning: Pre-Scan could not translate discovered names. Glossary enforcement will be limited."
            )

    logger.info(
        "Pre-scan summary: candidates=%s accepted=%s filtered=%s canonical=%s translated_now=%s unresolved=%s",
        candidate_count,
        accepted_count,
        filtered_count,
        len(graph._nodes),
        translated_now,
        unresolved_count,
    )

    # Update style guide with complete graph data
    # 1. Normalize "characters" schema for pipeline compatibility.
    graph_chars = graph.to_dict()["characters"]
    normalized_chars = []
    generated_glossary = []
    for char in graph_chars:
        canonical = str(char.get("canonical") or char.get("name") or "").strip()
        if not canonical:
            continue
        translated_name = str(char.get("translation") or "").strip()
        reading = str(char.get("reading", "")).strip()
        aliases_raw = sorted(set(char.get("aliases", []) or []))
        if not _looks_like_exportable_canonical(canonical, reading, translated_name):
            continue
        if not translated_name and canonical not in preserved_canonicals:
            continue
        if not translated_name and _is_all_kana_candidate(canonical):
            if not _has_non_kana_alias(aliases_raw, canonical):
                continue
        alias_objs = []
        for alias in aliases_raw:
            alias_source = str(alias).strip()
            if not alias_source or alias_source == canonical:
                continue
            if not _looks_like_clean_name_surface(alias_source):
                continue
            alias_obj = _build_alias_object(
                alias_source,
                canonical,
                reading,
                translated_name,
                mecab,
                translator=translator,
                settings=settings,
            )
            if alias_obj.get("target") and _looks_like_clean_name_surface(str(alias_obj.get("source", "")).strip()):
                alias_objs.append(alias_obj)
                generated_glossary.append(_alias_obj_to_glossary_entry(alias_obj))
        normalized_chars.append(
            {
                "canonical": canonical,
                "original": canonical,
                "name": translated_name or canonical,
                "translation": translated_name,
                "reading": reading,
                "gender": str(char.get("gender", "")).strip() or "unknown",
                "info": str(char.get("info", "")).strip(),
                "aliases": alias_objs,
                "context": char.get("context", []),
            }
        )
        if translated_name:
            canonical_entry = {
                "source": canonical,
                "target": translated_name,
                "priority": "hard",
                "auto": True,
                "type": "proper_noun",
                "pattern": "canonical",
            }
            if reading:
                canonical_entry["reading"] = reading
            generated_glossary.append(canonical_entry)
    style_guide["characters"] = normalized_chars
    
    # 2. Update "glossary" list with flattened entries for translator
    # (merge with existing glossary to keep manual entries)
    glossary_map = {}
    for item in style_guide.get("glossary", []) or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        if source:
            glossary_map[source] = item

    for entry in generated_glossary:
        source = str(entry.get("source", "")).strip()
        if source:
            glossary_map[source] = entry
        
    style_guide["glossary"] = list(glossary_map.values())
    try:
        from app.pipeline.controller import _sanitize_style_guide
        style_guide = _sanitize_style_guide(style_guide, settings.target_lang)
    except Exception as e:
        logger.warning(f"Pre-scan: final style-guide sanitization failed: {e}")
    
    # === CRITICAL: Free all GPU resources before main translation ===
    # Prescan creates its own detector/OCR/NER instances.
    # If not freed, main pipeline creates duplicates, causing 2x VRAM usage.
    
    logger.info("Pre-scan cleanup: Freeing detector, OCR, and NER from VRAM...")
    
    # 1. Unload NER model
    if ner:
        try:
            ner.unload()
        except Exception:
            pass
    
    # 2. Unload Detector
    if detector and _local_detector:
        try:
            if hasattr(detector, "unload"):
                detector.unload()
            elif hasattr(detector, "close"):
                detector.close()
            # Explicitly delete reference
            del detector
        except Exception as e:
            logger.warning(f"Pre-scan cleanup: Failed to unload detector: {e}")
            
    # 3. Unload OCR Engine
    if ocr_engine and _local_ocr:
        try:
            if hasattr(ocr_engine, "unload"):
                ocr_engine.unload()
            elif hasattr(ocr_engine, "close"):
                ocr_engine.close()
            # Explicitly delete reference
            del ocr_engine
        except Exception as e:
            logger.warning(f"Pre-scan cleanup: Failed to close OCR engine: {e}")
    
    # 4. Force garbage collection and clear CUDA cache
    try:
        import gc
        gc.collect()
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Pre-scan cleanup: CUDA cache cleared successfully")
    except Exception as e:
        logger.warning(f"Pre-scan cleanup: Failed to clear CUDA cache: {e}")
    
    translated_glossary_count = sum(1 for item in style_guide.get("glossary", []) if item.get("target"))
    if message_callback:
        message_callback(
            f"Pre-scan complete: {len(graph._nodes)} canonical names, {translated_glossary_count} translated glossary terms."
        )
        if len(graph._nodes) > 0 and translated_glossary_count == 0:
            message_callback(
                "Warning: Pre-Scan produced zero translated glossary terms; name consistency in page translation may be reduced."
            )
    
    return style_guide


def _coerce_confidence(value) -> float:
    try:
        conf = float(value)
    except Exception:
        conf = 0.0
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


def _update_candidate_stats(
    stats: dict,
    surface: str,
    page_index: int,
    ocr_conf: float,
    ner_conf: float,
    reading: str,
) -> None:
    item = stats.setdefault(
        surface,
        {
            "count": 0,
            "pages": set(),
            "ocr_sum": 0.0,
            "ner_sum": 0.0,
            "reading": "",
        },
    )
    item["count"] += 1
    item["pages"].add(page_index)
    item["ocr_sum"] += _coerce_confidence(ocr_conf)
    item["ner_sum"] += _coerce_confidence(ner_conf)
    if reading and (not item["reading"] or len(reading) > len(item["reading"])):
        item["reading"] = reading


def _pick_character_canonical(char_data: dict) -> str:
    for key in ("original", "canonical", "name"):
        value = str(char_data.get(key, "")).strip()
        if value:
            return value
    return ""


def _alias_to_source(alias) -> str:
    if isinstance(alias, dict):
        for key in ("source", "original", "canonical", "name"):
            value = str(alias.get(key, "")).strip()
            if value:
                return value
        return ""
    return str(alias).strip()


def _is_name_like(text: str) -> bool:
    if not text:
        return False
    if len(text) < 2 or len(text) > 20:
        return False
    has_cjk = any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in text)
    has_alpha = any(ch.isalpha() for ch in text)
    return has_cjk or has_alpha


def _is_loose_kana_candidate(text: str) -> bool:
    if not text:
        return False
    if _has_honorific_suffix(text):
        return False
    for ch in text:
        code = ord(ch)
        if not (0x3040 <= code <= 0x30FF or ch == "ー"):
            return False
    return True


def _is_hiragana_only(text: str) -> bool:
    text = str(text or "")
    return bool(text) and all((0x3040 <= ord(ch) <= 0x309F) or ch == "ー" for ch in text)


def _is_katakana_only(text: str) -> bool:
    text = str(text or "")
    return bool(text) and all((0x30A0 <= ord(ch) <= 0x30FF) or ch == "ー" for ch in text)


def _is_all_kana_candidate(text: str) -> bool:
    text = str(text or "")
    return bool(text) and all((0x3040 <= ord(ch) <= 0x30FF) or ch == "ー" for ch in text)


def _has_honorific_suffix(text: str) -> bool:
    text = str(text or "")
    if not text:
        return False
    relaxed_variants = {
        "さぁん": "さん",
        "さあん": "さん",
        "さーん": "さん",
        "ちゃぁん": "ちゃん",
        "ちゃあん": "ちゃん",
        "ちゃーん": "ちゃん",
        "くぅん": "くん",
        "くうん": "くん",
        "くーん": "くん",
    }
    relaxed = text
    for variant, normalized in relaxed_variants.items():
        if relaxed.endswith(variant):
            relaxed = f"{relaxed[:-len(variant)]}{normalized}"
            break

    false_honorific_words = {
        "みなさん",
        "皆さん",
        "たくさん",
        "沢山",
        "おじさん",
        "オジサン",
        "おばさん",
        "オバサン",
    }
    if relaxed in false_honorific_words:
        return False
    if relaxed.endswith(("みなさん", "皆さん", "たくさん", "沢山")):
        return False

    for suffix in ("さん", "くん", "ちゃん", "さま", "様", "先生", "先輩", "殿"):
        if not relaxed.endswith(suffix):
            continue
        base = relaxed[: -len(suffix)] if len(suffix) < len(relaxed) else ""
        if not base:
            continue
        if any(ch in base for ch in ("の", "が", "を", "に", "へ", "と", "も", "や", "か", "は", "で")):
            continue
        return True
    return False


def _is_japanese_word_char(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF
        or 0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
    )


def _context_boundary_ratio(surface: str, contexts: list[str]) -> float:
    surface = str(surface or "").strip()
    if not surface:
        return 0.0

    hits = 0
    bounded = 0
    for sentence in contexts or []:
        text = str(sentence or "")
        if not text:
            continue
        start = 0
        while True:
            idx = text.find(surface, start)
            if idx < 0:
                break
            hits += 1
            left = text[idx - 1] if idx > 0 else ""
            right_index = idx + len(surface)
            right = text[right_index] if right_index < len(text) else ""
            if not _is_japanese_word_char(left) and not _is_japanese_word_char(right):
                bounded += 1
            start = idx + len(surface)
    if hits <= 0:
        return 0.0
    return bounded / float(hits)


def _has_non_kana_alias(aliases: list[str] | set[str], canonical: str = "") -> bool:
    canonical = str(canonical or "").strip()
    for alias in aliases or []:
        alias = str(alias or "").strip()
        if alias and alias != canonical and not _is_loose_kana_candidate(alias):
            return True
    return False


def _is_supported_name_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF
        or 0x4E00 <= code <= 0x9FFF
        or ch in {"ー", "・", "々", "ヶ", "ケ", "ヴ"}
    )


def _is_kana(ch: str) -> bool:
    code = ord(ch)
    return 0x3040 <= code <= 0x30FF


def _is_cjk_term(text: str) -> bool:
    for ch in str(text or ""):
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF or 0x4E00 <= code <= 0x9FFF:
            return True
    return False


def _looks_like_clean_name_surface(text: str) -> bool:
    text = str(text or "").strip()
    if not text or len(text) > 12:
        return False
    if not all(_is_supported_name_char(ch) for ch in text):
        return False
    if all(0x4E00 <= ord(ch) <= 0x9FFF for ch in text) and len(text) > 6:
        return False
    for honorific in ("さん", "くん", "ちゃん", "様", "先生", "先輩", "殿", "君", "氏"):
        pos = text.find(honorific)
        if pos >= 0 and pos + len(honorific) < len(text):
            return False
    return _is_name_like(text)


def _looks_like_exportable_canonical(canonical: str, reading: str, translation: str) -> bool:
    canonical = str(canonical or "").strip()
    reading = str(reading or "").strip()
    translation = str(translation or "").strip()
    if not canonical or not _looks_like_clean_name_surface(canonical):
        return False
    if _is_cjk_term(canonical) and (not reading or not all(_is_kana(ch) for ch in reading)):
        return False
    if translation and len(translation) > 20:
        return False
    return True


def _should_auto_translate_canonical(node) -> bool:
    canonical = str(getattr(node, "canonical", "") or "").strip()
    if not canonical or getattr(node, "translation", None):
        return False
    if not _is_all_kana_candidate(canonical):
        return True
    return _has_non_kana_alias(getattr(node, "aliases", set()) or set(), canonical)


def _score_character_node(node, candidate_stats: dict) -> tuple[float, int, int, float, float]:
    total_count = 0
    page_set = set()
    ocr_sum = 0.0
    ner_sum = 0.0

    for alias in node.aliases:
        stats = candidate_stats.get(alias)
        if not stats:
            continue
        count = int(stats.get("count", 0))
        total_count += count
        page_set.update(stats.get("pages", set()))
        ocr_sum += float(stats.get("ocr_sum", 0.0))
        ner_sum += float(stats.get("ner_sum", 0.0))

    if total_count <= 0:
        return 0.0, 0, 0, 0.0, 0.0

    page_count = len(page_set)
    avg_ocr = ocr_sum / total_count
    avg_ner = ner_sum / total_count

    freq_component = min(total_count / 3.0, 1.0) * 0.45
    page_component = min(page_count / 2.0, 1.0) * 0.20
    ocr_component = _coerce_confidence(avg_ocr) * 0.25
    ner_component = _coerce_confidence(avg_ner) * 0.10
    score = freq_component + page_component + ocr_component + ner_component
    return score, total_count, page_count, avg_ocr, avg_ner


def _should_keep_character(node, score: float, total_count: int, page_count: int, avg_ocr: float, avg_ner: float) -> bool:
    canonical = str(getattr(node, "canonical", "") or "").strip()
    if not canonical:
        return False

    if _is_loose_kana_candidate(canonical):
        boundary_ratio = _context_boundary_ratio(canonical, getattr(node, "context_sentences", []) or [])
        has_honorific = _has_honorific_suffix(canonical)

        if len(canonical) <= 2 and not has_honorific:
            return False
        if boundary_ratio < 0.35 and total_count < 3 and not has_honorific:
            return False
        if _is_hiragana_only(canonical) and not has_honorific:
            if total_count < 3 or page_count < 2 or avg_ocr < 0.50:
                return False
        if _is_katakana_only(canonical) and len(canonical) <= 3 and not has_honorific:
            if total_count < 3 and page_count < 2:
                return False
        if total_count < 2 and page_count < 2 and avg_ner < 0.82:
            return False

    if score >= 0.58:
        return True
    if total_count >= 3 and avg_ocr >= 0.45:
        return True
    if total_count >= 2 and page_count >= 2 and avg_ocr >= 0.35:
        return True
    if avg_ner >= 0.82 and avg_ocr >= 0.40:
        return True
    if _is_name_like(canonical) and score >= 0.45 and avg_ocr >= 0.65:
        return True
    return False


def _filter_graph_by_confidence(graph, candidate_stats: dict, preserved_canonicals: set):
    from app.nlp.character_graph import CharacterGraph

    filtered_graph = CharacterGraph()
    kept_canonicals = set()
    accepted_surfaces = set()

    for canonical, node in graph._nodes.items():
        keep = False
        if canonical in preserved_canonicals:
            keep = True
        else:
            score, total_count, page_count, avg_ocr, avg_ner = _score_character_node(node, candidate_stats)
            keep = _should_keep_character(node, score, total_count, page_count, avg_ocr, avg_ner)

        if not keep:
            continue

        filtered_graph.add_character(
            canonical=canonical,
            reading=node.canonical_reading,
            translation=node.translation,
            gender=node.gender,
            info=node.info,
        )
        kept_canonicals.add(canonical)
        for alias in node.aliases:
            if alias in candidate_stats:
                accepted_surfaces.add(alias)

    for canonical in kept_canonicals:
        node = graph._nodes.get(canonical)
        if not node:
            continue
        for alias in node.aliases:
            if alias == canonical:
                continue
            alias_reading = ""
            alias_stats = candidate_stats.get(alias)
            if alias_stats:
                alias_reading = str(alias_stats.get("reading", "")).strip()
            filtered_graph.add_alias(alias, canonical, alias_reading=alias_reading or None)
        for sentence in node.context_sentences[:5]:
            filtered_graph.add_context_sentence(canonical, sentence)

    return filtered_graph, accepted_surfaces


def _resolve_prescan_model_name(settings: "PipelineSettings", translator) -> str:
    model_name = str(getattr(translator, "model_name", "")).strip()
    if model_name:
        return model_name

    if settings.translator_backend == "GGUF":
        return settings.gguf_model_path

    model_name = str(settings.ollama_model or "").strip()
    if model_name and model_name != "auto-detect":
        return model_name

    try:
        from app.models.ollama import list_models
        models = list_models()
        if models:
            return models[0]
    except Exception:
        pass
    return "sakura"


def _translate_graph_nodes_with_active_client(
    graph,
    translator,
    settings: "PipelineSettings",
    message_callback: Optional[Callable[[str], None]] = None,
) -> int:
    from app.pipeline.controller import _sanitize_glossary_target

    if translator is None:
        return 0

    terms_to_translate = [
        node.canonical
        for node in graph._nodes.values()
        if _should_auto_translate_canonical(node)
    ]
    if not terms_to_translate:
        return 0

    translated_count = 0
    used_batch = False
    if hasattr(translator, "translate_glossary"):
        try:
            translations_map = translator.translate_glossary(
                terms_to_translate,
                settings.source_lang,
                settings.target_lang,
            )
            if isinstance(translations_map, dict):
                used_batch = True
                for term in terms_to_translate:
                    raw = str(translations_map.get(term, "")).strip()
                    cleaned = _clean_name_translation(raw, term)
                    if not cleaned and _is_all_han(term):
                        cleaned = term
                    cleaned = _sanitize_glossary_target(cleaned, term, settings.target_lang)
                    if not cleaned:
                        continue
                    if term in graph._nodes:
                        graph._nodes[term].translation = cleaned
                        translated_count += 1
        except Exception as e:
            logger.warning(f"Pre-scan batch glossary translation failed: {e}")

    remaining = [
        node.canonical
        for node in graph._nodes.values()
        if _should_auto_translate_canonical(node)
    ]
    if not remaining or not hasattr(translator, "generate"):
        return translated_count

    model_name = _resolve_prescan_model_name(settings, translator)
    if message_callback and not used_batch:
        message_callback("Pre-Scan: Falling back to per-name translation with active translator...")

    for term in remaining:
        node = graph._nodes.get(term)
        context = getattr(node, "context_sentences", []) if node else []
        prompt = _build_name_prompt_with_context(term, context, settings.target_lang)
        try:
            result = translator.generate(
                model_name,
                prompt,
                timeout=60,
                options={"num_predict": 60, "temperature": 0.1},
            )
            cleaned = _clean_name_translation(str(result).strip(), term)
            if not cleaned and _is_all_han(term):
                cleaned = term
            cleaned = _sanitize_glossary_target(cleaned, term, settings.target_lang)
            if not cleaned:
                continue
            if term in graph._nodes:
                graph._nodes[term].translation = cleaned
                translated_count += 1
        except Exception as e:
            logger.debug(f"Pre-scan translation failed for '{term}': {e}")
    return translated_count


def _build_alias_object(
    alias_source: str,
    canonical_source: str,
    canonical_reading: str,
    translated_name: str,
    mecab,
    translator=None,
    settings: "PipelineSettings" | None = None,
) -> dict:
    alias_reading = ""
    pattern = ""
    hint = ""
    if mecab and mecab.is_available():
        try:
            alias_reading = _normalize_name(mecab.get_reading(alias_source))
            detected_pattern, _, detected_hint = mecab.detect_pattern(alias_source, alias_reading or alias_source)
            pattern = str(detected_pattern or "").strip()
            hint = str(detected_hint or "").strip()
        except Exception:
            pass

    alias_target = _resolve_alias_translation(
        alias_source=alias_source,
        alias_reading=alias_reading,
        canonical_source=canonical_source,
        canonical_reading=canonical_reading,
        canonical_translation=translated_name,
        mecab=mecab,
        translator=translator,
        settings=settings,
        pattern=pattern,
        hint=hint,
    )

    return {
        "source": alias_source,
        "target": alias_target,
        "reading": alias_reading,
        "pattern": pattern,
        "hint": hint,
    }


def _resolve_alias_translation(
    alias_source: str,
    alias_reading: str,
    canonical_source: str,
    canonical_reading: str,
    canonical_translation: str,
    mecab,
    translator=None,
    settings: "PipelineSettings" | None = None,
    pattern: str = "",
    hint: str = "",
) -> str:
    from app.pipeline.controller import _sanitize_glossary_target

    canonical_translation = str(canonical_translation or "").strip()
    if not canonical_translation:
        return ""

    alias_source = str(alias_source or "").strip()
    canonical_source = str(canonical_source or "").strip()
    canonical_reading = str(canonical_reading or "").strip()
    alias_reading = str(alias_reading or "").strip()
    if not alias_source:
        return ""
    if alias_source == canonical_source or alias_reading == canonical_reading:
        return canonical_translation

    base_reading = alias_reading
    if mecab and mecab.is_available():
        try:
            _, detected_base, detected_hint = mecab.detect_pattern(alias_source, alias_reading or alias_source)
            if detected_base:
                base_reading = detected_base
            if detected_hint and not hint:
                hint = str(detected_hint).strip()
        except Exception:
            pass

    segments = _split_canonical_name_segments(
        canonical_source,
        canonical_translation,
        canonical_reading,
        mecab,
    )
    matched = _match_alias_segment(base_reading, alias_reading, alias_source, segments)
    if matched:
        segment_target = str(matched.get("target", "")).strip() or canonical_translation
        segment_reading = str(matched.get("reading", "")).strip()
        if pattern in {"plain", "", "canonical"} and base_reading == segment_reading:
            return _sanitize_glossary_target(segment_target, alias_source, settings.target_lang if settings else "Simplified Chinese")

        heuristic_pattern = pattern
        if pattern in {"plain", "", "canonical"} and base_reading and segment_reading and base_reading != segment_reading:
            heuristic_pattern = "chan"
        heuristic = _heuristic_alias_target(segment_target, heuristic_pattern)
        if heuristic:
            return _sanitize_glossary_target(heuristic, alias_source, settings.target_lang if settings else "Simplified Chinese")

        translated = _translate_alias_with_active_client(
            translator,
            settings,
            alias_source,
            hint,
            segment_target,
        )
        if translated:
            return _sanitize_glossary_target(translated, alias_source, settings.target_lang if settings else "Simplified Chinese")
        return _sanitize_glossary_target(segment_target, alias_source, settings.target_lang if settings else "Simplified Chinese")
    return ""


def _split_canonical_name_segments(
    canonical_source: str,
    canonical_translation: str,
    canonical_reading: str,
    mecab,
) -> list[dict]:
    segments: list[dict] = []
    if mecab and getattr(mecab, "tagger", None):
        try:
            raw_tokens = []
            for word in mecab.tagger(canonical_source):
                surface = str(getattr(word, "surface", "") or "")
                if not surface.strip():
                    continue
                try:
                    kana = getattr(word.feature, "kana", None) or surface
                except Exception:
                    kana = surface
                raw_tokens.append({"source": surface, "reading": mecab._to_hiragana(kana)})
            if raw_tokens and "".join(token["source"] for token in raw_tokens) == canonical_source:
                segments = raw_tokens
        except Exception:
            segments = []

    if not segments:
        segments = [{"source": canonical_source, "reading": canonical_reading or canonical_source}]

    if _is_all_han(canonical_translation) and _is_all_han(canonical_source):
        if sum(len(seg["source"]) for seg in segments) == len(canonical_translation):
            offset = 0
            for seg in segments:
                seg_len = len(seg["source"])
                seg["target"] = canonical_translation[offset : offset + seg_len]
                offset += seg_len
        else:
            for seg in segments:
                seg["target"] = seg["source"]
    else:
        for seg in segments[:-1]:
            seg["target"] = seg["source"] if _is_all_han(seg["source"]) else ""
        segments[-1]["target"] = canonical_translation

    return [seg for seg in segments if str(seg.get("source", "")).strip()]


def _match_alias_segment(
    base_reading: str,
    alias_reading: str,
    alias_source: str,
    segments: list[dict],
) -> dict | None:
    if not segments:
        return None
    for seg in segments:
        if base_reading and base_reading == seg.get("reading"):
            return seg
        if alias_source and alias_source == seg.get("source"):
            return seg
    for seg in segments:
        seg_reading = str(seg.get("reading", "")).strip()
        if not seg_reading or len(base_reading) < 2:
            continue
        if seg_reading.startswith(base_reading) or base_reading.startswith(seg_reading):
            return seg
    for seg in segments:
        seg_reading = str(seg.get("reading", "")).strip()
        if not seg_reading or len(alias_reading) < 2:
            continue
        if seg_reading.startswith(alias_reading) or alias_reading.startswith(seg_reading):
            return seg
    return None


def _translate_alias_with_active_client(
    translator,
    settings: "PipelineSettings" | None,
    alias_source: str,
    hint: str,
    base_translation: str,
) -> str:
    if translator is None or settings is None or not hasattr(translator, "generate"):
        return ""
    # Default path stays heuristic-only for speed and stability.
    # LLM alias translation is reserved for the experimental discovery workflow.
    if not getattr(settings, "use_ollama_discovery", False):
        return ""
    model_name = _resolve_prescan_model_name(settings, translator)
    target_lang = settings.target_lang
    if target_lang not in {"Simplified Chinese", "Traditional Chinese"}:
        return ""

    if hint:
        prompt = (
            f"'{alias_source}'是对人物'{base_translation}'的{hint}。\n"
            f"把这个称呼翻译成{target_lang}。\n"
            "只输出译名，不要解释。"
        )
    else:
        prompt = (
            f"'{alias_source}'是人物'{base_translation}'的简称或昵称。\n"
            f"把这个称呼翻译成{target_lang}。\n"
            "只输出译名，不要解释。"
        )
    try:
        result = translator.generate(
            model_name,
            prompt,
            timeout=30,
            options={"num_predict": 30, "temperature": 0.1},
        )
        return _clean_name_translation(str(result or "").strip(), alias_source)
    except Exception:
        return ""


def _heuristic_alias_target(base_target: str, pattern: str) -> str:
    base_target = str(base_target or "").strip()
    if not base_target:
        return ""
    if pattern in {"chan", "tan", "cchi", "rin", "pyon"}:
        if len(base_target) <= 2:
            return f"小{base_target}"
        return base_target
    if pattern == "reduplication":
        tail = base_target[-1]
        if len(base_target) <= 2:
            return f"小{base_target}{tail}"
        return f"{base_target}{tail}"
    return base_target


def _alias_obj_to_glossary_entry(alias_obj: dict) -> dict:
    entry = {
        "source": alias_obj.get("source", ""),
        "target": alias_obj.get("target", ""),
        "priority": "hard",
        "auto": True,
        "type": "proper_noun",
    }
    reading = str(alias_obj.get("reading", "")).strip()
    pattern = str(alias_obj.get("pattern", "")).strip()
    hint = str(alias_obj.get("hint", "")).strip()
    if reading:
        entry["reading"] = reading
    if pattern:
        entry["pattern"] = pattern
    if hint:
        entry["hint"] = hint
    return entry


def _load_image_cv2(image_path: str):
    """Load image once (Unicode-safe) for repeated region crops."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    try:
        with open(image_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _crop_image(image, bbox: list) -> any:
    """Crop image region from a loaded cv2 image."""
    if image is None:
        return None
    
    x, y, w, h = bbox[:4]
    x, y = max(0, int(x)), max(0, int(y))
    w, h = max(1, int(w)), max(1, int(h))
    
    crop = image[y:y+h, x:x+w]
    if crop.size == 0:
        return None
    
    return crop


def _batch_translate_nodes(
    nodes: list,
    settings: "PipelineSettings",
    message_callback: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Translate a batch of character nodes using context.
    
    Returns:
        Number of successful translations
    """
    if not nodes:
        return 0
    
    # Initialize translation client
    try:
        if settings.translator_backend == "GGUF":
            from app.translate.gguf_client import GGUFClient
            n_gpu = settings.gguf_n_gpu_layers
            client = GGUFClient(
                model_path=settings.gguf_model_path,
                prompt_style=settings.gguf_prompt_style,
                n_ctx=settings.gguf_n_ctx,
                n_gpu_layers=n_gpu,
                n_threads=settings.gguf_n_threads,
                n_batch=settings.gguf_n_batch,
            )
            model_name = settings.gguf_model_path
        else:
            from app.translate.ollama_client import OllamaClient
            client = OllamaClient()
            model_name = settings.ollama_model
    except Exception as e:
        if message_callback:
            message_callback(f"Failed to initialize translator: {e}")
        return 0
    
    # Resolve model name
    if model_name == "auto-detect":
        from app.models.ollama import list_models
        models = list_models()
        model_name = models[0] if models else "sakura"
    
    count = 0
    # Translate each node
    for node in nodes[:50]:  # Limit to avoid overload
        try:
            prompt = _build_name_prompt_with_context(
                node.canonical, 
                node.context_sentences, 
                settings.target_lang
            )
            
            result = client.generate(
                model_name,
                prompt,
                timeout=30,
                options={"num_predict": 50, "temperature": 0.1},
            )
            
            if result and result.strip():
                clean = _clean_name_translation(result.strip(), node.canonical)
                if clean:
                    node.translation = clean
                    count += 1
        
        except Exception as e:
            logger.debug(f"Failed to translate name '{node.canonical}': {e}")
            continue
    
    # Cleanup
    try:
        if hasattr(client, "close"):
            client.close()
    except Exception:
        pass
    
    return count


def _build_name_prompt_with_context(name: str, context: list[str], target_lang: str) -> str:
    """Build a translation prompt for a character name with identification context."""
    examples = ""
    if context:
        # Take up to 3 short examples to provide context
        valid_ctx = [s for s in context if len(s) < 100][:3]
        if valid_ctx:
            examples = "\n上下文(Context):\n" + "\n".join([f"- {s}" for s in valid_ctx])
            
    if target_lang in ["Simplified Chinese", "Traditional Chinese"]:
        return f"""将以下日语人名翻译成中文，只输出中文译名，不要解释。
请根据上下文判断性别和语境（例如可爱、正式、古风）。
{examples}
人名：{name}
译名："""
    else:
        return f"""Translate the following Japanese name to {target_lang}.
Output only the translated name, no explanation.
Use context to determine gender and tone.
{examples}
Name: {name}
Translation:"""


def _clean_name_translation(translation: str, source: str) -> str:
    """Clean up a name translation result.
    
    Applies multiple filters to reject garbage LLM output:
    1. Strip leading/trailing whitespace and quotes
    2. Remove leading context markers (- or *)
    3. Take first line only
    4. Absolute max length check (names shouldn't be sentences)
    5. Reject prompt fragment echoes
    6. Reject if looks like a full sentence (contains common sentence patterns)
    """
    if not translation:
        return ""
        
    cleaned = translation.strip()
    
    # Take first line only
    if "\n" in cleaned:
        cleaned = cleaned.split("\n")[0].strip()
    
    # Remove leading context markers (LLM sometimes echoes "- context" lines)
    while cleaned.startswith("-") or cleaned.startswith("*") or cleaned.startswith("•"):
        cleaned = cleaned[1:].strip()
    
    # Remove quotes
    cleaned = cleaned.strip("\"'""''「」『』")

    # Prefer the shortest punctuation-delimited fragment that still looks like the name.
    if any(sep in cleaned for sep in "，,。！？：；:; "):
        parts = [part.strip() for part in re.split(r"[，,。！？：；:;\s]+", cleaned) if part.strip()]
        if parts:
            for part in reversed(parts):
                if part == source or part.endswith(source) or source.endswith(part):
                    cleaned = part
                    break

    # Remove trailing punctuation
    cleaned = cleaned.rstrip("。.，,、：:")
    
    # ABSOLUTE MAX LENGTH CHECK
    # A translated name should NEVER be longer than ~15 characters
    # (Even long names like "Alexander Hamilton" = 18 chars)
    if len(cleaned) > 20:
        return ""
    
    # Relative length check (backup)
    if len(cleaned) > max(len(source) * 4, 12):
        return ""
    
    # Reject prompt fragment echoes
    bad_markers = ["翻译", "译名", "输出", "Translation", "Name", "人名", "上下文", "Context"]
    if any(m in cleaned for m in bad_markers):
        return ""
    
    # Reject sentence-like patterns (names don't have these)
    sentence_markers = ["是", "了", "的", "吗", "呢", "啊", "吧", "啦", "呀", "哦", "嘛", 
                        "还", "很", "真", "在", "有", "没", "不", "会", "能", "要", "让"]
    # If more than 1 sentence marker, likely a sentence not a name
    marker_count = sum(1 for m in sentence_markers if m in cleaned)
    if marker_count > 1:
        return ""
    
    # Reject if it contains spaces (names shouldn't have spaces in CJK)
    # Exception: English names can have spaces
    if " " in cleaned and not any(ord(c) < 128 for c in cleaned):
        return ""

    # For kanji-only names in JP->CN, preserve the original if the model wrapped it in context.
    if _is_all_han(source) and (cleaned == source or source in cleaned):
        return source
    
    return cleaned


def _is_all_han(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return False
    return all(0x4E00 <= ord(ch) <= 0x9FFF for ch in text)


def _polygon_to_rect(poly: list) -> list:
    """Convert a polygon (list of [x, y] points) to [x, y, w, h] rect."""
    if not poly:
        return [0, 0, 0, 0]
    
    xs = [try_float(p[0]) for p in poly]
    ys = [try_float(p[1]) for p in poly]
    
    x = min(xs)
    y = min(ys)
    w = max(xs) - x
    h = max(ys) - y
    return [x, y, w, h]


def try_float(v):
    try:
        return float(v)
    except:
        return 0.0


def _normalize_name(name: str) -> str:
    """Normalize a name extracted from OCR.
    
    Removes spurious spaces that OCR engines sometimes insert between
    CJK characters, and cleans up common artifacts like nakaguro with spaces.
    """
    import re
    
    if not name:
        return ""
    
    cleaned = name.strip()
    
    # Remove spaces around nakaguro (middle dot) - common OCR artifact
    # e.g., "クリスティアーネ ・ フリードリヒ" -> "クリスティアーネ・フリードリヒ"
    cleaned = re.sub(r'\s*・\s*', '・', cleaned)
    cleaned = re.sub(r'\s*·\s*', '·', cleaned)  # Unicode middle dot variant
    
    # Check if name is primarily CJK (Japanese/Chinese characters)
    cjk_count = sum(1 for c in cleaned if ord(c) > 0x3000)
    ascii_count = sum(1 for c in cleaned if ord(c) < 128 and c.isalpha())
    
    if cjk_count > ascii_count:
        # Remove all spaces for CJK-dominant names
        cleaned = cleaned.replace(" ", "").replace("\u3000", "")
    
    # Remove common OCR artifacts from edges
    cleaned = cleaned.strip("・-─―")
    
    # -------------------------------------------------------------------------
    # CRITICAL QUALITY FILTER: Reject "Bridge Nodes" and Garbage
    # -------------------------------------------------------------------------
    # 1. Reject names with non-name punctuation (Quotes, Brackets, Punctuation)
    #    Real names do not contain: " ' ( ) ! ? … 。 、
    #    "Mayu" is fine. "Mayu!" is garbage. "Mayu”Chris" is a Bridge Node.
    INVALID_CHARS = {'"', "'", '”', '“', '’', '‘', '「', '」', '（', '）', '(', ')', '!', '！', '?', '？', '…', '。', '、'}
    if any(char in INVALID_CHARS for char in cleaned):
         return ""

    # 2. Reject names starting with common interjections (creating "Naa Yamato" -> Yamato link)
    #    "Yaa" (Hey), "Naa" (Hey), "Oi" (Hey), "Eeto" (Umm)
    INTERJECTIONS = ("やあ", "なあ", "おい", "えーと", "あの", "その", "えっ", "あッ", "ねえ")
    if cleaned.startswith(INTERJECTIONS):
         return ""

    # 3. Reject names that are just suffixes
    if cleaned.startswith(("さん", "くん", "ちゃん", "様", "殿")):
         return ""
    # -------------------------------------------------------------------------

    # Skip very short names (likely noise), unless it's a single Kanji
    if len(cleaned) < 2:
        # Check if single char is Kanji (roughly 4E00-9FFF)
        if len(cleaned) == 1 and 0x4E00 <= ord(cleaned) <= 0x9FFF:
            return cleaned
        return ""
    
    return cleaned
