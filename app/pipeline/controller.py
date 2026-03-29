# -*- coding: utf-8 -*-
"""Pipeline controller placeholder."""
from __future__ import annotations
import os
import time
from datetime import datetime, timezone
import sys
from dataclasses import dataclass
from typing import Iterable
from app.pipeline.filters import TextFilter
from PySide6 import QtCore
from app.io.project import default_project_dict, save_project
from app.io.style_guide import default_style_guide, load_style_guide
from app.pipeline.steps import build_output_path, build_page_record
from app.models.ollama import list_models
from app.translate.prompts import build_translation_prompt, build_batch_translation_prompt, build_entity_extraction_prompt
import tempfile
import re

import logging

logger = logging.getLogger(__name__)
_GLOSSARY_DEBUG = os.getenv("MT_DEBUG_GLOSSARY") == "1"

class PipelineStatus(QtCore.QObject):
    progress_changed = QtCore.Signal(int)
    eta_changed = QtCore.Signal(str)
    page_changed = QtCore.Signal(int, int)
    message = QtCore.Signal(str)
    queue_reset = QtCore.Signal(list)
    queue_item = QtCore.Signal(int, str)
    total_time_changed = QtCore.Signal(str)
    page_time_changed = QtCore.Signal(str)
    page_ready = QtCore.Signal(int, dict)
    consistency_issue = QtCore.Signal(list)  # Pages needing glossary update
    # Two-Pass Pipeline signals
    prescan_started = QtCore.Signal()
    prescan_progress = QtCore.Signal(int)
    prescan_finished = QtCore.Signal()


@dataclass
class PipelineSettings:
    import_dir: str
    export_dir: str
    json_path: str
    output_suffix: str
    source_lang: str
    target_lang: str
    ollama_model: str
    style_guide_path: str
    font_name: str
    use_gpu: bool
    filter_background: bool
    filter_strength: str
    detector_engine: str
    ocr_engine: str
    inpaint_mode: str
    font_detection: str
    translator_backend: str
    # Generation Options
    ollama_temperature: float
    ollama_top_p: float
    ollama_context: int
    gguf_temperature: float
    gguf_top_p: float
    gguf_model_path: str
    gguf_prompt_style: str
    gguf_n_ctx: int
    gguf_n_gpu_layers: int
    gguf_n_threads: int
    gguf_n_batch: int
    fast_mode: bool
    auto_glossary: bool
    # New settings
    detector_input_size: int
    inpaint_model_id: str
    use_ollama_discovery: bool = False
    files_whitelist: List[str] | None = None
    discovery_model: str | None = None # Model to use for discovery (None=Auto)
    discovery_backend: str = "Ollama" # "Ollama" or "GGUF"
    prescan_enabled: bool = False  # Run pre-scan to build glossary before translation
    prescan_use_ner: bool = False  # Optional heavy NER enhancement for pre-scan
    debug_ocr: bool = False  # Save OCR crop images for debugging


class PipelineWorker(QtCore.QThread):
    progress_changed = QtCore.Signal(int)
    eta_changed = QtCore.Signal(str)
    page_changed = QtCore.Signal(int, int)
    message = QtCore.Signal(str)
    queue_reset = QtCore.Signal(list)
    queue_item = QtCore.Signal(int, str)
    total_time_changed = QtCore.Signal(str)
    page_time_changed = QtCore.Signal(str)
    page_ready = QtCore.Signal(int, dict)
    consistency_issue = QtCore.Signal(list)
    # Two-Pass Pipeline signals
    prescan_started = QtCore.Signal()
    prescan_progress = QtCore.Signal(int)
    prescan_finished = QtCore.Signal()

    def __init__(self, settings: PipelineSettings, parent=None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        images = _list_images(self._settings.import_dir)
        
        # Filter by whitelist if provided (for re-translation)
        if self._settings.files_whitelist:
            whitelist_names = set(os.path.basename(f) for f in self._settings.files_whitelist)
            # Find matching images in the import dir
            images = [img for img in images if os.path.basename(img) in whitelist_names]
            
        total = len(images)
        self.queue_reset.emit(images)
        if total == 0:
            self.message.emit("No images found in import folder.")
            return
        if self._settings.fast_mode:
            self._settings.detector_engine = "PaddleOCR"
            self._settings.inpaint_mode = "fast"
            self._settings.font_detection = "off"
            self._settings.filter_strength = "normal"
            self.message.emit("Fast Mode: detector=PaddleOCR, inpaint=fast, font detection=off.")
        if not os.path.isdir(self._settings.export_dir):
            try:
                os.makedirs(self._settings.export_dir, exist_ok=True)
            except OSError:
                self.message.emit("Failed to create export folder.")
                return

        start_time = time.time()
        from app.detect.paddle_detector import PaddleTextDetector
        from app.ocr.manga_ocr_engine import MangaOcrEngine
        from app.translate.ollama_client import OllamaClient
        from app.render.renderer import render_translations

        ocr_engine = None
        font_detector = None
        auto_glossary_state = None
        pages = []
        try:
            if self._settings.ocr_engine == "MangaOCR":
                worker_error = None
                try:
                    ocr_engine = MangaOcrEngine(self._settings.use_gpu)
                except Exception as exc:
                    if _is_torch_missing(exc):
                        worker_error = exc
                        ocr_engine = None
                    else:
                        try:
                            from app.ocr.manga_ocr_worker import MangaOcrWorker
                            self.message.emit("MangaOCR in-process failed; using worker process.")
                            ocr_engine = MangaOcrWorker(use_gpu=self._settings.use_gpu)
                        except Exception as inner_exc:
                            worker_error = inner_exc
                            ocr_engine = None
                    if ocr_engine is None:
                        if _is_torch_missing(exc) or _is_torch_missing(worker_error):
                            self.message.emit(
                                "MangaOCR unavailable (PyTorch not installed). Falling back to PaddleOCR."
                            )
                        else:
                            self.message.emit(_friendly_model_error(worker_error or exc))
                            self.message.emit("MangaOCR failed; falling back to PaddleOCR.")
                        try:
                            from app.ocr.paddle_ocr_recognizer import PaddleOcrRecognizer
                            ocr_engine = PaddleOcrRecognizer(self._settings.use_gpu)
                            self._settings.ocr_engine = "PaddleOCR"
                        except Exception as fallback_exc:
                            self.message.emit(_friendly_model_error(fallback_exc))
                            return
            else:
                try:
                    from app.ocr.paddle_ocr_recognizer import PaddleOcrRecognizer
                    ocr_engine = PaddleOcrRecognizer(self._settings.use_gpu)
                except Exception as inner_exc:
                    self.message.emit(_friendly_model_error(inner_exc))
                    return

            if self._settings.font_detection != "off":
                try:
                    from app.render.font_detection import FontDetection
                    font_detector = FontDetection(mode=self._settings.font_detection)
                except Exception as exc:
                    self.message.emit(_friendly_model_error(exc))
                    font_detector = None

            try:
                if self._settings.detector_engine == "ComicTextDetector":
                    from app.detect.comic_text_detector import ComicTextDetector
                    detector = ComicTextDetector(self._settings.use_gpu)
                else:
                    detector = PaddleTextDetector(self._settings.use_gpu)
            except Exception as exc:
                self.message.emit(_friendly_model_error(exc))
                return
            background_detector = None
            if not self._settings.filter_background:
                if self._settings.detector_engine == "PaddleOCR":
                    background_detector = detector
                else:
                    try:
                        background_detector = PaddleTextDetector(self._settings.use_gpu)
                    except Exception as exc:
                        self.message.emit(_friendly_model_error(exc))
                        background_detector = None

            try:
                if self._settings.translator_backend == "GGUF":
                    from app.translate.gguf_client import GGUFClient
                    n_gpu_layers = self._settings.gguf_n_gpu_layers
                    # Auto-detect prompt style from filename if generic settings used
                    prompt_style = self._settings.gguf_prompt_style
                    if "sakura" in self._settings.gguf_model_path.lower() and prompt_style == "qwen":
                        prompt_style = "sakura"
                        
                    ollama = GGUFClient(
                        model_path=self._settings.gguf_model_path,
                        prompt_style=prompt_style,
                        n_ctx=self._settings.gguf_n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        n_threads=self._settings.gguf_n_threads,
                        n_batch=self._settings.gguf_n_batch,
                    )
                    if n_gpu_layers != 0 and not getattr(ollama, "gpu_offload", True):
                        self.message.emit(
                            "GGUF is running in CPU mode. For speed, install a CUDA-enabled llama-cpp-python "
                            "build or switch to Ollama."
                        )
                else:
                    ollama = OllamaClient()
                    if not ollama.is_available():
                        self.message.emit("Ollama server is not running. Start it with: ollama serve")
                        return
            except Exception as exc:
                self.message.emit(_friendly_model_error(exc))
                return
            model_name = (
                self._settings.gguf_model_path
                if self._settings.translator_backend == "GGUF"
                else self._settings.ollama_model
            )
            resolved_model = _resolve_model(self._settings.ollama_model)
            if self._settings.translator_backend == "Ollama":
                if resolved_model and self._settings.ollama_model != "auto-detect":
                    available = list_models()
                    if available and resolved_model not in available:
                        self.message.emit(f"Ollama model not found: {resolved_model}")
                        return
            elif not self._settings.gguf_model_path:
                self.message.emit("GGUF model path is required for GGUF backend.")
                return
            
            # Ensure model name is set on the client for glossary translation
            if hasattr(ollama, "translate_glossary"):
                 # Use resolved model for Ollama, or path for GGUF (though GGUF uses internal path)
                 setattr(ollama, "model_name", resolved_model or model_name)

            if self._settings.auto_glossary and not self._settings.style_guide_path:
                self._settings.style_guide_path = os.path.join(self._settings.export_dir, "style_guide.json")
            style_guide = _load_style_guide(self._settings.style_guide_path)
            if self._settings.auto_glossary and self._settings.style_guide_path and not os.path.isfile(self._settings.style_guide_path):
                try:
                    from app.io.style_guide import save_style_guide
                    save_style_guide(self._settings.style_guide_path, style_guide)
                except Exception:
                    pass
            context_window = []
            translation_cache: dict[str, str] = {}
            project = default_project_dict()
            project["project"]["name"] = os.path.basename(self._settings.import_dir.rstrip("\\/"))
            project["project"]["language"]["source"] = _lang_code(self._settings.source_lang)
            project["project"]["language"]["target"] = _lang_code(self._settings.target_lang)
            project["project"]["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            project["project"]["model"]["detector"] = self._settings.detector_engine
            project["project"]["model"]["ocr"] = self._settings.ocr_engine
            if self._settings.translator_backend == "GGUF":
                project["project"]["model"]["translator"] = f"gguf:{self._settings.gguf_model_path}"
            else:
                project["project"]["model"]["translator"] = f"ollama:{self._settings.ollama_model}"
            project["project"]["style_guide"] = self._settings.style_guide_path or ""
            auto_glossary_state = None
            if self._settings.auto_glossary:
                auto_glossary_state = {"counts": {}, "map": {}}
            
            # Pre-Scan Mode: Build complete glossary before translation
            if self._settings.prescan_enabled and self._settings.auto_glossary:
                self.prescan_started.emit()
                self.message.emit("Pre-Scan Mode: Building glossary before translation...")
                try:
                    from app.pipeline.prescan import prescan_for_glossary
                    style_guide = prescan_for_glossary(
                        import_dir=self._settings.import_dir,
                        images=images,
                        style_guide=style_guide,
                        settings=self._settings,
                        progress_callback=lambda p: self.prescan_progress.emit(p),
                        message_callback=lambda m: self.message.emit(f"[Pre-Scan] {m}"),
                        stop_check=lambda: self._stop_requested,
                        translator=ollama,
                        detector=detector,
                        ocr_engine=ocr_engine,
                    )
                    # Save the updated style guide
                    if self._settings.style_guide_path:
                        from app.io.style_guide import save_style_guide
                        save_style_guide(self._settings.style_guide_path, style_guide)
                    self.message.emit(f"Pre-Scan complete: {len(style_guide.get('glossary', []))} glossary entries.")
                except Exception as e:
                    self.message.emit(f"Pre-Scan failed: {e}. Continuing with normal translation.")
                    import logging
                    logging.getLogger(__name__).exception("Pre-scan error")
                finally:
                    self.prescan_finished.emit()
            for index, name in enumerate(images, start=1):
                if self._stop_requested:
                    self.message.emit("Stopped")
                    return

                page_start = time.time()
                self.queue_item.emit(index - 1, "processing")
                self.page_changed.emit(index, total)

                source_path = os.path.join(self._settings.import_dir, name)
                output_path = build_output_path(self._settings.export_dir, name, self._settings.output_suffix)

                try:
                    regions = _process_page(
                        source_path,
                        detector,
                        ocr_engine,
                        ollama,
                        model_name,
                        style_guide,
                        context_window,
                        self._settings.target_lang,
                        self._settings.source_lang,
                        self._settings.font_name,
                        self._settings.filter_background,
                        self._settings.filter_strength,
                        font_detector,
                        translation_cache,
                        background_detector,

                        auto_glossary_state,
                        image_input_size=self._settings.detector_input_size,
                        style_guide_path=self._settings.style_guide_path,
                        allow_ollama_discovery=self._settings.use_ollama_discovery,
                        discovery_model=self._settings.discovery_model,
                        settings=self._settings,
                    )
                    if auto_glossary_state is not None:
                        new_client = auto_glossary_state.pop("translation_client", None)
                        if new_client is not None and new_client is not ollama:
                            ollama = new_client
                except Exception as exc:
                    page_elapsed = time.time() - page_start
                    self.queue_item.emit(index - 1, f"error ({_format_seconds(page_elapsed)}): {exc}")
                    self.message.emit(f"Failed to process {name}: {exc}")
                    continue

                try:
                    render_translations(
                        source_path,
                        output_path,
                        regions,
                        self._settings.font_name,
                        inpaint_mode=self._settings.inpaint_mode,
                        use_gpu=self._settings.use_gpu,
                        model_id=self._settings.inpaint_model_id,
                    )
                except Exception as exc:
                    page_elapsed = time.time() - page_start
                    self.queue_item.emit(index - 1, f"error ({_format_seconds(page_elapsed)}): {exc}")
                    self.message.emit(f"Failed to render {name}: {exc}")
                    continue

                page_id = os.path.splitext(name)[0]
                page_record = build_page_record(source_path, page_id, regions, output_path)
                pages.append(page_record)
                self.page_ready.emit(index - 1, page_record)
                
                # Track glossary size at this page for consistency checking
                if auto_glossary_state is not None:
                    with _glossary_lock:
                        current_glossary_size = len(auto_glossary_state.get("map", {}))
                        snapshots = auto_glossary_state.setdefault("page_snapshots", {})
                        snapshots[index - 1] = current_glossary_size

                page_elapsed = time.time() - page_start
                self.page_time_changed.emit(f"Page: {_format_seconds(page_elapsed)}")
                self.queue_item.emit(index - 1, f"done ({_format_seconds(page_elapsed)})")
                progress = int(index / total * 100)
                self.progress_changed.emit(progress)

                elapsed = time.time() - start_time
                self.total_time_changed.emit(f"Total: {_format_seconds(elapsed)}")
                avg = elapsed / index
                remaining = avg * (total - index)
                self.eta_changed.emit(_format_eta(remaining))

                # --- PER-PAGE MEMORY CLEANUP ---
                # Prevent memory accumulation over long chapters (fixes 2GB+ leak)
                try:
                    del regions
                except NameError:
                    pass
                import gc
                gc.collect()
                
                # Clear CUDA cache every 5 pages to balance speed vs memory
                if self._settings.use_gpu and index % 5 == 0:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

            project["pages"] = pages
            json_path = self._settings.json_path or os.path.join(self._settings.export_dir, "project.json")
            try:
                save_project(json_path, project)
            except OSError:
                self.message.emit("Failed to write project JSON.")
            
            # --- MEMORY CLEANUP START ---
            # Flush Python Garbage Collector
            import gc
            gc.collect()
            
            # Flush PyTorch VRAM Cache (if used)
            if self._settings.use_gpu:
                 try:
                     import torch
                     if torch.cuda.is_available():
                         torch.cuda.empty_cache()
                 except Exception:
                     pass
            # --- MEMORY CLEANUP END ---
            
            total_elapsed = time.time() - start_time
            self.total_time_changed.emit(f"Total: {_format_seconds(total_elapsed)}")
            self.message.emit("Completed")
        finally:
            if auto_glossary_state and self._settings.style_guide_path:
                try:
                    # Force final discovery if buffer has remaining text
                    with _glossary_lock:
                        remaining_buffer = auto_glossary_state.get("buffer", [])
                        is_running = auto_glossary_state.get("is_running", False)

                    if remaining_buffer and not is_running:
                        self.message.emit("Running final Auto-Glossary discovery...")
                        # Run synchronously (not in thread) to ensure completion
                        use_deep_scan = bool(self._settings.use_ollama_discovery)
                        discovery_client = ollama
                        created_client = None
                        discovery_model = self._settings.discovery_model
                        if use_deep_scan:
                            backend = self._settings.discovery_backend
                            if backend == "GGUF" or (discovery_model and ".gguf" in discovery_model.lower()):
                                target_path = str(discovery_model or "").strip()
                                if target_path and os.path.isfile(target_path):
                                    if hasattr(ollama, "_model_path") and os.path.abspath(target_path) == os.path.abspath(getattr(ollama, "_model_path", "")):
                                        discovery_client = ollama
                                    else:
                                        from app.translate.gguf_client import GGUFClient
                                        n_gpu_layers = self._settings.gguf_n_gpu_layers
                                        created_client = GGUFClient(
                                            model_path=target_path,
                                            prompt_style="extract",
                                            n_ctx=2048,
                                            n_gpu_layers=n_gpu_layers,
                                            n_threads=max(1, self._settings.gguf_n_threads),
                                            n_batch=min(128, self._settings.gguf_n_batch),
                                        )
                                        discovery_client = created_client
                                else:
                                    self.message.emit("Deep Scan GGUF model path is invalid for final discovery.")
                                    use_deep_scan = False
                            elif backend == "Ollama":
                                if hasattr(ollama, "list_models"):
                                    discovery_client = ollama
                                elif self._settings.use_ollama_discovery:
                                    try:
                                        from app.translate.ollama_client import OllamaClient
                                        new_client = OllamaClient()
                                        if new_client.is_available():
                                            discovery_client = new_client
                                            created_client = new_client
                                        else:
                                            use_deep_scan = False
                                    except Exception:
                                        use_deep_scan = False
                        if use_deep_scan and discovery_client:
                            _run_sakura_discovery(
                                discovery_client,
                                model_name,
                                self._settings.source_lang,
                                self._settings.target_lang,
                                auto_glossary_state,
                                style_guide,
                                self._settings.style_guide_path,
                                discovery_model,
                            )
                        else:
                            _run_discovery(
                                ollama,
                                model_name,
                                self._settings.source_lang,
                                self._settings.target_lang,
                                auto_glossary_state,
                                style_guide,
                                self._settings.style_guide_path,
                                bool(ollama and hasattr(ollama, "generate")),
                            )
                        if created_client is not None and hasattr(created_client, "close"):
                            try:
                                created_client.close()
                            except Exception:
                                pass

                    # Ensure we have the latest data from background threads
                    with _glossary_lock:
                        learned_map = auto_glossary_state.get("map", {})
                        learned_chars = auto_glossary_state.get("characters", [])

                    if learned_map or learned_chars:
                        from app.io.style_guide import save_style_guide, load_style_guide
                        # Re-load to avoid overwriting external edits
                        current_sg = _load_style_guide(self._settings.style_guide_path)
                        updated_sg = _merge_glossary(current_sg, learned_map, learned_chars)
                        updated_sg = _sanitize_style_guide(updated_sg, self._settings.target_lang)
                        save_style_guide(self._settings.style_guide_path, updated_sg)
                        self.message.emit(
                            f"Auto-Glossary: Saved {len(learned_map)} terms, {len(learned_chars)} characters."
                        )
                except Exception as e:
                    self.message.emit(f"Failed to save Auto-Glossary data: {e}")
            
            # Consistency Check: Compare early pages vs final glossary
            # SKIP if running in re-translation mode (files_whitelist is set)
            # to prevent infinite loop: re-translate → consistency check → dialog → re-translate...
            if auto_glossary_state is None:
                pass
            elif self._settings.files_whitelist:
                self.message.emit("Skipping consistency check (re-translation mode).")
            elif self._settings.prescan_enabled:
                self.message.emit("Skipping consistency check (Pre-Scan enabled).")
            else:
                try:
                    final_style = _load_style_guide(self._settings.style_guide_path)
                    cleaned_style = _sanitize_style_guide(final_style, self._settings.target_lang)
                    if cleaned_style is not final_style and self._settings.style_guide_path:
                        from app.io.style_guide import save_style_guide
                        save_style_guide(self._settings.style_guide_path, cleaned_style)
                    inconsistent_pages = _find_inconsistent_pages(pages, cleaned_style)
                    if inconsistent_pages:
                        self.message.emit(
                            f"Consistency check: {len(inconsistent_pages)} pages may have "
                            f"outdated name translations."
                        )
                        # Emit signal for UI to handle
                        self.consistency_issue.emit(inconsistent_pages)
                except Exception as e:
                    self.message.emit(f"Consistency check failed: {e}")

            try:
                if hasattr(ocr_engine, "close"):
                    ocr_engine.close()
            except Exception:
                pass


class PipelineController(QtCore.QObject):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.status = PipelineStatus()
        self._running = False
        self._worker: PipelineWorker | None = None

    def start(self, settings: PipelineSettings) -> None:
        if self._running:
            return
        if not settings.import_dir:
            self.status.message.emit("Import folder is required.")
            return
        if not settings.export_dir:
            self.status.message.emit("Export folder is required.")
            return
        self._running = True
        self._worker = PipelineWorker(settings, self)
        self._worker.progress_changed.connect(self.status.progress_changed.emit)
        self._worker.eta_changed.connect(self.status.eta_changed.emit)
        self._worker.page_changed.connect(self.status.page_changed.emit)
        self._worker.total_time_changed.connect(self.status.total_time_changed.emit)
        self._worker.page_time_changed.connect(self.status.page_time_changed.emit)
        self._worker.message.connect(self.status.message.emit)
        self._worker.queue_reset.connect(self.status.queue_reset.emit)
        self._worker.queue_item.connect(self.status.queue_item.emit)
        self._worker.page_ready.connect(self.status.page_ready.emit)
        self._worker.consistency_issue.connect(self.status.consistency_issue.emit)
        # Two-Pass Pipeline signals
        self._worker.prescan_started.connect(self.status.prescan_started.emit)
        self._worker.prescan_progress.connect(self.status.prescan_progress.emit)
        self._worker.prescan_finished.connect(self.status.prescan_finished.emit)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self.status.message.emit("Started")

    def stop(self) -> None:
        if not self._running:
            return
        if self._worker:
            self._worker.request_stop()
        self.status.message.emit("Stopping...")

    def _on_finished(self):
        self._running = False
        self._worker = None

    def start_deep_scan(self, settings: PipelineSettings):
        """Start deep scan worker."""
        if self._running:
            return
        
        self.deep_scan_worker = DeepScanWorker(settings)
        # Relay signals? For now just simple finished
        self.deep_scan_worker.finished.connect(self._on_deep_scan_finished)
        self.deep_scan_worker.start()
        
    def _on_deep_scan_finished(self):
        self.status.message.emit("Deep scan completed. Glossary updated.")
        self.status.consistency_issue.emit([]) # Signal to maybe refresh? 
        # Actually Main Window handles the dialog logic, it waits for this worker to finish?
        # We'll rely on the worker reference in MainWindow if we want to block interaction.


class DeepScanWorker(QtCore.QThread):
    finished = QtCore.Signal()
    
    def __init__(self, settings: PipelineSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        
    def run(self):
        try:
            # Load project pages to get text
            # We assume the project is located at settings.json_path
            if not os.path.exists(self.settings.json_path):
                return
                
            import json
            from app.translate.ollama_client import OllamaClient
            from app.models.ollama import list_models
            
            with open(self.settings.json_path, 'r', encoding='utf-8') as f:
                project = json.load(f)
                
            pages = project.get("pages", [])
            accumulated = []
            if isinstance(pages, dict):
                sorted_keys = sorted(pages.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
                page_items = [pages[k] for k in sorted_keys]
            else:
                page_items = pages
            for page in page_items:
                if not isinstance(page, dict):
                    continue
                blocks = page.get("regions", []) or page.get("blocks", [])
                for b in blocks:
                    if isinstance(b, dict) and b.get("ocr_text"):
                        t = str(b["ocr_text"]).replace("\n", "").strip()
                        if t:
                            accumulated.append(t)
            
            if not accumulated:
                return

            # Hybrid Strategy:
            # Even if user checks "GGUF" for translation speed, we can attempt
            # to use a smart Ollama model (like Qwen) for discovery if available.
            
            backend = getattr(self.settings, "discovery_backend", "Ollama")
            discovery_model = getattr(self.settings, "discovery_model", None)
            model_to_use = self.settings.ollama_model

            ollama = None
            if backend == "GGUF" or (discovery_model and ".gguf" in str(discovery_model).lower()):
                if discovery_model and "sakura" in str(discovery_model).lower():
                    print("DeepScan: Sakura GGUF is translation-only; skipping Deep Scan.")
                    return
                if discovery_model and os.path.isfile(discovery_model):
                    from app.translate.gguf_client import GGUFClient
                    n_gpu_layers = self.settings.gguf_n_gpu_layers
                    ollama = GGUFClient(
                        model_path=discovery_model,
                        prompt_style="extract",
                        n_ctx=2048,
                        n_gpu_layers=n_gpu_layers,
                        n_threads=max(1, self.settings.gguf_n_threads),
                        n_batch=min(128, self.settings.gguf_n_batch),
                    )
                    model_to_use = "gguf_model"
                else:
                    print("DeepScan: GGUF backend selected but model path is invalid")
                    return
            else:
                ollama = OllamaClient()
                if not ollama.is_available():
                    print("DeepScan: Ollama server is not running")
                    return
                if discovery_model and str(discovery_model).strip() and "auto" not in str(discovery_model).lower():
                    model_to_use = str(discovery_model).strip()
                if model_to_use and "sakura" in model_to_use.lower():
                    model_to_use = ""
                if not model_to_use or "auto" in model_to_use.lower():
                    available_models = list_models()
                    qwen = next((m for m in available_models if "qwen" in m.lower()), None)
                    non_sakura = next((m for m in available_models if "sakura" not in m.lower()), None)
                    model_to_use = qwen if qwen else (non_sakura if non_sakura else "")
            
            if not model_to_use:
                # No model found
                print("DeepScan: No Ollama model found")
                return
                
            print(f"DeepScan: using model {model_to_use}")
            
            if not ollama:
                print("DeepScan: No discovery client available")
                return
            if not model_to_use:
                print("DeepScan: No model found")
                return
            # Load style guide
            base_style = _load_style_guide(self.settings.style_guide_path)
            
            # Run discovery
            # Mock state
            state = {"buffer": accumulated}
            
            _run_sakura_discovery(
                ollama=ollama,
                model=model_to_use,
                source_lang=self.settings.source_lang,
                target_lang=self.settings.target_lang,
                state=state,
                base_style=base_style,
                style_guide_path=self.settings.style_guide_path
            )
            
        except Exception as e:
            print(f"Deep scan error: {e}")
        finally:
            if "ollama" in locals() and hasattr(ollama, "close"):
                try:
                    ollama.close()
                except Exception:
                    pass
            self.finished.emit()


def _list_images(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    names = []
    for entry in os.listdir(folder):
        _, ext = os.path.splitext(entry)
        if ext.lower() in allowed:
            names.append(entry)
    names.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)])
    return names


def _format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "00:00"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _lang_code(label: str) -> str:
    mapping = {
        "Japanese": "ja",
        "Simplified Chinese": "zh-Hans",
        "English": "en",
    }
    return mapping.get(label, label)


def _friendly_model_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "paddleocr" in lowered:
        return "PaddleOCR is not installed. Install it with: pip install paddleocr"
    if "export_model.py" in lowered or "jit.save" in lowered:
        return "PaddleOCR export failed. Try unchecking 'Enable GPU when available' and retry."
    if "failed to load torch" in lowered:
        return (
            "Torch failed to load (DLL dependency error). Restart the app after installing conda PyTorch. "
            "If it persists, reboot Windows to refresh DLL search paths."
        )
    if "no module named 'torch'" in lowered:
        return (
            "PyTorch is not installed in the current environment. Install it or switch OCR Engine to PaddleOCR. "
            "Suggested: pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )
    if "manga-ocr" in lowered or "manga_ocr" in lowered:
        return f"MangaOCR failed to load: {text}"
    if "comictextdetector" in lowered or "comic-text-detector" in lowered or "utils.general" in lowered:
        return (
            "ComicTextDetector is not ready. Download comictextdetector.pt.onnx (CPU) or "
            "comictextdetector.pt (GPU) from https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1 "
            "and place it under models/comic-text-detector."
        )
    if "llama-cpp-python is not installed" in lowered or "no module named 'llama_cpp'" in lowered:
        return (
            "GGUF backend failed: llama-cpp-python is missing in the current environment. "
            "Install it with: pip install llama-cpp-python, or switch Translator backend to Ollama."
        )
    if "gguf model not found:" in lowered:
        return f"GGUF backend failed: {text}"
    if "gguf" in lowered or "llama-cpp-python" in lowered or "llama_cpp" in lowered:
        return f"GGUF backend failed: {text}"
    if "yuzumarker" in lowered or "font detection" in lowered:
        return (
            "Font detection failed to initialize. Ensure the font model checkpoint is set and dependencies are installed."
        )
    if "numpy" in lowered and "abi" in lowered:
        return (
            "NumPy ABI mismatch. Reinstall numpy and the OCR deps. "
            "Suggested: pip install -U numpy==1.26.4 paddleocr manga-ocr"
        )
    if "shm.dll" in lowered or "winerror 127" in lowered:
        return (
            "PyTorch DLL load failed. Reinstall torch in the conda env. "
            "Suggested: pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )
    return f"Failed to initialize models: {text}"


def _is_torch_missing(exc: Exception | None) -> bool:
    if exc is None:
        return False
    return "no module named 'torch'" in str(exc).lower()


def _load_style_guide(path: str):
    if path and os.path.isfile(path):
        try:
            # Handle empty or corrupt files gracefully
            if os.path.getsize(path) == 0:
                return default_style_guide()
                
            guide = load_style_guide(path)
            
            # --- SANITIZATION START ---
            # Remove hallucinated entries (e.g. "Thinking" output leaking into "Source")
            if "glossary" in guide and isinstance(guide["glossary"], list):
                cleaned = []
                for item in guide["glossary"]:
                    if not isinstance(item, dict): continue
                    source = str(item.get("source", "")).strip()
                    # Filter: Too long source (likely a sentence/instruction) or contains prohibited keywords
                    if len(source) > 30 or "处理用户" in source or "Need to" in source or "require" in source:
                        continue
                    cleaned.append(item)
                guide["glossary"] = cleaned
            # --- SANITIZATION END ---
            
            return guide
        except Exception:
            # Return default if file is corrupt (prevent crash)
            return default_style_guide()
    return default_style_guide()


_paddle_fallback_instance = None
_ocr_debug_counter = 0

def _is_valid_japanese(text: str) -> float:
    """
    Score how likely text is valid Japanese (0.0 to 1.0).
    Higher score = more valid Japanese characters.
    """
    if not text:
        return 0.0
    valid = 0
    for c in text:
        code = ord(c)
        # Hiragana, Katakana, Kanji, punctuation
        if (0x3040 <= code <= 0x30FF or  # Hiragana + Katakana
            0x4E00 <= code <= 0x9FFF or  # Kanji
            0x3000 <= code <= 0x303F or  # Japanese punctuation
            c in '!?。、…・「」『』（）'):
            valid += 1
    return valid / len(text) if text else 0.0

def _recognize_with_fallback(ocr_engine, crop, settings, bbox=None) -> tuple[str, float]:
    """
    OCR recognition using MangaOCR.
    For wide boxes (impact text), compares MangaOCR and PaddleOCR results
    and picks the one with more valid Japanese characters.
    """
    global _paddle_fallback_instance, _ocr_debug_counter
    text = ""
    conf = 1.0
    
    # Detect wide boxes (likely impact/title text)
    is_wide_box = False
    if bbox and len(bbox) >= 4:
        x, y, w, h = bbox[:4]
        if h > 0 and w > h * 2.5:  # Width > 2.5x height (stricter threshold)
            is_wide_box = True
    
    # DEBUG: Save crop images
    if settings and getattr(settings, 'debug_ocr', False):
        try:
            import os
            debug_dir = os.path.join(settings.export_dir, "ocr_debug")
            os.makedirs(debug_dir, exist_ok=True)
            crop_path = os.path.join(debug_dir, f"crop_{_ocr_debug_counter:04d}_bbox_{bbox}.png")
            if hasattr(crop, 'save'):
                crop.save(crop_path)
                print(f"[OCR DEBUG] Saved crop: {crop_path}")
            _ocr_debug_counter += 1
        except Exception as e:
            print(f"[OCR DEBUG] Failed to save crop: {e}")
    
    # Use MangaOCR (primary engine)
    # Match original repo: No padding. Pass crop directly.
    padded_main = crop

    if hasattr(ocr_engine, "recognize_with_confidence"):
        text, conf = ocr_engine.recognize_with_confidence(padded_main)
    else:
        text = ocr_engine.recognize(padded_main)
        conf = 1.0
    
    # DEBUG: Log OCR result
    if settings and getattr(settings, 'debug_ocr', False):
        print(f"[OCR DEBUG] bbox={bbox} -> MangaOCR='{text}' conf={conf:.3f}")
    
    # For wide boxes (impact text), try PaddleOCR ONLY if MangaOCR fails or is weak
    # "Extreme cases" fallback as requested
    if is_wide_box and settings:
        manga_score = _is_valid_japanese(text)
        
        # Only try fallback if MangaOCR result is poor
        # Threshold: < 0.5 valid Japanese OR very short text (< 2 chars)
        if manga_score < 0.5 or len(text.strip()) < 2:
            if _paddle_fallback_instance is None:
                try:
                    from app.ocr.paddle_ocr_recognizer import PaddleOcrRecognizer
                    _paddle_fallback_instance = PaddleOcrRecognizer(settings.use_gpu if settings else False)
                except Exception as e:
                    print(f"[OCR] Failed to load PaddleOCR: {e}")
            
            if _paddle_fallback_instance:
                try:
                    p_text = _paddle_fallback_instance.recognize(padded_main)
                    p_text = p_text.replace(" ", "") if p_text else ""
                    
                    paddle_score = _is_valid_japanese(p_text)
                    manga_stripped = text.replace(" ", "")
                    
                    if settings and getattr(settings, 'debug_ocr', False):
                        print(f"[OCR DEBUG] Fallback Triggered | MangaOCR='{text}'({manga_score:.2f}) vs PaddleOCR='{p_text}'({paddle_score:.2f})")
                    
                    # Only switch if PaddleOCR is significantly better
                    if paddle_score > manga_score and len(p_text) >= len(manga_stripped):
                        if settings and getattr(settings, 'debug_ocr', False):
                            print(f"[OCR DEBUG] Using PaddleOCR result (Rescue)")
                        text = p_text
                        conf = 0.9
                except Exception:
                    pass
                finally:
                    # CRITICAL: Unload PaddleOCR immediately to prevent VRAM leak/contention
                    # This fallback is rare, so we prioritize memory over reload speed
                    try:
                        if hasattr(_paddle_fallback_instance, "unload"):
                            _paddle_fallback_instance.unload()
                        del _paddle_fallback_instance
                        _paddle_fallback_instance = None
                        import gc
                        gc.collect()
                    except Exception:
                        pass
                    _paddle_fallback_instance = None
        else:
             if settings and getattr(settings, 'debug_ocr', False):
                 print(f"[OCR DEBUG] Skipping PaddleOCR fallback (MangaOCR Score {manga_score:.2f} >= 0.5)")

    if settings and getattr(settings, 'debug_ocr', False):
         print(f"[OCR CRITICAL] Chosen Text: '{text}' (ValidScore: {_is_valid_japanese(text):.2f}) for bbox={bbox}")

    return _clean_ocr_text(text), conf


def _process_page(
    image_path: str,
    detector,
    ocr_engine,
    ollama,
    model: str,
    style_guide: dict,
    context_window: list,
    target_lang: str,
    source_lang: str,
    font_name: str,
    filter_background: bool,
    filter_strength: str,
    font_detector,
    translation_cache: dict[str, str],
    background_detector,
    auto_glossary_state,
    image_input_size: int = 1024,
    style_guide_path: str = "",
    allow_ollama_discovery: bool = False,
    discovery_model: str | None = None,
    settings: PipelineSettings | None = None,
) -> list[dict]:
    # Initialize Filter
    text_filter = TextFilter(settings)

    if not image_path or not os.path.exists(image_path):
        return []
    image_size = _get_image_size(image_path)
    page_image = _load_image_for_crop(image_path)
    detections = _detect_regions(
        detector,
        image_path,
        image_size,
        input_size=image_input_size,
        use_gpu=bool(settings and settings.use_gpu),
    )
        
    merge = getattr(detector, "merge_mode", "auto") != "none"
    groups = _merge_detections(detections, image_size, merge=merge)
    groups = _sort_groups(groups)
    if not groups:
        groups = [{"bbox": _polygon_to_bbox(p), "polygons": [p], "conf": float(c or 0.0)} for p, c in detections]
    bubble_boxes = [g["bbox"] for g in groups]
    if background_detector is not None:
        bg_detections = _detect_regions(
            background_detector,
            image_path,
            image_size,
            input_size=image_input_size,
            use_gpu=bool(settings and settings.use_gpu),
        )
        for polygon, conf in bg_detections:
            try:
                bbox = _polygon_to_bbox(polygon)
            except Exception:
                continue
            if any(_overlap_ratio(bbox, bb) > 0.2 for bb in bubble_boxes):
                continue
            groups.append(
                {
                    "bbox": bbox,
                    "polygons": [polygon],
                    "conf": float(conf or 0.0),
                    "bg_text": True,
                }
            )
    groups = _dedupe_groups(groups)
    groups = _sort_groups(groups)
    regions = []
    pending_texts: dict[str, list[str]] = {}
    glossary_texts: list[str] = []
    for idx, group in enumerate(groups):
        bbox = group["bbox"]
        polygons = group["polygons"]
        det_conf = group["conf"]
        is_bg_group = bool(group.get("bg_text"))
        if is_bg_group:
            crop = _crop_image(image_path, bbox, image_obj=page_image)
            if crop is None:
                continue
            ocr_text, ocr_conf = _recognize_with_fallback(ocr_engine, crop, settings, bbox)
            if not ocr_text:
                continue
            region_type, semantic_bg, semantic_ignore, semantic_review, render_updates = _classify_semantic_region(
                ocr_text,
                bbox,
                image_size,
                det_conf,
                ocr_conf,
                page_image,
                text_filter,
                initial_bg=True,
            )
            skip_text = _should_skip_text(ocr_text, bbox, image_size) if semantic_bg else False
            region = _region_record(
                idx,
                polygons,
                bbox,
                ocr_text,
                "",
                det_conf,
                bg_text=semantic_bg,
                needs_review=semantic_review or skip_text,
                ignore=semantic_ignore or skip_text,
                font_name=font_name,
                region_type=region_type,
                ocr_conf=ocr_conf,
                render_updates=render_updates,
            )
            regions.append(region)
            if region.get("ignore"):
                continue
            glossary_texts.append(ocr_text)
            cached = translation_cache.get(ocr_text)
            if cached is not None:
                region["translation"] = cached
            else:
                pending_texts.setdefault(ocr_text, []).append(region["region_id"])
            continue
        bg_text, needs_review = _classify_region(
            bbox,
            image_size,
            det_conf,
            filter_background,
            filter_strength,
        )
        if bg_text:
            crop = _crop_image(image_path, bbox, image_obj=page_image)
            if crop is None:
                continue
            ocr_text, ocr_conf = _recognize_with_fallback(ocr_engine, crop, settings, bbox)
            if not ocr_text:
                continue
            region_type, semantic_bg, semantic_ignore, semantic_review, render_updates = _classify_semantic_region(
                ocr_text,
                bbox,
                image_size,
                det_conf,
                ocr_conf,
                page_image,
                text_filter,
                initial_bg=bg_text,
            )
            skip_text = _should_skip_text(ocr_text, bbox, image_size)
            ignore = semantic_ignore or bool(filter_background and skip_text and semantic_bg)
            
            region = _region_record(
                idx,
                polygons,
                bbox,
                ocr_text,
                "",
                det_conf,
                bg_text=semantic_bg,
                needs_review=needs_review or semantic_review or skip_text,
                ignore=ignore,
                font_name=font_name,
                region_type=region_type,
                ocr_conf=ocr_conf,
                render_updates=render_updates,
            )
            regions.append(region)
            if ignore:
                continue
            glossary_texts.append(ocr_text)
            cached = translation_cache.get(ocr_text)
            if cached is not None:
                region["translation"] = cached
            else:
                pending_texts.setdefault(ocr_text, []).append(region["region_id"])
            continue
        crop = _crop_image(image_path, bbox, image_obj=page_image)
        if crop is None:
            continue
        ocr_text, ocr_conf = _recognize_with_fallback(ocr_engine, crop, settings, bbox)
        if not ocr_text:
            continue
        region_type, semantic_bg, semantic_ignore, semantic_review, render_updates = _classify_semantic_region(
            ocr_text,
            bbox,
            image_size,
            det_conf,
            ocr_conf,
            page_image,
            text_filter,
            initial_bg=False,
        )
        if region_type == "speech_bubble" and _should_ignore_speech_fragment(ocr_text, bbox, image_size, ocr_conf):
            semantic_ignore = True
            semantic_review = True
        if semantic_ignore:
            regions.append(
                _region_record(
                    idx,
                    polygons,
                    bbox,
                    ocr_text,
                    "",
                    det_conf,
                    bg_text=semantic_bg,
                    needs_review=True,
                    ignore=True,
                    font_name=font_name,
                    region_type=region_type,
                    ocr_conf=ocr_conf,
                    render_updates=render_updates,
                )
            )
            continue
        glossary_texts.append(ocr_text)
        # REMOVED: _should_skip_text filter for speech bubbles
        # Speech bubbles detected by the detector should NEVER be filtered
        # They are legitimate dialogue that must always be translated
        detected_font = None
        if font_detector is not None:
            try:
                detected_font = font_detector.detect(crop)
            except Exception:
                detected_font = None



        # REMOVED: TextFilter check for speech bubbles
        # Speech bubbles detected by the detector should NEVER be filtered
        # They are legitimate dialogue - always translate them
        
        region = _region_record(
            idx,
            polygons,
            bbox,
            ocr_text,
            "",
            det_conf,
            bg_text=semantic_bg,
            needs_review=needs_review or semantic_review,
            ignore=False,
            font_name=font_name,
            region_type=region_type,
            ocr_conf=ocr_conf,
            render_updates=render_updates,
        )
        if detected_font:
            if target_lang == "Simplified Chinese" and not _is_font_allowed_for_cn(detected_font):
                detected_font = None
        if detected_font:
            render = region.get("render", {})
            render["font"] = detected_font
            region["render"] = render
        regions.append(region)
        cached = translation_cache.get(ocr_text)
        if cached is not None:
            region["translation"] = cached
        elif region.get("ignore") and text_filter.should_ignore(ocr_text, "background_text"):
            # Skip background text if the filter agrees it's skippable (SFX)
            # This allows Plot Descriptions (which don't look like SFX) to pass through.
            pass
        else:
            pending_texts.setdefault(ocr_text, []).append(region["region_id"])
    active_style_guide = style_guide
    
    # Skip runtime discovery if Pre-Scan is enabled (glossary is already built)
    should_run_discovery = (
        auto_glossary_state is not None 
        and glossary_texts 
        and not (settings and settings.prescan_enabled)
    )
    
    if should_run_discovery:
        if _GLOSSARY_DEBUG:
            import tempfile
            log_path = os.path.join(tempfile.gettempdir(), "auto_glossary_debug.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"  -> Calling _apply_auto_glossary with {len(glossary_texts)} texts\n")
        active_style_guide = _apply_auto_glossary(
            style_guide,
            auto_glossary_state,
            glossary_texts,
            ollama,
            model,
            source_lang,
            target_lang,
            style_guide_path=style_guide_path,
            allow_ollama=allow_ollama_discovery,
            discovery_model=discovery_model,
            settings=settings,
            mecab_only=not allow_ollama_discovery,
        )
        if auto_glossary_state is not None:
            new_client = auto_glossary_state.get("translation_client")
            if new_client is not None:
                ollama = new_client
    elif auto_glossary_state is not None:
        if _GLOSSARY_DEBUG:
            import tempfile
            log_path = os.path.join(tempfile.gettempdir(), "auto_glossary_debug.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("  -> SKIPPED: glossary_texts is EMPTY\n")
    if pending_texts:
        prompt_style_guide = _build_page_style_guide(
            active_style_guide,
            list(pending_texts.keys()),
        )
        context_lines = _recent_context_lines(context_window)
        items = []
        id_to_text: dict[str, str] = {}
        for idx, text in enumerate(pending_texts.keys()):
            item_id = f"t{idx:03d}"
            items.append({"id": item_id, "text": text})
            id_to_text[item_id] = text
        translations = _batch_translate(
            ollama,
            model,
            source_lang,
            target_lang,
            prompt_style_guide,
            items,
            context_lines=context_lines,
            settings=settings,
        )
        text_to_translation: dict[str, str] = {}
        if translations:
            for item_id, translation in translations.items():
                text = id_to_text.get(item_id)
                if text is not None:
                    # Apply glossary enforcement to ensure consistent name translations
                    enforced = _enforce_glossary(translation, text, active_style_guide)
                    text_to_translation[text] = enforced
        if not text_to_translation:
            for text in pending_texts.keys():
                raw_trans = _translate_single(
                    ollama,
                    model,
                    source_lang,
                    target_lang,
                    prompt_style_guide,
                    text,
                    context_lines=context_lines,
                    settings=settings,
                )
                # Apply glossary enforcement
                text_to_translation[text] = _enforce_glossary(raw_trans, text, active_style_guide)
        for text, region_ids in pending_texts.items():
            is_bubble = False
            for region in regions:
                if region["region_id"] in region_ids and region.get("type") == "speech_bubble":
                    is_bubble = True
                    break
            
            translation, lang_ok = _ensure_target_language(
                ollama,
                _resolve_model(model),
                source_lang,
                target_lang,
                text,
                text_to_translation.get(text, ""),
                is_bubble=is_bubble,
            )
            if translation:
                translation_cache[text] = translation
            for region in regions:
                if region["region_id"] in region_ids:
                    region["translation"] = translation
                    if not lang_ok:
                        region["flags"]["needs_review"] = True
    
    # Update context window
    # Collect confident translations to add to context
    page_context = []
    for region in regions:
        if region.get("ignore"):
            continue
        original = region.get("ocr_text", "").strip()
        trans = region.get("translation", "").strip()
        if original and trans and not region.get("flags", {}).get("needs_review"):
             page_context.append(f"{original} -> {trans}")
    
    # Keep last 10 lines of context to avoid overflow
    if page_context:
        context_window.extend(page_context)
        # simplistic list slice
        while len(context_window) > 12:
            context_window.pop(0)
            
    return regions


def _resolve_model(model: str) -> str:
    if model == "auto-detect":
        models = list_models()
        if models:
            preferred = [
                "aya:35b",
                "huihui_ai/qwen3-abliterated:32b",
                "huihui_ai/qwen3-abliterated:14b",
                "qwen3-coder:30b",
                "dolphin3:8b",
            ]
            for name in preferred:
                if name in models:
                    return name
            return models[0]
        return "aya:35b"
    return model


def _recent_context_lines(context_window: list, max_lines: int = 6) -> list[str]:
    if not context_window:
        return []
    return [str(line).strip() for line in context_window[-max_lines:] if str(line).strip()]


def _iter_character_sources(entry: dict) -> Iterable[str]:
    if not isinstance(entry, dict):
        return []
    values = []
    for key in ("original", "canonical", "name"):
        value = str(entry.get(key, "")).strip()
        if value:
            values.append(value)
    for alias in entry.get("aliases", []) or []:
        if isinstance(alias, dict):
            value = str(alias.get("source", "")).strip()
        else:
            value = str(alias).strip()
        if value:
            values.append(value)
    return values


def _match_count(texts: list[str], term: str) -> int:
    if not term:
        return 0
    return sum(1 for text in texts if _contains_term(text, term))


def _build_page_style_guide(
    style_guide: dict,
    source_texts: Iterable[str],
    max_glossary: int = 24,
    max_characters: int = 10,
) -> dict:
    if not isinstance(style_guide, dict):
        return default_style_guide()

    texts = [str(text).strip() for text in source_texts if str(text).strip()]
    if not texts:
        return style_guide

    glossary = style_guide.get("glossary", []) or []
    characters = style_guide.get("characters", []) or []
    if len(glossary) <= max_glossary and len(characters) <= max_characters:
        return style_guide

    selected_glossary = list(glossary) if len(glossary) <= max_glossary else []
    glossary_candidates = []
    if len(glossary) > max_glossary:
        for item in glossary:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            if not source or not target:
                continue
            match_count = _match_count(texts, source)
            if match_count <= 0:
                continue
            priority = str(item.get("priority", "")).strip().lower()
            score = (1000 if priority == "hard" else 0) + (match_count * 100) + len(source)
            glossary_candidates.append((score, item))
        glossary_candidates.sort(key=lambda pair: pair[0], reverse=True)
        seen_sources = set()
        for _, item in glossary_candidates:
            source = str(item.get("source", "")).strip()
            if source and source not in seen_sources:
                selected_glossary.append(item)
                seen_sources.add(source)
            if len(selected_glossary) >= max_glossary:
                break

    selected_characters = list(characters) if len(characters) <= max_characters else []
    character_candidates = []
    if len(characters) > max_characters:
        for raw_entry in characters:
            entry = _normalize_character_entry(raw_entry)
            if not entry:
                continue
            score = 0
            for source in _iter_character_sources(entry):
                score += _match_count(texts, source) * 100
                score += len(source)
            if score <= 0:
                continue
            character_candidates.append((score, entry))
        character_candidates.sort(key=lambda pair: pair[0], reverse=True)
        for _, entry in character_candidates[:max_characters]:
            selected_characters.append(entry)

    filtered = dict(style_guide)
    filtered["glossary"] = selected_glossary
    filtered["characters"] = selected_characters
    return filtered


def _polygon_to_bbox(polygon: list) -> list:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _bbox_to_polygon(bbox: list) -> list:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _merge_detections(detections: list, image_size: tuple[int, int], merge: bool = True) -> list:
    if not detections:
        return []
    groups = []
    for polygon, conf in detections:
        try:
            bbox = _polygon_to_bbox(polygon)
        except Exception:
            continue
        groups.append({"bbox": bbox, "polygons": [polygon], "conf": float(conf or 0.0)})
    if not groups or not merge:
        return []
    changed = True
    while changed:
        changed = False
        result = []
        while groups:
            current = groups.pop(0)
            merged = False
            for i, other in enumerate(groups):
                if _should_merge(current["bbox"], other["bbox"], image_size):
                    current["bbox"] = _union_box(current["bbox"], other["bbox"])
                    current["polygons"].extend(other["polygons"])
                    current["conf"] = max(current["conf"], other["conf"])
                    groups.pop(i)
                    merged = True
                    changed = True
                    break
            result.append(current)
            if merged:
                groups = result + groups
                result = []
                break
        if not changed:
            groups = result
    return groups


def _sort_groups(groups: list) -> list:
    """Sort groups in manga reading order (Right-to-Left, Top-to-Bottom)."""
    # Simply sort by Y then -X? No, manga is columns. Vertical columns from right to left.
    # Actually, R-to-L is primary. Top-to-Bottom is secondary within column.
    # But often checking Y first then -X is better for "standard" text detection sorts.
    # Standard "Manga" order: 
    # 1. Top-Right quadrant
    # 2. Bottom-Right quadrant
    # ...
    # A simple robust heuristic: Sort by -RightX + Y*0.1? No.
    # Let's use: Top-to-Bottom as primary, Right-to-Left as secondary?
    # No, Manga is Right-to-Left *Pages*, but bubbles?
    # Usually: Top Right -> Bottom Right -> Top Left -> Bottom Left.
    # So we sort primarily by -CenterX, but we need to group vertical lines.
    # Let's try a simple sort: (Y // 100, -X). Rough banding.
    if not groups:
        return []
    
    def sort_key(g):
        bbox = g["bbox"]
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        # Use simple banding logic to handle slight misalignments
        return (int(cy / 300), -cx) 
        # This is very rough. 
        # Better: recursively partition? 
        # Let's stick to standard reading order logic: Vertical columns starting from right.
        # But 'ComicTextDetector' usually gives them unsorted.
        # A clearer sort:  - (Right Edge), then Top.
        # But top bubbles in right col come before bottom bubbles in right col.
        # So: Band by X?
    
    # Let's use a simpler heuristic common in OCR:
    # Sort by -X (Right to Left).
    # Then for items with similar X, sort by Y.
    # But if a bubble is far top-left vs near top-right...
    # Correct order: 1 (Top Right), 2 (Bottom Right), 3 (Top Left).
    # So Primary: -X (Right). Secondary: Y (Top).
    # But pure -X is bad because slight X difference overrides massive Y difference.
    # It should be: Sort by columns.
    
    # Revised Logic:
    # 1. Sort all by -X.
    # 2. Group into "Right", "Center", "Left" columns?
    # Too complex.
    
    # Let's assume standard R-L, T-B:
    # Just sort by -RightX is usually decent for columns.
    # Let's do: Sort by (sum of X+Y?) No.
    
    # Let's use the logic found in existing manga-ocr tools:
    # Sort by Y-coordinate first? No, that's English/Webtoon (Top to Bottom).
    # Manga is R-L. 
    # Actually, most sophisticated tools use a graph or precise column detection.
    # For now, let's implement a robust "Top-Right to Bottom-Left" sort:
    # Score = - (X + (ImageHeight - Y))?
    
    # Let's keep it simple and robust for now:
    # Sort by -RightX. (Rightmost first).
    # If X is within a threshold (e.g. 50px), consider them same column, then sort by Y.
    
    return sorted(groups, key=lambda g: (- (g["bbox"][0] + g["bbox"][2]), g["bbox"][1]))


def _dedupe_groups(groups: list, overlap_threshold: float = 0.85) -> list:
    if not groups:
        return []
    deduped = []
    for group in groups:
        bbox = group.get("bbox")
        if not bbox:
            continue
        if any(_overlap_ratio(bbox, existing.get("bbox", bbox)) >= overlap_threshold for existing in deduped):
            continue
        deduped.append(group)
    return deduped


def _load_image_for_crop(image_path: str):
    """Load an RGB image once for repeated region crops."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(image_path) as img:
            return img.convert("RGB")
    except Exception:
        return None


def _crop_image(image_path: str, bbox: list, expand_wide: bool = True, image_obj=None):
    """Crop image at bbox. Optionally expands wide regions to capture clipped text."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = image_obj if image_obj is not None else _load_image_for_crop(image_path)
        if img is None:
            return None
        img_w, img_h = img.size
        x, y, w, h = [int(v) for v in bbox]

        # Expand wide regions (likely impact text with clipped edges)
        # Detection often clips sides of stylized horizontal text
        if expand_wide and h > 0 and w > h * 2:
            # Expand by 15% of width on each side for wide text
            expand = int(w * 0.15)
            x = max(0, x - expand)
            # Recalculate width to reach original right edge + expansion
            x_right = min(img_w, int(bbox[0]) + int(bbox[2]) + expand)
            w = x_right - x

        return img.crop((x, y, x + w, y + h))
    except Exception:
        return None


def _merge_bboxes(bboxes: list, image_size: tuple[int, int]) -> list:
    if not bboxes:
        return []
    boxes = [_expand_box(b, 8, image_size) for b in bboxes]
    changed = True
    while changed:
        changed = False
        result = []
        while boxes:
            current = boxes.pop(0)
            merged = False
            for i, other in enumerate(boxes):
                if _should_merge(current, other, image_size):
                    current = _union_box(current, other)
                    boxes.pop(i)
                    merged = True
                    changed = True
                    break
            result.append(current)
            if merged:
                boxes = result + boxes
                result = []
                break
        if not changed:
            boxes = result
    return boxes


def _should_merge(a: list, b: list, image_size: tuple[int, int]) -> bool:
    if _boxes_overlap(a, b):
        return _overlap_ratio(a, b) >= 0.25
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    x_overlap = not (ax2 < bx or bx2 < ax)
    y_overlap = not (ay2 < by or by2 < ay)
    v_gap = min(abs(by - ay2), abs(ay - by2))
    h_gap = min(abs(bx - ax2), abs(ax - bx2))
    if x_overlap and v_gap <= max(6, min(ah, bh) * 0.25):
        return _union_area_ratio(a, b, image_size) <= 0.03
    if y_overlap and h_gap <= max(6, min(aw, bw) * 0.2):
        return _union_area_ratio(a, b, image_size) <= 0.03
    return False


def _boxes_overlap(a: list, b: list) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay)


def _union_box(a: list, b: list) -> list:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return [x0, y0, x1 - x0, y1 - y0]


def _expand_box(box: list, padding: int, image_size: tuple[int, int]) -> list:
    img_w, img_h = image_size
    x, y, w, h = box
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(img_w, x + w + padding) if img_w else x + w + padding
    y1 = min(img_h, y + h + padding) if img_h else y + h + padding
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _union_area_ratio(a: list, b: list, image_size: tuple[int, int]) -> float:
    img_w, img_h = image_size
    if img_w <= 0 or img_h <= 0:
        return 0.0
    area = img_w * img_h
    union = _union_box(a, b)
    return (union[2] * union[3]) / area


def _overlap_ratio(a: list, b: list) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    min_area = min(aw * ah, bw * bh)
    return inter / max(1, min_area)


def _clean_translation(text: str) -> str:
    cleaned = text.strip()
    lowered = cleaned.lower()
    if lowered.startswith("translation:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    if cleaned.startswith("文字："):
        cleaned = cleaned.split("：", 1)[1].strip()
    if cleaned.startswith("文本："):
        cleaned = cleaned.split("：", 1)[1].strip()
    if cleaned.startswith("原文："):
        cleaned = cleaned.split("：", 1)[1].strip()
    if cleaned.startswith("翻译："):
        cleaned = cleaned.split("：", 1)[1].strip()
    if cleaned.startswith("翻译："):
        cleaned = cleaned.split("：", 1)[1].strip()
    if cleaned.startswith("译文："):
        cleaned = cleaned.split("：", 1)[1].strip()
    if "translates to" in lowered:
        parts = cleaned.split("translates to", 1)
        cleaned = parts[1].strip() if len(parts) > 1 else cleaned
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    cleaned = re.sub(r"<[^>]*>", "", cleaned)
    cleaned = re.sub(r"<\s*e=\d+\s*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\be=\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"e=\d+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("□", "")
    cleaned = re.sub(r"(?:口|□){2,}", "", cleaned)
    if _placeholder_ratio(cleaned) >= 0.15:
        cleaned = cleaned.replace("口", "")
    if _placeholder_ratio(cleaned) >= 0.25:
        return ""
    lines = [line for line in cleaned.splitlines() if line.strip()]
    filtered = []
    strip_phrases = [
        "文本：",
        "文本:",
        "仅需翻译",
        "只需翻译",
        "只翻译",
        "不要任何标签",
        "不要任何引号",
        "不要任何解释",
        "不要任何说明",
        "不要任何注释",
        "不要任何多余",
        "不要标签",
        "不要引号",
        "不要解释",
        "不要说明",
        "不要注释",
        "不要多余",
        "只输出译文",
        "仅输出译文",
        "输出译文",
        "只输出翻译",
        "译文如下",
        "翻译如下",
        # Kana-related prompt phrases (from retry prompts)
        "重要：",
        "重要:",
        "你的回答中",
        "绝对不能包含",
        "不能包含",
        "日语假名",
        "ひらがな",
        "カタカナ",
        "只能使用",
        "纯中文",
        "汉字进行翻译",
        "进行翻译",
        "将下面的日语翻译成",
        "翻译成简体中文",
        "只输出简体中文",
        "不要片假名",
        "不要平假名",
        "罗马音或英文",
        "日语原文",
        "请将日语",
    ]
    for line in lines:
        head = line.strip()
        lower = head.lower()
        if (
            lower.startswith("text:")
            or lower.startswith("文本:")
            or lower.startswith("文本：")
            or lower.startswith("context:")
            or lower.startswith("input:")
            or lower.startswith("重要：")
            or lower.startswith("重要:")
            or "return only the translation" in lower
            or "output only the translation" in lower
            or "no labels" in lower
            or "no quotes" in lower
            or "no explanations" in lower
            or "<<text>>" in lower
            or "<</text>>" in lower
            # Chinese/Japanese prompt leak patterns
            or "ひらがな" in head
            or "カタカナ" in head
            or "日语假名" in head
            or "绝对不能包含" in head
            or "纯中文汉字" in head
            or "只能使用" in head
            or "进行翻译" in head
            or "翻译成简体中文" in head
            or "将下面的日语翻译成" in head
            or "请将日语" in head
            or "日语原文" in head
        ):
            continue
        head = head.replace("文本：", "").replace("文本:", "")
        if _is_punct_only(head):
            continue
        for phrase in strip_phrases:
            head = head.replace(phrase, "")
        if not head.strip():
            continue
        filtered.append(head)
    cleaned = "\n".join(filtered).strip()
    if cleaned.startswith("\"") and cleaned.endswith("\""):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("`") and cleaned.endswith("`"):
        cleaned = cleaned[1:-1].strip()
    if "Return only the translation" in cleaned:
        cleaned = cleaned.split("Return only the translation", 1)[0].strip()
    cleaned = cleaned.strip("<> ")
    return cleaned


def _sanitize_glossary_target(target: str, source: str, target_lang: str) -> str:
    if not target:
        return ""
    cleaned = _clean_translation(target)
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[0].strip()
    cleaned = cleaned.strip().strip("“”\"' ").rstrip("。.，,")
    if not cleaned:
        return ""
    leak_markers = [
        "回复格式",
        "回復格式",
        "回复格式：",
        "回復格式：",
        "不要标点",
        "不要標點",
        "只输出",
        "只輸出",
        "只输出译文",
        "只輸出譯文",
    ]
    if _looks_like_prompt_leak(cleaned) or any(m in cleaned for m in leak_markers):
        return ""
    if target_lang in ["Simplified Chinese", "Traditional Chinese"]:
        if not _language_ok(target_lang, cleaned):
            return ""
        if _is_cjk_term(source) and _is_cjk_term(cleaned):
            digit_chars = set("0123456789０１２３４５６７８９一二三四五六七八九十百千万亿兩两")
            if not any(ch in digit_chars for ch in source) and any(ch in digit_chars for ch in cleaned):
                return ""
            extra_len = len(cleaned) - len(source)
            if len(source) <= 3 and extra_len >= 3:
                expansion_markers = (
                    "这里",
                    "那边",
                    "这个",
                    "那个",
                    "这些",
                    "那些",
                    "二楼",
                    "一楼",
                    "三楼",
                    "四楼",
                    "楼",
                    "习惯",
                    "地方",
                    "浴场",
                    "学园",
                    "学生",
                    "少女",
                    "休息",
                    "休息场",
                    "场",
                    "家",
                )
                if any(marker in cleaned for marker in expansion_markers):
                    return ""
    return cleaned



def _enforce_glossary(
    translation: str,
    source_text: str,
    style_guide: dict,
) -> str:
    """
    Post-process translation to enforce glossary term consistency.
    
    If source text contains a glossary source term, ensure the translation
    uses the correct target term. This fixes LLM inconsistency issues.
    
    Args:
        translation: The LLM translation output
        source_text: The original Japanese text
        style_guide: The style guide containing glossary entries
        
    Returns:
        Translation with glossary terms enforced
    """
    if not translation or not source_text:
        return translation
    
    glossary = style_guide.get("glossary", [])
    if not glossary:
        return translation
    
    # Build a mapping of source -> target for terms found in source text
    terms_to_enforce = []
    
    for item in glossary:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        
        if not source or not target:
            continue
        
        # Check if this glossary term appears in the source text
        if source in source_text:
            terms_to_enforce.append((source, target))
    
    if not terms_to_enforce:
        return translation
    
    # For each term that should be in the translation, check and fix
    result = translation
    
    for _, correct_target in terms_to_enforce:
        # Skip if target is already present
        if correct_target in result:
            continue
        
        # Calculate expected length of the name translation (in characters)
        target_len = len(correct_target)
        
        # For kana-based names (like まゆ), the model might have produced
        # a different Chinese transliteration (like 真由 instead of 麻由)
        # Look for Chinese character sequences of similar length to replace
        
        # Find all Chinese character sequences in the result
        # We look for sequences of length target_len
        chinese_sequences = set(re.findall(r'[\u4e00-\u9fff]{' + str(target_len) + '}', result))
        
        for seq in chinese_sequences:
            if seq == correct_target:
                continue
            
            # Check context to see if this sequence looks like a name
            # We use regex to ensure we only replace instances that look like names
            name_patterns = [
                (r'(' + re.escape(seq) + r')([酱桑君小姐先生老师])', 1),   # Name + honorific
                (r'(' + re.escape(seq) + r')((的|吗|呢|啊|吧|呀|哦|哇))', 1), # Name + particle
                (r'((是|叫|找|给|对|跟|和|与|爱|恨))(' + re.escape(seq) + r')', 3), # Verb + name
                (r'^(' + re.escape(seq) + r')($|[，。！？])', 1), # Start/End or standalone
                (r'([，。！？])(' + re.escape(seq) + r')([，。！？]|$)', 2), # Surrounded by punct
            ]
            
            replaced = False
            for pattern, group_idx in name_patterns:
                # If pattern matches, replace ONLY that instance
                if re.search(pattern, result):
                    # We found a context match. Now safely replace.
                    # Note: simple replace() is still risky regarding multiple occurrences of same word used differently
                    # But if we found "seq小姐", it's likely a name. 
                    # We'll replace all occurrences if we find strong evidence it's a name anywhere.
                    # This is a compromise.
                    result = result.replace(seq, correct_target)
                    replaced = True
                    break
            
            if replaced:
                pass 
                
    return result


import threading
import json
import re
from app.io.style_guide import save_style_guide

_glossary_lock = threading.Lock()


def _extract_names_heuristic(texts: list[str]) -> list[str]:
    """
    DEPRECATED: Old heuristic extraction, kept as fallback if MeCab unavailable.
    Looks for repeated katakana sequences (common for character names in manga).
    """
    from collections import Counter
    
    # Katakana pattern (2+ chars, common for names)
    katakana_pattern = re.compile(r'[\u30A0-\u30FF]{2,}')
    
    all_katakana = []
    for text in texts:
        matches = katakana_pattern.findall(text)
        all_katakana.extend(matches)
    
    # Count occurrences - names appear multiple times
    counts = Counter(all_katakana)
    
    # Filter: names should appear at least 2 times and be 2-8 chars (typical name length)
    potential_names = [
        name for name, count in counts.items()
        if count >= 2 and 2 <= len(name) <= 8
    ]
    
    # Also look for common suffixes that indicate names
    name_suffixes = ['さん', 'ちゃん', 'くん', '君', '様', '先生', '先輩', '殿']
    for text in texts:
        for suffix in name_suffixes:
            # Pattern: word + suffix
            pattern = re.compile(rf'([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]{{1,6}}){re.escape(suffix)}')
            matches = pattern.findall(text)
            potential_names.extend(matches)
    
    # Filter common stopwords
    blacklist = {
        "学校", "先生", "同級生", "委員長", "部長", "会長", "社長", "校長",
        "今日", "明日", "昨日", "今年", "来年", "先輩", "後輩", "毎日", "毎朝",
        "日本", "東京", "大阪", "中国", "全国", "本当", "本当に", "嘘",
        "時間", "場所", "気持ち", "問題", "事情", "理由", "意味",
        "能力", "危険", "危機", "戦争", "世界", "宇宙", "地球", "人間",
        "私", "僕", "俺", "自分", "貴様", "お前", "あなた", "アンタ", "君", "我",
        "彼", "彼女", "あいつ", "こいつ", "そいつ", "誰", "何", "何処",
        "男", "女", "人", "奴", "子供", "大人", "生徒", "教師", "医者", "刑事",
        "教室", "部屋", "家", "町", "都市", "国", "王", "城", "村",
    }
    
    # Deduplicate and filter
    unique_names = set(potential_names)
    return [n for n in unique_names if n not in blacklist]


def _extract_kanji_name_heuristic(text: str) -> list[str]:
    """Fallback: extract likely Kanji names from honorifics and repetition."""
    if not text:
        return []
    from collections import Counter

    honorifics = ["さん", "くん", "ちゃん", "様", "先生", "先輩", "殿", "君", "氏"]
    honorific_pattern = re.compile(
        rf"([\u4E00-\u9FFF]{{2,6}})(?:{'|'.join(honorifics)})"
    )
    names = set(m.group(1) for m in honorific_pattern.finditer(text))

    # Repetition fallback (3+ Kanji, appears 3+ times)
    pattern = re.compile(r"[\u4E00-\u9FFF]{3,6}")
    matches = pattern.findall(text)
    counts = Counter(matches)
    blacklist = {
        "学校", "先生", "同級生", "委員長", "部長", "会長", "社長", "校長",
        "今日", "明日", "昨日", "今年", "来年", "先輩", "後輩", "毎日", "毎朝",
        "日本", "東京", "大阪", "中国", "全国", "本当", "本当に", "嘘",
        "時間", "場所", "気持ち", "問題", "事情", "理由", "意味",
        "能力", "危険", "危機", "戦争", "世界", "宇宙", "地球", "人間",
        "私", "僕", "俺", "自分", "貴様", "お前", "あなた", "アンタ", "君", "我",
        "彼", "彼女", "あいつ", "こいつ", "そいつ", "誰", "何", "何処",
        "男", "女", "人", "奴", "子供", "大人", "生徒", "教師", "医者", "刑事",
        "教室", "部屋", "家", "町", "都市", "国", "王", "城", "村",
    }
    for name, count in counts.items():
        if count >= 3 and name not in blacklist:
            names.add(name)
    return list(names)


def _translate_name(ollama, model: str, name: str, target_lang: str) -> str:
    """Translate a proper noun using a simple, focused prompt."""
    if target_lang == "Simplified Chinese":
        prompt = f"把日语人名'{name}'翻译成中文。\n回复格式：只输出翻译后的名字，不要标点、不要解释。"
    elif target_lang == "Traditional Chinese":
        prompt = f"把日語人名'{name}'翻譯成繁體中文。\n回復格式：只輸出翻譯後的名字，不要標點、不要解釋。"
    else:
        prompt = f"Translate the Japanese name '{name}' to {target_lang}.\nFormat: Output ONLY the translated name, nothing else."
    
    try:
        result = ollama.generate(model, prompt, timeout=30, options={"num_predict": 30, "temperature": 0.1})
        if result:
            cleaned = _sanitize_glossary_target(result.strip(), name, target_lang)
            if cleaned:
                return cleaned
    except Exception:
        pass
    return ""


def _translate_alias(ollama, model: str, alias: str, hint: str, base_trans: str, target_lang: str) -> str:
    """
    Translate an alias with pattern context.
    The 'hint' comes from MeCab suffix detection (e.g., "亲昵的称呼" for -chan).
    """
    if target_lang == "Simplified Chinese":
        if hint:
            # For names with suffixes like -chan, -san
            prompt = f"'{alias}'是'{base_trans}'的{hint}。把'{alias}'翻译成中文名。\n回复格式：只输出翻译后的名字，不要其他内容。"
        else:
            # For plain aliases
            prompt = f"'{alias}'是人名'{base_trans}'的简称或别称。把'{alias}'翻译成中文。\n回复格式：只输出翻译后的名字，不要标点、不要解释。"
    elif target_lang == "Traditional Chinese":
        if hint:
            prompt = f"'{alias}'是'{base_trans}'的{hint}。把'{alias}'翻譯成繁體中文名。\n回復格式：只輸出翻譯後的名字，不要其他內容。"
        else:
            prompt = f"'{alias}'是人名'{base_trans}'的簡稱或別稱。把'{alias}'翻譯成繁體中文。\n回復格式：只輸出翻譯後的名字，不要標點、不要解釋。"
    else:
        prompt = f"'{alias}' is a nickname for '{base_trans}'. Translate '{alias}' to {target_lang}.\nFormat: Output ONLY the translated name, nothing else."
    
    try:
        result = ollama.generate(model, prompt, timeout=30, options={"num_predict": 30, "temperature": 0.1})
        if result:
            cleaned = _sanitize_glossary_target(result.strip(), alias, target_lang)
            if cleaned:
                return cleaned
    except Exception:
        pass
    return ""

def _parse_json_list(text: str) -> list:
    """Robustly parse a JSON list from LLM output."""
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except:
        pass
    
    # Try finding list pattern
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except:
            pass
    return []

def _is_garbage(text: str) -> bool:
    """Check if text is likely OCR noise."""
    if not text or len(text.strip()) < 2:
        return True
    # Check if all symbols/numbers (no letters/cjk)
    # Using a simple heuristic: must have at least one CJK or letter
    if not re.search(r"[a-zA-Z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
        return True
    return False

def _accumulate_text(state: dict, text: str):
    """Accumulate text for batched analysis."""
    if not text or _is_garbage(text):
        return
    with _glossary_lock:
        buffer = state.setdefault("buffer", [])
        buffer.append(text)
        if len(buffer) > 300:
            buffer.pop(0)
        


def _trigger_discovery_if_needed(
    state: dict,
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    base_style: dict,
    style_guide_path: str,
    allow_ollama: bool = False,
    discovery_model: str | None = None,
    settings: PipelineSettings | None = None,
):
    """Check buffer size and trigger background discovery if threshold met."""
    import tempfile
    log_path = os.path.join(tempfile.gettempdir(), "auto_glossary_debug.log")
    
    if not state:
        return
        
    # User choice: If not allowed to use Ollama for discovery, specifically check if we are using GGUF
    # If users use Ollama for translation, 'ollama' object is valid.
    # If users use GGUF, 'ollama' passed here might be None or a dummy?
    # Actually _process_page logic: if GGUF, ollama might be None.
    
    # If allow_ollama is False, and we are not using Ollama for translation (model is gguf?), skip.
    # But wait, if we are using Ollama for translation, then 'ollama' is valid and we SHOULD use it?
    # User said: "users can decide whether to use Ollama for our Auto-Glossary system"
    # This implies a global switch.
    
    if not allow_ollama and (not ollama or not hasattr(ollama, 'generate')):
         # Only allow if we are ALREADY using ollama for translation?
         # Or stricter: if use_ollama_discovery is False, NEVER do background discovery?
         # Let's assume the latter for safety/conflict avoidance.
         return

    # If we don't have an ollama client at all, we can't do it anyway
    if not ollama:
        return

    # Strategy for Hybrid Discovery:
    # 1. If we are already using Ollama (has list_models), use it.
    # 2. If allow_ollama is True: Instantiate a temporary OllamaClient.
    # 3. Else: Fall back to MeCab-only mode (using GGUF or whatever available for simple translation).
    
    # Logic for Deep Scan Client Resolution
    discovery_client = ollama
    is_real_ollama = hasattr(ollama, "list_models")
    use_deep_scan = False
    
    # Resolve backend preference.
    # MeCab-only mode must never invoke LLM discovery.
    backend = getattr(settings, "discovery_backend", "Ollama") if settings else "Ollama"
    if not allow_ollama:
        backend = "MeCab"

    # 1. GGUF Backend (LLM discovery path)
    if allow_ollama and (backend == "GGUF" or (discovery_model and ".gguf" in discovery_model.lower())):
        target_path = str(discovery_model or "").strip()
        translation_path = getattr(settings, "gguf_model_path", "") if settings else ""
        needs_swap = (
            settings
            and settings.translator_backend == "GGUF"
            and target_path
            and translation_path
            and os.path.abspath(target_path) != os.path.abspath(translation_path)
        )
        # Reuse existing GGUF client if it matches the target model (avoids double-load)
        if hasattr(ollama, "_model_path"):
            existing_path = getattr(ollama, "_model_path", "")
            if not target_path or (existing_path and os.path.abspath(target_path) == os.path.abspath(existing_path)):
                discovery_client = ollama
                use_deep_scan = True
                needs_swap = False
                logger.info("Deep Scan: Reusing current GGUF client for discovery.")

        if not use_deep_scan and target_path and os.path.isfile(target_path):
            try:
                from app.translate.gguf_client import GGUFClient
                if needs_swap and hasattr(ollama, "close"):
                    logger.info("Deep Scan: Swapping GGUF models to avoid dual load.")
                    ollama.close()
                logger.info(f"Deep Scan: Loading specialized GGUF model: {target_path}")
                n_gpu_layers = settings.gguf_n_gpu_layers if settings else 0
                discovery_client = GGUFClient(
                    model_path=target_path,
                    prompt_style="extract",
                    n_ctx=2048,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=max(1, settings.gguf_n_threads) if settings else 4,
                    n_batch=min(128, settings.gguf_n_batch) if settings else 64,
                )
                use_deep_scan = True
                logger.info("Deep Scan: GGUF enabled via Backend Selection.")
            except Exception as e:
                logger.error(f"Failed to load Deep Scan GGUF model: {e}")
                return
        elif not use_deep_scan:
            logger.warning("Deep Scan: GGUF backend selected but invalid path string.")

    # 2. Ollama Backend
    elif allow_ollama and backend == "Ollama":
        # If user explicitly wants Deep Scan via Ollama (allowed)
        if (discovery_model and discovery_model.lower() not in ["auto-detect", "none", ""]) or allow_ollama:
            if is_real_ollama:
                use_deep_scan = True
            else:
                try:
                    from app.translate.ollama_client import OllamaClient
                    new_client = OllamaClient()
                    if new_client.is_available():
                        discovery_client = new_client
                        use_deep_scan = True
                except Exception:
                    pass
            
    # Check buffer length
    with _glossary_lock:
        buffer = state.get("buffer", [])
        total_len = sum(len(s) for s in buffer)
        is_running = state.get("is_running", False)
    
    logger.debug(f"TRIGGER CHECK: total_len={total_len}, is_running={is_running}, deep_scan={use_deep_scan}")
    
    # Threshold: ~6000 chars to reduce LLM invocations and memory churn
    if total_len >= 6000 and not is_running:
        logger.info(f"TRIGGER: Starting discovery thread! (Deep Scan: {use_deep_scan})")
        
        with _glossary_lock:
            state["is_running"] = True
            state["had_live_discovery"] = True
            
        # Choose the correct worker function
        target_func = _run_sakura_discovery if use_deep_scan else _run_discovery
        
        if use_deep_scan:
            # Synchronous: Pause pipeline to prevent VRAM thrashing with LLM
            logger.info(f"STARTING DISCOVERY SYNCHRONOUSLY (Deep Scan Safe Mode)")
            try:
                 target_func(discovery_client, model, source_lang, target_lang, state, base_style, style_guide_path, discovery_model)
            except Exception as e:
                 logger.error(f"Discovery crashed: {e}")
            if discovery_client is not ollama and hasattr(discovery_client, "close"):
                 discovery_client.close()
            if settings and settings.translator_backend == "GGUF" and hasattr(ollama, "_model_path"):
                 target_path = str(getattr(settings, "gguf_model_path", "")).strip()
                 if target_path and os.path.isfile(target_path):
                     try:
                         from app.translate.gguf_client import GGUFClient
                         n_gpu_layers = settings.gguf_n_gpu_layers
                         state["translation_client"] = GGUFClient(
                             model_path=target_path,
                             prompt_style=settings.gguf_prompt_style,
                             n_ctx=settings.gguf_n_ctx,
                             n_gpu_layers=n_gpu_layers,
                             n_threads=settings.gguf_n_threads,
                             n_batch=settings.gguf_n_batch,
                         )
                         logger.info("Deep Scan: Reloaded translation GGUF client after swap.")
                     except Exception as e:
                         logger.error(f"Deep Scan: Failed to reload translation GGUF model: {e}")
            with _glossary_lock:
                 state["is_running"] = False
        else:
            # Asynchronous: Run MeCab in background (CPU only, safe for concurrency)
            logger.info(f"STARTING DISCOVERY IN BACKGROUND (MeCab Mode)")
            t = threading.Thread(
                target=target_func,
                args=(
                    discovery_client,
                    model,
                    source_lang,
                    target_lang,
                    state,
                    base_style,
                    style_guide_path,
                    bool(discovery_client and hasattr(discovery_client, "generate")),
                )
            )
            t.daemon = True
            t.start()


def _run_sakura_discovery(
    ollama,
    main_model: str,  # The model currently used by the main translation pipeline
    source_lang: str,
    target_lang: str,
    state: dict,
    base_style: dict,
    style_guide_path: str,
    target_model: str | None = None, # User-selected discovery model (None = Auto)
):
    """
    Background worker for Advanced Auto-Glossary discovery.
    """
    accumulated_text = list(state.get("buffer", []))
    if not accumulated_text:
        return
        
    with _glossary_lock:
         state["buffer"] = []

    # 1. Resolve Best Model for Extraction
    extraction_model = None
    
    extraction_model = None
    is_gguf_client = hasattr(ollama, "is_available") # Duck typing check for GGUFClient
    
    # Debug logging for model resolution
    available_models: list[str] = []
    if not is_gguf_client:
        try:
            available_models = list_models()
            logger.debug(f"Available Models: {available_models}")
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
    else:
        # GGUF Client doesn't list models, it HAS a model
        # The 'extraction_model' string is ignored by GGUF generate() usually, but strictly speaking
        # we treat the client as the model.
        extraction_model = "gguf_model" 
        logger.info("Deep Scan: Using GGUF Client.")

    try:
        main_model = str(main_model or "")
    # Check if main_model is a valid Ollama model (not a path)
        is_gguf_path = (
            os.path.sep in main_model
            or "/" in main_model
            or "\\" in main_model
            or main_model.lower().endswith(".gguf")
            or os.path.isfile(main_model)
        )
        if not is_gguf_path and "sakura" not in main_model.lower():
             extraction_model = main_model
             
        # Priority 1: Manual Override
        if target_model and target_model.lower() != "auto-detect" and "sakura" not in target_model.lower():
             extraction_model = target_model
             
        # Priority 2: Use Main Model if it's in Ollama list
        elif extraction_model and extraction_model in available_models:
             pass # extraction_model is already set to main_model
             
        # Priority 3: Smart Selection from Available
        elif not extraction_model:
            qwen_candidates = [m for m in available_models if "qwen" in m.lower() and "sakura" not in m.lower()]
            non_sakura_candidates = [m for m in available_models if "sakura" not in m.lower()]
            
            if qwen_candidates:
                extraction_model = qwen_candidates[0]
            elif non_sakura_candidates:
                extraction_model = non_sakura_candidates[0]
                
    except Exception:
        pass
        
    if extraction_model and "sakura" in extraction_model.lower() and not is_gguf_client:
        logger.warning("Deep Scan: Sakura is translation-only; skipping Deep Scan.")
        return

    # FORCE FALLBACK (Only for Ollama)
    if not extraction_model and not is_gguf_client:
        extraction_model = "huihui_ai/qwen3-abliterated:14b"
        logger.warning(f"No model matched. Forcing default '{extraction_model}'")

    # Final check
    if not extraction_model and not is_gguf_client:
         pass

    # Join text into chunks
    full_text = "\n".join(accumulated_text)
    # Join text into chunks
    full_text = "\n".join(accumulated_text)
    chunk_size = 800 # Reduced from 1500 to prevent timeouts
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    
    logger.info(f"Starting Discovery on {len(chunks)} chunks using {extraction_model}...")
    
    for i, chunk in enumerate(chunks):
        glossary_map = {}
        # Build prompt - simple line based is safer for weaker models
        # Build prompt using the shared robust prompt builder
        # This ensures we get JSON output and "Canonical" fields for nickname support
        prompt = build_entity_extraction_prompt(
            text_block=chunk,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        # Override for extracting model if it's very dumb (optional, but Qwen 14b is smart enough)
        # If extraction_model is explicitly "sakura", maybe fallback? 
        # But we assume Qwen/Smart model is used for Deep Scan as per design.
        
        # If using Qwen, we can try JSON for better structure, but line-based is universally robust.
        # Let's stick to line-based to be safe for all models including Sakura.
        
        try:
            # Increase timeout to 600s (10min) for very slow GPUs
            # Reduce num_predict to 1024 to save time
            result = ollama.generate(extraction_model, prompt, timeout=600, options={"num_predict": 1024, "temperature": 0.1})
            if not result:
                continue
            
            if not result:
                continue
            
            logger.debug(f"Chunk {i+1} Output:\n{result}\n---")

            # Parse JSON output
            # We use the robust parser from controller (already defined) or local logic
            current_extracted = _parse_json_list(result)
            
            # Post-process: Resolve Canonical Names (Nicknames -> Full Name Translation)
            # 1. First pass: Collect all "Canonical" -> "Translation" mappings
            #    e.g. Canonical: "Mayuzumi" -> Translation: "Xiao Dai"
            canonical_map = {}
            for item in current_extracted:
                if not isinstance(item, dict): continue
                canon = item.get("canonical", "").strip()
                raw_trans = item.get("translation", "").strip() or item.get("target", "").strip()
                source = item.get("text", "").strip() or item.get("source", "").strip()
                trans = _sanitize_glossary_target(raw_trans, canon or source, target_lang)
                
                # If this item IS the canonical form (source == canonical), save its translation
                if canon and trans and source == canon:
                    canonical_map[canon] = trans
            
            # 2. Second pass: Build the Glossary Map
            for item in current_extracted:
                if not isinstance(item, dict): continue
                
                source = item.get("text", "").strip() or item.get("source", "").strip()
                # Try finding translation in 'target' or 'translation' keys (prompts vary slightly)
                translation = item.get("translation", "").strip() or item.get("target", "").strip()
                type_ = item.get("type", "proper_noun")
                canon = item.get("canonical", "").strip()
                
                if not source or len(source) < 2:
                    continue
                if source in ["...", "、", "。"]: 
                    continue
                    
                # MAGIC: Canonical Name Logic
                # If we have a canonical name (e.g. Mayuzumi -> Xiao Dai)
                # And the current term is a nickname (e.g. Mayu-Mayu), 
                # resolving it is tricky.
                
                # Case 1: If the LLM was lazy and just copied the source (Target="Mayu-Mayu"), 
                # we SHOULD overwrite with canonical (Target="Xiao Dai") to be safe.
                # Case 2: If the LLM gave a specific variation (Target="Xiao Dai Dai"),
                # we should PRESERVE it.
                
                if canon and canon in canonical_map:
                    # Only overwrite if current translation is likely trash (same as source)
                    # or if it's completely empty.
                    is_lazy = (translation == source) or (not translation)
                    if is_lazy:
                        translation = canonical_map[canon]
                translation = _sanitize_glossary_target(translation, source, target_lang)
                
                if source and translation:
                    glossary_map[source] = {
                        "target": translation,
                        "type": type_,
                        "info": item.get("info", "") or f"Canon: {canon}" if canon else ""
                    }
                            
            # Update global glossary securely
            if glossary_map:
                # Re-load style guide inside lock to prevent race conditions with PipelineWorker
                with _glossary_lock:
                    # Update in-memory state for other components
                    state_map = state.setdefault("map", {})
                    state_map.update(glossary_map)
                    
                    # Update file on disk
                    try:
                        current_sg = _load_style_guide(style_guide_path)
                        # We pass None for characters because we only extracted glossary terms here
                        # (Actually we extracted Names as glossary terms, so putting them in glossary map is fine for now)
                        updated_sg = _merge_glossary(current_sg, glossary_map, None)
                        updated_sg = _sanitize_style_guide(updated_sg, target_lang)
                        save_style_guide(style_guide_path, updated_sg)
                    except Exception as io_err:
                        logger.error(f"Glossary Save Error: {str(io_err)}")
                    
        except Exception as e:
            logger.error(f"LLM Error in chunk {i+1}: {str(e)}")


def _run_discovery(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    state: dict,
    base_style: dict,
    style_guide_path: str,
    translate_entities: bool = False,
):
    """
    Background worker for Auto-Glossary discovery using MeCab.
    
    This function:
    1. Extracts proper nouns using MeCab (Japanese NLP)
    2. Groups names into canonical + aliases by reading matching
    3. Translates each name using focused prompts
    4. Saves results to style_guide.json
    """
    with _glossary_lock:
        buffer = list(state.get("buffer", []))
        state["buffer"] = []
    
    if not buffer:
        with _glossary_lock:
            state["is_running"] = False
        return

    full_text = "\n".join(buffer)
    
    # Debug log file (optional)
    log_path = None
    if _GLOSSARY_DEBUG:
        import tempfile
        log_path = os.path.join(tempfile.gettempdir(), "auto_glossary_debug.log")
    
    try:
        # Determine model to use
        resolved_model = _resolve_model(model)
        
        logger.info(f"--- MECAB DISCOVERY ---\nBuffer size: {len(full_text)} chars\nModel: {resolved_model}")
        
        # Try MeCab-based extraction first
        try:
            from app.nlp.mecab_extractor import MeCabExtractor, ExtractedName
            
            # Load user suffix config if available
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "suffixes.json")
            extractor = MeCabExtractor(config_path=config_path)
            
            if extractor.is_available:
                # Extract proper nouns
                names = extractor.extract_proper_nouns(full_text)
                
                if log_path:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"MeCab extracted {len(names)} proper nouns\n")
                        for name in names[:10]:
                            f.write(f"  - {name.surface} (reading: {name.reading}, pos: {name.pos})\n")
                
                # Fallback: add Kanji name-like chunks if MeCab misses full names
                if source_lang == "Japanese":
                    extra_names = _extract_kanji_name_heuristic(full_text)
                    if extra_names:
                        existing = {n.surface for n in names}
                        for surface in extra_names:
                            if surface not in existing:
                                names.append(ExtractedName(surface=surface, reading=surface, pos="固有名詞"))

                # Group into canonical + aliases
                groups = extractor.group_aliases(names)
                # Translate each group
                for group in groups:
                    # Translate canonical name first
                    canonical_trans = ""
                    if translate_entities:
                        canonical_trans = _translate_name(ollama, resolved_model, group.canonical, target_lang)
                        if not canonical_trans:
                            continue
                        
                    with _glossary_lock:
                        glossary_map = state.setdefault("map", {})
                        characters_list = state.setdefault("characters", [])
                        
                        if canonical_trans:
                            glossary_map[group.canonical] = {
                                "target": canonical_trans,
                                "reading": group.canonical_reading,
                                "pattern": "canonical",
                                "type": "proper_noun"
                            }
                        
                        # Track this as a character
                        char_entry = {
                            "name": canonical_trans or group.canonical,
                            "translation": canonical_trans,
                            "original": group.canonical,
                            "reading": group.canonical_reading,
                            "gender": "unknown",
                            "aliases": []
                        }
                    
                    # Translate each alias
                    for alias in group.aliases:
                        alias_source = alias["source"]
                        alias_hint = alias.get("hint", "")
                        
                        alias_trans = ""
                        if translate_entities:
                            alias_trans = _translate_alias(
                                ollama, resolved_model,
                                alias_source, alias_hint,
                                canonical_trans, target_lang
                            )
                            if not alias_trans:
                                continue
                        
                        with _glossary_lock:
                            if alias_trans:
                                glossary_map[alias_source] = {
                                    "target": alias_trans,
                                    "reading": alias.get("reading", ""),
                                    "pattern": alias.get("pattern", ""),
                                    "hint": alias.get("hint", ""),
                                    "type": "proper_noun"
                                }
                            
                            # Store full alias object with translation
                            alias_obj = dict(alias)
                            alias_obj["target"] = alias_trans
                            char_entry["aliases"].append(alias_obj)
                    
                    # Add character entry
                    with _glossary_lock:
                        # Check if character already exists
                        found = False
                        for existing in characters_list:
                            if existing.get("original") == group.canonical or existing.get("name") == canonical_trans:
                                # Merge aliases
                                existing_aliases = existing.setdefault("aliases", [])
                                for a in char_entry["aliases"]:
                                    if a not in existing_aliases:
                                        existing_aliases.append(a)
                                found = True
                                break
                        
                        if not found:
                            characters_list.append(char_entry)
                
                # Also translate standalone names (not in groups)
                with _glossary_lock:
                    glossary_map = state.setdefault("map", {})
                
                for name in names:
                    # Skip if already in glossary
                    if name.surface in glossary_map:
                        continue
                    if translate_entities:
                        trans = _translate_name(ollama, resolved_model, name.surface, target_lang)
                        if not trans:
                            continue
                        with _glossary_lock:
                            glossary_map[name.surface] = {
                                "target": trans,
                                "reading": name.reading,
                                "pattern": "standalone",
                                "type": "proper_noun"
                            }
            else:
                pass
                
        except ImportError:
            # Fallback to old heuristic method
            if translate_entities:
                heuristic_names = _extract_names_heuristic(buffer)
                for name in heuristic_names:
                    trans = _translate_name(ollama, resolved_model, name, target_lang)
                    if not trans:
                        continue
                    with _glossary_lock:
                        glossary_map = state.setdefault("map", {})
                        if name not in glossary_map:
                            glossary_map[name] = trans
        
        # Save to disk
        with _glossary_lock:
            chars = list(state.get("characters", []))
            g_map = dict(state.get("map", {}))
        
        merged_for_save = _merge_glossary(base_style, g_map, chars)
        merged_for_save = _sanitize_style_guide(merged_for_save, target_lang)
        if style_guide_path:
            try:
                save_style_guide(style_guide_path, merged_for_save)
            except Exception as e:
                print(f"Failed to save auto-glossary: {e}")

    except Exception as e:
        print(f"Discovery failed: {e}")
    finally:
        with _glossary_lock:
            state["is_running"] = False


def _apply_auto_glossary(
    base_style: dict,
    state: dict,
    texts: list[str],
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    style_guide_path: str = "",
    allow_ollama: bool = False,
    discovery_model: str | None = None,
    settings: PipelineSettings | None = None,
    mecab_only: bool = True,
) -> dict:
    # 1. Accumulate texts
    if texts:
        for t in texts:
             _accumulate_text(state, t)

    # 2. Trigger discovery
    if mecab_only:
        allow_ollama = False
        discovery_model = None
        settings = None
    _trigger_discovery_if_needed(
        state,
        ollama,
        model,
        source_lang,
        target_lang,
        base_style,
        style_guide_path,
        allow_ollama,
        discovery_model=discovery_model,
        settings=settings,
    )
    
    # 3. Read current state to merge
    with _glossary_lock:
         chars = list(state.get("characters", []))
         g_map = dict(state.get("map", {}))
         
    return _merge_glossary(base_style, g_map, chars)


def _batch_translate(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    style_guide: dict,
    items: list,
    context_lines: list[str] | None = None,
    settings: PipelineSettings | None = None,
) -> dict:
    resolved = _resolve_model(model)
    translations: dict = {}
    
    # Defaults
    temp = 0.2
    top_p = 0.9
    
    if settings:
        if settings.translator_backend == "GGUF":
             temp = settings.gguf_temperature
             top_p = settings.gguf_top_p
        else:
             temp = settings.ollama_temperature
             top_p = settings.ollama_top_p
             
    batch_size = 16
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        prompt = build_batch_translation_prompt(
            source_lang,
            target_lang,
            style_guide,
            chunk,
            context_lines=context_lines,
        )
        token_limit = _estimate_num_predict(chunk)
        try:
            raw = ollama.generate(
                resolved,
                prompt,
                timeout=600,
                options={"num_predict": token_limit, "temperature": temp, "top_p": top_p},
            )
        except Exception:
            return {}
        parsed = _parse_json_list(raw)
        if not isinstance(parsed, list):
            return {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            region_id = str(item.get("id", "")).strip()
            translation = _clean_translation(str(item.get("translation", "")).strip())
            if region_id:
                translations[region_id] = translation
    return translations


def _estimate_num_predict(items: list) -> int:
    if not items:
        return 128
    lengths = [len(str(item.get("text", ""))) for item in items if isinstance(item, dict)]
    total_len = sum(lengths)
    estimate = int(max(128, min(512, total_len * 3 + len(lengths) * 12)))
    return estimate


def _translate_single(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    style_guide: dict,
    text: str,
    context_lines: list[str] | None = None,
    settings: PipelineSettings | None = None,
) -> str:
    prompt = build_translation_prompt(
        source_lang,
        target_lang,
        style_guide,
        context_lines or [],
        text,
    )
    
    # Defaults
    temp = 0.2
    top_p = 0.9
    
    if settings:
        if settings.translator_backend == "GGUF":
             temp = settings.gguf_temperature
             top_p = settings.gguf_top_p
        else:
             temp = settings.ollama_temperature
             top_p = settings.ollama_top_p

    result = ollama.generate(
        _resolve_model(model),
        prompt,
        timeout=300,
    options={"num_predict": 160, "temperature": temp, "top_p": top_p},
    )
    cleaned = _clean_translation(result)
    if not cleaned and text.strip():
        # Fallback: Force translation
        retry_prompt = (
            f"Translate to {target_lang}. Translate exactly, do not skip. Output only the translation.\n"
            f"Text: {text}"
        )
        result = ollama.generate(
            _resolve_model(model),
            retry_prompt,
            timeout=300,
            options={"num_predict": 160, "temperature": temp},
        )
        cleaned = _clean_translation(result)
    return cleaned


def _ensure_target_language(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    ocr_text: str,
    translation: str,
    is_bubble: bool = False,
) -> tuple[str, bool]:
    if _looks_like_merged_batch_output(translation, ocr_text):
        translation = _translate_strict(ollama, model, source_lang, target_lang, ocr_text)
    elif _too_long_translation(translation, ocr_text):
        translation = _translate_brief(ollama, model, source_lang, target_lang, ocr_text)
    if _looks_like_prompt_leak(translation):
        translation = _translate_strict(ollama, model, source_lang, target_lang, ocr_text)
    
    # Only silence SFX/Empty if it's NOT a speech bubble.
    if not translation and TextFilter(None).should_ignore(ocr_text, "background_text") and not is_bubble:
        return "", True

    if _language_ok(target_lang, translation) and not _looks_like_prompt_leak(translation):
        return translation, True
    
    # Build retry prompt - be explicit about language requirements
    if target_lang == "Simplified Chinese":
        retry_prompt = (
            f"将下面的日语翻译成简体中文。\n"
            f"只输出简体中文译文，不要片假名、平假名、罗马音或英文。\n"
            f"日语原文: {ocr_text}\n"
        )
    else:
        retry_prompt = (
            f"Translate {source_lang} to {target_lang}.\n"
            "No English, no romaji, no explanations.\n"
            f"Text: {ocr_text}\n"
        )
    retry = _clean_translation(
        ollama.generate(
            model,
            retry_prompt,
            timeout=30,
            options={"num_predict": 128, "temperature": 0.1, "top_p": 0.9},
        )
    )
    if _looks_like_prompt_leak(retry):
        retry = _translate_strict(ollama, model, source_lang, target_lang, ocr_text)
    if _language_ok(target_lang, retry) and not _looks_like_prompt_leak(retry):
        return retry, True
    
    # Second retry for Chinese if still has Kana - be even more explicit
    if target_lang == "Simplified Chinese" and _kana_ratio(retry) > 0.05:
        final_prompt = (
            f"请将日语'{ocr_text}'翻译成中文。\n"
            f"重要：你的回答中绝对不能包含日语假名（ひらがな/カタカナ）。\n"
            f"只能使用纯中文汉字进行翻译。\n"
        )
        final = _clean_translation(
            ollama.generate(
                model,
                final_prompt,
                timeout=30,
                options={"num_predict": 128, "temperature": 0.05, "top_p": 0.9},
            )
        )
        if _language_ok(target_lang, final) and not _looks_like_prompt_leak(final):
            return final, True
        retry = final if final else retry
    
    if _looks_like_prompt_leak(retry or translation):
        return "", False
    return retry or translation, False


def _too_long_translation(translation: str, ocr_text: str) -> bool:
    if not translation or not ocr_text:
        return False
    if "\n" in translation:
        return True
    t_len = len(translation)
    o_len = len(ocr_text)
    if o_len <= 4:
        return t_len > max(12, o_len * 3)
    return t_len > o_len * 2.2


def _looks_like_merged_batch_output(translation: str, ocr_text: str) -> bool:
    if not translation:
        return False
    lines = [line.strip() for line in str(translation).splitlines() if line.strip()]
    if len(lines) >= 2:
        return True
    src_punct = sum(1 for ch in ocr_text if ch in "。！？!?…")
    dst_punct = sum(1 for ch in translation if ch in "。！？!?…")
    if dst_punct >= max(3, src_punct + 3) and len(translation) > max(24, len(ocr_text) * 1.6):
        return True
    return False


def _translate_brief(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    text: str,
) -> str:
    if target_lang == "Simplified Chinese":
        prompt = f"将以下{source_lang}翻译成简体中文，保持简短：{text}"
    elif target_lang == "English":
        prompt = f"Translate the following {source_lang} into English. Keep it short: {text}"
    else:
        prompt = f"Translate the following {source_lang} into {target_lang}. Keep it short: {text}"
    result = ollama.generate(
        _resolve_model(model),
        prompt,
        timeout=30,
        options={"num_predict": 128, "temperature": 0.1, "top_p": 0.9},
    )
    return _clean_translation(result)


def _language_ok(target_lang: str, text: str) -> bool:
    if not text:
        return False
    if target_lang == "Simplified Chinese":
        return _cjk_ratio(text) >= 0.3 and _kana_ratio(text) <= 0.1
    if target_lang == "English":
        return _cjk_ratio(text) < 0.2
    return True


def _looks_like_prompt_leak(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    markers = [
        "return only",
        "output only",
        "output only the translation",
        "no labels",
        "no quotes",
        "no explanations",
        "text to translate",
        "<<text>>",
        "<</text>>",
        "translation:",
    ]
    chinese_markers = [
        "只需翻译",
        "仅需翻译",
        "只翻译",
        "不要任何",
        "不要标签",
        "不要引号",
        "不要解释",
        "不要多余",
        "不要说明",
        "不要注释",
        "上下文",
        "译文",
        "只输出",
        "输出译文",
        "只输出翻译",
        "翻译如下",
        "文字：",
        "文本：",
        "原文：",
        "翻译：",
        "不要英文",
        "不要罗马音",
        "不要羅馬音",
        # Additional patterns seen in user-reported prompt leaks
        "不要用英语",
        "不要用罗马音",
        "不要加解释",
        "没有英语",
        "没有罗马音",
        "没有解释",
    ]
    if any(m in lowered for m in markers):
        return True
    for marker in chinese_markers:
        if marker in text:
            return True
    return False


def _translate_strict(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    text: str,
) -> str:
    if target_lang == "Simplified Chinese":
        prompt = f"将以下{source_lang}翻译成简体中文：{text}"
    elif target_lang == "English":
        prompt = f"Translate the following {source_lang} into English: {text}"
    else:
        prompt = f"Translate the following {source_lang} into {target_lang}: {text}"
    result = ollama.generate(
        _resolve_model(model),
        prompt,
        timeout=180,
        options={"num_predict": 160, "temperature": 0.1, "top_p": 0.9},
    )
    return _clean_translation(result)


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if _is_japanese(ch))
    return cjk / max(1, len(text))


def _kana_ratio(text: str) -> float:
    if not text:
        return 0.0
    kana = sum(1 for ch in text if _is_kana(ch))
    return kana / max(1, len(text))




def _should_skip_text(text: str, bbox: list, image_size: tuple[int, int]) -> bool:
    if not text:
        return True
    if _is_punct_only(text):
        return True
    if _placeholder_ratio(text) >= 0.15:
        return True
    
    # CRITICAL FIX: If text is strongly valid Japanese, NEVER skip it
    # This ensures short dialogue like "フ…", "そ", "え?" are always translated
    if _is_valid_japanese(text) >= 0.6:
        return False
    
    x, y, w, h = bbox
    area = w * h
    img_w, img_h = image_size
    page_area = img_w * img_h if img_w and img_h else 1
    ratio = area / page_area
    length = len(text)
    jp_ratio = _japanese_ratio(text)
    if length <= 2 and ratio < 0.003:
        # Check aspect ratio for very small boxes
        aspect = w / h if h else 0
        
        # FIX: "そ" and vertical text (tall/narrow) are often skipped by current aspect ratio check (0.3 < aspect < 3.5)
        # If it's strongly Japanese, KEEP IT regardless of aspect ratio
        if _is_valid_japanese(text) >= 0.5:
             return False

        if jp_ratio >= 0.6 and 0.3 < aspect < 3.5:
            return False
        return True
    if jp_ratio < 0.3 and length < 6:
        return True
    if jp_ratio < 0.2 and ratio < 0.006:
        return True
    return False


def _should_ignore_speech_fragment(
    text: str,
    bbox: list,
    image_size: tuple[int, int],
    ocr_conf: float,
) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return True
    if _is_punct_only(cleaned):
        return True
    if _placeholder_ratio(cleaned) >= 0.2:
        return True
    img_w, img_h = image_size
    page_area = max(1, img_w * img_h)
    _, _, w, h = bbox
    area_ratio = (max(1, w) * max(1, h)) / page_area
    kana_only = all(_is_kana(ch) or ch in {"ー", "・"} for ch in cleaned)
    narrow_box = min(max(1, w), max(1, h)) <= 42
    if len(cleaned) == 1:
        if kana_only and area_ratio < 0.0035 and ocr_conf < 0.985:
            return True
        if cleaned in {"っ", "ッ", "ー", "・"}:
            return True
    if len(cleaned) == 2 and kana_only and area_ratio < 0.0025 and ocr_conf < 0.96:
        return True
    if len(cleaned) == 3 and kana_only and narrow_box and area_ratio < 0.0015 and ocr_conf < 0.93:
        return True
    if len(cleaned) <= 3 and kana_only and narrow_box and area_ratio < 0.0009 and ocr_conf < 0.985:
        return True
    return False


def _classify_semantic_region(
    text: str,
    bbox: list,
    image_size: tuple[int, int],
    det_conf: float,
    ocr_conf: float,
    image_obj,
    text_filter: TextFilter,
    initial_bg: bool = False,
) -> tuple[str, bool, bool, bool, dict]:
    cleaned = str(text or "").strip()
    region_type = "background_text" if initial_bg else "speech_bubble"
    bg_text = bool(initial_bg)
    needs_review = det_conf < 0.6
    render_updates: dict[str, object] = {}

    if not cleaned:
        return region_type, bg_text, True, True, render_updates

    stats = _box_luma_stats_pil(image_obj, bbox)
    _, _, w, h = bbox
    aspect = w / max(1, h)
    thin_strip = h <= 28 and aspect >= 3.0
    katakana_ratio = _katakana_ratio_text(cleaned)
    contains_kanji = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in cleaned)
    mixed_scripts = _has_mixed_scripts(cleaned)
    has_latin = _has_latin_text(cleaned)

    if _looks_like_decorative_title_artifact(
        cleaned,
        bbox,
        image_size,
        det_conf,
        ocr_conf,
        mixed_scripts,
        has_latin,
    ):
        return "background_text", True, True, True, render_updates

    if _is_dark_caption_box(stats, cleaned):
        region_type = "background_text"
        bg_text = True
        render_updates = {"color": "#FFFFFF", "stroke": "#000000", "stroke_width": 1}

    if thin_strip and not bg_text:
        region_type = "background_text"
        bg_text = True

    if bg_text:
        if text_filter.should_ignore(cleaned, "background_text"):
            if not contains_kanji or katakana_ratio >= 0.6 or len(cleaned) <= 6:
                return region_type, bg_text, True, True, render_updates
        if _looks_like_background_artifact(cleaned, bbox, image_size, det_conf, ocr_conf, mixed_scripts):
            return region_type, bg_text, True, True, render_updates
        return region_type, bg_text, False, needs_review, render_updates

    if text_filter.should_ignore(cleaned, "speech_bubble") and _likely_sfx_effect_box(
        cleaned, bbox, image_size, ocr_conf
    ):
        return region_type, bg_text, True, True, render_updates

    return region_type, bg_text, False, needs_review, render_updates


def _box_luma_stats_pil(image_obj, bbox: list):
    if image_obj is None or not bbox:
        return None
    try:
        from PIL import ImageStat
    except Exception:
        return None
    try:
        img_w, img_h = image_obj.size
        x, y, w, h = [int(v) for v in bbox[:4]]
        x0 = max(0, min(x, img_w - 1))
        y0 = max(0, min(y, img_h - 1))
        x1 = max(x0 + 1, min(x + max(1, w), img_w))
        y1 = max(y0 + 1, min(y + max(1, h), img_h))
        crop = image_obj.crop((x0, y0, x1, y1)).convert("L")
        stat = ImageStat.Stat(crop)
        extrema = crop.getextrema()
        if not stat.mean or extrema is None:
            return None
        return float(stat.mean[0]), int(extrema[0]), int(extrema[1])
    except Exception:
        return None


def _is_dark_caption_box(stats, text: str) -> bool:
    if not stats or len(text) < 2:
        return False
    mean, low, high = stats
    if high >= 190:
        return False
    return mean < 125 and low < 110


def _katakana_ratio_text(text: str) -> float:
    if not text:
        return 0.0
    count = sum(1 for ch in text if 0x30A0 <= ord(ch) <= 0x30FF)
    return count / max(1, len(text))


def _has_mixed_scripts(text: str) -> bool:
    has_hira = any(0x3040 <= ord(ch) <= 0x309F for ch in text)
    has_kata = any(0x30A0 <= ord(ch) <= 0x30FF for ch in text)
    has_kanji = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in text)
    return sum(1 for flag in (has_hira, has_kata, has_kanji) if flag) >= 3


def _has_latin_text(text: str) -> bool:
    return any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in str(text or ""))


def _looks_like_decorative_title_artifact(
    text: str,
    bbox: list,
    image_size: tuple[int, int],
    det_conf: float,
    ocr_conf: float,
    mixed_scripts: bool,
    has_latin: bool,
) -> bool:
    if not text:
        return False
    if any(ch in text for ch in "。！？!?"):
        return False
    has_cjk = any(_is_cjk_char(ch) for ch in str(text or ""))
    _, _, w, h = bbox
    page_area = max(1, image_size[0] * image_size[1])
    area_ratio = (max(1, w) * max(1, h)) / page_area
    large_box = area_ratio >= 0.012 or (max(w, h) >= min(image_size) * 0.22)
    if has_latin and has_cjk and large_box:
        return True
    if has_latin and mixed_scripts and large_box:
        return True
    if has_latin and area_ratio >= 0.006 and ocr_conf < 0.995 and det_conf >= 0.85:
        return True
    return False


def _looks_like_background_artifact(
    text: str,
    bbox: list,
    image_size: tuple[int, int],
    det_conf: float,
    ocr_conf: float,
    mixed_scripts: bool,
) -> bool:
    _, _, w, h = bbox
    page_area = max(1, image_size[0] * image_size[1])
    area_ratio = (max(1, w) * max(1, h)) / page_area
    thin_strip = h <= 28 and w >= h * 3.0
    if thin_strip and mixed_scripts and ocr_conf < 0.95:
        return True
    if thin_strip and len(text) <= 8 and det_conf >= 0.95 and ocr_conf < 0.92:
        return True
    if area_ratio < 0.001 and _placeholder_ratio(text) > 0.0:
        return True
    return False


def _likely_sfx_effect_box(
    text: str,
    bbox: list,
    image_size: tuple[int, int],
    ocr_conf: float,
) -> bool:
    if any(ch in text for ch in "、。！？!?…"):
        return False
    _, _, w, h = bbox
    page_area = max(1, image_size[0] * image_size[1])
    area_ratio = (max(1, w) * max(1, h)) / page_area
    short = len(text) <= 6
    mostly_katakana = _katakana_ratio_text(text) >= 0.6
    return short and mostly_katakana and (min(w, h) <= 60 or area_ratio < 0.003) and ocr_conf < 0.995


def _japanese_ratio(text: str) -> float:
    if not text:
        return 0.0
    jp = sum(1 for ch in text if _is_japanese(ch))
    return jp / max(1, len(text))


def _placeholder_ratio(text: str) -> float:
    if not text:
        return 0.0
    placeholders = {"□", "口", "�"}
    count = sum(1 for ch in text if ch in placeholders)
    return count / max(1, len(text))


def _is_punct_only(text: str) -> bool:
    stripped = "".join(ch for ch in text if ch.strip())
    if not stripped:
        return True
    letters = sum(1 for ch in stripped if ch.isalnum() or _is_japanese(ch))
    return letters == 0


def _clean_ocr_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("□", "").replace("�", "")
    if _placeholder_ratio(cleaned) >= 0.2:
        cleaned = cleaned.replace("口", "")
    
    # For CJK text, remove ALL spaces (Japanese/Chinese don't use word spaces)
    # Use _is_valid_japanese score which correctly includes punctuation
    # If score > 0.4, it's likely Japanese/Chinese text
    stripped = cleaned.replace(" ", "")
    if stripped and _is_valid_japanese(stripped) > 0.4:
        # Remove all spaces from Japanese-dominant text
        cleaned = stripped
    
    # For non-CJK text, just normalize whitespace
    if " " in cleaned:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned


def _is_cjk_char(ch: str) -> bool:
    """Check if character is CJK (Chinese/Japanese/Korean)."""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF       # CJK Unified Ideographs
        or 0x3040 <= code <= 0x30FF    # Hiragana + Katakana
        or 0x3400 <= code <= 0x4DBF    # CJK Extension A
    )


def _is_japanese(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF
        or 0x4E00 <= code <= 0x9FFF
    )


def _is_kana(ch: str) -> bool:
    code = ord(ch)
    return 0x3040 <= code <= 0x30FF


def _is_font_allowed_for_cn(font_name: str) -> bool:
    if not font_name:
        return False
    allowed = {
        "Noto Sans CJK",
        "Microsoft YaHei",
        "SimSun",
        "SimHei",
    }
    name = font_name.strip().lower()
    for item in allowed:
        if item.lower() in name:
            return True
    return False


def _region_record(
    idx: int,
    polygon: list,
    bbox: list,
    ocr_text: str,
    translation: str,
    det_conf: float,
    bg_text: bool,
    needs_review: bool,
    ignore: bool,
    font_name: str,
    region_type: str = "speech_bubble",
    ocr_conf: float = 1.0,
    render_updates: dict | None = None,
) -> dict:
    render = {
        "font": font_name,
        "font_size": 0,
        "line_height": 1.2,
        "align": "center",
        "color": "#000000",
        "stroke": "#FFFFFF",
        "stroke_width": 2,
        "wrap_mode": "auto",
    }
    if isinstance(render_updates, dict):
        render.update({k: v for k, v in render_updates.items() if v is not None})
    return {
        "region_id": f"r{idx:03d}",
        "bbox": bbox,
        "polygon": polygon,
        "type": region_type,
        "ocr_text": ocr_text,
        "translation": translation,
        "confidence": {"det": det_conf, "ocr": ocr_conf, "trans": 1.0},
        "render": render,
        "flags": {"ignore": ignore, "bg_text": bg_text, "needs_review": needs_review},
    }

def _get_image_size(image_path: str) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError:
        return (0, 0)
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return (0, 0)


def _read_image_cv(image_path: str):
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


def _scale_polygon(polygon: list, scale: float) -> list:
    scaled = []
    for point in polygon:
        if point is None or len(point) < 2:
            continue
        scaled.append([float(point[0]) * scale, float(point[1]) * scale])
    return scaled


def _detect_with_scale(detector, image_path: str, image_size: tuple[int, int], target_long: int = 1280):
    image = _read_image_cv(image_path)
    if image is None or not hasattr(detector, "detect_image"):
        return detector.detect(image_path)
    try:
        import cv2
    except Exception:
        return detector.detect(image_path)
    h, w = image.shape[:2]
    long_edge = max(w, h)
    scale = 1.0
    if long_edge > target_long:
        scale = target_long / float(long_edge)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    detections = detector.detect_image(image)
    if scale != 1.0:
        inv = 1.0 / scale
        scaled = []
        for polygon, conf in detections:
            scaled.append((_scale_polygon(polygon, inv), conf))
        return scaled
    return detections


def _get_detector_fallback(detector, use_gpu: bool):
    fallback = getattr(detector, "_runtime_fallback_detector", None)
    if fallback is not None:
        return fallback
    from app.detect.paddle_detector import PaddleTextDetector

    fallback = PaddleTextDetector(use_gpu)
    setattr(detector, "_runtime_fallback_detector", fallback)
    return fallback


def _detect_regions(
    detector,
    image_path: str,
    image_size: tuple[int, int],
    input_size: int = 1024,
    use_gpu: bool = False,
    message_callback=None,
):
    if getattr(detector, "_runtime_fallback_active", False):
        fallback = _get_detector_fallback(detector, use_gpu)
        return _detect_with_scale(fallback, image_path, image_size, target_long=input_size)
    try:
        if hasattr(detector, "detect"):
            try:
                return detector.detect(image_path, input_size=input_size)
            except TypeError:
                return _detect_with_scale(detector, image_path, image_size, target_long=input_size)
        return _detect_with_scale(detector, image_path, image_size, target_long=input_size)
    except Exception as exc:
        detector_name = detector.__class__.__name__
        if detector_name != "ComicTextDetector":
            raise
        logger.warning("Detector failed on %s with %s. Falling back to PaddleTextDetector.", image_path, exc)
        if message_callback is not None:
            try:
                message_callback(f"ComicTextDetector failed on {os.path.basename(image_path)}; using Paddle fallback.")
            except Exception:
                pass
        fallback = _get_detector_fallback(detector, use_gpu)
        setattr(detector, "_runtime_fallback_active", True)
        return _detect_with_scale(fallback, image_path, image_size, target_long=input_size)


def _classify_region(
    bbox: list,
    image_size: tuple[int, int],
    det_conf: float,
    filter_background: bool,
    filter_strength: str,
) -> tuple[bool, bool]:
    img_w, img_h = image_size
    if img_w <= 0 or img_h <= 0:
        return False, det_conf < 0.6
    x, y, w, h = bbox
    area = w * h
    page_area = img_w * img_h
    if page_area <= 0:
        return False, det_conf < 0.6
    ratio = area / page_area
    aspect = w / h if h else 0
    margin_x = img_w * 0.02
    margin_y = img_h * 0.02
    near_edge = x < margin_x or y < margin_y or (x + w) > (img_w - margin_x) or (y + h) > (img_h - margin_y)

    aggressive = filter_strength == "aggressive"
    large_ratio = 0.12 if not aggressive else 0.09
    strip_ratio = 0.05 if not aggressive else 0.03
    edge_ratio = 0.03 if not aggressive else 0.02

    bg_text = False
    if ratio > large_ratio and (near_edge or aspect > 4):
        bg_text = True
    elif aspect > 5 and ratio > strip_ratio:
        bg_text = True
    elif near_edge and ratio > edge_ratio:
        bg_text = True

    if not filter_background:
        bg_text = False

    needs_review = det_conf < 0.6 or (bg_text and aggressive)
    return bg_text, needs_review


def _is_cjk_term(term: str) -> bool:
    for ch in term:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            return True
        if 0x3040 <= code <= 0x30FF:
            return True
        if 0xAC00 <= code <= 0xD7AF:
            return True
    return False


def _contains_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    if _is_cjk_term(term):
        return term in text
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _normalize_character_entry(entry: dict) -> dict:
    """Normalize character schema to a stable structure for all pipeline consumers."""
    if not isinstance(entry, dict):
        return {}
    original = str(entry.get("original") or entry.get("canonical") or "").strip()
    reading = str(entry.get("reading") or entry.get("canonical_reading") or "").strip()
    translation = str(entry.get("translation") or "").strip()
    name = str(entry.get("name") or "").strip()
    if not original and name:
        original = name
    if not name:
        name = translation or original
    aliases_raw = entry.get("aliases", []) or []
    aliases = []
    for alias in aliases_raw:
        if isinstance(alias, dict):
            source = str(alias.get("source", "")).strip()
            target = str(alias.get("target", "") or alias.get("translation", "")).strip()
            if source:
                aliases.append(
                    {
                        "source": source,
                        "target": target,
                        "reading": str(alias.get("reading", "")).strip(),
                        "pattern": str(alias.get("pattern", "")).strip(),
                        "hint": str(alias.get("hint", "")).strip(),
                    }
                )
        else:
            source = str(alias).strip()
            if source:
                aliases.append(
                    {
                        "source": source,
                        "target": translation,
                        "reading": "",
                        "pattern": "",
                        "hint": "",
                    }
                )
    return {
        "canonical": original,
        "original": original,
        "name": name,
        "translation": translation,
        "reading": reading,
        "gender": str(entry.get("gender", "")).strip(),
        "info": str(entry.get("info", "")).strip(),
        "aliases": aliases,
    }


def _find_inconsistent_pages(pages: list, style_guide: dict) -> list[int]:
    if not pages or not isinstance(style_guide, dict):
        return []
    term_targets: dict[str, set[str]] = {}
    glossary = style_guide.get("glossary", [])
    for item in glossary:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source", "")).strip()
        tgt = str(item.get("target", "")).strip()
        if len(src) < 2 or not tgt:
            continue
        term_targets.setdefault(src, set()).add(tgt)
    characters = style_guide.get("characters", [])
    if isinstance(characters, list):
        for raw_char in characters:
            char = _normalize_character_entry(raw_char)
            if not char:
                continue
            original = str(char.get("original", "")).strip()
            translation = str(char.get("translation", "")).strip()
            canonical_target = translation
            if original and canonical_target and canonical_target != original:
                term_targets.setdefault(original, set()).add(canonical_target)
            aliases = char.get("aliases", []) or []
            for alias in aliases:
                alias_source = str(alias.get("source", "")).strip()
                alias_target = str(alias.get("target", "")).strip()
                if not alias_source:
                    continue
                alias_targets = set()
                if alias_target and alias_target != alias_source:
                    alias_targets.add(alias_target)
                if canonical_target and canonical_target != alias_source:
                    alias_targets.add(canonical_target)
                if alias_targets:
                    term_targets.setdefault(alias_source, set()).update(alias_targets)
    if not term_targets:
        return []
    terms = list(term_targets.items())
    inconsistent_pages = []
    for page_idx, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        regions = page.get("regions", []) or page.get("blocks", [])
        for region in regions:
            if not isinstance(region, dict):
                continue
            flags = region.get("flags", {}) or {}
            if flags.get("ignore"):
                continue
            source_text = str(region.get("ocr_text", "")).strip()
            translation = str(region.get("translation", "")).strip()
            if not source_text or not translation:
                continue
            for src, targets in terms:
                if _contains_term(source_text, src):
                    if not any(_contains_term(translation, tgt) for tgt in targets):
                        inconsistent_pages.append(page_idx)
                        break
            if inconsistent_pages and inconsistent_pages[-1] == page_idx:
                break
    return inconsistent_pages


def _sanitize_style_guide(style_guide: dict, target_lang: str) -> dict:
    if not isinstance(style_guide, dict):
        return style_guide
    glossary = style_guide.get("glossary", [])
    cleaned_glossary = []
    changed = False
    # Normalize characters to a single schema.
    normalized_chars = []
    raw_chars = style_guide.get("characters", []) or []
    for raw_char in raw_chars:
        norm = _normalize_character_entry(raw_char)
        if norm and norm.get("original"):
            normalized_chars.append(norm)
    if raw_chars != normalized_chars:
        style_guide = dict(style_guide)
        style_guide["characters"] = normalized_chars
        changed = True

    # Collect aliases for name validation.
    alias_sources = set()
    for char in normalized_chars:
        original = str(char.get("original", "")).strip()
        if original:
            alias_sources.add(original)
        for alias in char.get("aliases", []) or []:
            src = str(alias.get("source", "")).strip()
            if src:
                alias_sources.add(src)
    honorifics = ("さん", "くん", "ちゃん", "様", "先生", "先輩", "殿", "君", "氏")
    for item in glossary:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        reading = str(item.get("reading", "")).strip()
        if item.get("auto"):
            has_honorific = any(h in source for h in honorifics)
            reading_is_kana = bool(reading) and all(_is_kana(ch) for ch in reading)
            source_is_kana = bool(source) and all(_is_kana(ch) for ch in source)
            if not (has_honorific or reading_is_kana or source_is_kana or source in alias_sources):
                if _is_cjk_term(source) and (not reading or reading == source) and len(source) <= 3:
                    changed = True
                    continue
                if not reading or reading == source:
                    changed = True
                    continue
            cleaned_target = _sanitize_glossary_target(target, source, target_lang)
            if not cleaned_target:
                changed = True
                continue
            if cleaned_target != target:
                new_item = dict(item)
                new_item["target"] = cleaned_target
                cleaned_glossary.append(new_item)
                changed = True
                continue
        cleaned_glossary.append(item)
    if changed:
        style_guide = dict(style_guide)
        style_guide["glossary"] = cleaned_glossary
    return style_guide


def _merge_glossary(style_guide: dict, new_map: dict, new_chars: list) -> dict:
    """Merge new glossary items into style guide."""
    # Ensure glossary list exists
    sg_glossary = style_guide.setdefault("glossary", [])
    
    # Map existing entries by source for quick lookup
    existing_map = {item["source"]: item for item in sg_glossary if "source" in item}
    
    for src, val in new_map.items():
        # Handle rich dict vs simple string
        if isinstance(val, dict):
            target = val.get("target", "")
            reading = val.get("reading", "")
            pattern = val.get("pattern", "")
            hint = val.get("hint", "")
            entry_type = val.get("type", "term")
        else:
            target = val
            reading = ""
            pattern = ""
            hint = ""
            entry_type = "term"
            
        if src not in existing_map:
            # Create new entry
            entry = {
                "source": src,
                "target": target,
                "priority": "hard",
                "auto": True
            }
            if reading: entry["reading"] = reading
            if pattern: entry["pattern"] = pattern
            if hint: entry["hint"] = hint
            if entry_type: entry["type"] = entry_type
            
            sg_glossary.append(entry)
            existing_map[src] = entry
        else:
            # Update existing if needed (e.g. add metadata)
            entry = existing_map[src]
            if entry.get("auto"):
                 if target and target != entry.get("target", ""):
                     entry["target"] = target
                 if reading and "reading" not in entry:
                     entry["reading"] = reading
                 if pattern and "pattern" not in entry:
                     entry["pattern"] = pattern
                 if hint and "hint" not in entry:
                     entry["hint"] = hint
    
    # Merge characters with normalized schema.
    sg_chars_raw = style_guide.setdefault("characters", [])
    sg_chars = []
    existing_chars = {}
    for c in sg_chars_raw:
        norm = _normalize_character_entry(c)
        if not norm or not norm.get("original"):
            continue
        key = norm.get("original")
        sg_chars.append(norm)
        existing_chars[key] = norm
    style_guide["characters"] = sg_chars

    if new_chars:
        for char in new_chars:
            norm_char = _normalize_character_entry(char)
            if not norm_char:
                continue
            original = norm_char.get("original", "").strip()
            if len(original) > 20 or "处理用户" in original or "需要" in original:
                continue
            if not original:
                continue

            existing = existing_chars.get(original)
            if existing is None:
                sg_chars.append(norm_char)
                existing_chars[original] = norm_char
                continue

            new_aliases = norm_char.get("aliases", [])
            # Fill canonical fields if the existing entry is incomplete.
            if not existing.get("translation") and norm_char.get("translation"):
                existing["translation"] = norm_char.get("translation")
            if (not existing.get("name") or existing.get("name") == original) and norm_char.get("name"):
                existing["name"] = norm_char.get("name")
            if not existing.get("reading") and norm_char.get("reading"):
                existing["reading"] = norm_char.get("reading")
            if not existing.get("gender") and norm_char.get("gender"):
                existing["gender"] = norm_char.get("gender")
            if not existing.get("info") and norm_char.get("info"):
                existing["info"] = norm_char.get("info")
            existing_aliases = existing.setdefault("aliases", [])
            existing_alias_sources = set()
            for a in existing_aliases:
                s = a.get("source") if isinstance(a, dict) else str(a)
                if s:
                    existing_alias_sources.add(s)
            for alias in new_aliases:
                src = alias.get("source") if isinstance(alias, dict) else str(alias)
                if src and src not in existing_alias_sources:
                    existing_aliases.append(alias)
                    existing_alias_sources.add(src)

    return style_guide
