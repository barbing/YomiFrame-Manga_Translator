# -*- coding: utf-8 -*-
"""Model downloader logic."""
import os
import requests
from PySide6 import QtCore
from app.config.defaults import (
    COMIC_TEXT_DETECTOR_GPU, 
    COMIC_TEXT_DETECTOR_CPU, 
    SAKURA_GGUF,
    QWEN_GGUF,
    BIG_LAMA,
    MANGA_OCR_BASE_URL,
    MANGA_OCR_FILES
)

import hashlib
from dataclasses import dataclass
from typing import List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tarfile
import zipfile
import shutil
from app.models.resolution import (
    has_paddle_runtime_models,
    resolve_manga_ocr_local_dir,
    resolve_manga_ocr_system_ref,
    resolve_ner_local_dir,
    resolve_ner_system_snapshot,
)

@dataclass
class DownloadTarget:
    url: str
    save_path: str
    label: str
    sha256: str = None  # Optional checksum

class ModelDownloader(QtCore.QObject):
    progress_changed = QtCore.Signal(int)
    status_changed = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)  # success, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel_requested = False
        self._session = self._create_session()
        self._pending_targets: List[DownloadTarget] = []

    def _create_session(self) -> requests.Session:
        """Create a robust requests session with retries."""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            "User-Agent": "YomiFrame/1.2.0"
        })
        return session

    def request_cancel(self):
        self._cancel_requested = True

    def _check_hf_cache(self, user: str, repo: str, filename: str = None) -> bool:
        """Check Hugging Face system caches (respects HF_HOME)."""
        try:
            # 1. Check environment variable
            hf_home = os.environ.get("HF_HOME")
            if not hf_home:
                hf_home = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
            
            # Standard HF cache structure: hub/models--user--repo/snapshots/hash/filename
            # We check if *any* snapshot exists and contains the file (if specified)
            model_dir = os.path.join(hf_home, "hub", f"models--{user}--{repo}")
            if not os.path.exists(model_dir):
                return False
                
            snapshots_dir = os.path.join(model_dir, "snapshots")
            if not os.path.exists(snapshots_dir):
                return False
            
            # Check all snapshots
            for snapshot in os.listdir(snapshots_dir):
                snap_path = os.path.join(snapshots_dir, snapshot)
                if not os.path.isdir(snap_path):
                    continue
                
                # If filename specified, check for it
                if filename:
                    if os.path.exists(os.path.join(snap_path, filename)):
                        return True
                else:
                    # If no filename, just existence of snapshot is enough
                    return True
                    
        except Exception:
            pass
        return False

    def check_comic_text_detector(self, models_dir: str) -> bool:
        """Check if ComicTextDetector models exist (Portable ONLY per user request)."""
        target_dir = os.path.join(models_dir, "comic-text-detector")
        path_gpu = os.path.join(target_dir, "comictextdetector.pt")
        path_cpu = os.path.join(target_dir, "comictextdetector.pt.onnx")
        
        return os.path.exists(path_gpu) and os.path.exists(path_cpu)

    def check_manga_ocr(self, models_dir: str) -> bool:
        """Check if MangaOCR models exist (System Priority > Portable)."""
        return bool(resolve_manga_ocr_system_ref() or resolve_manga_ocr_local_dir(models_dir))

    def check_big_lama(self, models_dir: str) -> bool:
        """Check if BigLama model exists (System Priority > Portable)."""
        # 1. System/Torch Cache (Priority)
        # Check TORCH_HOME or default ~/.cache/torch/hub/checkpoints
        try:
            torch_home = os.environ.get("TORCH_HOME")
            if not torch_home:
                torch_home = os.path.join(os.path.expanduser("~"), ".cache", "torch")
            
            cache_path = os.path.join(torch_home, "hub", "checkpoints", "big-lama.pt")
            if os.path.exists(cache_path):
                return True
        except Exception:
            pass

        # 2. Local check
        if os.path.exists(os.path.join(models_dir, "lama", "big-lama.pt")):
            return True
            
        return False

    def download_targets(self, targets: List[DownloadTarget]):
        """Generic downloader for a list of targets."""
        for target in targets:
            if self._cancel_requested:
                return

            # Check if file exists and verify checksum if provided
            if os.path.exists(target.save_path):
                if target.sha256:
                    self.status_changed.emit(f"Verifying {os.path.basename(target.save_path)}...")
                    if self._verify_checksum(target.save_path, target.sha256):
                        continue # File is good
                    else:
                        os.remove(target.save_path) # Corrupt, re-download
                else:
                    continue # Assume good if no checksum

            os.makedirs(os.path.dirname(target.save_path), exist_ok=True)
            if not self._download_file(target):
                return

    def queue_targets(self, targets: List[DownloadTarget]):
        """Queue targets for download."""
        self._pending_targets.extend(targets)

    def check_ner(self, models_dir: str) -> bool:
        """Check if NER model exists (System Priority > Portable)."""
        return bool(resolve_ner_system_snapshot() or resolve_ner_local_dir(os.path.join(models_dir, "ner")))

    def check_paddle_ocr(self, models_dir: str = "models") -> bool:
        """Check if PaddleOCR runtime models are installed."""
        return has_paddle_runtime_models(base_dir=models_dir)

    def _check_paddle_system_cache(self) -> bool:
        """Check ~/.paddleocr for existing models."""
        try:
            home = os.path.expanduser("~")
            base = os.path.join(home, ".paddleocr", "whl")
            if os.path.exists(base) and any(os.scandir(base)):
                return True
        except Exception:
            pass
        return False

    def prepare_ner(self, models_dir: str):
        """Queue NER model download."""
        self._ner_target_dir = os.path.join(models_dir, "ner")
        self._download_ner = True

    def prepare_paddle(self, models_dir: str):
        """Queue PaddleOCR manual model download."""
        target_dir = os.path.join(models_dir, "paddleocr")
        
        # Define the 3 models
        # Define the 3 models (Configured for Chinese PP-OCRv4)
        urls = [
             ("https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar", "Detection Model (v4)"),
             ("https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar", "Recognition Model (v4)"),
             ("https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar", "Classifier Model")
        ]
        
        targets = []
        for url, label in urls:
            filename = url.split("/")[-1]
            targets.append(DownloadTarget(
                url=url,
                save_path=os.path.join(target_dir, filename), 
                label=f"Downloading PaddleOCR {label}..."
            ))
            
        self.queue_targets(targets)

    def _perform_ner_download(self) -> bool:
        """Execute NER download using transformers."""
        try:
            from app.nlp.ner_extractor import download_ner_model
            
            def progress_adapter(percent):
                self.progress_changed.emit(percent)
                
            self.status_changed.emit("Downloading NER Model (bert-ner-japanese)...")
            return download_ner_model(self._ner_target_dir, progress_callback=progress_adapter)
        except Exception as e:
            self.finished.emit(False, f"NER Download failed: {e}")
            return False

    def process_queue(self):
        """Process queued targets (Slot)."""
        if not self._pending_targets and not getattr(self, "_download_ner", False):
            self.finished.emit(True, "No tasks.")
            return

        if self._pending_targets:
            self.download_targets(self._pending_targets)
            self._pending_targets.clear()
             
        if self._cancel_requested:
            self.finished.emit(False, "Cancelled")
            return

        # Execute NER download if queued
        if getattr(self, "_download_ner", False):
            success = self._perform_ner_download()
            self._download_ner = False
            if not success:
                return

        if self._cancel_requested:
            self.finished.emit(False, "Cancelled")
            return

        self.finished.emit(True, "All downloads completed.")


    def prepare_comic_text_detector(self, models_dir: str):
        """Queue ComicTextDetector models."""
        target_dir = os.path.join(models_dir, "comic-text-detector")
        targets = [
            DownloadTarget(
                COMIC_TEXT_DETECTOR_CPU,
                os.path.join(target_dir, "comictextdetector.pt.onnx"),
                "Downloading ComicTextDetector (CPU)..."
            ),
            DownloadTarget(
                COMIC_TEXT_DETECTOR_GPU,
                os.path.join(target_dir, "comictextdetector.pt"),
                "Downloading ComicTextDetector (GPU)..."
            )
        ]
        self.queue_targets(targets)

    def prepare_sakura(self, models_dir: str):
        """Queue Sakura GGUF model."""
        target_dir = os.path.join(models_dir, "sakura")
        targets = [
            DownloadTarget(
                SAKURA_GGUF,
                os.path.join(target_dir, "sakura-14b-qwen3-v1.5-q6k.gguf"),
                "Downloading Sakura 14B Q6k (Subject to network speed)..."
            )
        ]
        self.queue_targets(targets)

    def prepare_qwen(self, models_dir: str):
        """Queue Qwen GGUF model."""
        target_dir = os.path.join(models_dir, "qwen")
        targets = [
            DownloadTarget(
                QWEN_GGUF,
                os.path.join(target_dir, "Qwen3-14B-Q6_K.gguf"),
                "Downloading Qwen 14B Q6k (Subject to network speed)..."
            )
        ]
        self.queue_targets(targets)

    def prepare_manga_ocr(self, models_dir: str):
        """Queue MangaOCR models."""
        target_dir = os.path.join(models_dir, "manga-ocr")
        targets = []
        for filename in MANGA_OCR_FILES:
            targets.append(
                DownloadTarget(
                    url=MANGA_OCR_BASE_URL + filename,
                    save_path=os.path.join(target_dir, filename),
                    label=f"Downloading MangaOCR: {filename}"
                )
            )
        self.queue_targets(targets)

    def prepare_big_lama(self, models_dir: str):
        """Queue LaMa model."""
        target_dir = os.path.join(models_dir, "lama")
        targets = [
            DownloadTarget(
                BIG_LAMA,
                os.path.join(target_dir, "big-lama.pt"),
                "Downloading Inpainting Model (BigLama)..."
            )
        ]
        self.queue_targets(targets)

    def _verify_checksum(self, path: str, expected_sha256: str) -> bool:
        """Verify file checksum."""
        sha256 = hashlib.sha256()
        try:
            with open(path, 'rb') as f:
                while True:
                    data = f.read(65536)
                    if not data:
                        break
                    sha256.update(data)
            return sha256.hexdigest().lower() == expected_sha256.lower()
        except Exception:
            return False

    def _extract_archive(self, archive_path: str):
        """Extract .tar or .zip archives."""
        directory = os.path.dirname(archive_path)
        try:
            if archive_path.endswith(".tar"):
                with tarfile.open(archive_path, "r") as tar:
                    safe_members = []
                    for member in tar.getmembers():
                        # Block links and path traversal.
                        if member.issym() or member.islnk():
                            raise RuntimeError(f"Unsafe archive entry (link): {member.name}")
                        _safe_extract_path(directory, member.name)
                        safe_members.append(member)
                    tar.extractall(path=directory, members=safe_members)
            elif archive_path.endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    for member in zip_ref.infolist():
                        _safe_extract_path(directory, member.filename)
                        zip_ref.extract(member, directory)
            
            # Remove archive after extraction
            os.remove(archive_path)
            self.status_changed.emit(f"Extracted {os.path.basename(archive_path)}")
            return True
            
        except Exception as e:
            self.status_changed.emit(f"Extraction failed: {e}")
            return False

    def _download_file(self, target: DownloadTarget) -> bool:
        """Helper to download a single file with progress."""
        if self._cancel_requested:
            self.finished.emit(False, "Cancelled")
            return False

        self.status_changed.emit(target.label)
        try:
            # Connect timeout: 10s, Read timeout: 120s (tolerates slow streams)
            with self._session.get(target.url, stream=True, timeout=(10, 120)) as r:
                r.raise_for_status()
                total_header = r.headers.get("content-length")
                total_length = None
                if total_header:
                    try:
                        total_length = int(total_header)
                    except (TypeError, ValueError):
                        total_length = None

                dl = 0
                last_percent = -1
                with open(target.save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if self._cancel_requested:
                            f.close()
                            if os.path.exists(target.save_path):
                                os.remove(target.save_path)
                            self.finished.emit(False, "Cancelled")
                            return False
                        if not chunk:
                            continue
                        dl += len(chunk)
                        f.write(chunk)
                        if total_length and total_length > 0:
                            percent = int(100 * dl / total_length)
                            if percent > last_percent:
                                self.progress_changed.emit(percent)
                                last_percent = percent
                if total_length is None:
                    self.progress_changed.emit(100)

            # Post-download verification
            if not target.sha256:
                self.status_changed.emit(
                    f"Checksum not provided for {os.path.basename(target.save_path)}; integrity not fully verified."
                )
            if target.sha256 and not self._verify_checksum(target.save_path, target.sha256):
                 self.finished.emit(False, "Download failed: Checksum mismatch.")
                 return False

            # Post-download extraction
            if target.save_path.endswith(".tar") or target.save_path.endswith(".zip"):
                self.status_changed.emit("Extracting archive...")
                if not self._extract_archive(target.save_path):
                    self.finished.emit(False, "Download failed: Archive extraction error.")
                    return False

            return True
        except Exception as e:
            self.finished.emit(False, f"Download failed: {str(e)}")
            return False


def _safe_extract_path(base_dir: str, member_name: str) -> str:
    """Return validated extraction path; raise on traversal/absolute paths."""
    if not member_name:
        raise RuntimeError("Unsafe archive entry: empty filename")
    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
        raise RuntimeError(f"Unsafe archive entry: {member_name}")
    target_path = os.path.abspath(os.path.join(base_dir, member_name))
    base_abs = os.path.abspath(base_dir)
    try:
        inside = os.path.commonpath([base_abs, target_path]) == base_abs
    except ValueError:
        inside = False
    if not inside:
        raise RuntimeError(f"Unsafe archive entry: {member_name}")
    return target_path
