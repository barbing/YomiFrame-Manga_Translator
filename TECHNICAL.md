# YomiFrame - Technical Notes (v1.2.0)

## Overview
YomiFrame is a Windows desktop application for local manga translation. The pipeline detects text regions, OCRs Japanese text, translates it with a local backend, removes the original text, and renders translated text back into the page.

The current system is built around four principles:
- `ComicTextDetector` is the primary detector
- local inference should remain fast enough for volume translation
- proper nouns need cross-page memory, not only per-page prompt hints
- startup should pre-download fixed runtime assets before the first real translation run

Outputs:
- translated images
- `project.json`
- `style_guide.json` when Auto-Glossary is enabled

## High-Level Architecture
- UI: `PySide6`
- Pipeline orchestration: `app/pipeline/controller.py`
- Pre-scan and glossary building: `app/pipeline/prescan.py`
- Detection: `app/detect/*`
- OCR: `app/ocr/*`
- Translation: `app/translate/*`
- Rendering and text cleanup: `app/render/renderer.py`
- NLP / name extraction / alias graph: `app/nlp/*`
- Model download and resolution: `app/models/*`

## Main Runtime Flow
The main runtime flow lives in [controller.py](app/pipeline/controller.py).

Per page:
1. Load image.
2. Detect text regions.
3. OCR text regions.
4. Classify/filter regions.
5. Apply glossary and context-aware translation.
6. Remove original text.
7. Render translated text.
8. Save page output and update `project.json`.

The pipeline also emits UI-facing status messages, progress, ETA, and per-page timing.

## Detection
Primary detector:
- `ComicTextDetector`

Fallback detector:
- `PaddleTextDetector`

Design intent:
- `ComicTextDetector` should be the normal path
- `PaddleTextDetector` exists only as an emergency fallback when CTD is unavailable or crashes

Recent validation work focused on keeping the CTD path stable and avoiding silent degradation into Paddle-only behavior.

## OCR
Primary OCR path:
- `MangaOCR`

Fallback and rescue path:
- `PaddleOCR`

Runtime behavior:
- the app first tries in-process `MangaOCR`
- if that fails, it can use a worker-process fallback
- if MangaOCR is unavailable, it falls back to `PaddleOCR`
- for certain weak or wide OCR cases, Paddle can be used as a targeted rescue comparison instead of replacing MangaOCR globally

Relevant files:
- [manga_ocr_engine.py](app/ocr/manga_ocr_engine.py)
- [manga_ocr_worker.py](app/ocr/manga_ocr_worker.py)
- [manga_ocr_subprocess.py](app/ocr/manga_ocr_subprocess.py)
- [paddle_ocr_recognizer.py](app/ocr/paddle_ocr_recognizer.py)

## Translation Backends
Supported translation backends:
- `Ollama`
- `GGUF` via `llama-cpp-python`

Important product assumption:
- Sakura is treated as a translation model first, not a general extraction model

That means the noun-consistency system is designed to do most discovery with OCR + MeCab + graph logic first, then use the translation model for translation tasks rather than full extraction.

Relevant files:
- [ollama_client.py](app/translate/ollama_client.py)
- [gguf_client.py](app/translate/gguf_client.py)
- [prompts.py](app/translate/prompts.py)

## Auto-Glossary / Name Memory
The current glossary system is no longer just “consistent terms.” It is a lightweight name-memory layer.

Core components:
- [mecab_extractor.py](app/nlp/mecab_extractor.py)
- [character_graph.py](app/nlp/character_graph.py)
- [prescan.py](app/pipeline/prescan.py)

Current behavior:
- extracts likely names and aliases from OCR text
- groups aliases into canonical character/entity nodes
- prefers kanji canonical names when available
- handles many kana nickname variants more conservatively and more consistently
- exports cleaner `style_guide.json` entries

Examples of the kinds of relationships it now handles better:
- full name -> shortened surname
- given-name call forms
- honorific variants
- kana nicknames like `まゆ`, `まゆっち`, `まゆまゆ`

Known limitation:
- if the corpus never contains a canonical reference and only contains kana aliases, the system cannot reliably reconstruct the original kanji name from kana alone
- if a prior `style_guide.json` exists, it can supply that missing mapping

## Pre-Scan
`Build Glossary Before Translation` is the recommended quality mode for chapters and volumes.

Purpose:
- scan OCR text before page translation
- build glossary/name memory up front
- reduce early-page inconsistency

Design constraint:
- the default local workflow still needs to stay fast
- heavier extraction paths remain optional

The current implementation keeps pre-scan lightweight by default and avoids loading a second heavy extraction model unless explicitly requested.

## Experimental Deep Discovery
Deep Discovery remains in the codebase, but it is now treated as experimental.

Intended role:
- difficult edge cases
- developer investigation
- optional higher-cost discovery path

It is not the main noun-consistency strategy anymore.

UI status:
- user-facing wording now marks it as optional and slower
- standard workflow should keep it off

## Rendering
Rendering is handled in [renderer.py](app/render/renderer.py).

Main responsibilities:
- remove original text
- preserve readable bubble structure where possible
- handle dark captions and narration differently from standard speech bubbles
- avoid translating meaningless background/SFX fragments when they should remain original
- fit translated text into narrow or vertical regions without obvious overflow

Recent quality work focused on:
- reducing destructive cleanup
- suppressing junk overlays from bad OCR fragments
- improving dark-caption handling
- improving narrow vertical bubble fitting

The renderer still depends heavily on good upstream region grouping and OCR quality. If segmentation is poor, render-time fixes can only help partially.

## Region Semantics
The pipeline distinguishes different kinds of regions because they should not all be treated like ordinary speech bubbles.

Important classes:
- speech bubbles
- dark captions / narration boxes
- meaningful background text
- SFX / ignorable decorative fragments

This distinction is critical for both translation quality and image quality.

## Project JSON
Each page is saved with region-level structured data. Typical fields include:

```json
{
  "region_id": "r000",
  "bbox": [x, y, w, h],
  "polygon": [[[x, y], [x2, y2]]],
  "type": "speech_bubble|background_text",
  "ocr_text": "...",
  "translation": "...",
  "confidence": {
    "det": 1.0,
    "ocr": 1.0,
    "trans": 1.0
  },
  "render": {
    "font": "Microsoft YaHei",
    "font_size": 0,
    "line_height": 1.2,
    "align": "center"
  },
  "flags": {
    "ignore": false,
    "bg_text": false,
    "needs_review": false
  }
}
```

The JSON is designed for:
- review and re-render workflows
- manual correction
- later consistency checks

## Startup Pre-Downloader
The app checks for fixed runtime dependencies shortly after startup and offers to batch-download missing assets.

Current assets covered:
- `ComicTextDetector`
- `PaddleOCR`
- `MangaOCR`
- `Big-LAMA`
- Japanese NER model

Startup entry:
- [main_window.py](app/ui/main_window.py)

Download manager:
- [downloader.py](app/models/downloader.py)

Shared model resolution:
- [resolution.py](app/models/resolution.py)

Resolution order:
1. system cache / environment cache
2. project-local `models/`

This behavior is intentional:
- developers often already have model caches in their Python environment
- new users still get a portable project-local fallback

Recent hardening work aligned startup checks with runtime model resolution so a model marked “installed” at launch should not later trigger a surprise lazy download during translation.

## Model Resolution
Model resolution is now centralized instead of duplicated across loaders.

Why this matters:
- startup checks and runtime loaders must agree
- otherwise the app can report “ready” and still fail or redownload later

Shared resolver responsibilities:
- MangaOCR system/local lookup
- NER system/local lookup
- Paddle detector and recognizer lookup

Relevant file:
- [resolution.py](app/models/resolution.py)

## UI
Main UI file:
- [main_window.py](app/ui/main_window.py)

Important Home glossary controls:
- `Auto-Glossary (Name Memory)`
- `Build Glossary Before Translation`
- `Experimental Deep Discovery (Optional, slower)`

Current UI behavior:
- Auto-Glossary is the master switch
- if it is off, pre-scan and deep discovery are disabled
- wording now reflects the actual architecture rather than the older glossary design

## Performance
Current target:
- `6 pages under 2 minutes` on the reference PC

Validated recent behavior:
- full-volume GPU run around `15-16s/page`
- recent full-volume validation remained within the user’s runtime budget

The main cost drivers are:
- detector runtime
- OCR runtime
- translation backend/runtime configuration
- page density
- inpainting/render complexity

## Error Handling and Fallbacks
The app is designed to surface errors clearly in the UI and avoid silent failures.

Examples:
- missing MangaOCR dependencies -> readable error and fallback path
- missing detector assets -> user-visible model guidance
- startup model checks -> pre-download prompt before translation begins

The goal is graceful fallback where reasonable, but not silent quality regression.

## Local Data and Repository Conventions
Not tracked in Git:
- `models/`
- `output/`
- `Test Manga/`

Tracked code of interest:
- `app/`
- `README.md`
- `TECHNICAL.md`
- `requirements.txt`

## Running
From repo root:

```powershell
python -m app.main
```

## Verification
Typical validation steps:
- quick syntax check with `python -m py_compile`
- startup smoke run
- focused 6-page timing run
- full-volume quality run when translation/rendering behavior changes materially
