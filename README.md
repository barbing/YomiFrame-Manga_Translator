# YomiFrame

YomiFrame is a Windows desktop app for local manga translation. It detects text, OCRs Japanese dialogue, translates it with local models, removes the original text, and renders readable translated text back into the page.

![Screenshot](assets/screenshot.png)

## What It Does
- Windows GUI built with PySide6
- Batch translation from folder to folder
- `ComicTextDetector` as the primary detector, with `PaddleOCR` fallback only when needed
- OCR with `MangaOCR` and `PaddleOCR`
- Translation with local `Ollama` or local `GGUF` models
- Auto-Glossary / Name Memory for consistent proper nouns and aliases
- Optional `Build Glossary Before Translation` mode for stronger volume-wide consistency
- Experimental Deep Discovery for difficult extraction cases
- Automatic first-launch pre-downloader for fixed runtime model assets
- Project JSON output plus `style_guide.json`

## Current Workflow
For normal use:

1. Choose the input folder and export folder.
2. Select source and target languages.
3. Keep `Auto-Glossary (Name Memory)` enabled.
4. Enable `Build Glossary Before Translation` for chapter or volume runs.
5. Leave `Experimental Deep Discovery` off unless default discovery is insufficient.
6. Start translation.

Recommended default for Japanese -> Simplified Chinese:
- Detector: `ComicTextDetector`
- OCR: `MangaOCR`
- Translator: `GGUF` with Sakura, or `Ollama` if that is your local setup
- Auto-Glossary: `On`
- Build Glossary Before Translation: `On` for full volumes
- Experimental Deep Discovery: `Off`

## Auto-Glossary / Name Memory
The glossary system is no longer just a flat term list. It now builds character/name memory from OCR text and uses that memory during translation.

Highlights:
- prefers canonical kanji names when available
- handles common kana aliases and nickname variants more reliably
- keeps exported `style_guide.json` cleaner and less prone to poisoned entries
- improves cross-page consistency for names, places, and references

Important limitation:
- if a chapter contains only kana aliases and never contains a canonical reference, the app cannot reliably reconstruct the original kanji form on its own
- if the user already has a `style_guide.json` from previous sessions, that guide can still provide the missing canonical mapping

## Pre-Downloader
On startup, the app checks for required fixed runtime assets and offers to download missing ones in one batch before translation starts.

Covered assets:
- `ComicTextDetector`
- `PaddleOCR`
- `MangaOCR`
- `Big-LAMA`
- Japanese NER model

Resolution order:
1. Existing system cache / environment cache
2. Project-local `models/` fallback

This keeps developer environments efficient while still supporting new users who need project-local downloads.

## Models
Local models are not committed to the repository.

Typical layout:

```text
models/
  comic-text-detector/
  manga-ocr/
  paddleocr/
  lama/
  sakura/
  qwen/
```

### GGUF
- Place `.gguf` files under `models/`
- The app scans `models/**.gguf`
- Example:
  - `models/sakura/sakura-14b-qwen3-v1.5-q6k.gguf`

### Ollama
- Install Ollama separately
- Ensure the Ollama server is running
- Use the model dropdown in the UI

## Installation
Environment target:
- Windows 10/11
- Python 3.10
- Conda environment recommended

Install:

```powershell
pip install -r requirements.txt
```

Notes:
- `requirements.txt` keeps CPU-safe defaults where possible
- for GPU acceleration on Windows, install the correct CUDA-enabled `torch`, `paddlepaddle-gpu`, and GPU-capable `llama-cpp-python` build in your own environment as needed

## Run
From the repo root:

```powershell
python -m app.main
```

## Output
Each run can generate:
- translated images in the export folder
- `project.json`
- `style_guide.json` when Auto-Glossary is enabled

`project.json` stores regions, OCR text, translations, and render settings so results can be reviewed and re-applied later.

## Performance
Reference validation on the current local test volume:
- full volume run completed within the target budget
- recent validated run averaged about `15-16s/page` on GPU
- target remains `6 pages under 2 minutes`

Actual speed depends on:
- detector choice
- OCR engine
- translator backend
- GPU availability
- page density

## Troubleshooting
### MangaOCR fails to load
The app will try the normal path first, then worker-process fallback, then `PaddleOCR` if needed.

### ComicTextDetector fails
`ComicTextDetector` is the intended default detector. `PaddleOCR` detector fallback exists only as an emergency path.

### Ollama warning
Start Ollama with:

```powershell
ollama serve
```

Or switch to GGUF.

### Missing models on first run
Allow the startup pre-downloader to fetch the required runtime assets.

### GPU not used for GGUF
Install a CUDA-capable `llama-cpp-python` build and verify your model/backend configuration.

## Repository Notes
- `Test Manga/` is local test data and is not tracked
- `models/` is local and not tracked
- runtime behavior is documented in [TECHNICAL.md](TECHNICAL.md)
- packaging notes are in [BUILD_EXE.md](BUILD_EXE.md)

## License
This project integrates third-party components under their own licenses. See `app/third_party/` for details.
