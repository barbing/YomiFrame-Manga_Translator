# -*- coding: utf-8 -*-
"""Default settings."""
from dataclasses import dataclass

# Model URLs
COMIC_TEXT_DETECTOR_GPU = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt"
COMIC_TEXT_DETECTOR_CPU = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt.onnx"
SAKURA_GGUF = "https://huggingface.co/SakuraLLM/Sakura-14B-Qwen3-v1.5-GGUF/resolve/main/sakura-14b-qwen3-v1.5-q6k.gguf"
QWEN_GGUF = "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q6_K.gguf"
BIG_LAMA = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
MANGA_OCR_BASE_URL = "https://huggingface.co/kha-white/manga-ocr-base/resolve/main/"
MANGA_OCR_FILES = [
    "config.json",
    "preprocessor_config.json", 
    "pytorch_model.bin",
    "special_tokens_map.json",
    "vocab.txt",
    "tokenizer_config.json"
]


@dataclass
class AppDefaults:
    source_language: str = "Japanese"
    target_language: str = "Simplified Chinese"
    output_suffix: str = "_translated"
    theme: str = "dark"
    json_path: str = ""
    import_dir: str = ""
    export_dir: str = ""
    font_name: str = "Microsoft YaHei"
    font_detection: str = "heuristic"
    detector_input_size: int = 1024
    detector_engine: str = "ComicTextDetector"
    ocr_engine: str = "MangaOCR"
    filter_strength: str = "normal"
    inpaint_mode: str = "ai"
    # Available models: 
    # - "dreMaz/AnimeMangaInpainting" (default, anime-focused)
    # - "runwayml/stable-diffusion-inpainting" (general, slower)
    inpaint_model: str = "dreMaz/AnimeMangaInpainting"
    translator_backend: str = "Ollama"
    gguf_model_path: str = ""
    gguf_prompt_style: str = "sakura"
    gguf_n_ctx: int = 4096
    gguf_n_gpu_layers: int = -1
    gguf_n_threads: int = 8
    gguf_n_batch: int = 256
    fast_mode: bool = False
    auto_glossary: bool = True
    
    # Generation Options
    ollama_temperature: float = 0.2
    ollama_top_p: float = 0.9
    ollama_context: int = 4096
    gguf_temperature: float = 0.2
    gguf_top_p: float = 0.95


def get_defaults() -> AppDefaults:
    return AppDefaults()
