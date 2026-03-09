# -*- coding: utf-8 -*-
"""GGUF translation backend using llama-cpp-python."""
from __future__ import annotations
from functools import lru_cache
import os
import struct
import threading
from typing import Optional

import logging

logger = logging.getLogger(__name__)


def _normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _llama_cpp_version() -> str:
    try:
        import llama_cpp
        return str(getattr(llama_cpp, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _read_gguf_architecture(model_path: str) -> str:
    """Best-effort parser for GGUF metadata to read general.architecture."""
    GGUF_MAGIC = b"GGUF"
    GGUF_TYPE_ARRAY = 9
    GGUF_TYPE_STRING = 8

    def _read_exact(fh, n: int) -> bytes:
        data = fh.read(n)
        if len(data) != n:
            raise ValueError("Unexpected EOF")
        return data

    def _read_u32(fh) -> int:
        return struct.unpack("<I", _read_exact(fh, 4))[0]

    def _read_u64(fh) -> int:
        return struct.unpack("<Q", _read_exact(fh, 8))[0]

    def _read_bool(fh) -> bool:
        return _read_exact(fh, 1) != b"\x00"

    def _read_string(fh) -> str:
        length = _read_u64(fh)
        raw = _read_exact(fh, length)
        return raw.decode("utf-8", errors="replace")

    def _skip_scalar(fh, value_type: int) -> None:
        scalar_sizes = {
            0: 1,   # uint8
            1: 1,   # int8
            2: 2,   # uint16
            3: 2,   # int16
            4: 4,   # uint32
            5: 4,   # int32
            6: 4,   # float32
            7: 1,   # bool
            10: 8,  # uint64
            11: 8,  # int64
            12: 8,  # float64
        }
        size = scalar_sizes.get(value_type)
        if size is None:
            raise ValueError(f"Unsupported GGUF value type: {value_type}")
        _read_exact(fh, size)

    def _skip_value(fh, value_type: int) -> None:
        if value_type == GGUF_TYPE_STRING:
            _read_string(fh)
            return
        if value_type == GGUF_TYPE_ARRAY:
            elem_type = _read_u32(fh)
            count = _read_u64(fh)
            if elem_type == GGUF_TYPE_STRING:
                for _ in range(count):
                    _read_string(fh)
            else:
                scalar_sizes = {
                    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8
                }
                elem_size = scalar_sizes.get(elem_type)
                if elem_size is None:
                    raise ValueError(f"Unsupported GGUF array type: {elem_type}")
                _read_exact(fh, elem_size * count)
            return
        _skip_scalar(fh, value_type)

    try:
        with open(model_path, "rb") as fh:
            if _read_exact(fh, 4) != GGUF_MAGIC:
                return ""
            version = _read_u32(fh)
            if version not in (2, 3):
                return ""
            _tensor_count = _read_u64(fh)
            metadata_count = _read_u64(fh)
            for _ in range(metadata_count):
                key = _read_string(fh)
                value_type = _read_u32(fh)
                if key == "general.architecture" and value_type == GGUF_TYPE_STRING:
                    return _read_string(fh)
                _skip_value(fh, value_type)
    except Exception:
        return ""
    return ""


@lru_cache(maxsize=2)
def _load_model(
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    n_threads: int,
    n_batch: int,
):
    try:
        from llama_cpp import Llama
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.error(f"Failed to import llama-cpp-python: {exc}")
        raise RuntimeError("llama-cpp-python is not installed") from exc
    logger.info(f"Loading GGUF Model: {model_path} (n_ctx={n_ctx}, layers={n_gpu_layers})")
    try:
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_batch=n_batch,
            verbose=False,
        )
    except Exception as exc:
        arch = _read_gguf_architecture(model_path)
        version = _llama_cpp_version()
        if arch:
            raise RuntimeError(
                f"Failed to load model from file: {model_path}. "
                f"Detected GGUF architecture '{arch}'. "
                f"Current llama-cpp-python version: {version}. "
                f"This runtime may not support '{arch}'. "
                "Try: pip install -U llama-cpp-python"
            ) from exc
        raise RuntimeError(
            f"Failed to load model from file: {model_path}. "
            f"Current llama-cpp-python version: {version}. "
            "The file may be incompatible/corrupt, or the runtime may be outdated. "
            "Try re-downloading the model and upgrading llama-cpp-python."
        ) from exc


def clear_gguf_cache() -> None:
    _load_model.cache_clear()


def _wrap_prompt(prompt: str, style: str) -> str:
    if "<|im_start|>" in prompt or "Human:" in prompt and "Assistant:" in prompt:
        return prompt
    style = (style or "plain").lower()
    if style == "sakura":
        system = (
            "你是一个日本二次元领域的日语翻译模型，可以流畅通顺地以日本轻小说/漫画/Galgame的风格将日文翻译成简体中文，"
            "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
            "只输出译文或JSON，不要附加说明。"
        )
        return (
            "<|im_start|>system\n"
            f"{system}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if style == "qwen":
        system = "You are a translation engine. Output only the translation or JSON."
        return (
            "<|im_start|>system\n"
            f"{system}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if style == "extract":
        system = "You are an information extraction engine. Output only JSON. Follow the user's instructions."
        return (
            "<|im_start|>system\n"
            f"{system}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return prompt


class GGUFClient:
    def __init__(
        self,
        model_path: str,
        prompt_style: str = "qwen",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int = 8,
        n_batch: int = 512,
    ) -> None:
        self._model_path = _normalize_path(model_path)
        if not os.path.isfile(self._model_path):
            raise RuntimeError(f"GGUF model not found: {self._model_path}")
        self._prompt_style = prompt_style
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_threads = n_threads
        self._n_batch = n_batch
        self.gpu_offload = False
        try:
            from llama_cpp import llama_cpp
            if hasattr(llama_cpp, "llama_supports_gpu_offload"):
                has_gpu = bool(llama_cpp.llama_supports_gpu_offload())
            else:
                has_gpu = hasattr(llama_cpp, "llama_cuda_init") or hasattr(llama_cpp, "llama_gpu_init")
            self.gpu_offload = has_gpu and self._n_gpu_layers != 0
            if has_gpu:
                 logger.info("GGUF: GPU acceleration available/enabled.")
        except Exception:
            logger.warning("GGUF: Failed to check/init GPU support.")
            self.gpu_offload = False
        if self._n_gpu_layers == -1:
             self._n_gpu_layers = 100 if self.gpu_offload else 0
             
        self._llama = _load_model(
            self._model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            n_threads=self._n_threads,
            n_batch=self._n_batch,
        )
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return os.path.isfile(self._model_path)
        
    def is_gpu_enabled(self) -> bool:
        return self.gpu_offload and self._n_gpu_layers > 0

    def generate(self, model: str, prompt: str, timeout: int = 600, options: Optional[dict] = None) -> str:
        with self._lock:
            opts = {"temperature": 0.2, "top_p": 0.95}
            if options:
                opts.update(options)
            max_tokens = int(opts.pop("num_predict", 256))
            stop = opts.pop("stop", None)
            if stop is None:
                stop = ["<|im_end|>", "<|im_start|>", "###", "Instruction:", "System:", "User:"]
            prompt_wrapped = _wrap_prompt(prompt, self._prompt_style)
            result = self._llama(
                prompt_wrapped,
                max_tokens=max_tokens,
                stop=stop,
                **opts,
            )
            # logger.debug(f"GGUF Generation stats: {result.get('usage', {})}") # Optional: log token usage
            choices = result.get("choices", [])
            if not choices:
                return ""
            return str(choices[0].get("text", "")).strip()

    def close(self) -> None:
        try:
            self._llama = None
            clear_gguf_cache()
        except Exception:
            pass

    def translate_glossary(self, terms: list[str], source_lang: str, target_lang: str) -> dict[str, str]:
        """Translate a batch of terms (glossary) using the LLM. Returns source->target map."""
        if not terms:
            return {}
        
        # Batching (chunk by 20 to avoid context overflow)
        results = {}
        batch_size = 20
        
        # Determine prompt language
        is_zh = target_lang in ["Simplified Chinese", "Traditional Chinese", "zh", "zh-CN", "zh-TW"]
        
        for i in range(0, len(terms), batch_size):
            chunk = terms[i:i+batch_size]
            
            # Build Prompt
            if is_zh:
                prompt_text = (
                    f"将以下日文人名列表翻译成中文。\n"
                    f"请输出严格的JSON格式，格式为 {{\"日文原名\": \"中文译名\"}}。\n"
                    f"不要添加任何解释或Markdown标记。\n\n"
                    f"待翻译列表：\n" + "\n".join([f"- {t}" for t in chunk])
                )
            else:
                prompt_text = (
                    f"Translate the following Japanese names to {target_lang}.\n"
                    f"Output strictly valid JSON format: {{\"Source Name\": \"Translated Name\"}}.\n"
                    f"No explanations or markdown.\n\n"
                    f"List:\n" + "\n".join([f"- {t}" for t in chunk])
                )

            try:
                # Generate
                response = self.generate(
                    model=self._model_path,
                    prompt=prompt_text,
                    options={"num_predict": 1024, "temperature": 0.1}
                )
                
                # Parse JSON
                # Parse JSON with robust recovery
                import json
                import re
                
                clean_response = response.strip()
                
                # 1. Try to find JSON block using simple outer brace matching
                # Python re doesn't support recursion (?R), so we use find/rfind
                start = clean_response.find('{')
                end = clean_response.rfind('}')
                
                chunk_map = {}
                success = False
                
                if start != -1 and end != -1 and end > start:
                    potential_json = clean_response[start:end+1]
                    try:
                        chunk_map = json.loads(potential_json)
                        success = True
                    except json.JSONDecodeError:
                        # Try to clean up common issues like trailing commas
                        try:
                            # Remove trailing commas before }
                            fixed_json = re.sub(r',(\s*\})', r'\1', potential_json)
                            chunk_map = json.loads(fixed_json)
                            success = True
                        except:
                            pass
                
                if success and isinstance(chunk_map, dict):
                    results.update(chunk_map)
                else:
                    logger.warning(f"Failed to parse glossary JSON: {clean_response[:50]}...")
                    # Fallback: simple line parsing (Key: Value or Key=Value)
                    # This catches Sakura's tendency to output "Src=Tgt" lines if strict JSON fails
                    for line in clean_response.split('\n'):
                        if ':' in line:
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                k = parts[0].strip(' "-')
                                v = parts[1].strip(' ",')
                                if k in terms:
                                    results[k] = v
                        elif '=' in line:
                            parts = line.split('=', 1)
                            if len(parts) == 2:
                                k = parts[0].strip(' "-')
                                v = parts[1].strip(' ",')
                                if k in terms:
                                    results[k] = v
                    
            except Exception as e:
                logger.error(f"Error translating glossary chunk: {e}")
                
        # Clean results (ensure keys match input terms)
        final_map = {}
        for term in terms:
            if term in results:
                final_map[term] = str(results[term]).strip()
        
        return final_map
