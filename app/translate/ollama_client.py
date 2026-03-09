# -*- coding: utf-8 -*-
"""Ollama API client."""
from __future__ import annotations
import requests
from typing import Optional

import logging

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url.rstrip("/")

    _availability_cache = {"timestamp": 0.0, "status": False}

    def is_available(self, timeout: int = 5) -> bool:
        import time
        now = time.time()
        
        # Check cache (limit checks to once every 3 seconds globally)
        if now - OllamaClient._availability_cache["timestamp"] < 3.0:
            return OllamaClient._availability_cache["status"]

        url = f"{self._base_url}/api/tags"
        available = False
        try:
            response = requests.get(url, timeout=timeout)
            available = response.status_code == 200
            if not available:
                logger.debug(f"Ollama check failed. Status: {response.status_code}")
        except requests.RequestException as e:
            # Downgrade to debug to prevent spam
            logger.debug(f"Ollama unavailable: {e}")
            available = False
            
        # Update cache
        OllamaClient._availability_cache = {"timestamp": now, "status": available}
        return available

    def generate(self, model: str, prompt: str, timeout: int = 600, options: Optional[dict] = None) -> str:
        url = f"{self._base_url}/api/generate"
        default_options = {"temperature": 0.2}
        if options:
            default_options.update(options)
        payload = {"model": model, "prompt": prompt, "stream": False, "options": default_options}
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Ollama generation success for model {model}")
        return str(data.get("response", "")).strip()
    def translate_glossary(self, terms: list[str], source_lang: str, target_lang: str) -> dict[str, str]:
        """Translate a batch of terms using the LLM. Requires self.model_name to be set."""
        if not terms:
            return {}
        
        # Default model if not set (fallback)
        model = getattr(self, "model_name", "sakura")
        
        results = {}
        batch_size = 20
        is_zh = target_lang in ["Simplified Chinese", "Traditional Chinese", "zh", "zh-CN", "zh-TW"]
        
        for i in range(0, len(terms), batch_size):
            chunk = terms[i:i+batch_size]
            
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
                response = self.generate(
                    model=model,
                    prompt=prompt_text,
                    options={"num_predict": 1024, "temperature": 0.1}
                )
                
                # Parse JSON
                clean_response = response
                if "```json" in clean_response:
                    clean_response = clean_response.split("```json")[1].split("```")[0]
                elif "```" in clean_response:
                    clean_response = clean_response.split("```")[1].split("```")[0]
                    
                import json
                try:
                    chunk_map = json.loads(clean_response)
                    if isinstance(chunk_map, dict):
                        results.update(chunk_map)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse glossary JSON (Ollama): {clean_response[:50]}...")
                    pass
            except Exception as e:
                logger.error(f"Error translating glossary chunk (Ollama): {e}")
                
        # Clean results
        final_map = {}
        for term in terms:
            if term in results:
                final_map[term] = str(results[term]).strip()
        
        return final_map
