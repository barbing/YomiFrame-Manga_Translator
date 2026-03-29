# -*- coding: utf-8 -*-
"""NER-based name extractor using bert-ner-japanese for accurate Japanese Name Entity Recognition.

This module provides a transformer-based NER extractor that can detect:
- Person names (人名)
- Organization names (組織名)
- Location names (地名)

Falls back to MeCab if the BERT model is unavailable.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple
from app.models.resolution import resolve_ner_local_dir, resolve_ner_system_snapshot

logger = logging.getLogger(__name__)

# Model configuration
NER_MODEL_ID = "jurabi/bert-ner-japanese"
NER_MODEL_SIZE_MB = 420  # Approximate size for download progress


@dataclass
class NEREntity:
    """An entity extracted by the NER model."""
    text: str           # The entity surface form
    label: str          # Entity type: PER, ORG, LOC, etc.
    start: int          # Start position in text
    end: int            # End position in text
    confidence: float   # Model confidence score


class NERExtractor:
    """
    Named Entity Recognition extractor using bert-ner-japanese.
    
    This provides higher accuracy than MeCab for detecting proper nouns,
    especially for:
    - Complex multi-part names
    - Foreign names written in katakana
    - Unusual or rare names
    
    Falls back to MeCab if the NER model cannot be loaded.
    
    Usage:
        extractor = NERExtractor()
        if extractor.is_available():
            entities = extractor.extract_names("佐藤太郎は東京に住んでいます")
            # [NEREntity(text='佐藤太郎', label='PER', ...)]
    """
    
    def __init__(self, model_dir: Optional[str] = None, force_cpu: bool = False):
        """
        Initialize the NER extractor.
        
        Args:
            model_dir: Directory to cache the model (defaults to models/ner/)
            force_cpu: If True, forces CPU inference even if GPU available
        """
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._available = False
        self._fallback_mecab = None
        
        if resolve_ner_system_snapshot():
            self._model_dir = None
        else:
            if model_dir is None:
                app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                model_dir = os.path.join(app_root, "models", "ner")
            self._model_dir = resolve_ner_local_dir(model_dir) or model_dir
        self._force_cpu = force_cpu
        
        # Try to initialize
        self._init_model()
    
    def _init_model(self) -> None:
        """Attempt to load the NER model."""
        try:
            # Check for transformers
            from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
            import torch
            
            # Determine device
            device = -1  # CPU
            if not self._force_cpu:
                if torch.cuda.is_available():
                    device = 0
                    logger.info("NER: Using CUDA GPU")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = 0  # MPS uses device 0
                    logger.info("NER: Using Apple MPS")
            
            # Load model with caching
            cache_dir = self._model_dir if (self._model_dir and os.path.exists(self._model_dir)) else None
            
            logger.info(f"NER: Loading model '{NER_MODEL_ID}'...")
            
            # Explicitly load model and tokenizer to handle cache_dir correctly
            tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_ID, cache_dir=cache_dir)
            model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_ID, cache_dir=cache_dir)
            
            self._pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=device,
            )
            
            self._available = True
            logger.info("NER: Model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"NER: transformers not available: {e}")
            self._init_mecab_fallback()
            
        except Exception as e:
            logger.warning(f"NER: Failed to load model: {e}")
            self._init_mecab_fallback()
    
    def _init_mecab_fallback(self) -> None:
        """Initialize MeCab as fallback."""
        try:
            from app.nlp.mecab_extractor import MeCabExtractor
            self._fallback_mecab = MeCabExtractor()
            logger.info("NER: Using MeCab fallback")
        except Exception as e:
            logger.warning(f"NER: MeCab fallback also failed: {e}")
    
    def is_available(self) -> bool:
        """Check if NER model or fallback is available."""
        return self._available or (self._fallback_mecab is not None 
                                    and self._fallback_mecab.is_available())
    
    def is_ner_model_loaded(self) -> bool:
        """Check if the actual NER model is loaded (not fallback)."""
        return self._available
    
    def unload(self) -> None:
        """Unload the model from memory to free up resources."""
        if self._pipeline:
            del self._pipeline
            self._pipeline = None
        
        self._available = False
        
        # Clear CUDA cache if possible
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                logger.info("NER: Model unloaded and CUDA cache cleared")
        except Exception as e:
            logger.warning(f"NER: Failed to clear cache: {e}")
    
    def extract_entities(self, text: str) -> List[NEREntity]:
        """
        Extract all named entities from text.
        
        Args:
            text: Japanese text to analyze
            
        Returns:
            List of NEREntity objects for all detected entities
        """
        if not text or not text.strip():
            return []
        
        if self._available and self._pipeline:
            return self._extract_with_bert(text)
        elif self._fallback_mecab:
            return self._extract_with_mecab_fallback(text)
        else:
            return []
    
    def extract_names(self, text: str) -> List[NEREntity]:
        """
        Extract only person names (PER entities) from text.
        
        Args:
            text: Japanese text to analyze
            
        Returns:
            List of NEREntity objects for detected person names
        """
        entities = self.extract_entities(text)
        return [e for e in entities if e.label in ("PER", "人名", "PERSON")]
    
    def _extract_with_bert(self, text: str) -> List[NEREntity]:
        """Extract entities using the BERT NER model."""
        try:
            results = self._pipeline(text)
            entities = []
            
            for result in results:
                # Handle different output formats
                if isinstance(result, dict):
                    entity = NEREntity(
                        text=result.get("word", "").replace("##", ""),
                        label=result.get("entity_group", result.get("entity", "UNK")),
                        start=result.get("start", 0),
                        end=result.get("end", 0),
                        confidence=result.get("score", 0.0)
                    )
                    # Filter low-confidence results
                    if entity.confidence > 0.5 and len(entity.text) > 0:
                        entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return []
    
    def _extract_with_mecab_fallback(self, text: str) -> List[NEREntity]:
        """Extract entities using MeCab as fallback."""
        if not self._fallback_mecab:
            return []
            
        try:
            names = self._fallback_mecab.extract_proper_nouns(text)
            entities = []
            
            for name in names:
                # Map MeCab pos to NER labels
                label_map = {
                    "人名": "PER",
                    "地名": "LOC", 
                    "組織名": "ORG",
                    "人名_外来": "PER",
                    "人名_推測": "PER",
                }
                label = label_map.get(name.pos, "PER")
                
                entities.append(NEREntity(
                    text=name.surface,
                    label=label,
                    start=text.find(name.surface),
                    end=text.find(name.surface) + len(name.surface),
                    confidence=0.7  # Lower confidence for MeCab
                ))
            
            return entities
            
        except Exception as e:
            logger.error(f"MeCab fallback failed: {e}")
            return []
    
    def to_extracted_names(self, entities: List[NEREntity]) -> list:
        """
        Convert NER entities to ExtractedName format for compatibility.
        
        Args:
            entities: List of NEREntity objects
            
        Returns:
            List of ExtractedName objects
        """
        from app.nlp.mecab_extractor import ExtractedName
        
        return [
            ExtractedName(
                surface=e.text,
                reading=e.text,  # NER doesn't provide reading
                pos=e.label
            )
            for e in entities
        ]


def check_ner_model_available() -> Tuple[bool, str]:
    """
    Check if NER model can be loaded.
    
    Returns:
        Tuple of (available, message)
    """
    try:
        from transformers import pipeline
        return True, "transformers library available"
    except ImportError:
        return False, "transformers library not installed"


def download_ner_model(model_dir: Optional[str] = None, progress_callback=None) -> bool:
    """
    Download the NER model for offline use.
    
    Args:
        model_dir: Target directory for the model (cache_dir)
        progress_callback: Optional callback(percent: int) for progress updates
        
    Returns:
        True if successful
    """
    try:
        from huggingface_hub import snapshot_download

        if progress_callback:
            progress_callback(10)

        logger.info(f"Downloading NER model: {NER_MODEL_ID} to {model_dir or 'default cache'}")

        # Ensure directory exists if specified
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        if progress_callback:
            progress_callback(30)

        # Use Hub snapshot download so startup pre-download does not instantiate the model.
        snapshot_download(repo_id=NER_MODEL_ID, cache_dir=model_dir)

        if progress_callback:
            progress_callback(100)

        logger.info("NER model download complete")
        return True

    except ImportError:
        # Fallback for environments where huggingface_hub is unavailable.
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            if progress_callback:
                progress_callback(10)

            logger.info(f"Downloading NER model (fallback): {NER_MODEL_ID} to {model_dir or 'default cache'}")
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)

            AutoTokenizer.from_pretrained(NER_MODEL_ID, cache_dir=model_dir)
            if progress_callback:
                progress_callback(50)
            AutoModelForTokenClassification.from_pretrained(NER_MODEL_ID, cache_dir=model_dir)
            if progress_callback:
                progress_callback(100)

            logger.info("NER model download complete")
            return True
        except Exception as e:
            logger.error(f"Failed to download NER model: {e}")
            return False
    except Exception as e:
        logger.error(f"Failed to download NER model: {e}")
        return False
