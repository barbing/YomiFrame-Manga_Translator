# -*- coding: utf-8 -*-
"""
Text filtering logic for the pipeline.
Responsible for deciding which text regions should be translated vs ignored.
"""
from __future__ import annotations
import re

class TextFilter:
    """
    Centralized filter for determining if text should be processed or ignored.
    Encapsulates heuristics like SFX detection, punctuation checks, and script ratio.
    """
    
    def __init__(self, settings=None):
        self.settings = settings

    def should_ignore(self, text: str, region_type: str = "background_text") -> bool:
        """
        Decide if a text should be ignored (not translated).
        
        Args:
            text: The OCR text content.
            region_type: The type of region (e.g., 'speech_bubble', 'background_text').
            
        Returns:
            True if the text should be ignored, False if it should be translated.
        """
        if not text:
            return True

        # 1. Basic Garbage Check
        if self._is_punct_only(text):
            return True
            
        # 2. SFX Filter Heuristic
        # Only apply strict SFX filtering to background text or ambiguous regions.
        # Bubbles are usually safe, but sometimes the detector misclassifies SFX as bubbles.
        # So we apply a smart filter:
        # If it matches SFX patterns AND has NO punctuation AND is mostly Katakana -> Ignore.
        if self._looks_like_sfx(text):
            has_punct = any(ch in text for ch in "、。！？!.?…~～")
            if not has_punct:
                # "Purupuru" (Katakana) -> Ignore
                # "Hai hai" (Hiragana) -> Keep (Dialogue)
                if self._is_mostly_katakana(text):
                    return True
                    
        return False

    def _looks_like_sfx(self, text: str) -> bool:
        """
        Check if text resembles Sound Effects (SFX).
        """
        if not text:
            return False
        trimmed = text.strip()
        if not trimmed or len(trimmed) > 6:
            return False
            
        # Common non-SFX short words
        common_words = {
            "うん", "うう", "ええ", "はい", "いや", "いいえ",
            "え", "あ", "おい", "ね", "な", "うそ",
        }
        if trimmed in common_words:
            return False
            
        # Repetition check (e.g., PukuPuku)
        if len(trimmed) % 2 == 0:
            half = len(trimmed) // 2
            if trimmed[:half] == trimmed[half:]:
                return True
        
        # Short text with Katakana small tsu 'ッ' or long vowel 'ー' is likely SFX
        if len(trimmed) <= 4:
             if "ッ" in trimmed or "ー" in trimmed:
                 return True
                 
        # Repeated same char (e.g., ざざ, どど)
        # BUT exempt common dialogue expressions (アアア, イイイ, ハハハ, etc.)
        # These are valid emphasis/laughter sounds in dialogue
        if len(set(trimmed)) <= 1:
            dialogue_vowels = {"ア", "イ", "ウ", "エ", "オ", "ハ", "あ", "い", "う", "え", "お", "は"}
            if trimmed[0] in dialogue_vowels:
                return False  # Valid dialogue expression
            return True
            
        return False

    def _is_punct_only(self, text: str) -> bool:
        """Check if text is only punctuation."""
        return not any(ch.isalnum() or self._is_japanese(ch) for ch in text)

    def _is_japanese(self, ch: str) -> bool:
        """Check if char is Japanese (Kana/Kanji)."""
        code = ord(ch)
        return (0x3040 <= code <= 0x30FF) or (0x4E00 <= code <= 0x9FFF)

    def _is_mostly_katakana(self, text: str) -> bool:
        """Check if text is primarily Katakana (0x30A0 - 0x30FF)."""
        if not text:
            return False
        katakana_count = sum(1 for ch in text if 0x30A0 <= ord(ch) <= 0x30FF)
        return (katakana_count / len(text)) > 0.5
