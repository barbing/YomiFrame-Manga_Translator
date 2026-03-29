# -*- coding: utf-8 -*-
"""MeCab-based Japanese NLP extractor for proper noun extraction and alias grouping."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractedName:
    """A proper noun extracted from Japanese text."""
    surface: str      # 黛由紀江
    reading: str      # まゆずみゆきえ
    pos: str          # 人名 / 地名 / 組織名


@dataclass
class AliasGroup:
    """A group of names that refer to the same entity."""
    canonical: str              # 黛由紀江
    canonical_reading: str      # まゆずみゆきえ
    aliases: list[dict] = field(default_factory=list)  # [{"source": "まゆちゃん", "pattern": "chan", "hint": "..."}]


# Default suffix configuration (bundled with app)
DEFAULT_SUFFIXES = {
    # Affectionate/Cute
    "ちゃん": {"pattern": "chan", "hint": "亲昵的称呼"},
    "たん": {"pattern": "tan", "hint": "可爱的称呼"},
    "っち": {"pattern": "cchi", "hint": "亲昵的称呼"},
    "りん": {"pattern": "rin", "hint": "可爱的称呼"},
    "ぴょん": {"pattern": "pyon", "hint": "可爱的称呼"},
    
    # Formal/Polite
    "さん": {"pattern": "san", "hint": "礼貌的称呼"},
    "さま": {"pattern": "sama", "hint": "尊敬的称呼"},
    "様": {"pattern": "sama", "hint": "尊敬的称呼"},
    "氏": {"pattern": "shi", "hint": "正式称呼"},
    
    # Gender-based
    "くん": {"pattern": "kun", "hint": "对男性的称呼"},
    "君": {"pattern": "kun", "hint": "对男性的称呼"},
    "坊": {"pattern": "bou", "hint": "对小男孩的称呼"},
    "嬢": {"pattern": "jou", "hint": "对年轻女性的称呼"},
    "お嬢様": {"pattern": "ojousama", "hint": "对年轻小姐的尊称"},
    
    # Professional/Title (kanji forms)
    "先生": {"pattern": "sensei", "hint": "老师/医生/专家"},
    "先輩": {"pattern": "senpai", "hint": "前辈"},
    "後輩": {"pattern": "kouhai", "hint": "后辈"},
    "教授": {"pattern": "kyouju", "hint": "教授"},
    "博士": {"pattern": "hakase", "hint": "博士"},
    "部長": {"pattern": "buchou", "hint": "部长"},
    "課長": {"pattern": "kachou", "hint": "课长"},
    "社長": {"pattern": "shachou", "hint": "社长"},
    "会長": {"pattern": "kaichou", "hint": "会长"},
    "隊長": {"pattern": "taichou", "hint": "队长"},
    "団長": {"pattern": "danchou", "hint": "团长"},
    "艦長": {"pattern": "kanchou", "hint": "舰长"},
    "店長": {"pattern": "tenchou", "hint": "店长"},
    
    # Professional/Title (hiragana forms for reading matches)
    "せんせい": {"pattern": "sensei", "hint": "老师/医生/专家"},
    "せんぱい": {"pattern": "senpai", "hint": "前辈"},
    "こうはい": {"pattern": "kouhai", "hint": "后辈"},
    "きょうじゅ": {"pattern": "kyouju", "hint": "教授"},
    "はかせ": {"pattern": "hakase", "hint": "博士"},
    
    # Family
    "兄": {"pattern": "nii", "hint": "哥哥"},
    "姉": {"pattern": "nee", "hint": "姐姐"},
    "にい": {"pattern": "nii", "hint": "哥哥"},
    "ねえ": {"pattern": "nee", "hint": "姐姐"},
    "おにいちゃん": {"pattern": "oniichan", "hint": "哥哥（亲昵）"},
    "おねえちゃん": {"pattern": "oneechan", "hint": "姐姐（亲昵）"},
    "おにいさん": {"pattern": "oniisan", "hint": "哥哥（礼貌）"},
    "おねえさん": {"pattern": "oneesan", "hint": "姐姐（礼貌）"},
    
    # Archaic/Noble
    "殿": {"pattern": "dono", "hint": "殿下（古风）"},
    "卿": {"pattern": "kyou", "hint": "卿（古风）"},
    "姫": {"pattern": "hime", "hint": "公主"},
    "王": {"pattern": "ou", "hint": "国王"},
    "殿下": {"pattern": "denka", "hint": "殿下"},
    "陛下": {"pattern": "heika", "hint": "陛下"},
    
    # Informal
    "親分": {"pattern": "oyabun", "hint": "老大/大哥"},
    "兄貴": {"pattern": "aniki", "hint": "大哥（黑道）"},
    "姐御": {"pattern": "aneki", "hint": "大姐（黑道）"},
}


class MeCabExtractor:
    """
    Japanese NLP extractor using MeCab for proper noun extraction and alias grouping.
    
    Uses fugashi (MeCab wrapper) to:
    1. Extract proper nouns (固有名詞) with readings
    2. Detect naming patterns (suffixes, reduplication)
    3. Group aliases by reading substring matching
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the MeCab extractor.
        
        Args:
            config_path: Optional path to user-defined suffixes.json
        """
        try:
            import fugashi
            self.tagger = fugashi.Tagger()
            self._mecab_available = True
        except ImportError:
            self.tagger = None
            self._mecab_available = False
        except Exception:
            self.tagger = None
            self._mecab_available = False
        
        # Load suffix configuration
        self.suffixes = dict(DEFAULT_SUFFIXES)
        
        # Load user extensions if available
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    user_suffixes = json.load(f)
                    # Filter out comment keys
                    user_suffixes = {k: v for k, v in user_suffixes.items() if not k.startswith("_")}
                    self.suffixes.update(user_suffixes)
            except Exception:
                pass  # Use defaults on error
        
        # Pre-sort suffixes by length (longest first) for matching
        self._sorted_suffixes = sorted(self.suffixes.keys(), key=len, reverse=True)

        # Extra extraction controls (optional terms.json alongside suffixes.json)
        self.whitelist_terms = set()
        self.blacklist_terms = set()
        self.title_suffixes = {
            "王", "姫", "皇", "王女", "王子",
            "魔王", "勇者", "魔女", "剣士", "騎士",
            "隊長", "団長", "副隊長", "班長",
            "会長", "社長", "部長", "課長", "主任",
            "師匠", "先生", "博士", "教授",
            "総長", "監督", "長", "司令", "司令官",
        }
        self.title_terms = {
            "魔王", "勇者", "姫", "王女", "王子", "皇女", "皇子",
            "魔女", "剣聖", "聖女", "賢者", "騎士団", "魔王軍",
        }
        self.kana_alias_blacklist = {
            "これ", "それ", "あれ", "どれ",
            "ここ", "そこ", "あそこ", "どこ",
            "だれ", "なに", "いつ", "どう",
            "はい", "いいえ", "うん", "ええ",
            "ああ", "えー", "うー", "へえ",
        }

        if config_path:
            terms_path = os.path.join(os.path.dirname(config_path), "terms.json")
            if os.path.exists(terms_path):
                try:
                    with open(terms_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.whitelist_terms.update(data.get("whitelist", []) or [])
                        self.blacklist_terms.update(data.get("blacklist", []) or [])
                        self.title_suffixes.update(data.get("title_suffixes", []) or [])
                        self.title_terms.update(data.get("title_terms", []) or [])
                        self.kana_alias_blacklist.update(data.get("kana_alias_blacklist", []) or [])
                except Exception:
                    pass

    def is_available(self) -> bool:
        """Check if MeCab is available."""
        return self._mecab_available
    
    def detect_pattern(self, name: str, reading: str) -> tuple[str, str, str]:
        """
        Detect naming pattern and extract base reading.
        
        Args:
            name: The surface form (e.g., "まゆちゃん")
            reading: The reading in hiragana/katakana
            
        Returns:
            Tuple of (pattern_type, base_reading, hint_for_translation)
        """
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
        relaxed_name = str(name or "")
        relaxed_reading = str(reading or "")
        for variant, normalized in relaxed_variants.items():
            if relaxed_name.endswith(variant):
                relaxed_name = f"{relaxed_name[:-len(variant)]}{normalized}"
            if relaxed_reading.endswith(variant):
                relaxed_reading = f"{relaxed_reading[:-len(variant)]}{normalized}"

        # Check for known suffixes (longest match first)
        for suffix in self._sorted_suffixes:
            # Check if name or reading ends with this suffix
            if name.endswith(suffix):
                info = self.suffixes[suffix]
                base = reading[:-len(suffix)] if reading.endswith(suffix) else reading
                return info["pattern"], base, info.get("hint", "")
            elif reading.endswith(suffix):
                info = self.suffixes[suffix]
                base = reading[:-len(suffix)]
                return info["pattern"], base, info.get("hint", "")
            elif relaxed_name.endswith(suffix) or relaxed_reading.endswith(suffix):
                info = self.suffixes[suffix]
                base = relaxed_reading[:-len(suffix)] if relaxed_reading.endswith(suffix) else relaxed_reading
                return info["pattern"], base, info.get("hint", "")
        
        # Check for reduplication (まゆまゆ → まゆ)
        if len(reading) >= 4 and len(reading) % 2 == 0:
            half = len(reading) // 2
            if reading[:half] == reading[half:]:
                return "reduplication", reading[:half], "可爱/亲昵的叫法"
        
        # Plain name (no known suffix detected)
        # Don't guess unknown suffixes - let model handle translation directly
        return "plain", reading, ""
    
    def extract_proper_nouns(self, text: str) -> list[ExtractedName]:
        """
        Extract all proper nouns with readings from Japanese text.
        
        Handles MeCab splitting compound names (黛由紀江 → 黛 + 由紀江)
        and suffix separation (まゆちゃん → まゆ + ちゃん).
        
        Args:
            text: Japanese text to analyze
            
        Returns:
            List of ExtractedName objects
        """
        if not self._mecab_available or not self.tagger:
            return []
        
        # First pass: collect all tokens with their features
        tokens = []
        for word in self.tagger(text):
            try:
                pos1 = word.feature.pos1
                pos2 = word.feature.pos2
                pos3 = word.feature.pos3 if hasattr(word.feature, 'pos3') else ""
                kana = getattr(word.feature, 'kana', None) or word.surface
            except AttributeError:
                continue
            
            tokens.append({
                'surface': word.surface,
                'pos1': pos1,
                'pos2': pos2,
                'pos3': pos3,
                'kana': self._to_hiragana(kana),
            })
        

        # Second pass: merge adjacent proper nouns and attach suffixes
        results = []
        seen = set()
        i = 0
        
        # Common words that MeCab sometimes misclassifies as proper nouns
        # These should NEVER be merged into compound names
        garbage_words = {
            # Time words
            "今", "昨", "明", "先", "後", "毎日", "今日", "昨日", "明日", "朝", "昼", "夜",
            # Demonstratives  
            "此", "其", "彼", "これ", "それ", "あれ", "どれ", "ここ", "そこ", "あそこ", "どこ",
            # Pronouns (MeCab sometimes tags these as proper nouns in dialogue)
            "私", "僕", "俺", "自分", "貴様", "お前", "あなた", "アンタ", "君", "我", "余",
            "彼", "彼女", "あいつ", "こいつ", "そいつ",
            # Interrogatives
            "何", "誰", "何処", "いつ", "どう", "なぜ",
            # Interjections
            "ヤダ", "イヤ", "ダメ", "ウソ", "ホント", "マジ",
            "オイ", "ネェ", "アレ", "コレ", "ハイ", "イイエ", "ウン", "ソウ",
            # Generic Places
            "学校", "高校", "大学", "教室", "部屋", "家", "世界", "国", "町", "都市",
            # Generic Roles (Standalone) - Suffixes handle these when attached to names
            "先生", "生徒", "教師", "医者", "刑事", "警察",
            "父", "母", "兄", "弟", "姉", "妹", "両親", "子供",
            "男", "女", "人", "人間", "奴",
        }
        
        while i < len(tokens):
            t = tokens[i]
            
            # Check if this is a proper noun
            if t['pos1'] == "名詞" and t['pos2'] == "固有名詞":
                surface = t['surface']
                reading = t['kana']
                pos3 = t['pos3']
                
                # Skip garbage words - don't merge them!
                if surface in garbage_words or surface in self.blacklist_terms:
                    i += 1
                    continue
                # Filter by subcategory (person/place/org). Skip if unknown.
                allowed = False
                if surface in self.whitelist_terms:
                    allowed = True
                elif pos3:
                    if "人名" in pos3 or "地名" in pos3 or "組織" in pos3:
                        allowed = True
                if not allowed:
                    i += 1
                    continue
                
                # Merge adjacent proper nouns (黛 + 由紀江 → 黛由紀江)
                # Allow all types: person names, places, orgs, techniques
                start = i
                j = i + 1
                while j < len(tokens):
                    next_t = tokens[j]
                    
                    # Stop at garbage words - don't merge them
                    if next_t['surface'] in garbage_words or next_t['surface'] in self.blacklist_terms:
                        break
                    
                    # Continue merging if next token is also a proper noun OR a general noun (often names)
                    # This fixes splitting like "川神" (Proper) + "百代" (General/Common Noun)
                    if next_t['pos1'] == "名詞" and (next_t['pos2'] == "固有名詞" or next_t['pos2'] == "一般"):
                        surface += next_t['surface']
                        reading += next_t['kana']
                        j += 1
                    # Also attach suffixes (ちゃん, さん, etc.)
                    elif next_t['pos1'] == "接尾辞" and next_t['pos2'] == "名詞的":
                        surface += next_t['surface']
                        reading += next_t['kana']
                        j += 1
                    else:
                        break
                
                i = j  # Skip merged tokens
                
                # Skip if result is too short
                if len(surface) < 2:
                    continue
                
                # Add the merged name
                if surface not in seen:
                    seen.add(surface)
                    results.append(ExtractedName(
                        surface=surface,
                        reading=reading,
                        pos=pos3 or "固有名詞"
                    ))
                
                # Also add individual components if compound (for alias matching)
                # e.g., for 黛由紀江, also add 由紀江 separately
                if j - start > 1:
                    for k in range(start, j):
                        t2 = tokens[k]
                        if t2['pos1'] == "名詞" and t2['pos2'] == "固有名詞":
                            if len(t2['surface']) >= 2 and t2['surface'] not in seen:
                                seen.add(t2['surface'])
                                results.append(ExtractedName(
                                    surface=t2['surface'],
                                    reading=t2['kana'],
                                    pos=t2['pos3'] or "固有名詞"
                                ))
            else:
                i += 1

        # Title/role extraction (kanji patterns) + whitelist terms
        if text:
            title_matches = set()
            for term in self.title_terms:
                if term and term in text:
                    title_matches.add(term)
            # Suffix-based titles (e.g., 〇〇隊長)
            if self.title_suffixes:
                suffix_pattern = "|".join(sorted(self.title_suffixes, key=len, reverse=True))
                try:
                    import re
                    title_regex = re.compile(rf"[\\u4E00-\\u9FFF]{{1,6}}(?:{suffix_pattern})")
                    for m in title_regex.findall(text):
                        title_matches.add(m)
                except Exception:
                    pass
            for term in title_matches:
                if term in seen or term in self.blacklist_terms:
                    continue
                seen.add(term)
                results.append(ExtractedName(
                    surface=term,
                    reading=term,
                    pos="称号"
                ))
            for term in self.whitelist_terms:
                if term and term in text and term not in seen:
                    seen.add(term)
                    results.append(ExtractedName(
                        surface=term,
                        reading=term,
                        pos="白名单"
                    ))

        # Apply NER heuristics to boost name candidates MeCab missed
        boosted = self._boost_name_candidates(text, seen)
        results.extend(boosted)

        return results
    
    def _boost_name_candidates(self, text: str, already_seen: set) -> list[ExtractedName]:
        """
        Apply regex-based heuristics to catch names MeCab might miss.
        
        Targets:
        - Long katakana names (likely foreign names): アレクサンドラ
        - Katakana + suffix patterns: マリアちゃん, ジョンさん
        - Short capitalized katakana (2-4 chars): ミカ, サラ
        - Compound names with の/・: 佐藤の兄さん, マルコ・ポーロ
        
        Args:
            text: Japanese text to analyze
            already_seen: Set of names already extracted by MeCab
            
        Returns:
            List of additional ExtractedName objects
        """
        import re
        boosted = []
        
        # Pattern 1: Long katakana names (4+ chars, likely foreign names)
        # e.g., アレクサンドラ, クリスティーナ, エリザベス
        katakana_long = re.findall(r'[ァ-ヺー]{4,}', text)
        for name in katakana_long:
            if name not in already_seen and len(name) <= 12:
                already_seen.add(name)
                boosted.append(ExtractedName(
                    surface=name,
                    reading=self._to_hiragana(name),
                    pos="人名_外来"
                ))
        
        # Pattern 2: Katakana + Japanese suffix (foreign name + honorific)
        # e.g., ジョンさん, マリアちゃん, アリス先生
        katakana_suffix_pattern = re.compile(
            r'([ァ-ヺー]{2,8})(さん|ちゃん|くん|さま|様|せんせい|先生|せんぱい|先輩)'
        )
        for match in katakana_suffix_pattern.finditer(text):
            name = match.group(1)
            if name not in already_seen:
                already_seen.add(name)
                boosted.append(ExtractedName(
                    surface=name,
                    reading=self._to_hiragana(name),
                    pos="人名_外来"
                ))
        
        # Pattern 3: Short katakana names in quotes or followed by particles
        # e.g., 「ミカ」, ミカが, ミカは
        katakana_context_pattern = re.compile(
            r'[「『]([ァ-ヺー]{2,4})[」』]|([ァ-ヺー]{2,4})[がはをにの]'
        )
        for match in katakana_context_pattern.finditer(text):
            name = match.group(1) or match.group(2)
            if name and name not in already_seen:
                already_seen.add(name)
                boosted.append(ExtractedName(
                    surface=name,
                    reading=self._to_hiragana(name),
                    pos="人名_推測"
                ))
        
        # Pattern 4: Names with separator (の or ・)
        # e.g., マルコ・ポーロ -> マルコ, ポーロ
        separator_pattern = re.compile(r'([ァ-ヺー一-龯]{2,6})[・･]([ァ-ヺー一-龯]{2,6})')
        for match in separator_pattern.finditer(text):
            for name in [match.group(1), match.group(2)]:
                if name not in already_seen:
                    already_seen.add(name)
                    boosted.append(ExtractedName(
                        surface=name,
                        reading=self._to_hiragana(name) if re.match(r'^[ァ-ヺー]+$', name) else name,
                        pos="人名_推測"
                    ))

        # Pattern 5: Hiragana nicknames with honorifics
        # e.g., まゆちゃん, ゆきえさん
        hira_suffix_pattern = re.compile(
            r'([ぁ-ゖー]{2,6}(?:ちゃん|たん|くん|さん|さま|せんせい|せんぱい))'
        )
        for match in hira_suffix_pattern.finditer(text):
            alias = match.group(1)
            base = alias
            for suffix in self._sorted_suffixes:
                if alias.endswith(suffix):
                    base = alias[:-len(suffix)]
                    break
            if not base or base in self.kana_alias_blacklist or alias in already_seen:
                continue
            already_seen.add(alias)
            boosted.append(ExtractedName(
                surface=alias,
                reading=self._to_hiragana(alias),
                pos="人名_愛称"
            ))

        # Pattern 6: Hiragana reduplication nicknames
        # e.g., まゆまゆ
        hira_redup_pattern = re.compile(r'([ぁ-ゖー]{2,4})\1')
        for match in hira_redup_pattern.finditer(text):
            alias = match.group(0)
            base = match.group(1)
            if base in self.kana_alias_blacklist or alias in already_seen:
                continue
            already_seen.add(alias)
            boosted.append(ExtractedName(
                surface=alias,
                reading=self._to_hiragana(alias),
                pos="人名_愛称"
            ))

        # Pattern 7: Short hiragana names in quotes or directly followed by particles/punctuation
        hira_context_pattern = re.compile(
            r'(?:^|[「『（(、。！？!?…\s])([ぁ-ゖー]{2,4})(?=[がはをにへのとも、。！？!?…」』）)\s])'
        )
        for match in hira_context_pattern.finditer(text):
            name = match.group(1)
            if not name or name in already_seen or name in self.kana_alias_blacklist:
                continue
            already_seen.add(name)
            boosted.append(ExtractedName(
                surface=name,
                reading=self._to_hiragana(name),
                pos="人名_推測"
            ))
        
        return boosted
    
    def _to_hiragana(self, text: str) -> str:
        """Convert katakana to hiragana for consistent matching."""
        result = []
        for char in text:
            # Katakana range: 0x30A0 - 0x30FF
            # Hiragana range: 0x3040 - 0x309F
            code = ord(char)
            if 0x30A0 <= code <= 0x30FF:
                # Convert to hiragana
                result.append(chr(code - 0x60))
            else:
                result.append(char)
        return "".join(result)

    def _is_kana_only(self, text: str) -> bool:
        if not text:
            return False
        for char in text:
            code = ord(char)
            if not (0x3040 <= code <= 0x30FF):
                return False
        return True
    
    def group_aliases(self, names: list[ExtractedName]) -> list[AliasGroup]:
        """
        Group names by reading substring matching.
        
        The longest name is considered canonical. Shorter names whose reading
        is a substring of the canonical reading are considered aliases.
        
        Args:
            names: List of extracted names
            
        Returns:
            List of AliasGroup objects
        """
        if not names:
            return []
        
        # Sort by reading length (longest first = likely canonical)
        sorted_names = sorted(names, key=lambda n: len(n.reading), reverse=True)
        
        groups = []
        used = set()
        
        for name in sorted_names:
            if name.surface in used:
                continue
            
            # Start a new group with this as canonical
            group = AliasGroup(
                canonical=name.surface,
                canonical_reading=name.reading,
                aliases=[]
            )
            used.add(name.surface)
            
            # Find aliases (names whose base reading is substring of canonical)
            for other in sorted_names:
                if other.surface in used:
                    continue
                if other.surface == name.surface:
                    continue
                
                pattern, base_reading, hint = self.detect_pattern(other.surface, other.reading)
                
                # Check if base_reading is substring of canonical reading and surface looks related
                if base_reading and base_reading in name.reading:
                    # Require some surface overlap to reduce false merges
                    surface_ok = other.surface in name.surface or name.surface in other.surface
                    length_ok = len(base_reading) >= 2 and (len(base_reading) / max(1, len(name.reading))) >= 0.3
                    if not surface_ok and not length_ok:
                        # Allow kana-only nicknames if they are a substring of reading
                        if self._is_kana_only(other.surface):
                            if other.surface in self.kana_alias_blacklist:
                                continue
                            if len(base_reading) < 2:
                                continue
                        else:
                            continue
                    group.aliases.append({
                        "source": other.surface,
                        "reading": other.reading,
                        "pattern": pattern,
                        "hint": hint,
                    })
                    used.add(other.surface)
            
            # Only keep groups that have aliases (canonical + at least one alias)
            if group.aliases:
                groups.append(group)
        
        return groups
    
    def extract_and_group(self, text: str) -> list[AliasGroup]:
        """
        Convenience method: extract proper nouns and group aliases in one call.
        
        Args:
            text: Japanese text to analyze
            
        Returns:
            List of AliasGroup objects
        """
        names = self.extract_proper_nouns(text)
        return self.group_aliases(names)

    def get_reading(self, text: str) -> str:
        """Get hiragana reading for text."""
        if not self._mecab_available or not self.tagger:
            return text
        
        reading = ""
        for word in self.tagger(text):
            try:
                kana = getattr(word.feature, 'kana', None) or word.surface
                reading += self._to_hiragana(kana)
            except AttributeError:
                reading += word.surface
        return reading
