# -*- coding: utf-8 -*-
"""Character Alias Graph for linking name variants and improving translation consistency.

This module provides a graph-based approach to character name management,
linking together different forms of the same character's name (aliases)
to ensure consistent translation across all variants.

Key features:
- Detect name aliases based on reading similarity
- Build a graph of character relationships
- Provide canonical name lookups for any alias
- Support honorific suffix variations (まゆ → まゆちゃん, まゆさん)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Set, List
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CharacterNode:
    """A character with their canonical name and all known aliases."""
    canonical: str  # The primary/full name (e.g., 黛由紀江)
    canonical_reading: str  # Reading in hiragana (e.g., まゆずみゆきえ)
    aliases: Set[str] = field(default_factory=set)  # All known forms
    translation: Optional[str] = None  # The translated name
    gender: str = ""  # Gender hint (e.g., "Male", "Female")
    info: str = ""  # Additional info (e.g., "Main Character")
    context_sentences: List[str] = field(default_factory=list)  # Example sentences


class CharacterGraph:
    """
    A graph structure for managing character names and their aliases.
    
    The graph links together different forms of the same character's name
    to ensure consistent translation. For example:
    
    黛由紀江 (canonical)
      ├── まゆずみ (surname only)
      ├── まゆ (nickname)
      ├── まゆちゃん (nickname + chan)
      └── まゆずみゆきえ (full reading)
    
    All these forms should translate to the same English name.
    """
    
    def __init__(self):
        self._nodes: Dict[str, CharacterNode] = {}  # canonical -> node
        self._alias_index: Dict[str, str] = {}  # alias -> canonical
        self._reading_index: Dict[str, str] = {}  # reading -> canonical
        
    def add_character(self, canonical: str, reading: str, 
                      translation: Optional[str] = None,
                      gender: str = "", info: str = "") -> CharacterNode:
        """Add a new character to the graph."""
        if canonical in self._nodes:
            node = self._nodes[canonical]
            if translation: node.translation = translation
            if gender: node.gender = gender
            if info: node.info = info
            return node
            
        node = CharacterNode(
            canonical=canonical,
            canonical_reading=reading,
            translation=translation,
            gender=gender,
            info=info
        )
        node.aliases.add(canonical)
        self._nodes[canonical] = node
        self._alias_index[canonical] = canonical
        self._reading_index[reading] = canonical
        
        return node
    
    def add_alias(self, alias: str, canonical: str, 
                  alias_reading: Optional[str] = None) -> bool:
        """
        Link an alias to a canonical character name.
        
        Args:
            alias: The alias form (e.g., まゆちゃん)
            canonical: The canonical name (e.g., 黛由紀江)
            alias_reading: Optional reading for the alias
            
        Returns:
            True if successfully linked, False if canonical not found
        """
        if canonical not in self._nodes:
            return False
            
        # If alias exists as a canonical node, remove it (merge)
        if alias in self._nodes and alias != canonical:
            # Merge context if any
            alias_node = self._nodes[alias]
            self._nodes[canonical].context_sentences.extend(alias_node.context_sentences)
            # Re-map alias's existing aliases to the new canonical
            for sub_alias in alias_node.aliases:
                if sub_alias != alias:
                    self._alias_index[sub_alias] = canonical
                    self._nodes[canonical].aliases.add(sub_alias)
            # Remove old node
            del self._nodes[alias]
            
        node = self._nodes[canonical]
        node.aliases.add(alias)
        self._alias_index[alias] = canonical
        
        if alias_reading:
            self._reading_index[alias_reading] = canonical
            
        return True
    
    def find_canonical(self, name: str) -> Optional[str]:
        """Find the canonical form for any name variant."""
        return self._alias_index.get(name)
    
    def find_by_reading(self, reading: str) -> Optional[str]:
        """Find canonical name by reading (or reading prefix)."""
        # Exact match first
        if reading in self._reading_index:
            return self._reading_index[reading]
            
        # Prefix match (for nickname detection)
        for full_reading, canonical in self._reading_index.items():
            if full_reading.startswith(reading) or reading.startswith(full_reading):
                return canonical
                
        return None
    
    def get_translation(self, name: str) -> Optional[str]:
        """Get the translation for any form of a character's name."""
        canonical = self.find_canonical(name)
        if canonical and canonical in self._nodes:
            return self._nodes[canonical].translation
        return None
    
    def set_translation(self, canonical: str, translation: str) -> bool:
        """Set the translation for a character."""
        if canonical not in self._nodes:
            return False
        self._nodes[canonical].translation = translation
        return True
    
    def add_context_sentence(self, canonical: str, sentence: str) -> bool:
        """Add a context sentence for a character (for translation hints)."""
        if canonical not in self._nodes:
            return False
        node = self._nodes[canonical]
        if len(node.context_sentences) < 5:  # Limit context
            node.context_sentences.append(sentence)
        return True
    
    def auto_link_aliases(self, extracted_names: list) -> int:
        """
        Automatically detect and link aliases based on reading similarity.
        
        This uses several heuristics:
        1. Full reading match -> same character
        2. Reading prefix match -> likely nickname
        3. Same surface with different suffix -> honorific variation
        
        Args:
            extracted_names: List of ExtractedName objects
            
        Returns:
            Number of new links created
        """
        links_created = 0
        
        # Group by reading to find potential matches
        by_reading: Dict[str, List] = defaultdict(list)
        for name in extracted_names:
            by_reading[name.reading].append(name)
        
        # Find longest name for each reading group (likely canonical)
        for reading, names in by_reading.items():
            if not names:
                continue
                
            # Sort by surface length (longest is canonical)
            sorted_names = sorted(names, key=lambda n: -len(n.surface))
            canonical_name = sorted_names[0]
            
            # Add canonical if not exists
            if canonical_name.surface not in self._nodes:
                self.add_character(canonical_name.surface, canonical_name.reading)
            
            # Link shorter forms as aliases (if any)
            if len(sorted_names) > 1:
                for name in sorted_names[1:]:
                    if name.surface != canonical_name.surface:
                        # CRITICAL: Normalize aliases too
                        if self.add_alias(name.surface, canonical_name.surface):
                            links_created += 1
        
        # Check for prefix matches (e.g., まゆ <-> まゆずみ)
        all_readings = list(self._reading_index.keys())
        for reading in all_readings:
            # Handle Reduplication (e.g. MayuMayu -> Mayu)
            base_reading = reading
            if len(reading) >= 4 and reading[:len(reading)//2] == reading[len(reading)//2:]:
                base_reading = reading[:len(reading)//2]
            
            for other_reading, other_canonical in list(self._reading_index.items()):
                if reading == other_reading:
                    continue
                
                match = False
                
                # 1. Direct Prefix/Inclusion (Must be significant length)
                if len(reading) > 2 and len(other_reading) > 2 and (reading in other_reading or other_reading in reading):
                    match = True
                
                # 2. Suffix Match (Stricter: Min length 4 for suffix logic to avoid short collisions)
                elif len(reading) >= 4 and len(other_reading) >= 4 and (other_reading.endswith(reading) or reading.endswith(other_reading)):
                    match = True
                
                # 3. Reduplication Base Match (MayuMayu -> [Mayu] -> Mayuzumi)
                elif base_reading != reading and len(base_reading) >= 2 and (other_reading.startswith(base_reading) or base_reading.startswith(other_reading)):
                    match = True
                
                # 4. Shared Prefix (REMOVED: Too aggressive, causes unrelated names to merge)
                # elif reading[:2] == other_reading[:2] and len(reading) >= 3 and len(other_reading) >= 3:
                #      match = True

                if match:
                    # Resolve current canonicals (handle chaining merges)
                    current_canon_1 = self._reading_index.get(reading)
                    current_canon_2 = other_canonical
                    
                    node1_name = self.find_canonical(current_canon_1)
                    node2_name = self.find_canonical(current_canon_2)
                    
                    if node1_name and node2_name and node1_name != node2_name:
                        # Merge: shorter one becomes alias of longer one
                        node1 = self._nodes.get(node1_name)
                        node2 = self._nodes.get(node2_name)
                        
                        if node1 and node2:
                            if len(node1_name) > len(node2_name):
                                self.add_alias(node2_name, node1_name)
                                links_created += 1
                            else:
                                self.add_alias(node1_name, node2_name)
                                links_created += 1
        
        return links_created
    
    def to_dict(self) -> dict:
        """Serialize the graph to a dictionary for storage."""
        return {
            "characters": [
                {
                    "canonical": node.canonical,
                    "name": node.canonical,  # Backward compat
                    "reading": node.canonical_reading,
                    "aliases": list(node.aliases),
                    "translation": node.translation,
                    "gender": node.gender,
                    "info": node.info,
                    "context": node.context_sentences
                }
                for node in self._nodes.values()
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CharacterGraph":
        """Deserialize a graph from a dictionary."""
        graph = cls()
        for char_data in data.get("characters", []):
            # Support both 'canonical' and 'name' keys
            canonical = char_data.get("canonical") or char_data.get("name")
            if not canonical:
                continue
                
            node = graph.add_character(
                canonical=canonical,
                reading=char_data.get("reading", ""),
                translation=char_data.get("translation"),
                gender=char_data.get("gender", ""),
                info=char_data.get("info", "")
            )
            for alias in char_data.get("aliases", []):
                if alias != canonical:
                    graph.add_alias(alias, canonical)
            for sentence in char_data.get("context", []):
                graph.add_context_sentence(canonical, sentence)
        return graph
    
    def to_glossary_entries(self) -> list:
        """
        Convert the graph to glossary format for style_guide.json.
        
        Returns:
            List of glossary entries [{source, target, note}]
        """
        entries = []
        for node in self._nodes.values():
            if not node.translation:
                continue
                
            # Add all aliases as glossary entries pointing to same translation
            for alias in node.aliases:
                entries.append({
                    "source": alias,
                    "target": node.translation,
                    "note": f"Alias of {node.canonical}" if alias != node.canonical else ""
                })
        
        return entries
    
    def merge_from_glossary(self, glossary: list) -> int:
        """
        Import existing glossary entries into the graph.
        
        Args:
            glossary: List of glossary entries from style_guide.json
            
        Returns:
            Number of entries imported
        """
        imported = 0
        for entry in glossary:
            source = entry.get("source", "")
            target = entry.get("target", "")
            if not source or not target:
                continue
                
            # Check if this is already an alias
            canonical = self.find_canonical(source)
            if canonical:
                # Update translation if not set
                if not self._nodes[canonical].translation:
                    self.set_translation(canonical, target)
            else:
                # New character
                self.add_character(
                    canonical=source,
                    reading=source,  # No reading info from glossary
                    translation=target
                )
            imported += 1
            
        return imported
    
    def __len__(self) -> int:
        return len(self._nodes)
    
    def __repr__(self) -> str:
        return f"CharacterGraph({len(self._nodes)} characters, {len(self._alias_index)} aliases)"
