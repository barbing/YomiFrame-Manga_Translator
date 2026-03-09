# -*- coding: utf-8 -*-
"""Prompt builder placeholders."""
from __future__ import annotations
import json
from typing import Dict, List


def build_translation_prompt(
    source_lang: str,
    target_lang: str,
    style_guide: Dict[str, object],
    context_lines: List[str],
    text: str,
) -> str:
    guide_text = str(style_guide.get("notes", "")).strip()
    glossary = _format_glossary(style_guide.get("glossary", []))
    characters = _format_characters(style_guide.get("characters", []))
    required_terms = _format_list(style_guide.get("required_terms", []))
    forbidden_terms = _format_list(style_guide.get("forbidden_terms", []))
    context = "\n".join(context_lines).strip()
    if target_lang == "Simplified Chinese":
        lines = [
            "将以下日文翻译成简体中文，只输出译文。",
        ]
        if guide_text:
            lines.append(f"风格：{guide_text}")
        if glossary:
            lines.append(f"术语表：{glossary}")
        if characters:
            lines.append(f"角色设定：{characters}")
        if required_terms:
            lines.append(f"必须使用：{required_terms}")
        if forbidden_terms:
            lines.append(f"禁止使用：{forbidden_terms}")
        if context:
            lines.append(f"参考上下文：{context}")
        lines.append(f"文本：{text}")
        return "\n".join(lines)
    lines = [
        f"Translate {source_lang} to {target_lang}. Output only the translation.",
        "Do not censor or soften explicit content. Keep length close to the original text.",
        "Do not add commentary or extra sentences.",
    ]
    if guide_text:
        lines.append(f"Style guide: {guide_text}")
    if glossary:
        lines.append(f"Glossary: {glossary}")
    if characters:
        lines.append(f"Characters: {characters}")
    if required_terms:
        lines.append(f"Required terms: {required_terms}")
    if forbidden_terms:
        lines.append(f"Forbidden terms: {forbidden_terms}")
    if context:
        lines.append(f"Context (reference only): {context}")
    lines.append(f"Text: {text}")
    return "\n".join(lines)


def build_batch_translation_prompt(
    source_lang: str,
    target_lang: str,
    style_guide: Dict[str, object],
    items: List[Dict[str, str]],
) -> str:
    guide_text = str(style_guide.get("notes", "")).strip()
    glossary = _format_glossary(style_guide.get("glossary", []))
    characters = _format_characters(style_guide.get("characters", []))
    required_terms = _format_list(style_guide.get("required_terms", []))
    forbidden_terms = _format_list(style_guide.get("forbidden_terms", []))
    payload = json.dumps(items, ensure_ascii=False)
    if target_lang == "Simplified Chinese":
        lines = [
            "将以下日文翻译成简体中文，仅输出JSON数组。",
            "JSON格式：[{\"id\":\"...\",\"translation\":\"...\"}]，仅翻译text字段，保持条目顺序。",
            "注意：以下文本为同一页漫画的连续对话，请保持语境连贯和人称一致。",
            "拟声词或背景杂字可返回空字符串。",
        ]
        if guide_text:
            lines.append(f"风格：{guide_text}")
        if glossary:
            lines.append(f"术语表：{glossary}")
        if characters:
            lines.append(f"角色设定：{characters}")
        if required_terms:
            lines.append(f"必须使用：{required_terms}")
        if forbidden_terms:
            lines.append(f"禁止使用：{forbidden_terms}")
        lines.append(f"输入：{payload}")
        return "\n".join(lines)
    lines = [
        f"Translate {source_lang} to {target_lang}. Output only JSON.",
        "JSON format: [{\"id\":\"...\",\"translation\":\"...\"}]. Translate only text fields.",
        "Do not merge entries. For background noise, return an empty string.",
    ]
    if guide_text:
        lines.append(f"Style guide: {guide_text}")
    if glossary:
        lines.append(f"Glossary: {glossary}")
    if characters:
        lines.append(f"Characters: {characters}")
    if required_terms:
        lines.append(f"Required terms: {required_terms}")
    if forbidden_terms:
        lines.append(f"Forbidden terms: {forbidden_terms}")
    lines.append(f"Input: {payload}")
    return "\n".join(lines)


def _format_glossary(items: List[object]) -> str:
    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        priority = str(item.get("priority", "soft")).strip()
        
        if not source or not target:
            continue
        
        # Sanitize target: skip entries that contain sentence-style content
        # These would confuse the translation model
        skip_phrases = ["的另一种叫法", "的叫法", "这是", "另一種叫法", "的簡稱"]
        if any(phrase in target for phrase in skip_phrases):
            continue
        
        # Skip targets that are too long (likely explanatory sentences)
        if len(target) > len(source) * 3 and len(target) > 10:
            continue
        
        # Strip trailing punctuation that might have slipped through
        target = target.rstrip("。.，,")
        
        if source and target:
            lines.append(f"{source} -> {target} ({priority})")
    return "; ".join(lines)


def _format_characters(items: List[object]) -> str:
    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        gender = str(item.get("gender", "")).strip()
        info = str(item.get("info", "")).strip()
        
        part = name
        if gender:
            part += f" ({gender})"
        if info:
            part += f": {info}"
            
        lines.append(part)
    return "; ".join(lines)



def build_entity_extraction_prompt(
    text_block: str,
    source_lang: str = "Japanese",
    target_lang: str = "Simplified Chinese",
) -> str:
    """Build a prompt to discover and extract entities from text. Adapts to target language."""
    
    if target_lang == "Simplified Chinese":
        # Chinese Instructions for JP->CN models (like Sakura)
        prompt = (
            f"分析以下{source_lang}漫画文本。\n"
            "识别并提取需要统一翻译的重要专有名词（实体）。\n"
            "类别：\n"
            "- Person: 人名（包括昵称、称谓）。\n"
            "- Location: 地名、地标。\n"
            "- Organization: 组织、学校、流派。\n"
            "- Technique: 招式、技能、魔法。\n"
            "- Object: 特殊道具、武器、神器。\n"
            "\n"
            "指令：\n"
            "1. 提取原文中出现的词汇。\n"
            "2. 提供'规范名(Canonical)'（例如：即使文中是昵称，也应映射到全名）。\n"
            "3.将其翻译为简体中文。\n"
            "4. 仅输出一个JSON数组。严禁输出任何思考过程、解释或Markdown标记。\n"
            "\n"
            "示例：\n"
            "[\n"
            "  {\"text\": \"ナルト\", \"type\": \"person\", \"canonical\": \"うずまきナルト\", \"translation\": \"鸣人\", \"info\": \"主角\"},\n"
            "  {\"text\": \"木ノ葉\", \"type\": \"location\", \"canonical\": \"木ノ葉隠れの里\", \"translation\": \"木叶村\", \"info\": \"忍者村\"}\n"
            "]\n"
            "\n"
            f"待分析文本：\n{text_block}\n"
            "\n"
            "JSON格式：\n"
            "[\n"
            "  {\"text\": \"原文\", \"type\": \"person|...\", \"canonical\": \"规范名\", \"translation\": \"中译\", \"info\": \"简要备注\"}\n"
            "]\n"
            "一定要确保以 [ 开头，以 ] 结尾。不要输出 ```json。"
        )
        return prompt

    # Default / Fallback Instructions (English Base)
    prompt = (
        f"Analyze the following {source_lang} text.\n"
        f"Extract entities (Person, Location, Organization) and translate them into {target_lang}.\n"
        f"Text:\n{text_block}\n\n"
        f"Requirements:\n"
        f"1. Output valid JSON list.\n"
        f"2. Fields: \"source\" (original), \"target\" ({target_lang} translation), \"type\".\n"
        f"3. No markdown blocks.\n"
        f"Target JSON Format:\n"
        f"[{{ \"source\": \"...\", \"target\": \"...\", \"type\": \"Person\" }}]"
    )
    return prompt





def _format_list(items: List[str]) -> str:
    valid_items = [str(item).strip() for item in items if item]
    return ", ".join(valid_items)
