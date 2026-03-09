# -*- coding: utf-8 -*-
"""Simple renderer for translated text."""
from __future__ import annotations
import os
import re
from typing import Dict, List, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont, ImageStat
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None
    ImageFont = None
    ImageStat = None

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    np = None


def render_translations(
    image_path: str,
    output_path: str,
    regions: List[Dict[str, object]],
    font_name: str,
    inpaint_mode: str = "fast",
    use_gpu: bool = True,
    model_id: str = "dreMaz/AnimeMangaInpainting",
) -> None:
    if Image is None:
        raise RuntimeError("Pillow is not installed.")
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        working = img
        img_w, img_h = img.size
        render_regions: List[Tuple[str, Tuple[int, int, int, int], str, Tuple[int, int, int] | None]] = []
        background_boxes: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int] | None]] = []
        text_mask = None
        bubble_text_mask = None
        bubble_area_mask = None
        other_text_mask = None
        if cv2 is not None and np is not None:
            text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            bubble_text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            bubble_area_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            other_text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        img_np = np.array(img) if cv2 is not None and np is not None else None

        for region in regions:
            if not isinstance(region, dict):
                continue
            raw_text = str(region.get("translation", "")).strip()
            text = _normalize_text(raw_text)
            flags = region.get("flags", {}) or {}
            if not text or flags.get("ignore"):
                continue
            region_type = str(region.get("type", "") or "")
            is_background = region_type == "background_text" or bool(flags.get("bg_text"))
            bbox = region.get("bbox", [0, 0, 0, 0])
            polygon = region.get("polygon")
            poly_bounds = _polygon_bounds(polygon)
            if poly_bounds and _box_area(poly_bounds) < _box_area(bbox) * 0.85:
                x, y, w, h = [int(v) for v in poly_bounds]
            else:
                x, y, w, h = [int(v) for v in bbox]
            mask_pad = _fill_padding(w, h)
            text_pad = _text_padding(w, h)
            mx0 = max(0, x - mask_pad)
            my0 = max(0, y - mask_pad)
            mx1 = min(img_w, x + w + mask_pad)
            my1 = min(img_h, y + h + mask_pad)
            tx0 = max(0, x - text_pad)
            ty0 = max(0, y - text_pad)
            tx1 = min(img_w, x + w + text_pad)
            ty1 = min(img_h, y + h + text_pad)
            base_box = (x, y, x + w, y + h)
            render_box = (tx0, ty0, tx1, ty1)
            forced_color = None
            if img_np is not None:
                stats = _box_luma_stats(img_np, base_box)
                if stats:
                    mean, p20, p80 = stats
                    if p80 < 95:
                        forced_color = (255, 255, 255)
                        if region_type != "speech_bubble" and not is_background:
                            is_background = True
                    elif p20 > 175:
                        forced_color = (0, 0, 0)
            if img_np is not None and text_mask is not None and not is_background:
                region_mask, bubble_box, bubble_mask = _region_masks(
                    img_np,
                    (mx0, my0, mx1, my1),
                    bbox,
                    polygon,
                )
                if region_mask is not None:
                    if bubble_mask is not None:
                        region_mask = cv2.bitwise_and(region_mask, bubble_mask)
                    text_mask = cv2.bitwise_or(text_mask, region_mask)
                    if bubble_mask is not None and bubble_text_mask is not None:
                        bubble_text_mask = cv2.bitwise_or(bubble_text_mask, region_mask)
                        if bubble_area_mask is not None:
                            bubble_area_mask = cv2.bitwise_or(bubble_area_mask, bubble_mask)
                    elif other_text_mask is not None:
                        other_text_mask = cv2.bitwise_or(other_text_mask, region_mask)
                if bubble_box is not None:
                    bubble_area = _box_area([bubble_box[0], bubble_box[1], bubble_box[2] - bubble_box[0], bubble_box[3] - bubble_box[1]])
                    base_area = _box_area([base_box[0], base_box[1], base_box[2] - base_box[0], base_box[3] - base_box[1]])
                    if base_area and bubble_area >= base_area * 0.6:
                        limited = _intersect_box(render_box, bubble_box)
                        render_box = limited or bubble_box
                    else:
                        render_box = base_box
                else:
                    render_box = base_box
            else:
                render_box = base_box
            if is_background:
                pad = max(1, int(min(w, h) * 0.02))
                render_box = _shrink_box(render_box, pad)
                bg_pad = min(28, max(4, int(min(w, h) * 0.18)))
                fill_color = _estimate_box_fill(img_np, base_box) if img_np is not None else None
                expanded_box = None
                if img_np is not None:
                    stats = _box_luma_stats(img_np, base_box)
                    if stats:
                        _mean, _p20, _p80 = stats
                        if _p80 < 95:
                            expanded_box = _expand_dark_box(img_np, base_box)
                if expanded_box is not None:
                    ex0, ey0, ex1, ey1 = expanded_box
                    render_box = _intersect_box(render_box, expanded_box) or expanded_box
                    background_boxes.append(((ex0, ey0, ex1, ey1), fill_color))
                else:
                    background_boxes.append(
                        (
                            (
                                max(0, x - bg_pad),
                                max(0, y - bg_pad),
                                min(img_w, x + w + bg_pad),
                                min(img_h, y + h + bg_pad),
                            ),
                            fill_color,
                        )
                    )
            else:
                render_box = _shrink_box(render_box, max(2, int(min(w, h) * 0.03)))
            render = region.get("render") or {}
            if not isinstance(render, dict):
                render = {}
            region_font = render.get("font") or font_name
            if _has_cjk(text) and _is_cjk_unsupported_font(region_font):
                region_font = font_name
            render_regions.append((text, render_box, region_font, forced_color))

        if render_regions:
            # Combine background boxes into a single mask for inpainting
            if background_boxes and cv2 is not None and np is not None:
                 if text_mask is None:
                     text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                 
                 bg_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                 for box, _ in background_boxes:
                     bx0, by0, bx1, by1 = [int(v) for v in box]
                     # Ensure we don't draw outside bounds
                     bx0 = max(0, min(bx0, img_w))
                     by0 = max(0, min(by0, img_h))
                     bx1 = max(bx0, min(bx1, img_w))
                     by1 = max(by0, min(by1, img_h))
                     cv2.rectangle(bg_mask, (bx0, by0), (bx1, by1), 255, thickness=-1)
                 
                 # Combine with existing text mask
                 text_mask = cv2.bitwise_or(text_mask, bg_mask)
                 # Also update bubble_text_mask if we want them treated similarly, 
                 # but usually background text is distinct. 
                 # However, to ensure they are removed, we add to text_mask which is used by _apply_text_removal.

            if bubble_area_mask is not None and bubble_text_mask is not None and bubble_area_mask.any():
                if cv2 is not None and np is not None:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    bubble_area_mask = cv2.morphologyEx(bubble_area_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                working = _apply_bubble_fill(working, bubble_area_mask, bubble_text_mask, img_np)
            
            # Now apply text removal for everything (including background text which is now in text_mask)
            if text_mask is not None and text_mask.any():
                working = _apply_text_removal(working, text_mask, inpaint_mode, use_gpu, model_id=model_id)

        draw = ImageDraw.Draw(working)
        median_height = 0
        preferred_size = None
        if render_regions:
            heights = sorted(max(1, box[3] - box[1]) for _, box, _, _ in render_regions)
            median_height = heights[len(heights) // 2]
            preferred_size = max(12, int(median_height * 0.33))
        for text, box, region_font, forced_color in render_regions:
            x0, y0, x1, y1 = box
            w = max(1, x1 - x0)
            h = max(1, y1 - y0)
            local_preferred = None
            if preferred_size and h >= preferred_size * 2:
                local_preferred = min(preferred_size, max(12, int(h * 0.7)))
            base_font = _fit_font(draw, text, w, h, region_font, preferred_size=local_preferred)
            best_lines = _wrap_text(draw, text, base_font, w)
            best_font = base_font
            best_height = sum(_text_height(base_font, line) for line in best_lines)
            max_lines = max(1, min(6, int(h / max(1, _text_height(base_font, "A"))) + 1))
            for lines_count in range(2, max_lines + 1):
                test_lines = _wrap_text(draw, text, base_font, w, max_lines=lines_count)
                test_height = sum(_text_height(base_font, line) for line in test_lines)
                if test_height <= h and len(test_lines) > len(best_lines):
                    best_lines = test_lines
                    best_height = test_height
            if best_height > h and local_preferred:
                for size in range(local_preferred - 1, 9, -1):
                    test_font = _load_font(_find_font_path(region_font), size, _sample_char(text))
                    test_lines = _wrap_text(draw, text, test_font, w, max_lines=max_lines)
                    test_height = sum(_text_height(test_font, line) for line in test_lines)
                    if test_height <= h:
                        best_font = test_font
                        best_lines = test_lines
                        best_height = test_height
                        break
            offset_y = y0 + max(0, (h - best_height) // 2)
            fill_color = forced_color or _resolve_text_color(working, box)
            for line in best_lines:
                bbox = best_font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
                offset_x = x0 + max(0, (w - line_width) // 2) - bbox[0]
                draw.text((offset_x, offset_y - bbox[1]), line, fill=fill_color, font=best_font)
                offset_y += line_height
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save with high quality to prevent JPEG compression artifacts
        ext = os.path.splitext(output_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            working.save(output_path, quality=95, optimize=True)
        else:
            working.save(output_path)


def _apply_text_removal(image, text_mask, mode: str, use_gpu: bool, model_id: str = "dreMaz/AnimeMangaInpainting"):
    if text_mask is None:
        return image
    mode = (mode or "fast").lower()
    
    if cv2 is None or np is None:
        print("[TextRemoval] No CV2/numpy, using simple white mask")
        return _apply_white_mask(image, text_mask)
    
    # Always try white bubble fill first (cleanest option)
    bubble_fill = _white_bubble_fill(image, text_mask)
    if bubble_fill is not None:
        print("[TextRemoval] Used WHITE BUBBLE FILL")
        return bubble_fill
    
    # Try uniform fill for simple backgrounds (handles most cases)
    uniform_fill = _apply_uniform_fill(image, text_mask)
    if uniform_fill is not None:
        print("[TextRemoval] Used UNIFORM FILL")
        return uniform_fill
    
    # For remaining complex areas: use AI inpainting if enabled, otherwise CV2
    if mode == "ai" and use_gpu:
        try:
            print(f"[TextRemoval] Using AI INPAINT with model: {model_id}")
            from app.render.inpaint_ai import ai_inpaint
            result = ai_inpaint(image, text_mask, use_gpu=use_gpu, model_id=model_id)
            print("[TextRemoval] AI INPAINT Success")
            return result
        except Exception as e:
            print(f"[TextRemoval] AI INPAINT failed: {e}, falling back to CV2")
    
    print("[TextRemoval] Used CV2 INPAINT (fallback)")
    # CV2 inpainting with moderate dilation for complex areas
    img_np = np.array(image)
    kernel_size = max(5, int(max(text_mask.shape) * 0.004))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(text_mask, kernel, iterations=2)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(img_bgr, dilated, 5, cv2.INPAINT_NS)
    rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _process_fast_inpaint(image, mask):
    if cv2 is None or np is None:
        return image
    img_np = np.array(image)
    kernel_size = max(3, int(max(mask.shape) * 0.0015))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(img_bgr, dilated, 3, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _apply_background_fill(
    image,
    boxes: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int] | None]],
):
    if cv2 is None or np is None or not boxes:
        return image
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    for box, fill_color in boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(x0 + 1, min(x1, w))
        y1 = max(y0 + 1, min(y1, h))
        if fill_color is None:
            samples = _sample_box_border(img_np, x0, y0, x1, y1, pad=3)
            if samples.size < 3:
                samples = img_np[y0:y1, x0:x1].reshape(-1, 3)
            if samples.size < 3:
                continue
            median = np.median(samples, axis=0)
            fill_color = tuple(int(v) for v in median.tolist())
        img_np[y0:y1, x0:x1] = np.array(fill_color, dtype=np.uint8)
    return Image.fromarray(img_np)


def _box_luma_stats(img_np, box: Tuple[int, int, int, int]):
    if cv2 is None or np is None:
        return None
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = img_np.shape[:2]
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    crop = img_np[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    if gray.size == 0:
        return None
    mean = float(gray.mean())
    p20 = float(np.percentile(gray, 20))
    p80 = float(np.percentile(gray, 80))
    return mean, p20, p80


def _estimate_box_fill(img_np, box: Tuple[int, int, int, int]):
    stats = _box_luma_stats(img_np, box)
    if stats is None:
        return None
    mean, p20, p80 = stats
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = img_np.shape[:2]
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    crop = img_np[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    pixels = crop.reshape(-1, 3)
    gray_flat = gray.reshape(-1)
    if p80 < 95:
        mask = gray_flat <= np.percentile(gray_flat, 50)
    elif p20 > 175:
        mask = gray_flat >= np.percentile(gray_flat, 50)
    else:
        mask = slice(None)
    sample = pixels[mask] if not isinstance(mask, slice) else pixels
    if sample.size < 3:
        sample = pixels
    median = np.median(sample.reshape(-1, 3), axis=0)
    return tuple(int(v) for v in median.tolist())

def _sample_box_border(img_np, x0: int, y0: int, x1: int, y1: int, pad: int = 3):
    h, w = img_np.shape[:2]
    pad = max(1, pad)
    ix0 = min(max(x0 + pad, 0), w)
    iy0 = min(max(y0 + pad, 0), h)
    ix1 = max(min(x1 - pad, w), ix0 + 1)
    iy1 = max(min(y1 - pad, h), iy0 + 1)
    outer = np.zeros((h, w), dtype=np.uint8)
    inner = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(outer, (x0, y0), (x1 - 1, y1 - 1), 255, thickness=-1)
    cv2.rectangle(inner, (ix0, iy0), (ix1 - 1, iy1 - 1), 255, thickness=-1)
    border = cv2.subtract(outer, inner)
    return img_np[border > 0].reshape(-1, 3)


def _expand_dark_box(img_np, box: Tuple[int, int, int, int]):
    if cv2 is None or np is None:
        return None
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = img_np.shape[:2]
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    pad = int(max(x1 - x0, y1 - y0) * 1.5)
    rx0 = max(0, x0 - pad)
    ry0 = max(0, y0 - pad)
    rx1 = min(w, x1 + pad)
    ry1 = min(h, y1 + pad)
    roi = img_np[ry0:ry1, rx0:rx1]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    thresh = 120
    _, dark = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels = cv2.connectedComponents(dark)
    if num_labels <= 1:
        return None
    cy = int((y0 + y1) / 2) - ry0
    cx = int((x0 + x1) / 2) - rx0
    if cy < 0 or cx < 0 or cy >= labels.shape[0] or cx >= labels.shape[1]:
        return None
    label = labels[cy, cx]
    if label == 0:
        return None
    ys, xs = np.where(labels == label)
    if ys.size == 0 or xs.size == 0:
        return None
    bx0 = int(xs.min()) + rx0
    by0 = int(ys.min()) + ry0
    bx1 = int(xs.max()) + rx0
    by1 = int(ys.max()) + ry0
    base_area = max(1, (x1 - x0) * (y1 - y0))
    expanded_area = max(1, (bx1 - bx0) * (by1 - by0))
    if expanded_area > base_area * 8:
        return None
    return (bx0, by0, bx1, by1)


def _apply_uniform_fill(image, text_mask):
    if cv2 is None or np is None:
        return None
    if text_mask is None:
        return None
    if not np.any(text_mask):
        return None
    img_np = np.array(image)
    num_labels, labels = cv2.connectedComponents(text_mask)
    if num_labels <= 1:
        return None
    result = img_np.copy()
    applied = False
    skipped_complex = False  # Track if we skipped any complex backgrounds
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        if component.sum() < 8:
            continue
        
        # First, sample border to determine background complexity
        sample_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sample_dilated = cv2.dilate(component, sample_kernel, iterations=2)
        sample_border = cv2.subtract(sample_dilated, component)
        samples = img_np[sample_border > 0]
        
        if samples.size < 9:
            # Not enough samples - skip, let AI handle it
            skipped_complex = True
            continue
        
        sample_arr = samples.reshape(-1, 3)
        std = float(sample_arr.std(axis=0).mean())
        median = np.median(sample_arr, axis=0)
        
        # ONLY handle uniform backgrounds - skip complex ones for AI
        if std <= 15:
            # Very uniform (white bubble): moderate dilation to preserve borders
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(component, kernel, iterations=2)
            result[dilated > 0] = median.astype(np.uint8)
            applied = True
        elif std <= 30:
            # Moderately uniform: minimal dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(component, kernel, iterations=1)
            result[dilated > 0] = median.astype(np.uint8)
            applied = True
        else:
            # Complex background (artwork): SKIP - let AI handle it
            skipped_complex = True
            continue
    
    # Only return if we processed some regions AND didn't skip any complex ones
    # If we skipped complex regions, return None so AI inpainting handles everything
    if applied and not skipped_complex:
        return Image.fromarray(result)
    return None


def _apply_bubble_fill(image, bubble_mask, text_mask, reference_np):
    if bubble_mask is None or text_mask is None:
        return image
    if cv2 is None or np is None:
        return _apply_white_mask(image, text_mask)
    result = np.array(image)
    ref = reference_np if reference_np is not None else result
    num_labels, labels = cv2.connectedComponents(bubble_mask)
    if num_labels <= 1:
        return _apply_white_mask(image, text_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    text_binary = (text_mask > 0).astype(np.uint8)
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        if component.sum() == 0:
            continue
        text_region = cv2.bitwise_and(component, text_binary)
        if text_region.sum() == 0:
            continue
        ys, xs = np.where(text_region > 0)
        if ys.size == 0 or xs.size == 0:
            continue
        tx0, ty0, tx1, ty1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        pad = max(4, int(max(tx1 - tx0, ty1 - ty0) * 0.35))
        bx0 = max(0, tx0 - pad)
        by0 = max(0, ty0 - pad)
        bx1 = min(component.shape[1] - 1, tx1 + pad)
        by1 = min(component.shape[0] - 1, ty1 + pad)
        local = np.zeros_like(component)
        local[by0 : by1 + 1, bx0 : bx1 + 1] = component[by0 : by1 + 1, bx0 : bx1 + 1]
        halo = cv2.dilate(text_region, kernel, iterations=1)
        sample_mask = cv2.bitwise_and(local, cv2.bitwise_not(halo))
        samples = ref[sample_mask > 0]
        if samples.size < 12:
            inner = cv2.erode(component, kernel, iterations=2)
            if inner.sum() > 0:
                samples = ref[inner > 0]
        if samples.size < 3:
            color = (255, 255, 255)
        else:
            median = np.median(samples.reshape(-1, 3), axis=0)
            color = tuple(int(v) for v in median.tolist())
        result[text_region > 0] = color
    return Image.fromarray(result)


def _apply_white_mask(image, text_mask):
    if cv2 is None or np is None:
        draw = ImageDraw.Draw(image)
        return image
    img_np = np.array(image)
    # Aggressively dilate mask to ensure complete text coverage
    kernel_size = max(7, int(max(text_mask.shape) * 0.006))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(text_mask, kernel, iterations=4)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    img_np[dilated > 0] = (255, 255, 255)
    return Image.fromarray(img_np)


def _resolve_text_color(image, box, default=(0, 0, 0)):
    if ImageStat is None or image is None:
        return default
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = max(x0 + 1, x1)
    y1 = max(y0 + 1, y1)
    try:
        # Sample a larger area to include the bubble rim/background
        pad = max(10, int(min(x1 - x0, y1 - y0) * 0.15))
        bx0 = max(0, x0 - pad)
        by0 = max(0, y0 - pad)
        bx1 = min(image.width, x1 + pad)
        by1 = min(image.height, y1 + pad)
        
        crop = image.crop((bx0, by0, bx1, by1))
        
        # Mask out the center (text area) to only checking the rim
        mask = Image.new("L", crop.size, 255)
        draw = ImageDraw.Draw(mask)
        # Inner box relative to crop
        ix0 = x0 - bx0
        iy0 = y0 - by0
        ix1 = x1 - bx0
        iy1 = y1 - by0
        draw.rectangle((ix0, iy0, ix1, iy1), fill=0)
        
        stat = ImageStat.Stat(crop, mask)
        if not stat.mean or len(stat.mean) < 3:
            return default
            
        mean = stat.mean
        lum = 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2]
        
        # If background is bright, use black text. If dark, use white.
        if lum > 140:
             return (0, 0, 0)
        elif lum < 100:
             return (255, 255, 255)
             
    except Exception:
        return default
    return default


def _region_masks(
    img_np,
    pad_box: Tuple[int, int, int, int],
    bbox: List[int],
    polygon,
):
    if cv2 is None or np is None:
        return None, None
    x0, y0, x1, y1 = pad_box
    roi = img_np[y0:y1, x0:x1]
    if roi.size == 0:
        return None, None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=1)
    if white.mean() < 5:
        _, white = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=1)

    text_mask = np.zeros((white.shape[0], white.shape[1]), dtype=np.uint8)
    polys = _normalize_polygons(polygon)
    if polys:
        try:
            for poly in polys:
                poly = np.array(poly, dtype=np.int32)
                poly[:, 0] = poly[:, 0] - x0
                poly[:, 1] = poly[:, 1] - y0
                cv2.fillPoly(text_mask, [poly], 255)
        except Exception:
            text_mask = _rect_mask(text_mask.shape, bbox, x0, y0)
    else:
        text_mask = _rect_mask(text_mask.shape, bbox, x0, y0)
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)

    use_mask = white
    text_pixels = int((text_mask > 0).sum())
    if text_pixels:
        _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)
        white_overlap = int(((white > 0) & (text_mask > 0)).sum())
        dark_overlap = int(((dark > 0) & (text_mask > 0)).sum())
        if dark_overlap > max(4, white_overlap * 1.2):
            use_mask = dark
    bubble_box, bubble_mask = _find_bubble_box(use_mask, text_mask)
    if bubble_mask is not None:
        bubble_pixels = gray[bubble_mask > 0]
        if bubble_pixels.size > 0:
            bubble_mean = float(bubble_pixels.mean())
            if bubble_mean > 165:
                threshold = min(140, bubble_mean - 35)
                stroke = (gray < threshold).astype(np.uint8) * 255
            elif bubble_mean < 90:
                threshold = max(170, bubble_mean + 60)
                stroke = (gray > threshold).astype(np.uint8) * 255
            else:
                stroke = None
            if stroke is not None:
                stroke = cv2.bitwise_and(stroke, bubble_mask)
                stroke = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE, kernel, iterations=1)
                stroke = cv2.dilate(stroke, kernel, iterations=1)
                text_mask = cv2.bitwise_or(text_mask, stroke)
    adaptive_block = max(15, int(min(gray.shape[:2]) / 8) * 2 + 1)
    base_limit = cv2.dilate(text_mask, kernel, iterations=1)
    if bubble_mask is not None:
        base_limit = cv2.bitwise_and(base_limit, bubble_mask)
    try:
        ink_dark = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            adaptive_block,
            12,
        )
        ink_light = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            adaptive_block,
            12,
        )
        dark_overlap = int(((ink_dark > 0) & (base_limit > 0)).sum())
        light_overlap = int(((ink_light > 0) & (base_limit > 0)).sum())
        if max(dark_overlap, light_overlap) > 5:
            ink = ink_dark if dark_overlap >= light_overlap else ink_light
            ink = cv2.bitwise_and(ink, base_limit)
            ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=1)
            ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel, iterations=1)
            ink = cv2.dilate(ink, kernel, iterations=1)
            text_mask = cv2.bitwise_or(text_mask, ink)
    except Exception:
        pass
    if bubble_mask is not None:
        # Expand text removal slightly but keep it strictly inside the bubble area.
        expanded = cv2.dilate(text_mask, kernel, iterations=1)
        text_mask = cv2.bitwise_and(expanded, bubble_mask)
    full_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = text_mask
    if bubble_box:
        bx0, by0, bx1, by1 = bubble_box
        bubble_box = (x0 + bx0, y0 + by0, x0 + bx1, y0 + by1)
        bubble_box = _ensure_box_contains(bubble_box, pad_box)
    if bubble_mask is not None:
        full_bubble = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        full_bubble[y0:y1, x0:x1] = bubble_mask
        bubble_mask = full_bubble
    return full_mask, bubble_box, bubble_mask


def _ensure_box_contains(outer, inner):
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return (
        min(ox0, ix0),
        min(oy0, iy0),
        max(ox1, ix1),
        max(oy1, iy1),
    )


def _rect_mask(shape, bbox, x0, y0):
    mask = np.zeros(shape, dtype=np.uint8)
    x, y, w, h = [int(v) for v in bbox]
    rx0 = max(0, x - x0)
    ry0 = max(0, y - y0)
    rx1 = min(shape[1], rx0 + w)
    ry1 = min(shape[0], ry0 + h)
    cv2.rectangle(mask, (rx0, ry0), (rx1, ry1), 255, thickness=-1)
    return mask


def _find_bubble_box(white_mask, text_mask):
    num_labels, labels = cv2.connectedComponents(white_mask)
    best_label = 0
    best_overlap = 0
    if num_labels <= 1:
        return None, None
    for label in range(1, num_labels):
        overlap = np.sum((labels == label) & (text_mask > 0))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    if best_label == 0:
        return None, None
    text_pixels = int((text_mask > 0).sum())
    ys, xs = np.where(labels == best_label)
    if ys.size == 0 or xs.size == 0:
        return None, None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    area = (x1 - x0) * (y1 - y0)
    total_area = white_mask.shape[0] * white_mask.shape[1]
    if total_area > 0 and area / total_area > 0.6:
        return None, None
    if text_pixels and area > max(text_pixels * 25, 16000):
        return None, None
    bubble_mask = (labels == best_label).astype(np.uint8) * 255
    return (x0, y0, x1, y1), bubble_mask


def _intersect_box(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _limit_box(box, limit):
    bx0, by0, bx1, by1 = box
    lx0, ly0, lx1, ly1 = limit
    limited = _intersect_box(box, limit)
    if not limited:
        return box
    # If the limit is significantly smaller, use it; otherwise keep the original.
    bw = bx1 - bx0
    bh = by1 - by0
    lw = limited[2] - limited[0]
    lh = limited[3] - limited[1]
    if lw * lh < bw * bh * 0.85:
        return limited
    return box


def _normalize_polygons(polygon) -> List[List[List[float]]]:
    if polygon is None:
        return []
    if hasattr(polygon, "tolist"):
        try:
            polygon = polygon.tolist()
        except Exception:
            return []
    if isinstance(polygon, (list, tuple)) and len(polygon) > 0:
        first = polygon[0]
        if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (int, float)):
            return [polygon]
        if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (list, tuple)):
            return [p for p in polygon if p]
    return []


def _polygon_bounds(polygon) -> Tuple[int, int, int, int] | None:
    polys = _normalize_polygons(polygon)
    if not polys:
        return None
    xs = []
    ys = []
    for poly in polys:
        for point in poly:
            xs.append(point[0])
            ys.append(point[1])
    if not xs or not ys:
        return None
    x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _box_area(box) -> int:
    if not box:
        return 0
    if len(box) == 4:
        return int(max(1, box[2]) * max(1, box[3]))
    return 0


def _shrink_box(box, padding):
    x0, y0, x1, y1 = box
    x0 = x0 + padding
    y0 = y0 + padding
    x1 = x1 - padding
    y1 = y1 - padding
    if x1 <= x0 or y1 <= y0:
        return box
    return (x0, y0, x1, y1)


def _wrap_text(draw, text: str, font, max_width: int, max_lines: int | None = None) -> List[str]:
    words = _tokenize_text(text)
    lines: List[str] = []
    current = ""
    has_space = " " in text
    for idx, ch in enumerate(words):
        candidate = (current + " " + ch).strip() if has_space else current + ch
        if _is_punct_only(ch) and current:
            current = f"{current}{ch}"
            continue
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = ch
            if max_lines is not None and len(lines) >= max_lines - 1:
                remaining = words[idx + 1 :]
                if remaining:
                    current = _join_tokens([current] + remaining, has_space)
                break
    if current:
        lines.append(current)
    lines = _fix_leading_punct(lines)
    return lines


def _join_tokens(tokens: List[str], has_space: bool) -> str:
    if not tokens:
        return ""
    if not has_space:
        return "".join(tokens)
    combined = tokens[0]
    for token in tokens[1:]:
        if _is_punct_only(token):
            combined = f"{combined}{token}"
        else:
            combined = f"{combined} {token}"
    return combined


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = " ".join(part for part in cleaned.split("\n") if part.strip())
    if _has_cjk(cleaned):
        cleaned = re.sub(r"[.．]{2,}", "…", cleaned)
        cleaned = re.sub(r"…{2,}", "…", cleaned)
        cleaned = cleaned.replace("．", "。")
        cleaned = cleaned.replace(":", "：").replace(";", "；")
        cleaned = cleaned.replace("!", "！").replace("?", "？")
        cleaned = cleaned.replace(".", "。")
        cleaned = re.sub(r"\s*([。，、！？：；…])\s*", r"\1", cleaned)
    return cleaned.strip()


def _tokenize_text(text: str) -> List[str]:
    if " " in text:
        parts = [p for p in text.split(" ") if p]
        tokens: List[str] = []
        for part in parts:
            if _is_punct_only(part) and tokens:
                tokens[-1] = f"{tokens[-1]}{part}"
            else:
                tokens.append(part)
        return tokens
    tokens = []
    index = 0
    while index < len(text):
        ch = text[index]
        if ch in {".", "．"}:
            end = index
            while end < len(text) and text[end] in {".", "．"}:
                end += 1
            tokens.append(text[index:end])
            index = end
            continue
        if ch == "…":
            end = index
            while end < len(text) and text[end] == "…":
                end += 1
            tokens.append(text[index:end])
            index = end
            continue
        if _is_punct_char(ch) and tokens:
            tokens[-1] = f"{tokens[-1]}{ch}"
        else:
            tokens.append(ch)
        index += 1
    return tokens


def _is_punct_char(ch: str) -> bool:
    return ch in {
        "。",
        "．",
        "，",
        "、",
        "！",
        "？",
        "：",
        "；",
        "…",
        ".",
        ",",
        "!",
        "?",
        ":",
        ";",
        "·",
        "—",
        "～",
        "…",
    }


def _is_punct_only(text: str) -> bool:
    if not text:
        return True
    stripped = "".join(ch for ch in text if ch.strip())
    if not stripped:
        return True
    return all(_is_punct_char(ch) for ch in stripped)


def _fix_leading_punct(lines: List[str]) -> List[str]:
    if len(lines) <= 1:
        return lines
    fixed: List[str] = [lines[0]]
    for line in lines[1:]:
        line = line.lstrip()
        if line and _is_punct_char(line[0]):
            fixed[-1] = f"{fixed[-1]}{line[0]}"
            remainder = line[1:].lstrip()
            if remainder:
                fixed.append(remainder)
        else:
            fixed.append(line)
    return fixed


def _text_height(font, text: str) -> int:
    bbox = font.getbbox(text)
    return max(1, int(bbox[3] - bbox[1]))


def _fill_padding(w: int, h: int) -> int:
    base = max(12, int(min(w, h) * 0.35), int(max(w, h) * 0.1))
    return min(base, 80)


def _text_padding(w: int, h: int) -> int:
    base = max(6, int(min(w, h) * 0.12), int(max(w, h) * 0.03))
    return min(base, 28)


def _fit_font(
    draw,
    text: str,
    max_width: int,
    max_height: int,
    font_name: str,
    preferred_size: int | None = None,
):
    font_path = _find_font_path(font_name)
    sample = _sample_char(text)
    start_size = min(72, max(14, int(max_height * 0.75)))
    min_size = max(10, int(max_height * 0.18))
    if preferred_size is not None:
        target = max(min_size, min(preferred_size, start_size))
        font = _load_font(font_path, target, sample)
        lines = _wrap_text(draw, text, font, max_width)
        total_height = sum(font.getbbox(line)[3] for line in lines)
        if total_height <= max_height:
            return font
    for size in range(start_size, min_size - 1, -1):
        font = _load_font(font_path, size, sample)
        lines = _wrap_text(draw, text, font, max_width)
        total_height = sum(font.getbbox(line)[3] for line in lines)
        if total_height <= max_height:
            return font
    for size in range(min_size - 1, 9, -1):
        font = _load_font(font_path, size, sample)
        lines = _wrap_text(draw, text, font, max_width)
        total_height = sum(font.getbbox(line)[3] for line in lines)
        if total_height <= max_height:
            return font
    return _load_font(font_path, 10, sample)


def _load_font(font_path: str | None, size: int, sample: str):
    if ImageFont is None:
        raise RuntimeError("Pillow is not installed.")
    candidates = []
    if font_path and os.path.exists(font_path):
        candidates.append(font_path)
    candidates.extend(_fallback_font_paths())
    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        try:
            font = ImageFont.truetype(path, size=size)
        except Exception:
            continue
        if _font_supports_text(font, sample):
            return font
    return ImageFont.load_default()


def _font_supports_text(font, text: str) -> bool:
    if not text:
        return True
    try:
        mask = font.getmask(text)
        return mask.getbbox() is not None
    except Exception:
        return False


def _sample_char(text: str) -> str:
    for ch in text:
        if _is_cjk(ch):
            return ch
    return "A"


def _find_font_path(font_name: str) -> str | None:
    if not font_name:
        return None
    if os.path.isfile(font_name):
        return font_name
    fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    if not os.path.isdir(fonts_dir):
        return None
    lowered = font_name.lower().replace(" ", "")
    fallback_fonts = [
        "Noto Sans CJK",
        "Microsoft YaHei",
        "SimSun",
        "MS Gothic",
        "Yu Gothic",
        "Meiryo",
    ]
    for entry in os.listdir(fonts_dir):
        name, ext = os.path.splitext(entry)
        if ext.lower() not in {".ttf", ".otf", ".ttc"}:
            continue
        if lowered in name.lower().replace(" ", ""):
            return os.path.join(fonts_dir, entry)
    for font in fallback_fonts:
        lowered = font.lower().replace(" ", "")
        for entry in os.listdir(fonts_dir):
            name, ext = os.path.splitext(entry)
            if ext.lower() not in {".ttf", ".otf", ".ttc"}:
                continue
            if lowered in name.lower().replace(" ", ""):
                return os.path.join(fonts_dir, entry)
    known_files = [
        "msyh.ttc",
        "msyhbd.ttc",
        "simsun.ttc",
        "simhei.ttf",
        "msgothic.ttc",
        "yugothic.ttf",
        "yugothicui.ttf",
        "meiryo.ttc",
    ]
    for fname in known_files:
        candidate = os.path.join(fonts_dir, fname)
        if os.path.isfile(candidate):
            return candidate
    return None


def _fallback_font_paths() -> List[str]:
    fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    if not os.path.isdir(fonts_dir):
        return []
    known_files = [
        "msyh.ttc",
        "msyhbd.ttc",
        "simsun.ttc",
        "simhei.ttf",
        "msgothic.ttc",
        "yugothic.ttf",
        "yugothicui.ttf",
        "meiryo.ttc",
    ]
    paths = []
    for fname in known_files:
        candidate = os.path.join(fonts_dir, fname)
        if os.path.isfile(candidate):
            paths.append(candidate)
    return paths


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF


def _has_cjk(text: str) -> bool:
    return any(_is_cjk(ch) for ch in text)


def _is_cjk_unsupported_font(font_name: str) -> bool:
    if not font_name:
        return False
    name = font_name.lower()
    return "gothic" in name or "meiryo" in name


def _white_bubble_fill(image, text_mask):
    if cv2 is None or np is None:
        return None
    ys, xs = np.where(text_mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad = 12
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(text_mask.shape[1] - 1, x1 + pad)
    y1 = min(text_mask.shape[0] - 1, y1 + pad)
    img_np = np.array(image)
    roi = img_np[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    if mean > 220 and std < 18:
        filled = _apply_white_mask(image, text_mask)
        return filled
    return None
