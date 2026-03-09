# -*- coding: utf-8 -*-
"""AI inpainting using LaMa (Large Mask Inpainting)."""
from __future__ import annotations
from functools import lru_cache

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None


def clear_model_cache():
    """Clear the model cache to allow loading a different model."""
    _load_lama_model.cache_clear()


@lru_cache(maxsize=1)
def _load_lama_model(device: str):
    """Load the LaMa model."""
    import os
    import torch
    
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise RuntimeError("simple-lama-inpainting is not installed. Run: pip install simple-lama-inpainting") from exc
    
    # Check if we have a local model that should be copied to cache
    app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    local_path = os.path.join(app_root, "models", "lama", "big-lama.pt")
    if os.path.exists(local_path):
        try:
            hub_dir = torch.hub.get_dir()
            cache_dir = os.path.join(hub_dir, "checkpoints")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "big-lama.pt")
            
            # Prefer system cache (if it exists, do NOT overwrite from local)
            if not os.path.exists(cache_path):
                print(f"[AI Inpaint] Copying local model to cache: {cache_path}")
                import shutil
                shutil.copy2(local_path, cache_path)
            else:
                 print(f"[AI Inpaint] Found existing system model: {cache_path}")
        except Exception as e:
            print(f"[AI Inpaint] Failed to copy local model: {e}")
    
    print(f"[AI Inpaint] Loading SimpleLama model on {device}")
    lama = SimpleLama(device=torch.device(device))
    print("[AI Inpaint] SimpleLama model loaded successfully")
    return lama


def ai_inpaint(image, mask, use_gpu: bool = True, model_id: str = "dreMaz/AnimeMangaInpainting"):
    """Perform AI inpainting using LaMa model.
    
    Args:
        image: PIL Image to inpaint
        mask: numpy array mask (255 = inpaint, 0 = keep)
        use_gpu: whether to use GPU (CUDA)
        model_id: (ignored - uses default SimpleLama)
    
    Returns:
        PIL Image with inpainted regions
    """
    if Image is None:
        raise RuntimeError("Pillow is not installed.")
    
    # Import cv2 and numpy for mask processing
    try:
        import cv2
        import numpy as np
    except ImportError:
        cv2 = None
        np = None
    
    if cv2 is None or np is None:
        raise RuntimeError("cv2 and numpy are required for AI inpainting")
    
    # Dilate mask to ensure full text coverage - balanced to avoid removing bubbles
    kernel_size = max(5, int(max(mask.shape) * 0.005))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    
    device = "cuda" if use_gpu else "cpu"
    
    try:
        lama = _load_lama_model(device)
    except Exception as e:
        print(f"[AI Inpaint] Failed to load LaMa model: {e}")
        raise
    
    # Convert mask to PIL Image
    mask_image = Image.fromarray(dilated_mask).convert("L")
    
    # Get bounding box of mask
    bbox = mask_image.getbbox()
    if not bbox:
        return image
    
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    
    # Add padding for better context
    pad = max(32, int(max(w, h) * 0.2))
    cx0 = max(0, x0 - pad)
    cy0 = max(0, y0 - pad)
    cx1 = min(image.width, x1 + pad)
    cy1 = min(image.height, y1 + pad)
    
    crop_w = cx1 - cx0
    crop_h = cy1 - cy0
    
    # Crop for processing
    crop_img = image.crop((cx0, cy0, cx1, cy1))
    crop_mask = mask_image.crop((cx0, cy0, cx1, cy1))
    
    # Run LaMa inpainting
    print(f"[AI Inpaint] Processing region: {crop_w}x{crop_h}")
    result = lama(crop_img, crop_mask)
    
    # LaMa might return a different size, resize to match original crop
    if result.size != (crop_w, crop_h):
        print(f"[AI Inpaint] Resizing result from {result.size} to {(crop_w, crop_h)}")
        result = result.resize((crop_w, crop_h), Image.LANCZOS)
    
    # Ensure mask is same size as result
    if crop_mask.size != result.size:
        crop_mask = crop_mask.resize(result.size, Image.NEAREST)
    
    # Composite only within mask region
    out_crop = Image.composite(result, crop_img, crop_mask)
    
    # Paste back into original image
    out = image.copy()
    out.paste(out_crop, (cx0, cy0))
    
    print("[AI Inpaint] Success")
    return out
