import numpy as np
from PIL import Image, ImageDraw, ImageFont

def overlay_mask(img: Image.Image, mask: np.ndarray, alpha=0.45):
    base = np.asarray(img).astype(np.float32)
    color = np.zeros_like(base); color[...,1] = 255  # green
    m = (mask.astype(bool))[..., None]
    out = (base * (1 - alpha) + color * alpha * m).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)

def annotate(img: Image.Image, text: str):
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 22), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return img
