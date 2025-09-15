#!/usr/bin/env python3
"""
SegFormer runner with LETTERBOX pre/post, CONFIDENCE-based foreground, and NEAREST mask upsampling.

- No anisotropic stretch: letterbox to a target long side (or run native with do_resize=False).
- Foreground from max softmax probability (class-agnostic), thresholded at tau.
- Connected components on fg mask; keep top-k near center with min area ratio.
- Returns binary mask(s) in original image coordinates.
"""

from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from skimage.measure import label, regionprops
    _HAVE_SKIMAGE = True
except Exception:
    _HAVE_SKIMAGE = False


def letterbox_reflect_np(img: np.ndarray, target_long_side: int) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """Return (padded_img, pads) where pads=(top, bottom, left, right)."""
    h, w = img.shape[:2]
    if target_long_side is None or target_long_side <= 0:
        return img, (0,0,0,0)
    long_side = max(h, w)
    if long_side == target_long_side:
        return img, (0,0,0,0)
    scale = target_long_side / float(long_side)
    nh = max(1, int(round(h * scale))); nw = max(1, int(round(w * scale)))
    pil = Image.fromarray(img, mode="RGB").resize((nw, nh), resample=Image.BILINEAR)
    arr = np.array(pil, dtype=np.uint8)
    top = (target_long_side - nh) // 2
    bottom = target_long_side - nh - top
    left = (target_long_side - nw) // 2
    right = target_long_side - nw - left
    arr = np.pad(arr, ((top,bottom),(left,right),(0,0)), mode='reflect')
    return arr, (top, bottom, left, right)


def unletterbox(mask: np.ndarray, pads: Tuple[int,int,int,int], orig_hw: Tuple[int,int]) -> np.ndarray:
    """Remove letterbox padding and resize back to original with NEAREST."""
    top, bottom, left, right = pads
    h0, w0 = orig_hw
    cropped = mask[top:mask.shape[0]-bottom, left:mask.shape[1]-right]
    pil = Image.fromarray((cropped.astype(np.uint8) * 255), mode="L").resize((w0, h0), resample=Image.NEAREST)
    return (np.array(pil) > 127).astype(np.uint8)


def _cc_largest_k_center(mask: np.ndarray, k:int=2, min_area_ratio:float=0.002) -> List[np.ndarray]:
    """Keep up to k components prioritized by center proximity, then area."""
    h, w = mask.shape
    min_area = max(1000, int(min_area_ratio*h*w))
    if _HAVE_SKIMAGE:
        labeled = label(mask, connectivity=2)
        props = regionprops(labeled)
        center = np.array([h/2.0, w/2.0], dtype=np.float32)
        cand = []
        for pr in props:
            if pr.area < min_area:
                continue
            dist = float(np.linalg.norm(np.array(pr.centroid) - center))
            cand.append((dist, -pr.area, pr.label))
        cand.sort(key=lambda t: (t[0], t[1]))
        out = []
        for _,__, lab in cand[:k]:
            out.append((labeled == lab).astype(np.uint8))
        return out if out else [mask.astype(np.uint8)]
    # Fallback: single mask if skimage not available
    return [mask.astype(np.uint8)]


class SegFormerRunner:
    def __init__(self, model, processor, device: Optional[str] = None, target_long_side: Optional[int] = None, tau: float = 0.6):
        """
        model: HuggingFace SegFormerForSemanticSegmentation
        processor: SegformerImageProcessor with do_resize=False (we letterbox ourselves) and do_normalize=True
        target_long_side: if set, letterbox inputs to this size; else run at native resolution.
        tau: threshold on max softmax probability for foreground.
        """
        self.model = model.eval()
        self.processor = processor
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.target_long_side = target_long_side
        self.tau = float(tau)

    @torch.no_grad()
    def segment(self, img_uint8: np.ndarray, topk:int=1, return_probs: bool=False):
        """Return dict with binary masks in original coordinates and optional prob map."""
        h0, w0 = img_uint8.shape[:2]
        if self.target_long_side:
            padded, pads = letterbox_reflect_np(img_uint8, self.target_long_side)
        else:
            padded, pads = img_uint8, (0,0,0,0)

        inputs = self.processor(images=Image.fromarray(padded, mode="RGB"),
                                return_tensors="pt", do_resize=False, do_normalize=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # logits: [B, C, H', W']
        logits = outputs.logits  # float32
        probs = F.softmax(logits, dim=1)      # [B, C, H', W']
        fg_prob, _ = probs.max(dim=1)         # [B, H', W'] class-agnostic foreground confidence

        # Upsample to padded size
        fg_prob_up = F.interpolate(fg_prob.unsqueeze(1), size=(padded.shape[0], padded.shape[1]),
                                   mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
        fg_mask = (fg_prob_up[0] >= self.tau).astype(np.uint8)

        # Unletterbox to original size
        fg_mask_orig = unletterbox(fg_mask, pads, (h0, w0))

        # Connected components; keep top-k by center proximity then area
        masks = _cc_largest_k_center(fg_mask_orig, k=max(1, int(topk)))

        out = {"masks": masks}
        if return_probs:
            out["fg_prob_padded"] = fg_prob_up[0]  # for debugging/visualization
        return out