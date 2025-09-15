import numpy as np
from PIL import Image
from typing import Iterable
from skimage.measure import label, regionprops

def gt_mask_from_path(path, thresholds: Iterable[int], min_abs_area=1000, min_rel_area=0.002):
    """
    Deterministic GT via thresholding.
    - keeps ALL components above size floor (preserves holes)
    - foreground=1, background=0
    """
    gray = np.array(Image.open(path).convert("L"))
    H, W = gray.shape
    img_area = H * W

    for thr in thresholds:
        cand = (gray > thr).astype(np.uint8)
        lab = label(cand, connectivity=2)
        out = np.zeros_like(cand, dtype=np.uint8)
        for r in regionprops(lab):
            if r.area >= max(min_abs_area, min_rel_area * img_area):
                out[lab == r.label] = 1
        if out.any():
            return out
    return np.zeros_like(gray, dtype=np.uint8)
