import json
from pathlib import Path
import numpy as np
from skimage.morphology import closing, disk

from data_io.naming import scan_pairs
from data_io.gt import gt_mask_from_path
from config import RAW_DIR, THRESHOLDS

def calib_radius_by_size(max_r=25):
    """
    For each (sid, area) with both concave_nofill and concave OUT images,
    pick r minimizing | area(closing(nofill,r)) - area(concave) |.
    Return median r per size ('small','large').
    """
    src = Path(RAW_DIR)
    change_pairs, _ = scan_pairs(src)
    bucket = {"small": [], "large": []}

    for (sid, ctype, area), group in change_pairs.items():
        if ctype != "concave_nofill" or "out" not in group:
            continue
        key_c = (sid, "concave", area)
        if key_c not in change_pairs or "out" not in change_pairs[key_c]:
            continue
        size = group["size"]  # small/large
        m_n = gt_mask_from_path(group["out"], THRESHOLDS).astype(bool)
        m_c = gt_mask_from_path(change_pairs[key_c]["out"], THRESHOLDS).astype(bool)
        target_A = int(m_c.sum())

        best = None
        for r in range(1, max_r + 1):
            A = int(closing(m_n, footprint=disk(r)).sum())
            err = abs(A - target_A)
            best = (err, r) if best is None or err < best[0] else best
        bucket[size].append(best[1])

    r_small = int(np.median(bucket["small"])) if bucket["small"] else 5
    r_large = int(np.median(bucket["large"])) if bucket["large"] else 9

    # Enforce minimum radius and separation between small/large
    MIN_R = 3
    SEP   = 2
    r_small = max(MIN_R, r_small)
    r_large = max(MIN_R, r_large)
    if r_large < r_small + SEP:
        r_large = r_small + SEP
    return {"small": r_small, "large": r_large}

def main(out_json="runs/radii.json"):
    radii = calib_radius_by_size()
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(radii, open(out_json, "w"), indent=2)
    print("Calibrated radii:", radii)

if __name__ == "__main__":
    main()
