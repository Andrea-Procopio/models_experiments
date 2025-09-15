# diag_calib_pairs.py
from pathlib import Path
from data_io.naming import scan_pairs
from data_io.gt import gt_mask_from_path
from config import RAW_DIR, THRESHOLDS

change_pairs, _ = scan_pairs(Path(RAW_DIR))
n = 0
for (sid, ctype, area), grp in change_pairs.items():
    if ctype!="concave_nofill" or "out" not in grp: continue
    key_c = (sid, "concave", area)
    if key_c in change_pairs and "out" in change_pairs[key_c]:
        n += 1
print("usable pairs:", n)
