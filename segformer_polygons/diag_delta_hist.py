# diag_delta_hist.py
from pathlib import Path
from data_io.naming import scan_pairs
from data_io.gt import gt_mask_from_path
from config import RAW_DIR, THRESHOLDS
import numpy as np

change_pairs, _ = scan_pairs(Path(RAW_DIR))
rows = []
for (sid, ctype, area), grp in change_pairs.items():
    if "out" not in grp: continue
    m = gt_mask_from_path(grp["out"], THRESHOLDS)
    rows.append((grp["size"], int(m.sum()), sid, ctype, area))
import statistics as s
for sz in ("small","large"):
    vals = [a for (size,a,_,_,_) in rows if size==sz]
    if vals:
        print(sz, "n=", len(vals), "meanA=", s.mean(vals), "medianA=", s.median(vals))
