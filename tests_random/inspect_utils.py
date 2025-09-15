from pathlib import Path
from data_io.naming import scan_pairs
from data_io.gt import gt_mask_from_path
from skimage.morphology import closing, disk
from config import RAW_DIR, THRESHOLDS

cp,_ = scan_pairs(Path(RAW_DIR))
for (sid, ctype, area), grp in cp.items():
    if ctype!="concave_nofill" or "out" not in grp: continue
    key_c = (sid,"concave",area)
    if key_c not in cp or "out" not in cp[key_c]: continue
    m_n = gt_mask_from_path(grp["out"], THRESHOLDS).astype(bool)
    m_c = gt_mask_from_path(cp[key_c]["out"], THRESHOLDS).astype(bool)
    print("sid",sid,"area",area,"size",grp["size"], "A_nofill", int(m_n.sum()), "A_conc", int(m_c.sum()))
    for r in [1,2,4,6,8,10]:
        A = int(closing(m_n, footprint=disk(r)).sum())
        print("  r=",r,"A_close",A,"Δ=",A-int(m_c.sum()))
    break