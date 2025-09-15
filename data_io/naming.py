import re
from pathlib import Path
from typing import Dict, Tuple, Optional

# Case-insensitive, ignore any starting with "catch_shape"
PAT_CHANGE = re.compile(
    r"^(?!catch_shape).*shape(?P<sid>\d+)_(?P<ctype>convex|concave|concave_nofill)_area(?P<area>[12])_(?P<phase>init|out)\.png$",
    re.IGNORECASE,
)
PAT_NOCHG = re.compile(
    r"^(?!catch_shape).*shape(?P<sid>\d+)_no_change(?P<var>[1-6])_init\.png$",
    re.IGNORECASE,
)

def parse_name(name: str) -> Optional[Dict]:
    m = PAT_CHANGE.match(name)
    if m:
        d = m.groupdict()
        sid = int(d["sid"])
        area = int(d["area"])
        size = "small" if area == 1 else "large"
        return {
            "kind": "chg",
            "sid": sid,
            "ctype": d["ctype"].lower(),
            "area": area,
            "size": size,
            "phase": d["phase"].lower(),
        }
    m = PAT_NOCHG.match(name)
    if m:
        d = m.groupdict()
        sid = int(d["sid"])
        var = int(d["var"])
        return {"kind": "nochg", "sid": sid, "var": var}
    return None

def scan_pairs(src: Path):
    """
    Returns:
      change[(sid, ctype, area)] -> dict {'init': Path, 'out': Path, 'size': 'small'|'large'}
      nochange[sid] -> list[Path]
    """
    change = {}
    nochg = {}
    for p in src.glob("*.png"):
        info = parse_name(p.name)
        if not info:
            continue
        if info["kind"] == "chg":
            key = (info["sid"], info["ctype"], info["area"])
            change.setdefault(key, {"size": info["size"]})[info["phase"]] = p
        else:
            nochg.setdefault(info["sid"], []).append(p)
    return change, nochg
