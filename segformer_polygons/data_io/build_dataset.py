import json, random, shutil
from pathlib import Path
from typing import Tuple, List
from PIL import Image
import numpy as np
from tqdm import tqdm

from .naming import scan_pairs, parse_name
from .gt import gt_mask_from_path

from config import RAW_DIR, DATA_DIR, VAL_FRACTION, RNG_SEED, THRESHOLDS

def split_by_shape(shape_ids: List[int], val_frac: float, seed: int):
    rnd = random.Random(seed)
    ids = sorted(set(shape_ids))
    rnd.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    return set(ids[n_val:]), set(ids[:n_val])

def save_mask(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)

def save_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def build():
    src = Path(RAW_DIR)
    out = Path(DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)

    change_pairs, nochange = scan_pairs(src)
    all_sids = {sid for (sid, _, _) in change_pairs.keys()} | set(nochange.keys())
    train_ids, val_ids = split_by_shape(list(all_sids), VAL_FRACTION, RNG_SEED)

    meta = []

    def process_one(split: str, img_path: Path, sid: int, ctype: str, area: int, size: str, phase: str):
        mask = gt_mask_from_path(img_path, THRESHOLDS)
        img_out = out / split / "images" / img_path.name
        msk_out = out / split / "masks" / img_path.name
        save_image(img_path, img_out)
        save_mask(mask, msk_out)
        meta.append({
            "split": split,
            "sid": sid,
            "ctype": ctype,
            "area": area,
            "size": size,
            "phase": phase,
            "image": str(img_out),
            "mask": str(msk_out),
        })

    # Changed cases
    for (sid, ctype, area), group in tqdm(change_pairs.items(), desc="changed"):
        size = group["size"]
        split = "train" if sid in train_ids else "val"
        for phase in ["init", "out"]:
            if phase in group:
                process_one(split, group[phase], sid, ctype, area, size, phase)

    # No-change cases
    for sid, paths in tqdm(nochange.items(), desc="no_change"):
        split = "train" if sid in train_ids else "val"
        for p in paths:
            info = parse_name(p.name)
            # info is 'nochg'
            process_one(split, p, sid, "no_change", area=0, size="na", phase="init")

    with open(out / "meta_shape.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. Train items: {sum(1 for m in meta if m['split']=='train')} | Val items: {sum(1 for m in meta if m['split']=='val')}")

if __name__ == "__main__":
    build()
