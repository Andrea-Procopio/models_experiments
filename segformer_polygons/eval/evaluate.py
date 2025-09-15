import json, os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from config import DATA_DIR, MODEL_ID, OVERLAY_OUT
from training.dataset import load_datasets  # reuse processor
from eval.metrics import iou_dice, boundary_f1, concavity_fill_index, overfill_index, underfill_index
from viz.render import overlay_mask, annotate

def load_model(checkpoint_dir: str):
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, processor, device

def predict_mask(model, processor, device, pil_img: Image.Image):
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs)
    # back to native size
    pred = processor.post_process_semantic_segmentation(
        out, target_sizes=[pil_img.size[::-1]]
    )[0].cpu().numpy().astype(np.uint8)
    return (pred == 1).astype(np.uint8)  # binary 0/1

def evaluate(ckpt_dir: str, radii_json: str, split="val", save_overlays=True):
    model, processor, device = load_model(ckpt_dir)
    meta = json.load(open(Path(DATA_DIR) / "meta_shape.json"))
    items = [m for m in meta if m["split"] == split]

    radii = json.load(open(radii_json)) if Path(radii_json).exists() else {"small": 6, "large": 10}

    rows = []
    out_dir = Path(OVERLAY_OUT) / split
    if save_overlays:
        out_dir.mkdir(parents=True, exist_ok=True)

    for rec in tqdm(items, desc=f"eval-{split}"):
        img = Image.open(rec["image"]).convert("RGB")
        gt  = (np.array(Image.open(rec["mask"]).convert("L")) > 0).astype(np.uint8)

        pred = predict_mask(model, processor, device, img)

        iou, dice = iou_dice(pred, gt)
        bf1 = boundary_f1(pred, gt)

        size = rec["size"]
        ctype = rec["ctype"]
        # CFI only meaningful for concavity-related classes, but it's defined generally.
        r = radii.get(size, 0) if size in ("small","large") else 0
        cfi = concavity_fill_index(pred, gt, r) if r > 0 else 0.0
        ofi = overfill_index(pred, gt, r) if r > 0 else 0.0
        ufi = underfill_index(pred, gt, r) if r > 0 else 0.0

        rows.append({
            "sid": rec["sid"], "ctype": ctype, "size": size, "phase": rec["phase"],
            "IoU": iou, "Dice": dice, "BoundaryF1": bf1, "CFI": cfi, "OFI": ofi, 
            "UFI": ufi, "image": rec["image"], "mask": rec["mask"]
        })

        if save_overlays:
            ov = overlay_mask(img, pred)
            txt = f"{ctype}/{size} IoU={iou:.3f} Dice={dice:.3f} CFI={cfi:.3f} OFI={ofi:.3f} UFI={ufi:.3f}"
            ov = annotate(ov, txt)
            ov.save(out_dir / (Path(rec["image"]).name.replace(".png", "_pred.png")))

    # aggregate by (ctype, size)
    import itertools, statistics as stats
    groups = {}
    keyfunc = lambda r: (r["ctype"], r["size"])
    for k, group in itertools.groupby(sorted(rows, key=keyfunc), key=keyfunc):
        g = list(group)
        def mean(field): 
            vals = [r[field] for r in g if not (np.isnan(r[field]) or r[field] is None)]
            return float(sum(vals) / max(1, len(vals)))
        groups[k] = {
            "count": len(g),
            "IoU": mean("IoU"),
            "Dice": mean("Dice"),
            "BoundaryF1": mean("BoundaryF1"),
            "CFI": mean("CFI"),
            "OFI": mean("OFI"),
            "UFI": mean("UFI"),
        }

    out_json = Path(ckpt_dir) / f"eval_{split}.json"
    # Convert tuple-keyed groups to a list of serializable records for JSON
    groups_list = [{"ctype": k[0], "size": k[1], **v} for k, v in groups.items()]
    json.dump({"per_image": rows, "by_group": groups_list}, open(out_json, "w"), indent=2)
    print("Saved:", out_json)
    print("Grouped metrics:")
    for (ctype, size), m in groups.items():
        print(f"{ctype:15s} {size:6s} n={m['count']:3d} | IoU {m['IoU']:.3f}  Dice {m['Dice']:.3f}  BF1 {m['BoundaryF1']:.3f}  CFI {m['CFI']:.3f}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to fine-tuned checkpoint dir (training output)")
    ap.add_argument("--radii", default="runs/radii.json", help="Calibrated radii JSON (from calibration.py)")
    ap.add_argument("--split", default="val", choices=["train","val"])
    ap.add_argument("--no_overlays", action="store_true")
    args = ap.parse_args()
    evaluate(args.ckpt, args.radii, split=args.split, save_overlays=not args.no_overlays)

if __name__ == "__main__":
    main()