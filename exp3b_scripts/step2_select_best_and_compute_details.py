#!/usr/bin/env python3
"""
Step 2: Select optimal masks and compute per-image details for Exp 3B.

Inputs from step1:
- pairs.json
- processed_images/<model_prefix>_<image_name>/proposals/mask_<image_name>_###.png

This script mirrors the selection logic in ChangeDetectionExperiment:
- Size filtering by ratio to blob (min_size_ratio, max_size_ratio)
- IoU-based selection with minimum IoU threshold

Outputs:
- processed_images/.../frames_masks_nonmem/mask_<image_name>.png (chosen mask)
- threshold_results/per_image_detailed.json (list of entries with base, type, areas, area_change)
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

import sys
CUR_DIR = Path(__file__).resolve().parent
PARENT_DIR = CUR_DIR.parent
GRANDPARENT_DIR = PARENT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(GRANDPARENT_DIR))

from exp3Change import ChangeDetectionExperiment


def load_binary(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert('L'))
    return (arr > 0).astype(bool)


def compute_blob_stats(mask: np.ndarray) -> Dict[str, float]:
    if mask is None or mask.sum() == 0:
        return {'area': 0.0, 'centroid_x': 0.0, 'centroid_y': 0.0, 'perimeter': 0.0}
    mask_uint8 = mask.astype(np.uint8)
    labeled = label(mask_uint8, connectivity=2)
    props = regionprops(labeled)
    if not props:
        return {'area': 0.0, 'centroid_x': 0.0, 'centroid_y': 0.0, 'perimeter': 0.0}
    prop = props[0]
    cy, cx = prop.centroid
    return {'area': float(prop.area), 'centroid_x': float(cx), 'centroid_y': float(cy), 'perimeter': float(prop.perimeter)}


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def choose_best_mask(blob: np.ndarray,
                     candidates: List[np.ndarray],
                     min_size_ratio: float,
                     max_size_ratio: float,
                     iou_threshold: float) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if not candidates:
        return None, None
    blob_area = float(blob.sum())

    # Helper: resize candidate to blob shape if needed (nearest to keep binary)
    def resize_to_blob_shape(mask: np.ndarray) -> np.ndarray:
        if mask.shape == blob.shape:
            return mask.astype(bool)
        target_h, target_w = blob.shape
        pil = Image.fromarray((mask.astype(np.uint8) * 255))
        pil = pil.resize((target_w, target_h), resample=Image.NEAREST)
        arr = np.array(pil)
        return (arr > 0).astype(bool)

    # Normalize candidate shapes before filtering and IoU
    normalized: List[np.ndarray] = [resize_to_blob_shape(m) for m in candidates]

    filtered: List[np.ndarray] = []
    for m in normalized:
        area = float(m.sum())
        ratio = (area / blob_area) if blob_area > 0 else float('inf')
        if min_size_ratio <= ratio <= max_size_ratio:
            filtered.append(m)
    if not filtered:
        return None, None
    ious = np.array([compute_iou(blob, m) for m in filtered], dtype=np.float32)
    best_idx = int(ious.argmax())
    best_iou = float(ious[best_idx])
    if best_iou < iou_threshold:
        return None, None
    return filtered[best_idx].astype(bool), -best_iou


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3B modular step2: select masks and compute per-image details")
    # Expect step1_dir to be results/<arch>/<model_tag>/step1_out
    parser.add_argument("--step1_dir", type=Path, default=Path("/Users/andreaprocopio/Desktop/object_reps_neural/hugging_face/model_experiments/exp3_general/exp3b_modular/results/segformer/nvidia_segformer-b0-finetuned-ade-512-512/step1_out"))
    # Output to results/<arch>/<model_tag>/step2_out
    parser.add_argument("--output_dir", type=Path, default=Path("/Users/andreaprocopio/Desktop/object_reps_neural/hugging_face/model_experiments/exp3_general/exp3b_modular/results/segformer/nvidia_segformer-b0-finetuned-ade-512-512/step2_out"))
    parser.add_argument("--model_interface", type=str, default="segformer", choices=["segformer", "detr", "maskrcnn", "sam"]) 
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--iou_threshold", type=float, default=0.1)
    parser.add_argument("--max_size_ratio", type=float, default=7.0)
    parser.add_argument("--min_size_ratio", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed_images_dir = args.output_dir / "processed_images"
    threshold_results_dir = args.output_dir / "threshold_results"
    processed_images_dir.mkdir(parents=True, exist_ok=True)
    threshold_results_dir.mkdir(parents=True, exist_ok=True)

    # Create a CDE instance to reuse naming and type parsing
    # Note: We'll only use its helpers (_get_model_prefix, _img_type_from_name)
    # Model is not required here.
    # Align interface name to ensure model_prefix identical to step1
    class SegFormerInterface:
        def __init__(self, model_name: str):
            self.model_name = model_name
    dummy_if = SegFormerInterface(args.model_name or args.model_interface)
    cde = ChangeDetectionExperiment(model_interface=dummy_if, output_dir=str(args.output_dir),
                                    iou_threshold=args.iou_threshold,
                                    max_size_ratio=args.max_size_ratio,
                                    min_size_ratio=args.min_size_ratio)

    pairs_path = args.step1_dir / "pairs.json"
    if not pairs_path.exists():
        raise FileNotFoundError(f"pairs.json not found in {args.step1_dir}")
    pairs = json.loads(pairs_path.read_text())

    # Infer model_prefix consistent with step1: results/<arch>/<model_tag>/step1_out
    try:
        # step1_dir = .../results/<arch>/<model_tag>/step1_out
        model_tag = args.step1_dir.parent.name
        arch = args.step1_dir.parent.parent.name
        model_prefix = f"{arch}_{model_tag}"
    except Exception:
        # Fallback to type-based prefix
        model_prefix = cde._get_model_prefix()

    per_image_details: List[Dict] = []

    for item in pairs:
        base = item["base"]
        init_path = item["init"]
        out_path = item["out"]

        # Load proposals from step1
        def load_proposals(which: str) -> List[np.ndarray]:
            image_name = f"{base}_{which}"
            prop_dir = (args.step1_dir / "processed_images" / f"{model_prefix}_{image_name}" / "proposals")
            if not prop_dir.exists():
                return []
            mask_paths = sorted(prop_dir.glob(f"mask_{image_name}_*.png"))
            return [load_binary(p) for p in mask_paths]

        # Prefer the saved blob from step1; fallback to proxy from proposals.
        def proxy_blob(which: str, candidates: List[np.ndarray]) -> Optional[np.ndarray]:
            image_name = f"{base}_{which}"
            blob_path = (args.step1_dir / "processed_images" / f"{model_prefix}_{image_name}" / "blobs" / f"blob_{image_name}.png")
            if blob_path.exists():
                return load_binary(blob_path)
            best = None
            best_area = 0
            for m in candidates:
                area = int(m.sum())
                if area > best_area:
                    best = m
                    best_area = area
            return best

        init_candidates = load_proposals("init")
        out_candidates = load_proposals("out")

        init_blob = proxy_blob("init", init_candidates)
        out_blob = proxy_blob("out", out_candidates)

        # Select best masks using same constraints
        chosen_init, _ = choose_best_mask(init_blob, init_candidates, args.min_size_ratio, args.max_size_ratio, args.iou_threshold) if init_blob is not None else (None, None)
        chosen_out, _ = choose_best_mask(out_blob, out_candidates, args.min_size_ratio, args.max_size_ratio, args.iou_threshold) if out_blob is not None else (None, None)

        # Fallback to blob if selection fails
        if chosen_init is None:
            chosen_init = init_blob if init_blob is not None else np.zeros((1,1), dtype=bool)
        if chosen_out is None:
            chosen_out = out_blob if out_blob is not None else chosen_init

        # Save chosen masks
        for which, chosen in [("init", chosen_init), ("out", chosen_out)]:
            image_name = f"{base}_{which}"
            mask_dir = processed_images_dir / f"{model_prefix}_{image_name}" / "frames_masks_nonmem"
            mask_dir.mkdir(parents=True, exist_ok=True)
            out_file = mask_dir / f"mask_{image_name}.png"
            Image.fromarray((chosen.astype(np.uint8) * 255)).save(out_file)

        # Stats and details
        init_stats = compute_blob_stats(chosen_init)
        out_stats = compute_blob_stats(chosen_out)
        area_change_ratio = None
        if init_stats['area'] > 0:
            area_change_ratio = abs(out_stats['area'] - init_stats['area']) / init_stats['area']

        img_type = cde._img_type_from_name(base)
        per_image_details.append({
            'base': base,
            'type': img_type,
            'before_mask': str(processed_images_dir / f"{model_prefix}_{base}_init" / "frames_masks_nonmem" / f"mask_{base}_init.png"),
            'after_mask': str(processed_images_dir / f"{model_prefix}_{base}_out" / "frames_masks_nonmem" / f"mask_{base}_out.png"),
            'area_before': int(init_stats['area']),
            'area_after': int(out_stats['area']),
            'area_change': area_change_ratio
        })

    # Write only a flat copy at threshold_results root for step3 convenience
    (threshold_results_dir / "per_image_detailed.json").write_text(json.dumps(per_image_details, indent=2))

    print(f"Done. Wrote per_image_detailed.json with {len(per_image_details)} entries to {threshold_results_dir}")


if __name__ == "__main__":
    main()


