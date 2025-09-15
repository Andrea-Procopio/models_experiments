#!/usr/bin/env python3
"""
Step 1: Generate proposal masks for Exp 3B images (no selection).

This script replicates the early part of the pipeline used by exp3b_correlation.py
and ChangeDetectionExperiment, but stops after generating and saving all proposal
candidate masks for each image, without choosing the best mask.

Outputs:
- processed_images/<model_prefix>_<image_name>/
  - proposals/mask_{i}.png (binary 0/255 for each candidate component)
  - original_segmentation/<image_name>_original_segmentation.png (if available)
- pairs.json (list of (base, init_path, out_path)) to drive later steps

Usage:
  python step1_generate_proposals.py \
    --images_dir /path/to/Exp3b_Images \
    --output_dir /tmp/exp3b_modular/step1 \
    --model_interface segformer \
    --model_name nvidia/segformer-b1-finetuned-ade-512-512 \
    --resume
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import sys
CUR_DIR = Path(__file__).resolve().parent
PARENT_DIR = CUR_DIR.parent  # exp3_general
GRANDPARENT_DIR = PARENT_DIR.parent  # hugging_face/model_experiments
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(GRANDPARENT_DIR))

from exp3Change import ChangeDetectionExperiment
from segformer.segformer_interface_v2_adapter import SegFormerInterfaceAdapter as SegFormerInterface
from detr.detr_interface import DetrInterface
from maskrcnn.maskrcnn_interface import MaskRCNNInterface
from sam.sam_interface import SAMInterface


def save_candidates_as_pngs(candidates: List[np.ndarray], dest_dir: Path, image_name: str) -> List[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for idx, mask in enumerate(candidates):
        mask_uint8 = (mask.astype(np.uint8) * 255)
        out_path = dest_dir / f"mask_{image_name}_{idx:03d}.png"
        Image.fromarray(mask_uint8).save(out_path)
        saved.append(str(out_path))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3B modular step1: generate proposal masks")
    parser.add_argument("--images_dir", type=Path, default=Path("/Users/andreaprocopio/Desktop/object_reps_neural/hugging_face/model_experiments/Exp3b_Images"))
    # results/<architecture>/<model_tag>/step1_out
    parser.add_argument("--output_dir", type=Path, default=Path("/Users/andreaprocopio/Desktop/object_reps_neural/hugging_face/model_experiments/exp3_general/exp3b_modular/results"))
    parser.add_argument("--model_interface", type=str, default="segformer", choices=["segformer", "detr", "maskrcnn", "sam"]) 
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Default models like exp3b_correlation
    if args.model_name is None:
        if args.model_interface == "segformer":
            args.model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        elif args.model_interface == "detr":
            args.model_name = "facebook/detr-resnet-50-panoptic"
        elif args.model_interface == "maskrcnn":
            args.model_name = "maskrcnn_resnet50_fpn"
        elif args.model_interface == "sam":
            args.model_name = "facebook/sam-vit-base"

    # Prepare nested output dirs: results/<arch>/<model_tag>/step1_out
    arch = args.model_interface
    model_tag = args.model_name.replace('/', '_') if args.model_name else arch
    step_out_dir = args.output_dir / arch / model_tag / "step1_out"
    step_out_dir.mkdir(parents=True, exist_ok=True)
    processed_images_dir = step_out_dir / "processed_images"
    processed_images_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model interface
    if args.model_interface == "segformer":
        model_if = SegFormerInterface(model_name=args.model_name)
    elif args.model_interface == "detr":
        model_if = DetrInterface(model_name=args.model_name)
    elif args.model_interface == "maskrcnn":
        model_if = MaskRCNNInterface(model_name=args.model_name)
        model_if.set_detection_sensitivity("max")
    elif args.model_interface == "sam":
        model_if = SAMInterface(model_name=args.model_name)
    else:
        raise ValueError(f"Unknown model interface: {args.model_interface}")

    # Create a CDE instance to reuse helper methods (will not run full experiment)
    # Use an internal dir to avoid empty threshold_results at the step root
    cde_internal_dir = step_out_dir / "cde_internal"
    cde = ChangeDetectionExperiment(model_interface=model_if, output_dir=str(cde_internal_dir))

    # Load model once
    if not hasattr(model_if, "model") or model_if.model is None:
        model_if.load_model()

    # Find pairs using the same logic
    pairs: List[Tuple[str, str, str]] = cde._find_image_pairs(str(args.images_dir))

    # Save pairs for downstream steps
    (step_out_dir / "pairs.json").write_text(json.dumps([
        {"base": b, "init": i, "out": o} for (b, i, o) in pairs
    ], indent=2))

    # Process each image in the pairs and save proposals
    # Include model tag in prefix for clearer provenance
    model_prefix = f"{arch}_{model_tag}"
    manifest: Dict[str, Dict[str, List[str]]] = {}

    for base, init_path, out_path in pairs:
        for which, img_path in [("init", init_path), ("out", out_path)]:
            image_name = f"{base}_{which}"

            # Skip if already done and resume requested
            prop_dir = processed_images_dir / f"{model_prefix}_{image_name}" / "proposals"
            overlay_dir = processed_images_dir / f"{model_prefix}_{image_name}" / "original_segmentation"
            blobs_dir = processed_images_dir / f"{model_prefix}_{image_name}" / "blobs"
            if args.resume and prop_dir.exists() and any(prop_dir.glob("mask_*.png")):
                saved = sorted(str(p) for p in prop_dir.glob("mask_*.png"))
                manifest.setdefault(base, {})[which] = saved
                continue

            # Load image
            frame = np.array(Image.open(img_path).convert('RGB'))
            H, W, _ = frame.shape

            # Optional overlay like CDE
            try:
                overlay_dir.mkdir(parents=True, exist_ok=True)
                cde._save_original_segmentation_overlay(frame, str(overlay_dir), image_name)
            except Exception:
                pass

            # Save blob detection mask and overlay like CDE
            try:
                blobs_dir.mkdir(parents=True, exist_ok=True)
                blob = cde._detect_blob(frame)
                if blob is not None:
                    Image.fromarray((blob.astype(np.uint8) * 255)).save(blobs_dir / f"blob_{image_name}.png")
                    cde._save_blob_overlay(frame, blob, str(blobs_dir), image_name)
            except Exception:
                pass

            # Generate candidate masks
            candidates = cde._run_model_inference(frame, H, W)
            saved = save_candidates_as_pngs(candidates, prop_dir, image_name)
            manifest.setdefault(base, {})[which] = saved

    # Write manifest for step 2
    (step_out_dir / "proposals_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Remove empty threshold_results if created internally
    try:
        thr_dir = cde_internal_dir / "threshold_results"
        if thr_dir.exists() and not any(thr_dir.iterdir()):
            thr_dir.rmdir()
    except Exception:
        pass

    print(f"Done. Proposals saved under: {processed_images_dir}")


if __name__ == "__main__":
    main()


