"""
Experiment 3B – model-vs-human correlation on object-change detection.

* Takes BEFORE / AFTER image pairs (same file-name stem + '_init', '_out').
* Runs ChangeDetectionExperiment once to get the best mask for each image.
* Computes |area_after – area_before| / area_before for every pair.
* Sweeps %-change thresholds from 0.1% to 5% in 0.1% steps (default) and classifies
  each pair as "different" (1) or "same" (0).
* Correlates those binary model decisions with mean human judgements.
* Saves JSON results, .txt summary, and three PNG plots.

Usage
-----
python exp3b_correlation.py \
    --images_dir  /path/to/exp3b_imgs \
    --human_csv   /path/to/human_data.csv \
    --output_dir  /tmp/exp3b_out \
    --model_interface segformer \
    --model_name  nvidia/segformer-b0-finetuned-ade-512-512 \
    --resume
"""

## remember to activate the virtual environment before running the script
## source venv_exp3b/bin/activate
## remember to change the paths in the script to the correct ones
## 
## For SAM models, also ensure SAM dependencies are installed:
## pip install -r sam/requirements.txt


from __future__ import annotations
import argparse, json, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import torch  # only to enforce no-grad

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exp3Change import ChangeDetectionExperiment
from segformer.segformer_interface import SegFormerInterface
from detr.detr_interface import DetrInterface
from maskrcnn.maskrcnn_interface import MaskRCNNInterface
from sam.sam_interface import SAMInterface


# helper functions
# ----------------------------------------------------------------------
def load_human_data(csv_path: Path) -> Dict[str, float]:
    """Return {fullShapeName: mean_diff_response (0-1)}."""
    df = (pd.read_csv(csv_path)
            .query("~shape.str.contains('catch_shape', na=False)", engine="python")
            .query("response in ['same','different']"))
    df["binary"] = (df["response"] == "different").astype(float)
    means = df.groupby("fullShapeName")["binary"].mean()
    return {shape_name: mean for shape_name, mean in means.items()}


def collect_area_ratios(cde: ChangeDetectionExperiment) -> Dict[str, float]:
    """
    Read CDE's `threshold_results_dir/*/per_image_detailed.json` and return
    {base_stem: area_change_ratio}.
    We look only at the *first* results file because the ratio is
    threshold-independent.
    """
    res_files = list(Path(cde.threshold_results_dir).rglob("per_image_detailed.json"))
    if not res_files:
        raise FileNotFoundError(
            f"No per_image_detailed.json found under {cde.threshold_results_dir}. "
            "Did exp3Change run successfully?")
    data = json.loads(Path(res_files[0]).read_text())
    return {d["base"]: d["area_change"] for d in data}


def load_base_types(cde: ChangeDetectionExperiment) -> Dict[str, str]:
    """Return {base_stem: type} from the first per_image_detailed.json."""
    res_files = list(Path(cde.threshold_results_dir).rglob("per_image_detailed.json"))
    if not res_files:
        raise FileNotFoundError(
            f"No per_image_detailed.json found under {cde.threshold_results_dir}. "
            "Did exp3Change run successfully?")
    data = json.loads(Path(res_files[0]).read_text())
    return {d["base"]: d.get("type", "unknown") for d in data}


def compute_category_rates(values_by_base: Dict[str, float],
                           base_to_type: Dict[str, str],
                           is_binary: bool) -> Dict[str, float]:
    """
    Compute mean value per category.
    - values_by_base: {base: float} where float is either binary (0/1) or probability [0,1]
    - base_to_type: {base: type}
    - is_binary: True if values are 0/1 decisions
    Returns {type: mean_rate}
    """
    buckets: Dict[str, List[float]] = {}
    for base, val in values_by_base.items():
        t = base_to_type.get(base)
        if t is None:
            continue
        buckets.setdefault(t, []).append(float(val))
    return {t: (float(np.mean(v)) if len(v) > 0 else float("nan")) for t, v in buckets.items()}


def rmse_by_categories(model_rates: Dict[str, float], human_rates: Dict[str, float]) -> float:
    """Unweighted RMSE across categories present in both dicts."""
    keys = sorted(set(model_rates.keys()) & set(human_rates.keys()))
    if not keys:
        return float("nan")
    diffs = [(model_rates[k] - human_rates[k]) ** 2 for k in keys
             if not np.isnan(model_rates[k]) and not np.isnan(human_rates[k])]
    if not diffs:
        return float("nan")
    return float(np.sqrt(np.mean(diffs)))


def decide_different(area_ratio: float, thr: float) -> int:
    """Return 1 if |Δarea|/area_before > thr else 0."""
    return int(area_ratio > thr)


def correlations(model: List[int], human: List[float]) -> Tuple[float, float]:
    """Pearson r, Spearman ρ. Return (np.nan, np.nan) if <2 samples."""
    if len(model) < 2 or len(set(model)) == 1:
        return float("nan"), float("nan")
    return pearsonr(model, human)[0], spearmanr(model, human)[0]


# main function
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, default="/Users/andreaprocopio/Desktop/object_reps_neural/hugging_face/model_experiments/Exp3b_Images")
    parser.add_argument("--human_csv", type=Path, default="/Users/andreaprocopio/Desktop/object_reps_neural/detr/EXP_3_CHANGE/Data_processed/Data/exp3b_data.csv")
    parser.add_argument("--output_dir", type=Path, default="/Users/andreaprocopio/Desktop/object_reps_neural/hugging_face/model_experiments/exp3b_results_include_no_change")
    parser.add_argument("--model_interface", type=str, default="segformer",
                        choices=["segformer", "detr", "maskrcnn", "sam"], 
                        help="Model interface to use")
    parser.add_argument("--model_name", default=None,
                        help="Hugging-Face checkpoint for SegFormer/DETR or torchvision model name for MaskRCNN")
    parser.add_argument("--thresholds", default=None,
                        help="Comma-separated list of thresholds. If None, uses same thresholds as exp3Change.py")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Set default model names based on interface choice
    if args.model_name is None:
        if args.model_interface == "segformer":
            args.model_name = "nvidia/segformer-b1-finetuned-ade-512-512"
        elif args.model_interface == "detr":
            args.model_name = "facebook/detr-resnet-50-panoptic"
        elif args.model_interface == "maskrcnn":
            args.model_name = "maskrcnn_resnet50_fpn"
        elif args.model_interface == "sam":
            args.model_name = "facebook/sam-vit-base"  # Using sam-base for faster testing
        else:
            raise ValueError(f"Unknown model interface: {args.model_interface}")

    ## ensures that when we don't overwrite results when we run the script multiple times
    ## with different models, in the output we'll get a new folder named after each model
    model_tag = args.model_name.replace('/', '_')
    args.output_dir = args.output_dir / model_tag

    # Generate thresholds (same as exp3Change.py if not specified)
    if args.thresholds is None:
        # 0.1% to 5% in 0.1% steps, then 6% to 20% in 1% steps
        fine_thresholds = np.arange(0.001, 0.051, 0.001)
        coarse_thresholds = np.arange(0.06, 0.201, 0.01)
        thr_values = np.concatenate([fine_thresholds, coarse_thresholds])
        thr_values = [round(float(x), 3) for x in thr_values]
    else:
        thr_values = [round(float(x), 3) for x in args.thresholds.split(",")]

    # run segmentation once (or resume)
    torch.set_grad_enabled(False)
    
    # Initialize model interface based on choice
    if args.model_interface == "segformer":
        model_if = SegFormerInterface(model_name=args.model_name)
    elif args.model_interface == "detr":
        model_if = DetrInterface(model_name=args.model_name)
    elif args.model_interface == "maskrcnn":
        model_if = MaskRCNNInterface(model_name=args.model_name)
        # Set high detection sensitivity for simple blob detection
        model_if.set_detection_sensitivity("max")
    elif args.model_interface == "sam":
        model_if = SAMInterface(model_name=args.model_name)
    else:
        raise ValueError(f"Unknown model interface: {args.model_interface}")
    
    cde = ChangeDetectionExperiment(model_interface=model_if,
                                    output_dir=str(args.output_dir / "cde"))
    cde.run_full_experiment(images_dir=str(args.images_dir), resume=args.resume)

    # area-ratio for each pair
    area_ratio = collect_area_ratios(cde)                 # {base: ratio}
    base_types = load_base_types(cde)                     # {base: type}

    # human averages
    human_mean = load_human_data(args.human_csv)          # {shape_type: mean}

    # threshold sweep: correlation (global) and category-rate RMSE
    corr_table = []
    rmse_table = []
    for thr in thr_values:
        model_dec = []
        human_dec = []
        # Build per-base model binary decision for category rates
        model_by_base: Dict[str, float] = {}
        for base, ratio in area_ratio.items():
            # base name should match exactly with human data
            if base not in human_mean:
                continue
            model_dec.append(decide_different(ratio, thr))
            human_dec.append(human_mean[base])
            model_by_base[base] = decide_different(ratio, thr)
        r_p, r_s = correlations(model_dec, human_dec)
        corr_table.append(dict(threshold=thr,
                               pearson=r_p, spearman=r_s,
                               n=len(model_dec)))
        print(f"thr={thr:.3f} : r_P={r_p:.3f} r_S={r_s:.3f}  (n={len(model_dec)})")

        # Category-rate RMSE vs humans
        # Human category means (probabilities) from human_mean and base_types
        human_by_base = {b: p for b, p in human_mean.items() if b in base_types}
        human_cat_rates = compute_category_rates(human_by_base, base_types, is_binary=False)
        model_cat_rates = compute_category_rates(model_by_base, base_types, is_binary=True)
        rmse_val = rmse_by_categories(model_cat_rates, human_cat_rates)
        rmse_table.append(dict(threshold=thr, rmse=rmse_val,
                               model_cat_rates=model_cat_rates,
                               human_cat_rates=human_cat_rates))

    # save JSON
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "correlation_results.json").write_text(json.dumps(corr_table, indent=2))

    # TXT summary
    best_row = max(corr_table, key=lambda d: d["pearson"])
    best_rmse_row = min([r for r in rmse_table if not np.isnan(r["rmse"])], key=lambda d: d["rmse"]) if rmse_table else None
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    txt = (f"Exp 3B summary ({ts})\n"
           f"Model: {args.model_interface} - {args.model_name}\n"
           f"max Pearson = {best_row['pearson']:.3f} at thr {best_row['threshold']:.3f} "
           f"(Spearman {best_row['spearman']:.3f}, n={best_row['n']})\n")
    if best_rmse_row is not None:
        txt += (f"min category-RMSE = {best_rmse_row['rmse']:.4f} at thr {best_rmse_row['threshold']:.3f}\n")
    (out / "summary.txt").write_text(txt)
    print(txt)

    ## plot RMSE vs threshold (category rates)
    rmse_thr_arr = [d["threshold"] for d in rmse_table if not np.isnan(d["rmse"])]
    rmse_values = [d["rmse"] for d in rmse_table if not np.isnan(d["rmse"])]

    if rmse_thr_arr and rmse_values:
        plt.figure(figsize=(9, 6))
        plt.plot(rmse_thr_arr, rmse_values, "o-", label="Category RMSE")
        if best_rmse_row is not None:
            best_thr_rmse = best_rmse_row["threshold"]
            plt.axvline(best_thr_rmse, ls=":", color="k",
                        label=f"min RMSE at {best_thr_rmse:.3f}")
        plt.xlabel("Δ area threshold")
        plt.ylabel("Category RMSE (model vs human)")
        plt.title(f"Model-human category RMSE vs threshold ({args.model_interface})")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "rmse_vs_threshold.png", dpi=300)
        plt.close()

    # Plot model vs human category rates at best-RMSE threshold
    if best_rmse_row is not None:
        types_order = sorted(set(list(best_rmse_row["human_cat_rates"].keys()) + list(best_rmse_row["model_cat_rates"].keys())))
        human_vals = [best_rmse_row["human_cat_rates"].get(t, float("nan")) * 100 for t in types_order]
        model_vals = [best_rmse_row["model_cat_rates"].get(t, float("nan")) * 100 for t in types_order]

        x = np.arange(len(types_order))
        width = 0.38
        plt.figure(figsize=(7, 4.5))
        plt.bar(x - width/2, human_vals, width, label="Human", color="lightgray", edgecolor="black")
        plt.bar(x + width/2, model_vals, width, label="Model", color="#4C78A8", edgecolor="black")
        plt.xticks(x, types_order, rotation=20)
        plt.ylabel("Detection rate (%)")
        plt.ylim(0, 100)
        plt.title(f"Model vs Human category rates (thr={best_rmse_row['threshold']:.3f}, RMSE={best_rmse_row['rmse']:.3f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "category_rates_model_vs_human_best_rmse.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
