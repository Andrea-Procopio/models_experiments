#!/usr/bin/env python3
"""
Step 3: Correlation, RMSE and plotting for Exp 3B (modular).

Inputs from step2:
- threshold_results/per_image_detailed.json (list of {base, type, area_before, area_after, area_change})

Also requires:
- human_csv: path to human judgements (exp3b_data.csv)

Outputs:
- correlation_results.json
- summary.txt (max Pearson, min RMSE thresholds)
- rmse_vs_threshold.png
- category_rates_model_vs_human_best_rmse.png
"""

from __future__ import annotations
import argparse
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

import sys
CUR_DIR = Path(__file__).resolve().parent
PARENT_DIR = CUR_DIR.parent
GRANDPARENT_DIR = PARENT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(GRANDPARENT_DIR))


def load_human_data(csv_path: Path) -> Dict[str, float]:
    df = (pd.read_csv(csv_path)
            .query("~shape.str.contains('catch_shape', na=False)", engine="python")
            .query("response in ['same','different']"))
    df["binary"] = (df["response"] == "different").astype(float)
    means = df.groupby("fullShapeName")["binary"].mean()
    return {shape_name: mean for shape_name, mean in means.items()}


def load_area_ratios_and_types(step2_dir: Path) -> Tuple[Dict[str, float], Dict[str, str]]:
    details_path = step2_dir / "threshold_results" / "per_image_detailed.json"
    if not details_path.exists():
        # fallback to the 1.0_comparison location used in step2
        alt = step2_dir / "threshold_results" / "1.0_comparison" / "per_image_detailed.json"
        details_path = alt
    data = json.loads(details_path.read_text())
    area_ratio = {d["base"]: d["area_change"] for d in data}
    base_types = {d["base"]: d.get("type", "unknown") for d in data}
    return area_ratio, base_types


def decide_different(area_ratio: float, thr: float) -> int:
    return int(area_ratio is not None and area_ratio > thr)


def correlations(model: List[int], human: List[float]) -> Tuple[float, float]:
    if len(model) < 2 or len(set(model)) == 1:
        return float("nan"), float("nan")
    return pearsonr(model, human)[0], spearmanr(model, human)[0]


def compute_category_rates(values_by_base: Dict[str, float], base_to_type: Dict[str, str]) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {}
    for base, val in values_by_base.items():
        t = base_to_type.get(base)
        if t is None:
            continue
        buckets.setdefault(t, []).append(float(val))
    return {t: (float(np.mean(v)) if len(v) > 0 else float("nan")) for t, v in buckets.items()}


def rmse_by_categories(model_rates: Dict[str, float], human_rates: Dict[str, float]) -> float:
    keys = sorted(set(model_rates.keys()) & set(human_rates.keys()))
    if not keys:
        return float("nan")
    diffs = [(model_rates[k] - human_rates[k]) ** 2 for k in keys
             if not np.isnan(model_rates[k]) and not np.isnan(human_rates[k])]
    if not diffs:
        return float("nan")
    return float(np.sqrt(np.mean(diffs)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3B modular step3: analyze and plot")
    # Expect step2_dir to be results/<arch>/<model_tag>/step2_out
    parser.add_argument("--step2_dir", type=Path, default=Path("/Users/andreaprocopio/Desktop/object_reps_neural/hugging_face/model_experiments/exp3_general/exp3b_modular/results/segformer/nvidia_segformer-b0-finetuned-ade-512-512/step2_out"))
    parser.add_argument("--human_csv", type=Path, default=Path("/Users/andreaprocopio/Desktop/object_reps_neural/detr/EXP_3_CHANGE/Data_processed/Data/exp3b_data.csv"))
    # Output under results/<arch>/<model_tag>/step3_out
    parser.add_argument("--output_dir", type=Path, default=Path("/Users/andreaprocopio/Desktop/object_reps_neural/hugging_face/model_experiments/exp3_general/exp3b_modular/results/segformer/nvidia_segformer-b0-finetuned-ade-512-512/step3_out"))
    parser.add_argument("--model_interface", type=str, default="segformer")
    parser.add_argument("--model_name", type=str, default="nvidia/segformer-b1-finetuned-ade-512-512")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated thresholds; default matches exp3b_correlation")
    args = parser.parse_args()

    # A single directory per model to match requested structure
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # thresholds grid
    if args.thresholds is None:
        fine_thresholds = np.arange(0.001, 0.051, 0.001)
        coarse_thresholds = np.arange(0.06, 0.201, 0.01)
        thr_values = np.concatenate([fine_thresholds, coarse_thresholds])
        thr_values = [round(float(x), 3) for x in thr_values]
    else:
        thr_values = [round(float(x), 3) for x in args.thresholds.split(',')]

    # load inputs
    area_ratio, base_types = load_area_ratios_and_types(args.step2_dir)
    human_mean = load_human_data(args.human_csv)

    corr_table: List[Dict] = []
    rmse_table: List[Dict] = []

    for thr in thr_values:
        model_dec: List[int] = []
        human_dec: List[float] = []
        model_by_base: Dict[str, float] = {}

        for base, ratio in area_ratio.items():
            if base not in human_mean:
                continue
            dec = decide_different(ratio, thr)
            model_dec.append(dec)
            human_dec.append(human_mean[base])
            model_by_base[base] = dec

        r_p, r_s = correlations(model_dec, human_dec)
        corr_table.append(dict(threshold=thr, pearson=r_p, spearman=r_s, n=len(model_dec)))

        # Category RMSE
        human_by_base = {b: p for b, p in human_mean.items() if b in base_types}
        human_cat_rates = compute_category_rates(human_by_base, base_types)
        model_cat_rates = compute_category_rates(model_by_base, base_types)
        rmse_val = rmse_by_categories(model_cat_rates, human_cat_rates)
        rmse_table.append(dict(threshold=thr, rmse=rmse_val,
                               model_cat_rates=model_cat_rates,
                               human_cat_rates=human_cat_rates))

    # save JSON and summary
    (out / "correlation_results.json").write_text(json.dumps(corr_table, indent=2))
    best_row = max(corr_table, key=lambda d: (d["pearson"] if not np.isnan(d["pearson"]) else -np.inf))
    rmse_rows = [r for r in rmse_table if not np.isnan(r["rmse"])]
    best_rmse_row = (min(rmse_rows, key=lambda d: d["rmse"]) if rmse_rows else None)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    txt = (f"Exp 3B modular summary ({ts})\n"
           f"Model: {args.model_interface} - {args.model_name}\n"
           f"max Pearson = {best_row['pearson']:.3f} at thr {best_row['threshold']:.3f} "
           f"(Spearman {best_row['spearman']:.3f}, n={best_row['n']})\n")
    if best_rmse_row is not None:
        txt += (f"min category-RMSE = {best_rmse_row['rmse']:.4f} at thr {best_rmse_row['threshold']:.3f}\n")
    (out / "summary.txt").write_text(txt)
    print(txt)

    # RMSE vs threshold plot
    rmse_thr_arr = [d["threshold"] for d in rmse_table if not np.isnan(d["rmse"])]
    rmse_values = [d["rmse"] for d in rmse_table if not np.isnan(d["rmse"])]
    if rmse_thr_arr and rmse_values:
        plt.figure(figsize=(9, 6))
        plt.plot(rmse_thr_arr, rmse_values, "o-", label="Category RMSE")
        if best_rmse_row is not None:
            best_thr_rmse = best_rmse_row["threshold"]
            plt.axvline(best_thr_rmse, ls=":", color="k", label=f"min RMSE at {best_thr_rmse:.3f}")
        plt.xlabel("Δ area threshold")
        plt.ylabel("Category RMSE (model vs human)")
        plt.title(f"Model-human category RMSE vs threshold ({args.model_interface})")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "rmse_vs_threshold.png", dpi=300)
        plt.close()

    # Category bar chart at best RMSE
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

    # Additionally, generate detection bar charts for every threshold like exp3Change
    # We reconstruct detections per threshold using the binary model decisions
    for row in corr_table:
        thr = row["threshold"]
        pct_label = f"{thr * 100:.1f}".rstrip('0').rstrip('.')
        dir_out = out / "threshold_results" / f"{pct_label}_comparison"
        dir_out.mkdir(parents=True, exist_ok=True)

        # Build detections dict using area_ratio and base_types
        detections = {t: [] for t in sorted(set(base_types.values()))}
        for base, ratio in area_ratio.items():
            t = base_types.get(base, "unknown")
            detected = 1 if (ratio is not None and ratio > thr) else 0
            detections.setdefault(t, []).append(detected)

        # Compute mean rates and a simplified SEM like in exp3Change
        types = sorted(detections.keys())
        rates = np.array([(np.mean(detections[t]) * 100 if detections[t] else 0.0) for t in types])
        def sem_bin(vals: List[int]) -> float:
            n = len(vals)
            if n == 0:
                return 0.0
            p = np.mean(vals)
            return float(np.sqrt(p * (1 - p) / n) * 100 * 0.5)
        sems = np.array([sem_bin(detections[t]) for t in types])

        # Overall plot
        x = np.arange(len(types))
        plt.figure(figsize=(4.8, 4))
        for i, t in enumerate(types):
            plt.bar(x[i], rates[i], 1.0, color='lightgray', edgecolor='black', hatch='//', yerr=sems[i], capsize=5)
        plt.xticks([])
        plt.ylabel('% Detection Rate')
        plt.ylim(0, 100)
        plt.title(f'Threshold = {pct_label}%')
        plt.tight_layout()
        plt.savefig(dir_out / 'overall_comparison.png', dpi=200)
        plt.close()

        # Three-condition plot (concave, concave_nofill, convex) if present
        three = ['concave', 'concave_nofill', 'convex']
        colors = [
            (255/255, 188/255, 78/255),
            (209/255, 168/255, 95/255),
            (79/255, 168/255, 78/255)
        ]
        x2 = np.arange(len(three)) * 2.1
        plt.figure(figsize=(4.5, 6))
        for i, t in enumerate(three):
            if t in types:
                idx = types.index(t)
                plt.bar(x2[i], rates[idx], 2.1, color=colors[i], edgecolor='black', hatch='//', yerr=sems[idx], capsize=5)
        plt.xticks([])
        plt.ylabel('% Noticing Change')
        plt.ylim(0, 100)
        plt.title(f'Threshold = {pct_label}%')
        plt.tight_layout()
        plt.savefig(dir_out / 'three_comparison.png', dpi=200)
        plt.close()


if __name__ == "__main__":
    main()


