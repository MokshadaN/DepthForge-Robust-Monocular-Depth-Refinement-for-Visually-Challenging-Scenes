"""
Phase 2 — Experiment 1: Cross-camera robustness summary.

Reads metrics.json from each per-camera output directory and assembles
the Experiment 1 results table (Table 1 in the report).

Usage
-----
    python phase2/summarise_exp1.py
        --results-dirs
            phase1/output
            phase1/output_scannetpp
            phase1/output_matterport3d
        --camera-labels
            "Fisheye-outdoor (KITTI-360)"
            "Fisheye-indoor (ScanNet++)"
            "360-ERP (Matterport3D)"
        --output phase2/results/exp1_camera_metrics.csv

Columns emitted
---------------
camera, MAE, RMSE, AbsRel, SqRel, delta_1, N,
size_small_AbsRel, size_medium_AbsRel, size_large_AbsRel,
dist_0_5_AbsRel, dist_5_20_AbsRel, dist_20p_AbsRel
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_metrics(results_dir: Path) -> dict:
    p = results_dir / "metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"No metrics.json in {results_dir}")
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def flatten_metrics(label: str, m: dict) -> dict:
    ov = m.get("overall", {})
    strat = m.get("stratified", {})
    sz    = m.get("size_stratified", {})
    return {
        "camera":               label,
        "MAE":                  ov.get("MAE",     ""),
        "RMSE":                 ov.get("RMSE",    ""),
        "AbsRel":               ov.get("AbsRel",  ""),
        "SqRel":                ov.get("SqRel",   ""),
        "delta_1":              ov.get("delta_1", ""),
        "N":                    ov.get("n",       ""),
        "size_small_AbsRel":    sz.get("small",   {}).get("AbsRel", ""),
        "size_medium_AbsRel":   sz.get("medium",  {}).get("AbsRel", ""),
        "size_large_AbsRel":    sz.get("large",   {}).get("AbsRel", ""),
        "dist_0_5_AbsRel":      strat.get("0-5 m",  {}).get("AbsRel", ""),
        "dist_5_20_AbsRel":     strat.get("5-20 m", {}).get("AbsRel", ""),
        "dist_20p_AbsRel":      strat.get("20+ m",  {}).get("AbsRel", ""),
    }


def print_table(rows: list[dict]) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print("  Experiment 1 — Cross-Camera Robustness (Table 1)")
    print(sep)
    print(f"  {'Camera':<35} {'MAE':>7} {'RMSE':>7} {'AbsRel':>8} {'δ<1.25':>8} {'N':>5}")
    print(f"  {'-'*85}")
    for r in rows:
        mae  = f"{float(r['MAE']):.4f}"   if r['MAE']   != "" else "  N/A  "
        rmse = f"{float(r['RMSE']):.4f}"  if r['RMSE']  != "" else "  N/A  "
        absr = f"{float(r['AbsRel']):.4f}" if r['AbsRel'] != "" else "  N/A  "
        d1   = f"{float(r['delta_1']):.4f}" if r['delta_1'] != "" else "  N/A  "
        n    = str(r['N']) if r['N'] != "" else "N/A"
        print(f"  {r['camera']:<35} {mae:>7} {rmse:>7} {absr:>8} {d1:>8} {n:>5}")
    print(sep + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Exp 1 cross-camera summary")
    p.add_argument("--results-dirs", nargs="+",
                   default=[
                       "phase1/output",
                       "phase1/output_scannetpp",
                       "phase1/output_matterport3d",
                   ])
    p.add_argument("--camera-labels", nargs="+",
                   default=[
                       "Fisheye-outdoor (KITTI-360)",
                       "Fisheye-indoor (ScanNet++)",
                       "360-ERP (Matterport3D)",
                   ])
    p.add_argument("--output", default="phase2/results/exp1_camera_metrics.csv")
    args = p.parse_args()

    if len(args.results_dirs) != len(args.camera_labels):
        raise ValueError("--results-dirs and --camera-labels must have the same length")

    rows = []
    for rdir, label in zip(args.results_dirs, args.camera_labels):
        d = Path(rdir).resolve()
        try:
            m = load_metrics(d)
            row = flatten_metrics(label, m)
            rows.append(row)
            print(f"  ✓ {label:40s} AbsRel={row['AbsRel']}")
        except FileNotFoundError as e:
            print(f"  ✗ {label:40s} {e}")
            rows.append({"camera": label, **{k: "" for k in [
                "MAE","RMSE","AbsRel","SqRel","delta_1","N",
                "size_small_AbsRel","size_medium_AbsRel","size_large_AbsRel",
                "dist_0_5_AbsRel","dist_5_20_AbsRel","dist_20p_AbsRel",
            ]}})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print_table(rows)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
