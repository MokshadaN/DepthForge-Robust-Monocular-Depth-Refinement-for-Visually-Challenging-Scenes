"""
Phase 3 Benchmark & Progression Summarizer.

Collects metrics across Phase 1 (Baseline Box), Phase 2 (Center Trimming), and
Phase 3 (Segmentation + ERP Perspective Unwarping) and generates a cross-phase
progression summary table and CSV.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List


RUNS = [
    # KITTI-360
    {"camera": "Fisheye Outdoor (KITTI-360)", "phase": "Phase 1 (BBox)", "json": "phase1/output/metrics.json"},
    {"camera": "Fisheye Outdoor (KITTI-360)", "phase": "Phase 2 (Center 0.35)", "json": "phase2/results/kitti360_comparison/comparison_metrics.json"},
    {"camera": "Fisheye Outdoor (KITTI-360)", "phase": "Phase 3 (Seg Mask)", "json": "phase3/output_kitti360/metrics.json"},

    # ScanNet++
    {"camera": "Fisheye Indoor (ScanNet++)", "phase": "Phase 1 (BBox)", "json": "phase1/output_scannetpp/metrics.json"},
    {"camera": "Fisheye Indoor (ScanNet++)", "phase": "Phase 2 (Center 0.35)", "json": "phase1/output_scannetpp/metrics.json"},
    {"camera": "Fisheye Indoor (ScanNet++)", "phase": "Phase 3 (Seg Mask)", "json": "phase3/output_scannetpp/metrics.json"},

    # Matterport3D
    {"camera": "360-ERP (Matterport3D)", "phase": "Phase 1 (BBox)", "json": "phase1/output_matterport3d/metrics.json"},
    {"camera": "360-ERP (Matterport3D)", "phase": "Phase 2 (Center 0.35)", "json": "phase1/output_matterport3d/metrics.json"},
    {"camera": "360-ERP (Matterport3D)", "phase": "Phase 3 (Cubemap+Seg)", "json": "phase3/output_matterport3d/metrics.json"},
]


def load_run_metric(path_str: str) -> Dict:
    p = Path(path_str)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "overall" in data:
        return data["overall"]
    if "UniDAC" in data:
        return data["UniDAC"]
    return data


def main():
    rows = []
    print("\n" + "=" * 92)
    print("  Progression Benchmark: Phase 1 -> Phase 2 -> Phase 3 Object-Level Distance Estimation")
    print("=" * 92)
    hdr = f"  {'Camera Projection':<32} {'Pipeline Phase':<24} {'MAE (m)':>9} {'RMSE (m)':>10} {'AbsRel':>9} {'d<1.25':>8} {'N':>4}"
    print(hdr)
    print("  " + "-" * 88)

    for run in RUNS:
        m = load_run_metric(run["json"])
        mae = m.get("MAE", float("nan"))
        rmse = m.get("RMSE", float("nan"))
        absr = m.get("AbsRel", float("nan"))
        d1 = m.get("delta_1", float("nan"))
        n = m.get("n", 0)

        row_str = f"  {run['camera']:<32} {run['phase']:<24} {mae:>8.4f}m {rmse:>9.4f}m {absr:>9.4f} {d1:>8.4f} {n:>4}"
        print(row_str)

        rows.append({
            "camera": run["camera"],
            "phase": run["phase"],
            "mae_m": f"{mae:.4f}" if mae == mae else "",
            "rmse_m": f"{rmse:.4f}" if rmse == rmse else "",
            "abs_rel": f"{absr:.4f}" if absr == absr else "",
            "delta_1": f"{d1:.4f}" if d1 == d1 else "",
            "n_objects": n,
        })

    print("=" * 92 + "\n")

    out_csv = Path("phase3/results/phase_progression_metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["camera", "phase", "mae_m", "rmse_m", "abs_rel", "delta_1", "n_objects"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary table saved to -> {out_csv}\n")


if __name__ == "__main__":
    main()
