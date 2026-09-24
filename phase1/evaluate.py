"""
Phase 1.2 — Ground-truth evaluation helpers.

New in 1.2
----------
* Scale-align mode: computes a per-frame median scale factor and reports
  both raw and aligned metrics side-by-side.
* Distance-stratified metrics: 0-5 m / 5-20 m / 20+ m buckets.
* Coverage filter: boxes whose valid_px / box_area < min_valid_ratio are
  excluded from metrics (same filter as pipeline.py).
* Per-class breakdown (carried over from 1.1).

Usage
-----
Called from pipeline.py when --gt-depth is supplied, or run standalone:

    python phase1/evaluate.py \\
        --detections phase1/output/object_depths.csv \\
        --depth-npy  phase1/output/depth.npy \\
        --gt-depth   demo/input/kitti360_depth.png \\
        --depth-scale 256 \\
        --scale-align \\
        --output-dir phase1/output
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
try:
    import OpenEXR as _openexr
    import Imath as _imath
    _HAS_OPENEXR = True
except ImportError:
    _HAS_OPENEXR = False
try:
    import imageio.v3 as _iio3
    _HAS_IMAGEIO3 = True
except ImportError:
    _HAS_IMAGEIO3 = False


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_gt_depth(gt_path: Path, depth_scale: float = 256.0) -> np.ndarray:
    """
    Load GT depth as float32 metres. Supports:
    - uint16 PNG  (KITTI-360, ScanNet++): divided by depth_scale
    - float32 EXR (Matterport3D):         used as-is (depth_scale=1.0)
    
    Zero / invalid pixels are set to NaN.
    """
    gt_path = Path(gt_path)
    if gt_path.suffix.lower() == ".exr":
        depth = None
        # --- Try 1: OpenCV (works if compiled with OpenEXR) ---
        depth_cv = cv2.imread(str(gt_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth_cv is not None:
            depth = (depth_cv[:, :, 0] if depth_cv.ndim == 3 else depth_cv).astype(np.float32)
        # --- Try 2: OpenEXR Python binding (pip install OpenEXR) ---
        elif _HAS_OPENEXR:
            try:
                f = _openexr.InputFile(str(gt_path))
                hdr = f.header()
                dw  = hdr["dataWindow"]
                W   = dw.max.x - dw.min.x + 1
                H   = dw.max.y - dw.min.y + 1
                # pick first channel (R, Y, or Z)
                ch_name = next(iter(hdr["channels"].keys()))
                ch_type = hdr["channels"][ch_name].type
                pt = _imath.PixelType.FLOAT
                raw = f.channel(ch_name, _imath.PixelType(pt))
                depth = np.frombuffer(raw, dtype=np.float32).reshape(H, W).copy()
            except Exception as _e:
                raise IOError(f"OpenEXR failed on {gt_path}: {_e}") from _e
        # --- Try 3: imageio.v3 with openexr plugin ---
        elif _HAS_IMAGEIO3:
            try:
                arr = _iio3.imread(str(gt_path), plugin="openexr")
                depth = (arr[:, :, 0] if arr.ndim == 3 else arr).astype(np.float32)
            except Exception as _e:
                raise IOError(
                    f"Could not read EXR file: {gt_path}\n"
                    f"  imageio error: {_e}\n"
                    f"  Fix: pip install OpenEXR"
                ) from _e
        else:
            raise IOError(
                f"Cannot read EXR file: {gt_path}\n"
                f"  No EXR reader available. Fix: pip install OpenEXR"
            )
    else:
        img = np.array(Image.open(str(gt_path)))
        depth = img.astype(np.float32) / depth_scale
    depth[depth <= 0] = np.nan
    return depth


# ---------------------------------------------------------------------------
# Per-box GT depth aggregation (mirrors pipeline.py logic exactly)
# ---------------------------------------------------------------------------

def gt_depth_in_box(
    gt_depth: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mask: Optional[np.ndarray] = None,
    mode: str = "center",
    center_fraction: float = 0.50,
    low_pct: float = 10.0,
    high_pct: float = 90.0,
    min_pixels: int = 5,
) -> Tuple[float, int]:
    """Aggregate GT depth pixels inside bbox or instance mask with the same logic as pred."""
    if mask is not None and mode == "mask":
        valid_mask = (mask > 0) & np.isfinite(gt_depth) & (gt_depth > 0)
        values = gt_depth[valid_mask]
        if values.size < min_pixels:
            return float("nan"), 0
        if values.size >= 10:
            lo, hi = np.percentile(values, [low_pct, high_pct])
            values = values[(values >= lo) & (values <= hi)]
        return float(np.median(values)), int(values.size)

    x1, y1, x2, y2 = bbox
    h, w = gt_depth.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return float("nan"), 0

    if mode in ("center", "weighted"):
        bw, bh = x2 - x1, y2 - y1
        mx = int((1.0 - center_fraction) * bw / 2.0)
        my = int((1.0 - center_fraction) * bh / 2.0)
        x1, y1, x2, y2 = x1 + mx, y1 + my, x2 - mx, y2 - my

    roi = gt_depth[y1:y2, x1:x2]
    values = roi[np.isfinite(roi) & (roi > 0)]
    if values.size < min_pixels:
        return float("nan"), 0

    if mode == "fg_cluster" and values.size >= 15:
        lo_v, hi_v = np.percentile(values, [5, 95])
        if hi_v - lo_v > 0.35:
            split_thresh = float(np.median(values))
            near_cluster = values[values <= split_thresh]
            if near_cluster.size >= int(0.20 * values.size):
                values = near_cluster

    if values.size >= 10:
        lo, hi = np.percentile(values, [low_pct, high_pct])
        values = values[(values >= lo) & (values <= hi)]

    return float(np.median(values)), int(values.size)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, gts: np.ndarray) -> Dict:
    """MAE / RMSE / AbsRel / SqRel / delta<1.25 over valid (pred, gt) pairs."""
    mask = np.isfinite(preds) & np.isfinite(gts) & (gts > 0)
    n = int(mask.sum())
    if n == 0:
        nan = float("nan")
        return {k: nan for k in ["MAE", "RMSE", "AbsRel", "SqRel", "delta_1", "n"]}

    p, g = preds[mask], gts[mask]
    diff = p - g
    return {
        "MAE":     round(float(np.mean(np.abs(diff))), 4),
        "RMSE":    round(float(math.sqrt(np.mean(diff ** 2))), 4),
        "AbsRel":  round(float(np.mean(np.abs(diff) / g)), 4),
        "SqRel":   round(float(np.mean(diff ** 2 / g)), 4),
        "delta_1": round(float(np.mean(np.maximum(p / g, g / p) < 1.25)), 4),
        "n":       n,
    }


# ---------------------------------------------------------------------------
# Distance-stratified metrics
# ---------------------------------------------------------------------------

DISTANCE_BINS = [
    ("0-5 m",  0.0,  5.0),
    ("5-20 m", 5.0, 20.0),
    ("20+ m", 20.0, float("inf")),
]

# Object size bins by bounding-box pixel area
SIZE_BINS = [
    ("small",  0,      5_000),     # distant / small objects
    ("medium", 5_000,  50_000),    # typical near-field
    ("large",  50_000, float("inf")),  # close / large objects
]


def stratified_metrics(
    preds: np.ndarray, gts: np.ndarray
) -> Dict[str, Dict]:
    """
    Split (pred, gt) pairs by GT depth range and compute metrics per bin.
    Only pairs where both values are finite are considered.
    """
    result: Dict[str, Dict] = {}
    valid = np.isfinite(preds) & np.isfinite(gts) & (gts > 0)
    for label, lo, hi in DISTANCE_BINS:
        mask = valid & (gts >= lo) & (gts < hi)
        if mask.sum() == 0:
            result[label] = {k: float("nan") for k in ["MAE", "RMSE", "AbsRel", "delta_1", "n"]}
            result[label]["n"] = 0
        else:
            m = compute_metrics(preds[mask], gts[mask])
            result[label] = m
    return result


def size_stratified_metrics(
    preds: np.ndarray,
    gts: np.ndarray,
    box_areas: np.ndarray,
) -> Dict[str, Dict]:
    """
    Split (pred, gt) pairs by bounding-box pixel area and compute metrics.
    box_areas : 1-D array of (x2-x1)*(y2-y1) for each detection (same order as preds/gts).
    """
    result: Dict[str, Dict] = {}
    valid = np.isfinite(preds) & np.isfinite(gts) & (gts > 0)
    for label, lo, hi in SIZE_BINS:
        mask = valid & (box_areas >= lo) & (box_areas < hi)
        if mask.sum() == 0:
            result[label] = {k: float("nan") for k in ["MAE", "RMSE", "AbsRel", "delta_1", "n"]}
            result[label]["n"] = 0
        else:
            m = compute_metrics(preds[mask], gts[mask])
            result[label] = m
    return result


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

def per_class_metrics(numeric_rows: List[Dict]) -> Dict[str, Dict]:
    buckets: Dict[str, Tuple[List, List]] = defaultdict(lambda: ([], []))
    for r in numeric_rows:
        pd, gd = r["pred_depth_m"], r["gt_depth_m"]
        if np.isfinite(pd) and np.isfinite(gd):
            buckets[r["class"]][0].append(pd)
            buckets[r["class"]][1].append(gd)
    return {
        cls: compute_metrics(np.array(ps), np.array(gs))
        for cls, (ps, gs) in buckets.items()
    }


# ---------------------------------------------------------------------------
# Evaluation CSV schema
# ---------------------------------------------------------------------------

EVAL_FIELDS = [
    "image", "object_id", "class", "confidence",
    "x1", "y1", "x2", "y2", "roi_mode",
    "pred_depth_m", "pred_depth_aligned_m", "gt_depth_m",
    "abs_error_m", "abs_error_aligned_m",
    "rel_error",   "rel_error_aligned",
    "pred_valid_px", "gt_valid_px",
    "coverage_ratio", "filtered", "filter_reason",
]


def save_eval_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    detections: List[Dict],
    depth_pred: np.ndarray,
    gt_depth_path: Path,
    output_dir: Path,
    depth_scale: float = 256.0,
    roi_mode: str = "center",
    center_fraction: float = 0.50,
    low_pct: float = 10.0,
    high_pct: float = 90.0,
    conf_threshold: float = 0.40,
    min_valid_ratio: float = 0.20,
    scale_align: bool = False,
    image_name: str = "image",
) -> Dict:
    """
    Full evaluation entry point.  Writes eval_results.csv and metrics.json.

    Parameters
    ----------
    detections    : list of dicts from pipeline.py (includes 'filtered', 'coverage_ratio')
    depth_pred    : float32 HxW predicted depth (metres)
    gt_depth_path : uint16 GT depth PNG
    scale_align   : if True, multiply depth_pred by median(GT) / median(pred)
                    and report aligned metrics in addition to raw
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not gt_depth_path.exists():
        raise FileNotFoundError(f"GT depth not found: {gt_depth_path}")

    gt_depth = load_gt_depth(gt_depth_path, depth_scale=depth_scale)

    # Resize GT to pred shape if needed (KITTI-360 GT is 1400×1400)
    ph, pw = depth_pred.shape[:2]
    gh, gw = gt_depth.shape[:2]
    if (gh, gw) != (ph, pw):
        gt_depth_r = cv2.resize(gt_depth, (pw, ph), interpolation=cv2.INTER_NEAREST)
        gt_depth_r[gt_depth_r <= 0] = np.nan
    else:
        gt_depth_r = gt_depth

    # --- Scale alignment (median calibration) --------------------------
    scale_factor: Optional[float] = None
    depth_aligned = depth_pred.copy()
    if scale_align:
        pred_median = float(np.nanmedian(depth_pred))
        gt_median   = float(np.nanmedian(gt_depth_r))
        if pred_median > 0 and gt_median > 0:
            scale_factor = gt_median / pred_median
            depth_aligned = depth_pred * scale_factor

    # --- Build per-object rows -----------------------------------------
    rows: List[Dict] = []
    numeric_rows: List[Dict] = []
    pred_all, gt_all = [], []
    pred_aligned_all = []

    for det in detections:
        bbox    = det["bbox"]
        conf    = det["confidence"]
        pred_d  = det["estimated_depth_m"]
        pred_px = det.get("valid_depth_pixels", 0)
        cov     = det.get("coverage_ratio", None)
        is_filt = det.get("filtered", False)
        f_reason= det.get("filter_reason", "")

        # Aligned prediction for this object (use scale-aligned map)
        pred_d_aligned = (
            pred_d * scale_factor
            if (scale_align and scale_factor is not None and np.isfinite(pred_d))
            else float("nan")
        )

        det_mode = det.get("roi_mode", roi_mode)
        gt_d, gt_px = gt_depth_in_box(
            gt_depth_r, bbox,
            mask=det.get("mask"),
            mode=det_mode,
            center_fraction=center_fraction,
            low_pct=low_pct,
            high_pct=high_pct,
        )

        def _err(p, g):
            return abs(p - g) if np.isfinite(p) and np.isfinite(g) else float("nan")
        def _rel(p, g):
            ae = _err(p, g)
            return (ae / g) if np.isfinite(ae) and g > 0 else float("nan")

        abs_err     = _err(pred_d, gt_d)
        rel_err     = _rel(pred_d, gt_d)
        abs_err_al  = _err(pred_d_aligned, gt_d)
        rel_err_al  = _rel(pred_d_aligned, gt_d)

        x1, y1, x2, y2 = bbox

        def _fmt(v): return f"{v:.4f}" if np.isfinite(v) else ""

        rows.append({
            "image":                image_name,
            "object_id":            det["object_id"],
            "class":                det["class"],
            "confidence":           f"{conf:.6f}",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "roi_mode":             roi_mode,
            "pred_depth_m":         _fmt(pred_d),
            "pred_depth_aligned_m": _fmt(pred_d_aligned),
            "gt_depth_m":           _fmt(gt_d),
            "abs_error_m":          _fmt(abs_err),
            "abs_error_aligned_m":  _fmt(abs_err_al),
            "rel_error":            _fmt(rel_err),
            "rel_error_aligned":    _fmt(rel_err_al),
            "pred_valid_px":        pred_px,
            "gt_valid_px":          gt_px,
            "coverage_ratio":       f"{cov:.3f}" if cov is not None else "",
            "filtered":             "1" if is_filt else "0",
            "filter_reason":        f_reason,
        })

        numeric_rows.append({
            "class":        det["class"],
            "pred_depth_m": pred_d,
            "pred_depth_aligned_m": pred_d_aligned,
            "gt_depth_m":   gt_d,
            "box_area":     (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
            "gt_range":     (
                "0-5 m"  if np.isfinite(gt_d) and gt_d < 5   else
                "5-20 m" if np.isfinite(gt_d) and gt_d < 20  else
                "20+ m"  if np.isfinite(gt_d)                 else "unknown"
            ),
        })

        if not is_filt:
            pred_all.append(pred_d)
            gt_all.append(gt_d)
            pred_aligned_all.append(pred_d_aligned)

    save_eval_csv(rows, output_dir / "eval_results.csv")

    pa  = np.array(pred_all,         dtype=np.float32)
    paa = np.array(pred_aligned_all, dtype=np.float32)
    ga  = np.array(gt_all,           dtype=np.float32)

    # Collect bounding-box areas for the same unfiltered detections that
    # contributed to pred_all / gt_all (same order, same filter).
    box_areas_eval = np.array(
        [
            (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])
            for det in detections
            if not det.get("filtered", False)
        ],
        dtype=np.float32,
    )


    overall         = compute_metrics(pa,  ga)
    overall_aligned = compute_metrics(paa, ga) if scale_align else {}
    strat_raw       = stratified_metrics(pa,  ga)
    strat_aligned   = stratified_metrics(paa, ga) if scale_align else {}
    size_strat_raw  = size_stratified_metrics(pa, ga, box_areas_eval)
    per_cls         = per_class_metrics(numeric_rows)

    summary = {
        "overall":                overall,
        "overall_scale_aligned":  overall_aligned,
        "scale_factor":           scale_factor,
        "stratified":             strat_raw,
        "stratified_aligned":     strat_aligned,
        "size_stratified":        size_strat_raw,
        "per_class":              per_cls,
        "n_detections":           len(detections),
        "n_evaluated":            int(np.isfinite(pa).sum()),
        "conf_threshold":         conf_threshold,
        "min_valid_ratio":        min_valid_ratio,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_metrics(summary: Dict, title: str = "Phase-1.2 Evaluation Results") -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Detections : {summary['n_detections']} total  |  "
          f"{summary['n_evaluated']} passed filters")
    sf = summary.get("scale_factor")
    if sf is not None:
        print(f"  Scale factor (median-align): {sf:.4f}")

    def _row(label, m):
        mae  = m.get("MAE",     float("nan"))
        rmse = m.get("RMSE",    float("nan"))
        absr = m.get("AbsRel",  float("nan"))
        d1   = m.get("delta_1", float("nan"))
        n    = m.get("n", 0)
        return f"  {label:<22} {mae:>7.4f}m {rmse:>7.4f}m {absr:>8.4f} {d1:>8.4f} {n:>5}"

    header = f"  {'Mode':<22} {'MAE':>8} {'RMSE':>8} {'AbsRel':>9} {'δ<1.25':>9} {'N':>5}"
    print(f"\n{header}")
    print(f"  {'-'*62}")
    print(_row("Raw prediction", summary["overall"]))
    if summary.get("overall_scale_aligned"):
        print(_row("Scale-aligned", summary["overall_scale_aligned"]))

    # Size-stratified
    size_strat = summary.get("size_stratified", {})
    if size_strat:
        print(f"\n  Object-size stratified (bbox area, raw):")
        print(f"  {'-'*62}")
        for band, m in size_strat.items():
            print(_row(f"  {band}", m))

    # Distance-stratified
    strat = summary.get("stratified", {})
    strat_al = summary.get("stratified_aligned", {})
    if strat:
        print(f"\n  Distance-stratified (raw):")
        print(f"  {'-'*62}")
        for band, m in strat.items():
            print(_row(f"  {band}", m))
    if strat_al:
        print(f"\n  Distance-stratified (scale-aligned):")
        print(f"  {'-'*62}")
        for band, m in strat_al.items():
            print(_row(f"  {band}", m))

    # Per-class
    per_cls = summary.get("per_class", {})
    if per_cls:
        print(f"\n  Per-class (raw):")
        print(f"  {'-'*62}")
        for cls, m in per_cls.items():
            print(_row(f"  {cls}", m))

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-1.2 GT evaluation")
    p.add_argument("--detections",      required=True, help="object_depths.csv from pipeline.py")
    p.add_argument("--depth-npy",       required=True, help="depth.npy from pipeline.py")
    p.add_argument("--gt-depth",        required=True, help="GT depth PNG (uint16)")
    p.add_argument("--depth-scale",     type=float, default=256.0)
    p.add_argument("--output-dir",      default="phase1/output")
    p.add_argument("--roi-mode",        default="center", choices=["center", "bbox", "weighted"])
    p.add_argument("--center-fraction", type=float, default=0.50)
    p.add_argument("--low-pct",         type=float, default=10.0)
    p.add_argument("--high-pct",        type=float, default=90.0)
    p.add_argument("--conf",            type=float, default=0.40)
    p.add_argument("--min-valid-ratio", type=float, default=0.20)
    p.add_argument("--scale-align",     action="store_true",
                   help="Report scale-aligned metrics alongside raw metrics.")
    return p.parse_args()


if __name__ == "__main__":
    import sys
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    args = _parse_args()

    dets: List[Dict] = []
    with open(args.detections, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            filtered = row.get("filtered", "0") == "1"
            dets.append({
                "object_id":          i,
                "class":              row["class"],
                "confidence":         float(row["confidence"]),
                "bbox":               (int(row["x1"]), int(row["y1"]),
                                       int(row["x2"]), int(row["y2"])),
                "estimated_depth_m":  float(row["estimated_depth_m"]) if row["estimated_depth_m"] else float("nan"),
                "valid_depth_pixels": int(row.get("valid_depth_pixels") or 0),
                "coverage_ratio":     float(row["coverage_ratio"]) if row.get("coverage_ratio") else None,
                "filtered":           filtered,
                "filter_reason":      row.get("filter_reason", ""),
            })

    depth_pred = np.load(args.depth_npy).astype(np.float32)

    summary = evaluate(
        detections=dets,
        depth_pred=depth_pred,
        gt_depth_path=Path(args.gt_depth),
        output_dir=Path(args.output_dir),
        depth_scale=args.depth_scale,
        roi_mode=args.roi_mode,
        center_fraction=args.center_fraction,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
        conf_threshold=args.conf,
        min_valid_ratio=args.min_valid_ratio,
        scale_align=args.scale_align,
        image_name=Path(args.detections).stem,
    )

    print_metrics(summary)
    out = Path(args.output_dir)
    print(f"Eval CSV  -> {out / 'eval_results.csv'}")
    print(f"Metrics   -> {out / 'metrics.json'}")
