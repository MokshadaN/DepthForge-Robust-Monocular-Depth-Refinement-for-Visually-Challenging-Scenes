"""
Phase 1.2 — Multi-image batch evaluator.

Runs the YOLO + UniDAC pipeline across many images and aggregates
object-level metrics (MAE / RMSE / AbsRel / delta<1.25) over all detections.

Usage
-----
    python phase1/batch_eval.py \\
        --image-dir  datasets/kitti360/images/ \\
        --gt-dir     datasets/kitti360/depth/ \\
        --yolo-model yolo11n.pt \\
        --unidac-checkpoint checkpoints/unidac.pt \\
        --output-dir phase1/batch_output \\
        --scale-align

Split-file support (one stem per line, no extension):
    python phase1/batch_eval.py \\
        --split splits/kitti360/val.txt \\
        --image-dir  datasets/kitti360/images/ \\
        --gt-dir     datasets/kitti360/depth/ \\
        --output-dir phase1/batch_output

Notes
-----
* GT depth files must have the same stem as the RGB files, e.g.
      images/0000.png  <->  depth/0000.png
* Depth PNGs must be uint16 with the standard KITTI-360 scale of 256.
* Detections below --conf-min or with coverage < --min-valid-ratio
  are excluded from the aggregated metrics (same rules as single-image mode).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase1.evaluate import (
    compute_metrics,
    evaluate,
    gt_depth_in_box,
    load_gt_depth,
    per_class_metrics,
    print_metrics,
    stratified_metrics,
)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(args: argparse.Namespace) -> None:
    image_dir  = Path(args.image_dir).resolve()
    gt_dir     = Path(args.gt_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Collect image list -------------------------------------------
    if args.split:
        split_path = Path(args.split).resolve()
        stems = [l.strip() for l in split_path.read_text().splitlines() if l.strip()]
        image_paths = []
        for stem in stems:
            for ext in (".png", ".jpg", ".jpeg"):
                p = image_dir / (stem + ext)
                if p.exists():
                    image_paths.append(p)
                    break
    else:
        image_paths = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg")
            for p in image_dir.glob(ext)
        )

    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    print(f"Batch eval: {len(image_paths)} images")
    print(f"Output   : {output_dir}\n")

    # ---- Lazy-load models --------------------------------------------
    print("Loading YOLO ...")
    from ultralytics import YOLO
    detector = YOLO(args.yolo_model)

    print("Loading UniDAC ...")
    # Import lazily to avoid slow startup when model isn't needed.
    from phase1.pipeline import UniDACDepthRunner, extract_depth_from_bbox
    depth_runner = UniDACDepthRunner(
        checkpoint=Path(args.unidac_checkpoint).resolve(),
        device=args.device,
    )

    # ---- Per-image loop -----------------------------------------------
    all_rows:   List[Dict] = []   # raw eval rows across all images
    skip_count = 0
    t0 = time.time()

    batch_csv_path = output_dir / "batch_eval_results.csv"

    for idx, img_path in enumerate(image_paths):
        stem   = img_path.stem
        gt_png = gt_dir / (stem + ".png")
        if not gt_png.exists():
            print(f"  [{idx+1}/{len(image_paths)}] {stem} — GT not found, skipping")
            skip_count += 1
            continue

        print(f"  [{idx+1}/{len(image_paths)}] {stem} ...", end=" ", flush=True)
        t1 = time.time()

        try:
            image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            orig_h, orig_w = image_bgr.shape[:2]

            # ---- Depth prediction ------------------------------------
            depth = depth_runner.predict(img_path, output_size=(orig_h, orig_w))

            # ---- YOLO detection -------------------------------------
            yolo_results = detector.predict(
                source=image_bgr,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )

            detections: List[Dict] = []
            names = detector.names
            obj_id = 0
            for result in yolo_results:
                if result.boxes is None:
                    continue
                boxes   = result.boxes.xyxy.detach().cpu().numpy()
                confs   = result.boxes.conf.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                for box, conf, cls_id in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = np.round(box).astype(int).tolist()
                    dv, vpx, roi_box = extract_depth_from_bbox(
                        depth, (x1, y1, x2, y2),
                        mode=args.roi_mode,
                        center_fraction=args.center_fraction,
                        low_pct=args.low_pct,
                        high_pct=args.high_pct,
                    )
                    rx1, ry1, rx2, ry2 = roi_box
                    roi_area = max(1, (rx2 - rx1) * (ry2 - ry1))
                    cov = vpx / roi_area
                    filtered = (float(conf) < args.conf_min) or (cov < args.min_valid_ratio)
                    detections.append({
                        "object_id":          obj_id,
                        "class":              names[int(cls_id)],
                        "confidence":         float(conf),
                        "bbox":               (x1, y1, x2, y2),
                        "estimated_depth_m":  dv,
                        "valid_depth_pixels": vpx,
                        "coverage_ratio":     round(cov, 3),
                        "filtered":           filtered,
                        "filter_reason":      "",
                    })
                    obj_id += 1

            # ---- Scale alignment ------------------------------------
            scale_factor: Optional[float] = None
            if args.scale_align:
                gt_tmp = load_gt_depth(gt_png, depth_scale=args.depth_scale)
                ph, pw = depth.shape[:2]
                if gt_tmp.shape != (ph, pw):
                    gt_tmp = cv2.resize(gt_tmp, (pw, ph), interpolation=cv2.INTER_NEAREST)
                    gt_tmp[gt_tmp <= 0] = np.nan
                pm = float(np.nanmedian(depth))
                gm = float(np.nanmedian(gt_tmp))
                if pm > 0 and gm > 0:
                    scale_factor = gm / pm

            # ---- Per-image evaluation --------------------------------
            img_summary = evaluate(
                detections=detections,
                depth_pred=depth,
                gt_depth_path=gt_png,
                output_dir=output_dir / stem,
                depth_scale=args.depth_scale,
                roi_mode=args.roi_mode,
                center_fraction=args.center_fraction,
                low_pct=args.low_pct,
                high_pct=args.high_pct,
                conf_threshold=args.conf_min,
                min_valid_ratio=args.min_valid_ratio,
                scale_align=args.scale_align,
                image_name=img_path.name,
            )

            # Collect raw rows from the per-image CSV
            row_csv = output_dir / stem / "eval_results.csv"
            if row_csv.exists():
                with row_csv.open(newline="", encoding="utf-8") as f:
                    all_rows.extend(list(csv.DictReader(f)))

            elapsed = time.time() - t1
            ov = img_summary["overall"]
            print(f"MAE={ov.get('MAE', float('nan')):.3f}m  "
                  f"AbsRel={ov.get('AbsRel', float('nan')):.3f}  "
                  f"n={ov.get('n', 0)}  ({elapsed:.1f}s)")

        except Exception as exc:
            print(f"ERROR: {exc}")
            skip_count += 1

    # ---- Aggregate across all images ---------------------------------
    print(f"\nProcessed {len(image_paths) - skip_count} / {len(image_paths)} images "
          f"({skip_count} skipped)  |  total time: {time.time()-t0:.1f}s")

    if not all_rows:
        print("No valid rows collected — cannot compute aggregate metrics.")
        return

    def _safe_float(v):
        try:
            return float(v) if v else float("nan")
        except (ValueError, TypeError):
            return float("nan")

    pred_all  = np.array([_safe_float(r.get("pred_depth_m", "")) for r in all_rows], dtype=np.float32)
    gt_all    = np.array([_safe_float(r.get("gt_depth_m",   "")) for r in all_rows], dtype=np.float32)
    pred_al   = np.array([_safe_float(r.get("pred_depth_aligned_m", "")) for r in all_rows], dtype=np.float32)

    # Only keep unfiltered rows for aggregate metrics
    filt_mask = np.array([r.get("filtered", "0") == "1" for r in all_rows])
    keep = ~filt_mask
    pred_all  = pred_all[keep]
    gt_all    = gt_all[keep]
    pred_al   = pred_al[keep]

    aggregate = {
        "overall":                compute_metrics(pred_all, gt_all),
        "overall_scale_aligned":  compute_metrics(pred_al,  gt_all) if args.scale_align else {},
        "stratified":             stratified_metrics(pred_all, gt_all),
        "stratified_aligned":     stratified_metrics(pred_al,  gt_all) if args.scale_align else {},
        "scale_factor":           None,
        "n_detections":           len(all_rows),
        "n_evaluated":            int(keep.sum()),
        "conf_threshold":         args.conf_min,
        "min_valid_ratio":        args.min_valid_ratio,
        "n_images":               len(image_paths) - skip_count,
        "n_skipped":              skip_count,
    }

    with (output_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    # Write combined CSV
    if all_rows:
        fields = list(all_rows[0].keys())
        with batch_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    print_metrics(aggregate, title="Batch Aggregate Results")
    print(f"Combined CSV     -> {batch_csv_path}")
    print(f"Aggregate JSON   -> {output_dir / 'aggregate_metrics.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-1.2 batch evaluation")

    # Input
    p.add_argument("--image-dir",  required=True, help="Directory of RGB images")
    p.add_argument("--gt-dir",     required=True, help="Directory of GT depth PNGs (uint16)")
    p.add_argument("--split",      default=None,
                   help="Optional text file listing image stems (one per line).")

    # Models
    p.add_argument("--yolo-model",        default="yolo11n.pt")
    p.add_argument("--unidac-checkpoint", default=str(REPO_ROOT / "checkpoints/unidac.pt"))
    p.add_argument("--device",            default="auto")

    # Output
    p.add_argument("--output-dir", default="phase1/batch_output")

    # Detection
    p.add_argument("--conf",     type=float, default=0.25, help="YOLO detection threshold")
    p.add_argument("--conf-min", type=float, default=0.40, help="Hard metric filter threshold")
    p.add_argument("--iou",      type=float, default=0.70)
    p.add_argument("--imgsz",    type=int,   default=640)

    # Aggregation
    p.add_argument("--roi-mode",        default="center", choices=["center", "bbox", "weighted"])
    p.add_argument("--center-fraction", type=float, default=0.50)
    p.add_argument("--low-pct",         type=float, default=10.0)
    p.add_argument("--high-pct",        type=float, default=90.0)
    p.add_argument("--min-valid-ratio", type=float, default=0.20)

    # Evaluation
    p.add_argument("--depth-scale",  type=float, default=256.0)
    p.add_argument("--scale-align",  action="store_true",
                   help="Apply median scale calibration per image before metrics.")

    return p.parse_args()


if __name__ == "__main__":
    run_batch(_parse_args())
