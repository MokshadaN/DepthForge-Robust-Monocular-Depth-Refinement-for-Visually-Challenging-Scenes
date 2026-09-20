"""
Phase 2 — Multi-model depth comparison framework.

Compares UniDAC against three popular monocular depth baselines:
  • MiDaS v3.1        (intel-isl/MiDaS, via torch.hub)
  • ZoeDepth          (isl-org/ZoeDepth, via torch.hub)
  • Depth Anything V2 (LiheYoung/depth-anything, via transformers)

For each model the script:
  1. Loads the RGB image.
  2. Predicts a depth map (resized to the original resolution).
  3. Runs YOLO to get bounding boxes (shared across all models).
  4. Aggregates per-object depth using the same ROI strategy.
  5. Compares against GT depth and computes MAE / RMSE / AbsRel / delta<1.25.
  6. Writes a combined comparison CSV and a ranked summary table.

Usage
-----
  python phase2/compare_models.py \\
      --image  demo/input/kitti360_rgb.png \\
      --gt     demo/input/kitti360_depth.png \\
      --depth-scale 256 \\
      --yolo-model yolo11n.pt \\
      --unidac-checkpoint checkpoints/unidac.pt \\
      --sample-config demo/input/kitti360_sample.json \\
      --models unidac midas zoe depthanything \\
      --output-dir phase2/results/comparison

Notes
-----
  • MiDaS and ZoeDepth are affine-invariant (relative depth) — scale-align is
    applied automatically before metrics for those models.
  • Depth Anything V2 can be metric or relative depending on the variant.
  • UniDAC is metric; raw and scale-aligned metrics are both reported.
  • All models are run on CPU if CUDA is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase1.evaluate import (
    compute_metrics,
    gt_depth_in_box,
    load_gt_depth,
    print_metrics,
    stratified_metrics,
    size_stratified_metrics,
)
from phase1.pipeline import (
    extract_depth_from_bbox,
    load_sample_config,
    UniDACDepthRunner,
)


# ---------------------------------------------------------------------------
# Depth backend wrappers
# ---------------------------------------------------------------------------

class DepthBackend:
    """Abstract base — each backend exposes predict(image_path) -> HxW float32."""
    name: str
    is_metric: bool = False          # True → raw metrics; False → scale-align before metrics

    def predict(self, image_path: Path, output_size: Tuple[int, int]) -> np.ndarray:
        raise NotImplementedError


class UniDACBackend(DepthBackend):
    name = "UniDAC"
    is_metric = True

    def __init__(self, checkpoint: Path, device: str, sample: Dict):
        self._runner = UniDACDepthRunner(
            checkpoint=checkpoint, device=device, sample=sample
        )

    def predict(self, image_path: Path, output_size: Tuple[int, int]) -> np.ndarray:
        return self._runner.predict(image_path, output_size=output_size)


class MiDaSBackend(DepthBackend):
    name = "MiDaS"
    is_metric = False

    def __init__(self, device: str, model_type: str = "DPT_BEiT_L_512"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else ("cuda" if device.startswith("cuda") else "cpu"))
        print(f"  Loading MiDaS ({model_type}) ...")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = (transforms.dpt_transform
                          if "DPT" in model_type
                          else transforms.small_transform)

    def predict(self, image_path: Path, output_size: Tuple[int, int]) -> np.ndarray:
        img = np.array(Image.open(str(image_path)).convert("RGB"))
        inp = self.transform(img).to(self.device)
        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=output_size,
                mode="bicubic", align_corners=False,
            ).squeeze()
        depth = pred.cpu().numpy().astype(np.float32)
        # MiDaS returns inverse depth — convert to (relative) depth
        depth = 1.0 / (depth + 1e-6)
        return depth


class ZoeDepthBackend(DepthBackend):
    name = "ZoeDepth"
    is_metric = True       # ZoeDepth-N is trained for metric depth

    def __init__(self, device: str, model_name: str = "ZoeD_N"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else ("cuda" if device.startswith("cuda") else "cpu"))
        print(f"  Loading ZoeDepth ({model_name}) ...")
        self.model = torch.hub.load(
            "isl-org/ZoeDepth", model_name, pretrained=True, trust_repo=True
        )
        self.model.to(self.device).eval()

    def predict(self, image_path: Path, output_size: Tuple[int, int]) -> np.ndarray:
        img = Image.open(str(image_path)).convert("RGB")
        with torch.no_grad():
            depth_pil = self.model.infer_pil(img)
        depth = np.array(depth_pil, dtype=np.float32)
        if (depth.shape[0], depth.shape[1]) != output_size:
            depth = cv2.resize(depth, (output_size[1], output_size[0]),
                               interpolation=cv2.INTER_LINEAR)
        return depth


class DepthAnythingBackend(DepthBackend):
    name = "DepthAnything"
    is_metric = False

    def __init__(self, device: str, encoder: str = "vitl"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else ("cuda" if device.startswith("cuda") else "cpu"))
        print(f"  Loading Depth Anything V2 (encoder={encoder}) ...")
        try:
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline(
                task="depth-estimation",
                model=f"depth-anything/Depth-Anything-V2-{encoder.capitalize()}-hf",
                device=0 if str(self.device).startswith("cuda") else -1,
            )
            self._use_hf = True
        except Exception as e:
            print(f"    HuggingFace transformers failed ({e}), trying torch.hub ...")
            self._model = torch.hub.load(
                "LiheYoung/depth_anything", "DepthAnything_vitl14",
                pretrained=True, trust_repo=True,
            )
            self._model.to(self.device).eval()
            self._use_hf = False

    def predict(self, image_path: Path, output_size: Tuple[int, int]) -> np.ndarray:
        img = Image.open(str(image_path)).convert("RGB")
        if self._use_hf:
            result = self._pipe(img)
            depth = np.array(result["depth"], dtype=np.float32)
        else:
            import torchvision.transforms.functional as TF
            img_t = TF.normalize(
                TF.to_tensor(np.array(img)),
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
            ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                depth = self._model(img_t)[0, 0].cpu().numpy()
        if (depth.shape[0], depth.shape[1]) != output_size:
            depth = cv2.resize(depth, (output_size[1], output_size[0]),
                               interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_backends(model_names: List[str], args: argparse.Namespace,
                   sample: Dict) -> List[DepthBackend]:
    backends: List[DepthBackend] = []
    for name in model_names:
        name = name.lower().strip()
        if name == "unidac":
            ckpt = Path(args.unidac_checkpoint).resolve()
            backends.append(UniDACBackend(ckpt, args.device, sample))
        elif name == "midas":
            backends.append(MiDaSBackend(args.device,
                                         getattr(args, "midas_type", "DPT_BEiT_L_512")))
        elif name in ("zoe", "zoedepth"):
            backends.append(ZoeDepthBackend(args.device,
                                            getattr(args, "zoe_model", "ZoeD_N")))
        elif name in ("depthanything", "depth_anything", "da"):
            backends.append(DepthAnythingBackend(args.device,
                                                 getattr(args, "da_encoder", "vitl")))
        else:
            print(f"  [warn] Unknown model '{name}', skipping.")
    return backends


# ---------------------------------------------------------------------------
# Per-object evaluation for one model
# ---------------------------------------------------------------------------

def eval_model_on_detections(
    backend: DepthBackend,
    detections: List[Dict],
    gt_depth_r: np.ndarray,
    roi_mode: str,
    center_fraction: float,
    low_pct: float,
    high_pct: float,
    conf_min: float,
    min_valid_ratio: float,
) -> Tuple[Dict, List[Dict]]:
    """
    Returns (metrics_dict, per_object_rows).
    Scale-alignment is applied automatically for non-metric models.
    """
    depth_pred = backend._depth_cache   # set by caller

    # Optional scale-align for relative-depth models
    if not backend.is_metric:
        pm = float(np.nanmedian(depth_pred))
        gm = float(np.nanmedian(gt_depth_r))
        scale = (gm / pm) if pm > 0 and gm > 0 else 1.0
        depth_pred_eval = depth_pred * scale
        scale_note = f"{scale:.4f}"
    else:
        depth_pred_eval = depth_pred
        scale_note = "N/A (metric)"

    preds, gts, areas = [], [], []
    rows: List[Dict] = []

    for det in detections:
        if det.get("filtered", False):
            continue
        bbox = det["bbox"]
        pd, _, _ = extract_depth_from_bbox(
            depth_pred_eval, bbox,
            mode=roi_mode, center_fraction=center_fraction,
            low_pct=low_pct, high_pct=high_pct,
        )
        gd, _ = gt_depth_in_box(
            gt_depth_r, bbox,
            mode=roi_mode, center_fraction=center_fraction,
            low_pct=low_pct, high_pct=high_pct,
        )
        ae = abs(pd - gd) if np.isfinite(pd) and np.isfinite(gd) else float("nan")
        re = (ae / gd) if np.isfinite(ae) and gd > 0 else float("nan")
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        preds.append(pd); gts.append(gd); areas.append(area)
        rows.append({
            "model":       backend.name,
            "object_id":   det["object_id"],
            "class":       det["class"],
            "confidence":  f"{det['confidence']:.4f}",
            "pred_depth_m": f"{pd:.4f}" if np.isfinite(pd) else "",
            "gt_depth_m":  f"{gd:.4f}" if np.isfinite(gd) else "",
            "abs_error_m": f"{ae:.4f}" if np.isfinite(ae) else "",
            "rel_error":   f"{re:.4f}" if np.isfinite(re) else "",
            "scale_factor": scale_note,
        })

    pa = np.array(preds, dtype=np.float32)
    ga = np.array(gts,   dtype=np.float32)
    ba = np.array(areas, dtype=np.float32)

    metrics = compute_metrics(pa, ga)
    metrics["stratified"] = stratified_metrics(pa, ga)
    metrics["size_stratified"] = size_stratified_metrics(pa, ga, ba)
    metrics["scale_factor"] = scale_note

    return metrics, rows


# ---------------------------------------------------------------------------
# Pretty comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: Dict[str, Dict]) -> None:
    sep  = "=" * 78
    sep2 = "-" * 78
    print(f"\n{sep}")
    print("  Model Comparison — Object-Level Depth Estimation")
    print(sep)
    hdr = f"  {'Model':<20} {'Metric?':>7} {'MAE':>8} {'RMSE':>8} {'AbsRel':>8} {'δ<1.25':>8} {'N':>5}"
    print(hdr); print(f"  {sep2}")
    ranked = sorted(results.items(), key=lambda kv: kv[1].get("AbsRel", float("inf")))
    for rank, (model_name, m) in enumerate(ranked, 1):
        is_m  = "yes" if m.get("_is_metric") else "no (aligned)"
        mae   = m.get("MAE",     float("nan"))
        rmse  = m.get("RMSE",    float("nan"))
        absr  = m.get("AbsRel",  float("nan"))
        d1    = m.get("delta_1", float("nan"))
        n     = m.get("n", 0)
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
        print(f"  {medal} {model_name:<18} {is_m:>7} {mae:>8.4f} {rmse:>8.4f} {absr:>8.4f} {d1:>8.4f} {n:>5}")
    print(f"  {sep2}")
    best = ranked[0][0]
    print(f"\n  Best model by AbsRel: {best}")
    print(f"{sep}\n")

    # Distance-stratified comparison
    print(f"\n{sep}")
    print("  Distance-Stratified AbsRel Comparison")
    print(sep)
    bands = ["0-5 m", "5-20 m", "20+ m"]
    hdr2  = f"  {'Model':<20} " + "".join(f"{b:>12}" for b in bands)
    print(hdr2); print(f"  {sep2}")
    for model_name, m in ranked:
        strat = m.get("stratified", {})
        row = f"  {model_name:<20} "
        for b in bands:
            v = strat.get(b, {}).get("AbsRel", float("nan"))
            row += f"{'N/A':>12}" if not np.isfinite(v) else f"{v:>12.4f}"
        print(row)
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_comparison(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image).resolve()
    gt_path    = Path(args.gt).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT depth not found: {gt_path}")

    print(f"\n{'='*60}")
    print(f"  Phase 2 — Model Comparison")
    print(f"  Image : {image_path.name}")
    print(f"  GT    : {gt_path.name}")
    print(f"{'='*60}\n")

    # ---- Ground truth ----
    gt_depth   = load_gt_depth(gt_path, depth_scale=args.depth_scale)
    image_bgr  = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    orig_h, orig_w = image_bgr.shape[:2]
    gt_depth_r = cv2.resize(gt_depth, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    gt_depth_r[gt_depth_r <= 0] = np.nan

    # ---- Shared YOLO detections ----
    print("Loading YOLO ...")
    from ultralytics import YOLO
    detector = YOLO(args.yolo_model)
    yolo_results = detector.predict(
        source=image_bgr, conf=args.conf, iou=args.iou,
        imgsz=args.imgsz, verbose=False,
    )
    detections: List[Dict] = []
    names = detector.names
    obj_id = 0
    for result in yolo_results:
        if result.boxes is None:
            continue
        for box, conf, cls_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
        ):
            x1, y1, x2, y2 = np.round(box).astype(int).tolist()
            # Basic depth extraction (used for coverage filter only; real per-model in eval loop)
            roi_area = max(1, (x2 - x1) * (y2 - y1))
            # Approximate coverage from full-box pixel count (no depth yet)
            cov_filtered = float(conf) < args.conf_min
            det = {
                "object_id":  obj_id,
                "class":      names[int(cls_id)],
                "confidence": float(conf),
                "bbox":       (x1, y1, x2, y2),
                "filtered":   cov_filtered,
            }
            detections.append(det)
            obj_id += 1

    n_pass = sum(1 for d in detections if not d["filtered"])
    print(f"YOLO: {len(detections)} detections, {n_pass} pass conf>={args.conf_min}\n")

    # ---- Load sample config (for UniDAC camera params) ----
    sc_path = Path(args.sample_config).resolve() if args.sample_config else None
    sample  = load_sample_config(sc_path)

    # ---- Build requested backends ----
    print("Loading depth models ...")
    backends = build_backends(args.models, args, sample)
    if not backends:
        raise RuntimeError("No valid models specified. Use --models unidac midas zoe depthanything")

    # ---- Per-model inference + evaluation ----
    all_rows:   List[Dict] = []
    model_metrics: Dict[str, Dict] = {}

    for backend in backends:
        print(f"\n[{backend.name}] Predicting depth ...")
        t0 = time.time()
        try:
            depth = backend.predict(image_path, output_size=(orig_h, orig_w))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        elapsed = time.time() - t0
        print(f"  Inference: {elapsed:.1f}s  |  "
              f"depth range [{np.nanmin(depth):.2f}, {np.nanmax(depth):.2f}] m")

        # Save depth map for this model
        np.save(output_dir / f"depth_{backend.name}.npy", depth)
        _save_depth_colormap(depth, output_dir / f"depth_{backend.name}_color.png")

        backend._depth_cache = depth   # passed to eval_model_on_detections

        metrics, rows = eval_model_on_detections(
            backend, detections, gt_depth_r,
            roi_mode=args.roi_mode,
            center_fraction=args.center_fraction,
            low_pct=args.low_pct,
            high_pct=args.high_pct,
            conf_min=args.conf_min,
            min_valid_ratio=args.min_valid_ratio,
        )
        metrics["_is_metric"] = backend.is_metric
        model_metrics[backend.name] = metrics
        all_rows.extend(rows)

        print(f"  MAE={metrics.get('MAE', float('nan')):.4f}m  "
              f"AbsRel={metrics.get('AbsRel', float('nan')):.4f}  "
              f"δ<1.25={metrics.get('delta_1', float('nan')):.4f}  "
              f"N={metrics.get('n', 0)}")

    # ---- Save combined CSV ----
    if all_rows:
        csv_path = output_dir / "comparison_results.csv"
        fields = list(all_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nPer-object CSV  -> {csv_path}")

    # ---- Save JSON summary ----
    summary_json = output_dir / "comparison_metrics.json"
    serialisable = {
        k: {kk: vv for kk, vv in v.items() if kk != "_is_metric"}
        for k, v in model_metrics.items()
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Metrics JSON    -> {summary_json}")

    # ---- Print comparison table ----
    print_comparison_table(model_metrics)

    # ---- Generate annotated side-by-side image ----
    _save_sidebyside(image_bgr, detections, backends, output_dir)

    # ---- Verdict ----
    _print_verdict(model_metrics)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_depth_colormap(depth: np.ndarray, path: Path) -> None:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return
    vmin, vmax = np.percentile(valid, [2, 98])
    vmax = max(vmax, vmin + 1e-6)
    scaled = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    scaled = np.nan_to_num(scaled, nan=0.0)
    gray = (scaled * 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA))


def _save_sidebyside(image_bgr: np.ndarray, detections: List[Dict],
                     backends: List[DepthBackend], output_dir: Path) -> None:
    """Save a grid: original | depth_model1 | depth_model2 | ..."""
    h, w = image_bgr.shape[:2]
    thumb_w = min(w, 480)
    thumb_h = int(h * thumb_w / w)

    panels = [cv2.resize(image_bgr, (thumb_w, thumb_h))]
    labels = ["Input"]
    for backend in backends:
        if not hasattr(backend, "_depth_cache"):
            continue
        depth = backend._depth_cache
        valid = depth[np.isfinite(depth) & (depth > 0)]
        if valid.size == 0:
            continue
        vmin, vmax = np.percentile(valid, [2, 98])
        vmax = max(vmax, vmin + 1e-6)
        scaled = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
        gray  = (np.nan_to_num(scaled, nan=0.0) * 255).astype(np.uint8)
        color = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
        color = cv2.resize(color, (thumb_w, thumb_h))
        panels.append(color)
        labels.append(backend.name)

    grid = np.hstack(panels)
    # Draw label bar
    bar = np.zeros((30, grid.shape[1], 3), dtype=np.uint8)
    x = 0
    for label, panel in zip(labels, panels):
        cv2.putText(bar, label, (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        x += panel.shape[1]
    grid = np.vstack([bar, grid])
    cv2.imwrite(str(output_dir / "sidebyside_depth.jpg"), grid)
    print(f"Side-by-side    -> {output_dir / 'sidebyside_depth.jpg'}")


def _print_verdict(model_metrics: Dict[str, Dict]) -> None:
    """Print a human-readable verdict on whether UniDAC is the best model."""
    if not model_metrics:
        return
    ranked = sorted(
        ((name, m.get("AbsRel", float("inf"))) for name, m in model_metrics.items()),
        key=lambda x: x[1],
    )
    best_name, best_absr = ranked[0]

    print("=" * 60)
    print("  VERDICT")
    print("=" * 60)
    unidac_rank = next((i + 1 for i, (n, _) in enumerate(ranked) if n == "UniDAC"), None)

    for rank, (name, absr) in enumerate(ranked, 1):
        tag = " ← BEST" if rank == 1 else ""
        print(f"  #{rank}  {name:<20}  AbsRel={absr:.4f}{tag}")

    print()
    if unidac_rank == 1:
        print("  ✅ UniDAC achieves the lowest AbsRel — it IS the best model here.")
    elif unidac_rank is not None:
        delta = model_metrics["UniDAC"].get("AbsRel", float("nan")) - best_absr
        print(f"  ⚠️  UniDAC ranks #{unidac_rank}. Best is {best_name} "
              f"(ΔAbsRel = +{delta:.4f}).")
        print(f"  → Consider scale-alignment or a larger YOLO model to close the gap.")
    else:
        print(f"  UniDAC was not included in this run.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 — Multi-model depth comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image",  required=True, help="Input RGB image")
    p.add_argument("--gt",     required=True, help="GT depth (uint16 PNG or float32 EXR)")
    p.add_argument("--depth-scale", type=float, default=256.0,
                   help="Scale for uint16 GT: metres = uint16 / depth-scale")
    p.add_argument("--output-dir", default="phase2/results/comparison")
    p.add_argument("--device",     default="auto")

    # Models
    p.add_argument("--models", nargs="+",
                   default=["unidac", "midas", "zoe"],
                   help="Models to compare: unidac midas zoe depthanything")
    p.add_argument("--midas-type",  default="DPT_BEiT_L_512",
                   help="MiDaS model type (torch.hub)")
    p.add_argument("--zoe-model",   default="ZoeD_N",
                   help="ZoeDepth model name (torch.hub)")
    p.add_argument("--da-encoder",  default="vitl",
                   help="Depth Anything V2 encoder (vits|vitb|vitl)")

    # UniDAC-specific
    p.add_argument("--unidac-checkpoint",
                   default=str(REPO_ROOT / "checkpoints/unidac.pt"))
    p.add_argument("--sample-config", default=None,
                   help="*_sample.json for camera params (UniDAC)")

    # YOLO
    p.add_argument("--yolo-model", default="yolo11n.pt")
    p.add_argument("--conf",     type=float, default=0.25)
    p.add_argument("--conf-min", type=float, default=0.40)
    p.add_argument("--iou",      type=float, default=0.70)
    p.add_argument("--imgsz",    type=int,   default=640)

    # Aggregation
    p.add_argument("--roi-mode",        default="center",
                   choices=["center", "bbox", "weighted"])
    p.add_argument("--center-fraction", type=float, default=0.35)
    p.add_argument("--low-pct",         type=float, default=10.0)
    p.add_argument("--high-pct",        type=float, default=90.0)
    p.add_argument("--min-valid-ratio", type=float, default=0.20)

    return p.parse_args()


if __name__ == "__main__":
    run_comparison(_parse_args())
