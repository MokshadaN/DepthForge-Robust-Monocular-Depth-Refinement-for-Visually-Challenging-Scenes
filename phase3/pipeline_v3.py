"""
Phase 3 Pipeline: Advanced Projection-Aware & Segmentation-Guided Object Distance Estimation.

Key Enhancements in Phase 3
---------------------------
1. YOLOv11 Instance Segmentation (yolo11n-seg.pt) for exact polygon mask extraction.
2. 360° ERP Tangential Perspective Unwarping (cubemap patching) to eliminate equirectangular distortion.
3. Mask-Guided & Gradient-Aware Depth Aggregation: extracts depth strictly within object polygons,
   filtering out silhouette edge artifacts and background leakage.
4. Scale-calibrated & distance-stratified evaluation across all camera projection geometries.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from ultralytics import YOLO

# Resolve repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase1.pipeline import UniDACDepthRunner, load_sample_config
from phase1.evaluate import (
    evaluate,
    print_metrics,
    load_gt_depth,
)
from phase3.erp_patch import (
    extract_cubemap_patches,
    perspective_from_erp,
    reproject_patch_mask_to_erp,
)


# ---------------------------------------------------------------------------
# Mask & Depth Aggregation Engine
# ---------------------------------------------------------------------------

def extract_mask_guided_depth(
    depth_map: np.ndarray,
    mask: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    trim_pct: Tuple[float, float] = (15.0, 85.0),
    gradient_filter: bool = True,
    grad_thresh_factor: float = 2.5,
) -> Tuple[float, int, float, float]:
    """
    Extract metric depth strictly within an object's segmentation mask,
    with gradient edge filtering and trimmed percentile aggregation.
    """
    valid_mask = (mask > 0) & np.isfinite(depth_map) & (depth_map > 0.05) & (depth_map < 150.0)
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count < 4:
        return float("nan"), 0, float("nan"), float("nan")

    depth_pixels = depth_map[valid_mask]

    # Compute spatial gradient inside bounding ROI for edge outlier rejection
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        roi_depth = np.copy(depth_map[max(0, y1):min(depth_map.shape[0], y2), max(0, x1):min(depth_map.shape[1], x2)])
    else:
        roi_depth = np.copy(depth_map)

    roi_depth = np.nan_to_num(roi_depth, nan=0.0)
    gx = cv2.Sobel(roi_depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    if bbox is not None:
        full_grad = np.zeros_like(depth_map, dtype=np.float32)
        full_grad[max(0, y1):min(depth_map.shape[0], y2), max(0, x1):min(depth_map.shape[1], x2)] = grad_mag
    else:
        full_grad = grad_mag

    grad_pixels = full_grad[valid_mask]
    mean_grad = float(np.mean(grad_pixels)) if grad_pixels.size > 0 else 0.0

    if gradient_filter and valid_count > 15:
        med_grad = float(np.median(grad_pixels))
        grad_cutoff = max(med_grad * grad_thresh_factor, 0.15)
        stable_mask = grad_pixels <= grad_cutoff
        if np.count_nonzero(stable_mask) >= 5:
            depth_pixels = depth_pixels[stable_mask]

    # Trimmed percentile median
    p_low, p_high = np.percentile(depth_pixels, [trim_pct[0], trim_pct[1]])
    trimmed = depth_pixels[(depth_pixels >= p_low) & (depth_pixels <= p_high)]

    if trimmed.size == 0:
        trimmed = depth_pixels

    depth_val = float(np.median(trimmed))
    depth_std = float(np.std(trimmed)) if trimmed.size > 1 else 0.0

    return depth_val, int(trimmed.size), depth_std, mean_grad


# ---------------------------------------------------------------------------
# YOLOv11 Segmenter & Perspective Runner
# ---------------------------------------------------------------------------

class YOLOSegmenter:
    """YOLOv11 Instance Segmentation & Bounding Box detector."""

    def __init__(self, model_path: str = "yolo11n-seg.pt", conf_thresh: float = 0.25):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, image_bgr: np.ndarray) -> List[Dict]:
        """Detect objects with polygon segmentation masks on a single image."""
        h, w = image_bgr.shape[:2]
        results = self.model(image_bgr, conf=self.conf_thresh, verbose=False)[0]

        detections = []
        if len(results.boxes) == 0:
            return detections

        has_masks = results.masks is not None

        for idx, box in enumerate(results.boxes):
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(w, xyxy[2]), min(h, xyxy[3])

            if has_masks and idx < len(results.masks.data):
                raw_mask = results.masks.data[idx].cpu().numpy()
                mask = cv2.resize((raw_mask > 0.5).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255

            detections.append({
                "object_id": idx,
                "class": cls_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "mask": mask,
            })
        return detections

    def detect_erp_cubemap(self, erp_bgr: np.ndarray, fov_deg: float = 90.0) -> List[Dict]:
        """
        Detect objects on a 360° ERP image by extracting perspective patches,
        running YOLO segmentation, and reprojecting masks back to ERP coordinates.
        """
        erp_h, erp_w = erp_bgr.shape[:2]
        patches = extract_cubemap_patches(erp_bgr, fov_deg=fov_deg, yaws=[0.0, 90.0, 180.0, 270.0])

        all_detections = []
        obj_id = 0

        # Ignore non-indoor/erroneous classes on indoor panoramas if conf is low
        ignore_classes = {"train", "airplane"}

        for patch in patches:
            p_img = patch["image"]
            map_x = patch["map_x"]
            map_y = patch["map_y"]

            p_dets = self.detect(p_img)
            for det in p_dets:
                if det["class"] in ignore_classes:
                    continue

                # Reproject perspective mask to ERP
                erp_mask = reproject_patch_mask_to_erp(det["mask"], map_x, map_y, (erp_h, erp_w))
                ys, xs = np.where(erp_mask > 0)
                if len(xs) < 25:
                    continue

                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

                all_detections.append({
                    "object_id": obj_id,
                    "class": det["class"],
                    "confidence": det["confidence"],
                    "bbox": (x1, y1, x2, y2),
                    "mask": erp_mask,
                    "source_yaw": patch["yaw"],
                })
                obj_id += 1

        # Non-maximum suppression / overlap deduplication across overlapping patches
        filtered = self._deduplicate_erp_detections(all_detections, (erp_h, erp_w))
        return filtered

    @staticmethod
    def _deduplicate_erp_detections(detections: List[Dict], shape: Tuple[int, int], iou_thresh: float = 0.40) -> List[Dict]:
        """Filter out duplicate detections across adjacent perspective patch seams."""
        if not detections:
            return []
        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        kept = []
        for det in sorted_dets:
            m1 = det["mask"] > 0
            overlap = False
            for k in kept:
                if k["class"] == det["class"]:
                    m2 = k["mask"] > 0
                    inter = np.count_nonzero(m1 & m2)
                    union = np.count_nonzero(m1 | m2)
                    if union > 0 and (inter / union) > iou_thresh:
                        overlap = True
                        break
            if not overlap:
                kept.append(det)

        for i, d in enumerate(kept):
            d["object_id"] = i
        return kept


# ---------------------------------------------------------------------------
# Visualisation & Annotation
# ---------------------------------------------------------------------------

def draw_phase3_visualisation(
    image_bgr: np.ndarray,
    detections: List[Dict],
    depth_map: np.ndarray,
    output_dir: Path,
) -> None:
    """Render annotated boxes, polygon segmentation masks, and distance tags."""
    h, w = image_bgr.shape[:2]
    annotated = image_bgr.copy()
    mask_overlay = image_bgr.copy()

    # Distinct color palette
    palette = [
        (46, 204, 113),  # emerald green
        (52, 152, 219),  # peter river blue
        (155, 89, 182),  # amethyst purple
        (241, 196, 15),  # sunflower yellow
        (230, 126, 34),  # carrot orange
        (231, 76, 60),   # alizarin red
        (26, 188, 156),  # turquoise
    ]

    for det in detections:
        color = palette[det["object_id"] % len(palette)]
        mask = det.get("mask")
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        depth_val = det.get("pred_depth_m", float("nan"))
        cls_name = det["class"]
        conf = det["confidence"]

        # Blend segmentation mask
        if mask is not None and np.count_nonzero(mask) > 0:
            poly_mask = mask > 0
            mask_overlay[poly_mask] = (
                mask_overlay[poly_mask] * 0.45 + np.array(color, dtype=np.uint8) * 0.55
            ).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color, 2)

        # Distance badge color: Green <= 3m, Yellow 3-10m, Red > 10m
        if np.isfinite(depth_val):
            badge_color = (46, 204, 113) if depth_val <= 3.5 else ((241, 196, 15) if depth_val <= 10.0 else (231, 76, 60))
            depth_str = f"{depth_val:.2f}m"
        else:
            badge_color = (128, 128, 128)
            depth_str = "N/A"

        label = f"#{det['object_id']:02d} {cls_name} {conf:.2f} | {depth_str}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), badge_color, 2)

        # Text banner
        font_scale = 0.52
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty1 = max(0, y1 - th - 6)
        cv2.rectangle(annotated, (x1, ty1), (x1 + tw + 6, ty1 + th + 6), badge_color, -1)
        cv2.putText(annotated, label, (x1 + 3, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    # Blend overlay
    blended = cv2.addWeighted(annotated, 0.7, mask_overlay, 0.3, 0)
    cv2.imwrite(str(output_dir / "annotated_result.jpg"), blended)

    # Save depth colormap
    valid_d = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
    if valid_d.size > 0:
        vmin, vmax = np.percentile(valid_d, [2, 98])
        vmax = max(vmax, vmin + 1e-4)
        norm_d = np.clip((depth_map - vmin) / (vmax - vmin), 0, 1)
        norm_d = np.nan_to_num(norm_d, nan=0.0)
        depth_color = cv2.applyColorMap((norm_d * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        cv2.imwrite(str(output_dir / "depth_color.png"), depth_color)


# ---------------------------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------------------------

def run_phase3_pipeline(
    image_path: str,
    sample_config_path: str,
    checkpoint_path: str,
    yolo_model_path: str = "yolo11n-seg.pt",
    gt_depth_path: Optional[str] = None,
    output_dir: str = "phase3/output",
    conf_threshold: float = 0.25,
    eval_threshold: float = 0.35,
    use_cubemap_unwarping: bool = True,
    device: str = "auto",
) -> Dict:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(sample_config_path, "r") as f:
        cfg = json.load(f)
    is_erp = bool(cfg.get("is_erp", False))
    camera_name = cfg.get("camera_name", Path(image_path).stem)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    raw_h, raw_w = image_bgr.shape[:2]
    sample = load_sample_config(Path(sample_config_path).resolve())
    is_erp = bool(sample.get("erp", False))
    camera_name = sample.get("dataset_name", Path(image_path).stem)

    print("=" * 65)
    print(f"  UniDAC Phase 3 — Advanced Distance Estimation")
    print(f"  Camera  : {camera_name} (ERP={is_erp})")
    print(f"  Image   : {image_path}")
    print("=" * 65)

    # 1. Depth Estimation using tested UniDACDepthRunner
    print("[1/4] Computing full-resolution metric depth map...")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    depth_runner = UniDACDepthRunner(
        checkpoint=checkpoint,
        device=device,
        download_backbone=False,
        sample=sample,
    )
    depth_map = depth_runner.predict(Path(image_path).resolve(), output_size=(raw_h, raw_w))
    np.save(str(out_dir / "depth.npy"), depth_map)

    # 2. Object Detection & Instance Segmentation
    print("[2/4] Running YOLOv11 instance segmentation...")
    segmenter = YOLOSegmenter(yolo_model_path, conf_thresh=conf_threshold)

    if is_erp and use_cubemap_unwarping:
        print("      Applying 360° Tangential Perspective Unwarping...")
        detections = segmenter.detect_erp_cubemap(image_bgr, fov_deg=90.0)
    else:
        detections = segmenter.detect(image_bgr)

    print(f"      Detected {len(detections)} object instances.")

    # 3. Mask-Guided Depth Extraction
    print("[3/4] Extracting mask-guided metric depths...")
    for det in detections:
        mask = det["mask"]
        bbox = det["bbox"]
        pval, valid_px, std, mean_grad = extract_mask_guided_depth(
            depth_map, mask, bbox=bbox, trim_pct=(15.0, 85.0), gradient_filter=True
        )
        det["pred_depth_m"] = pval
        det["valid_pixels"] = valid_px
        det["depth_std"] = std
        det["mean_grad"] = mean_grad
        det["filtered"] = det["confidence"] < eval_threshold

        tag = f"{pval:.2f}m (std={std:.2f}m)" if np.isfinite(pval) else "N/A"
        print(f"      #{det['object_id']:02d} {det['class']:<14} conf={det['confidence']:.2f} depth={tag}")

    # 4. Evaluation against GT (if available)
    eval_metrics = {}
    if gt_depth_path and Path(gt_depth_path).exists():
        print("\n[4/4] Evaluating against ground truth depth...")
        depth_scale = float(cfg.get("depth_scale", 256.0 if gt_depth_path.endswith(".png") else 1.0))
        eval_detections = []
        for det in detections:
            d_eval = copy.deepcopy(det)
            d_eval["estimated_depth_m"] = det["pred_depth_m"]
            d_eval["valid_depth_pixels"] = det["valid_pixels"]
            d_eval["coverage_ratio"] = 1.0
            d_eval["roi_mode"] = "mask"
            eval_detections.append(d_eval)

        eval_metrics = evaluate(
            detections=eval_detections,
            depth_pred=depth_map,
            gt_depth_path=Path(gt_depth_path),
            output_dir=out_dir,
            depth_scale=depth_scale,
            roi_mode="mask",
            scale_align=False,
            image_name=Path(image_path).name,
        )
        print_metrics(eval_metrics, title="Phase 3 Evaluation Results")
    else:
        print("\n[4/4] No GT depth provided; skipping evaluation report.")

    # Visualisation
    draw_phase3_visualisation(image_bgr, detections, depth_map, out_dir)
    print(f"\nArtifacts saved to -> {out_dir}")
    return eval_metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UniDAC Phase 3 Pipeline")
    p.add_argument("--image", required=True, help="Path to input RGB image")
    p.add_argument("--sample-config", required=True, help="Camera sample config JSON")
    p.add_argument("--gt-depth", default=None, help="Path to GT depth map (PNG/EXR)")
    p.add_argument("--unidac-checkpoint", default="checkpoints/unidac.pt", help="UniDAC checkpoint")
    p.add_argument("--yolo-model", default="yolo11n-seg.pt", help="YOLOv11 segmentation model")
    p.add_argument("--output-dir", default="phase3/output", help="Output directory")
    p.add_argument("--conf-thresh", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--eval-thresh", type=float, default=0.35, help="Evaluation confidence cutoff")
    p.add_argument("--no-cubemap", action="store_true", help="Disable ERP cubemap unwarping")
    p.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_phase3_pipeline(
        image_path=args.image,
        sample_config_path=args.sample_config,
        checkpoint_path=args.unidac_checkpoint,
        yolo_model_path=args.yolo_model,
        gt_depth_path=args.gt_depth,
        output_dir=args.output_dir,
        conf_threshold=args.conf_thresh,
        eval_threshold=args.eval_thresh,
        use_cubemap_unwarping=not args.no_cubemap,
        device=args.device,
    )
