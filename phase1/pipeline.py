"""
Phase 1.2: YOLO + UniDAC object-level distance estimation — improved baseline.

New in 1.2
----------
* Gaussian-weighted depth aggregation (--roi-mode weighted)
* Confidence-aware bounding-box visualisation (green/yellow/red by conf tier)
* Box-area coverage filter (--min-valid-ratio) to remove occluded/truncated detections
* Scale-align flag (--scale-align) for test-time median calibration
* Distance-stratified evaluation metrics (0-5 m / 5-20 m / 20+ m)
* Batch evaluator: phase1/batch_eval.py

No model training or modification is performed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from ultralytics import YOLO

# Make imports resolve when this script is run from anywhere inside the repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unidac.models.unidac import UniDAC
from unidac.utils.erp_geometry import (
    cam_to_erp_patch_fast,
    erp_patch_to_cam_fast,
)
from unidac.dataloaders.dataset import resize_for_input
from phase1.evaluate import evaluate, print_metrics


KITT360_SAMPLE = {
    "config_file": REPO_ROOT / "configs/test/dac_dinov3l+dpt_outdoor_test_kitti360.json",
    "image_filename": REPO_ROOT / "demo/input/kitti360_rgb.png",
    "grid_fisheye": REPO_ROOT / "splits/kitti360/grid_fisheye_02.npy",
    "camera_params": {
        "dataset": "kitti360",
        "fx": 1.3363220825849971e3,
        "fy": 1.3357883350012958e3,
        "cx": 7.1694323510126321e2,
        "cy": 7.0576498308221585e2,
        "xi": 2.2134047507854890,
        "k1": 1.6798235660113681e-02,
        "k2": 1.6548773243373522,
        "p1": 4.2223943394772046e-04,
        "p2": 4.2462134260997584e-04,
        "camera_model": "MEI",
    },
    "crop_wfov": 180,
    "fwd_sz": (704, 704),
    "erp": False,
}


def _snap_to_stride(v: int, stride: int = 16) -> int:
    """Round v to nearest multiple of stride (UniDAC median_pooling requires this)."""
    return max(stride, round(v / stride) * stride)


def load_sample_config(json_path: Optional[Path]) -> Dict:
    """
    Load a *_sample.json file (e.g. demo/input/kitti360_sample.json) and
    normalise it into the same structure as KITT360_SAMPLE so the rest of
    the pipeline needs no changes.

    fwd_sz is automatically snapped to multiples of 16 — UniDAC's
    scale_est.py median_pooling patch stride requires this (the KITTI-360
    default of 704 = 44x16 already satisfies it, but ScanNet++'s
    [500, 750] does not).

    If json_path is None the built-in KITT360_SAMPLE is returned.
    """
    if json_path is None:
        return KITT360_SAMPLE

    json_path = Path(json_path).resolve()
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = raw.get("dataset_name", "kitti360")

    # Map dataset name -> UniDAC config file
    _config_map = {
        "kitti360":     "configs/test/dac_dinov3l+dpt_outdoor_test_kitti360.json",
        "scannetpp":    "configs/test/dac_dinov3l+dpt_indoor_test_scannetpp.json",
        "matterport3d": "configs/test/dac_dinov3l+dpt_indoor_test_m3d.json",
        "kitti":        "configs/test/dac_dinov3l+dpt_outdoor_test_kitti.json",
        "nyu":          "configs/test/dac_dinov3l+dpt_indoor_test_nyu.json",
    }
    config_rel = _config_map.get(dataset, _config_map["kitti360"])

    fisheye_grid = raw.get("fishey_grid") or raw.get("grid_fisheye")
    raw_fwd      = raw.get("fwd_sz", [704, 704])

    # Snap fwd_sz to multiples of 16 — required by UniDAC's median_pooling.
    # e.g. ScanNet++ [500, 750] -> [496, 752]  |  Matterport3D [512, 1024] -> unchanged
    fwd_sz = tuple(_snap_to_stride(int(v), 16) for v in raw_fwd)
    if list(fwd_sz) != [int(v) for v in raw_fwd]:
        print(f"      [config] fwd_sz snapped {list(raw_fwd)} -> {list(fwd_sz)} "
              f"(nearest multiples of 16)")

    return {
        "dataset_name":   dataset,
        "config_file":    REPO_ROOT / config_rel,
        "grid_fisheye":   REPO_ROOT / fisheye_grid if fisheye_grid else None,
        "camera_params":  raw.get("cam_params", raw.get("camera_params", {})),
        "crop_wfov":      raw.get("crop_wFoV", raw.get("crop_wfov", 180)),
        "fwd_sz":         fwd_sz,
        "erp":            bool(raw.get("erp", False)),
        "depth_scale":    float(raw.get("depth_scale", 256.0)),
        "gt_depth":       REPO_ROOT / raw["annotation_filename_depth"] if raw.get("annotation_filename_depth") else None,
        "image_filename": REPO_ROOT / raw["image_filename"] if raw.get("image_filename") else None,
    }



class UniDACDepthRunner:
    """Minimal inference wrapper around the official UniDAC model/demo path."""

    def __init__(
        self,
        checkpoint: Path,
        device: str = "auto",
        download_backbone: bool = False,
        sample: Optional[Dict] = None,
    ):
        self.device = self._resolve_device(device)
        self.checkpoint = checkpoint
        self.sample = sample if sample is not None else KITT360_SAMPLE

        config_path = self.sample["config_file"]
        if not config_path.exists():
            raise FileNotFoundError(f"UniDAC config not found: {config_path}")
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"UniDAC checkpoint not found: {checkpoint}\n"
                "Download the official pretrained checkpoint and place it at checkpoints/unidac.pt."
            )

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        self.config = copy.deepcopy(config)

        if not download_backbone:
            self.config["model"]["pixel_encoder"]["pretrained"] = None

        self.model = UniDAC.build(self.config)
        self.model.load_pretrained(str(checkpoint))
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        return torch.device(device)

    @torch.no_grad()
    def predict(self, image_path: Path, output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Predict metric depth and remap into the original camera coordinate frame.

        Supports three camera modes controlled by self.sample:
          - Fisheye (MEI / OPENCV_FISHEYE): erp=False, grid_fisheye supplied
          - 360 ERP (Matterport3D / Pano3D):  erp=True, no fisheye grid needed

        Returns
        -------
        float32 depth array (original_height, original_width).
        Invalid pixels are NaN.
        """
        sample = self.sample
        is_erp = sample.get("erp", False)

        image_path = Path(image_path)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        org_h, org_w = image.shape[:2]

        out_h, out_w = output_size if output_size is not None else (org_h, org_w)

        cano_sz = self.config["data"]["cano_sz"]
        fwd_h, fwd_w = sample["fwd_sz"]
        norm = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

        # =================================================================
        # ERP branch: image is already equirectangular — skip the fisheye
        # cam_to_erp_patch_fast projection and feed it directly.
        # =================================================================
        if is_erp:
            # Resize the ERP image to the model's expected forward size
            image_resized = cv2.resize(image, (fwd_w, fwd_h), interpolation=cv2.INTER_LINEAR)
            image_tensor  = TF.normalize(
                TF.to_tensor(image_resized), **norm
            ).unsqueeze(0).to(self.device)

            # Uniform lat/long grids for a full equirectangular image
            lat_1d  = np.linspace(-np.pi / 2,  np.pi / 2, fwd_h, dtype=np.float32)
            long_1d = np.linspace(-np.pi,       np.pi,     fwd_w, dtype=np.float32)
            lat_grid  = np.tile(lat_1d[:, None],  (1, fwd_w))
            long_grid = np.tile(long_1d[None, :], (fwd_h, 1))

            lat_range  = torch.tensor([-np.pi / 2, np.pi / 2], dtype=torch.float32)
            long_range = torch.tensor([-np.pi,     np.pi],     dtype=torch.float32)

            # Attention mask: all valid pixels
            attn_mask = np.ones((fwd_h, fwd_w), dtype=np.float32)

            batch_attn_mask  = torch.from_numpy(attn_mask).unsqueeze(0).unsqueeze(0).to(self.device)
            batch_lat_range  = lat_range.unsqueeze(0).to(self.device)
            batch_long_range = long_range.unsqueeze(0).to(self.device)
            batch_lat_grid   = torch.from_numpy(lat_grid).unsqueeze(0).to(self.device)
            batch_long_grid  = torch.from_numpy(long_grid).unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds, _, _ = self.model(
                    image_tensor, batch_lat_range, batch_long_range,
                    attn_mask=batch_attn_mask,
                    lat_grid=batch_lat_grid,
                    long_grid=batch_long_grid,
                )

            depth_np = preds[0, 0].detach().cpu().numpy().astype(np.float32)
            if (depth_np.shape[0] != out_h) or (depth_np.shape[1] != out_w):
                depth_np = cv2.resize(depth_np, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            depth_np[~np.isfinite(depth_np)] = np.nan
            return depth_np

        # =================================================================
        # Fisheye branch: project camera image into ERP patch first
        # =================================================================
        image_f     = image.astype(np.float32) / 255.0
        dummy_depth = np.zeros((org_h, org_w, 1), dtype=np.float32)
        dummy_valid = np.zeros_like(dummy_depth)
        phi  = np.array(0, dtype=np.float32)
        roll = np.array(0, dtype=np.float32)
        theta = 0

        crop_width  = int(cano_sz[0] * sample.get("crop_wfov", 180) / 180)
        crop_height = int(crop_width * fwd_h / fwd_w)

        erp_image, _, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image_f, dummy_depth, dummy_valid,
            theta, phi, crop_height, crop_width,
            cano_sz[0], cano_sz[0] * 2,
            sample["camera_params"], roll, scale_fac=None,
        )

        lat_range  = torch.tensor([float(np.min(latitude)),  float(np.max(latitude))],  dtype=torch.float32)
        long_range = torch.tensor([float(np.min(longitude)), float(np.max(longitude))], dtype=torch.float32)

        resized_image, _, _, pred_scale_factor, attn_mask, lat_grid, long_grid = resize_for_input(
            (erp_image * 255.0).astype(np.uint8),
            np.zeros((erp_image.shape[0], erp_image.shape[1], 1), dtype=np.float32),
            (fwd_h, fwd_w), None,
            [erp_image.shape[0], erp_image.shape[1]], 1.0,
            padding_rgb=[0, 0, 0], mask=erp_mask,
            lat_grid=latitude, long_grid=longitude,
        )

        image_tensor      = TF.normalize(TF.to_tensor(resized_image), **norm).unsqueeze(0).to(self.device)
        batch_attn_mask   = TF.to_tensor((attn_mask > 0).astype(np.float32)).unsqueeze(0).to(self.device)
        batch_lat_range   = lat_range.unsqueeze(0).to(self.device)
        batch_long_range  = long_range.unsqueeze(0).to(self.device)
        batch_lat_grid    = torch.tensor(lat_grid).unsqueeze(0).to(self.device)
        batch_long_grid   = torch.tensor(long_grid).unsqueeze(0).to(self.device)

        preds, _, _ = self.model(
            image_tensor, batch_lat_range, batch_long_range,
            attn_mask=batch_attn_mask, lat_grid=batch_lat_grid, long_grid=batch_long_grid,
        )
        preds      = preds * pred_scale_factor
        depth_erp  = preds[0].detach().cpu()
        erp_h      = float(cano_sz[0]) * float(pred_scale_factor)


        # Fisheye: use the pre-computed grid lookup table
        grid_path = sample.get("grid_fisheye")
        if grid_path is None:
            raise ValueError(
                "sample['grid_fisheye'] is required for non-ERP (fisheye) cameras. "
                "Set erp=True for 360-degree images."
            )
        grid_path = Path(grid_path)
        if not grid_path.exists():
            raise FileNotFoundError(f"Fisheye lookup table not found: {grid_path}")

        grid = np.load(grid_path)
        if grid.shape[0] != out_h or grid.shape[1] != out_w:
            grid_xy    = cv2.resize(grid[:, :, :3], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            grid_valid = cv2.resize(grid[:, :, 3],  (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            grid = np.concatenate([grid_xy, grid_valid[..., None]], axis=2)

        dataset_name = sample.get("dataset_name", sample.get("camera_params", {}).get("dataset", "kitti360"))
        _, depth_out, _, active_mask = erp_patch_to_cam_fast(
            image_tensor[0].detach().cpu(), depth_erp,
            batch_attn_mask[0].detach().cpu(),
            0.0, 0.0,
            out_h=out_h, out_w=out_w,
            erp_h=erp_h, erp_w=erp_h * 2,
            cam_params={"dataset": dataset_name},
            fisheye_grid2ray=grid,
        )

        depth  = depth_out[0, 0].numpy().astype(np.float32)
        active = active_mask[0].numpy().astype(bool)
        depth[~active] = np.nan
        return depth


def enhance_textureless_image(
    image_bgr: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8),
    denoise: bool = False,
) -> np.ndarray:
    """
    Enhance contrast and shading cues on low-texture / textureless scenes
    using CLAHE on the luminance (L) channel in LAB color space.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    if denoise:
        enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
    return enhanced


def extract_depth_from_bbox(
    depth: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mask: Optional[np.ndarray] = None,
    mode: str = "center",
    center_fraction: float = 0.50,
    low_pct: float = 10.0,
    high_pct: float = 90.0,
) -> Tuple[float, int, float, float, Tuple[int, int, int, int]]:
    """
    Robust object-level depth aggregation for standard and textureless scenes.

    Modes
    -----
    bbox       : entire bounding box median
    center     : inner (center_fraction) of box, trimmed percentile median
    weighted   : Gaussian-weighted average over inner box (pixels closer to centre contribute more)
    mask       : depth aggregated strictly inside instance segmentation mask
    fg_cluster : foreground-background cluster separation (isolates foreground object from planar wall/floor)

    Returns
    -------
    (depth_value, valid_pixels, depth_std, mean_gradient, roi_box)
    """
    x1, y1, x2, y2 = bbox
    h, w = depth.shape[:2]

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return float("nan"), 0, float("nan"), float("nan"), (x1, y1, x2, y2)

    # Determine ROI boundaries
    if mode in ("bbox", "fg_cluster", "mask"):
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2
    elif mode in ("center", "weighted"):
        bw = x2 - x1
        bh = y2 - y1
        margin_x = int((1.0 - center_fraction) * bw / 2.0)
        margin_y = int((1.0 - center_fraction) * bh / 2.0)
        rx1, ry1 = x1 + margin_x, y1 + margin_y
        rx2, ry2 = x2 - margin_x, y2 - margin_y
    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'bbox', 'center', 'weighted', 'mask', or 'fg_cluster'")

    # Extract mask or rectangular ROI
    if mask is not None and (mode == "mask" or mode == "fg_cluster"):
        obj_mask_roi = mask[ry1:ry2, rx1:rx2]
        roi = depth[ry1:ry2, rx1:rx2]
        valid_mask = obj_mask_roi & np.isfinite(roi) & (roi > 0)
    else:
        roi = depth[ry1:ry2, rx1:rx2]
        valid_mask = np.isfinite(roi) & (roi > 0)

    roi_h, roi_w = roi.shape[:2]
    valid_count = int(valid_mask.sum())

    if valid_count == 0:
        return float("nan"), 0, float("nan"), float("nan"), (rx1, ry1, rx2, ry2)

    # Compute surface depth gradient (surface planarity / slope metric)
    if roi_h > 2 and roi_w > 2:
        filled_roi = np.where(valid_mask, roi, np.nanmedian(roi[valid_mask]))
        gy, gx = np.gradient(filled_roi)
        grad_mag = np.sqrt(gx**2 + gy**2)
        mean_gradient = float(np.nanmean(grad_mag[valid_mask])) if valid_mask.any() else 0.0
    else:
        mean_gradient = 0.0

    values = roi[valid_mask]

    # Mode: fg_cluster (foreground cluster isolation for textureless backgrounds)
    if mode == "fg_cluster" and values.size >= 15:
        lo_v, hi_v = np.percentile(values, [5, 95])
        depth_span = hi_v - lo_v
        if depth_span > 0.35:
            # Separate foreground (nearer) from background plane (farther)
            split_thresh = float(np.median(values))
            near_cluster = values[values <= split_thresh]
            if near_cluster.size >= int(0.20 * values.size):
                values = near_cluster

    # Mode: weighted Gaussian
    if mode == "weighted" and roi_h > 0 and roi_w > 0:
        cy, cx = (roi_h - 1) / 2.0, (roi_w - 1) / 2.0
        sigma = max(min(roi_h, roi_w) / 2.0, 1.0)
        yy, xx = np.mgrid[0:roi_h, 0:roi_w]
        gauss = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        w_map = gauss * valid_mask
        w_sum = w_map.sum()
        if w_sum > 0:
            vals = roi[valid_mask]
            ws   = w_map[valid_mask]
            if vals.size >= 10:
                lo, hi = np.percentile(vals, [low_pct, high_pct])
                keep = (vals >= lo) & (vals <= hi)
                vals, ws = vals[keep], ws[keep]
            if ws.sum() > 0:
                depth_val = float(np.average(vals, weights=ws))
                depth_std = float(np.sqrt(np.average((vals - depth_val) ** 2, weights=ws)))
                return depth_val, valid_count, depth_std, mean_gradient, (rx1, ry1, rx2, ry2)

    # Trimmed percentile median for center / bbox / mask / fg_cluster
    if values.size >= 10:
        lo, hi = np.percentile(values, [low_pct, high_pct])
        values = values[(values >= lo) & (values <= hi)]

    depth_val = float(np.median(values))
    depth_std = float(np.std(values)) if values.size > 0 else 0.0

    return depth_val, valid_count, depth_std, mean_gradient, (rx1, ry1, rx2, ry2)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    vmin, vmax = np.percentile(valid, [2, 98]) if valid.size > 10 else (float(valid.min()), float(valid.max()))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    scaled = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    scaled = np.nan_to_num(scaled, nan=0.0)
    gray = (scaled * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)


def save_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "object_id",
        "class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "roi_mode",
        "estimated_depth_m",
        "valid_depth_pixels",
        "depth_std_m",
        "surface_gradient",
        "coverage_ratio",
        "filtered",
        "filter_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draw_results(
    image_bgr: np.ndarray,
    detections: List[Dict],
    conf_min: float = 0.40,
) -> np.ndarray:
    """
    Draw bounding boxes and instance segmentation masks with confidence-aware colour coding.
    """
    output = image_bgr.copy()
    overlay = output.copy()
    has_mask = False

    for det in detections:
        mask = det.get("mask")
        conf = det["confidence"]
        if mask is not None:
            has_mask = True
            colour = (0, 220, 0) if conf >= conf_min else (0, 200, 255) if conf >= 0.30 else (0, 60, 220)
            overlay[mask] = colour

    if has_mask:
        output = cv2.addWeighted(overlay, 0.35, output, 0.65, 0)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label    = det["class"]
        conf     = det["confidence"]
        depth    = det["estimated_depth_m"]
        cov      = det.get("coverage_ratio", None)
        std_d    = det.get("depth_std_m", None)

        # Colour by confidence tier
        if conf >= conf_min:
            colour = (0, 220, 0)      # green
        elif conf >= 0.30:
            colour = (0, 200, 255)    # yellow-orange
        else:
            colour = (0, 60, 220)     # red

        thickness = max(2, int((y2 - y1) / 120))
        cv2.rectangle(output, (x1, y1), (x2, y2), colour, thickness)

        if np.isfinite(depth):
            line1 = f"{label} {conf:.2f} | {depth:.2f}m"
            if std_d is not None and np.isfinite(std_d) and std_d > 0:
                line1 += f" (+/-{std_d:.2f})"
        else:
            line1 = f"{label} {conf:.2f} | depth N/A"

        font_scale = max(0.50, min(0.85, (x2 - x1) / 280))
        text_y = max(22, y1 - 8)
        cv2.putText(output, line1, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, 2, cv2.LINE_AA)

        if cov is not None:
            line2 = f"cov={cov:.2f}"
            cv2.putText(output, line2, (x1, text_y + int(font_scale * 26)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.65,
                        (200, 200, 200), 1, cv2.LINE_AA)
    return output


def run_pipeline(args: argparse.Namespace) -> None:
    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    print("[1/5] Loading YOLO...")
    detector = YOLO(args.yolo_model)

    print("[2/5] Loading UniDAC...")
    checkpoint = Path(args.unidac_checkpoint).expanduser().resolve()

    # Load sample config (camera type, fisheye grid, ERP flag, …)
    sample_config_path = Path(args.sample_config).resolve() if args.sample_config else None
    sample = load_sample_config(sample_config_path)
    print(f"      Camera: {sample.get('dataset_name', 'kitti360')}  "
          f"ERP={sample.get('erp', False)}  "
          f"fwd_sz={sample.get('fwd_sz')}")

    # Override depth-scale from the sample config if not explicitly passed
    if args.sample_config and not args.depth_scale_override:
        inferred_scale = sample.get("depth_scale", 256.0)
        if args.depth_scale == 256.0 and inferred_scale != 256.0:
            args.depth_scale = inferred_scale
            print(f"      GT depth scale inferred from sample config: {inferred_scale}")

    depth_runner = UniDACDepthRunner(
        checkpoint=checkpoint,
        device=args.device,
        download_backbone=args.download_dinov3_backbone,
        sample=sample,
    )

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"OpenCV could not read: {image_path}")
    orig_h, orig_w = image_bgr.shape[:2]

    # Preprocessing for textureless / low-contrast scenes
    image_for_yolo = image_bgr
    depth_input_path = image_path

    if args.enhance_textureless:
        print(f"[Prep] Applying CLAHE contrast enhancement for textureless scene (clip={args.clahe_clip})...")
        enhanced_bgr = enhance_textureless_image(image_bgr, clip_limit=args.clahe_clip)
        enhanced_path = output_dir / f"enhanced_{image_path.name}"
        cv2.imwrite(str(enhanced_path), enhanced_bgr)
        image_for_yolo = enhanced_bgr
        if args.enhance_depth_input:
            depth_input_path = enhanced_path

    print("[3/5] Running UniDAC depth estimation...")
    depth = depth_runner.predict(depth_input_path, output_size=(orig_h, orig_w))
    if depth.shape != (orig_h, orig_w):
        raise RuntimeError(
            f"Depth/image alignment failed: depth={depth.shape}, image={(orig_h, orig_w)}"
        )

    np.save(output_dir / "depth.npy", depth)
    cv2.imwrite(str(output_dir / "depth_color.png"), colorize_depth(depth))

    print("[4/5] Running YOLO object detection / segmentation...")
    yolo_results = detector.predict(
        source=image_for_yolo,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False,
    )

    detections: List[Dict] = []
    rows: List[Dict] = []
    object_id = 0
    conf_min: float = args.conf_min
    min_valid_ratio: float = args.min_valid_ratio

    names = detector.names
    for result in yolo_results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)

        masks = None
        if result.masks is not None and len(result.masks) == len(boxes):
            raw_masks = result.masks.data.detach().cpu().numpy()
            masks = []
            for m in raw_masks:
                m_resized = cv2.resize(m.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) > 0.5
                masks.append(m_resized)

        for idx, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes)):
            x1, y1, x2, y2 = np.round(box).astype(int).tolist()
            obj_mask = masks[idx] if masks is not None else None

            # Mode selection: auto-use mask if available and requested or default
            effective_roi_mode = args.roi_mode
            if obj_mask is not None and args.roi_mode in ("center", "mask"):
                effective_roi_mode = "mask"

            depth_value, valid_pixels, depth_std, mean_grad, roi_box = extract_depth_from_bbox(
                depth,
                (x1, y1, x2, y2),
                mask=obj_mask,
                mode=effective_roi_mode,
                center_fraction=args.center_fraction,
                low_pct=args.low_pct,
                high_pct=args.high_pct,
            )

            # Coverage ratio: valid depth pixels / total ROI pixels
            rx1, ry1, rx2, ry2 = roi_box
            roi_area = max(1, (rx2 - rx1) * (ry2 - ry1))
            coverage = valid_pixels / roi_area

            cov_filtered = coverage < min_valid_ratio
            conf_filtered = float(conf) < conf_min
            filtered = conf_filtered or cov_filtered

            class_name = names[int(cls_id)]
            det = {
                "object_id":          object_id,
                "class":              str(class_name),
                "confidence":         float(conf),
                "bbox":               (x1, y1, x2, y2),
                "mask":               obj_mask,
                "roi_mode":           effective_roi_mode,
                "estimated_depth_m":  depth_value,
                "valid_depth_pixels": valid_pixels,
                "depth_std_m":        depth_std,
                "surface_gradient":   mean_grad,
                "coverage_ratio":     round(coverage, 3),
                "filtered":           filtered,
                "filter_reason":      (
                    "low_conf" if conf_filtered and not cov_filtered
                    else "low_cov" if cov_filtered and not conf_filtered
                    else "low_conf+low_cov" if filtered
                    else ""
                ),
            }
            detections.append(det)
            rows.append({
                "image":              str(image_path.name),
                "object_id":          object_id,
                "class":              str(class_name),
                "confidence":         f"{float(conf):.6f}",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "roi_mode":           effective_roi_mode,
                "estimated_depth_m":  f"{depth_value:.6f}" if np.isfinite(depth_value) else "",
                "valid_depth_pixels": valid_pixels,
                "depth_std_m":        f"{depth_std:.4f}" if np.isfinite(depth_std) else "",
                "surface_gradient":   f"{mean_grad:.4f}" if np.isfinite(mean_grad) else "",
                "coverage_ratio":     f"{coverage:.3f}",
                "filtered":           "1" if filtered else "0",
                "filter_reason":      det["filter_reason"],
            })
            object_id += 1

    print("[5/5] Saving fused results...")
    annotated = draw_results(image_bgr, detections, conf_min=conf_min)
    cv2.imwrite(str(output_dir / "annotated_result.jpg"), annotated)
    save_csv(rows, output_dir / "object_depths.csv")

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\nDone.")
    print(f"Annotated image : {output_dir / 'annotated_result.jpg'}")
    print(f"Depth map       : {output_dir / 'depth.npy'}")
    print(f"Depth visual    : {output_dir / 'depth_color.png'}")
    print(f"Detections      : {output_dir / 'object_depths.csv'}")
    print(f"\nDetections (conf>={args.conf:.2f}, eval threshold>={conf_min:.2f}):")
    for det in detections:
        depth_text = f"{det['estimated_depth_m']:.2f} m" if np.isfinite(det["estimated_depth_m"]) else "N/A"
        std_text = f" (std={det['depth_std_m']:.2f}m)" if np.isfinite(det['depth_std_m']) and det['depth_std_m'] > 0 else ""
        reason = det.get("filter_reason", "")
        flag = f"  [{reason.upper()}]" if reason else ""
        print(
            f"  #{det['object_id']:02d} {det['class']:<15} "
            f"conf={det['confidence']:.2f}  cov={det['coverage_ratio']:.2f}  "
            f"depth={depth_text}{std_text}{flag}"
        )

    # ---------------------------------------------------------------
    # Optional evaluation against ground-truth depth
    # ---------------------------------------------------------------
    if args.gt_depth:
        gt_path = Path(args.gt_depth).expanduser().resolve()
        print(f"\n[Eval] Comparing against GT depth: {gt_path.name}")
        summary = evaluate(
            detections=detections,
            depth_pred=depth,
            gt_depth_path=gt_path,
            output_dir=output_dir,
            depth_scale=args.depth_scale,
            roi_mode=args.roi_mode,
            center_fraction=args.center_fraction,
            low_pct=args.low_pct,
            high_pct=args.high_pct,
            conf_threshold=conf_min,
            min_valid_ratio=min_valid_ratio,
            scale_align=args.scale_align,
            image_name=image_path.name,
        )
        print_metrics(summary)
        print(f"Eval CSV  : {output_dir / 'eval_results.csv'}")
        print(f"Metrics   : {output_dir / 'metrics.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 YOLO + UniDAC pipeline (with Textureless Image Support)")
    parser.add_argument(
        "--image",
        default=str(KITT360_SAMPLE["image_filename"]),
        help="Input RGB image. Defaults to UniDAC's KITTI-360 sample.",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolo11n.pt",
        help="Ultralytics checkpoint (e.g. yolo11n.pt or yolo11n-seg.pt for instance segmentation).",
    )
    parser.add_argument(
        "--unidac-checkpoint",
        default=str(REPO_ROOT / "checkpoints/unidac.pt"),
        help="Path to official UniDAC checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "phase1/output"),
        help="Directory for output files.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--conf",     type=float, default=0.25, help="YOLO detection confidence threshold.")
    parser.add_argument("--conf-min", type=float, default=0.40,
                        help="Hard confidence floor — detections below this are excluded from metrics.")
    parser.add_argument("--iou",   type=float, default=0.70, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int,   default=640,  help="YOLO inference image size (px).")
    parser.add_argument(
        "--roi-mode",
        choices=["center", "bbox", "weighted", "mask", "fg_cluster"],
        default="center",
        help=(
            "Depth aggregation strategy: "
            "'center' = trimmed median of inner box, "
            "'weighted' = Gaussian-weighted mean (closer to centre = higher weight), "
            "'bbox' = full bounding box median, "
            "'mask' = strictly inside segmentation mask (if available), "
            "'fg_cluster' = 2-cluster foreground-background separation for textureless scenes."
        ),
    )
    parser.add_argument(
        "--center-fraction", type=float, default=0.50,
        help="Fraction of the box used by --roi-mode center/weighted (default 0.50).",
    )
    parser.add_argument("--low-pct",  type=float, default=10.0,
                        help="Lower percentile for depth trimming (default 10).")
    parser.add_argument("--high-pct", type=float, default=90.0,
                        help="Upper percentile for depth trimming (default 90).")
    parser.add_argument(
        "--min-valid-ratio", type=float, default=0.20,
        help="Minimum fraction of valid depth pixels in the ROI. "
             "Boxes below this are flagged low_cov and excluded from metrics.",
    )

    # ---- Textureless Image Handling ----
    parser.add_argument(
        "--enhance-textureless", action="store_true",
        help="Apply CLAHE contrast enhancement on luminance channel to boost shading/contours in textureless scenes.",
    )
    parser.add_argument(
        "--clahe-clip", type=float, default=2.5,
        help="CLAHE contrast clip limit (default 2.5).",
    )
    parser.add_argument(
        "--enhance-depth-input", action="store_true",
        help="Also feed contrast-enhanced image into UniDAC (default feeds enhanced only to YOLO).",
    )

    # ---- Camera / dataset config (Phase 2) ----
    parser.add_argument(
        "--sample-config", default=None,
        help=(
            "Path to a *_sample.json file (e.g. demo/input/kitti360_sample.json). "
            "Selects the correct UniDAC config, camera params, fisheye grid, "
            "ERP flag, and GT depth scale automatically. "
            "Supported datasets: kitti360, scannetpp, matterport3d, kitti, nyu."
        ),
    )
    # ---- Evaluation args ----
    parser.add_argument(
        "--gt-depth", default=None,
        help="Optional GT depth file (uint16 PNG or float32 EXR). Enables evaluation.",
    )
    parser.add_argument(
        "--depth-scale", type=float, default=256.0,
        help="Scale factor: GT_metres = uint16 / depth-scale (default 256). "
             "Auto-inferred from --sample-config when available.",
    )
    parser.add_argument(
        "--depth-scale-override", action="store_true",
        help="Keep --depth-scale even when --sample-config supplies a different value.",
    )
    parser.add_argument(
        "--scale-align", action="store_true",
        help="Apply median-based scale calibration to align pred depth to GT before metrics.",
    )
    parser.add_argument(
        "--download-dinov3-backbone", action="store_true",
        help="Use the official config and download DINOv3 backbone separately.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
