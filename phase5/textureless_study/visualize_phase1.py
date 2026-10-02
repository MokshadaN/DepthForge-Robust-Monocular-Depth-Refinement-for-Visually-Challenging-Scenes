"""Qualitative panel generator for Phase 1.

Generates comparative 5-pane visualizations for samples:
  [RGB Image | Low-Texture Mask | Ground Truth Depth | Predicted Depth | Error Map (|D - D_gt| / D_gt)]
"""

import os
import sys
import numpy as np
import cv2

workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from texture_mask import texture_map, low_texture_mask


def colorize_depth(depth: np.ndarray, vmin: float = None, vmax: float = None, cmap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Map single-channel depth map to RGB colormap."""
    valid = (depth > 1e-3) & np.isfinite(depth)
    if not valid.any():
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    vmin = vmin if vmin is not None else float(np.percentile(depth[valid], 2))
    vmax = vmax if vmax is not None else float(np.percentile(depth[valid], 98))
    norm_d = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    d_u8 = (norm_d * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(d_u8, cmap)
    color[~valid] = 0
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def colorize_error(rel_error: np.ndarray, max_err: float = 0.35) -> np.ndarray:
    """Map relative error (|pred - gt| / gt) to JET/TURBO heat map."""
    norm_err = np.clip(rel_error / max(max_err, 1e-4), 0.0, 1.0)
    err_u8 = (norm_err * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(err_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def create_sample_panel(
    rgb: np.ndarray,
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    tau: float,
    sample_id: str = "sample",
    model_name: str = "UniDAC",
    out_dir: str = "results/phase1/visualizations",
):
    """Generate and save side-by-side diagnostic visualization panel."""
    os.makedirs(out_dir, exist_ok=True)
    H, W = rgb.shape[:2]

    # 1. Low-texture mask
    T = texture_map(rgb)
    lt_mask = low_texture_mask(T, tau)
    mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
    mask_vis[lt_mask] = [230, 80, 80]      # Red for low-texture
    mask_vis[~lt_mask] = [80, 200, 120]    # Green for textured

    # Blend mask with RGB
    mask_overlay = cv2.addWeighted(rgb, 0.6, mask_vis, 0.4, 0)

    # 2. Depth maps (shared vmin, vmax for fair comparison)
    valid_gt = (gt_depth > 1e-3) & np.isfinite(gt_depth)
    vmin = float(np.percentile(gt_depth[valid_gt], 1)) if valid_gt.any() else 0.5
    vmax = float(np.percentile(gt_depth[valid_gt], 99)) if valid_gt.any() else 8.0

    gt_vis = colorize_depth(gt_depth, vmin=vmin, vmax=vmax)
    pred_vis = colorize_depth(pred_depth, vmin=vmin, vmax=vmax)

    # 3. Error map (|pred - gt| / gt)
    valid_both = valid_gt & (pred_depth > 1e-3) & np.isfinite(pred_depth)
    rel_err = np.zeros_like(gt_depth)
    rel_err[valid_both] = np.abs(pred_depth[valid_both] - gt_depth[valid_both]) / gt_depth[valid_both]
    err_vis = colorize_error(rel_err, max_err=0.30)
    err_vis[~valid_both] = 0

    # 4. Add text annotations to panes
    def annotate(img, text):
        out = img.copy()
        cv2.putText(out, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    p1 = annotate(rgb, "RGB Input")
    p2 = annotate(mask_overlay, f"Texture Mask (tau={tau:.3f})")
    p3 = annotate(gt_vis, "Ground Truth Depth")
    p4 = annotate(pred_vis, f"{model_name} Depth")
    p5 = annotate(err_vis, "AbsRel Error Map")

    panel = np.hstack([p1, p2, p3, p4, p5])
    out_file = os.path.join(out_dir, f"{sample_id}_{model_name}_panel.png")
    cv2.imwrite(out_file, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved diagnostic panel: {out_file}")
    return out_file
