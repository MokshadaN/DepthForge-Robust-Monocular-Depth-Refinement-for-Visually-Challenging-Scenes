"""Diagnostic script to inspect UniDAC scale factors, intermediate tensors, and per-scene statistics."""

import os
import sys
import numpy as np
import torch
from PIL import Image
import cv2

workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from unidac_runner import UniDACRunner
from metrics import masked_metrics


def diagnose_sample(name: str, rgb: np.ndarray, gt: np.ndarray, K: tuple, crop_wFoV: int = 180):
    print(f"\n{'='*60}\nDIAGNOSING SAMPLE: {name}\n{'='*60}")
    H, W = rgb.shape[:2]
    fx, fy, cx, cy = K
    print(f"Image shape: {H}x{W}, Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    runner = UniDACRunner(repo="UniDAC", ckpt="UniDAC/checkpoints/unidac.pt")

    # Run step-by-step to inspect intermediate values
    cam = dict(dataset="perspective", fx=fx, fy=fy, cx=cx, cy=cy)
    img = rgb.astype(np.float32) / 255.0
    dep = np.zeros((H, W, 1), np.float32)
    valid = np.ones((H, W, 1), np.float32)

    cw = int(runner.cano[0] * crop_wFoV / 180)
    ch = int(cw * 512 / 704)

    img_e, dep_e, _, erp_mask, lat, lon = runner._c2e(
        img, dep, valid, 0, np.float32(0), ch, cw,
        runner.cano[0], runner.cano[0] * 2, cam, np.float32(0)
    )
    if dep_e.ndim == 2:
        dep_e = dep_e[:, :, np.newaxis]

    img_r, dep_r, pad, psf, attn, lat_g, lon_g = runner._resize(
        (img_e * 255.0).astype(np.uint8), dep_e, (512, 704), None,
        [img_e.shape[0], img_e.shape[1]], 1.0, padding_rgb=[0, 0, 0],
        mask=erp_mask, lat_grid=lat, long_grid=lon
    )

    import torchvision.transforms.functional as TF
    x = TF.normalize(TF.to_tensor(img_r), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(runner.dev)
    attn_t = TF.to_tensor((attn > 0).astype(np.float32)).unsqueeze(0).to(runner.dev)
    lat_r = torch.tensor([float(lat.min()), float(lat.max())])
    lon_r = torch.tensor([float(lon.min()), float(lon.max())])

    with torch.no_grad():
        preds_raw, _, _ = runner.model(
            x, lat_r.unsqueeze(0).to(runner.dev), lon_r.unsqueeze(0).to(runner.dev),
            attn_mask=attn_t, lat_grid=torch.tensor(lat_g).unsqueeze(0).to(runner.dev)
        )

    preds_scaled = preds_raw * psf
    erp_h = float(runner.cano[0]) * float(psf)

    # Convert back
    _, d_cam_scaled, _, act = runner._e2c(
        x[0].cpu(), preds_scaled[0].cpu(), attn_t[0].cpu(),
        0.0, np.float32(0), H, W, erp_h, erp_h * 2, cam
    )
    # Also test WITHOUT multiplying by psf
    _, d_cam_raw, _, _ = runner._e2c(
        x[0].cpu(), preds_raw[0].cpu(), attn_t[0].cpu(),
        0.0, np.float32(0), H, W, float(runner.cano[0]), float(runner.cano[0]) * 2, cam
    )

    z_scaled = runner.predict_z(rgb, fx, fy, cx, cy)[0]

    # Stats
    raw_np = preds_raw.cpu().numpy()
    scaled_np = preds_scaled.cpu().numpy()
    d_cam_np = d_cam_scaled[0, 0].numpy()

    valid_gt = (gt > 1e-3) & np.isfinite(gt)
    print(f"psf value: {psf}")
    print(f"Raw model pred (ERP):   min={raw_np.min():.3f}, max={raw_np.max():.3f}, med={np.median(raw_np):.3f}")
    print(f"Pred * psf (ERP):       min={scaled_np.min():.3f}, max={scaled_np.max():.3f}, med={np.median(scaled_np):.3f}")
    print(f"d_cam (scaled):         min={d_cam_np[d_cam_np>0].min():.3f}, max={d_cam_np.max():.3f}, med={np.median(d_cam_np[d_cam_np>0]):.3f}")
    print(f"z-depth (current):      min={z_scaled[z_scaled>0].min():.3f}, max={z_scaled.max():.3f}, med={np.median(z_scaled[z_scaled>0]):.3f}")
    if valid_gt.any():
        gt_valid = gt[valid_gt]
        print(f"GT depth:               min={gt_valid.min():.3f}, max={gt_valid.max():.3f}, med={np.median(gt_valid):.3f}")
        s_scaled = np.median(gt_valid) / np.median(z_scaled[valid_gt & (z_scaled > 0)])
        print(f"Alignment scale factor s = median(gt)/median(pred): {s_scaled:.4f}")
        res = masked_metrics(z_scaled, gt, np.ones((H, W), dtype=bool), valid_gt)
        print(f"Raw AbsRel: {res['raw']['absrel']:.4f}, Aligned AbsRel: {res['aligned']['absrel']:.4f}")


def main():
    # 1. Test on scannetpp demo sample
    scannet_rgb_path = "UniDAC/demo/input/scannetpp_rgb.jpg"
    scannet_depth_path = "UniDAC/demo/input/scannetpp_depth.png"
    if os.path.exists(scannet_rgb_path) and os.path.exists(scannet_depth_path):
        rgb = np.array(Image.open(scannet_rgb_path))
        depth = cv2.imread(scannet_depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
        K = (789.9, 791.5, 879.2, 584.8)
        diagnose_sample("ScanNet++ Demo Sample", rgb, depth, K, crop_wFoV=180)

    # 2. Test on synthetic test sample
    H, W = 480, 640
    img = np.full((H, W, 3), 180, dtype=np.uint8)
    img[:, 320:] = np.random.randint(20, 240, (H, 320, 3), dtype=np.uint8)
    gt = np.full((H, W), 2.5, dtype=np.float32)
    K = (518.85, 518.85, 320.0, 240.0)
    diagnose_sample("Synthetic Planar Test Sample (2.5m)", img, gt, K, crop_wFoV=180)


if __name__ == "__main__":
    main()
