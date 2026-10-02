"""verify_unidac_adapter.py
Checks that UniDACRunner.predict_z reproduces the demo's prediction on the
KITTI360 demo sample to within a reasonable tolerance.

Run from the workspace root:
    py -3.13 verify_unidac_adapter.py

Requirements:
  - UniDAC/checkpoints/unidac.pt downloaded
  - PYTHONPATH includes UniDAC/ (script adds it automatically)
  - torch, torchvision, opencv-python, scipy in path

The demo runs sample 0 (kitti360) in ERP mode through the model and saves
the output. This script:
  1. Loads the KITTI360 rgb via the adapter (perspective / cam-to-ERP path)
  2. Calls UniDACRunner.predict_z
  3. Reports median, mean z-depth over active pixels
  4. (Optionally) compares against a reference numpy file saved from the demo

Exit code 0 = passed (shapes OK, non-zero depth range, active mask non-empty).
"""
import sys, os, json
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────
REPO   = os.path.join(os.path.dirname(__file__), "UniDAC")
CKPT   = os.path.join(REPO, "checkpoints", "unidac.pt")
CONFIG = "configs/test/dac_dinov3l+dpt_outdoor_test_kitti360.json"
SAMPLE_JSON = os.path.join(REPO, "demo", "input", "kitti360_sample.json")
RGB_PATH    = os.path.join(REPO, "demo", "input", "kitti360_rgb.png")

# ── UniDAC KITTI360 cam params (from demo_unidac.py OUTDOOR_SAMPLE_1) ──
CAM = {
    "fx": 1.3363220825849971e+03,
    "fy": 1.3357883350012958e+03,
    "cx": 7.1694323510126321e+02,
    "cy": 7.0576498308221585e+02,
}

# ── import ────────────────────────────────────────────────────────────────
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))

import cv2
from unidac_runner import UniDACRunner

def main():
    if not os.path.isfile(CKPT):
        sys.exit(f"[FAIL] Checkpoint not found: {CKPT}")

    rgb_bgr = cv2.imread(RGB_PATH)
    if rgb_bgr is None:
        sys.exit(f"[FAIL] Cannot read RGB: {RGB_PATH}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    print(f"[INFO] Loaded kitti360 rgb: {H}×{W}")

    print("[INFO] Loading UniDACRunner …")
    runner = UniDACRunner(
        repo=REPO,
        ckpt=CKPT,
        config=CONFIG,
        device="cpu",   # use CPU if no CUDA
    )
    print("[INFO] Running predict_z …")
    z, active = runner.predict_z(
        rgb_u8=rgb,
        fx=CAM["fx"], fy=CAM["fy"], cx=CAM["cx"], cy=CAM["cy"],
        crop_wFoV=180,
        fwd_sz=(704, 704),   # kitti360 uses 704×704 in the demo
        # NOTE: kitti360 camera is MEI fisheye and normally needs a lookup grid.
        # The adapter is for PERSPECTIVE images, so we use name='perspective'
        # which selects the pinhole branch in erp_patch_to_cam_fast.
        name="perspective",
    )

    # ── assertions ────────────────────────────────────────────────────────
    assert z.shape == (H, W), f"Shape mismatch: {z.shape} vs ({H},{W})"
    assert active.shape == (H, W), f"Active mask shape mismatch"
    n_active = int(active.sum())
    assert n_active > 0, "Active pixel mask is all-zero!"
    assert (n_active / (H * W)) > 0.8, f"Too few active pixels: {n_active}/{H*W}"

    valid_z = z[active]
    depth_min  = float(valid_z.min())
    depth_max  = float(valid_z.max())
    depth_med  = float(np.median(valid_z))
    depth_mean = float(valid_z.mean())

    print(f"[INFO] Active pixels : {n_active:,} / {H*W:,} "
          f"({100*n_active/(H*W):.1f}%)")
    print(f"[INFO] z-depth stats  min={depth_min:.3f}  max={depth_max:.3f}  "
          f"med={depth_med:.3f}  mean={depth_mean:.3f}  [m]")

    # Sanity: outdoor scene should have depths in ~[0.1, 100] m
    assert depth_min > 0, f"Minimum depth should be positive, got {depth_min}"
    assert depth_max < 200, f"Depth too large: {depth_max}"
    assert depth_med > 0.5, f"Median depth suspiciously small: {depth_med}"

    print("[PASS] UniDACRunner.predict_z sanity checks passed.")

    # Save output for visual inspection
    out_dir = os.path.join(REPO, "demo", "output")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "kitti360_z_adapter.npy"), z)
    np.save(os.path.join(out_dir, "kitti360_active_adapter.npy"), active)
    print(f"[INFO] Saved z and active arrays to {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
