"""
Phase 8: Extended Multi-Domain Intermediate Visual Results Generator.
Generates comprehensive 6-column intermediate diagnostic panels for:
  1. Camouflage benchmark scenes
  2. Transparent glassware & refraction scenes
  3. Textureless architectural & blank wall scenes
  4. YOLO Object Detection + Distance Estimation Comparison
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase8.model_adapter import MultiScaleDilatedAdapter

P8_DIR      = Path("phase8").resolve()
DATA_DIR    = P8_DIR / "data"
SPLITS_F    = DATA_DIR / "splits.json"
RESULTS_DIR = P8_DIR / "results"
PANELS_DIR  = RESULTS_DIR / "panels"
PANELS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_FILE   = Path("checkpoints/phase8_best.pt").resolve()

IMG_SIZE = (480, 640)

def compute_sobel_energy(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(gx ** 2 + gy ** 2)
    return mag / (mag.max() + 1e-6)

def align_scale_shift(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    valid = (gt > 1e-3) & (pred > 1e-3)
    p, g = pred[valid], gt[valid]
    if len(p) < 10:
        return pred
    A = np.stack([p, np.ones_like(p)], axis=1)
    s, t = np.linalg.lstsq(A, g, rcond=None)[0]
    return np.clip(s * pred + t, 0.0, 1.0)

def generate_domain_panel(samples, out_path, title_prefix):
    n = len(samples)
    if n == 0:
        return
    # Columns: [RGB, Sobel Energy, Base Depth, Refined Depth, DPT Teacher, Error Map]
    fig, axes = plt.subplots(n, 6, figsize=(24, 3.8 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
        
    for i, item in enumerate(samples):
        name    = item["name"]
        rgb     = cv2.cvtColor(item["img_bgr"], cv2.COLOR_BGR2RGB)
        energy  = item["energy"]
        base_d  = item["base_d"]
        ref_d   = item["ref_d"]
        gt_d    = item["gt_d"]
        err_map = np.abs(ref_d - gt_d)
        
        # 1. RGB
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Input RGB\n({name})", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")
        
        # 2. Sobel Energy
        axes[i, 1].imshow(energy, cmap="bone")
        axes[i, 1].set_title("Sobel Energy (Edge Prior)", fontsize=11)
        axes[i, 1].axis("off")
        
        # 3. Base Depth
        axes[i, 2].imshow(base_d, cmap="inferno")
        axes[i, 2].set_title("Base Model (MiDaS)", fontsize=11)
        axes[i, 2].axis("off")
        
        # 4. Refined Depth
        axes[i, 3].imshow(ref_d, cmap="inferno")
        axes[i, 3].set_title("Phase 8 Refined (Ours)", fontsize=11, fontweight="bold", color="darkblue")
        axes[i, 3].axis("off")
        
        # 5. Teacher Pseudo-GT
        axes[i, 4].imshow(gt_d, cmap="inferno")
        axes[i, 4].set_title("DPT_Large Teacher GT", fontsize=11)
        axes[i, 4].axis("off")
        
        # 6. Error Residual Map
        im = axes[i, 5].imshow(err_map, cmap="plasma", vmin=0.0, vmax=0.4)
        axes[i, 5].set_title("Absolute Error Map (|Ours - GT|)", fontsize=11)
        axes[i, 5].axis("off")
        
    plt.suptitle(f"Phase 8 Intermediate Diagnostics & Depth Refinement: {title_prefix}", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved panel: {out_path.name}")

def main():
    print("=" * 80)
    print("  PHASE 8: GENERATING EXTENDED INTERMEDIATE RESULTS ACROSS ALL DOMAINS")
    print("=" * 80)

    with open(SPLITS_F, "r") as f:
        splits = json.load(f)
    test_records = splits["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = MultiScaleDilatedAdapter().to(device)
    adapter.load_state_dict(torch.load(CKPT_FILE, map_location=device, weights_only=True))
    adapter.eval()

    base_midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    small_transform = midas_transforms.small_transform

    camo_samples = []
    glass_samples = []
    tex_samples = []

    for rec in test_records:
        rgb_p = Path(rec["rgb"])
        gt_p  = Path(rec["depth_gt"])
        dom   = rec["domain"]
        name  = rec["name"]

        if not rgb_p.exists() or not gt_p.exists():
            continue

        img_bgr = cv2.imread(str(rgb_p))
        if img_bgr is None:
            continue
        img_res = cv2.resize(img_bgr, (IMG_SIZE[1], IMG_SIZE[0]))
        gt_np   = np.load(str(gt_p)).astype(np.float32)
        sobel_e = compute_sobel_energy(img_res)

        # Base depth
        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        inp = small_transform(img_rgb).to(device)
        with torch.no_grad():
            b_pred = base_midas(inp)
            b_pred = F.interpolate(b_pred.unsqueeze(1), size=IMG_SIZE, mode="bicubic", align_corners=False).squeeze()
            b_min, b_max = b_pred.min(), b_pred.max()
            if b_max > b_min:
                b_pred = (b_pred - b_min) / (b_max - b_min)
            base_np = b_pred.cpu().numpy()

            # Refined depth
            rgb_t  = torch.from_numpy(img_res).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            base_t = torch.from_numpy(base_np).unsqueeze(0).unsqueeze(0).float().to(device)
            energy_t = torch.from_numpy(sobel_e).unsqueeze(0).unsqueeze(0).float().to(device)
            ref_t  = adapter(base_t, rgb_t, energy_t)
            ref_np = ref_t.squeeze().cpu().numpy()

        base_aligned = align_scale_shift(base_np, gt_np)
        ref_aligned  = align_scale_shift(ref_np, gt_np)

        item = {
            "name": name,
            "domain": dom,
            "img_bgr": img_res,
            "energy": sobel_e,
            "base_d": base_aligned,
            "ref_d": ref_aligned,
            "gt_d": gt_np
        }

        if dom == "camo" and len(camo_samples) < 4:
            camo_samples.append(item)
        elif dom == "transparent" and len(glass_samples) < 4:
            glass_samples.append(item)
        elif dom == "textureless" and len(tex_samples) < 4:
            tex_samples.append(item)

        if len(camo_samples) >= 4 and len(glass_samples) >= 4 and len(tex_samples) >= 4:
            print("  Collected 4 samples for all domains, stopping inference loop early.")
            break

    print("\nGenerating intermediate comparison figures...")
    generate_domain_panel(camo_samples, PANELS_DIR / "panel_1_camouflage_intermediate_results.png", "Biological Camouflage")
    generate_domain_panel(glass_samples, PANELS_DIR / "panel_2_transparent_glass_intermediate_results.png", "Transparent Glassware & Refraction")
    generate_domain_panel(tex_samples, PANELS_DIR / "panel_3_textureless_intermediate_results.png", "Textureless Architecture & Blank Walls")

    print("\n[COMPLETE] All extended intermediate results generated in:", PANELS_DIR)

if __name__ == "__main__":
    main()
