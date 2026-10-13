"""
Phase 8: Comprehensive Local Model Evaluator & Comparison Panel Generator.
Loads trained weights from checkpoints/phase8_best.pt and evaluates on local test data.
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase8.model_adapter import MultiScaleDilatedAdapter
from phase8.metrics import compute_metrics, aggregate_metrics, format_metrics

P8_DIR      = Path("phase8").resolve()
DATA_DIR    = P8_DIR / "data"
SPLITS_F    = DATA_DIR / "splits.json"
RESULTS_DIR = P8_DIR / "results"
PANELS_DIR  = RESULTS_DIR / "panels"
PANELS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_FILE   = Path("checkpoints/phase8_best.pt").resolve()

IMG_SIZE = (480, 640)

def compute_sobel_energy(img_bgr: np.ndarray) -> torch.Tensor:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(gx ** 2 + gy ** 2)
    norm = mag / (mag.max() + 1e-6)
    return torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float()

def align_scale_shift(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    valid = (gt > 1e-3) & (pred > 1e-3)
    p = pred[valid]
    g = gt[valid]
    if len(p) < 10:
        return pred
    A = np.stack([p, np.ones_like(p)], axis=1)
    s, t = np.linalg.lstsq(A, g, rcond=None)[0]
    return np.clip(s * pred + t, 0.0, 1.0)

def main():
    print("=" * 80)
    print("  PHASE 8: LOCAL MODEL EVALUATION & VISUALIZATION")
    print("=" * 80)
    
    if not CKPT_FILE.exists():
        print(f"Error: Checkpoint not found at {CKPT_FILE}")
        return
        
    if not SPLITS_F.exists():
        print(f"Error: Splits not found at {SPLITS_F}")
        return

    with open(SPLITS_F, "r") as f:
        splits = json.load(f)
    test_records = splits["test"]
    print(f"Loaded {len(test_records)} test samples from {SPLITS_F}")

    # Load Adapter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = MultiScaleDilatedAdapter().to(device)
    state = torch.load(CKPT_FILE, map_location=device, weights_only=True)
    adapter.load_state_dict(state)
    adapter.eval()
    print(f"Loaded model weights from {CKPT_FILE} ({sum(p.numel() for p in adapter.parameters())} params)")

    # Load MiDaS Small for base depth inference
    print("Loading base MiDaS Small model...")
    base_midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    small_transform = midas_transforms.small_transform

    domain_results = {"camo": {"base": [], "refined": []},
                      "transparent": {"base": [], "refined": []},
                      "textureless": {"base": [], "refined": []},
                      "all": {"base": [], "refined": []}}

    sample_plots = []

    print("\nRunning evaluation on test set...")
    for idx, rec in enumerate(test_records, 1):
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
        gt_np = np.load(str(gt_p)).astype(np.float32)

        # Base depth inference
        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        inp = small_transform(img_rgb).to(device)
        with torch.no_grad():
            b_pred = base_midas(inp)
            b_pred = F.interpolate(b_pred.unsqueeze(1), size=IMG_SIZE, mode="bicubic", align_corners=False).squeeze()
            b_min, b_max = b_pred.min(), b_pred.max()
            if b_max > b_min:
                b_pred = (b_pred - b_min) / (b_max - b_min)
            base_np = b_pred.cpu().numpy()

            # Adapter forward pass
            rgb_t  = torch.from_numpy(img_res).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            base_t = torch.from_numpy(base_np).unsqueeze(0).unsqueeze(0).float().to(device)
            energy = compute_sobel_energy(img_res).to(device)
            ref_t  = adapter(base_t, rgb_t, energy)
            ref_np = ref_t.squeeze().cpu().numpy()

        t_base = torch.from_numpy(base_np).float()
        t_ref  = torch.from_numpy(ref_np).float()
        t_gt   = torch.from_numpy(gt_np).float()

        m_base = compute_metrics(t_base, t_gt, align_scale=True)
        m_ref  = compute_metrics(t_ref, t_gt, align_scale=True)

        domain_results["all"]["base"].append(m_base)
        domain_results["all"]["refined"].append(m_ref)
        if dom in domain_results:
            domain_results[dom]["base"].append(m_base)
            domain_results[dom]["refined"].append(m_ref)

        if len(sample_plots) < 6:
            sample_plots.append((name, dom, img_res, base_np, ref_np, gt_np))

        if idx % 20 == 0 or idx == len(test_records):
            print(f"  [{idx}/{len(test_records)}] processed")

    # Print summary table
    print("\n" + "=" * 90)
    print("  PHASE 8 FINAL SCALE-ALIGNED BENCHMARK METRICS (TEST SET)")
    print("=" * 90)
    print(f"| {'Domain':<15} | {'AbsRel (lower)':<20} | {'d1 1.25 (higher)':<18} | {'SILog (lower)':<16} |")
    print("-" * 90)

    for dom_k in ["all", "camo", "transparent", "textureless"]:
        b_list = domain_results[dom_k]["base"]
        r_list = domain_results[dom_k]["refined"]
        if not b_list:
            continue
        b_agg = aggregate_metrics(b_list)
        r_agg = aggregate_metrics(r_list)
        imp = (b_agg['AbsRel'] - r_agg['AbsRel']) / b_agg['AbsRel'] * 100
        dom_title = "Overall (All)" if dom_k == "all" else dom_k.capitalize()
        print(f"| {dom_title:<15} | {b_agg['AbsRel']:.4f} -> {r_agg['AbsRel']:.4f} ({imp:+5.1f}%) | {b_agg['d1']:.4f} -> {r_agg['d1']:.4f} | {b_agg['SILog']:.4f} -> {r_agg['SILog']:.4f} |")
    print("=" * 90)

    # Save visual comparison panels
    n_samples = len(sample_plots)
    if n_samples > 0:
        fig, axes = plt.subplots(n_samples, 4, figsize=(18, 4 * n_samples))
        if n_samples == 1:
            axes = [axes]
        for row, (name, dom, img_bgr, b_d, r_d, g_d) in enumerate(sample_plots):
            axes[row][0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            axes[row][0].set_title(f"Input RGB ({dom})\n{name}", fontsize=10)
            axes[row][0].axis("off")

            axes[row][1].imshow(b_d, cmap="inferno")
            axes[row][1].set_title("MiDaS Base Depth", fontsize=10)
            axes[row][1].axis("off")

            axes[row][2].imshow(r_d, cmap="inferno")
            axes[row][2].set_title("Phase 8 Refined (Ours)", fontsize=10, fontweight="bold")
            axes[row][2].axis("off")

            axes[row][3].imshow(g_d, cmap="inferno")
            axes[row][3].set_title("DPT_Large Pseudo-GT", fontsize=10)
            axes[row][3].axis("off")

        plt.tight_layout()
        out_panel = PANELS_DIR / "phase8_local_comparison_panel.png"
        plt.savefig(str(out_panel), dpi=150)
        plt.close()
        print(f"\nVisual comparison panel saved -> {out_panel}")

if __name__ == "__main__":
    main()
