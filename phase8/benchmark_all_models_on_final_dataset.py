"""
Phase 8: Memory-Efficient Multi-Model Benchmark Suite on the Final Dataset.
Compares models sequentially to maintain tiny RAM footprint (<200 MB):
  1. MiDaS Small (21.2M)
  2. Phase 8 Refined Adapter (Ours - 21.2M + 0.08M)
  3. DPT_Large (Precomputed Teacher Ground Truth - 345M)

Evaluates on the held-out test split across:
  - Biological Camouflage
  - Transparent Glassware
  - Textureless Architecture
"""
from __future__ import annotations

import gc
import json
import sys
import time
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
from phase8.metrics import compute_metrics, aggregate_metrics

P8_DIR      = Path("phase8").resolve()
DATA_DIR    = P8_DIR / "data"
SPLITS_F    = DATA_DIR / "splits.json"
RESULTS_DIR = P8_DIR / "results"
CHARTS_DIR  = RESULTS_DIR / "charts"
PANELS_DIR  = RESULTS_DIR / "panels"
for d in [CHARTS_DIR, PANELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CKPT_FILE   = Path("checkpoints/phase8_best.pt").resolve()
IMG_SIZE    = (480, 640)

def compute_sobel_energy(img_bgr: np.ndarray) -> torch.Tensor:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(gx ** 2 + gy ** 2)
    norm = mag / (mag.max() + 1e-6)
    return torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float()

def norm_depth_01(arr: np.ndarray) -> np.ndarray:
    p_low  = np.percentile(arr, 2)
    p_high = np.percentile(arr, 98)
    if p_high > p_low:
        norm = np.clip((arr - p_low) / (p_high - p_low + 1e-6), 0.0, 1.0)
        return norm.astype(np.float32)
    return np.zeros_like(arr, dtype=np.float32)

def main():
    print("=" * 90, flush=True)
    print("  PHASE 8: COMPREHENSIVE MULTI-MODEL BENCHMARK SUITE", flush=True)
    print("=" * 90, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on: {device}", flush=True)

    # Load splits
    with open(SPLITS_F, "r") as f:
        splits = json.load(f)
    test_records = splits["test"]

    # Select balanced test subset (8 Camo, 6 Glass, 6 Textureless)
    camo_recs = [r for r in test_records if r["domain"] == "camo"][:8]
    glass_recs = [r for r in test_records if r["domain"] == "transparent"][:6]
    tex_recs = [r for r in test_records if r["domain"] == "textureless"][:6]
    eval_records = camo_recs + glass_recs + tex_recs
    print(f"Evaluating on {len(eval_records)} balanced test samples (Camo: {len(camo_recs)}, Glass: {len(glass_recs)}, Texless: {len(tex_recs)})...", flush=True)

    # 1. Load Base MiDaS Small
    print("\n[1/2] Running Base MiDaS Small Inference...", flush=True)
    midas_small = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    small_trans = midas_transforms.small_transform

    base_preds = {}
    for idx, rec in enumerate(eval_records, 1):
        img_bgr = cv2.imread(rec["rgb"])
        img_res = cv2.resize(img_bgr, (IMG_SIZE[1], IMG_SIZE[0]))
        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        inp_s = small_trans(img_rgb).to(device)
        with torch.no_grad():
            p = midas_small(inp_s)
            p = F.interpolate(p.unsqueeze(1), size=IMG_SIZE, mode="bicubic", align_corners=False).squeeze().cpu().numpy()
            base_preds[rec["name"]] = norm_depth_01(p)

    del midas_small
    gc.collect()

    # 2. Load Phase 8 Refined Adapter
    print("[2/2] Running Phase 8 Refined Adapter Inference...", flush=True)
    adapter = MultiScaleDilatedAdapter().to(device)
    adapter.load_state_dict(torch.load(CKPT_FILE, map_location=device, weights_only=True))
    adapter.eval()

    refined_preds = {}
    for idx, rec in enumerate(eval_records, 1):
        img_bgr = cv2.imread(rec["rgb"])
        img_res = cv2.resize(img_bgr, (IMG_SIZE[1], IMG_SIZE[0]))
        rgb_t   = torch.from_numpy(img_res).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
        base_t  = torch.from_numpy(base_preds[rec["name"]]).unsqueeze(0).unsqueeze(0).float().to(device)
        energy  = compute_sobel_energy(img_res).to(device)
        with torch.no_grad():
            p_ref = adapter(base_t, rgb_t, energy).squeeze().cpu().numpy()
            refined_preds[rec["name"]] = p_ref

    # Compute Metrics across all domains
    models = ["MiDaS Small (Baseline)", "Phase 8 Refined (Ours)"]
    domain_scores = {m: {"all": [], "camo": [], "transparent": [], "textureless": []} for m in models}
    visual_samples = []

    for idx, rec in enumerate(eval_records, 1):
        name = rec["name"]
        dom  = rec["domain"]
        gt_np = np.load(rec["depth_gt"]).astype(np.float32)
        t_gt = torch.from_numpy(gt_np).float()

        # MiDaS Small
        t_base = torch.from_numpy(base_preds[name]).float()
        m_base = compute_metrics(t_base, t_gt, align_scale=True)
        domain_scores["MiDaS Small (Baseline)"]["all"].append(m_base)
        domain_scores["MiDaS Small (Baseline)"][dom].append(m_base)

        # Phase 8 Refined
        t_ref = torch.from_numpy(refined_preds[name]).float()
        m_ref = compute_metrics(t_ref, t_gt, align_scale=True)
        domain_scores["Phase 8 Refined (Ours)"]["all"].append(m_ref)
        domain_scores["Phase 8 Refined (Ours)"][dom].append(m_ref)

        if len(visual_samples) < 6:
            img_bgr = cv2.imread(rec["rgb"])
            img_res = cv2.resize(img_bgr, (IMG_SIZE[1], IMG_SIZE[0]))
            visual_samples.append((name, dom, img_res, base_preds[name], refined_preds[name], gt_np))

    # Print Leaderboard
    print("\n" + "=" * 105, flush=True)
    print("  MULTI-MODEL BENCHMARK LEADERBOARD (TEST SET - ALL DOMAINS)", flush=True)
    print("=" * 105, flush=True)
    print(f"| {'Model Architecture':<28} | {'Params':<12} | {'AbsRel (lower)':<14} | {'RMSE (lower)':<12} | {'d1 1.25 (higher)':<17} | {'SILog (lower)':<14} |", flush=True)
    print("-" * 105, flush=True)

    param_counts = {
        "MiDaS Small (Baseline)": "21.2M",
        "Phase 8 Refined (Ours)": "21.2M + 0.08M"
    }

    final_table_data = {}
    for m in models:
        agg = aggregate_metrics(domain_scores[m]["all"])
        final_table_data[m] = agg
        print(f"| {m:<28} | {param_counts[m]:<12} | {agg['AbsRel']:.4f}         | {agg['RMSE']:.4f}       | {agg['d1']:.4f}             | {agg['SILog']:.4f}         |", flush=True)
    print("=" * 105, flush=True)

    # Breakdown per domain
    print("\n" + "=" * 105, flush=True)
    print("  DOMAIN-STRATIFIED BREAKDOWN (AbsRel / SILog)", flush=True)
    print("=" * 105, flush=True)
    print(f"| {'Model':<28} | {'Camouflage (AbsRel / SILog)':<30} | {'Transparent Glass (AbsRel / SILog)':<36} |", flush=True)
    print("-" * 105, flush=True)
    for m in models:
        c_agg = aggregate_metrics(domain_scores[m]["camo"])
        g_agg = aggregate_metrics(domain_scores[m]["transparent"])
        print(f"| {m:<28} | {c_agg['AbsRel']:.4f} / {c_agg['SILog']:.4f}             | {g_agg['AbsRel']:.4f} / {g_agg['SILog']:.4f}                      |", flush=True)
    print("=" * 105, flush=True)

    # ── GENERATE COMPARATIVE CHARTS ───────────────────────────────────────────
    print("\nGenerating comparative charts...", flush=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    model_labels = ["MiDaS Small\n(Baseline)", "Phase 8 Refined\n(Ours)"]
    colors = ["#e74c3c", "#2ecc71"]

    absrel_vals = [final_table_data[m]["AbsRel"] for m in models]
    silog_vals  = [final_table_data[m]["SILog"] for m in models]

    ax1.bar(model_labels, absrel_vals, color=colors, width=0.45, edgecolor='black')
    ax1.set_title("Overall AbsRel Error (Lower = Better)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("AbsRel Error")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    ax2.bar(model_labels, silog_vals, color=colors, width=0.45, edgecolor='black')
    ax2.set_title("Overall SILog Error (Lower = Better)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("SILog Error")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    plt.suptitle("Multi-Model Depth Benchmark Comparison on Hard Optical Failure Cases", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    chart_p = CHARTS_DIR / "multi_model_benchmark_barchart.png"
    plt.savefig(str(chart_p), dpi=160)
    plt.close()
    print(f"  [+] Saved benchmark bar chart: {chart_p.name}", flush=True)

    # ── GENERATE SIDE-BY-SIDE PANEL ───────────────────────────────────────────
    print("Generating Multi-Model Side-by-Side Comparison Panel...", flush=True)
    n_vis = len(visual_samples)
    fig, axes = plt.subplots(n_vis, 4, figsize=(14, 3.2 * n_vis))
    if n_vis == 1: axes = np.expand_dims(axes, 0)

    for i, (name, dom, rgb_im, p_sm, p_rf, gt_im) in enumerate(visual_samples):
        # Resize for lightweight rendering
        rgb_thumb = cv2.resize(cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB), (320, 240))
        sm_thumb  = cv2.resize(p_sm, (320, 240))
        rf_thumb  = cv2.resize(p_rf, (320, 240))
        gt_thumb  = cv2.resize(gt_im, (320, 240))

        # 1. RGB
        axes[i, 0].imshow(rgb_thumb)
        axes[i, 0].set_title(f"Input RGB ({dom})\n{name}", fontsize=9, fontweight="bold")
        axes[i, 0].axis("off")

        # 2. MiDaS Small
        axes[i, 1].imshow(sm_thumb, cmap="inferno")
        axes[i, 1].set_title("MiDaS Small (21M Base)", fontsize=9)
        axes[i, 1].axis("off")

        # 3. Phase 8 Refined (Ours)
        axes[i, 2].imshow(rf_thumb, cmap="inferno")
        axes[i, 2].set_title("Phase 8 Refined (Ours)\n[21M + 0.08M]", fontsize=9, fontweight="bold", color="green")
        axes[i, 2].axis("off")

        # 4. Pseudo-GT (DPT_Large)
        axes[i, 3].imshow(gt_thumb, cmap="inferno")
        axes[i, 3].set_title("DPT_Large Pseudo-GT", fontsize=9)
        axes[i, 3].axis("off")

    plt.suptitle("Comprehensive Multi-Model Side-by-Side Visual Comparison", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    multi_panel_p = PANELS_DIR / "all_models_side_by_side_panel.png"
    plt.savefig(str(multi_panel_p), dpi=100)
    plt.close("all")
    print(f"  [+] Saved multi-model comparison panel: {multi_panel_p.name}", flush=True)
    print("\n[COMPLETE] Multi-model benchmark suite finished successfully!", flush=True)

if __name__ == "__main__":
    main()
