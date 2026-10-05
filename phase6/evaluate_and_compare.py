"""
Phase 6: Head-to-Head Comparison: Before vs. After Fine-Tuning UniDAC on Camouflage & Textureless Scenes.
"""

from __future__ import annotations

import sys
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase1.pipeline import load_sample_config, UniDACDepthRunner
from phase4.compare_all_models import MiDaSModelBackend
from phase6.model_adapter import TextureAwareResidualHead
from phase6.train_adapter import compute_energy_tensor


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })


def compute_metrics(depth_map: np.ndarray, energy_map: np.ndarray):
    # Boundary Sharpness (BSI)
    gx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    bsi = float(np.percentile(grad, 90))

    # Low-texture roughness
    low_tex = energy_map < 0.20
    if np.count_nonzero(low_tex) > 50:
        lap = np.abs(cv2.Laplacian(depth_map, cv2.CV_32F))
        roughness = float(np.mean(lap[low_tex]))
    else:
        roughness = 0.0

    return bsi, roughness


def run_evaluation():
    setup_style()
    camo_dir = Path("phase5/data/camo_scenes").resolve()
    out_dir = Path("phase6/results").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("  PHASE 6: COMPREHENSIVE BEFORE VS. AFTER FINE-TUNING EVALUATION")
    print("=" * 88)

    # 1. Load Pretrained Base UniDAC
    print("Loading Pretrained Base UniDAC...")
    cfg = load_sample_config(Path("demo/input/scannetpp_sample.json").resolve())
    unidac_runner = UniDACDepthRunner(
        checkpoint=Path("checkpoints/unidac.pt").resolve(),
        sample=cfg
    )

    # 2. Load Fine-Tuned Adapter
    print("Loading Fine-Tuned Camouflage & Texture Adapter...")
    adapter = TextureAwareResidualHead()
    ckpt_p = Path("checkpoints/unidac_camo_adapted.pt").resolve()
    adapter.load_state_dict(torch.load(str(ckpt_p), weights_only=True))
    adapter.eval()

    # 3. Load MiDaS Baseline
    print("Loading MiDaS v3.1 Baseline...")
    midas_backend = MiDaSModelBackend("DPT_Hybrid", device="auto")

    scenes = sorted(list(camo_dir.glob("*.jpg")))
    results = []
    panel_paths = []

    temp_p = out_dir / "temp_eval.jpg"

    for idx, sc in enumerate(scenes, 1):
        scene_name = sc.stem
        img_bgr = cv2.imread(str(sc))
        h, w = img_bgr.shape[:2]

        print(f"[{idx}/{len(scenes)}] Evaluating {scene_name}...")

        # A. Baseline UniDAC (Before)
        cv2.imwrite(str(temp_p), img_bgr)
        depth_before = unidac_runner.predict(temp_p, output_size=(h, w))

        # B. Fine-Tuned UniDAC (After)
        rgb_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        depth_tensor = torch.from_numpy(depth_before).unsqueeze(0).unsqueeze(0).float()
        energy_tensor = compute_energy_tensor(img_bgr)
        
        with torch.no_grad():
            depth_after_tensor = adapter(depth_tensor, rgb_tensor, energy_tensor)
        depth_after = depth_after_tensor.squeeze().cpu().numpy()

        # C. MiDaS Baseline
        midas_raw = midas_backend.predict(img_bgr, {}, output_size=(h, w))
        scale = float(np.median(depth_before) / (np.median(midas_raw) + 1e-6))
        depth_midas = midas_raw * scale

        energy_np = energy_tensor.squeeze().cpu().numpy()

        # Compute Metrics
        bsi_before, r_before = compute_metrics(depth_before, energy_np)
        bsi_after, r_after = compute_metrics(depth_after, energy_np)
        bsi_midas, r_midas = compute_metrics(depth_midas, energy_np)

        # Difference map
        diff_map = np.abs(depth_after - depth_before)

        results.append({
            "scene": scene_name,
            "bsi_before": bsi_before, "bsi_after": bsi_after, "bsi_midas": bsi_midas,
            "r_before": r_before, "r_after": r_after, "r_midas": r_midas
        })

        # Visual Formatting
        def to_magma(d_map):
            d_min, d_max = np.percentile(d_map, 2), np.percentile(d_map, 98)
            if d_max - d_min < 1e-4:
                d_max = d_min + 1.0
            norm = np.clip((d_map - d_min) / (d_max - d_min), 0, 1)
            return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

        def to_turbo(d_map):
            norm = np.clip(d_map / (np.percentile(d_map, 99) + 1e-4), 0, 1)
            return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

        tw, th = 320, int(h * 320 / w)
        p_rgb = cv2.resize(img_bgr, (tw, th))
        p_bef = cv2.resize(to_magma(depth_before), (tw, th))
        p_aft = cv2.resize(to_magma(depth_after), (tw, th))
        p_dif = cv2.resize(to_turbo(diff_map), (tw, th))
        p_mid = cv2.resize(to_magma(depth_midas), (tw, th))

        def add_banner(img, title, subtitle, color_bar=(24, 24, 24)):
            banner = np.zeros((48, img.shape[1], 3), dtype=np.uint8)
            banner[:] = color_bar
            cv2.putText(banner, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(banner, subtitle, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
            return np.vstack([banner, img])

        p1 = add_banner(p_rgb, "1. Input RGB", scene_name.replace("_", " ").title())
        p2 = add_banner(p_bef, "2. UniDAC (Before)", f"BSI = {bsi_before:.3f} | Rough = {r_before:.4f}", (40, 30, 20))
        p3 = add_banner(p_aft, "3. UniDAC Fine-Tuned (After)", f"BSI = {bsi_after:.3f} | Rough = {r_after:.4f}", (20, 50, 20))
        p4 = add_banner(p_dif, "4. Refinement Gain Map", "Delta Depth (|After - Before|)")
        p5 = add_banner(p_mid, "5. MiDaS v3.1", f"BSI = {bsi_midas:.3f} | Rough = {r_midas:.4f}", (20, 20, 50))

        composite = np.hstack([p1, p2, p3, p4, p5])
        panel_save = out_dir / f"panel_before_vs_after_{scene_name}.jpg"
        cv2.imwrite(str(panel_save), composite)
        panel_paths.append(panel_save)

    if temp_p.exists():
        temp_p.unlink()

    # -----------------------------------------------------------------------
    # Summary Bar Chart: Before vs. After
    # -----------------------------------------------------------------------
    avg_bsi_bef = np.mean([r["bsi_before"] for r in results])
    avg_bsi_aft = np.mean([r["bsi_after"] for r in results])
    avg_bsi_mid = np.mean([r["bsi_midas"] for r in results])

    avg_r_bef = np.mean([r["r_before"] for r in results])
    avg_r_aft = np.mean([r["r_after"] for r in results])
    avg_r_mid = np.mean([r["r_midas"] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    models = ["MiDaS v3.1\n(Baseline)", "UniDAC Base\n(Before)", "UniDAC Fine-Tuned\n(After - Ours)"]
    x = np.arange(len(models))

    # Boundary Sharpness
    bsi_vals = [avg_bsi_mid, avg_bsi_bef, avg_bsi_aft]
    bars1 = ax1.bar(x, bsi_vals, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="#2c3e50", width=0.48)
    ax1.set_ylabel("Boundary Step Intensity (BSI)")
    ax1.set_title("Boundary Separation Gain (Higher = Sharper)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylim(0, max(bsi_vals) * 1.35)
    for b in bars1:
        h_val = b.get_height()
        ax1.annotate(f"{h_val:.3f}", xy=(b.get_x() + b.get_width()/2, h_val), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # Residual Roughness
    r_vals = [avg_r_mid, avg_r_bef, avg_r_aft]
    bars2 = ax2.bar(x, r_vals, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="#2c3e50", width=0.48)
    ax2.set_ylabel("Residual Roughness in Low-Texture (m⁻¹)")
    ax2.set_title("Planar Surface Flatness (Lower = Smoother)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylim(0, max(r_vals) * 1.35)
    for b in bars2:
        h_val = b.get_height()
        ax2.annotate(f"{h_val:.4f}", xy=(b.get_x() + b.get_width()/2, h_val), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("UniDAC Fine-Tuning Impact: Pre-Trained vs. Camouflage & Texture Adapted Model", y=1.02)
    plt.tight_layout()
    chart_p = out_dir / "chart_before_vs_after_finetuning.png"
    plt.savefig(str(chart_p), dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # Assemble Master Before vs After Showcase
    # -----------------------------------------------------------------------
    target_width = 1600
    loaded = []
    for p_path in panel_paths:
        img = cv2.imread(str(p_path))
        h, w = img.shape[:2]
        scaled_h = int(h * target_width / w)
        loaded.append(cv2.resize(img, (target_width, scaled_h)))

    banner = np.zeros((80, target_width, 3), dtype=np.uint8)
    banner[:] = [15, 20, 28]
    cv2.putText(banner, "Head-to-Head: Pre-Trained UniDAC vs. Fine-Tuned Texture-Adapted Model", 
                (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(banner, "Empirical Evaluation across Marine, Terrestrial, Amphibian Camouflage & Monochromatic Interiors", 
                (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 200), 1, cv2.LINE_AA)

    divider = np.zeros((6, target_width, 3), dtype=np.uint8)
    divider[:] = [30, 80, 50]

    all_blocks = [banner, divider]
    for idx, panel in enumerate(loaded):
        all_blocks.append(panel)
        if idx < len(loaded) - 1:
            all_blocks.append(divider)

    master_showcase = np.vstack(all_blocks)
    showcase_p = out_dir / "master_before_vs_after_showcase.jpg"
    cv2.imwrite(str(showcase_p), master_showcase, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print("\n" + "=" * 80)
    print("  BEFORE VS. AFTER FINE-TUNING SUMMARY TABLE")
    print("=" * 80)
    print(f"  {'Scene Name':<32} | {'Before BSI':<10} | {'After BSI':<10} | {'Gain':<10}")
    print("-" * 80)
    for r in results:
        gain = f"+{((r['bsi_after'] - r['bsi_before'])/(r['bsi_before'] + 1e-5))*100:.1f}%" if r['bsi_after'] >= r['bsi_before'] else f"{((r['bsi_after'] - r['bsi_before'])/(r['bsi_before'] + 1e-5))*100:.1f}%"
        print(f"  {r['scene']:<32} | {r['bsi_before']:>10.3f} | {r['bsi_after']:>10.3f} | {gain:>10}")
    print("-" * 80)
    print(f"  {'AVERAGE':<32} | {avg_bsi_bef:>10.3f} | {avg_bsi_aft:>10.3f} | +{((avg_bsi_aft-avg_bsi_bef)/(avg_bsi_bef+1e-5))*100:.1f}%")
    print("=" * 80 + "\n")
    print(f"Summary Chart Saved -> {chart_p}")
    print(f"Master Showcase Saved -> {showcase_p}\n")


if __name__ == "__main__":
    run_evaluation()
