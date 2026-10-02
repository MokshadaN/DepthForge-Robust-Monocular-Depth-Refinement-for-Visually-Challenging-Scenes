"""
Phase 5: Comprehensive Benchmark on Authentic Camouflage & Textureless Scenes.
Evaluates UniDAC vs. MiDaS v3.1 vs. Depth baselines and TAR/Photometric Refinements.
"""

from __future__ import annotations

import sys
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase1.pipeline import load_sample_config, UniDACDepthRunner
from phase4.compare_all_models import MiDaSModelBackend


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


# ---------------------------------------------------------------------------
# Texture Map & Boundary Metrics
# ---------------------------------------------------------------------------

def compute_texture_energy(image_bgr: np.ndarray) -> np.ndarray:
    """Compute combined gradient + variance texture energy."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Sobel gradient
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Local variance
    mean = cv2.blur(gray, (7, 7))
    mean_sq = cv2.blur(gray**2, (7, 7))
    variance = np.maximum(mean_sq - mean**2, 0.0)
    std_dev = np.sqrt(variance)
    
    energy = 0.5 * (grad_mag / (grad_mag.max() + 1e-5)) + 0.5 * (std_dev / (std_dev.max() + 1e-5))
    return energy


def enhance_camo_input(image_bgr: np.ndarray) -> np.ndarray:
    """Multi-scale CLAHE + Guided Bilateral Edge Injection."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)
    l_enh = cv2.bilateralFilter(l_enh, d=5, sigmaColor=35, sigmaSpace=35)
    lab_enh = cv2.merge([l_enh, a, b])
    return cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)


def compute_boundary_sharpness(depth_map: np.ndarray) -> float:
    """Calculate mean Sobel step gradient across the depth map."""
    depth_clean = np.nan_to_num(depth_map, nan=0.0)
    gx = cv2.Sobel(depth_clean, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_clean, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    # 90th percentile gradient represents boundary sharpness
    return float(np.percentile(grad, 90))


def compute_surface_smoothness(depth_map: np.ndarray, low_texture_mask: np.ndarray) -> float:
    """Calculate residual roughness in low-texture regions (lower = cleaner)."""
    if np.count_nonzero(low_texture_mask) < 50:
        return 0.0
    depth_clean = np.nan_to_num(depth_map, nan=0.0)
    lap = np.abs(cv2.Laplacian(depth_clean, cv2.CV_32F))
    return float(np.mean(lap[low_texture_mask]))


# ---------------------------------------------------------------------------
# Benchmark Execution
# ---------------------------------------------------------------------------

def run_camo_benchmark():
    setup_style()
    data_dir = Path("phase5/data/camo_scenes").resolve()
    out_dir = Path("phase5/results").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("  PHASE 5: COMPREHENSIVE TEXTURELESS & AUTHENTIC CAMOUFLAGE BENCHMARK")
    print("=" * 88)

    scenes = sorted(list(data_dir.glob("*.jpg")))
    print(f"Found {len(scenes)} authentic test scenes in {data_dir}\n")

    # Load Models
    print("Initializing UniDAC Runner...")
    sample_cfg = load_sample_config(Path("demo/input/scannetpp_sample.json").resolve())
    unidac_runner = UniDACDepthRunner(
        checkpoint=Path("checkpoints/unidac.pt").resolve(),
        sample=sample_cfg
    )

    print("Initializing MiDaS v3.1 DPT_Hybrid Baseline...")
    midas_backend = MiDaSModelBackend("DPT_Hybrid", device="auto")

    results_table = []
    panel_paths = []

    for idx, scene_path in enumerate(scenes, 1):
        scene_name = scene_path.stem
        img_bgr = cv2.imread(str(scene_path))
        h, w = img_bgr.shape[:2]

        print(f"[{idx}/{len(scenes)}] Evaluating scene: {scene_name} ({w}x{h})...")

        # 1. Texture Map & Homogeneity Mask
        energy = compute_texture_energy(img_bgr)
        low_tex_mask = energy < 0.25

        # 2. UniDAC Standard
        temp_p = out_dir / f"temp_{scene_name}.jpg"
        cv2.imwrite(str(temp_p), img_bgr)
        unidac_depth = unidac_runner.predict(temp_p, output_size=(h, w))

        # 3. UniDAC Enhanced (Proposed CLAHE + Bilateral Injection)
        img_enh = enhance_camo_input(img_bgr)
        cv2.imwrite(str(temp_p), img_enh)
        unidac_enh_depth = unidac_runner.predict(temp_p, output_size=(h, w))
        temp_p.unlink(missing_ok=True)

        # 4. MiDaS v3.1 Baseline
        midas_raw = midas_backend.predict(img_bgr, {}, output_size=(h, w))
        # Scale align MiDaS to median of UniDAC for fair visual & metric comparison
        scale = float(np.median(unidac_depth) / (np.median(midas_raw) + 1e-6))
        midas_depth = midas_raw * scale

        # Metrics
        bsi_mid = compute_boundary_sharpness(midas_depth)
        bsi_uni = compute_boundary_sharpness(unidac_depth)
        bsi_enh = compute_boundary_sharpness(unidac_enh_depth)

        rough_mid = compute_surface_smoothness(midas_depth, low_tex_mask)
        rough_uni = compute_surface_smoothness(unidac_depth, low_tex_mask)
        rough_enh = compute_surface_smoothness(unidac_enh_depth, low_tex_mask)

        results_table.append({
            "scene": scene_name,
            "bsi_midas": bsi_mid, "bsi_unidac": bsi_uni, "bsi_enhanced": bsi_enh,
            "rough_midas": rough_mid, "rough_unidac": rough_uni, "rough_enhanced": rough_enh
        })

        # Generate Tangible Visual Panel
        def to_magma(d_map):
            d_min, d_max = np.percentile(d_map, 2), np.percentile(d_map, 98)
            if d_max - d_min < 1e-4:
                d_max = d_min + 1.0
            norm = np.clip((d_map - d_min) / (d_max - d_min), 0, 1)
            return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

        def to_inferno(t_map):
            norm = np.clip(t_map, 0, 1)
            return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

        tw, th = 320, int(h * 320 / w)
        p_rgb = cv2.resize(img_bgr, (tw, th))
        p_tex = cv2.resize(to_inferno(energy), (tw, th))
        p_mid = cv2.resize(to_magma(midas_depth), (tw, th))
        p_uni = cv2.resize(to_magma(unidac_depth), (tw, th))
        p_enh = cv2.resize(to_magma(unidac_enh_depth), (tw, th))

        def add_banner(img, title, subtitle):
            banner = np.zeros((48, img.shape[1], 3), dtype=np.uint8)
            banner[:] = [24, 24, 24]
            cv2.putText(banner, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(banner, subtitle, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
            return np.vstack([banner, img])

        p1 = add_banner(p_rgb, "1. Input RGB", scene_name.replace("_", " ").title())
        p2 = add_banner(p_tex, "2. Texture Energy Map", "Low Gradient Homogeneity")
        p3 = add_banner(p_mid, "3. MiDaS v3.1", f"BSI = {bsi_mid:.3f} | Rough = {rough_mid:.3f}")
        p4 = add_banner(p_uni, "4. UniDAC Standard", f"BSI = {bsi_uni:.3f} | Rough = {rough_uni:.3f}")
        p5 = add_banner(p_enh, "5. UniDAC Enhanced (Ours)", f"BSI = {bsi_enh:.3f} | Rough = {rough_enh:.3f}")

        composite = np.hstack([p1, p2, p3, p4, p5])
        panel_save = out_dir / f"panel_{scene_name}.jpg"
        cv2.imwrite(str(panel_save), composite)
        panel_paths.append(panel_save)

    # -----------------------------------------------------------------------
    # Summary Metrics Chart
    # -----------------------------------------------------------------------
    avg_bsi_mid = np.mean([r["bsi_midas"] for r in results_table])
    avg_bsi_uni = np.mean([r["bsi_unidac"] for r in results_table])
    avg_bsi_enh = np.mean([r["bsi_enhanced"] for r in results_table])

    avg_r_mid = np.mean([r["rough_midas"] for r in results_table])
    avg_r_uni = np.mean([r["rough_unidac"] for r in results_table])
    avg_r_enh = np.mean([r["rough_enhanced"] for r in results_table])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    models = ["MiDaS v3.1\n(Traditional)", "UniDAC\n(Standard)", "UniDAC Enhanced\n(Proposed)"]
    x = np.arange(len(models))

    # Boundary Sharpness
    bsi_vals = [avg_bsi_mid, avg_bsi_uni, avg_bsi_enh]
    bars1 = ax1.bar(x, bsi_vals, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="#2c3e50", width=0.48)
    ax1.set_ylabel("Boundary Step Intensity (BSI)")
    ax1.set_title("Camouflage Boundary Separation (Higher = Sharper)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylim(0, max(bsi_vals) * 1.3)
    for b in bars1:
        h_val = b.get_height()
        ax1.annotate(f"{h_val:.3f}", xy=(b.get_x() + b.get_width()/2, h_val), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # Surface Roughness / Artifacts
    r_vals = [avg_r_mid, avg_r_uni, avg_r_enh]
    bars2 = ax2.bar(x, r_vals, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="#2c3e50", width=0.48)
    ax2.set_ylabel("Residual Roughness in Low-Texture (m⁻¹)")
    ax2.set_title("Textureless Planar Smoothness (Lower = Cleaner)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylim(0, max(r_vals) * 1.3)
    for b in bars2:
        h_val = b.get_height()
        ax2.annotate(f"{h_val:.4f}", xy=(b.get_x() + b.get_width()/2, h_val), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Authentic Camouflage & Textureless Benchmark: UniDAC vs. Baselines & Enhancements", y=1.02)
    plt.tight_layout()
    chart_p = out_dir / "chart_camo_benchmark_summary.png"
    plt.savefig(str(chart_p), dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 80)
    print("  AUTHENTIC CAMOUFLAGE & TEXTURELESS BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  {'Scene Name':<32} | {'MiDaS BSI':<10} | {'UniDAC BSI':<10} | {'Ours BSI':<10}")
    print("-" * 80)
    for r in results_table:
        print(f"  {r['scene']:<32} | {r['bsi_midas']:>10.3f} | {r['bsi_unidac']:>10.3f} | {r['bsi_enhanced']:>10.3f}")
    print("-" * 80)
    print(f"  {'AVERAGE':<32} | {avg_bsi_mid:>10.3f} | {avg_bsi_uni:>10.3f} | {avg_bsi_enh:>10.3f}  <-- +{((avg_bsi_enh-avg_bsi_mid)/avg_bsi_mid)*100:.1f}% vs MiDaS")
    print("=" * 80 + "\n")
    print(f"Summary Chart Saved -> {chart_p}")
    print(f"Visual Panels Generated -> {len(panel_paths)} panels in {out_dir}\n")


if __name__ == "__main__":
    run_camo_benchmark()
