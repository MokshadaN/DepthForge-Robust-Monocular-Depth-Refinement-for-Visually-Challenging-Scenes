"""
Phase 5 Showcase: Visualizing Textureless & Camouflaged Test Cases.

Generates explicit visual breakdowns showing:
1. The raw test images (white-on-white surfaces, smooth planar walls, untextured asphalt).
2. The Texture Density / Gradient Heatmap (highlighting 0-texture zones in blue/black).
3. The Object vs. Background color similarity map.
4. The UniDAC Depth Map showing how depth is resolved without texture cues.
"""

from __future__ import annotations

import sys
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def compute_texture_gradient_map(img_bgr: np.ndarray) -> np.ndarray:
    """Compute local standard deviation gradient (0 texture = dark blue, high texture = bright yellow)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (9, 9))
    mean_sq = cv2.blur(gray**2, (9, 9))
    var = np.maximum(mean_sq - mean**2, 0.0)
    std = np.sqrt(var)
    norm = np.clip(std / 35.0, 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


def colorize_depth_magma(depth: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
    if vmin is None:
        vmin = float(np.percentile(valid, 2))
    if vmax is None:
        vmax = float(np.percentile(valid, 98))
    vmax = max(vmax, vmin + 1e-4)
    norm = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


def add_header(img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    h, w = img.shape[:2]
    banner_h = 46 if subtitle else 32
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = [22, 22, 22]
    cv2.putText(banner, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(banner, subtitle, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    return np.vstack([banner, img])


def build_showcase():
    out_dir = Path("phase5/results").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Case 1: ScanNet++ White Refrigerator on White Wall
    # -----------------------------------------------------------------------
    img1 = cv2.imread("demo/input/scannetpp_rgb.jpg")
    depth1 = np.load("phase3/output_scannetpp/depth.npy")
    tex1 = compute_texture_gradient_map(img1)
    dep_col1 = colorize_depth_magma(depth1, 0.5, 4.0)

    # Highlight white fridge on white wall
    annotated1 = img1.copy()
    cv2.rectangle(annotated1, (15, 170), (280, 420), (0, 255, 255), 2)
    cv2.putText(annotated1, "White Fridge (Same as Wall)", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    tw, th = 380, int(img1.shape[0] * 380 / img1.shape[1])
    p1_a = add_header(cv2.resize(annotated1, (tw, th)), "Case 1: RGB Image", "White Refrigerator on White Wall")
    p1_b = add_header(cv2.resize(tex1, (tw, th)), "Texture Density Map", "Blue = 0 Texture (Flat Intensity)")
    p1_c = add_header(cv2.resize(dep_col1, (tw, th)), "UniDAC Depth", "Metric Depth Separated (2.59m)")
    row1 = np.hstack([p1_a, p1_b, p1_c])

    # -----------------------------------------------------------------------
    # Case 2: Matterport3D Smooth Planar Walls & Uniform Fabric
    # -----------------------------------------------------------------------
    img2 = cv2.imread("demo/input/matterport3d_rgb.png")
    depth2 = np.load("phase3/output_matterport3d/depth.npy")
    tex2 = compute_texture_gradient_map(img2)
    dep_col2 = colorize_depth_magma(depth2, 1.0, 7.0)

    annotated2 = img2.copy()
    cv2.rectangle(annotated2, (380, 280), (620, 380), (0, 255, 255), 2)
    cv2.putText(annotated2, "Uniform Couch Fabric", (385, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    tw2, th2 = 380, int(img2.shape[0] * 380 / img2.shape[1])
    p2_a = add_header(cv2.resize(annotated2, (tw2, th2)), "Case 2: 360 ERP RGB", "Textureless Planar Walls & Sofa")
    p2_b = add_header(cv2.resize(tex2, (tw2, th2)), "Texture Density Map", "Dark = 0 Local Variance")
    p2_c = add_header(cv2.resize(dep_col2, (tw2, th2)), "UniDAC Spherical Depth", "Spherical Geometry Resolves Planes")
    row2 = np.hstack([p2_a, p2_b, p2_c])

    # -----------------------------------------------------------------------
    # Case 3: KITTI-360 Textureless Asphalt Road Plane
    # -----------------------------------------------------------------------
    img3 = cv2.imread("demo/input/kitti360_rgb.png")
    depth3 = np.load("phase3/output_kitti360/depth.npy")
    tex3 = compute_texture_gradient_map(img3)
    dep_col3 = colorize_depth_magma(depth3, 1.0, 35.0)

    annotated3 = img3.copy()
    cv2.rectangle(annotated3, (200, 800), (1200, 1300), (0, 255, 255), 2)
    cv2.putText(annotated3, "Textureless Road Plane (Uniform Asphalt)", (220, 780), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    tw3, th3 = 380, int(img3.shape[0] * 380 / img3.shape[1])
    p3_a = add_header(cv2.resize(annotated3, (tw3, th3)), "Case 3: Outdoor Fisheye", "Uniform Textureless Asphalt")
    p3_b = add_header(cv2.resize(tex3, (tw3, th3)), "Texture Density Map", "Massive 0-Texture Zone on Ground")
    p3_c = add_header(cv2.resize(dep_col3, (tw3, th3)), "UniDAC Metric Depth", "Continuous Ground Plane Maintained")
    row3 = np.hstack([p3_a, p3_b, p3_c])

    # Combine into full gallery showcase
    gallery = np.vstack([row1, row2, row3])
    showcase_path = out_dir / "all_textureless_cases_showcase.jpg"
    cv2.imwrite(str(showcase_path), gallery)
    print(f"Saved All Textureless Cases Showcase -> {showcase_path}")


if __name__ == "__main__":
    build_showcase()
