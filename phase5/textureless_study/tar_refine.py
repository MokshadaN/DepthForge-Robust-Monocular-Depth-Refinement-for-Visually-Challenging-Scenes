"""Texture-Aware Refinement (TAR) for Monocular Metric Depth.

Implements the Phase 3 pipeline:
  1. Low-texture segmentation (gradient, variance, entropy mask)
  2. Connected components extraction
  3. Back-projection to 3D point cloud per component
  4. RANSAC plane fitting (aX + bY + cZ + d = 0) with confidence scoring
  5. Plane gating (minimum component size, inlier ratio, low residual)
  6. Analytical ray-plane depth projection: Z = -d / (a*(u-cx)/fx + b*(v-cy)/fy + c)
  7. Depth discontinuity & component boundary protection with smooth alpha falloff
"""

import numpy as np
import cv2
from texture_mask import texture_map, low_texture_mask


def fit_plane_ransac(
    pts: np.ndarray,
    n_iter: int = 150,
    dist_thresh: float = 0.03,
    min_inlier_ratio: float = 0.65,
    max_residual: float = 0.05,
    seed: int = 42,
):
    """Fit a plane (aX + bY + cZ + d = 0, with a^2+b^2+c^2=1) via RANSAC.

    Parameters
    ----------
    pts : np.ndarray (N, 3)
        3D points (X, Y, Z).
    n_iter : int
        RANSAC iterations.
    dist_thresh : float
        Point-to-plane distance threshold in meters.
    min_inlier_ratio : float
        Threshold on fraction of points supporting the plane.
    max_residual : float
        Max allowed mean residual of inliers.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    plane : tuple (a, b, c, d) or None
        Normalized plane equation coefficients such that aX + bY + cZ + d = 0.
    inlier_mask : np.ndarray (N,) bool
        Boolean mask of inlier points.
    confidence : dict
        Details {'inlier_ratio': float, 'mean_residual': float, 'is_valid': bool}
    """
    N = len(pts)
    if N < 30:
        return None, np.zeros(N, dtype=bool), {"inlier_ratio": 0.0, "mean_residual": 999.0, "is_valid": False}

    rng = np.random.default_rng(seed)
    best_plane = None
    best_inliers = np.zeros(N, dtype=bool)
    best_score = -1

    for _ in range(n_iter):
        idx = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = pts[idx[0]], pts[idx[1]], pts[idx[2]]

        # Normal vector n = (p2 - p1) x (p3 - p1)
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-7:
            continue
        n = n / norm_n
        d = -float(np.dot(n, p1))

        # Distances from all points: |n . p + d|
        dists = np.abs(np.dot(pts, n) + d)
        inliers = dists < dist_thresh
        score = int(inliers.sum())

        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_plane = (float(n[0]), float(n[1]), float(n[2]), float(d))

    if best_plane is None or best_score < 3:
        return None, np.zeros(N, dtype=bool), {"inlier_ratio": 0.0, "mean_residual": 999.0, "is_valid": False}

    # Refit plane on all inliers via PCA / SVD for higher accuracy
    inlier_pts = pts[best_inliers]
    if len(inlier_pts) >= 3:
        centroid = inlier_pts.mean(axis=0)
        uu, dd, vv = np.linalg.svd(inlier_pts - centroid)
        n = vv[2]  # normal is eigenvector with smallest singular value
        n = n / np.linalg.norm(n)
        d = -float(np.dot(n, centroid))
        # Recompute inliers and residual with refitted plane
        dists = np.abs(np.dot(pts, n) + d)
        best_inliers = dists < dist_thresh
        best_plane = (float(n[0]), float(n[1]), float(n[2]), float(d))

    inlier_ratio = float(best_inliers.sum()) / float(N)
    mean_res = float(np.mean(np.abs(np.dot(pts[best_inliers], best_plane[:3]) + best_plane[3]))) if best_inliers.sum() > 0 else 999.0
    is_valid = (inlier_ratio >= min_inlier_ratio) and (mean_res <= max_residual)

    conf = {"inlier_ratio": inlier_ratio, "mean_residual": mean_res, "is_valid": is_valid}
    return best_plane, best_inliers, conf


def compute_boundary_weight_map(
    mask: np.ndarray,
    depth: np.ndarray,
    band_pixels: int = 5,
    depth_edge_rel_thresh: float = 0.08,
) -> np.ndarray:
    """Compute smooth blending weights [0, 1] that drop to 0 near mask & depth boundaries.

    Parameters
    ----------
    mask : np.ndarray (H, W) bool
        Connected component or active refinement mask.
    depth : np.ndarray (H, W) float32
        Input depth map.
    band_pixels : int
        Width in pixels over which refinement weight fades to zero.
    depth_edge_rel_thresh : float
        Relative jump in depth to consider a depth boundary discontinuity.

    Returns
    -------
    weights : np.ndarray (H, W) float32 in [0, 1]
    """
    if not mask.any():
        return np.zeros_like(depth, dtype=np.float32)

    # 1. Component boundary distance transform
    mask_u8 = mask.astype(np.uint8)
    dist_mask = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)

    # 2. Depth discontinuity detector
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_d = np.sqrt(gx**2 + gy**2)
    rel_grad = grad_d / np.maximum(depth, 1e-2)
    depth_edges = (rel_grad > depth_edge_rel_thresh).astype(np.uint8)

    # Distance to depth discontinuities
    inv_depth_edges = (1 - depth_edges).astype(np.uint8)
    dist_depth_edge = cv2.distanceTransform(inv_depth_edges, cv2.DIST_L2, 3)

    effective_dist = np.minimum(dist_mask, dist_depth_edge)
    weights = np.clip(effective_dist / max(band_pixels, 1), 0.0, 1.0).astype(np.float32)
    # Cosine smooth falloff: 0.5 * (1 - cos(pi * w))
    weights = 0.5 * (1.0 - np.cos(np.pi * weights))
    return weights * mask.astype(np.float32)


def refine_tar(
    depth: np.ndarray,
    rgb: np.ndarray,
    K: tuple,
    tau: float,
    min_comp_pixels: int = 400,
    min_inlier_ratio: float = 0.70,
    max_residual: float = 0.04,
    boundary_band: int = 6,
) -> tuple:
    """Texture-Aware Refinement (TAR) pipeline on a single image.

    Parameters
    ----------
    depth : np.ndarray (H, W) float32
        Predicted metric z-depth (m).
    rgb : np.ndarray (H, W, 3) uint8
        RGB image.
    K : tuple (fx, fy, cx, cy)
        Camera intrinsics.
    tau : float
        Threshold for low-texture mask.
    min_comp_pixels : int
        Minimum size of connected low-texture region to attempt plane fitting.
    min_inlier_ratio : float
        RANSAC inlier ratio requirement.
    max_residual : float
        RANSAC inlier mean residual threshold (m).
    boundary_band : int
        Pixel radius of boundary protection transition zone.

    Returns
    -------
    refined_depth : np.ndarray (H, W) float32
        Refined metric z-depth map.
    debug_info : dict
        Dictionary with intermediate masks, fitted plane count, etc.
    """
    H, W = depth.shape
    fx, fy, cx, cy = K

    # Step 1: Texture Map & Low-Texture Mask
    T = texture_map(rgb)
    lt_mask = low_texture_mask(T, tau) & (depth > 1e-3) & np.isfinite(depth)

    # Step 2: Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        lt_mask.astype(np.uint8), connectivity=8
    )

    refined = depth.copy()
    refined_mask = np.zeros((H, W), dtype=bool)
    fitted_planes = []

    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    ray_x = (u_grid - cx) / fx
    ray_y = (v_grid - cy) / fy

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_comp_pixels:
            continue

        comp_mask = (labels == label)
        # Back-project 3D points
        z_comp = depth[comp_mask]
        rx_comp = ray_x[comp_mask]
        ry_comp = ray_y[comp_mask]

        X = rx_comp * z_comp
        Y = ry_comp * z_comp
        Z = z_comp
        pts = np.stack([X, Y, Z], axis=-1)

        plane, inliers, conf = fit_plane_ransac(
            pts,
            n_iter=200,
            dist_thresh=0.03,
            min_inlier_ratio=min_inlier_ratio,
            max_residual=max_residual,
        )

        if not conf["is_valid"] or plane is None:
            continue

        a, b, c, d = plane
        denom = a * ray_x + b * ray_y + c
        valid_denom = np.abs(denom) > 1e-4
        z_plane = np.zeros_like(depth)
        z_plane[valid_denom] = -d / denom[valid_denom]

        valid_fit = comp_mask & valid_denom & (z_plane > 0.1) & (z_plane < 150.0)
        if not valid_fit.any():
            continue

        # Step 7: Boundary protection & blending
        weights = compute_boundary_weight_map(
            valid_fit, depth, band_pixels=boundary_band
        )

        refined = (1.0 - weights) * refined + weights * z_plane
        refined_mask = refined_mask | (weights > 0.05)
        fitted_planes.append({
            "label": int(label),
            "area": int(area),
            "plane": (float(a), float(b), float(c), float(d)),
            "conf": conf,
        })

    debug_info = {
        "texture_map": T,
        "low_texture_mask": lt_mask,
        "refined_mask": refined_mask,
        "planes": fitted_planes,
    }
    return refined.astype(np.float32), debug_info
