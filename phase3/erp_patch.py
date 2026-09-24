"""
Phase 3 Geometry Module: ERP <-> Perspective Tangent Patch Extractor & Reprojector.

Allows running standard 2D object detectors (YOLO) on undistorted perspective crops
extracted from 360° Equirectangular Panoramic (ERP) images, and reprojecting
the resulting bounding boxes / segmentation masks back into ERP coordinate space.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


def perspective_from_erp(
    erp_img: np.ndarray,
    fov_deg: float = 90.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    out_hw: Tuple[int, int] = (512, 512),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a rectilinear perspective patch from a 360° equirectangular image.

    Parameters
    ----------
    erp_img : np.ndarray (H, W, C) or (H, W)
    fov_deg : horizontal field of view in degrees (e.g. 90.0)
    yaw_deg : horizontal rotation angle in degrees [-180, 180]
    pitch_deg : vertical tilt angle in degrees [-90, 90]
    roll_deg : roll angle in degrees
    out_hw : (height, width) of output perspective image

    Returns
    -------
    persp_img : np.ndarray (out_h, out_w, C)
    map_x : np.ndarray (out_h, out_w) float32 ERP x coordinates
    map_y : np.ndarray (out_h, out_w) float32 ERP y coordinates
    """
    out_h, out_w = out_hw
    erp_h, erp_w = erp_img.shape[:2]

    # Focal length for given FOV
    f = 0.5 * out_w / np.tan(0.5 * np.radians(fov_deg))
    cx, cy = (out_w - 1) / 2.0, (out_h - 1) / 2.0

    # Grid of pixel coordinates in perspective camera
    u, v = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
    x_c = (u - cx) / f
    y_c = (v - cy) / f
    z_c = np.ones_like(x_c)

    # Normalize camera ray vectors
    norm = np.sqrt(x_c**2 + y_c**2 + z_c**2)
    rays_c = np.stack([x_c / norm, y_c / norm, z_c / norm], axis=-1)  # (H, W, 3)

    # Rotation matrix (Yaw around Y, Pitch around X, Roll around Z)
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    # R = R_yaw @ R_pitch @ R_roll
    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ], dtype=np.float32)

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ], dtype=np.float32)

    R_roll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    R = R_yaw @ R_pitch @ R_roll

    # Transform rays to world ERP frame
    rays_w = rays_c @ R.T  # (H, W, 3) where [X, Y, Z]

    # Convert 3D direction rays to spherical (lat, lon)
    # Longitude: -pi to +pi (mapped to 0..erp_w)
    # Latitude: -pi/2 to +pi/2 (mapped to 0..erp_h)
    lon = np.arctan2(rays_w[..., 0], rays_w[..., 2])   # X, Z
    lat = np.arcsin(np.clip(-rays_w[..., 1], -1.0, 1.0)) # -Y (image Y is down)

    # Map lon/lat to ERP pixel coordinates
    map_x = ((lon / (2 * np.pi) + 0.5) * erp_w).astype(np.float32) % erp_w
    map_y = (((0.5 - lat / np.pi)) * erp_h).astype(np.float32)
    map_y = np.clip(map_y, 0, erp_h - 1)

    persp_img = cv2.remap(
        erp_img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP
    )
    return persp_img, map_x, map_y


def extract_cubemap_patches(
    erp_img: np.ndarray,
    fov_deg: float = 90.0,
    yaws: Optional[List[float]] = None,
    out_hw: Tuple[int, int] = (512, 512),
) -> List[Dict]:
    """
    Extract multiple overlapping perspective tangent patches around the 360° horizon.
    Default: 4 orthogonal views (yaw = 0°, 90°, 180°, 270°) with 90° FOV covering 360°.
    """
    if yaws is None:
        yaws = [0.0, 90.0, 180.0, 270.0]

    patches = []
    for yaw in yaws:
        patch_img, map_x, map_y = perspective_from_erp(
            erp_img, fov_deg=fov_deg, yaw_deg=yaw, pitch_deg=0.0, out_hw=out_hw
        )
        patches.append({
            "yaw": yaw,
            "fov": fov_deg,
            "image": patch_img,
            "map_x": map_x,
            "map_y": map_y,
        })
    return patches


def reproject_patch_mask_to_erp(
    patch_mask: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    erp_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Reproject a binary mask (e.g. YOLO instance mask) from perspective patch to full ERP.

    Parameters
    ----------
    patch_mask : (H_p, W_p) uint8 / bool
    map_x, map_y : (H_p, W_p) mapping grids returned from perspective_from_erp
    erp_shape : (H_erp, W_erp)

    Returns
    -------
    erp_mask : (H_erp, W_erp) uint8 binary mask (0 or 255)
    """
    erp_h, erp_w = erp_shape[:2]
    erp_mask = np.zeros((erp_h, erp_w), dtype=np.uint8)

    # Coordinates where patch mask is active
    active_y, active_x = np.where(patch_mask > 0)
    if len(active_x) == 0:
        return erp_mask

    target_xs = np.clip(np.round(map_x[active_y, active_x]).astype(int), 0, erp_w - 1)
    target_ys = np.clip(np.round(map_y[active_y, active_x]).astype(int), 0, erp_h - 1)

    erp_mask[target_ys, target_xs] = 255

    # Fill small interpolation holes via morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erp_mask = cv2.morphologyEx(erp_mask, cv2.MORPH_CLOSE, kernel)
    return erp_mask


def reproject_bbox_to_erp(
    bbox: Tuple[int, int, int, int],
    map_x: np.ndarray,
    map_y: np.ndarray,
    erp_shape: Tuple[int, int],
) -> Tuple[int, int, int, int, np.ndarray]:
    """
    Reproject a perspective bounding box into ERP space, returning an ERP bounding box
    and the exact reprojected mask.
    """
    x1, y1, x2, y2 = bbox
    patch_h, patch_w = map_x.shape[:2]
    box_mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    box_mask[max(0, y1):min(patch_h, y2), max(0, x1):min(patch_w, x2)] = 255

    erp_mask = reproject_patch_mask_to_erp(box_mask, map_x, map_y, erp_shape)
    ys, xs = np.where(erp_mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0, erp_mask)

    erp_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return (*erp_bbox, erp_mask)
