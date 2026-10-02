"""Shape2.5D dataset loader for Phase-1 evaluation.

Paper: "3D Shape Perception from Monocular Vision, Touch and Shape Priors"
The depth maps are stored as PNG (uint16) or EXR/HDR, one per image.

Expected layout:
  <root>/
    images/     -> *.png / *.jpg
    depth/      -> *.png (uint16, mm) or *.exr (float, metres)
    intrinsics/ -> *.txt or one shared camera.txt with fx fy cx cy per line
    split_test.txt  (one image stem per line, optional)

Returns {id, rgb, gt, K, valid}.
GT is z-depth in metres (uint16 PNG assumed to be millimetres -> metres).
"""
import os, glob
import numpy as np
import cv2
from typing import Iterator, Dict, Any, Optional

# Fallback if per-image intrinsics are not available
SHAPE25D_FX = 600.0
SHAPE25D_FY = 600.0
SHAPE25D_CX = 320.0
SHAPE25D_CY = 240.0
SHAPE25D_MAX_DEPTH = 5.0   # metres


def _read_intrinsics(path: str):
    """Try to read fx fy cx cy from a plain text file (one per line or space-separated)."""
    with open(path) as f:
        nums = [float(x) for x in f.read().split()]
    if len(nums) >= 4:
        return tuple(nums[:4])   # fx fy cx cy
    return None


def _load_depth(path: str) -> Optional[np.ndarray]:
    if not os.path.isfile(path):
        return None
    if path.lower().endswith(".exr"):
        arr = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    else:
        arr = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if arr is None:
        return None
    arr = arr.astype(np.float32)
    # heuristic: if max > 100 assume mm, convert to metres
    if arr.max() > 100:
        arr /= 1000.0
    return arr


def load_shape25d(
    root: str,
    split_file: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield Shape2.5D samples one at a time.

    If no split_file is given, all images in ``<root>/images/`` are used.
    """
    # ---- build list of stems ----
    if split_file and os.path.isfile(split_file):
        with open(split_file) as f:
            stems = [l.strip() for l in f if l.strip()]
    else:
        img_dir = os.path.join(root, "images")
        stems = [
            os.path.splitext(os.path.basename(p))[0]
            for p in sorted(glob.glob(os.path.join(img_dir, "*.*")))
        ]

    # ---- shared intrinsics? ----
    shared_K = None
    for cand in ["camera.txt", "intrinsics.txt", "K.txt"]:
        p = os.path.join(root, cand)
        if os.path.isfile(p):
            shared_K = _read_intrinsics(p)
            break

    count = 0
    for stem in stems:
        if max_samples is not None and count >= max_samples:
            break

        # RGB
        rgb_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            cand = os.path.join(root, "images", stem + ext)
            if os.path.isfile(cand):
                rgb_path = cand
                break
        if rgb_path is None:
            continue

        # Depth
        depth_path = None
        for ext in [".png", ".exr"]:
            cand = os.path.join(root, "depth", stem + ext)
            if os.path.isfile(cand):
                depth_path = cand
                break
        if depth_path is None:
            continue

        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        gt = _load_depth(depth_path)
        if gt is None:
            continue

        # per-image intrinsics?
        K = shared_K
        if K is None:
            for ext in [".txt", ".json"]:
                cand = os.path.join(root, "intrinsics", stem + ext)
                if os.path.isfile(cand):
                    K = _read_intrinsics(cand)
                    break
        if K is None:
            K = (SHAPE25D_FX, SHAPE25D_FY, SHAPE25D_CX, SHAPE25D_CY)

        valid = (gt > 0.01) & (gt < SHAPE25D_MAX_DEPTH)
        yield {
            "id": stem,
            "rgb": rgb,
            "gt": gt,
            "K": K,
            "valid": valid,
        }
        count += 1


def load_shape25d_list(root: str, split_file: Optional[str] = None, max_samples: Optional[int] = None):
    return list(load_shape25d(root, split_file=split_file, max_samples=max_samples))
