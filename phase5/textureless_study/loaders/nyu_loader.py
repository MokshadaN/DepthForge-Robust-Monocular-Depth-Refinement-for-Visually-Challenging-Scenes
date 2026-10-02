"""NYU Depth V2 loader for Phase-1 evaluation.

Expected dataset layout (standard MAT or RGB+depth-PNG layout):
  <root>/
    rgb/        (or images/)    -> *.png / *.jpg, 640x480
    depth/      (or annotations/) -> *.png, uint16, depth = pixel / 1000.0 metres
    splits/     nyu_test_split.txt  (one relative rgb path per line)

Returns samples: {id, rgb: HxWx3 u8, gt: HxW float32 z-depth [m], K: (fx,fy,cx,cy), valid: bool HxW}

NYU canonical intrinsics (used when K is not per-image):
  fx = 518.8579, fy = 519.4696, cx = 325.5824, cy = 253.7362
  Image size: 640 x 480
"""
import os
import numpy as np
import cv2
from typing import Iterator, Dict, Any, Optional

# Standard NYU intrinsics (focal length in pixels, principal point)
NYU_FX = 518.8579
NYU_FY = 519.4696
NYU_CX = 325.5824
NYU_CY = 253.7362
NYU_K = (NYU_FX, NYU_FY, NYU_CX, NYU_CY)

NYU_MAX_DEPTH = 10.0   # metres – GT beyond this is invalid
NYU_DEPTH_SCALE = 1000.0  # raw uint16 -> metres


def _load_depth_png(path: str) -> np.ndarray:
    """Load a 16-bit depth PNG and convert to float32 metres."""
    raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if raw is None:
        raise FileNotFoundError(f"Cannot read depth: {path}")
    return raw.astype(np.float32) / NYU_DEPTH_SCALE


def load_nyu(
    root: str,
    split_file: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield NYU Depth V2 samples one at a time.

    Parameters
    ----------
    root : str
        Dataset root, e.g. ``/data/nyu_depth_v2``.
    split_file : str, optional
        Path to a text file with one RGB filename per line (relative to root).
        Defaults to ``<root>/splits/nyu_test_split.txt``.
    max_samples : int, optional
        If set, stop after this many samples.
    """
    if split_file is None:
        split_file = os.path.join(root, "splits", "nyu_test_split.txt")
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file) as f:
        rgb_rel_paths = [l.strip() for l in f if l.strip()]

    count = 0
    for rel in rgb_rel_paths:
        if max_samples is not None and count >= max_samples:
            break

        rgb_path = os.path.join(root, rel)
        # Infer depth path: mirror rgb/ -> depth/, same stem, .png
        depth_rel = rel.replace("rgb/", "depth/").replace("images/", "depth/")
        stem = os.path.splitext(depth_rel)[0]
        depth_path = stem + ".png"
        depth_full = os.path.join(root, depth_path)

        if not os.path.isfile(rgb_path):
            continue
        if not os.path.isfile(depth_full):
            continue

        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)  # HxWx3 uint8
        gt = _load_depth_png(depth_full)  # HxW float32 metres

        valid = (gt > 0.01) & (gt < NYU_MAX_DEPTH)

        sample_id = os.path.splitext(os.path.basename(rel))[0]
        yield {
            "id": sample_id,
            "rgb": rgb,
            "gt": gt,
            "K": NYU_K,
            "valid": valid,
        }
        count += 1


def load_nyu_list(root: str, split_file: Optional[str] = None, max_samples: Optional[int] = None):
    """Convenience wrapper — returns a list instead of a generator."""
    return list(load_nyu(root, split_file=split_file, max_samples=max_samples))
