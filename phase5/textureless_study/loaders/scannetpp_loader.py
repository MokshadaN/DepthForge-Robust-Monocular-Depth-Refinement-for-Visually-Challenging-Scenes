"""ScanNet++ loader for Phase-1 evaluation (perspective RGB frames only).

Expected layout (standard ScanNet++ structure):
  <root>/
    data/
      <scene_id>/
        dslr/
          resized_images/   -> *.JPG
          resized_depth/    -> *.png  (uint16, depth in mm)
          grid_fisheye.npy  (lookup table for fisheye undistortion, optional)
          nerfstudio_transforms.json  (contains camera intrinsics per-frame)
    splits/
      val_frames.txt   (one path relative to data/ per line, e.g. scene_id/dslr/frame_000.JPG)

Returns {id, rgb, gt, K, valid}.
GT is z-depth in metres. For fisheye frames the grid table converts euclid->z properly.
"""
import os, json, glob
import numpy as np
import cv2
from typing import Iterator, Dict, Any, Optional

SCANNETPP_MAX_DEPTH = 5.0   # metres
SCANNETPP_DEPTH_SCALE = 1000.0  # mm -> m


def _load_scene_intrinsics(scene_root: str) -> Optional[tuple]:
    """Read (fx, fy, cx, cy) from nerfstudio_transforms.json if present."""
    tf_path = os.path.join(scene_root, "nerfstudio_transforms.json")
    if not os.path.isfile(tf_path):
        return None
    with open(tf_path) as f:
        meta = json.load(f)
    # nerfstudio stores fl_x, fl_y, cx, cy at top level
    try:
        return (float(meta["fl_x"]), float(meta["fl_y"]),
                float(meta["cx"]), float(meta["cy"]))
    except KeyError:
        return None


def load_scannetpp(
    root: str,
    split_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    require_depth: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield ScanNet++ DSLR samples one at a time.

    Parameters
    ----------
    root : str
        Dataset root (contains ``data/`` sub-directory).
    split_file : str, optional
        Text file with lines like ``scene_id/dslr/frame_000.JPG`` (relative to ``data/``).
        If omitted, all scenes' resized_images/*.JPG are enumerated.
    max_samples : int, optional
        Stop after this many samples.
    require_depth : bool
        If True (default), skip frames with no matching depth file.
    """
    data_dir = os.path.join(root, "data")

    # ---- build (scene_id, frame_stem) list ----
    entries = []  # list of (scene_id, rgb_abs_path, depth_abs_path_or_None)
    if split_file and os.path.isfile(split_file):
        with open(split_file) as f:
            rel_paths = [l.strip() for l in f if l.strip()]
        for rel in rel_paths:
            abs_rgb = os.path.join(data_dir, rel)
            scene_id = rel.split("/")[0]
            stem = os.path.splitext(os.path.basename(rel))[0]
            dep = os.path.join(data_dir, scene_id, "dslr", "resized_depth", stem + ".png")
            entries.append((scene_id, abs_rgb, dep if os.path.isfile(dep) else None))
    else:
        for scene_id in sorted(os.listdir(data_dir)):
            img_dir = os.path.join(data_dir, scene_id, "dslr", "resized_images")
            if not os.path.isdir(img_dir):
                continue
            for rgb_path in sorted(glob.glob(os.path.join(img_dir, "*.*"))):
                stem = os.path.splitext(os.path.basename(rgb_path))[0]
                dep = os.path.join(data_dir, scene_id, "dslr", "resized_depth", stem + ".png")
                entries.append((scene_id, rgb_path, dep if os.path.isfile(dep) else None))

    # ---- yield ----
    count = 0
    _K_cache: Dict[str, Optional[tuple]] = {}
    for scene_id, rgb_path, dep_path in entries:
        if max_samples is not None and count >= max_samples:
            break
        if require_depth and dep_path is None:
            continue
        if not os.path.isfile(rgb_path):
            continue

        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

        gt = np.zeros(rgb.shape[:2], dtype=np.float32)
        if dep_path and os.path.isfile(dep_path):
            raw = cv2.imread(dep_path, cv2.IMREAD_ANYDEPTH)
            if raw is not None:
                gt = raw.astype(np.float32) / SCANNETPP_DEPTH_SCALE

        # intrinsics
        if scene_id not in _K_cache:
            scene_root = os.path.join(data_dir, scene_id, "dslr")
            _K_cache[scene_id] = _load_scene_intrinsics(scene_root)
        K = _K_cache[scene_id] or (600.0, 600.0, rgb.shape[1] / 2, rgb.shape[0] / 2)

        valid = (gt > 0.01) & (gt < SCANNETPP_MAX_DEPTH)
        sample_id = f"{scene_id}__{os.path.splitext(os.path.basename(rgb_path))[0]}"
        yield {
            "id": sample_id,
            "rgb": rgb,
            "gt": gt,
            "K": K,
            "valid": valid,
        }
        count += 1


def load_scannetpp_list(root: str, split_file: Optional[str] = None, max_samples: Optional[int] = None):
    return list(load_scannetpp(root, split_file=split_file, max_samples=max_samples))
