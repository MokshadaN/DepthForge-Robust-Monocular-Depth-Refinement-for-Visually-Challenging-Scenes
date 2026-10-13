"""
Phase 8 — Data Builder.

Builds a supervised depth dataset from the 522 real images in unified_500:
  Step 1: Copy each real photo to phase8/data/rgb/
  Step 2: Run MiDaS DPT_Large (teacher) on every image → pseudo-GT depth
           Saved as float32 .npy in phase8/data/pseudo_gt/
           NOTE: DPT_Large is a DIFFERENT model from UniDAC — training against
           its output is cross-model supervision, NOT circular self-distillation.
  Step 3: Scene-level grouped split (70/15/15) → phase8/data/splits.json
           All views of the same scene stay in the same partition.

Why DPT_Large as pseudo-GT?
  - It was trained on 1.5M+ diverse images via MiDaS mixing strategy
  - It produces excellent relative depth structure on indoor / object scenes
  - The SILog loss is scale-invariant, so absolute scale errors in the
    pseudo-GT do not mislead training — only depth RATIOS matter
  - For the camouflage domain there is no public RGB-D ground-truth dataset;
    cross-model pseudo-GT is the strongest available supervision signal

Run:
    python phase8/build_phase8_data.py
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ── Paths ────────────────────────────────────────────────────────────────────
SRC_DIR    = Path("phase7/data/unified_500").resolve()
P8_DIR     = Path("phase8/data").resolve()
RGB_DIR    = P8_DIR / "rgb"
DEPTH_DIR  = P8_DIR / "pseudo_gt"
SPLITS_F   = P8_DIR / "splits.json"

RGB_DIR.mkdir(parents=True, exist_ok=True)
DEPTH_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (480, 640)   # (H, W) — fixed resolution for all samples


# ── Teacher model ─────────────────────────────────────────────────────────────
def load_teacher():
    """Load MiDaS DPT_Large as the pseudo-GT teacher."""
    from phase4.compare_all_models import MiDaSModelBackend
    print("  Loading MiDaS DPT_Large (teacher model)...")
    teacher = MiDaSModelBackend("DPT_Large", device="cpu")
    print("  DPT_Large loaded OK\n")
    return teacher


# ── Pseudo-GT generation ──────────────────────────────────────────────────────
def generate_pseudo_gt(src_files: list[Path], teacher) -> list[dict]:
    """
    For each source image:
      1. Resize to TARGET_SIZE, save to rgb/
      2. Run DPT_Large → depth, normalise to [0,1] range, save as .npy to pseudo_gt/
      3. Return list of metadata dicts {name, domain, rgb_path, depth_path}
    """
    records = []
    n = len(src_files)

    for idx, src in enumerate(src_files, 1):
        name = src.stem          # e.g. camo_0001
        domain = name.split("_")[0]  # camo | transparent | textureless

        rgb_out   = RGB_DIR   / f"{name}.jpg"
        depth_out = DEPTH_DIR / f"{name}.npy"

        # --- RGB copy (resized) ---
        img = cv2.imread(str(src))
        if img is None:
            print(f"  [SKIP] Cannot read {src.name}")
            continue
        img_res = cv2.resize(img, (TARGET_SIZE[1], TARGET_SIZE[0]))
        cv2.imwrite(str(rgb_out), img_res)

        # --- DPT_Large pseudo-GT depth ---
        depth_raw = teacher.predict(img_res, sample_cfg={},
                                    output_size=(TARGET_SIZE[0], TARGET_SIZE[1]))
        # Normalise to [0.01, 1.0] (strictly positive for log-loss)
        d_min = float(np.percentile(depth_raw, 2))
        d_max = float(np.percentile(depth_raw, 98))
        depth_norm = np.clip((depth_raw - d_min) / (d_max - d_min + 1e-6), 0.01, 1.0)
        np.save(str(depth_out), depth_norm.astype(np.float32))

        records.append({
            "name":       name,
            "domain":     domain,
            "rgb":        str(rgb_out.relative_to(REPO_ROOT)),
            "depth_gt":   str(depth_out.relative_to(REPO_ROOT)),
        })

        if idx % 50 == 0 or idx == n:
            print(f"  [{idx:>3}/{n}] processed — latest: {name}")

    return records


# ── Scene-level split ─────────────────────────────────────────────────────────
def make_splits(records: list[dict]) -> dict:
    """
    Group records by scene root (strip _view* suffix).
    Split scenes 70/15/15; all views of the same scene stay together.
    """
    scenes: dict[str, list] = {}
    for r in records:
        key = r["name"].split("_view")[0] if "_view" in r["name"] else r["name"]
        scenes.setdefault(key, []).append(r)

    keys = sorted(scenes.keys())
    rng  = random.Random(42)
    rng.shuffle(keys)

    n      = len(keys)
    n_tr   = int(n * 0.70)
    n_va   = int(n * 0.15)

    train_r = [r for k in keys[:n_tr]         for r in scenes[k]]
    val_r   = [r for k in keys[n_tr:n_tr+n_va] for r in scenes[k]]
    test_r  = [r for k in keys[n_tr+n_va:]    for r in scenes[k]]

    def domain_counts(recs):
        dc = {}
        for r in recs:
            dc[r["domain"]] = dc.get(r["domain"], 0) + 1
        return dc

    total = len(records)
    dc = domain_counts(records)

    return {
        "total":              total,
        "unique_scenes":      n,
        "leakage_protection": "Strict scene-level grouping — all variants of a scene share a partition.",
        "teacher_model":      "MiDaS DPT_Large (cross-model pseudo-GT, not UniDAC self-distillation)",
        "target_size":        list(TARGET_SIZE),
        "domain_counts":      dc,
        "domain_percent":     {k: f"{100*v/total:.1f}%" for k, v in dc.items()},
        "split_counts": {
            "train": len(train_r), "val": len(val_r), "test": len(test_r)
        },
        "train": train_r,
        "val":   val_r,
        "test":  test_r,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def build():
    print("=" * 84)
    print("  PHASE 8: SUPERVISED DEPTH DATASET BUILDER (DPT_Large pseudo-GT)")
    print("=" * 84)

    src_files = sorted(SRC_DIR.glob("*.jpg"))
    if not src_files:
        raise FileNotFoundError(
            f"No images found in {SRC_DIR}\n"
            "Run phase7/reorganize_clean_dataset.py first."
        )
    print(f"\nSource: {len(src_files)} images from {SRC_DIR}")

    teacher = load_teacher()

    print(f"Generating DPT_Large pseudo-GT for {len(src_files)} images...")
    print(f"(~2–4 s/image on CPU = {len(src_files)*3//60}–{len(src_files)*4//60} min total)\n")

    records = generate_pseudo_gt(src_files, teacher)
    splits  = make_splits(records)

    with open(SPLITS_F, "w") as fp:
        json.dump(splits, fp, indent=2)

    dc = splits["domain_counts"]
    sc = splits["split_counts"]
    tot = splits["total"]

    print("\n" + "=" * 84)
    print("  BUILD COMPLETE")
    print("=" * 84)
    print(f"  Total images    : {tot}")
    print(f"  Unique scenes   : {splits['unique_scenes']}")
    print(f"  Teacher model   : MiDaS DPT_Large (cross-model, not circular)")
    print(f"  Domain balance  :")
    for k, v in dc.items():
        print(f"    {k:<14}: {v:>4}  ({100*v/tot:.1f}%)")
    print(f"  Train / Val / Test : {sc['train']} / {sc['val']} / {sc['test']}")
    print(f"  Splits saved    -> {SPLITS_F}\n")


if __name__ == "__main__":
    build()
