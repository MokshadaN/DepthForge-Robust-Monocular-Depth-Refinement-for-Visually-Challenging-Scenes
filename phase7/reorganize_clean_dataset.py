"""
Phase 7: Reorganize Dataset with Exactly 2 Dual-Scale Views per Scene and Scene-Level Grouped Splitting.
Guarantees zero data leakage between Train, Validation, and Test partitions.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
import cv2

DATA_DIR = Path("phase7/data/unified_500").resolve()
CAMO_SOURCE = Path("phase7/data/camo_500").resolve()
SPLITS_FILE = Path("phase7/data/unified_splits.json").resolve()


def clean_and_rebuild_dataset():
    print("=" * 88)
    print("  PHASE 7: DATASET PURGE & 2-VIEW SCENE-GROUPED RECONSTRUCTION")
    print("=" * 88)

    # 1. Clean out existing unified_500 directory
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Add all 500 unique camouflage scenes from CAMO-FS
    camo_files = sorted(list(CAMO_SOURCE.glob("*.jpg")))
    print(f"Transferring {len(camo_files)} distinct biological camouflage scenes from CAMO-FS...")
    for idx, f in enumerate(camo_files):
        shutil.copy(f, DATA_DIR / f"camo_{idx:04d}.jpg")

    # 3. Add 2 Dual-Scale Views per Real Transparent & Textureless Scene
    # Sourced from high-resolution authentic photographs
    real_transparent_urls = [
        ("glass_01_wine", "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=1000&q=85"),
        ("glass_02_beaker", "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?w=1000&q=85"),
        ("glass_03_water_cup", "https://images.unsplash.com/photo-1559839914-17aae19cec71?w=1000&q=85"),
        ("glass_04_bottle", "https://images.unsplash.com/photo-1592945403244-b3fbafd7f539?w=1000&q=85"),
        ("glass_05_window", "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=1000&q=85")
    ]

    real_texless_urls = [
        ("texless_01_white_room", "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=1000&q=85"),
        ("texless_02_white_wall", "https://images.unsplash.com/photo-1513519245088-0e12902e5a38?w=1000&q=85"),
        ("texless_03_kitchen", "https://images.unsplash.com/photo-1556911220-e15b29be8c8f?w=1000&q=85"),
        ("texless_04_blank_studio", "https://images.unsplash.com/photo-1507089947368-19c1da9775ae?w=1000&q=85"),
        ("texless_05_staircase_wall", "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=1000&q=85"),
        ("texless_06_desk_surface", "https://images.unsplash.com/photo-1497366216548-37526070297c?w=1000&q=85")
    ]

    import urllib.request
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UniDACResearch/1.0"}

    def fetch_and_save_dual_views(item_list, prefix):
        for name, url in item_list:
            temp_target = DATA_DIR / f"temp_{name}.jpg"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    with open(temp_target, "wb") as fp:
                        fp.write(resp.read())
                
                img = cv2.imread(str(temp_target))
                if img is None:
                    continue
                h, w = img.shape[:2]

                # View 1: Wide Context View (85% central crop)
                h1, w1 = int(h * 0.85), int(w * 0.85)
                y1, x1 = int((h - h1) / 2), int((w - w1) / 2)
                view1 = img[y1:y1+h1, x1:x1+w1]
                cv2.imwrite(str(DATA_DIR / f"{prefix}_{name}_view1_wide.jpg"), view1)

                # View 2: Focused Detail View (60% focused crop)
                h2, w2 = int(h * 0.60), int(w * 0.60)
                y2, x2 = int(h * 0.25), int(w * 0.25)
                view2 = img[y2:y2+h2, x2:x2+w2]
                cv2.imwrite(str(DATA_DIR / f"{prefix}_{name}_view2_detail.jpg"), view2)

                print(f"  -> Generated exactly 2 dual views for: {name}")
                temp_target.unlink(missing_ok=True)
            except Exception as e:
                print(f"  -> Error fetching {name}: {e}")

    print("\nFetching transparent glassware scenes (2 views each)...")
    fetch_and_save_dual_views(real_transparent_urls, "transparent")

    print("\nFetching textureless architectural scenes (2 views each)...")
    fetch_and_save_dual_views(real_texless_urls, "textureless")

    # 4. Group-Based Splitting (Never split views of the same scene across train and test)
    all_files = sorted([f.name for f in DATA_DIR.glob("*.jpg")])
    
    # Extract unique scene roots
    scenes = {}
    for f in all_files:
        if "_view" in f:
            scene_root = f.split("_view")[0]
        else:
            scene_root = f.split(".")[0]
        if scene_root not in scenes:
            scenes[scene_root] = []
        scenes[scene_root].append(f)

    scene_keys = sorted(list(scenes.keys()))
    rng = random.Random(42)
    rng.shuffle(scene_keys)

    n_scenes = len(scene_keys)
    n_train_s = int(n_scenes * 0.70)
    n_val_s   = int(n_scenes * 0.15)

    train_scenes = scene_keys[:n_train_s]
    val_scenes   = scene_keys[n_train_s:n_train_s + n_val_s]
    test_scenes  = scene_keys[n_train_s + n_val_s:]

    train_files = [f for s in train_scenes for f in scenes[s]]
    val_files   = [f for s in val_scenes for f in scenes[s]]
    test_files  = [f for s in test_scenes for f in scenes[s]]

    splits_data = {
        "total_images": len(all_files),
        "total_unique_scenes": len(scene_keys),
        "leakage_protection": "Strict Scene-Level Grouping (All views of a scene stay in same partition)",
        "counts": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files)
        },
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files
    }

    with open(SPLITS_FILE, "w") as fp:
        json.dump(splits_data, fp, indent=2)

    print("\n" + "=" * 80)
    print("  DATASET REORGANIZATION COMPLETE & LEAKAGE-FREE")
    print("=" * 80)
    print(f"  Total Verified Images:     {len(all_files)}")
    print(f"  Total Unique Scenes:       {len(scene_keys)}")
    print(f"  - Train Partition:         {len(train_files)} images ({len(train_scenes)} scenes)")
    print(f"  - Validation Partition:    {len(val_files)} images ({len(val_scenes)} scenes)")
    print(f"  - Blind Test Partition:    {len(test_files)} images ({len(test_scenes)} scenes)")
    print(f"  Metadata Saved -> {SPLITS_FILE}\n")


if __name__ == "__main__":
    clean_and_rebuild_dataset()
