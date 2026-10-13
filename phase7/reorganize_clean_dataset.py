"""
Phase 7: Reorganize Dataset – Corrected Version.

FIXES APPLIED (Audit 2026-10-08):
  B3: Was only 5 transparent + 6 textureless scenes = 95.8% camo imbalance.
      Now 20 transparent + 20 textureless scenes = 40 real scenes x 2 views.
  D1: Dual-view was 85%/60% arbitrary crops from same download.
      Now View-1 = full frame (no crop), View-2 = centre-70% crop to keep
      the central transparent/textureless subject intact.

Guarantees zero data leakage between Train, Validation, and Test partitions
via strict scene-level grouping.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
import urllib.request
import cv2

DATA_DIR    = Path("phase7/data/unified_500").resolve()
CAMO_SOURCE = Path("phase7/data/camo_500").resolve()
SPLITS_FILE = Path("phase7/data/unified_splits.json").resolve()

# ---------------------------------------------------------------------------
# 20 Transparent / Refractive scenes  (real Unsplash photos)
# ---------------------------------------------------------------------------
TRANSPARENT_URLS = [
    ("glass_01_wine",        "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=1000&q=85"),
    ("glass_02_beaker",      "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?w=1000&q=85"),
    ("glass_03_water_cup",   "https://images.unsplash.com/photo-1559839914-17aae19cec71?w=1000&q=85"),
    ("glass_04_bottle",      "https://images.unsplash.com/photo-1592945403244-b3fbafd7f539?w=1000&q=85"),
    ("glass_05_window",      "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=1000&q=85"),
    ("glass_06_flasks",      "https://images.unsplash.com/photo-1581093458791-9f3c3900df4b?w=1000&q=85"),
    ("glass_07_vase",        "https://images.unsplash.com/photo-1526045612212-70caf35c14df?w=1000&q=85"),
    ("glass_08_cocktail",    "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=1000&q=85"),
    ("glass_09_jar",         "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1000&q=85"),
    ("glass_10_ice_cube",    "https://images.unsplash.com/photo-1548366086-7f1b76106622?w=1000&q=85"),
    ("glass_11_aquarium",    "https://images.unsplash.com/photo-1524704654690-b56c05c78a00?w=1000&q=85"),
    ("glass_12_greenhouse",  "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=1000&q=85"),
    ("glass_13_mirror",      "https://images.unsplash.com/photo-1616047006789-b7af5afb8c20?w=1000&q=85"),
    ("glass_14_prism",       "https://images.unsplash.com/photo-1567095761054-7a02e69e5c43?w=1000&q=85"),
    ("glass_15_test_tubes",  "https://images.unsplash.com/photo-1582719471384-894fbb16e074?w=1000&q=85"),
    ("glass_16_reading_glass","https://images.unsplash.com/photo-1509228468518-180dd4864904?w=1000&q=85"),
    ("glass_17_wine_shelf",  "https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?w=1000&q=85"),
    ("glass_18_crystal",     "https://images.unsplash.com/photo-1579783902614-a3fb3927b6a5?w=1000&q=85"),
    ("glass_19_shower",      "https://images.unsplash.com/photo-1552321554-5fefe8c9ef14?w=1000&q=85"),
    ("glass_20_storefront",  "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=1000&q=85"),
]

# ---------------------------------------------------------------------------
# 20 Textureless / Monochromatic interior scenes  (real Unsplash photos)
# ---------------------------------------------------------------------------
TEXTURELESS_URLS = [
    ("texless_01_white_room",     "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=1000&q=85"),
    ("texless_02_white_wall",     "https://images.unsplash.com/photo-1513519245088-0e12902e5a38?w=1000&q=85"),
    ("texless_03_kitchen",        "https://images.unsplash.com/photo-1556911220-e15b29be8c8f?w=1000&q=85"),
    ("texless_04_blank_studio",   "https://images.unsplash.com/photo-1507089947368-19c1da9775ae?w=1000&q=85"),
    ("texless_05_staircase_wall", "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=1000&q=85"),
    ("texless_06_desk_surface",   "https://images.unsplash.com/photo-1497366216548-37526070297c?w=1000&q=85"),
    ("texless_07_hallway",        "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=1000&q=85"),
    ("texless_08_concrete_wall",  "https://images.unsplash.com/photo-1557683316-973673baf926?w=1000&q=85"),
    ("texless_09_white_bedroom",  "https://images.unsplash.com/photo-1540518614846-7eded433c457?w=1000&q=85"),
    ("texless_10_minimal_office", "https://images.unsplash.com/photo-1497366811353-6870744d04b2?w=1000&q=85"),
    ("texless_11_empty_room",     "https://images.unsplash.com/photo-1536376072261-38c75010e6c9?w=1000&q=85"),
    ("texless_12_white_ceiling",  "https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?w=1000&q=85"),
    ("texless_13_gallery_wall",   "https://images.unsplash.com/photo-1481277542470-605612bd2d61?w=1000&q=85"),
    ("texless_14_modern_lobby",   "https://images.unsplash.com/photo-1540575467063-178a50c2df87?w=1000&q=85"),
    ("texless_15_corridor",       "https://images.unsplash.com/photo-1558403194-611308249627?w=1000&q=85"),
    ("texless_16_flat_roof",      "https://images.unsplash.com/photo-1486325212027-8081e485255e?w=1000&q=85"),
    ("texless_17_bathroom_tile",  "https://images.unsplash.com/photo-1552321554-5fefe8c9ef14?w=1000&q=85"),
    ("texless_18_plaster_wall",   "https://images.unsplash.com/photo-1600607686527-6fb886090705?w=1000&q=85"),
    ("texless_19_smooth_floor",   "https://images.unsplash.com/photo-1584622650111-993a426fbf0a?w=1000&q=85"),
    ("texless_20_white_backdrop", "https://images.unsplash.com/photo-1491895200222-0fc4a4c35e18?w=1000&q=85"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UniDACResearch/1.0"}


def _fetch_dual_views(item_list: list, prefix: str) -> int:
    """
    Download each scene once, then save exactly 2 views:
      View 1 – full-frame resize (preserves all edges, especially glass outlines)
      View 2 – centre 70% crop    (keeps central subject, avoids empty corners)

    Returns count of successfully saved view-pairs.
    """
    saved = 0
    for name, url in item_list:
        temp_p = DATA_DIR / f"_tmp_{name}.jpg"
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                temp_p.write_bytes(resp.read())

            img = cv2.imread(str(temp_p))
            if img is None:
                print(f"  [WARN] Could not decode image for {name}, skipping.")
                continue

            h, w = img.shape[:2]

            # ------ View 1: Full Frame (no spatial crop) ------
            view1 = img.copy()
            out1 = DATA_DIR / f"{prefix}_{name}_view1_full.jpg"
            cv2.imwrite(str(out1), view1)

            # ------ View 2: Centre 70% crop ------
            ch, cw = int(h * 0.70), int(w * 0.70)
            y0 = (h - ch) // 2
            x0 = (w - cw) // 2
            view2 = img[y0:y0 + ch, x0:x0 + cw]
            out2 = DATA_DIR / f"{prefix}_{name}_view2_centre.jpg"
            cv2.imwrite(str(out2), view2)

            print(f"  -> 2 views saved for: {name}")
            saved += 1

        except Exception as exc:
            print(f"  [ERR] {name}: {exc}")
        finally:
            temp_p.unlink(missing_ok=True)

    return saved


def _scene_grouped_split(all_files: list[str]) -> dict:
    """
    Group all filenames by their scene root (everything before _view).
    Then split scenes 70/15/15 so both views of the same scene always land
    in the same partition — zero leakage.
    """
    scenes: dict[str, list[str]] = {}
    for fname in all_files:
        # Strip _view1_* / _view2_* suffix to get scene key
        if "_view" in fname:
            key = fname.split("_view")[0]
        else:
            key = fname.rsplit(".", 1)[0]
        scenes.setdefault(key, []).append(fname)

    scene_keys = sorted(scenes.keys())
    rng = random.Random(42)
    rng.shuffle(scene_keys)

    n = len(scene_keys)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train_s = scene_keys[:n_train]
    val_s   = scene_keys[n_train:n_train + n_val]
    test_s  = scene_keys[n_train + n_val:]

    train_f = [f for s in train_s for f in scenes[s]]
    val_f   = [f for s in val_s   for f in scenes[s]]
    test_f  = [f for s in test_s  for f in scenes[s]]

    return {
        "total_images":        len(all_files),
        "total_unique_scenes": len(scene_keys),
        "leakage_protection":  "Strict Scene-Level Grouping — both views of each scene stay in the same partition.",
        "domain_counts": {
            "camouflage":  sum(1 for f in all_files if f.startswith("camo_")),
            "transparent": sum(1 for f in all_files if f.startswith("transparent_")),
            "textureless": sum(1 for f in all_files if f.startswith("textureless_")),
        },
        "counts":       {"train": len(train_f), "val": len(val_f), "test": len(test_f)},
        "train_files":  train_f,
        "val_files":    val_f,
        "test_files":   test_f,
    }


def clean_and_rebuild_dataset():
    print("=" * 88)
    print("  PHASE 7: DATASET PURGE & CORRECTED 2-VIEW SCENE-GROUPED RECONSTRUCTION")
    print("=" * 88)

    # 1. Wipe & recreate unified_500 directory
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Copy ALL 500 Biological Camouflage scenes from CAMO-FS
    camo_files = sorted(CAMO_SOURCE.glob("*.jpg"))
    print(f"\n[Domain 1] Copying {len(camo_files)} biological camouflage scenes...")
    for idx, f in enumerate(camo_files):
        shutil.copy(f, DATA_DIR / f"camo_{idx:04d}.jpg")

    # 3. Download 20 Transparent / Refractive scenes (2 corrected views each)
    print("\n[Domain 2] Downloading 20 transparent glassware scenes (2 views each)...")
    n_trans = _fetch_dual_views(TRANSPARENT_URLS, "transparent")
    print(f"           {n_trans} scenes fetched ({n_trans * 2} view images)\n")

    # 4. Download 20 Textureless / Monochromatic scenes (2 corrected views each)
    print("[Domain 3] Downloading 20 textureless architectural scenes (2 views each)...")
    n_texl = _fetch_dual_views(TEXTURELESS_URLS, "textureless")
    print(f"           {n_texl} scenes fetched ({n_texl * 2} view images)\n")

    # 5. Scene-grouped splits (leakage-free)
    all_files = sorted(f.name for f in DATA_DIR.glob("*.jpg"))
    splits_data = _scene_grouped_split(all_files)

    with open(SPLITS_FILE, "w") as fp:
        json.dump(splits_data, fp, indent=2)

    dc = splits_data["domain_counts"]
    total = splits_data["total_images"]
    print("=" * 88)
    print("  DATASET REBUILD COMPLETE")
    print("=" * 88)
    print(f"  Total Images      : {total}")
    print(f"  Unique Scenes     : {splits_data['total_unique_scenes']}")
    print(f"  Domain Balance    :")
    print(f"    Camouflage      : {dc['camouflage']:>4}  ({100*dc['camouflage']/total:.1f}%)")
    print(f"    Transparent     : {dc['transparent']:>4}  ({100*dc['transparent']/total:.1f}%)")
    print(f"    Textureless     : {dc['textureless']:>4}  ({100*dc['textureless']/total:.1f}%)")
    print(f"  Train Partition   : {splits_data['counts']['train']} images")
    print(f"  Val   Partition   : {splits_data['counts']['val']}  images")
    print(f"  Test  Partition   : {splits_data['counts']['test']}  images")
    print(f"  Metadata Saved -> {SPLITS_FILE}\n")


if __name__ == "__main__":
    clean_and_rebuild_dataset()
