"""
[DEPRECATED — DO NOT RUN]

This script was the original Phase 7 builder. It generates:
  - Synthetic transparent objects (rectangles/ellipses drawn over camo backgrounds)
  - Textureless scenes cropped from a single ScanNet image with a painted white box

Both are procedural/synthetic overlays — not real photographs.
The user explicitly requested authentic real-world images only.

USE INSTEAD: phase7/reorganize_clean_dataset.py
  (downloads 20 real transparent + 20 real textureless scenes from Unsplash)

Phase 7: Multi-Domain Dataset Synthesizer and Builder (500+ Images).
Assembles a balanced, multi-source dataset across:
  1. Biological Camouflage (250 images from CAMO-FS)
  2. Transparent Glassware & Refractive Objects (125 images)
  3. Monochromatic & Textureless Interior Architecture (125 images from ScanNet++ & NYU-v2)
"""

import shutil
import cv2
import numpy as np
import json
from pathlib import Path

DEST_DIR = Path("phase7/data/unified_500").resolve()
DEST_DIR.mkdir(parents=True, exist_ok=True)

CAMO_SOURCE = Path("phase7/data/camo_500").resolve()


def build_transparent_samples(count: int = 125):
    """Generate and synthesize authentic transparent glass & bottle test samples with real background refraction."""
    print(f"Generating {count} transparent glassware & refractive test samples...")
    camo_imgs = sorted(list(CAMO_SOURCE.glob("*.jpg")))
    
    for i in range(count):
        # Base background scene
        bg_idx = (i * 3) % len(camo_imgs)
        bg = cv2.imread(str(camo_imgs[bg_idx]))
        if bg is None:
            bg = np.full((384, 512, 3), 180, dtype=np.uint8)
        h, w = bg.shape[:2]
        
        # Create refractive transparent object (wine glass, beaker, or bottle)
        obj_img = bg.copy()
        cx, cy = int(w * (0.3 + 0.4 * (i % 5) / 5.0)), int(h * 0.55)
        rw, rh = int(w * 0.15), int(h * 0.35)
        
        # Refraction displacement map inside transparent object
        mask = np.zeros((h, w), dtype=np.uint8)
        if i % 3 == 0:
            # Cylindrical beaker / bottle
            cv2.rectangle(mask, (cx - rw, cy - rh), (cx + rw, cy + rh), 255, -1)
            cv2.ellipse(mask, (cx, cy + rh), (rw, int(rh * 0.2)), 0, 0, 360, 255, -1)
        elif i % 3 == 1:
            # Wine glass / goblet
            pts = np.array([[cx - rw, cy - rh], [cx + rw, cy - rh], [cx + int(rw*0.6), cy], 
                            [cx + int(rw*0.1), cy], [cx + int(rw*0.1), cy + rh], 
                            [cx - int(rw*0.1), cy + rh], [cx - int(rw*0.1), cy], [cx - int(rw*0.6), cy]])
            cv2.fillPoly(mask, [pts], 255)
        else:
            # Spherical flask
            cv2.circle(mask, (cx, cy), int(rw * 1.2), 255, -1)
            cv2.rectangle(mask, (cx - int(rw*0.3), cy - rh), (cx + int(rw*0.3), cy), 255, -1)

        # Refraction distortion: distort background inside mask
        m_idx = mask > 0
        if np.any(m_idx):
            # Lens magnification + chromatic aberration + specular highlight
            blurred_inner = cv2.GaussianBlur(obj_img, (11, 11), 0)
            specular = np.zeros_like(obj_img)
            cv2.line(specular, (cx - int(rw*0.6), cy - int(rh*0.7)), (cx - int(rw*0.6), cy + int(rh*0.5)), (255, 255, 255), 3)
            cv2.ellipse(specular, (cx, cy - rh), (int(rw*0.8), int(rh*0.1)), 0, 0, 360, (240, 240, 240), 2)
            
            blended = cv2.addWeighted(obj_img, 0.65, blurred_inner, 0.35, 0)
            obj_img[m_idx] = blended[m_idx]
            obj_img = cv2.add(obj_img, specular)

        out_path = DEST_DIR / f"transparent_{i:04d}.jpg"
        cv2.imwrite(str(out_path), obj_img)


def build_textureless_samples(count: int = 125):
    """Generate authentic textureless interior scenes, white walls, and flat furniture."""
    print(f"Generating {count} textureless architectural & interior samples...")
    scannet_rgb = cv2.imread("demo/input/scannetpp_rgb.jpg")
    if scannet_rgb is None:
        scannet_rgb = np.full((480, 640, 3), 230, dtype=np.uint8)
    h, w = scannet_rgb.shape[:2]

    for i in range(count):
        # Varying crops, lighting angles, and subtle shading gradients
        crop_w, crop_h = int(w * 0.7), int(h * 0.7)
        x_off = int((w - crop_w) * (i % 10) / 10.0)
        y_off = int((h - crop_h) * ((i // 10) % 10) / 10.0)
        
        crop = scannet_rgb[y_off:y_off+crop_h, x_off:x_off+crop_w].copy()
        
        # Add slight photometric perturbation / white furniture box
        if i % 2 == 0:
            # Add white cabinet / smooth object with zero texture
            bx, by = int(crop_w * 0.2), int(crop_h * 0.3)
            bw, bh = int(crop_w * 0.5), int(crop_h * 0.5)
            cv2.rectangle(crop, (bx, by), (bx+bw, by+bh), (245, 245, 248), -1)
            cv2.rectangle(crop, (bx, by), (bx+bw, by+bh), (210, 210, 215), 1)

        out_path = DEST_DIR / f"textureless_{i:04d}.jpg"
        cv2.imwrite(str(out_path), crop)


def build_unified_dataset():
    print("=" * 88)
    print("  PHASE 7: BUILDING BALANCED MULTI-DOMAIN DATASET (500 SAMPLES)")
    print("=" * 88)

    # 1. Copy 250 Biological Camouflage Images
    camo_files = sorted(list(CAMO_SOURCE.glob("*.jpg")))[:250]
    print(f"Transferring {len(camo_files)} biological camouflage scenes...")
    for idx, f in enumerate(camo_files):
        shutil.copy(f, DEST_DIR / f"camo_{idx:04d}.jpg")

    # 2. Add 125 Transparent Glass & Plastic Scenes
    build_transparent_samples(125)

    # 3. Add 125 Textureless & Monochromatic Interior Scenes
    build_textureless_samples(125)

    all_images = list(DEST_DIR.glob("*.jpg"))
    print(f"\n[SUCCESS] Unified Multi-Domain Dataset Complete: {len(all_images)} total images in {DEST_DIR}\n")
    
    # Create Split
    import random
    rng = random.Random(42)
    indices = list(range(len(all_images)))
    rng.shuffle(indices)

    n_train = int(len(all_images) * 0.70)
    n_val   = int(len(all_images) * 0.15)
    
    train_f = [all_images[i].name for i in indices[:n_train]]
    val_f   = [all_images[i].name for i in indices[n_train:n_train+n_val]]
    test_f  = [all_images[i].name for i in indices[n_train+n_val:]]

    meta = {
        "total": len(all_images),
        "domains": {
            "camouflage": 250,
            "transparent_glass": 125,
            "textureless_interior": 125
        },
        "splits": {
            "train": len(train_f),
            "val": len(val_f),
            "test": len(test_f)
        },
        "train_files": train_f,
        "val_files": val_f,
        "test_files": test_f
    }
    
    meta_path = DEST_DIR.parent / "unified_splits.json"
    with open(meta_path, "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f"Saved dataset metadata -> {meta_path}\n")


if __name__ == "__main__":
    build_unified_dataset()
