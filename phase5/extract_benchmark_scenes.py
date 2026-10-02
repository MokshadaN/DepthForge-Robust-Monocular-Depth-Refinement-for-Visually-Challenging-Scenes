"""
Crop and isolate individual authentic camouflage test scenes from official COD benchmark sheets.
"""

import cv2
import numpy as np
from pathlib import Path

DATA_DIR = Path("phase5/data/camouflaged").resolve()
OUT_DIR = Path("phase5/data/camo_scenes").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_scenes():
    # 1. Marine Camouflage: Cuttlefish & Coral / Sand
    aquatic_sheet = cv2.imread(str(DATA_DIR / "cod10k_sample_aquatic.png"))
    if aquatic_sheet is not None:
        h, w = aquatic_sheet.shape[:2]
        # Crop 2 distinct marine scenes
        scene_marine_1 = aquatic_sheet[int(h*0.05):int(h*0.48), int(w*0.02):int(w*0.48)]
        scene_marine_2 = aquatic_sheet[int(h*0.05):int(h*0.48), int(w*0.52):int(w*0.98)]
        cv2.imwrite(str(OUT_DIR / "scene_01_marine_flounder.jpg"), scene_marine_1)
        cv2.imwrite(str(OUT_DIR / "scene_02_marine_cuttlefish.jpg"), scene_marine_2)
        print(f"Extracted marine scenes: {scene_marine_1.shape}, {scene_marine_2.shape}")

    # 2. Terrestrial Camouflage: Owl on Bark & Gecko on Lichen
    terr_sheet = cv2.imread(str(DATA_DIR / "cod10k_sample_terrestrial.png"))
    if terr_sheet is not None:
        h, w = terr_sheet.shape[:2]
        scene_terr_1 = terr_sheet[int(h*0.05):int(h*0.48), int(w*0.02):int(w*0.48)]
        scene_terr_2 = terr_sheet[int(h*0.52):int(h*0.95), int(w*0.52):int(w*0.98)]
        cv2.imwrite(str(OUT_DIR / "scene_03_terrestrial_owl.jpg"), scene_terr_1)
        cv2.imwrite(str(OUT_DIR / "scene_04_terrestrial_gecko.jpg"), scene_terr_2)
        print(f"Extracted terrestrial scenes: {scene_terr_1.shape}, {scene_terr_2.shape}")

    # 3. Amphibian / Foliage Camouflage
    amph_sheet = cv2.imread(str(DATA_DIR / "cod10k_sample_amphibian.png"))
    if amph_sheet is not None:
        h, w = amph_sheet.shape[:2]
        scene_amph = amph_sheet[int(h*0.05):int(h*0.92), int(w*0.02):int(w*0.48)]
        cv2.imwrite(str(OUT_DIR / "scene_05_amphibian_frog.jpg"), scene_amph)
        print(f"Extracted amphibian scene: {scene_amph.shape}")

    # 4. ScanNet++ Zero-Texture White Furniture Case
    scannet_rgb = cv2.imread("demo/input/scannetpp_rgb.jpg")
    if scannet_rgb is not None:
        cv2.imwrite(str(OUT_DIR / "scene_06_indoor_white_furniture.jpg"), scannet_rgb)
        print("Included ScanNet++ Indoor Textureless Scene.")

    print(f"\nAll extracted scenes ready in: {OUT_DIR}")

if __name__ == "__main__":
    extract_scenes()
