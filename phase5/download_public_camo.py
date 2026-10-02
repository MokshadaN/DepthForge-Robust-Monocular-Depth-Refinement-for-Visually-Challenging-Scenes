"""
Download authentic public camouflaged and textureless test samples from open public sources.
Sources: Public datasets / Wikimedia Commons public domain scientific repositories.
"""

import os
import sys
import urllib.request
import cv2
import numpy as np
from pathlib import Path

DEST_DIR = Path("phase5/data/public_benchmark").resolve()
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Curated set of high-difficulty camouflage & textureless samples:
SAMPLES = [
    {
        "name": "camo_01_flounder_sand",
        "description": "Peacock Flounder perfectly matching textured sand/gravel substrate",
        "category": "marine_camouflage",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Bothus_mancus_camouflaged.jpg/1280px-Bothus_mancus_camouflaged.jpg"
    },
    {
        "name": "camo_02_screech_owl_bark",
        "description": "Eastern Screech Owl seamlessly matching tree bark texture and color",
        "category": "terrestrial_camouflage",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Megascops_asio_Eastern_Screech_Owl_camouflaged.jpg/1280px-Megascops_asio_Eastern_Screech_Owl_camouflaged.jpg"
    },
    {
        "name": "camo_03_mossy_gecko",
        "description": "Mossy Leaf-tailed Gecko (Uroplatus sikorae) on lichen/moss tree surface",
        "category": "reptile_camouflage",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Uroplatus_sikorae_01.jpg/1280px-Uroplatus_sikorae_01.jpg"
    },
    {
        "name": "camo_04_craugastor_frog",
        "description": "Fitzinger's Robber Frog camouflaged in dead leaf litter",
        "category": "amphibian_camouflage",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Craugastor_fitzingeri_%28camouflaged%29.jpg/1280px-Craugastor_fitzingeri_%28camouflaged%29.jpg"
    },
    {
        "name": "textureless_05_white_interior",
        "description": "Monochromatic white interior wall, white cabinetry, low gradient cues",
        "category": "textureless_interior",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Interior_design_living_room_white.jpg/1280px-Interior_design_living_room_white.jpg"
    }
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def download_samples():
    print(f"Downloading {len(SAMPLES)} public benchmark samples to {DEST_DIR}...")
    downloaded = []
    
    for item in SAMPLES:
        target_path = DEST_DIR / f"{item['name']}.jpg"
        print(f"Fetching: {item['name']} ({item['category']})...")
        try:
            req = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                with open(target_path, "wb") as f:
                    f.write(data)
            
            # Verify valid image
            img = cv2.imread(str(target_path))
            if img is not None and img.shape[0] > 100:
                print(f"  -> OK: {img.shape[1]}x{img.shape[0]} px ({target_path.stat().st_size // 1024} KB)")
                downloaded.append((item, target_path))
            else:
                print(f"  -> Corrupt image format, removing.")
                target_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"  -> Error downloading {item['name']}: {e}")
            
    print(f"\nCompleted: {len(downloaded)}/{len(SAMPLES)} verified public samples downloaded.\n")
    return downloaded

if __name__ == "__main__":
    download_samples()
