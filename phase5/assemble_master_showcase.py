"""
Assemble all 6 camouflage and textureless visual panels into a single high-impact showcase banner.
"""

import cv2
import numpy as np
from pathlib import Path

RES_DIR = Path("phase5/results").resolve()

def assemble_master_showcase():
    panel_files = [
        "panel_scene_01_marine_flounder.jpg",
        "panel_scene_02_marine_cuttlefish.jpg",
        "panel_scene_03_terrestrial_owl.jpg",
        "panel_scene_04_terrestrial_gecko.jpg",
        "panel_scene_05_amphibian_frog.jpg",
        "panel_scene_06_indoor_white_furniture.jpg"
    ]
    
    loaded = []
    target_width = 1600
    for p_name in panel_files:
        p_path = RES_DIR / p_name
        if p_path.exists():
            img = cv2.imread(str(p_path))
            h, w = img.shape[:2]
            scaled_h = int(h * target_width / w)
            img_res = cv2.resize(img, (target_width, scaled_h))
            loaded.append(img_res)
            
    if not loaded:
        print("No panels found to assemble.")
        return

    # Master title banner
    banner = np.zeros((80, target_width, 3), dtype=np.uint8)
    banner[:] = [15, 18, 24]
    cv2.putText(banner, "UniDAC vs. Baselines: Authentic Camouflage & Textureless Benchmark Suite", 
                (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(banner, "Comparative Depth Reconstruction across Marine, Terrestrial, Amphibian & Monochromatic Indoor Scenes", 
                (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 200, 220), 1, cv2.LINE_AA)

    # Divider bar
    divider = np.zeros((6, target_width, 3), dtype=np.uint8)
    divider[:] = [40, 60, 80]

    all_blocks = [banner, divider]
    for idx, panel in enumerate(loaded):
        all_blocks.append(panel)
        if idx < len(loaded) - 1:
            all_blocks.append(divider)

    master_showcase = np.vstack(all_blocks)
    out_p = RES_DIR / "master_camouflage_showcase_panel.jpg"
    cv2.imwrite(str(out_p), master_showcase, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Saved master showcase ({master_showcase.shape[1]}x{master_showcase.shape[0]} px) -> {out_p}")

if __name__ == "__main__":
    assemble_master_showcase()
