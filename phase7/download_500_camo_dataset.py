"""
Phase 7: Download 500 authentic camouflaged images from the CAMO-FS benchmark on HuggingFace.
Uses ThreadPoolExecutor for fast concurrent downloading.
"""

import urllib.request
import concurrent.futures
import time
from pathlib import Path

OUT_DIR = Path("phase7/data/camo_500").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://huggingface.co/datasets/Thanhdat3010/CAMO-FS/resolve/main/images/images"
TOTAL_IMAGES = 500

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UniDACResearch/1.0"
}


def download_single_image(idx: int):
    filename = f"camouflage_{idx:08d}.jpg"
    target_path = OUT_DIR / filename
    if target_path.exists() and target_path.stat().st_size > 1000:
        return filename, True
        
    url = f"{BASE_URL}/{filename}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = resp.read()
            with open(target_path, "wb") as f:
                f.write(data)
        return filename, True
    except Exception as e:
        return filename, False


def download_dataset():
    print(f"Starting concurrent download of {TOTAL_IMAGES} images to {OUT_DIR}...")
    start_time = time.time()
    success_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_single_image, i): i for i in range(TOTAL_IMAGES)}
        for future in concurrent.futures.as_completed(futures):
            fname, ok = future.result()
            if ok:
                success_count += 1
                if success_count % 50 == 0 or success_count == TOTAL_IMAGES:
                    print(f"  Downloaded: [{success_count}/{TOTAL_IMAGES}] images ({time.time()-start_time:.1f}s)")
                    
    elapsed = time.time() - start_time
    print(f"\n[COMPLETE] Successfully downloaded {success_count}/{TOTAL_IMAGES} camouflage images in {elapsed:.1f} seconds.\n")


if __name__ == "__main__":
    download_dataset()
