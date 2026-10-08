"""
Phase 7: Dataset Manager and Deterministic Split Generator for 500+ Images.
Creates Train (70%), Validation (15%), and Test (15%) partitions.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
import cv2

DATA_DIR = Path("phase7/data/camo_500").resolve()
SPLITS_FILE = Path("phase7/data/dataset_splits.json").resolve()
SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)


def prepare_dataset_splits(seed: int = 42):
    image_files = sorted(list(DATA_DIR.glob("*.jpg")))
    total = len(image_files)
    print(f"Discovered {total} valid images in {DATA_DIR}")

    if total == 0:
        raise FileNotFoundError(f"No images found in {DATA_DIR}")

    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled_indices = list(range(total))
    rng.shuffle(shuffled_indices)

    # 70% Train, 15% Val, 15% Test
    n_train = int(total * 0.70)
    n_val = int(total * 0.15)
    n_test = total - n_train - n_val

    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:n_train + n_val]
    test_indices = shuffled_indices[n_train + n_val:]

    train_files = [image_files[i].name for i in train_indices]
    val_files = [image_files[i].name for i in val_indices]
    test_files = [image_files[i].name for i in test_indices]

    splits = {
        "total_images": total,
        "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "counts": {"train": len(train_files), "val": len(val_files), "test": len(test_files)},
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    with open(SPLITS_FILE, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Dataset Split Successfully Configured:")
    print(f"  - Train Set:      {len(train_files)} images (70%)")
    print(f"  - Validation Set: {len(val_files)} images (15%)")
    print(f"  - Test Set:       {len(test_files)} images (15%)")
    print(f"Metadata saved -> {SPLITS_FILE}\n")
    return splits


if __name__ == "__main__":
    prepare_dataset_splits()
