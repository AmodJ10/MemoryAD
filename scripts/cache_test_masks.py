"""
Cache ground truth masks for P-AUROC evaluation.

Saves test masks as .npy files alongside existing cached features,
so P-AUROC can be computed without re-loading from disk each time.

Usage:
    .venv\\Scripts\\python.exe scripts/cache_test_masks.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.data_utils.dataset import AnomalyDataset

DATASET_ROOT = "data/mvtec_ad"
FEATURE_DIR = "data/features/dinov2_vitb14"
MASK_SIZE = 518  # Match the input size used for feature extraction

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def cache_masks_for_category(category):
    """Cache test masks for one category."""
    out_path = Path(FEATURE_DIR) / category / "test_masks.npy"
    if out_path.exists():
        print(f"  {category}: already cached")
        return

    dataset = AnomalyDataset(
        root=DATASET_ROOT,
        category=category,
        split="test",
        input_size=MASK_SIZE,
        mask_size=MASK_SIZE,
    )

    masks = []
    for i in range(len(dataset)):
        sample = dataset[i]
        masks.append(sample["mask"].numpy())  # [1, H, W]

    masks_arr = np.stack(masks)  # [N, 1, H, W]
    np.save(out_path, masks_arr)
    print(f"  {category}: saved {masks_arr.shape} -> {out_path}")


def main():
    print("=" * 60)
    print("Caching test masks for P-AUROC evaluation")
    print("=" * 60)

    for cat in CATEGORIES:
        cache_masks_for_category(cat)

    print("\nDone! Masks saved to data/features/dinov2_vitb14/<category>/test_masks.npy")


if __name__ == "__main__":
    main()
