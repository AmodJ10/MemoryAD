"""
Pre-compute and cache DINOv2 features for all VisA categories.

Usage:
    .venv\\Scripts\\python.exe scripts\\precompute_features_visa.py

Output: data/features/dinov2_vitb14_visa/<category>/train_features.npy
                                                  /test_features.npy
                                                  /test_labels.npy
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from src.backbones.dinov2 import DINOv2Extractor
from src.data_utils.dataset import get_category_dataloaders


DATASET_ROOT = "data/visa"
FEATURE_DIR = "data/features/dinov2_vitb14_visa"
BACKBONE_NAME = "dinov2_vitb14"
LAYERS = [7, 11]
INPUT_SIZE = 518
BATCH_SIZE = 4

CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum",
    "fryum", "macaroni1", "macaroni2",
    "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


def extract_and_save(backbone, category, feature_dir, dataset_root):
    """Extract features for one category and save to disk."""
    cat_dir = Path(feature_dir) / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    # Check if already cached
    train_path = cat_dir / "train_features.npy"
    if train_path.exists():
        print(f"  {category}: already cached, skipping")
        return

    # Train features (patch-level)
    train_loader, test_loader = get_category_dataloaders(
        root=dataset_root, category=category,
        dataset_type="visa",
        input_size=INPUT_SIZE, batch_size=BATCH_SIZE,
    )

    print(f"  {category}: extracting train features ({len(train_loader.dataset)} images)...")
    train_feats = []
    for batch in tqdm(train_loader, desc=f"    train", leave=False):
        feats = backbone.extract(batch["image"])  # [B, P, D]
        B, P, D = feats.shape
        train_feats.append(feats.reshape(B * P, D).cpu().numpy())
    train_feats = np.concatenate(train_feats, axis=0)
    np.save(cat_dir / "train_features.npy", train_feats)
    print(f"    -> train: {train_feats.shape}")

    # Test features (per-image, keep structure for scoring)
    print(f"  {category}: extracting test features ({len(test_loader.dataset)} images)...")
    test_feats_list = []
    test_labels = []
    for batch in tqdm(test_loader, desc=f"    test", leave=False):
        feats = backbone.extract(batch["image"])  # [B, P, D]
        labels = batch["label"].numpy()
        for i in range(feats.shape[0]):
            test_feats_list.append(feats[i].cpu().numpy())  # [P, D]
            test_labels.append(labels[i])

    test_feats_arr = np.stack(test_feats_list)  # [N_test, P, D]
    test_labels_arr = np.array(test_labels)
    np.save(cat_dir / "test_features.npy", test_feats_arr)
    np.save(cat_dir / "test_labels.npy", test_labels_arr)
    print(f"    -> test: {test_feats_arr.shape}, labels: {test_labels_arr.shape}")


def main():
    print("=" * 60)
    print("Pre-computing DINOv2 features for VisA")
    print("=" * 60)

    # Check if VisA data exists
    if not Path(DATASET_ROOT).exists():
        print(f"\nERROR: VisA dataset not found at {DATASET_ROOT}")
        print("Please download VisA and extract to data/visa/")
        print("Expected structure: data/visa/<category>/train/good/*.JPG")
        sys.exit(1)

    t_start = time.time()

    # Load backbone once
    print(f"\nLoading {BACKBONE_NAME}...")
    backbone = DINOv2Extractor(
        model_name=BACKBONE_NAME, layers=LAYERS,
        aggregation="concat", use_fp16=True,
    )
    spatial_dims = backbone.get_spatial_dims(INPUT_SIZE)
    print(f"Feature dim: {backbone.feature_dim}, spatial: {spatial_dims}")

    # Save spatial dims for later use
    meta_dir = Path(FEATURE_DIR)
    meta_dir.mkdir(parents=True, exist_ok=True)
    np.save(meta_dir / "spatial_dims.npy", np.array(spatial_dims))
    np.save(meta_dir / "feature_dim.npy", np.array([backbone.feature_dim]))

    # Extract for each category
    print(f"\nExtracting features for {len(CATEGORIES)} categories...")
    for cat in CATEGORIES:
        extract_and_save(backbone, cat, FEATURE_DIR, DATASET_ROOT)

    elapsed = time.time() - t_start
    print(f"\nDone! Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Features saved to: {FEATURE_DIR}/")

    # Print summary
    total_size = 0
    for cat in CATEGORIES:
        cat_dir = Path(FEATURE_DIR) / cat
        for f in cat_dir.glob("*.npy"):
            total_size += f.stat().st_size
    print(f"Total disk usage: {total_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
