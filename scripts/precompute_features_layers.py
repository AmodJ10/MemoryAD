"""
Pre-compute DINOv2 features with alternate layer configurations.

Used for the layer selection ablation (E11). Each config saves to a
separate feature directory.

Usage:
    .venv\\Scripts\\python.exe scripts/precompute_features_layers.py --layers 6 11
    .venv\\Scripts\\python.exe scripts/precompute_features_layers.py --layers 7 9 11
    .venv\\Scripts\\python.exe scripts/precompute_features_layers.py --layers 11
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from src.backbones.dinov2 import DINOv2Extractor
from src.data_utils.dataset import get_category_dataloaders

DATASET_ROOT = "data/mvtec_ad"
BACKBONE_NAME = "dinov2_vitb14"
INPUT_SIZE = 518
BATCH_SIZE = 4

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def extract_and_save(backbone, category, feature_dir):
    """Extract features for one category and save."""
    cat_dir = Path(feature_dir) / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    train_path = cat_dir / "train_features.npy"
    if train_path.exists():
        print(f"  {category}: already cached, skipping")
        return

    train_loader, test_loader = get_category_dataloaders(
        root=DATASET_ROOT, category=category,
        input_size=INPUT_SIZE, batch_size=BATCH_SIZE,
    )

    # Train features
    print(f"  {category}: extracting train features...")
    train_feats = []
    for batch in tqdm(train_loader, desc=f"    train", leave=False):
        feats = backbone.extract(batch["image"])
        B, P, D = feats.shape
        train_feats.append(feats.reshape(B * P, D).cpu().numpy())
    train_feats = np.concatenate(train_feats, axis=0)
    np.save(cat_dir / "train_features.npy", train_feats)
    print(f"    -> train: {train_feats.shape}")

    # Test features
    print(f"  {category}: extracting test features...")
    test_feats_list = []
    test_labels = []
    for batch in tqdm(test_loader, desc=f"    test", leave=False):
        feats = backbone.extract(batch["image"])
        labels = batch["label"].numpy()
        for i in range(feats.shape[0]):
            test_feats_list.append(feats[i].cpu().numpy())
            test_labels.append(labels[i])

    test_feats_arr = np.stack(test_feats_list)
    test_labels_arr = np.array(test_labels)
    np.save(cat_dir / "test_features.npy", test_feats_arr)
    np.save(cat_dir / "test_labels.npy", test_labels_arr)
    print(f"    -> test: {test_feats_arr.shape}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute DINOv2 features with custom layers")
    parser.add_argument("--layers", nargs="+", type=int, required=True,
                        help="Layer indices to extract (e.g. --layers 6 11)")
    args = parser.parse_args()

    layers = args.layers
    layers_str = "_".join(map(str, layers))
    feature_dir = f"data/features/dinov2_vitb14_l{layers_str}"

    print("=" * 60)
    print(f"Pre-computing DINOv2 features (layers={layers})")
    print(f"Output: {feature_dir}/")
    print("=" * 60)

    t_start = time.time()

    backbone = DINOv2Extractor(
        model_name=BACKBONE_NAME, layers=layers,
        aggregation="concat", use_fp16=True,
    )
    spatial_dims = backbone.get_spatial_dims(INPUT_SIZE)
    print(f"Feature dim: {backbone.feature_dim}, spatial: {spatial_dims}")

    meta_dir = Path(feature_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    np.save(meta_dir / "spatial_dims.npy", np.array(spatial_dims))
    np.save(meta_dir / "feature_dim.npy", np.array([backbone.feature_dim]))

    for cat in CATEGORIES:
        extract_and_save(backbone, cat, feature_dir)

    elapsed = time.time() - t_start
    print(f"\nDone! Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Features saved to: {feature_dir}/")


if __name__ == "__main__":
    main()
