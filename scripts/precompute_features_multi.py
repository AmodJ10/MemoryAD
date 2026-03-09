"""
Pre-compute features for multiple backbones (CLIP, WideResNet) for MVTec AD.

Usage:
    .venv\\Scripts\\python.exe scripts\\precompute_features_multi.py --backbone clip
    .venv\\Scripts\\python.exe scripts\\precompute_features_multi.py --backbone wrn
    .venv\\Scripts\\python.exe scripts\\precompute_features_multi.py --backbone all
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from src.data_utils.dataset import get_category_dataloaders

DATASET_ROOT = "data/mvtec_ad"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

BACKBONE_CONFIG = {
    "clip": {
        "feature_dir": "data/features/clip_vitl14",
        "input_size": 224,
        "batch_size": 8,  # CLIP is lighter than DINOv2
    },
    "wrn": {
        "feature_dir": "data/features/wide_resnet50",
        "input_size": 224,
        "batch_size": 16,  # WRN is lightweight
    },
}


def create_backbone(name):
    """Create and return a backbone extractor."""
    if name == "clip":
        from src.backbones.clip_backbone import CLIPExtractor
        backbone = CLIPExtractor(
            model_name="ViT-L-14", pretrained="openai",
            layers=[18, 23], aggregation="concat", use_fp16=True,
        )
        print(f"CLIP ViT-L/14: dim={backbone.feature_dim}, spatial={backbone.get_spatial_dims(224)}")
        return backbone
    elif name == "wrn":
        from src.backbones.wideresnet import WideResNetExtractor
        backbone = WideResNetExtractor(
            layers=["layer2", "layer3"], aggregation="concat", use_fp16=True,
        )
        print(f"WideResNet-50: dim={backbone.feature_dim}, spatial={backbone.get_spatial_dims(224)}")
        return backbone
    else:
        raise ValueError(f"Unknown backbone: {name}")


def extract_and_save(backbone, category, feature_dir, input_size, batch_size):
    """Extract features for one category and save to disk."""
    cat_dir = Path(feature_dir) / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    train_path = cat_dir / "train_features.npy"
    if train_path.exists():
        print(f"  {category}: already cached, skipping")
        return

    train_loader, test_loader = get_category_dataloaders(
        root=DATASET_ROOT, category=category,
        input_size=input_size, batch_size=batch_size,
    )

    # Train features
    print(f"  {category}: extracting train features...")
    train_feats = []
    for batch in tqdm(train_loader, desc=f"    train", leave=False):
        feats = backbone.extract(batch["image"])  # [B, P, D]
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
        feats = backbone.extract(batch["image"])  # [B, P, D]
        labels = batch["label"].numpy()
        actual_B = min(feats.shape[0], len(labels))
        for i in range(actual_B):
            test_feats_list.append(feats[i].cpu().numpy())
            test_labels.append(labels[i])

    test_feats_arr = np.stack(test_feats_list)
    test_labels_arr = np.array(test_labels)
    np.save(cat_dir / "test_features.npy", test_feats_arr)
    np.save(cat_dir / "test_labels.npy", test_labels_arr)
    print(f"    -> test: {test_feats_arr.shape}, labels: {test_labels_arr.shape}")


def run_backbone(name):
    """Precompute features for a single backbone."""
    cfg = BACKBONE_CONFIG[name]
    feature_dir = cfg["feature_dir"]
    input_size = cfg["input_size"]
    batch_size = cfg["batch_size"]

    print(f"\n{'='*60}")
    print(f"Pre-computing {name.upper()} features for MVTec AD")
    print(f"{'='*60}")

    t0 = time.time()
    backbone = create_backbone(name)

    # Save spatial dims metadata
    meta_dir = Path(feature_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    spatial_dims = backbone.get_spatial_dims(input_size)
    np.save(meta_dir / "spatial_dims.npy", np.array(spatial_dims))
    np.save(meta_dir / "feature_dim.npy", np.array([backbone.feature_dim]))

    for cat in CATEGORIES:
        extract_and_save(backbone, cat, feature_dir, input_size, batch_size)

    elapsed = time.time() - t0
    print(f"\n{name.upper()} done! {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Features saved to: {feature_dir}/")

    # Free GPU memory
    del backbone
    torch.cuda.empty_cache()

    return feature_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["clip", "wrn", "all"])
    args = parser.parse_args()

    if args.backbone == "all":
        for name in ["clip", "wrn"]:
            run_backbone(name)
    else:
        run_backbone(args.backbone)


if __name__ == "__main__":
    main()
