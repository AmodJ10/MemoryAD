"""
Create lightweight demo assets for judge-friendly runs.

Outputs:
1) Tiny MVTec-style image subset under demo/demo_data/mvtec_subset
2) Tiny cached feature subset under demo/demo_features/dinov2_vitb14
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS]


def resolve_mask(gt_root: Path, defect_type: str, image_path: Path) -> Path | None:
    mask_path = gt_root / defect_type / image_path.name
    if mask_path.exists():
        return mask_path

    png_mask = mask_path.with_suffix(".png")
    if png_mask.exists():
        return png_mask

    mvtec_mask = mask_path.with_name(f"{image_path.stem}_mask.png")
    if mvtec_mask.exists():
        return mvtec_mask

    return None


def copy_files(paths: Iterable[Path], dst_dir: Path) -> list[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in paths:
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def create_demo_dataset(
    source_root: Path,
    target_root: Path,
    categories: list[str],
    train_good: int,
    test_good: int,
    test_anom: int,
) -> dict:
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    summary = {"categories": {}}

    for category in categories:
        src_cat = source_root / category
        dst_cat = target_root / category

        train_good_imgs = list_images(src_cat / "train" / "good")[:train_good]
        copied_train = copy_files(train_good_imgs, dst_cat / "train" / "good")

        test_good_imgs = list_images(src_cat / "test" / "good")[:test_good]
        copied_good = copy_files(test_good_imgs, dst_cat / "test" / "good")

        defect_dirs = [
            d for d in sorted((src_cat / "test").iterdir())
            if d.is_dir() and d.name != "good"
        ]
        if not defect_dirs:
            raise RuntimeError(f"No anomalous test folders found for category: {category}")

        defect_dir = defect_dirs[0]
        defect_name = defect_dir.name
        defect_imgs = list_images(defect_dir)[:test_anom]
        copied_defect = copy_files(defect_imgs, dst_cat / "test" / defect_name)

        gt_root = src_cat / "ground_truth"
        copied_masks = 0
        for img in copied_defect:
            src_img = defect_dir / img.name
            mask = resolve_mask(gt_root, defect_name, src_img)
            if mask is not None:
                copy_files([mask], dst_cat / "ground_truth" / defect_name)
                copied_masks += 1

        summary["categories"][category] = {
            "train_good": len(copied_train),
            "test_good": len(copied_good),
            "test_anomalous": len(copied_defect),
            "anomaly_type": defect_name,
            "masks": copied_masks,
        }

    return summary


def create_demo_feature_cache(
    source_feature_root: Path,
    target_feature_root: Path,
    categories: list[str],
    train_images: int,
    test_good: int,
    test_anom: int,
) -> dict:
    if target_feature_root.exists():
        shutil.rmtree(target_feature_root)
    target_feature_root.mkdir(parents=True, exist_ok=True)

    spatial_dims = np.load(source_feature_root / "spatial_dims.npy")
    feature_dim = np.load(source_feature_root / "feature_dim.npy")
    np.save(target_feature_root / "spatial_dims.npy", spatial_dims)
    np.save(target_feature_root / "feature_dim.npy", feature_dim)

    patch_count = int(np.prod(spatial_dims))
    summary = {"categories": {}, "patch_count": patch_count}

    for category in categories:
        src_cat = source_feature_root / category
        dst_cat = target_feature_root / category
        dst_cat.mkdir(parents=True, exist_ok=True)

        train_features = np.load(src_cat / "train_features.npy")
        keep_train_patches = min(train_images * patch_count, train_features.shape[0])
        demo_train = train_features[:keep_train_patches]

        test_features = np.load(src_cat / "test_features.npy")
        test_labels = np.load(src_cat / "test_labels.npy")

        good_idx = np.where(test_labels == 0)[0][:test_good]
        bad_idx = np.where(test_labels == 1)[0][:test_anom]

        if len(good_idx) == 0 or len(bad_idx) == 0:
            raise RuntimeError(
                f"Category '{category}' does not have enough positive/negative test samples in cached features."
            )

        keep_idx = np.concatenate([good_idx, bad_idx])
        demo_test_features = test_features[keep_idx]
        demo_test_labels = test_labels[keep_idx]

        np.save(dst_cat / "train_features.npy", demo_train)
        np.save(dst_cat / "test_features.npy", demo_test_features)
        np.save(dst_cat / "test_labels.npy", demo_test_labels)

        summary["categories"][category] = {
            "train_patches": int(demo_train.shape[0]),
            "test_images": int(demo_test_features.shape[0]),
            "test_good": int(np.sum(demo_test_labels == 0)),
            "test_anomalous": int(np.sum(demo_test_labels == 1)),
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tiny demo dataset and feature cache")
    parser.add_argument("--source-data", type=str, default="data/mvtec_ad")
    parser.add_argument("--source-features", type=str, default="data/features/dinov2_vitb14")
    parser.add_argument("--target-data", type=str, default="demo/demo_data/mvtec_subset")
    parser.add_argument("--target-features", type=str, default="demo/demo_features/dinov2_vitb14")
    parser.add_argument("--categories", nargs="+", default=["bottle", "cable"])
    parser.add_argument("--train-good", type=int, default=8)
    parser.add_argument("--test-good", type=int, default=4)
    parser.add_argument("--test-anom", type=int, default=4)
    parser.add_argument("--manifest", type=str, default="demo/demo_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_data = Path(args.source_data)
    source_features = Path(args.source_features)
    target_data = Path(args.target_data)
    target_features = Path(args.target_features)

    if not source_data.exists():
        raise FileNotFoundError(f"Source dataset folder not found: {source_data}")
    if not source_features.exists():
        raise FileNotFoundError(f"Source feature cache not found: {source_features}")

    dataset_summary = create_demo_dataset(
        source_root=source_data,
        target_root=target_data,
        categories=args.categories,
        train_good=args.train_good,
        test_good=args.test_good,
        test_anom=args.test_anom,
    )

    feature_summary = create_demo_feature_cache(
        source_feature_root=source_features,
        target_feature_root=target_features,
        categories=args.categories,
        train_images=args.train_good,
        test_good=args.test_good,
        test_anom=args.test_anom,
    )

    manifest = {
        "dataset": dataset_summary,
        "features": feature_summary,
        "paths": {
            "dataset_root": str(target_data).replace("\\", "/"),
            "feature_root": str(target_features).replace("\\", "/"),
        },
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Demo assets created successfully.")
    print(f"  Dataset root: {target_data}")
    print(f"  Feature root: {target_features}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
