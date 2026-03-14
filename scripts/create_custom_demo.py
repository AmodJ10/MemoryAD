"""
Create Custom Demo Package for MemoryAD.

This script generates a self-contained demo folder that includes:
  1. The trained model bundle (global coreset + metadata)
  2. A curated test dataset with both normal and anomalous images from
     every MVTec AD category
  3. The original experiment metrics baked into the bundle metadata

Usage:
    python scripts/create_custom_demo.py

    # Or with custom paths:
    python scripts/create_custom_demo.py \
        --dataset-root data/mvtec_ad \
        --results-json results/E1_mvtec_5task/results.json \
        --existing-bundle exports/mvtec_full_cpu/model_bundle \
        --output custom_demo
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _force_remove_readonly(func, path, exc_info):
    """Handle Windows read-only files during shutil.rmtree."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def copy_file(src: Path, dst: Path) -> None:
    """Copy a single file, creating parent dirs as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def curate_test_images(
    dataset_root: Path,
    output_root: Path,
    categories: list[str],
    good_per_cat: int = 4,
    bad_per_cat: int = 4,
) -> dict[str, Any]:
    """
    Build a compact, representative test set by copying a handful of
    normal ('good') and anomalous images from each category.
    """
    if output_root.exists():
        shutil.rmtree(output_root, onerror=_force_remove_readonly)
    output_root.mkdir(parents=True)

    manifest: dict[str, Any] = {"categories": {}, "totals": {"images": 0, "good": 0, "anomalous": 0}}

    for cat in categories:
        cat_dir = dataset_root / cat
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"

        if not test_dir.exists():
            print(f"  [WARN] Skipping {cat}: test dir not found at {test_dir}")
            continue

        # --- Good images ---
        good_dir = test_dir / "good"
        good_imgs: list[Path] = []
        if good_dir.exists():
            good_imgs = sorted(
                p for p in good_dir.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS
            )[:good_per_cat]

        for p in good_imgs:
            copy_file(p, output_root / cat / "test" / "good" / p.name)

        # --- Anomalous images (pick first defect type found) ---
        defect_dirs = sorted(
            d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"
        )
        bad_imgs: list[Path] = []
        defect_name = "unknown"
        if defect_dirs:
            chosen = defect_dirs[0]
            defect_name = chosen.name
            bad_imgs = sorted(
                p for p in chosen.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS
            )[:bad_per_cat]

            for p in bad_imgs:
                copy_file(p, output_root / cat / "test" / defect_name / p.name)
                # Also copy the ground-truth mask if available
                mask = _resolve_mask(gt_dir, defect_name, p.name)
                if mask is not None:
                    copy_file(mask, output_root / cat / "ground_truth" / defect_name / mask.name)

        n_good = len(good_imgs)
        n_bad = len(bad_imgs)
        manifest["categories"][cat] = {
            "defect_type": defect_name,
            "good": n_good,
            "anomalous": n_bad,
            "total": n_good + n_bad,
        }
        manifest["totals"]["images"] += n_good + n_bad
        manifest["totals"]["good"] += n_good
        manifest["totals"]["anomalous"] += n_bad
        print(f"  {cat}: {n_good} good + {n_bad} anomalous ({defect_name})")

    return manifest


def _resolve_mask(gt_root: Path, defect_type: str, image_name: str) -> Path | None:
    """Try to find the GT mask for a given test image."""
    base = gt_root / defect_type / image_name
    if base.exists():
        return base
    png = base.with_suffix(".png")
    if png.exists():
        return png
    stem_mask = gt_root / defect_type / f"{Path(image_name).stem}_mask.png"
    if stem_mask.exists():
        return stem_mask
    return None


def build_bundle(
    existing_bundle: Path,
    results_json: Path,
    output_bundle: Path,
) -> None:
    """
    Copy the existing model bundle (coreset + config) and enrich its
    metadata with the full experiment results.
    """
    if output_bundle.exists():
        shutil.rmtree(output_bundle, onerror=_force_remove_readonly)
    output_bundle.mkdir(parents=True)

    # Copy coreset
    src_coreset = existing_bundle / "coreset_global.npy"
    if not src_coreset.exists():
        raise FileNotFoundError(f"Coreset not found at {src_coreset}")
    shutil.copy2(src_coreset, output_bundle / "coreset_global.npy")

    # Load existing metadata and experiment results
    with (existing_bundle / "metadata.json").open("r") as f:
        metadata = json.load(f)

    with results_json.open("r") as f:
        results = json.load(f)

    # Enrich metadata with full results
    metadata["training_summary"] = {
        "final_mean_auroc": results["final_mean_auroc"],
        "avg_incremental_auroc": results["avg_incremental_auroc"],
        "forgetting_rate": results["forgetting_rate"],
        "forward_transfer": results.get("forward_transfer", 0.0),
        "total_time_seconds": results.get("total_time_seconds", 0.0),
    }
    metadata["per_category_results"] = results.get("per_category_final", {})
    metadata["coreset_stats"] = results.get("coreset_stats", {})

    with (output_bundle / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Bundle saved to {output_bundle}")
    print(f"  Coreset: {src_coreset.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a custom demo package for MemoryAD judges"
    )
    parser.add_argument(
        "--dataset-root", default="data/mvtec_ad",
        help="Root of the MVTec AD dataset",
    )
    parser.add_argument(
        "--results-json", default="results/E1_mvtec_5task/results.json",
        help="Path to the experiment results JSON",
    )
    parser.add_argument(
        "--existing-bundle", default="exports/mvtec_full_cpu/model_bundle",
        help="Path to the existing model bundle with coreset_global.npy",
    )
    parser.add_argument(
        "--output", default="custom_demo",
        help="Output directory for the demo package",
    )
    parser.add_argument(
        "--good-per-cat", type=int, default=4,
        help="Number of good images per category in the test set",
    )
    parser.add_argument(
        "--bad-per-cat", type=int, default=4,
        help="Number of anomalous images per category in the test set",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MemoryAD — Custom Demo Package Builder")
    print("=" * 60)

    # Step 1: Copy and enrich model bundle
    print("\n[1/3] Building model bundle...")
    build_bundle(
        existing_bundle=Path(args.existing_bundle),
        results_json=Path(args.results_json),
        output_bundle=output / "model_bundle",
    )

    # Step 2: Curate test images
    print("\n[2/3] Curating test images for all categories...")
    manifest = curate_test_images(
        dataset_root=Path(args.dataset_root),
        output_root=output / "test_data",
        categories=ALL_CATEGORIES,
        good_per_cat=args.good_per_cat,
        bad_per_cat=args.bad_per_cat,
    )

    # Save manifest
    with (output / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    # Step 3: Summary
    print(f"\n[3/3] Demo package created at: {output.resolve()}")
    print(f"  Model bundle:  {output / 'model_bundle'}")
    print(f"  Test data:     {output / 'test_data'}")
    print(f"  Total images:  {manifest['totals']['images']}")
    print(f"    Good:        {manifest['totals']['good']}")
    print(f"    Anomalous:   {manifest['totals']['anomalous']}")
    print("=" * 60)
    print("Done! Next: python scripts/custom_demo_server.py")


if __name__ == "__main__":
    main()
