"""
Reorganize VisA dataset from flat CSV-based layout to MVTec-style directory structure.

Before: data/visa/<category>/Data/Images/{Normal,Anomaly}/*.JPG
After:  data/visa/<category>/train/good/*.JPG
                             test/good/*.JPG
                             test/bad/*.JPG
                             ground_truth/bad/*.png

Uses split_csv/1cls.csv for train/test split assignment.

Usage:
    .venv\\Scripts\\python.exe scripts\\reorganize_visa.py
"""
import csv
import shutil
import sys
import os
from pathlib import Path
from collections import defaultdict

VISA_ROOT = Path("data/visa")
SPLIT_CSV = VISA_ROOT / "split_csv" / "1cls.csv"

CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum",
    "fryum", "macaroni1", "macaroni2",
    "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


def main():
    if not SPLIT_CSV.exists():
        print(f"ERROR: Split CSV not found: {SPLIT_CSV}")
        sys.exit(1)

    # Parse CSV
    entries = defaultdict(list)  # category -> [(split, label, image_path, mask_path)]
    with open(SPLIT_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["object"]
            split = row["split"]     # train or test
            label = row["label"]     # normal or anomaly
            image = row["image"]     # relative path like candle/Data/Images/Normal/0836.JPG
            mask = row.get("mask", "").strip()  # may be empty
            entries[cat].append((split, label, image, mask))

    print(f"Parsed {sum(len(v) for v in entries.values())} entries across {len(entries)} categories\n")

    for cat in CATEGORIES:
        if cat not in entries:
            print(f"  {cat}: NOT FOUND in CSV, skipping")
            continue

        cat_dir = VISA_ROOT / cat

        # Check if already reorganized
        if (cat_dir / "train" / "good").exists():
            n_train = len(list((cat_dir / "train" / "good").glob("*")))
            if n_train > 0:
                print(f"  {cat}: already reorganized ({n_train} train images), skipping")
                continue

        # Create target directories
        (cat_dir / "train" / "good").mkdir(parents=True, exist_ok=True)
        (cat_dir / "test" / "good").mkdir(parents=True, exist_ok=True)
        (cat_dir / "test" / "bad").mkdir(parents=True, exist_ok=True)
        (cat_dir / "ground_truth" / "bad").mkdir(parents=True, exist_ok=True)

        n_train = n_test_good = n_test_bad = 0

        for split, label, image_rel, mask_rel in entries[cat]:
            # Source image path
            src_image = VISA_ROOT / image_rel
            if not src_image.exists():
                print(f"    WARNING: {src_image} not found")
                continue

            if split == "train":
                dst = cat_dir / "train" / "good" / src_image.name
                shutil.copy2(src_image, dst)
                n_train += 1
            elif split == "test":
                if label == "normal":
                    dst = cat_dir / "test" / "good" / src_image.name
                    shutil.copy2(src_image, dst)
                    n_test_good += 1
                else:  # anomaly
                    dst = cat_dir / "test" / "bad" / src_image.name
                    shutil.copy2(src_image, dst)
                    n_test_bad += 1

                    # Copy mask if available
                    if mask_rel:
                        src_mask = VISA_ROOT / mask_rel
                        if src_mask.exists():
                            dst_mask = cat_dir / "ground_truth" / "bad" / src_mask.name
                            shutil.copy2(src_mask, dst_mask)

        print(f"  {cat}: train={n_train}, test_good={n_test_good}, test_bad={n_test_bad}")

    print("\nDone! VisA is now in MVTec-compatible directory structure.")


if __name__ == "__main__":
    main()
