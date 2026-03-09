"""Test data loader with real MVTec AD data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils.dataset import AnomalyDataset, get_category_dataloaders

DATASET_ROOT = "data/mvtec_ad"
CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
              "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
              "transistor", "wood", "zipper"]

print("Testing MVTec AD data loader...")
print(f"{'Category':<15} {'Train':>6} {'Test':>6} {'Anomalous':>10}")
print("-" * 40)

for cat in CATEGORIES:
    train_ds = AnomalyDataset(root=DATASET_ROOT, category=cat, split="train", input_size=518)
    test_ds = AnomalyDataset(root=DATASET_ROOT, category=cat, split="test", input_size=518)
    n_anomalous = sum(test_ds.labels)
    print(f"{cat:<15} {len(train_ds):>6} {len(test_ds):>6} {n_anomalous:>10}")

# Test loading one batch
print("\nLoading one batch from 'bottle'...")
train_loader, test_loader = get_category_dataloaders(
    root=DATASET_ROOT, category="bottle", input_size=518, batch_size=4
)
batch = next(iter(train_loader))
print(f"  Image shape: {batch['image'].shape}")
print(f"  Mask shape:  {batch['mask'].shape}")
print(f"  Labels:      {batch['label']}")
print(f"  Category:    {batch['category']}")

print("\n=== DATA LOADER TEST PASSED ===")
