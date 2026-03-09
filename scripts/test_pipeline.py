"""
Phase 3 — End-to-end pipeline test on real MVTec AD data.

Runs a 2-task toy experiment:
  Task 0: bottle, cable (3 categories from roadmap, using 2 for speed)
  Task 1: carpet, grid

Tests: DINOv2 extraction → coreset building → k-NN scoring → AUROC.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.backbones.dinov2 import DINOv2Extractor
from src.coreset.adaptive_manager import AdaptiveCoresetManager
from src.scoring.knn_scorer import KNNScorer
from src.evaluation.metrics import compute_auroc
from src.data_utils.dataset import get_category_dataloaders


DATASET_ROOT = "data/mvtec_ad"
BUDGET = 5000
K = 9

# 2-task experiment for quick validation
TASKS = [
    {"task_id": 0, "categories": ["bottle", "cable"]},
    {"task_id": 1, "categories": ["carpet", "grid"]},
]


def extract_features(backbone, loader, desc=""):
    """Extract patch features from all images in a loader."""
    all_features = []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"]
        feats = backbone.extract(images)  # [B, P, D]
        B, P, D = feats.shape
        all_features.append(feats.reshape(B * P, D).cpu().numpy())
    return np.concatenate(all_features, axis=0)


def evaluate_category(backbone, scorer, test_loader, spatial_dims):
    """Evaluate anomaly detection on one category."""
    all_scores = []
    all_labels = []
    for batch in test_loader:
        images = batch["image"]
        labels = batch["label"].numpy()
        feats = backbone.extract(images).cpu().numpy()  # [B, P, D]
        B = feats.shape[0]
        for i in range(B):
            score, _ = scorer.score_image(feats[i], spatial_dims)
            all_scores.append(score)
            all_labels.append(labels[i])
    return compute_auroc(np.array(all_labels), np.array(all_scores))


def main():
    print("=" * 60)
    print("Phase 3 — End-to-End Pipeline Test")
    print("=" * 60)

    # Load backbone
    print("\n1. Loading DINOv2 ViT-B/14...")
    t0 = time.time()
    backbone = DINOv2Extractor(
        model_name="dinov2_vitb14", layers=[7, 11],
        aggregation="concat", use_fp16=True
    )
    spatial_dims = backbone.get_spatial_dims(518)
    print(f"   Loaded in {time.time()-t0:.1f}s. Feature dim: {backbone.feature_dim}")
    print(f"   Spatial dims: {spatial_dims} = {spatial_dims[0]*spatial_dims[1]} patches/image")
    print(f"   VRAM used: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

    # Create coreset manager and scorer
    manager = AdaptiveCoresetManager(global_budget=BUDGET, strategy="proportional")
    scorer = KNNScorer(k=K)

    # Track results
    all_categories = []
    for task in TASKS:
        all_categories.extend(task["categories"])
    n_tasks = len(TASKS)
    n_cats = len(all_categories)
    auroc_matrix = np.full((n_tasks, n_cats), np.nan)

    categories_seen = []

    for task_idx, task in enumerate(TASKS):
        cats = task["categories"]
        print(f"\n--- Task {task_idx}: {cats} ---")

        # Extract features for new categories
        task_features = {}
        for cat in cats:
            print(f"   Extracting '{cat}' features...")
            train_loader, _ = get_category_dataloaders(
                root=DATASET_ROOT, category=cat, input_size=518, batch_size=4
            )
            feats = extract_features(backbone, train_loader, desc=f"   {cat}")
            task_features[cat] = feats
            print(f"   -> {feats.shape[0]} patches, {feats.shape[1]}D")

        # Add to coreset
        info = manager.add_task(task_features)
        print(f"   Coreset: {info['total_patches']} patches, {info['total_categories']} cats")
        print(f"   Per-cat: {info['budget_allocation']}")

        categories_seen.extend(cats)

        # Build scorer
        scorer.fit(manager.get_global_coreset())

        # Evaluate all seen categories
        print(f"   Evaluating on {len(categories_seen)} categories...")
        for cat in categories_seen:
            cat_idx = all_categories.index(cat)
            _, test_loader = get_category_dataloaders(
                root=DATASET_ROOT, category=cat, input_size=518, batch_size=4
            )
            auroc = evaluate_category(backbone, scorer, test_loader, spatial_dims)
            auroc_matrix[task_idx, cat_idx] = auroc
            print(f"   {cat:>12}: I-AUROC = {auroc:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("AUROC Matrix (rows=tasks, cols=categories):")
    header = "         " + "".join(f"{c:>12}" for c in all_categories)
    print(header)
    for t in range(n_tasks):
        row = f"Task {t}:  "
        for c in range(n_cats):
            v = auroc_matrix[t, c]
            row += f"{v:>12.4f}" if not np.isnan(v) else "         N/A"
        print(row)

    # Final metrics
    valid = ~np.isnan(auroc_matrix[-1])
    final_mean = np.mean(auroc_matrix[-1, valid])
    print(f"\nFinal Mean I-AUROC: {final_mean:.4f}")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    print("=" * 60)

    if final_mean > 0.80:
        print("\n[PASS] Pipeline working correctly -- AUROC > 0.80")
    else:
        print("\n[WARN] AUROC below 0.80 -- may need investigation")


if __name__ == "__main__":
    main()
