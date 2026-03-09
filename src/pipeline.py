"""
MemoryAD Pipeline -- Main experiment orchestrator.

Runs the full incremental anomaly detection pipeline:
1. Load config & task splits
2. For each task:
   a. Load features from cache (or extract live)
   b. Add to adaptive coreset manager
   c. Evaluate on ALL seen categories
3. Compute continual learning metrics
4. Save results

Supports two modes:
  - Cached mode (fast): loads pre-computed features from disk
  - Live mode (slow):   extracts features on-the-fly with DINOv2
"""

import os
import json
import time
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from .backbones.dinov2 import DINOv2Extractor
from .backbones.clip_backbone import CLIPExtractor
from .backbones.wideresnet import WideResNetExtractor
from .coreset.adaptive_manager import AdaptiveCoresetManager
from .scoring.knn_scorer import KNNScorer
from .evaluation.metrics import (
    compute_auroc,
    compute_pixel_auroc,
    compute_forgetting_rate,
    summarise_results,
)
from .data_utils.dataset import get_category_dataloaders
from .data_utils.feature_cache import FeatureCache


def create_backbone(config: dict):
    """Create feature extractor from config."""
    name = config["backbone"]["name"]
    layers = config["backbone"]["layers"]
    aggregation = config["backbone"].get("feature_aggregation", "concat")
    use_fp16 = config["backbone"].get("use_fp16", True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "dinov2" in name:
        return DINOv2Extractor(
            model_name=name, layers=layers,
            aggregation=aggregation, use_fp16=use_fp16, device=device,
        )
    elif "clip" in name.lower():
        return CLIPExtractor(
            model_name="ViT-L-14", layers=layers,
            aggregation=aggregation, use_fp16=use_fp16, device=device,
        )
    elif "resnet" in name.lower() or "wide" in name.lower():
        layer_names = [f"layer{l}" for l in layers]
        return WideResNetExtractor(
            layers=layer_names,
            aggregation=aggregation, use_fp16=use_fp16, device=device,
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")


def extract_features_for_category(
    backbone,
    data_loader,
    desc: str = "",
) -> np.ndarray:
    """
    Extract patch features for all images in a data loader.

    Returns: [total_patches, D] numpy array.
    """
    all_features = []

    for batch in tqdm(data_loader, desc=desc, leave=False):
        images = batch["image"]
        features = backbone.extract(images)  # [B, P, D]
        # Flatten batch: [B, P, D] -> [B*P, D]
        B, P, D = features.shape
        features = features.reshape(B * P, D)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def evaluate_on_category_cached(
    scorer: KNNScorer,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    spatial_dims,
) -> dict:
    """
    Evaluate anomaly detection using cached test features.

    Args:
        scorer: fitted KNNScorer
        test_features: [N_test, P, D] per-image patch features
        test_labels: [N_test] binary labels
        spatial_dims: (H, W) patch grid dimensions

    Returns: dict with I-AUROC.
    """
    # Use batch scoring for speed
    image_scores, _ = scorer.score_batch(test_features, spatial_dims)
    i_auroc = compute_auroc(test_labels, image_scores)
    return {"i_auroc": i_auroc, "p_auroc": 0.0}


def evaluate_on_category(
    backbone,
    scorer: KNNScorer,
    test_loader,
    spatial_dims,
) -> dict:
    """
    Evaluate anomaly detection on a single category's test set (live mode).

    Returns: dict with I-AUROC, P-AUROC, and per-image scores.
    """
    all_image_scores = []
    all_labels = []
    all_anomaly_maps = []
    all_gt_masks = []

    for batch in test_loader:
        images = batch["image"]
        labels = batch["label"].numpy()
        masks = batch["mask"].numpy()  # [B, 1, H, W]

        # Extract features
        features = backbone.extract(images)  # [B, P, D]
        features_np = features.cpu().numpy()

        # Score each image
        image_scores, anomaly_maps = scorer.score_batch(features_np, spatial_dims)

        all_image_scores.extend(image_scores.tolist())
        all_labels.extend(labels.tolist())

        if anomaly_maps is not None:
            all_anomaly_maps.append(anomaly_maps)
            all_gt_masks.append(masks[:, 0, :, :])  # Remove channel dim

    all_labels = np.array(all_labels)
    all_image_scores = np.array(all_image_scores)

    # Image-level AUROC
    i_auroc = compute_auroc(all_labels, all_image_scores)

    # Pixel-level AUROC (if anomaly maps available and we have anomalous images)
    p_auroc = 0.0
    if all_anomaly_maps and np.any(all_labels == 1):
        gt_masks = np.concatenate(all_gt_masks, axis=0)
        anomaly_maps = np.concatenate(all_anomaly_maps, axis=0)

        # Resize anomaly maps to match GT mask size if needed
        if anomaly_maps.shape[1:] != gt_masks.shape[1:]:
            from scipy.ndimage import zoom
            scale_h = gt_masks.shape[1] / anomaly_maps.shape[1]
            scale_w = gt_masks.shape[2] / anomaly_maps.shape[2]
            resized_maps = []
            for amap in anomaly_maps:
                resized_maps.append(zoom(amap, (scale_h, scale_w), order=1))
            anomaly_maps = np.array(resized_maps)

        p_auroc = compute_pixel_auroc(gt_masks, anomaly_maps)

    return {
        "i_auroc": i_auroc,
        "p_auroc": p_auroc,
    }


class MemoryADPipeline:
    """
    Main experimental pipeline for MemoryAD.

    Orchestrates the full incremental anomaly detection workflow:
    sequential task learning -> coreset management -> evaluation.

    Supports two modes:
    - use_cache=True (default): loads features from FeatureCache (fast)
    - use_cache=False: extracts features live with backbone (slow)
    """

    def __init__(
        self,
        config: dict,
        task_config: dict,
        output_dir: str = "results",
        use_cache: bool = True,
        feature_dir: str = "data/features/dinov2_vitb14",
        coreset_cache_dir: str = "data/coreset_cache",
    ):
        self.config = config
        self.task_config = task_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

        # Feature cache
        self.cache = None
        if use_cache:
            self.cache = FeatureCache(feature_dir)
            self.spatial_dims = self.cache.spatial_dims
            self.backbone = None  # Don't load backbone if using cache
            print(f"Using cached features from {feature_dir}")
            print(f"  Spatial dims: {self.spatial_dims}, Feature dim: {self.cache.feature_dim}")
        else:
            # Create backbone (loads model into GPU)
            print(f"Loading backbone: {config['backbone']['name']}...")
            self.backbone = create_backbone(config)
            input_size = config["backbone"].get("input_size", 518)
            self.spatial_dims = self.backbone.get_spatial_dims(input_size)

        # Create coreset manager with disk cache
        self.manager = AdaptiveCoresetManager(
            global_budget=config["coreset"]["global_budget"],
            strategy=config["coreset"]["strategy"],
            min_per_category=config["coreset"]["min_per_category"],
            coreset_cache_dir=coreset_cache_dir,
        )

        # Create scorer
        self.scorer = KNNScorer(
            k=config["scoring"]["k"],
            image_score_method=config["scoring"]["image_score_method"],
        )

        # Dataset config
        self.dataset_name = task_config["dataset"]["name"]
        self.dataset_root = task_config["dataset"]["root"]
        self.tasks = task_config["tasks"]

        # Results storage
        self.all_categories = []
        for task in self.tasks:
            self.all_categories.extend(task["categories"])

        n_tasks = len(self.tasks)
        n_cats = len(self.all_categories)
        self.auroc_matrix = np.full((n_tasks, n_cats), np.nan)

    def _load_train_features(self, category: str) -> np.ndarray:
        """Load train features from cache or extract live."""
        if self.use_cache and self.cache is not None:
            return self.cache.load_train_features(category)
        else:
            train_loader, _ = get_category_dataloaders(
                root=self.dataset_root,
                category=category,
                dataset_type=self.dataset_name,
                input_size=self.config["backbone"].get("input_size", 518),
                batch_size=self.config["backbone"].get("batch_size", 4),
            )
            return extract_features_for_category(
                self.backbone, train_loader, desc=f"  {category}"
            )

    def _evaluate_category(self, category: str) -> dict:
        """Evaluate on a category using cache or live extraction."""
        if self.use_cache and self.cache is not None:
            test_features, test_labels = self.cache.load_test_data(category)
            return evaluate_on_category_cached(
                self.scorer, test_features, test_labels, self.spatial_dims
            )
        else:
            _, test_loader = get_category_dataloaders(
                root=self.dataset_root,
                category=category,
                dataset_type=self.dataset_name,
                input_size=self.config["backbone"].get("input_size", 518),
                batch_size=self.config["backbone"].get("batch_size", 4),
            )
            return evaluate_on_category(
                self.backbone, self.scorer, test_loader, self.spatial_dims
            )

    def run(self) -> dict:
        """Run the full incremental experiment."""
        print(f"\n{'='*60}")
        print(f"MemoryAD Experiment")
        print(f"Dataset: {self.dataset_name}")
        print(f"Tasks: {len(self.tasks)}")
        print(f"Categories: {len(self.all_categories)}")
        print(f"Coreset budget: {self.config['coreset']['global_budget']}")
        mode = "cached" if self.use_cache else "live"
        print(f"Mode: {mode}")
        print(f"{'='*60}\n")

        categories_seen = []
        total_time = 0

        for task_idx, task in enumerate(self.tasks):
            task_categories = task["categories"]
            print(f"\n--- Task {task_idx} ---")
            print(f"New categories: {task_categories}")

            t_start = time.time()

            # Step 1: Load/extract features for new categories
            task_features = {}
            for cat in task_categories:
                features = self._load_train_features(cat)
                task_features[cat] = features
                print(f"  {cat}: {features.shape[0]} patches, dim={features.shape[1]}")

            # Step 2: Add to coreset manager
            info = self.manager.add_task(task_features)
            print(f"  Coreset: {info['total_patches']} patches across {info['total_categories']} categories")

            categories_seen.extend(task_categories)

            # Step 3: Build scorer with updated coreset
            global_coreset = self.manager.get_global_coreset()
            self.scorer.fit(global_coreset)

            # Step 4: Evaluate on ALL seen categories
            print(f"  Evaluating on {len(categories_seen)} categories...")
            for cat in categories_seen:
                cat_idx = self.all_categories.index(cat)
                result = self._evaluate_category(cat)
                self.auroc_matrix[task_idx, cat_idx] = result["i_auroc"]
                print(f"    {cat}: I-AUROC={result['i_auroc']:.4f}")

            t_elapsed = time.time() - t_start
            total_time += t_elapsed
            print(f"  Task {task_idx} completed in {t_elapsed:.1f}s")

        # Compute continual learning metrics
        joint_aurocs = np.nanmax(self.auroc_matrix, axis=0)
        results = summarise_results(
            self.auroc_matrix, joint_aurocs, self.all_categories
        )
        results["total_time_seconds"] = total_time
        results["auroc_matrix"] = self.auroc_matrix.tolist()
        results["coreset_stats"] = self.manager.get_stats()

        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Final Mean I-AUROC: {results['final_mean_auroc']:.4f}")
        print(f"Avg Incremental I-AUROC: {results['avg_incremental_auroc']:.4f}")
        print(f"Forgetting Rate: {results['forgetting_rate']:.4f}")
        print(f"Forward Transfer: {results['forward_transfer']:.4f}")
        print(f"Total Time: {total_time:.1f}s")
        print(f"{'='*60}\n")

        # Save results
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {results_path}")

        return results
