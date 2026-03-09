"""
Naive Baseline — Lower Bound.

Processes tasks sequentially but REPLACES the entire coreset with only
the current task's features. Complete catastrophic forgetting.
"""

import numpy as np
import torch
from typing import Dict, List
from tqdm import tqdm

from ..backbones.dinov2 import DINOv2Extractor
from ..coreset.greedy_coreset import greedy_coreset_selection
from ..scoring.knn_scorer import KNNScorer
from ..evaluation.metrics import compute_auroc
from ..data_utils.dataset import get_category_dataloaders


class NaiveBaseline:
    """Lower bound: coreset contains only the latest task's features."""

    def __init__(self, config: dict):
        self.config = config
        self.backbone = None
        self.scorer = KNNScorer(
            k=config["scoring"]["k"],
            image_score_method=config["scoring"]["image_score_method"],
        )
        self.budget = config["coreset"]["global_budget"]

    def _ensure_backbone(self):
        if self.backbone is None:
            self.backbone = DINOv2Extractor(
                model_name=self.config["backbone"]["name"],
                layers=self.config["backbone"]["layers"],
                aggregation=self.config["backbone"].get("feature_aggregation", "concat"),
                use_fp16=self.config["backbone"].get("use_fp16", True),
            )

    def _extract_features(self, loader, desc=""):
        all_feats = []
        for batch in tqdm(loader, desc=desc, leave=False):
            feats = self.backbone.extract(batch["image"])
            B, P, D = feats.shape
            all_feats.append(feats.reshape(B * P, D).cpu().numpy())
        return np.concatenate(all_feats, axis=0)

    def run(
        self,
        dataset_root: str,
        tasks: List[dict],
        all_categories: List[str],
        dataset_type: str = "mvtec_ad",
    ) -> dict:
        """
        Run naive baseline: for each task, replace entire coreset.

        Returns: dict with AUROC matrix and final per-category results.
        """
        self._ensure_backbone()
        input_size = self.config["backbone"].get("input_size", 518)
        batch_size = self.config["backbone"].get("batch_size", 4)
        spatial_dims = self.backbone.get_spatial_dims(input_size)

        n_tasks = len(tasks)
        n_cats = len(all_categories)
        auroc_matrix = np.full((n_tasks, n_cats), np.nan)
        categories_seen = []

        for task_idx, task in enumerate(tasks):
            cats = task["categories"]
            print(f"[Naive] Task {task_idx}: {cats}")

            # Extract features ONLY for current task categories
            task_features = []
            for cat in cats:
                train_loader, _ = get_category_dataloaders(
                    root=dataset_root, category=cat,
                    dataset_type=dataset_type,
                    input_size=input_size, batch_size=batch_size,
                )
                feats = self._extract_features(train_loader, desc=f"  {cat}")
                task_features.append(feats)

            # Replace entire coreset with only current task
            combined = np.concatenate(task_features, axis=0)
            coreset = greedy_coreset_selection(combined, budget=self.budget)
            self.scorer.fit(coreset)

            categories_seen.extend(cats)

            # Evaluate on ALL seen categories
            for cat in categories_seen:
                cat_idx = all_categories.index(cat)
                _, test_loader = get_category_dataloaders(
                    root=dataset_root, category=cat,
                    dataset_type=dataset_type,
                    input_size=input_size, batch_size=batch_size,
                )
                auroc = self._evaluate_category(test_loader, spatial_dims)
                auroc_matrix[task_idx, cat_idx] = auroc
                print(f"  {cat}: I-AUROC = {auroc:.4f}")

        # Final results
        valid = ~np.isnan(auroc_matrix[-1])
        mean_auroc = float(np.mean(auroc_matrix[-1, valid]))
        print(f"[Naive] Final Mean I-AUROC: {mean_auroc:.4f}")

        return {
            "method": "naive",
            "auroc_matrix": auroc_matrix.tolist(),
            "mean_auroc": mean_auroc,
            "per_category": {
                cat: float(auroc_matrix[-1, all_categories.index(cat)])
                for cat in categories_seen
                if not np.isnan(auroc_matrix[-1, all_categories.index(cat)])
            },
        }

    def _evaluate_category(self, test_loader, spatial_dims) -> float:
        all_scores, all_labels = [], []
        for batch in test_loader:
            feats = self.backbone.extract(batch["image"]).cpu().numpy()
            labels = batch["label"].numpy()
            for i in range(feats.shape[0]):
                score, _ = self.scorer.score_image(feats[i], spatial_dims)
                all_scores.append(score)
                all_labels.append(labels[i])
        return compute_auroc(np.array(all_labels), np.array(all_scores))
