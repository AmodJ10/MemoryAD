"""
Joint Baseline — Upper Bound.

Processes ALL categories at once using the same DINOv2 + coreset + k-NN
pipeline. No incremental setup — this is the performance ceiling.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from tqdm import tqdm

from ..backbones.dinov2 import DINOv2Extractor
from ..coreset.greedy_coreset import greedy_coreset_selection
from ..scoring.knn_scorer import KNNScorer
from ..evaluation.metrics import compute_auroc
from ..data_utils.dataset import get_category_dataloaders


class JointBaseline:
    """Upper bound: build coreset from ALL categories simultaneously."""

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
        all_categories: List[str],
        dataset_type: str = "mvtec_ad",
    ) -> dict:
        """
        Run joint baseline: extract all features, build single coreset, evaluate.

        Returns: dict with per-category I-AUROC and mean I-AUROC.
        """
        self._ensure_backbone()
        input_size = self.config["backbone"].get("input_size", 518)
        batch_size = self.config["backbone"].get("batch_size", 4)
        spatial_dims = self.backbone.get_spatial_dims(input_size)

        # Extract features for ALL categories
        print("[Joint] Extracting features for all categories...")
        all_features = []
        for cat in all_categories:
            train_loader, _ = get_category_dataloaders(
                root=dataset_root, category=cat,
                dataset_type=dataset_type,
                input_size=input_size, batch_size=batch_size,
            )
            feats = self._extract_features(train_loader, desc=f"  {cat}")
            all_features.append(feats)
            print(f"  {cat}: {feats.shape[0]} patches")

        # Build single coreset from ALL features
        combined = np.concatenate(all_features, axis=0)
        print(f"[Joint] Building coreset: {combined.shape[0]} -> {self.budget} patches...")
        coreset = greedy_coreset_selection(combined, budget=self.budget)
        self.scorer.fit(coreset)

        # Evaluate on all categories
        print("[Joint] Evaluating...")
        results = {}
        for cat in all_categories:
            _, test_loader = get_category_dataloaders(
                root=dataset_root, category=cat,
                dataset_type=dataset_type,
                input_size=input_size, batch_size=batch_size,
            )
            auroc = self._evaluate_category(test_loader, spatial_dims)
            results[cat] = auroc
            print(f"  {cat}: I-AUROC = {auroc:.4f}")

        mean_auroc = np.mean(list(results.values()))
        print(f"[Joint] Mean I-AUROC: {mean_auroc:.4f}")

        return {
            "method": "joint",
            "per_category": results,
            "mean_auroc": float(mean_auroc),
            "joint_aurocs": np.array([results[c] for c in all_categories]),
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
