"""
LwF Baseline — Learning without Forgetting + RD4AD.

Before each new task, saves the current decoder as a frozen teacher.
During training on the new task, adds a distillation loss that forces
the student decoder to match the teacher's outputs on the new data.

Reference: Li & Hoiem, "Learning without Forgetting", TPAMI 2018.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, List

from .rd4ad import RD4AD
from ..data_utils.dataset import AnomalyDataset, get_category_dataloaders
from ..evaluation.metrics import compute_auroc


class LwFBaseline:
    """LwF + RD4AD for continual anomaly detection."""

    def __init__(self, config: dict, alpha: float = 1.0, temperature: float = 2.0, epochs: int = 50):
        self.config = config
        self.alpha = alpha
        self.temperature = temperature
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create RD4AD model
        self.model = RD4AD(device=self.device)

        # Teacher (frozen old model)
        self.teacher_decoder = None

    def _distillation_loss(self, model, enc_feats, dec_feats):
        """Knowledge distillation loss against frozen teacher."""
        if self.teacher_decoder is None:
            return torch.tensor(0.0, device=self.device)

        # Get teacher outputs (no grad)
        with torch.no_grad():
            teacher_feats = self.teacher_decoder(enc_feats[-1])

        # MSE distillation between student and teacher decoder outputs
        dist_loss = torch.tensor(0.0, device=self.device)
        for student_f, teacher_f in zip(dec_feats, teacher_feats):
            if student_f.shape != teacher_f.shape:
                teacher_f = F.interpolate(
                    teacher_f, size=student_f.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            dist_loss = dist_loss + F.mse_loss(student_f, teacher_f)

        return self.alpha * dist_loss

    def run(
        self,
        dataset_root: str,
        tasks: List[dict],
        all_categories: List[str],
        dataset_type: str = "mvtec_ad",
    ) -> dict:
        """Run LwF+RD4AD across sequential tasks."""
        input_size = 256
        batch_size = 8
        n_tasks = len(tasks)
        n_cats = len(all_categories)
        auroc_matrix = np.full((n_tasks, n_cats), np.nan)
        categories_seen = []

        for task_idx, task in enumerate(tasks):
            cats = task["categories"]
            print(f"[LwF] Task {task_idx}: {cats}")

            # Save teacher before training on new task
            if task_idx > 0:
                self.teacher_decoder = copy.deepcopy(self.model.decoder)
                self.teacher_decoder.eval()
                for p in self.teacher_decoder.parameters():
                    p.requires_grad = False

            # Create combined loader for this task
            datasets = []
            for cat in cats:
                ds = AnomalyDataset(
                    root=dataset_root, category=cat, split="train",
                    dataset_type=dataset_type, input_size=input_size,
                )
                datasets.append(ds)

            combined_ds = ConcatDataset(datasets)
            train_loader = DataLoader(combined_ds, batch_size=batch_size, shuffle=True, num_workers=0)

            # Train with distillation loss
            print(f"  Training RD4AD with LwF (alpha={self.alpha})...", flush=True)
            self.model.train_on_loader(
                train_loader, epochs=self.epochs,
                extra_loss_fn=self._distillation_loss if task_idx > 0 else None,
            )

            categories_seen.extend(cats)

            # Evaluate on all seen categories
            print(f"  Evaluating on {len(categories_seen)} categories...", flush=True)
            for cat in categories_seen:
                cat_idx = all_categories.index(cat)
                _, test_loader = get_category_dataloaders(
                    root=dataset_root, category=cat,
                    dataset_type=dataset_type,
                    input_size=input_size, batch_size=batch_size,
                )
                auroc = self.model.evaluate(test_loader)
                auroc_matrix[task_idx, cat_idx] = auroc
                print(f"  {cat}: I-AUROC = {auroc:.4f}", flush=True)

        valid = ~np.isnan(auroc_matrix[-1])
        mean_auroc = float(np.mean(auroc_matrix[-1, valid]))
        print(f"[LwF] Final Mean I-AUROC: {mean_auroc:.4f}", flush=True)

        return {
            "method": "lwf_rd4ad",
            "auroc_matrix": auroc_matrix.tolist(),
            "mean_auroc": mean_auroc,
            "per_category": {
                cat: float(auroc_matrix[-1, all_categories.index(cat)])
                for cat in categories_seen
                if not np.isnan(auroc_matrix[-1, all_categories.index(cat)])
            },
        }
