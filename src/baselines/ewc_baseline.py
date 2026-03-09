"""
EWC Baseline — Elastic Weight Consolidation + RD4AD.

After each task, computes the Fisher Information matrix (diagonal approx)
on the current task's data and adds a regularisation penalty to prevent
important weights from changing when learning new tasks.

Reference: Kirkpatrick et al., "Overcoming Catastrophic Forgetting in
Neural Networks", PNAS 2017.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, List

from .rd4ad import RD4AD
from ..data_utils.dataset import AnomalyDataset, get_category_dataloaders
from ..evaluation.metrics import compute_auroc


class EWCBaseline:
    """EWC + RD4AD for continual anomaly detection."""

    def __init__(self, config: dict, ewc_lambda: float = 5000.0, epochs: int = 50):
        self.config = config
        self.ewc_lambda = ewc_lambda
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create RD4AD model
        self.model = RD4AD(device=self.device)

        # EWC storage
        self.fisher_matrices = []  # List of Fisher dicts per task
        self.old_params = []        # List of param snapshot dicts per task

    def _compute_fisher(self, data_loader: DataLoader):
        """Compute diagonal Fisher Information on the given data."""
        self.model.decoder.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.decoder.named_parameters() if p.requires_grad}

        n_samples = 0
        for batch in data_loader:
            images = batch["image"].to(self.device)
            self.model.decoder.zero_grad()

            enc_feats, dec_feats = self.model(images)
            loss = self.model.compute_loss(enc_feats, dec_feats)
            loss.backward()

            for n, p in self.model.decoder.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            n_samples += images.size(0)

        # Average
        for n in fisher:
            fisher[n] /= max(n_samples, 1)

        return fisher

    def _ewc_loss(self, model, enc_feats, dec_feats):
        """EWC regularisation penalty."""
        if not self.fisher_matrices:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for fisher, old_params in zip(self.fisher_matrices, self.old_params):
            for n, p in model.decoder.named_parameters():
                if n in fisher:
                    penalty = penalty + (fisher[n] * (p - old_params[n]) ** 2).sum()

        return self.ewc_lambda * penalty

    def run(
        self,
        dataset_root: str,
        tasks: List[dict],
        all_categories: List[str],
        dataset_type: str = "mvtec_ad",
    ) -> dict:
        """Run EWC+RD4AD across sequential tasks."""
        input_size = 256  # RD4AD uses 256x256 (WideResNet native)
        batch_size = 8
        n_tasks = len(tasks)
        n_cats = len(all_categories)
        auroc_matrix = np.full((n_tasks, n_cats), np.nan)
        categories_seen = []

        for task_idx, task in enumerate(tasks):
            cats = task["categories"]
            print(f"[EWC] Task {task_idx}: {cats}", flush=True)

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

            # Train with EWC penalty
            print(f"  Training RD4AD with EWC (lambda={self.ewc_lambda})...", flush=True)
            self.model.train_on_loader(
                train_loader, epochs=self.epochs,
                extra_loss_fn=self._ewc_loss if task_idx > 0 else None,
            )

            # Compute Fisher and save params for this task
            import time as _time
            t_fisher = _time.time()
            print("  Computing Fisher Information...", flush=True)
            fisher = self._compute_fisher(train_loader)
            print(f"  Fisher done in {_time.time()-t_fisher:.1f}s", flush=True)
            self.fisher_matrices.append(fisher)
            self.old_params.append(
                {n: p.data.clone() for n, p in self.model.decoder.named_parameters() if p.requires_grad}
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
        print(f"[EWC] Final Mean I-AUROC: {mean_auroc:.4f}", flush=True)

        return {
            "method": "ewc_rd4ad",
            "auroc_matrix": auroc_matrix.tolist(),
            "mean_auroc": mean_auroc,
            "per_category": {
                cat: float(auroc_matrix[-1, all_categories.index(cat)])
                for cat in categories_seen
                if not np.isnan(auroc_matrix[-1, all_categories.index(cat)])
            },
        }
