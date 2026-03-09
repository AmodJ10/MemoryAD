"""
Replay Baseline — Experience Replay + RD4AD.

Stores N normal images per old category in a replay buffer. When training
on a new task, mixes new data with replayed old data to maintain
performance on previously seen categories.

Reference: Chaudhry et al., "Continual Learning with Tiny Episodic
Memories", 2019.
"""

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from typing import Dict, List

from .rd4ad import RD4AD
from ..data_utils.dataset import AnomalyDataset, get_category_dataloaders
from ..evaluation.metrics import compute_auroc


class ReplayBaseline:
    """Replay + RD4AD for continual anomaly detection."""

    def __init__(self, config: dict, buffer_per_category: int = 10, epochs: int = 50):
        self.config = config
        self.buffer_per_category = buffer_per_category
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create RD4AD model
        self.model = RD4AD(device=self.device)

        # Replay buffer: list of (category_name, dataset_subset) pairs
        self.replay_datasets = []

    def _select_buffer_samples(self, dataset, n: int):
        """Randomly select n samples from a dataset for the replay buffer."""
        if len(dataset) <= n:
            return dataset
        indices = np.random.choice(len(dataset), size=n, replace=False)
        return Subset(dataset, indices.tolist())

    def run(
        self,
        dataset_root: str,
        tasks: List[dict],
        all_categories: List[str],
        dataset_type: str = "mvtec_ad",
    ) -> dict:
        """Run Replay+RD4AD across sequential tasks."""
        input_size = 256
        batch_size = 8
        n_tasks = len(tasks)
        n_cats = len(all_categories)
        auroc_matrix = np.full((n_tasks, n_cats), np.nan)
        categories_seen = []

        for task_idx, task in enumerate(tasks):
            cats = task["categories"]
            print(f"[Replay] Task {task_idx}: {cats}", flush=True)

            # Create datasets for current task
            new_datasets = []
            for cat in cats:
                ds = AnomalyDataset(
                    root=dataset_root, category=cat, split="train",
                    dataset_type=dataset_type, input_size=input_size,
                )
                new_datasets.append(ds)

                # Add samples to replay buffer for future tasks
                buffer = self._select_buffer_samples(ds, self.buffer_per_category)
                self.replay_datasets.append(buffer)

            # Combine new data + replay buffer
            all_datasets = new_datasets.copy()
            if task_idx > 0:
                # Add replay data from previous tasks
                # (exclude current task's buffer entries which are the last len(cats) entries)
                old_replays = self.replay_datasets[:-len(cats)]
                all_datasets.extend(old_replays)
                n_replay = sum(len(d) for d in old_replays)
                print(f"  Replay buffer: {n_replay} images from {len(old_replays)} old categories", flush=True)

            combined_ds = ConcatDataset(all_datasets)
            train_loader = DataLoader(combined_ds, batch_size=batch_size, shuffle=True, num_workers=0)

            # Train
            print(f"  Training RD4AD with replay ({len(combined_ds)} total images)...", flush=True)
            self.model.train_on_loader(train_loader, epochs=self.epochs)

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
        print(f"[Replay] Final Mean I-AUROC: {mean_auroc:.4f}", flush=True)

        return {
            "method": "replay_rd4ad",
            "auroc_matrix": auroc_matrix.tolist(),
            "mean_auroc": mean_auroc,
            "per_category": {
                cat: float(auroc_matrix[-1, all_categories.index(cat)])
                for cat in categories_seen
                if not np.isnan(auroc_matrix[-1, all_categories.index(cat)])
            },
        }
