"""
Adaptive Coreset Manager — the core novel component of MemoryAD.

Manages a fixed-budget global coreset that incrementally grows as new product
categories arrive. When a new task is added, the manager:
1. Extracts a coreset for the new categories
2. Re-allocates the global budget across all seen categories
3. Compresses older coresets to fit the new budget allocation
4. Merges everything into a single global coreset
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from .greedy_coreset import greedy_coreset_selection, random_coreset_selection


class AdaptiveCoresetManager:
    """
    Manages a fixed-budget global coreset across sequential tasks.

    The coreset is a collection of patch-level features from normal images
    of all product categories seen so far. When new categories arrive, the
    manager redistributes the budget and compresses older entries.

    This is inherently immune to catastrophic forgetting because:
    - No model weights are updated
    - Old category patches are preserved (possibly compressed) in the coreset
    - Anomaly detection uses nearest-neighbour scoring against the coreset
    """

    def __init__(
        self,
        global_budget: int = 10000,
        strategy: str = "proportional",
        min_per_category: int = 100,
        selection_method: str = "greedy",
        seed: int = 42,
        coreset_cache_dir: Optional[str] = None,
    ):
        """
        Args:
            global_budget:    Maximum total patches in the coreset.
            strategy:         Budget allocation strategy:
                              "proportional" — equal share per category
                              "weighted"     — proportional to training set size
                              "recency"      — newer categories get 1.5× weight
            min_per_category: Minimum patches per category.
            selection_method: "greedy" (k-center) or "random".
            seed:             Random seed for reproducibility.
            coreset_cache_dir: Optional directory for caching coreset results.
                               If set, coresets are saved/loaded as .npy files
                               keyed by (category, budget, method, seed).
        """
        self.global_budget = global_budget
        self.strategy = strategy
        self.min_per_category = min_per_category
        self.selection_method = selection_method
        self.seed = seed

        # O2: Disk cache for coreset results
        self.cache_dir = None
        if coreset_cache_dir:
            self.cache_dir = Path(coreset_cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.category_coresets: Dict[str, np.ndarray] = {}  # {category: [n, D]}
        self.category_sizes: Dict[str, int] = {}  # Original training set sizes
        self.task_history: List[List[str]] = []  # Which categories per task

    @property
    def categories_seen(self) -> List[str]:
        """All categories seen so far."""
        return list(self.category_coresets.keys())

    @property
    def total_patches(self) -> int:
        """Current total number of patches in the coreset."""
        return sum(c.shape[0] for c in self.category_coresets.values())

    @property
    def feature_dim(self) -> int:
        """Feature dimension (D)."""
        if self.category_coresets:
            return next(iter(self.category_coresets.values())).shape[1]
        return 0

    def add_task(self, task_features: Dict[str, np.ndarray]) -> dict:
        """
        Add a new task's categories to the coreset.

        Args:
            task_features: {category_name: [N_patches, D]} dict of patch
                           features for each new category in this task.

        Returns:
            info: Dict with budget allocation details and stats.
        """
        new_categories = list(task_features.keys())
        self.task_history.append(new_categories)

        # Store original sizes for weighted allocation
        for cat, feats in task_features.items():
            self.category_sizes[cat] = feats.shape[0]

        # Step 1: Compute new budget allocation
        all_categories = self.categories_seen + [c for c in new_categories if c not in self.category_coresets]
        budget_allocation = self._allocate_budget(all_categories)

        # Step 2: Build coresets for new categories
        for cat in new_categories:
            cat_budget = budget_allocation[cat]
            cat_features = task_features[cat]

            # O2: Check disk cache first
            dim = cat_features.shape[1]
            cached = self._load_cached_coreset(cat, cat_budget, dim)
            if cached is not None:
                self.category_coresets[cat] = cached
            elif self.selection_method == "greedy":
                coreset = greedy_coreset_selection(
                    cat_features, cat_budget, seed=self.seed
                )
                self.category_coresets[cat] = coreset
                self._save_cached_coreset(cat, cat_budget, dim, coreset)
            else:
                coreset = random_coreset_selection(
                    cat_features, cat_budget, seed=self.seed
                )
                self.category_coresets[cat] = coreset
                self._save_cached_coreset(cat, cat_budget, dim, coreset)

        # Step 3: Compress older categories to fit new allocation
        for cat in self.categories_seen:
            if cat in new_categories:
                continue  # Already handled above

            cat_budget = budget_allocation[cat]
            current_size = self.category_coresets[cat].shape[0]

            if current_size > cat_budget:
                # O3: Truncate instead of recompressing.
                # The first K points of a greedy coreset are already a good
                # K-center subset (selected in order of max coverage).
                self.category_coresets[cat] = self.category_coresets[cat][:cat_budget]
            # If current_size <= cat_budget, keep all (no expansion for old categories)

        # Build info dict
        info = {
            "task_id": len(self.task_history) - 1,
            "new_categories": new_categories,
            "total_categories": len(self.categories_seen),
            "budget_allocation": {cat: self.category_coresets[cat].shape[0] for cat in self.categories_seen},
            "total_patches": self.total_patches,
            "global_budget": self.global_budget,
        }

        return info

    def _allocate_budget(self, categories: List[str]) -> Dict[str, int]:
        """
        Allocate the global budget across all categories.

        Returns:
            Dict[str, int] mapping category name → allocated budget.
        """
        n_cats = len(categories)

        # Ensure min_per_category doesn't violate global_budget
        effective_min = min(self.min_per_category, self.global_budget // n_cats)

        if self.strategy == "proportional":
            # Equal share per category
            base = self.global_budget // n_cats
            allocation = {cat: max(base, effective_min) for cat in categories}

        elif self.strategy == "weighted":
            # Proportional to original training set size
            total_size = sum(self.category_sizes.get(cat, 1000) for cat in categories)
            allocation = {}
            for cat in categories:
                cat_size = self.category_sizes.get(cat, 1000)
                share = int(self.global_budget * cat_size / total_size)
                allocation[cat] = max(share, effective_min)

        elif self.strategy == "recency":
            # Newer categories get 1.5× weight
            weights = {}
            for task_idx, task_cats in enumerate(self.task_history):
                recency = 1.0 + 0.5 * (task_idx / max(len(self.task_history) - 1, 1))
                for cat in task_cats:
                    weights[cat] = recency
            # Handle categories not yet in history
            for cat in categories:
                if cat not in weights:
                    weights[cat] = 1.5  # Newest gets highest weight

            total_weight = sum(weights.get(cat, 1.0) for cat in categories)
            allocation = {}
            for cat in categories:
                share = int(self.global_budget * weights.get(cat, 1.0) / total_weight)
                allocation[cat] = max(share, effective_min)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Hard enforce: trim proportionally until total <= global_budget
        total = sum(allocation.values())
        if total > self.global_budget:
            overflow = total - self.global_budget
            # Sort categories by allocation (descending) and trim from largest
            sorted_cats = sorted(allocation.keys(), key=lambda c: allocation[c], reverse=True)
            for cat in sorted_cats:
                if overflow <= 0:
                    break
                # How much can we trim from this category?
                trimmable = allocation[cat] - effective_min
                trim = min(trimmable, overflow)
                if trim > 0:
                    allocation[cat] -= trim
                    overflow -= trim

        return allocation

    def get_global_coreset(self) -> np.ndarray:
        """
        Return the combined global coreset as a single [total_patches, D] array.
        """
        if not self.category_coresets:
            return np.empty((0, 0), dtype=np.float32)

        return np.concatenate(list(self.category_coresets.values()), axis=0)

    def get_category_labels(self) -> np.ndarray:
        """
        Return category label for each patch in the global coreset.
        Useful for analysis and visualisation.
        """
        labels = []
        for cat, coreset in self.category_coresets.items():
            labels.extend([cat] * coreset.shape[0])
        return np.array(labels)

    def get_stats(self) -> dict:
        """Return current state summary."""
        return {
            "total_categories": len(self.categories_seen),
            "total_patches": self.total_patches,
            "global_budget": self.global_budget,
            "per_category": {
                cat: coreset.shape[0] for cat, coreset in self.category_coresets.items()
            },
            "tasks_seen": len(self.task_history),
            "feature_dim": self.feature_dim,
        }

    def __repr__(self):
        return (
            f"AdaptiveCoresetManager(budget={self.global_budget}, "
            f"strategy={self.strategy}, categories={len(self.categories_seen)}, "
            f"patches={self.total_patches})"
        )

    # ── O2: Disk cache helpers ──────────────────────────────

    def _cache_key(self, category: str, budget: int, dim: int) -> str:
        """Deterministic cache filename for (category, budget, method, seed, dim)."""
        return f"{category}_b{budget}_{self.selection_method}_s{self.seed}_d{dim}.npy"

    def _load_cached_coreset(self, category: str, budget: int, dim: int) -> Optional[np.ndarray]:
        """Load coreset from disk cache if available."""
        if self.cache_dir is None:
            return None
        path = self.cache_dir / self._cache_key(category, budget, dim)
        if path.exists():
            data = np.load(path)
            if data.shape[0] == budget and data.shape[1] == dim:
                return data
        return None

    def _save_cached_coreset(self, category: str, budget: int, dim: int, coreset: np.ndarray):
        """Save coreset to disk cache."""
        if self.cache_dir is None:
            return
        path = self.cache_dir / self._cache_key(category, budget, dim)
        np.save(path, coreset)
