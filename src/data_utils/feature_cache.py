"""
Feature cache utilities.

Load pre-computed DINOv2 features from disk instead of re-extracting.
Features are stored as .npy files in data/features/<backbone>/<category>/.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


DEFAULT_FEATURE_DIR = "data/features/dinov2_vitb14"


class FeatureCache:
    """Load pre-computed features from disk."""

    def __init__(self, feature_dir: str = DEFAULT_FEATURE_DIR):
        self.feature_dir = Path(feature_dir)

        # Load metadata
        spatial_path = self.feature_dir / "spatial_dims.npy"
        if spatial_path.exists():
            self.spatial_dims = tuple(np.load(spatial_path).tolist())
        else:
            self.spatial_dims = (37, 37)  # Default for DINOv2 ViT-B/14 at 518

        dim_path = self.feature_dir / "feature_dim.npy"
        if dim_path.exists():
            self.feature_dim = int(np.load(dim_path)[0])
        else:
            self.feature_dim = 1536  # concat of 2x768

    def has_category(self, category: str) -> bool:
        """Check if features are cached for a category."""
        return (self.feature_dir / category / "train_features.npy").exists()

    def load_train_features(self, category: str) -> np.ndarray:
        """Load train patch features [N_patches, D]."""
        path = self.feature_dir / category / "train_features.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"No cached features for '{category}'. Run precompute_features.py first."
            )
        return np.load(path)

    def load_test_data(self, category: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test features and labels.

        Returns:
            features: [N_test, P, D] per-image patch features
            labels: [N_test] binary labels (0=normal, 1=anomalous)
        """
        cat_dir = self.feature_dir / category
        features = np.load(cat_dir / "test_features.npy")
        labels = np.load(cat_dir / "test_labels.npy")
        return features, labels

    def load_all_train(self, categories: List[str]) -> Dict[str, np.ndarray]:
        """Load train features for multiple categories."""
        return {cat: self.load_train_features(cat) for cat in categories}

    def available_categories(self) -> List[str]:
        """List all categories with cached features."""
        cats = []
        for d in sorted(self.feature_dir.iterdir()):
            if d.is_dir() and (d / "train_features.npy").exists():
                cats.append(d.name)
        return cats
