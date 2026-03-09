"""
k-NN anomaly scorer for MemoryAD.

Scores test patches by their distance to the nearest neighbours in the
global coreset. Patches far from any normal patch are likely anomalous.

GPU-accelerated: uses PyTorch CUDA for batch distance computation when CUDA
is available, with faiss CPU as fallback.
"""

import numpy as np
import torch
from typing import Optional, Tuple


def _get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class KNNScorer:
    """
    Anomaly scorer using k-nearest-neighbour distance.
    
    Given a global coreset of normal patch features, scores test images
    by computing the distance of each test patch to its k nearest
    neighbours in the coreset.
    
    Uses GPU-accelerated batch distance computation when CUDA is available.
    Falls back to faiss CPU index otherwise.
    """

    def __init__(
        self,
        k: int = 9,
        image_score_method: str = "max",
        top_k_percent: float = 1.0,
    ):
        """
        Args:
            k:                  Number of nearest neighbours.
            image_score_method: "max" — image score is max patch score.
                                "top_k_mean" — mean of top K% patches.
            top_k_percent:      Percentage of patches for top_k_mean.
        """
        self.k = k
        self.image_score_method = image_score_method
        self.top_k_percent = top_k_percent

        self._coreset = None       # numpy array for faiss fallback
        self._coreset_gpu = None   # torch tensor on GPU
        self._index = None         # faiss index (fallback)
        self._device = _get_device()
        self._use_gpu = self._device.type == "cuda"

    def fit(self, coreset: np.ndarray):
        """
        Build the nearest-neighbour index from the global coreset.

        Args:
            coreset: [M, D] numpy array of normal patch features.
        """
        coreset = coreset.astype(np.float32)
        self._coreset = coreset

        if self._use_gpu:
            # Keep coreset on GPU for fast batch distance computation
            self._coreset_gpu = torch.from_numpy(coreset).to(self._device)
        else:
            # Fallback: use faiss CPU index
            import faiss
            D = coreset.shape[1]
            self._index = faiss.IndexFlatL2(D)
            self._index.add(coreset)

    def _gpu_knn(self, queries: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated k-NN distance computation.
        
        Computes pairwise L2 distances between queries and coreset using
        PyTorch on CUDA, then returns the mean of k-nearest distances.
        
        For large query sets, processes in chunks to avoid OOM.
        
        Args:
            queries: [Q, D] numpy array of query features.
            
        Returns:
            scores: [Q] numpy array, mean k-NN distance per query.
        """
        Q, D = queries.shape
        M = self._coreset_gpu.shape[0]
        k = min(self.k, M)
        
        # Process in chunks to avoid GPU OOM
        # Each chunk needs Q_chunk * M * 4 bytes for distance matrix
        # For 8GB VRAM with coreset M=10K, D=1536: ~60MB per 1K queries
        max_chunk = max(1, min(Q, 2048))  # 2K queries at a time
        
        all_scores = np.empty(Q, dtype=np.float32)
        
        for start in range(0, Q, max_chunk):
            end = min(start + max_chunk, Q)
            chunk = torch.from_numpy(queries[start:end]).to(self._device)  # [C, D]
            
            # Compute pairwise squared L2 distances
            # ||q - c||^2 = ||q||^2 + ||c||^2 - 2*q.c
            q_sq = (chunk * chunk).sum(dim=1, keepdim=True)       # [C, 1]
            c_sq = (self._coreset_gpu * self._coreset_gpu).sum(dim=1, keepdim=True).T  # [1, M]
            dists = q_sq + c_sq - 2.0 * (chunk @ self._coreset_gpu.T)  # [C, M]
            dists.clamp_(min=0.0)
            
            # Get k-nearest distances
            topk_dists, _ = torch.topk(dists, k, dim=1, largest=False)  # [C, k]
            chunk_scores = topk_dists.mean(dim=1)  # [C]
            
            all_scores[start:end] = chunk_scores.cpu().numpy()
            
            del chunk, dists, topk_dists, chunk_scores
        
        return all_scores

    def score_patches(self, test_features: np.ndarray) -> np.ndarray:
        """
        Compute anomaly score for each test patch.

        Args:
            test_features: [P, D] numpy array of patch features from one test image.

        Returns:
            scores: [P] numpy array of anomaly scores per patch.
        """
        if self._coreset is None:
            raise RuntimeError("Must call fit() before scoring.")

        test_features = test_features.astype(np.float32)

        if self._use_gpu:
            return self._gpu_knn(test_features)
        else:
            # Fallback: faiss CPU
            distances, _ = self._index.search(test_features, self.k)
            return distances.mean(axis=1)

    def score_image(
        self,
        test_features: np.ndarray,
        spatial_dims: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Compute image-level and pixel-level anomaly scores.

        Args:
            test_features: [P, D] numpy array of patch features.
            spatial_dims:  (H, W) tuple for reshaping into anomaly map.

        Returns:
            image_score: float, image-level anomaly score.
            anomaly_map: [H, W] numpy array if spatial_dims given, else None.
        """
        patch_scores = self.score_patches(test_features)

        # Image-level score
        if self.image_score_method == "max":
            image_score = float(patch_scores.max())
        elif self.image_score_method == "top_k_mean":
            n_top = max(1, int(len(patch_scores) * self.top_k_percent / 100))
            top_scores = np.sort(patch_scores)[-n_top:]
            image_score = float(top_scores.mean())
        else:
            raise ValueError(f"Unknown method: {self.image_score_method}")

        # Anomaly map
        anomaly_map = None
        if spatial_dims is not None:
            H, W = spatial_dims
            if len(patch_scores) == H * W:
                anomaly_map = patch_scores.reshape(H, W)

        return image_score, anomaly_map

    def score_batch(
        self,
        batch_features: np.ndarray,
        spatial_dims: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Score a batch of test images.

        Args:
            batch_features: [B, P, D] numpy array.
            spatial_dims:   (H, W) for anomaly maps.

        Returns:
            image_scores: [B] array.
            anomaly_maps: [B, H, W] array if spatial_dims given.
        """
        if self._coreset is None:
            raise RuntimeError("Must call fit() before scoring.")

        B, P, D = batch_features.shape

        # Flatten and score all patches at once
        flat = batch_features.reshape(B * P, D).astype(np.float32)
        
        if self._use_gpu:
            patch_scores_flat = self._gpu_knn(flat)
        else:
            distances, _ = self._index.search(flat, self.k)
            patch_scores_flat = distances.mean(axis=1)
        
        patch_scores = patch_scores_flat.reshape(B, P)

        # Image-level scores
        if self.image_score_method == "max":
            image_scores = patch_scores.max(axis=1)
        elif self.image_score_method == "top_k_mean":
            n_top = max(1, int(P * self.top_k_percent / 100))
            sorted_scores = np.sort(patch_scores, axis=1)[:, -n_top:]
            image_scores = sorted_scores.mean(axis=1)
        else:
            raise ValueError(f"Unknown method: {self.image_score_method}")

        # Anomaly maps
        anomaly_maps = None
        if spatial_dims is not None:
            H, W = spatial_dims
            if P == H * W:
                anomaly_maps = patch_scores.reshape(B, H, W)

        return image_scores, anomaly_maps
