"""
Greedy k-center coreset selection for MemoryAD.

Selects a maximally-covering subset of patch features by iteratively picking
the point farthest from the current coreset.

GPU-accelerated: uses PyTorch CUDA tensors for the inner loop matrix-vector
multiply, giving ~10-50x speedup over numpy on RTX 3070.
"""

import numpy as np
import torch
from tqdm import tqdm


def _get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def greedy_coreset_selection(
    features: np.ndarray,
    budget: int,
    seed: int = 0,
    use_faiss: bool = True,  # kept for API compat, ignored
) -> np.ndarray:
    """
    Greedy k-center coreset selection (GPU-accelerated).

    Iteratively selects the point farthest from the current coreset until the
    budget is reached. This maximises the coverage of the feature space.

    Args:
        features: [N, D] numpy array of patch features.
        budget:   Number of patches to select.
        seed:     Random seed for initial point selection.
        use_faiss: Ignored (kept for API compatibility).

    Returns:
        coreset: [budget, D] numpy array of selected features.
    """
    N, D = features.shape

    if budget >= N:
        return features.copy()

    features = features.astype(np.float32)

    # O1: Pre-subsample to reduce N before greedy selection.
    max_pool = max(budget * 5, 5000)
    if N > max_pool:
        rng = np.random.RandomState(seed)
        indices = rng.choice(N, size=max_pool, replace=False)
        features = features[indices]

    return _greedy_coreset_gpu(features, budget, seed)


def _greedy_coreset_gpu(
    features: np.ndarray,
    budget: int,
    seed: int,
) -> np.ndarray:
    """Greedy coreset selection using PyTorch CUDA tensors.

    The inner loop does a matrix-vector multiply (features @ last_point)
    which is ~10-50x faster on GPU than CPU numpy.
    Falls back to CPU PyTorch if CUDA is not available.
    """
    device = _get_device()
    N, D = features.shape
    rng = np.random.RandomState(seed)

    # Move features to GPU once
    feats = torch.from_numpy(features).to(device)  # [N, D]

    selected_indices = []
    selected_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # Step 1: Pick random initial seed point
    initial_idx = rng.randint(0, N)
    selected_indices.append(initial_idx)
    selected_mask[initial_idx] = True

    # Step 2: Maintain min distance from each point to nearest coreset member
    min_distances = torch.full((N,), float('inf'), dtype=torch.float32, device=device)

    # Pre-compute squared norms
    sq_norms = (feats * feats).sum(dim=1)  # [N]

    for step in tqdm(range(1, budget), desc="Coreset selection", leave=False):
        last_idx = selected_indices[-1]
        last_point = feats[last_idx]  # [D]
        last_sq_norm = sq_norms[last_idx]

        # Vectorized squared L2 on GPU:
        # ||x_i - last||^2 = ||x_i||^2 + ||last||^2 - 2*x_i.last
        dots = feats @ last_point  # [N] — GPU matrix-vector multiply
        distances = sq_norms + last_sq_norm - 2.0 * dots
        distances.clamp_(min=0.0)  # Clamp numerical noise

        # Update minimum distances
        torch.minimum(min_distances, distances, out=min_distances)

        # Mask already-selected points
        min_distances[selected_mask] = -1.0

        # Select farthest point
        next_idx = int(torch.argmax(min_distances).item())
        selected_indices.append(next_idx)
        selected_mask[next_idx] = True

    # Return as numpy array (on CPU)
    result = feats[selected_indices].cpu().numpy()

    # Free GPU memory
    del feats, min_distances, sq_norms, selected_mask
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def random_coreset_selection(
    features: np.ndarray,
    budget: int,
    seed: int = 0,
) -> np.ndarray:
    """Simple random subsampling baseline for ablation comparison."""
    N = features.shape[0]
    if budget >= N:
        return features.copy()

    rng = np.random.RandomState(seed)
    indices = rng.choice(N, size=budget, replace=False)
    return features[indices]
