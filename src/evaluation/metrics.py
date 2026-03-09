"""
Evaluation metrics for MemoryAD.

Includes standard anomaly detection metrics (AUROC, AUPR, F1-max) and
continual learning metrics (forgetting rate, forward transfer, average
incremental AUROC).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from typing import Dict, List, Optional


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def compute_aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve."""
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, scores))


def compute_f1_max(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute maximum F1 score across all thresholds."""
    if len(np.unique(labels)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    return float(np.max(f1_scores))


def compute_pixel_auroc(
    gt_masks: np.ndarray,
    anomaly_maps: np.ndarray,
) -> float:
    """
    Compute pixel-level AUROC.

    Args:
        gt_masks:     [N, H, W] binary ground truth masks.
        anomaly_maps: [N, H, W] anomaly score maps.
    """
    gt_flat = gt_masks.flatten().astype(int)
    pred_flat = anomaly_maps.flatten()

    if len(np.unique(gt_flat)) < 2:
        return 0.0
    return float(roc_auc_score(gt_flat, pred_flat))


# ── Continual Learning Metrics ──────────────────────────────────────


def compute_forgetting_rate(
    auroc_matrix: np.ndarray,
) -> float:
    """
    Compute average forgetting rate from an AUROC matrix.

    The AUROC matrix has shape [T, C] where:
      auroc_matrix[t, c] = AUROC on category c after learning task t.

    Forgetting for category c introduced at task t_c:
      f_c = max_{t in [t_c, T-1)} auroc_matrix[t, c] - auroc_matrix[T-1, c]

    Average forgetting = mean of f_c over all categories.

    Args:
        auroc_matrix: [T, C] numpy array. NaN for categories not yet seen.

    Returns:
        Average forgetting rate (lower is better, 0 means no forgetting).
    """
    T, C = auroc_matrix.shape
    forgetting_values = []

    for c in range(C):
        # Find when category c was first seen
        valid_mask = ~np.isnan(auroc_matrix[:, c])
        if valid_mask.sum() < 2:
            continue

        valid_scores = auroc_matrix[valid_mask, c]
        max_score = np.max(valid_scores[:-1])  # Best score before final
        final_score = valid_scores[-1]  # Score after all tasks

        forgetting = max_score - final_score
        forgetting_values.append(max(0, forgetting))  # Clip at 0

    if not forgetting_values:
        return 0.0
    return float(np.mean(forgetting_values))


def compute_forward_transfer(
    auroc_matrix: np.ndarray,
    joint_aurocs: np.ndarray,
) -> float:
    """
    Compute forward transfer: how well does the model adapt to new categories
    compared to the joint training upper bound?

    Args:
        auroc_matrix: [T, C] numpy array.
        joint_aurocs: [C] numpy array — per-category AUROC from joint training.

    Returns:
        Average forward transfer (1.0 = matches joint, <1.0 = worse).
    """
    T, C = auroc_matrix.shape
    transfers = []

    for c in range(C):
        # Find the first task where c was seen
        valid_mask = ~np.isnan(auroc_matrix[:, c])
        if valid_mask.sum() == 0:
            continue

        first_valid_idx = np.argmax(valid_mask)
        first_auroc = auroc_matrix[first_valid_idx, c]

        if joint_aurocs[c] > 0:
            ft = first_auroc / joint_aurocs[c]
            transfers.append(ft)

    if not transfers:
        return 0.0
    return float(np.mean(transfers))


def compute_avg_incremental_auroc(auroc_matrix: np.ndarray) -> float:
    """
    Compute average incremental AUROC: mean of per-task average AUROC.

    For each task t, compute the mean AUROC across all categories seen so far,
    then average across all tasks.

    Args:
        auroc_matrix: [T, C] numpy array.
    """
    T, C = auroc_matrix.shape
    task_means = []

    for t in range(T):
        valid_mask = ~np.isnan(auroc_matrix[t, :])
        if valid_mask.sum() > 0:
            task_means.append(np.mean(auroc_matrix[t, valid_mask]))

    if not task_means:
        return 0.0
    return float(np.mean(task_means))


def summarise_results(
    auroc_matrix: np.ndarray,
    joint_aurocs: np.ndarray,
    category_names: List[str],
) -> dict:
    """
    Compute all continual learning metrics from the AUROC matrix.

    Returns a dict with all metrics.
    """
    T = auroc_matrix.shape[0]

    # Final AUROC (after last task)
    valid_final = ~np.isnan(auroc_matrix[-1])
    final_auroc = float(np.mean(auroc_matrix[-1, valid_final])) if valid_final.any() else 0.0

    return {
        "final_mean_auroc": final_auroc,
        "avg_incremental_auroc": compute_avg_incremental_auroc(auroc_matrix),
        "forgetting_rate": compute_forgetting_rate(auroc_matrix),
        "forward_transfer": compute_forward_transfer(auroc_matrix, joint_aurocs),
        "per_category_final": {
            cat: float(auroc_matrix[-1, i]) if not np.isnan(auroc_matrix[-1, i]) else None
            for i, cat in enumerate(category_names)
        },
        "num_tasks": T,
        "num_categories": len(category_names),
    }
