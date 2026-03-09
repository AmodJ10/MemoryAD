"""Quick integration test for all core MemoryAD modules."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def test_greedy_coreset():
    from src.coreset.greedy_coreset import greedy_coreset_selection
    features = np.random.randn(1000, 768).astype(np.float32)
    coreset = greedy_coreset_selection(features, budget=100, use_faiss=True)
    assert coreset.shape == (100, 768), f"Expected (100, 768), got {coreset.shape}"
    print(f"[PASS] Greedy coreset: {features.shape} -> {coreset.shape}")

def test_adaptive_manager():
    from src.coreset.adaptive_manager import AdaptiveCoresetManager
    manager = AdaptiveCoresetManager(global_budget=500, strategy="proportional")

    # Task 0: 3 categories
    task0 = {
        "bottle": np.random.randn(300, 768).astype(np.float32),
        "cable": np.random.randn(250, 768).astype(np.float32),
        "capsule": np.random.randn(200, 768).astype(np.float32),
    }
    info0 = manager.add_task(task0)
    assert info0["total_categories"] == 3
    assert info0["total_patches"] <= 500
    print(f"[PASS] Task 0: {info0['total_patches']} patches, {info0['total_categories']} cats")

    # Task 1: 3 more categories
    task1 = {
        "carpet": np.random.randn(280, 768).astype(np.float32),
        "grid": np.random.randn(220, 768).astype(np.float32),
        "hazelnut": np.random.randn(260, 768).astype(np.float32),
    }
    info1 = manager.add_task(task1)
    assert info1["total_categories"] == 6
    assert info1["total_patches"] <= 500
    print(f"[PASS] Task 1: {info1['total_patches']} patches, {info1['total_categories']} cats")
    print(f"  Budget allocation: {info1['budget_allocation']}")

def test_knn_scorer():
    from src.scoring.knn_scorer import KNNScorer
    scorer = KNNScorer(k=5)
    
    coreset = np.random.randn(500, 768).astype(np.float32)
    scorer.fit(coreset)

    test_patch = np.random.randn(100, 768).astype(np.float32)
    img_score, amap = scorer.score_image(test_patch, spatial_dims=(10, 10))
    assert amap.shape == (10, 10)
    print(f"[PASS] k-NN scorer: score={img_score:.4f}, map_shape={amap.shape}")

def test_metrics():
    from src.evaluation.metrics import compute_auroc, compute_forgetting_rate
    
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    auroc = compute_auroc(labels, scores)
    assert auroc == 1.0, f"Expected AUROC=1.0, got {auroc}"
    print(f"[PASS] AUROC: {auroc:.4f}")

    auroc_matrix = np.array([
        [0.95, np.nan, np.nan],
        [0.93, 0.96, np.nan],
        [0.90, 0.94, 0.97],
    ])
    fr = compute_forgetting_rate(auroc_matrix)
    assert fr >= 0, f"Forgetting rate should be >= 0"
    print(f"[PASS] Forgetting rate: {fr:.4f}")

if __name__ == "__main__":
    test_greedy_coreset()
    test_adaptive_manager()
    test_knn_scorer()
    test_metrics()
    print("\n=== ALL TESTS PASSED ===")
