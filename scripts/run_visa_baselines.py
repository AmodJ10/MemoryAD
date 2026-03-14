"""
Run Joint, Naive, and MemoryAD on VisA (4 tasks) with full CIL metrics.

Output: results/baseline_visa_results.json — I-AUROC, FR, Avg Inc, FT
for each method (Joint, Naive, MemoryAD).

Usage:
    .venv\\Scripts\\python.exe scripts\\run_visa_baselines.py
    .venv\\Scripts\\python.exe scripts\\run_visa_baselines.py --only independent
    .venv\\Scripts\\python.exe scripts\\run_visa_baselines.py --only independent joint
"""
import sys, os, time, json, gc, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils.feature_cache import FeatureCache
from src.coreset.adaptive_manager import AdaptiveCoresetManager
from src.coreset.greedy_coreset import greedy_coreset_selection
from src.scoring.knn_scorer import KNNScorer
from src.evaluation.metrics import compute_auroc
import yaml

TASKS = yaml.safe_load(open("configs/visa_4task.yaml"))["tasks"]
ALL_CATEGORIES = [c for t in TASKS for c in t["categories"]]
BUDGET = 10000
K = 9
FEATURE_DIR = "data/features/dinov2_vitb14_visa"
# Cap per-category patches to avoid RAM explosion (12 cats × 20K × 1536 × 4B ≈ 1.4GB)
MAX_PATCHES_PER_CAT = 20000


def _cleanup():
    """Free GPU memory and trigger garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_cached(scorer, cache, categories, spatial_dims):
    """Evaluate I-AUROC on categories using cached features."""
    results = {}
    for cat in categories:
        test_feats, test_labels = cache.load_test_data(cat)
        image_scores, _ = scorer.score_batch(test_feats, spatial_dims)
        results[cat] = compute_auroc(test_labels, image_scores)
    return results


def compute_cil_metrics(auroc_matrix, n_tasks, n_cats, all_categories, tasks,
                        joint_aurocs=None):
    """Compute I-AUROC, FR, Avg Inc, and FT."""
    am = np.array(auroc_matrix)

    # Final mean AUROC (last row, non-NaN)
    final_mean = float(np.nanmean(am[-1]))

    # Avg Incremental: mean of per-task mean AUROCs
    task_means = []
    for t in range(n_tasks):
        valid = ~np.isnan(am[t])
        if valid.any():
            task_means.append(float(np.mean(am[t, valid])))
    avg_inc = float(np.mean(task_means)) if task_means else 0.0

    # Forgetting Rate: for each category, max AUROC seen - final AUROC
    forgetting_vals = []
    for cat_idx in range(n_cats):
        col = am[:, cat_idx]
        valid = ~np.isnan(col)
        if valid.sum() > 1:
            valid_scores = col[valid]
            best = np.max(valid_scores[:-1])
            final = valid_scores[-1]
            forgetting_vals.append(max(0.0, best - final))
    forgetting_rate = float(np.mean(forgetting_vals)) if forgetting_vals else 0.0

    # Forward Transfer: AUROC at first exposure / joint AUROC
    ft_vals = []
    if joint_aurocs is not None:
        for t_idx, task in enumerate(tasks):
            for cat in task["categories"]:
                cat_idx = all_categories.index(cat)
                if not np.isnan(am[t_idx, cat_idx]) and joint_aurocs[cat_idx] > 0:
                    ft_vals.append(am[t_idx, cat_idx] / joint_aurocs[cat_idx])
    forward_transfer = float(np.mean(ft_vals)) if ft_vals else 0.0

    return {
        "final_mean_auroc": final_mean,
        "avg_incremental_auroc": avg_inc,
        "forgetting_rate": forgetting_rate,
        "forward_transfer": forward_transfer,
    }


# ── Joint (upper bound) ─────────────────────────────────
def run_joint(cache):
    print(f"\n{'='*60}")
    print("Joint (upper bound) — 4-task, 12 categories")
    print(f"{'='*60}")
    t0 = time.time()

    rng = np.random.RandomState(42)
    all_feats = []
    for cat in ALL_CATEGORIES:
        f = cache.load_train_features(cat)
        if f.shape[0] > MAX_PATCHES_PER_CAT:
            indices = rng.choice(f.shape[0], MAX_PATCHES_PER_CAT, replace=False)
            f = f[indices]
        all_feats.append(f)
        print(f"  {cat}: {f.shape[0]} patches")

    combined = np.concatenate(all_feats, axis=0)
    del all_feats
    print(f"  Total: {combined.shape[0]} patches -> coreset {BUDGET}...")
    coreset = greedy_coreset_selection(combined, budget=BUDGET)
    del combined
    _cleanup()

    scorer = KNNScorer(k=K)
    scorer.fit(coreset)

    cat_aurocs = evaluate_cached(scorer, cache, ALL_CATEGORIES, cache.spatial_dims)
    for cat in ALL_CATEGORIES:
        print(f"    {cat}: I-AUROC={cat_aurocs[cat]:.4f}")

    # Build auroc matrix (joint sees all categories at each task step)
    n_tasks = len(TASKS)
    n_cats = len(ALL_CATEGORIES)
    auroc_matrix = np.full((n_tasks, n_cats), np.nan)
    cats_seen = []
    for t_idx, task in enumerate(TASKS):
        cats_seen.extend(task["categories"])
        for cat in cats_seen:
            auroc_matrix[t_idx, ALL_CATEGORIES.index(cat)] = cat_aurocs[cat]

    # Joint per-category AUROCs (used as FT reference for other methods)
    joint_cat_aurocs = np.array([cat_aurocs[cat] for cat in ALL_CATEGORIES])

    elapsed = time.time() - t0
    metrics = compute_cil_metrics(auroc_matrix, n_tasks, n_cats, ALL_CATEGORIES,
                                  TASKS, joint_aurocs=joint_cat_aurocs)
    metrics["auroc_matrix"] = auroc_matrix.tolist()
    metrics["time_s"] = elapsed
    print(f"\nJoint done in {elapsed:.1f}s — I-AUROC={metrics['final_mean_auroc']:.4f}")
    del scorer
    _cleanup()
    return metrics, joint_cat_aurocs


# ── Naive (lower bound) ─────────────────────────────────
def run_naive(cache, joint_cat_aurocs):
    print(f"\n{'='*60}")
    print("Naive (lower bound) — 4-task, 12 categories")
    print(f"{'='*60}")
    t0 = time.time()

    scorer = KNNScorer(k=K)
    rng = np.random.RandomState(42)
    n_tasks = len(TASKS)
    n_cats = len(ALL_CATEGORIES)
    auroc_matrix = np.full((n_tasks, n_cats), np.nan)
    categories_seen = []

    for task_idx, task in enumerate(TASKS):
        cats = task["categories"]
        print(f"\n--- Task {task_idx} ---")

        # Naive: only use current task's features (forget everything else)
        task_feats = []
        for c in cats:
            f = cache.load_train_features(c)
            if f.shape[0] > MAX_PATCHES_PER_CAT:
                indices = rng.choice(f.shape[0], MAX_PATCHES_PER_CAT, replace=False)
                f = f[indices]
            task_feats.append(f)

        combined = np.concatenate(task_feats, axis=0)
        del task_feats
        coreset = greedy_coreset_selection(combined, budget=BUDGET)
        del combined
        _cleanup()
        scorer.fit(coreset)
        del coreset
        _cleanup()
        categories_seen.extend(cats)

        cat_aurocs = evaluate_cached(scorer, cache, categories_seen, cache.spatial_dims)
        for cat in categories_seen:
            auroc_matrix[task_idx, ALL_CATEGORIES.index(cat)] = cat_aurocs[cat]
            print(f"    {cat}: I-AUROC={cat_aurocs[cat]:.4f}")

    elapsed = time.time() - t0
    metrics = compute_cil_metrics(auroc_matrix, n_tasks, n_cats, ALL_CATEGORIES,
                                  TASKS, joint_aurocs=joint_cat_aurocs)
    metrics["auroc_matrix"] = auroc_matrix.tolist()
    metrics["time_s"] = elapsed
    print(f"\nNaive done in {elapsed:.1f}s — I-AUROC={metrics['final_mean_auroc']:.4f}")
    del scorer
    _cleanup()
    return metrics


# ── MemoryAD (ours) ──────────────────────────────────────
def run_memoryad(cache, joint_cat_aurocs):
    print(f"\n{'='*60}")
    print("MemoryAD (ours) — 4-task, 12 categories")
    print(f"{'='*60}")
    t0 = time.time()

    manager = AdaptiveCoresetManager(global_budget=BUDGET, strategy="proportional")
    scorer = KNNScorer(k=K)
    spatial_dims = cache.spatial_dims
    n_tasks = len(TASKS)
    n_cats = len(ALL_CATEGORIES)
    auroc_matrix = np.full((n_tasks, n_cats), np.nan)
    categories_seen = []

    for task_idx, task in enumerate(TASKS):
        cats = task["categories"]
        print(f"\n--- Task {task_idx} ---")
        rng = np.random.RandomState(42)
        task_features = {}
        for c in cats:
            feats = cache.load_train_features(c)
            if feats.shape[0] > MAX_PATCHES_PER_CAT:
                indices = rng.choice(feats.shape[0], MAX_PATCHES_PER_CAT, replace=False)
                feats = feats[indices]
            task_features[c] = feats
            print(f"  {c}: {feats.shape[0]} patches")

        manager.add_task(task_features)
        del task_features
        _cleanup()
        scorer.fit(manager.get_global_coreset())
        categories_seen.extend(cats)

        cat_aurocs = evaluate_cached(scorer, cache, categories_seen, spatial_dims)
        for cat in categories_seen:
            auroc_matrix[task_idx, ALL_CATEGORIES.index(cat)] = cat_aurocs[cat]
            print(f"    {cat}: I-AUROC={cat_aurocs[cat]:.4f}")

    elapsed = time.time() - t0
    metrics = compute_cil_metrics(auroc_matrix, n_tasks, n_cats, ALL_CATEGORIES,
                                  TASKS, joint_aurocs=joint_cat_aurocs)
    metrics["auroc_matrix"] = auroc_matrix.tolist()
    metrics["time_s"] = elapsed
    print(f"\nMemoryAD done in {elapsed:.1f}s — I-AUROC={metrics['final_mean_auroc']:.4f}")
    del scorer, manager
    _cleanup()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+",
                        choices=["joint", "naive", "memoryad"],
                        help="Run only specific methods (default: all)")
    args = parser.parse_args()
    run_methods = set(args.only) if args.only else {"joint", "naive", "memoryad"}

    cache = FeatureCache(FEATURE_DIR)
    print(f"Feature cache: {cache.feature_dir}")
    print(f"Spatial dims: {cache.spatial_dims}, Feature dim: {cache.feature_dim}")

    results = {}
    t_start = time.time()
    joint_cat_aurocs = None

    # Joint first (upper bound + provides FT reference)
    if "joint" in run_methods:
        joint_result, joint_cat_aurocs = run_joint(cache)
        results["Joint (upper)"] = joint_result

    if "naive" in run_methods:
        results["Naive (lower)"] = run_naive(cache, joint_cat_aurocs)

    if "memoryad" in run_methods:
        results["MemoryAD (ours)"] = run_memoryad(cache, joint_cat_aurocs)

    total_time = time.time() - t_start

    # Print comparison table
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON (4-task VisA)")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'I-AUROC':>10} {'FR':>8} {'Avg Inc':>10} {'FT':>8}")
    print("-" * 60)
    for method, m in sorted(results.items(),
                            key=lambda x: x[1]["final_mean_auroc"], reverse=True):
        print(f"{method:<20} {m['final_mean_auroc']:>10.4f} {m['forgetting_rate']:>8.4f} "
              f"{m['avg_incremental_auroc']:>10.4f} {m['forward_transfer']:>8.4f}")
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_visa_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to results/baseline_visa_results.json")


if __name__ == "__main__":
    main()
