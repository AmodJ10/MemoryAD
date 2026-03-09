"""
Run all baselines on full 5-task MVTec AD and compute CIL metrics.

Output: results/baseline_full_results.json — contains auroc_matrix, FR, AI, FT
for each method (Joint, Naive, MemoryAD, EWC, LwF, Replay).

Usage:
    .venv\\Scripts\\python.exe scripts\\run_baselines_full.py
    .venv\\Scripts\\python.exe scripts\\run_baselines_full.py --skip-rd4ad
"""
import sys, os, time, json, subprocess, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data_utils.feature_cache import FeatureCache
from src.coreset.adaptive_manager import AdaptiveCoresetManager
from src.coreset.greedy_coreset import greedy_coreset_selection
from src.scoring.knn_scorer import KNNScorer
from src.evaluation.metrics import compute_auroc

# ── Full 5-task MVTec AD config ─────────────────────────
TASKS = [
    {"categories": ["bottle", "cable", "capsule"]},
    {"categories": ["carpet", "grid", "hazelnut"]},
    {"categories": ["leather", "metal_nut", "pill"]},
    {"categories": ["screw", "tile", "toothbrush"]},
    {"categories": ["transistor", "wood", "zipper"]},
]
ALL_CATEGORIES = [c for t in TASKS for c in t["categories"]]
BUDGET = 10000
K = 9
FEATURE_DIR = "data/features/dinov2_vitb14"


def evaluate_cached(scorer, cache, categories, spatial_dims):
    """Evaluate I-AUROC on categories using cached features."""
    results = {}
    for cat in categories:
        test_feats, test_labels = cache.load_test_data(cat)
        image_scores, _ = scorer.score_batch(test_feats, spatial_dims)
        results[cat] = compute_auroc(test_labels, image_scores)
    return results


def compute_cil_metrics(auroc_matrix, n_tasks, n_cats, all_categories, tasks):
    """Compute forgetting rate, avg incremental AUROC, and forward transfer."""
    am = np.array(auroc_matrix)

    # Final mean AUROC (last row, non-NaN)
    valid_final = ~np.isnan(am[-1])
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
            best = np.nanmax(col)
            final = col[-1] if not np.isnan(col[-1]) else 0.0
            forgetting_vals.append(best - final)
    forgetting_rate = float(np.mean(forgetting_vals)) if forgetting_vals else 0.0

    # Forward Transfer: AUROC at first exposure for each category
    ft_vals = []
    cats_seen = []
    for t_idx, task in enumerate(tasks):
        for cat in task["categories"]:
            cat_idx = all_categories.index(cat)
            if not np.isnan(am[t_idx, cat_idx]):
                ft_vals.append(am[t_idx, cat_idx])
    forward_transfer = float(np.mean(ft_vals)) if ft_vals else 0.0

    return {
        "final_mean_auroc": final_mean,
        "avg_incremental_auroc": avg_inc,
        "forgetting_rate": forgetting_rate,
        "forward_transfer": forward_transfer,
    }


# ── MemoryAD ────────────────────────────────────────────
def run_memoryad(cache):
    print(f"\n{'='*60}")
    print("MemoryAD (ours) — 5-task, 15 categories")
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
        task_features = {}
        for c in cats:
            feats = cache.load_train_features(c)
            task_features[c] = feats
            print(f"  {c}: {feats.shape[0]} patches")

        manager.add_task(task_features)
        scorer.fit(manager.get_global_coreset())
        categories_seen.extend(cats)

        cat_aurocs = evaluate_cached(scorer, cache, categories_seen, spatial_dims)
        for cat in categories_seen:
            auroc_matrix[task_idx, ALL_CATEGORIES.index(cat)] = cat_aurocs[cat]
            print(f"    {cat}: I-AUROC={cat_aurocs[cat]:.4f}")

    elapsed = time.time() - t0
    metrics = compute_cil_metrics(auroc_matrix, n_tasks, n_cats, ALL_CATEGORIES, TASKS)
    metrics["auroc_matrix"] = auroc_matrix.tolist()
    metrics["time_s"] = elapsed
    print(f"\nMemoryAD done in {elapsed:.1f}s — I-AUROC={metrics['final_mean_auroc']:.4f}")
    return metrics


# ── Joint (upper bound) ─────────────────────────────────
def run_joint(cache):
    print(f"\n{'='*60}")
    print("Joint (upper bound) — 5-task, 15 categories")
    print(f"{'='*60}")
    t0 = time.time()

    rng = np.random.RandomState(42)
    all_feats = []
    for cat in ALL_CATEGORIES:
        f = cache.load_train_features(cat)
        all_feats.append(f)
        print(f"  {cat}: {f.shape[0]} patches")

    combined = np.concatenate(all_feats, axis=0)
    print(f"  Total: {combined.shape[0]} patches -> coreset {BUDGET}...")
    coreset = greedy_coreset_selection(combined, budget=BUDGET)

    scorer = KNNScorer(k=K)
    scorer.fit(coreset)

    cat_aurocs = evaluate_cached(scorer, cache, ALL_CATEGORIES, cache.spatial_dims)
    for cat in ALL_CATEGORIES:
        print(f"    {cat}: I-AUROC={cat_aurocs[cat]:.4f}")

    # Joint has no incremental setup — same result at all "tasks"
    n_tasks = len(TASKS)
    n_cats = len(ALL_CATEGORIES)
    auroc_matrix = np.full((n_tasks, n_cats), np.nan)
    cats_seen = []
    for t_idx, task in enumerate(TASKS):
        cats_seen.extend(task["categories"])
        for cat in cats_seen:
            auroc_matrix[t_idx, ALL_CATEGORIES.index(cat)] = cat_aurocs[cat]

    elapsed = time.time() - t0
    metrics = compute_cil_metrics(auroc_matrix, n_tasks, n_cats, ALL_CATEGORIES, TASKS)
    metrics["auroc_matrix"] = auroc_matrix.tolist()
    metrics["time_s"] = elapsed
    print(f"\nJoint done in {elapsed:.1f}s — I-AUROC={metrics['final_mean_auroc']:.4f}")
    return metrics


# ── Naive (lower bound) ─────────────────────────────────
def run_naive(cache):
    print(f"\n{'='*60}")
    print("Naive (lower bound) — 5-task, 15 categories")
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
            task_feats.append(f)

        combined = np.concatenate(task_feats, axis=0)
        coreset = greedy_coreset_selection(combined, budget=BUDGET)
        scorer.fit(coreset)

        categories_seen.extend(cats)

        cat_aurocs = evaluate_cached(scorer, cache, categories_seen, cache.spatial_dims)
        for cat in categories_seen:
            auroc_matrix[task_idx, ALL_CATEGORIES.index(cat)] = cat_aurocs[cat]
            print(f"    {cat}: I-AUROC={cat_aurocs[cat]:.4f}")

    elapsed = time.time() - t0
    metrics = compute_cil_metrics(auroc_matrix, n_tasks, n_cats, ALL_CATEGORIES, TASKS)
    metrics["auroc_matrix"] = auroc_matrix.tolist()
    metrics["time_s"] = elapsed
    print(f"\nNaive done in {elapsed:.1f}s — I-AUROC={metrics['final_mean_auroc']:.4f}")
    return metrics


# ── RD4AD baselines via subprocess ──────────────────────
_RD4AD_WORKER_FULL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_rd4ad_worker_full.py"
)


def run_rd4ad_subprocess(method, epochs=30):
    """Run an RD4AD baseline in a separate process and parse full results."""
    print(f"\n{'='*60}")
    print(f"{method.upper()} + RD4AD (subprocess, {epochs} epochs, 5-task)")
    print(f"{'='*60}")
    sys.stdout.flush()

    proc = subprocess.Popen(
        [sys.executable, _RD4AD_WORKER_FULL, method, str(epochs)],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    result = None
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("RESULT_JSON:"):
            result = json.loads(line.replace("RESULT_JSON:", ""))
        else:
            print(f"  {line}")
            sys.stdout.flush()

    proc.wait(timeout=3600)
    if proc.returncode != 0 or result is None:
        print(f"  ERROR: subprocess exited with code {proc.returncode}")
        return {"final_mean_auroc": 0.0, "forgetting_rate": 0.0,
                "avg_incremental_auroc": 0.0, "forward_transfer": 0.0}

    # Compute CIL metrics from auroc_matrix
    am = result["auroc_matrix"]
    n_tasks = len(TASKS)
    n_cats = len(ALL_CATEGORIES)
    metrics = compute_cil_metrics(am, n_tasks, n_cats, ALL_CATEGORIES, TASKS)
    metrics["auroc_matrix"] = am
    print(f"  {method.upper()} I-AUROC={metrics['final_mean_auroc']:.4f} FR={metrics['forgetting_rate']:.4f}")
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rd4ad", action="store_true",
                        help="Skip RD4AD baselines (EWC/LwF/Replay)")
    args = parser.parse_args()

    cache = FeatureCache(FEATURE_DIR)
    print(f"Feature cache: {cache.feature_dir}")
    print(f"Spatial dims: {cache.spatial_dims}, Feature dim: {cache.feature_dim}")

    results = {}
    t_start = time.time()

    # Feature-based methods (fast)
    results["MemoryAD (ours)"] = run_memoryad(cache)
    results["Joint (upper)"] = run_joint(cache)
    results["Naive (lower)"] = run_naive(cache)

    # RD4AD-based methods (slow, need GPU)
    if not args.skip_rd4ad:
        results["EWC + RD4AD"] = run_rd4ad_subprocess("ewc", epochs=30)
        results["LwF + RD4AD"] = run_rd4ad_subprocess("lwf", epochs=30)
        results["Replay + RD4AD"] = run_rd4ad_subprocess("replay", epochs=30)
    else:
        print("\n[SKIPPED] RD4AD baselines (--skip-rd4ad)")

    total_time = time.time() - t_start

    # Print comparison table
    print(f"\n{'='*60}")
    print("FULL BASELINE COMPARISON (5-task MVTec AD)")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'I-AUROC':>10} {'FR':>8} {'Avg Inc':>10} {'FT':>8}")
    print("-" * 60)
    for method, m in sorted(results.items(), key=lambda x: x[1]["final_mean_auroc"], reverse=True):
        print(f"{method:<20} {m['final_mean_auroc']:>10.4f} {m['forgetting_rate']:>8.4f} "
              f"{m['avg_incremental_auroc']:>10.4f} {m['forward_transfer']:>8.4f}")
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_full_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to results/baseline_full_results.json")


if __name__ == "__main__":
    main()
