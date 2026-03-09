"""
Test all baselines using PRE-COMPUTED features (cached .npy files).

Run precompute_features.py first, then this script.

RD4AD baselines (EWC/LwF/Replay) are run in separate subprocesses
to guarantee GPU memory is fully released between runs. This prevents
OOM on 8GB GPUs (WideResNet-50 is ~1.5GB per model).

Tasks:
  Task 0: bottle, cable
  Task 1: carpet, grid
"""
import sys, os, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data_utils.feature_cache import FeatureCache
from src.coreset.adaptive_manager import AdaptiveCoresetManager
from src.coreset.greedy_coreset import greedy_coreset_selection
from src.scoring.knn_scorer import KNNScorer
from src.evaluation.metrics import compute_auroc


TASKS = [
    {"categories": ["bottle", "cable"]},
    {"categories": ["carpet", "grid"]},
]
ALL_CATEGORIES = ["bottle", "cable", "carpet", "grid"]
BUDGET = 5000
K = 9
MAX_PATCHES_PER_CAT = 5000


def subsample_features(features, max_patches, rng=None):
    """Randomly subsample features if they exceed max_patches."""
    if features.shape[0] <= max_patches:
        return features
    if rng is None:
        rng = np.random.RandomState(42)
    indices = rng.choice(features.shape[0], size=max_patches, replace=False)
    return features[indices]


def evaluate_cached(scorer, cache, categories, spatial_dims):
    """Evaluate I-AUROC on categories using cached test features + batch scoring."""
    results = {}
    for cat in categories:
        test_feats, test_labels = cache.load_test_data(cat)
        image_scores, _ = scorer.score_batch(test_feats, spatial_dims)
        results[cat] = compute_auroc(test_labels, image_scores)
    return results


# ---- MemoryAD (ours) ----
def run_memoryad(cache):
    print("\n" + "="*60)
    print("MemoryAD (ours)")
    print("="*60)

    manager = AdaptiveCoresetManager(global_budget=BUDGET, strategy="proportional")
    scorer = KNNScorer(k=K)
    spatial_dims = cache.spatial_dims
    n_tasks = len(TASKS)
    n_cats = len(ALL_CATEGORIES)
    auroc_matrix = np.full((n_tasks, n_cats), np.nan)
    categories_seen = []

    for task_idx, task in enumerate(TASKS):
        cats = task["categories"]
        task_features = {}
        for c in cats:
            feats = cache.load_train_features(c)
            feats = subsample_features(feats, MAX_PATCHES_PER_CAT)
            task_features[c] = feats
            print(f"  {c}: {feats.shape[0]} patches (subsampled)")

        manager.add_task(task_features)
        scorer.fit(manager.get_global_coreset())
        categories_seen.extend(cats)

        cat_aurocs = evaluate_cached(scorer, cache, categories_seen, spatial_dims)
        for cat in categories_seen:
            auroc_matrix[task_idx, ALL_CATEGORIES.index(cat)] = cat_aurocs[cat]
            print(f"  {cat}: I-AUROC = {cat_aurocs[cat]:.4f}")

    valid = ~np.isnan(auroc_matrix[-1])
    return float(np.mean(auroc_matrix[-1, valid])), auroc_matrix


# ---- Joint (upper bound) ----
def run_joint(cache):
    print("\n" + "="*60)
    print("Joint (upper bound)")
    print("="*60)

    rng = np.random.RandomState(42)
    all_feats = []
    for cat in ALL_CATEGORIES:
        f = cache.load_train_features(cat)
        f = subsample_features(f, MAX_PATCHES_PER_CAT, rng)
        all_feats.append(f)
        print(f"  {cat}: {f.shape[0]} patches (subsampled)")

    combined = np.concatenate(all_feats, axis=0)
    print(f"  Total: {combined.shape[0]} patches -> coreset {BUDGET}...")
    coreset = greedy_coreset_selection(combined, budget=BUDGET)

    scorer = KNNScorer(k=K)
    scorer.fit(coreset)

    cat_aurocs = evaluate_cached(scorer, cache, ALL_CATEGORIES, cache.spatial_dims)
    for cat in ALL_CATEGORIES:
        print(f"  {cat}: I-AUROC = {cat_aurocs[cat]:.4f}")

    return float(np.mean(list(cat_aurocs.values())))


# ---- Naive (lower bound) ----
def run_naive(cache):
    print("\n" + "="*60)
    print("Naive (lower bound)")
    print("="*60)

    scorer = KNNScorer(k=K)
    rng = np.random.RandomState(42)
    n_tasks = len(TASKS)
    n_cats = len(ALL_CATEGORIES)
    auroc_matrix = np.full((n_tasks, n_cats), np.nan)
    categories_seen = []

    for task_idx, task in enumerate(TASKS):
        cats = task["categories"]

        task_feats = []
        for c in cats:
            f = cache.load_train_features(c)
            f = subsample_features(f, MAX_PATCHES_PER_CAT, rng)
            task_feats.append(f)

        combined = np.concatenate(task_feats, axis=0)
        coreset = greedy_coreset_selection(combined, budget=BUDGET)
        scorer.fit(coreset)

        categories_seen.extend(cats)

        cat_aurocs = evaluate_cached(scorer, cache, categories_seen, cache.spatial_dims)
        for cat in categories_seen:
            auroc_matrix[task_idx, ALL_CATEGORIES.index(cat)] = cat_aurocs[cat]
            print(f"  Task {task_idx} | {cat}: I-AUROC = {cat_aurocs[cat]:.4f}")

    valid = ~np.isnan(auroc_matrix[-1])
    return float(np.mean(auroc_matrix[-1, valid]))


# ---- RD4AD baselines via subprocess ----
_RD4AD_WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_rd4ad_worker.py"
)


def run_rd4ad_subprocess(method, epochs=30):
    """Run an RD4AD baseline in a separate process to guarantee GPU cleanup.
    
    Output is streamed live to the terminal so the user sees real-time progress.
    """
    print(f"\n{'='*60}")
    print(f"{method.upper()} + RD4AD (subprocess, {epochs} epochs)")
    print(f"{'='*60}")
    sys.stdout.flush()

    proc = subprocess.Popen(
        [sys.executable, _RD4AD_WORKER_SCRIPT, method, str(epochs)],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,  # line-buffered
    )

    auroc = 0.0
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("RESULT_JSON:"):
            auroc = json.loads(line.replace("RESULT_JSON:", ""))
        else:
            print(f"  {line}")
            sys.stdout.flush()

    proc.wait(timeout=1800)
    if proc.returncode != 0:
        print(f"  ERROR: subprocess exited with code {proc.returncode}")
        return 0.0

    print(f"  Mean I-AUROC: {auroc:.4f}")
    return auroc


def main():
    cache = FeatureCache()

    # Verify features are cached
    missing = [c for c in ALL_CATEGORIES if not cache.has_category(c)]
    if missing:
        print(f"ERROR: Missing cached features for: {missing}")
        print("Run: .venv\\Scripts\\python.exe scripts\\precompute_features.py")
        sys.exit(1)

    print(f"Feature cache: {cache.feature_dir}")
    print(f"Spatial dims: {cache.spatial_dims}, Feature dim: {cache.feature_dim}")
    print(f"Categories: {cache.available_categories()}")

    results = {}
    t_start = time.time()

    # Feature-based methods (fast with cache, no GPU needed)
    auroc, _ = run_memoryad(cache)
    results["MemoryAD (ours)"] = auroc

    results["Joint (upper)"] = run_joint(cache)
    results["Naive (lower)"] = run_naive(cache)

    # RD4AD-based methods — each in a SEPARATE PROCESS for guaranteed GPU cleanup
    results["EWC + RD4AD"] = run_rd4ad_subprocess("ewc", epochs=30)
    results["LwF + RD4AD"] = run_rd4ad_subprocess("lwf", epochs=30)
    results["Replay + RD4AD"] = run_rd4ad_subprocess("replay", epochs=30)

    total_time = time.time() - t_start

    # Print comparison table
    print("\n" + "="*60)
    print("BASELINE COMPARISON (2-task MVTec AD)")
    print("="*60)
    print(f"{'Method':<20} {'Mean I-AUROC':>12}")
    print("-" * 34)
    for method, auroc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{method:<20} {auroc:>12.4f}")
    print(f"\nTotal time: {total_time:.0f}s")
    print("="*60)

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to results/baseline_comparison.json")


if __name__ == "__main__":
    main()
