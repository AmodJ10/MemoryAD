"""
Run Joint and Naive baselines on VisA (4 tasks).
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils.feature_cache import FeatureCache
from src.coreset.greedy_coreset import greedy_coreset_selection
from src.scoring.knn_scorer import KNNScorer
from src.evaluation.metrics import compute_auroc
import yaml

TASKS = yaml.safe_load(open("configs/visa_4task.yaml"))["tasks"]
ALL_CATEGORIES = [c for t in TASKS for c in t["categories"]]
BUDGET = 10000
K = 9
FEATURE_DIR = "data/features/dinov2_vitb14_visa"

def evaluate_cached(scorer, cache, categories, spatial_dims):
    results = {}
    for cat in categories:
        test_feats, test_labels = cache.load_test_data(cat)
        image_scores, _ = scorer.score_batch(test_feats, spatial_dims)
        results[cat] = compute_auroc(test_labels, image_scores)
    return results

def compute_cil_metrics(auroc_matrix, n_tasks, n_cats, all_categories, tasks):
    am = np.array(auroc_matrix)
    final_mean = float(np.nanmean(am[-1]))
    task_means = [float(np.mean(am[t, ~np.isnan(am[t])])) for t in range(n_tasks) if (~np.isnan(am[t])).any()]
    avg_inc = float(np.mean(task_means)) if task_means else 0.0
    
    forgetting_vals = []
    for cat_idx in range(n_cats):
        col = am[:, cat_idx]
        valid = ~np.isnan(col)
        if valid.sum() > 1:
            best = np.nanmax(col)
            final = col[-1] if not np.isnan(col[-1]) else 0.0
            forgetting_vals.append(best - final)
    forgetting_rate = float(np.mean(forgetting_vals)) if forgetting_vals else 0.0
    
    return {
        "final_mean_auroc": final_mean,
        "avg_incremental_auroc": avg_inc,
        "forgetting_rate": forgetting_rate,
    }

def run_joint(cache):
    print(f"\nJoint (upper bound) — 4-task, 12 categories", flush=True)
    t0 = time.time()
    
    rng = np.random.RandomState(42)
    all_feats = []
    for cat in ALL_CATEGORIES:
        print(f"Loading {cat}...", flush=True)
        f = cache.load_train_features(cat)
        # Subsample immediately to avoid OOM (100k points per category is plenty)
        if f.shape[0] > 100000:
            indices = rng.choice(f.shape[0], 100000, replace=False)
            f = f[indices]
        all_feats.append(f)
    print(f"Loaded {sum([f.shape[0] for f in all_feats])} patches -> coreset {BUDGET}...", flush=True)
    combined = np.concatenate(all_feats, axis=0)
    coreset = greedy_coreset_selection(combined, budget=BUDGET)
    
    scorer = KNNScorer(k=K)
    scorer.fit(coreset)
    
    cat_aurocs = evaluate_cached(scorer, cache, ALL_CATEGORIES, cache.spatial_dims)
    for cat in ALL_CATEGORIES: print(f"  {cat}: {cat_aurocs[cat]:.4f}")
    
    n_tasks = len(TASKS)
    n_cats = len(ALL_CATEGORIES)
    auroc_matrix = np.full((n_tasks, n_cats), np.nan)
    cats_seen = []
    for t_idx, task in enumerate(TASKS):
        cats_seen.extend(task["categories"])
        for cat in cats_seen: auroc_matrix[t_idx, ALL_CATEGORIES.index(cat)] = cat_aurocs[cat]
            
    metrics = compute_cil_metrics(auroc_matrix, n_tasks, n_cats, ALL_CATEGORIES, TASKS)
    print(f"Joint done in {time.time()-t0:.1f}s — I-AUROC={metrics['final_mean_auroc']:.4f}")
    return metrics

def run_naive(cache):
    print(f"\nNaive (lower bound) — 4-task, 12 categories", flush=True)
    t0 = time.time()
    scorer = KNNScorer(k=K)
    n_tasks = len(TASKS)
    n_cats = len(ALL_CATEGORIES)
    auroc_matrix = np.full((n_tasks, n_cats), np.nan)
    categories_seen = []
    
    rng = np.random.RandomState(42)
    for task_idx, task in enumerate(TASKS):
        cats = task["categories"]
        print(f"  Task {task_idx}: loading {cats}", flush=True)
        task_feats = []
        for c in cats:
            f = cache.load_train_features(c)
            if f.shape[0] > 100000:
                indices = rng.choice(f.shape[0], 100000, replace=False)
                f = f[indices]
            task_feats.append(f)
        combined = np.concatenate(task_feats, axis=0)
        coreset = greedy_coreset_selection(combined, budget=BUDGET)
        scorer.fit(coreset)
        categories_seen.extend(cats)
        
        cat_aurocs = evaluate_cached(scorer, cache, categories_seen, cache.spatial_dims)
        for cat in categories_seen: auroc_matrix[task_idx, ALL_CATEGORIES.index(cat)] = cat_aurocs[cat]
            
    metrics = compute_cil_metrics(auroc_matrix, n_tasks, n_cats, ALL_CATEGORIES, TASKS)
    print(f"Naive done in {time.time()-t0:.1f}s — I-AUROC={metrics['final_mean_auroc']:.4f}")
    return metrics

def main():
    cache = FeatureCache(FEATURE_DIR)
    results = {}
    results["Joint (upper)"] = run_joint(cache)
    results["Naive (lower)"] = run_naive(cache)
    
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_visa_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to results/baseline_visa_results.json")

if __name__ == "__main__":
    main()
