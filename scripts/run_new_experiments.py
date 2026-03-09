"""
MemoryAD — New GPU Experiments (E8-E12)

Runs ablation and robustness experiments:
  E8:  k-NN k ablation (k in {1,3,5,9,15,25})
  E9:  Task ordering sensitivity (3 permutations)
  E10: Statistical significance (3 seeds, mean±std)
  E11: Layer selection ablation (requires feature re-extraction)
  E12: P-AUROC pixel-level evaluation

Usage:
    .venv\\Scripts\\python.exe scripts/run_new_experiments.py --only E8 E9 E10 E12
    .venv\\Scripts\\python.exe scripts/run_new_experiments.py --only E11  # after feature extraction
"""

import sys, os, time, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
import numpy as np
from pathlib import Path

from src.pipeline import MemoryADPipeline


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Configs ───────────────────────────────────────────────
DEFAULT_CONFIG = load_yaml("configs/default.yaml")
MVTEC_5TASK = load_yaml("configs/mvtec_5task.yaml")

MVTEC_FEATURE_DIR = "data/features/dinov2_vitb14"

ALL_CATEGORIES = [
    "bottle", "cable", "capsule",   # Task 0
    "carpet", "grid", "hazelnut",   # Task 1
    "leather", "metal_nut", "pill", # Task 2
    "screw", "tile", "toothbrush",  # Task 3
    "transistor", "wood", "zipper", # Task 4
]


def run_experiment(config, task_config, output_dir, feature_dir, label=""):
    """Run a single pipeline experiment and return results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    results_path = Path(output_dir) / "results.json"
    if results_path.exists():
        print(f"  [CACHED] Loading from {results_path}")
        with open(results_path) as f:
            return json.load(f)

    t0 = time.time()
    pipeline = MemoryADPipeline(
        config=config,
        task_config=task_config,
        output_dir=output_dir,
        use_cache=True,
        feature_dir=feature_dir,
    )
    results = pipeline.run()
    elapsed = time.time() - t0
    results["wall_time"] = elapsed
    print(f"\n  -> {label} done in {elapsed:.1f}s")
    return results


# ═══════════════════════════════════════════════════════════
# E8: k-NN k Ablation
# ═══════════════════════════════════════════════════════════
def run_e8():
    """Run k ablation: k in {1, 3, 5, 9, 15, 25}."""
    k_values = [1, 3, 5, 9, 15, 25]
    all_results = {}

    for k in k_values:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["scoring"]["k"] = k
        out_dir = f"results/E8_k_ablation/k_{k}"

        results = run_experiment(
            config, MVTEC_5TASK,
            output_dir=out_dir,
            feature_dir=MVTEC_FEATURE_DIR,
            label=f"E8: k ablation (k={k})",
        )
        all_results[k] = {
            "k": k,
            "mean_auroc": results["final_mean_auroc"],
            "forgetting_rate": results["forgetting_rate"],
            "avg_incremental_auroc": results["avg_incremental_auroc"],
        }

    # Print summary
    print(f"\n{'='*60}")
    print("E8: k Ablation Summary")
    print(f"{'='*60}")
    print(f"{'k':>4} | {'Mean I-AUROC':>12} | {'Forgetting':>10} | {'Avg Inc.':>9}")
    print("-" * 45)
    for k in k_values:
        r = all_results[k]
        print(f"{k:>4} | {r['mean_auroc']:>12.4f} | {r['forgetting_rate']:>10.4f} | {r['avg_incremental_auroc']:>9.4f}")

    # Save summary
    out_dir = Path("results/E8_k_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


# ═══════════════════════════════════════════════════════════
# E9: Task Ordering Sensitivity
# ═══════════════════════════════════════════════════════════
def run_e9():
    """Run task ordering sensitivity with 3 different permutations."""
    ordering_seeds = [42, 123, 456]
    all_results = {}

    for seed in ordering_seeds:
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(ALL_CATEGORIES).tolist()

        # Group into 5 tasks of 3
        task_config = copy.deepcopy(MVTEC_5TASK)
        task_config["tasks"] = [
            {"task_id": i, "categories": shuffled[i*3:(i+1)*3]}
            for i in range(5)
        ]

        config = copy.deepcopy(DEFAULT_CONFIG)
        out_dir = f"results/E9_ordering/seed_{seed}"

        order_str = " | ".join([",".join(shuffled[i*3:(i+1)*3]) for i in range(5)])
        print(f"\n  Order (seed={seed}): {order_str}")

        results = run_experiment(
            config, task_config,
            output_dir=out_dir,
            feature_dir=MVTEC_FEATURE_DIR,
            label=f"E9: Task ordering (seed={seed})",
        )
        all_results[seed] = {
            "seed": seed,
            "ordering": shuffled,
            "mean_auroc": results["final_mean_auroc"],
            "forgetting_rate": results["forgetting_rate"],
            "avg_incremental_auroc": results["avg_incremental_auroc"],
        }

    # Compute stats
    aurocs = [r["mean_auroc"] for r in all_results.values()]
    frs = [r["forgetting_rate"] for r in all_results.values()]

    print(f"\n{'='*60}")
    print("E9: Task Ordering Sensitivity Summary")
    print(f"{'='*60}")
    for seed in ordering_seeds:
        r = all_results[seed]
        print(f"  Seed {seed}: I-AUROC={r['mean_auroc']:.4f}, FR={r['forgetting_rate']:.4f}")
    print(f"\n  Mean I-AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  Mean FR:      {np.mean(frs):.4f} ± {np.std(frs):.4f}")

    summary = {
        "per_seed": all_results,
        "mean_auroc": float(np.mean(aurocs)),
        "std_auroc": float(np.std(aurocs)),
        "mean_fr": float(np.mean(frs)),
        "std_fr": float(np.std(frs)),
    }

    out_dir = Path("results/E9_ordering")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ═══════════════════════════════════════════════════════════
# E10: Statistical Significance (Multiple Seeds)
# ═══════════════════════════════════════════════════════════
def run_e10():
    """Run E1 with 3 different coreset seeds for error bars."""
    seeds = [42, 123, 456]
    all_results = {}

    for seed in seeds:
        config = copy.deepcopy(DEFAULT_CONFIG)
        out_dir = f"results/E10_seeds/seed_{seed}"

        # Override coreset cache dir so different seeds don't collide
        pipeline = MemoryADPipeline(
            config=config,
            task_config=MVTEC_5TASK,
            output_dir=out_dir,
            use_cache=True,
            feature_dir=MVTEC_FEATURE_DIR,
            coreset_cache_dir=f"data/coreset_cache_seed{seed}",
        )

        results_path = Path(out_dir) / "results.json"
        if results_path.exists():
            print(f"\n  [CACHED] Seed {seed}: loading from {results_path}")
            with open(results_path) as f:
                results = json.load(f)
        else:
            # Set the seed on the coreset manager
            pipeline.manager.seed = seed
            print(f"\n{'='*60}")
            print(f"  E10: Statistical significance (seed={seed})")
            print(f"{'='*60}")
            t0 = time.time()
            results = pipeline.run()
            results["wall_time"] = time.time() - t0

        all_results[seed] = {
            "seed": seed,
            "mean_auroc": results["final_mean_auroc"],
            "forgetting_rate": results["forgetting_rate"],
            "avg_incremental_auroc": results["avg_incremental_auroc"],
            "per_category_final": results.get("per_category_final", {}),
        }

    # Compute stats
    aurocs = [r["mean_auroc"] for r in all_results.values()]
    frs = [r["forgetting_rate"] for r in all_results.values()]

    print(f"\n{'='*60}")
    print("E10: Statistical Significance Summary")
    print(f"{'='*60}")
    for seed in seeds:
        r = all_results[seed]
        print(f"  Seed {seed}: I-AUROC={r['mean_auroc']:.4f}, FR={r['forgetting_rate']:.4f}")
    print(f"\n  I-AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  FR:      {np.mean(frs):.4f} ± {np.std(frs):.4f}")

    summary = {
        "per_seed": all_results,
        "mean_auroc": float(np.mean(aurocs)),
        "std_auroc": float(np.std(aurocs)),
        "mean_fr": float(np.mean(frs)),
        "std_fr": float(np.std(frs)),
    }

    out_dir = Path("results/E10_seeds")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ═══════════════════════════════════════════════════════════
# E11: Layer Selection Ablation
# ═══════════════════════════════════════════════════════════
def run_e11():
    """Run layer ablation. Requires pre-extracted features for each config."""
    layer_configs = {
        "l7_11":    {"layers": [7, 11],    "feature_dir": "data/features/dinov2_vitb14"},
        "l6_11":    {"layers": [6, 11],    "feature_dir": "data/features/dinov2_vitb14_l6_11"},
        "l7_9_11":  {"layers": [7, 9, 11], "feature_dir": "data/features/dinov2_vitb14_l7_9_11"},
        "l11_only": {"layers": [11],       "feature_dir": "data/features/dinov2_vitb14_l11"},
    }

    all_results = {}

    for name, lconfig in layer_configs.items():
        feature_dir = lconfig["feature_dir"]

        # Check if features exist
        if not Path(feature_dir).exists() or not (Path(feature_dir) / "bottle" / "train_features.npy").exists():
            print(f"\n  [SKIP] {name}: features not found at {feature_dir}")
            print(f"  Run: .venv\\Scripts\\python.exe scripts/precompute_features_layers.py --layers {' '.join(map(str, lconfig['layers']))}")
            continue

        config = copy.deepcopy(DEFAULT_CONFIG)
        config["backbone"]["layers"] = lconfig["layers"]
        out_dir = f"results/E11_layers/{name}"

        results = run_experiment(
            config, MVTEC_5TASK,
            output_dir=out_dir,
            feature_dir=feature_dir,
            label=f"E11: Layer ablation ({name}, layers={lconfig['layers']})",
        )
        all_results[name] = {
            "layers": lconfig["layers"],
            "layers_str": name,
            "mean_auroc": results["final_mean_auroc"],
            "forgetting_rate": results["forgetting_rate"],
            "avg_incremental_auroc": results["avg_incremental_auroc"],
        }

    if all_results:
        print(f"\n{'='*60}")
        print("E11: Layer Ablation Summary")
        print(f"{'='*60}")
        print(f"{'Layers':<15} | {'Mean I-AUROC':>12} | {'Forgetting':>10} | {'Avg Inc.':>9}")
        print("-" * 55)
        for name, r in all_results.items():
            print(f"{name:<15} | {r['mean_auroc']:>12.4f} | {r['forgetting_rate']:>10.4f} | {r['avg_incremental_auroc']:>9.4f}")

    out_dir = Path("results/E11_layers")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


# ═══════════════════════════════════════════════════════════
# E12: P-AUROC (Pixel-Level Evaluation)
# ═══════════════════════════════════════════════════════════
def run_e12():
    """Run P-AUROC evaluation using cached masks."""
    from src.data_utils.feature_cache import FeatureCache
    from src.coreset.adaptive_manager import AdaptiveCoresetManager
    from src.scoring.knn_scorer import KNNScorer
    from src.evaluation.metrics import compute_auroc, compute_pixel_auroc
    from scipy.ndimage import zoom

    config = copy.deepcopy(DEFAULT_CONFIG)
    cache = FeatureCache(MVTEC_FEATURE_DIR)
    task_config = MVTEC_5TASK
    tasks = task_config["tasks"]

    # Check if masks are cached
    sample_mask_path = Path(MVTEC_FEATURE_DIR) / "bottle" / "test_masks.npy"
    if not sample_mask_path.exists():
        print("\n  [ERROR] Test masks not cached. Run first:")
        print("  .venv\\Scripts\\python.exe scripts/cache_test_masks.py")
        return None

    manager = AdaptiveCoresetManager(
        global_budget=config["coreset"]["global_budget"],
        strategy=config["coreset"]["strategy"],
        min_per_category=config["coreset"]["min_per_category"],
    )
    scorer = KNNScorer(k=config["scoring"]["k"])
    categories_seen = []
    spatial_dims = cache.spatial_dims  # (37, 37)

    print(f"\n{'='*60}")
    print("  E12: P-AUROC Pixel-Level Evaluation")
    print(f"{'='*60}")

    per_category_pauroc = {}
    per_category_iauroc = {}

    for task_idx, task in enumerate(tasks):
        cats = task["categories"]

        # Load and add features
        task_features = {}
        for cat in cats:
            task_features[cat] = cache.load_train_features(cat)
        manager.add_task(task_features)
        categories_seen.extend(cats)

        # Fit scorer
        global_coreset = manager.get_global_coreset()
        scorer.fit(global_coreset)

    # Final evaluation on all categories with P-AUROC
    print(f"\n  Evaluating P-AUROC on {len(categories_seen)} categories...")
    for cat in categories_seen:
        test_features, test_labels = cache.load_test_data(cat)
        test_masks = np.load(Path(MVTEC_FEATURE_DIR) / cat / "test_masks.npy")

        # Score batch → get anomaly maps
        image_scores, anomaly_maps = scorer.score_batch(test_features, spatial_dims)

        # I-AUROC
        i_auroc = compute_auroc(test_labels, image_scores)
        per_category_iauroc[cat] = i_auroc

        # P-AUROC: upsample anomaly maps to mask size
        if anomaly_maps is not None and test_masks is not None:
            # anomaly_maps: [N, H_patch, W_patch]  (37×37)
            # test_masks: [N, 1, H_mask, W_mask]   (518×518)
            N = anomaly_maps.shape[0]
            H_mask = test_masks.shape[2] if test_masks.ndim == 4 else test_masks.shape[1]
            W_mask = test_masks.shape[3] if test_masks.ndim == 4 else test_masks.shape[2]

            # Upsample anomaly maps to mask resolution
            scale_h = H_mask / spatial_dims[0]
            scale_w = W_mask / spatial_dims[1]
            upsampled_maps = np.zeros((N, H_mask, W_mask), dtype=np.float32)
            for i in range(N):
                upsampled_maps[i] = zoom(anomaly_maps[i], (scale_h, scale_w), order=1)

            # Squeeze masks to [N, H, W]
            if test_masks.ndim == 4:
                gt_masks = test_masks[:, 0, :, :]
            else:
                gt_masks = test_masks

            # Only compute if there are anomalous images
            has_anomalous = np.any(gt_masks > 0)
            if has_anomalous:
                p_auroc = compute_pixel_auroc(gt_masks, upsampled_maps)
            else:
                p_auroc = 0.0
        else:
            p_auroc = 0.0

        per_category_pauroc[cat] = p_auroc
        print(f"    {cat}: I-AUROC={i_auroc:.4f}, P-AUROC={p_auroc:.4f}")

    # Summary
    valid_paurocs = [v for v in per_category_pauroc.values() if v > 0]
    mean_iauroc = np.mean(list(per_category_iauroc.values()))
    mean_pauroc = np.mean(valid_paurocs) if valid_paurocs else 0.0

    print(f"\n  Mean I-AUROC: {mean_iauroc:.4f}")
    print(f"  Mean P-AUROC: {mean_pauroc:.4f}")

    results = {
        "mean_iauroc": float(mean_iauroc),
        "mean_pauroc": float(mean_pauroc),
        "per_category_iauroc": {k: float(v) for k, v in per_category_iauroc.items()},
        "per_category_pauroc": {k: float(v) for k, v in per_category_pauroc.items()},
    }

    out_dir = Path("results/E12_pauroc")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─── Main ─────────────────────────────────────────────────
EXPERIMENTS = {
    "E8":  ("k ablation", run_e8),
    "E9":  ("Task ordering sensitivity", run_e9),
    "E10": ("Statistical significance", run_e10),
    "E11": ("Layer selection ablation", run_e11),
    "E12": ("P-AUROC pixel-level", run_e12),
}


def main():
    parser = argparse.ArgumentParser(description="MemoryAD — New Experiments (E8-E12)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only specific experiments (e.g. --only E8 E9)")
    args = parser.parse_args()

    if args.only:
        to_run = [e.upper() for e in args.only]
    else:
        to_run = list(EXPERIMENTS.keys())

    print("=" * 60)
    print("MemoryAD — New Experiment Suite (E8-E12)")
    print("=" * 60)
    print(f"Running: {', '.join(to_run)}")

    all_results = {}
    t_total = time.time()

    for exp_id in to_run:
        if exp_id not in EXPERIMENTS:
            print(f"Unknown experiment: {exp_id}, skipping")
            continue

        label, fn = EXPERIMENTS[exp_id]
        print(f"\n{'#'*60}")
        print(f"# {exp_id}: {label}")
        print(f"{'#'*60}")

        try:
            result = fn()
            all_results[exp_id] = result
            print(f"\n[OK] {exp_id} completed successfully")
        except Exception as e:
            print(f"\n[FAIL] {exp_id} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_id] = {"error": str(e)}

    elapsed = time.time() - t_total
    print(f"\nTotal wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
