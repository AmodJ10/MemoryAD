"""
MemoryAD — Run All Experiments (E1-E7)

Runs the complete experimental suite:
  E1: MVTec AD 5-task main result
  E2: VisA 4-task generalization
  E3: Budget ablation (1K/5K/10K/20K/50K)
  E4: Backbone ablation (DINOv2 only — CLIP/WRN need separate precompute)
  E5: 15-task scalability
  E6: CIL strategy comparison (proportional/weighted/recency)
  E7: Inference speed breakdown

Usage:
    .venv\\Scripts\\python.exe scripts\\run_all_experiments.py
    .venv\\Scripts\\python.exe scripts\\run_all_experiments.py --only E1 E3
    .venv\\Scripts\\python.exe scripts\\run_all_experiments.py --skip E2
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
MVTEC_15TASK = load_yaml("configs/mvtec_15task.yaml")
VISA_4TASK = load_yaml("configs/visa_4task.yaml")

MVTEC_FEATURE_DIR = "data/features/dinov2_vitb14"
VISA_FEATURE_DIR = "data/features/dinov2_vitb14_visa"


def run_experiment(config, task_config, output_dir, feature_dir, label=""):
    """Run a single pipeline experiment and return results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
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


# ─── E1: MVTec 5-task main result ─────────────────────────
def run_e1():
    config = copy.deepcopy(DEFAULT_CONFIG)
    return run_experiment(
        config, MVTEC_5TASK,
        output_dir="results/E1_mvtec_5task",
        feature_dir=MVTEC_FEATURE_DIR,
        label="E1: MVTec AD 5-task (DINOv2, budget=10K)",
    )


# ─── E2: VisA 4-task generalization ───────────────────────
def run_e2():
    config = copy.deepcopy(DEFAULT_CONFIG)
    # Check if VisA features exist
    if not Path(VISA_FEATURE_DIR).exists():
        print("\n[!] VisA features not found. Run precompute_features_visa.py first.")
        print(f"  Expected: {VISA_FEATURE_DIR}/")
        return None

    return run_experiment(
        config, VISA_4TASK,
        output_dir="results/E2_visa_4task",
        feature_dir=VISA_FEATURE_DIR,
        label="E2: VisA 4-task (DINOv2, budget=10K)",
    )


# ─── E3: Budget ablation ──────────────────────────────────
def run_e3():
    budgets = [1000, 5000, 10000, 20000, 50000]
    all_results = {}

    for budget in budgets:
        out_dir = f"results/E3_budget_{budget}"
        results_path = Path(out_dir) / "results.json"

        # Resume support: skip already-completed budget runs
        if results_path.exists():
            print(f"\n  [CACHED] E3 budget={budget} — loading from {results_path}")
            with open(results_path) as f:
                results = json.load(f)
            all_results[budget] = {
                "mean_auroc": results["final_mean_auroc"],
                "forgetting_rate": results["forgetting_rate"],
                "avg_incremental_auroc": results["avg_incremental_auroc"],
                "wall_time": results.get("wall_time", 0),
            }
            continue

        config = copy.deepcopy(DEFAULT_CONFIG)
        config["coreset"]["global_budget"] = budget

        results = run_experiment(
            config, MVTEC_5TASK,
            output_dir=out_dir,
            feature_dir=MVTEC_FEATURE_DIR,
            label=f"E3: Budget ablation (budget={budget})",
        )
        all_results[budget] = {
            "mean_auroc": results["final_mean_auroc"],
            "forgetting_rate": results["forgetting_rate"],
            "avg_incremental_auroc": results["avg_incremental_auroc"],
            "wall_time": results["wall_time"],
        }

    # Print summary table
    print(f"\n{'='*60}")
    print("E3: Budget Ablation Summary")
    print(f"{'='*60}")
    print(f"{'Budget':>8} | {'Mean I-AUROC':>12} | {'Forgetting':>10} | {'Avg Inc.':>9} | {'Time':>6}")
    print("-" * 55)
    for budget in budgets:
        r = all_results[budget]
        print(f"{budget:>8} | {r['mean_auroc']:>12.4f} | {r['forgetting_rate']:>10.4f} | {r['avg_incremental_auroc']:>9.4f} | {r['wall_time']:>5.0f}s")

    # Save combined results
    out_dir = Path("results/E3_budget_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


# ─── E4: Backbone ablation ────────────────────────────────
def run_e4():
    """Run ablation on different backbones."""
    backbones = {
        "DINOv2-B/14": "data/features/dinov2_vitb14",
        "CLIP-L/14": "data/features/clip_vitl14",
        "WRN-50": "data/features/wide_resnet50",
    }
    
    all_results = {}
    
    for name, feat_dir in backbones.items():
        if not Path(feat_dir).exists():
            print(f"\n[!] Features for {name} not found at {feat_dir}, skipping.")
            continue
            
        config = copy.deepcopy(DEFAULT_CONFIG)
        out_name = name.lower().replace("/", "_").replace("-", "")
        
        results = run_experiment(
            config, MVTEC_5TASK,
            output_dir=f"results/E4_backbone_{out_name}",
            feature_dir=feat_dir,
            label=f"E4: Backbone ablation ({name})",
        )
        all_results[name] = results

    print(f"\n{'='*60}")
    print("E4: Backbone Ablation Summary")
    print(f"{'='*60}")
    print(f"{'Backbone':>15} | {'Mean I-AUROC':>12} | {'Forgetting':>10}")
    print("-" * 45)
    for name, r in all_results.items():
        print(f"{name:>15} | {r['final_mean_auroc']:>12.4f} | {r['forgetting_rate']:>10.4f}")

    return all_results

    print(f"\n{'='*60}")
    print("E4: Backbone Ablation Summary")
    print(f"{'='*60}")
    print(f"{'Backbone':>20} | {'Mean I-AUROC':>12} | {'Forgetting':>10}")
    print("-" * 50)
    print(f"{'DINOv2-B/14':>20} | {results['final_mean_auroc']:>12.4f} | {results['forgetting_rate']:>10.4f}")
    print("  (CLIP-L/14 and WRN-50 require separate feature precomputation)")

    return {"dinov2_vitb14": results}


# ─── E5: 15-task scalability ──────────────────────────────
def run_e5():
    config = copy.deepcopy(DEFAULT_CONFIG)
    return run_experiment(
        config, MVTEC_15TASK,
        output_dir="results/E5_mvtec_15task",
        feature_dir=MVTEC_FEATURE_DIR,
        label="E5: MVTec AD 15-task scalability (DINOv2, budget=10K)",
    )


# ─── E6: CIL strategy comparison ─────────────────────────
def run_e6():
    strategies = ["proportional", "weighted", "recency"]
    all_results = {}

    for strategy in strategies:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["coreset"]["strategy"] = strategy

        results = run_experiment(
            config, MVTEC_5TASK,
            output_dir=f"results/E6_strategy_{strategy}",
            feature_dir=MVTEC_FEATURE_DIR,
            label=f"E6: CIL strategy ({strategy})",
        )
        all_results[strategy] = {
            "mean_auroc": results["final_mean_auroc"],
            "forgetting_rate": results["forgetting_rate"],
            "avg_incremental_auroc": results["avg_incremental_auroc"],
            "wall_time": results["wall_time"],
        }

    # Print summary
    print(f"\n{'='*60}")
    print("E6: CIL Strategy Comparison")
    print(f"{'='*60}")
    print(f"{'Strategy':>15} | {'Mean I-AUROC':>12} | {'Forgetting':>10} | {'Avg Inc.':>9}")
    print("-" * 55)
    for s in strategies:
        r = all_results[s]
        print(f"{s:>15} | {r['mean_auroc']:>12.4f} | {r['forgetting_rate']:>10.4f} | {r['avg_incremental_auroc']:>9.4f}")

    out_dir = Path("results/E6_strategy_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


# ─── E7: Inference speed breakdown ────────────────────────
def run_e7():
    """Measure component-level timing for a single 5-task run."""
    config = copy.deepcopy(DEFAULT_CONFIG)

    from src.data_utils.feature_cache import FeatureCache
    from src.coreset.adaptive_manager import AdaptiveCoresetManager
    from src.scoring.knn_scorer import KNNScorer
    from src.evaluation.metrics import compute_auroc

    cache = FeatureCache(MVTEC_FEATURE_DIR)
    task_config = MVTEC_5TASK
    tasks = task_config["tasks"]
    all_categories = []
    for t in tasks:
        all_categories.extend(t["categories"])

    # Timing accumulators
    t_feature_load = 0
    t_coreset = 0
    t_scoring_fit = 0
    t_evaluation = 0
    n_images_scored = 0

    manager = AdaptiveCoresetManager(
        global_budget=config["coreset"]["global_budget"],
        strategy=config["coreset"]["strategy"],
        min_per_category=config["coreset"]["min_per_category"],
    )
    scorer = KNNScorer(k=config["scoring"]["k"])
    categories_seen = []

    print(f"\n{'='*60}")
    print("  E7: Inference Speed Breakdown")
    print(f"{'='*60}")

    for task_idx, task in enumerate(tasks):
        cats = task["categories"]

        # Time: Feature loading
        t0 = time.time()
        task_features = {}
        for cat in cats:
            task_features[cat] = cache.load_train_features(cat)
        t_feature_load += time.time() - t0

        # Time: Coreset update
        t0 = time.time()
        manager.add_task(task_features)
        t_coreset += time.time() - t0

        # Time: Scorer fit
        t0 = time.time()
        global_coreset = manager.get_global_coreset()
        scorer.fit(global_coreset)
        t_scoring_fit += time.time() - t0

        categories_seen.extend(cats)

        # Time: Evaluation
        t0 = time.time()
        for cat in categories_seen:
            test_features, test_labels = cache.load_test_data(cat)
            scores, _ = scorer.score_batch(test_features, cache.spatial_dims)
            n_images_scored += len(test_labels)
        t_evaluation += time.time() - t0

    total = t_feature_load + t_coreset + t_scoring_fit + t_evaluation

    results = {
        "feature_load_s": round(t_feature_load, 3),
        "coreset_update_s": round(t_coreset, 3),
        "scorer_fit_s": round(t_scoring_fit, 3),
        "evaluation_s": round(t_evaluation, 3),
        "total_s": round(total, 3),
        "images_scored": n_images_scored,
        "ms_per_image": round(t_evaluation / max(n_images_scored, 1) * 1000, 2),
    }

    print(f"\n  Component Breakdown:")
    print(f"  {'Feature loading':.<30} {t_feature_load:>8.3f}s")
    print(f"  {'Coreset update':.<30} {t_coreset:>8.3f}s")
    print(f"  {'k-NN scorer fit':.<30} {t_scoring_fit:>8.3f}s")
    print(f"  {'Evaluation':.<30} {t_evaluation:>8.3f}s")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':.<30} {total:>8.3f}s")
    print(f"\n  Images scored: {n_images_scored}")
    print(f"  Throughput: {results['ms_per_image']:.2f} ms/image")

    out_dir = Path("results/E7_inference_speed")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─── Main ─────────────────────────────────────────────────
EXPERIMENTS = {
    "E1": ("MVTec 5-task main result", run_e1),
    "E2": ("VisA 4-task generalization", run_e2),
    "E3": ("Budget ablation", run_e3),
    "E4": ("Backbone ablation", run_e4),
    "E5": ("15-task scalability", run_e5),
    "E6": ("CIL strategy comparison", run_e6),
    "E7": ("Inference speed", run_e7),
}


def main():
    parser = argparse.ArgumentParser(description="MemoryAD — Run All Experiments")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only specific experiments (e.g. --only E1 E3)")
    parser.add_argument("--skip", nargs="+", default=None,
                        help="Skip specific experiments (e.g. --skip E2)")
    args = parser.parse_args()

    # Determine which experiments to run
    if args.only:
        to_run = [e.upper() for e in args.only]
    else:
        to_run = list(EXPERIMENTS.keys())

    if args.skip:
        skip = {e.upper() for e in args.skip}
        to_run = [e for e in to_run if e not in skip]

    print("=" * 60)
    print("MemoryAD — Full Experiment Suite")
    print("=" * 60)
    print(f"Running: {', '.join(to_run)}")
    print()

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

    # Final summary
    print(f"\n\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for exp_id in to_run:
        if exp_id in all_results:
            r = all_results[exp_id]
            if r is None:
                print(f"  {exp_id}: SKIPPED (missing data)")
            elif "error" in r:
                print(f"  {exp_id}: FAILED — {r['error']}")
            elif isinstance(r, dict) and "final_mean_auroc" in r:
                print(f"  {exp_id}: Mean I-AUROC = {r['final_mean_auroc']:.4f}, "
                      f"Forgetting = {r['forgetting_rate']:.4f}")
            else:
                print(f"  {exp_id}: completed")

    print(f"\nTotal wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to: results/")


if __name__ == "__main__":
    main()
