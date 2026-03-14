"""
MemoryAD — CLI entry point.

Usage:
    python -m scripts.run_experiment --config configs/default.yaml --tasks configs/mvtec_5task.yaml --output results/E1_mvtec_5task

    python -m scripts.run_experiment --config configs/default.yaml --tasks configs/visa_4task.yaml --output results/E2_visa_4task
"""

import argparse
import yaml
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import MemoryADPipeline


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(default: dict, override: dict) -> dict:
    """Deep-merge override into default."""
    merged = default.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def main():
    parser = argparse.ArgumentParser(description="MemoryAD — Continual Anomaly Detection")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to default config YAML")
    parser.add_argument("--tasks", type=str, required=True,
                        help="Path to task split YAML (e.g. configs/mvtec_5task.yaml)")
    parser.add_argument("--output", type=str, default="results/experiment",
                        help="Output directory for results")
    parser.add_argument("--budget", type=int, default=None,
                        help="Override coreset budget")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Override backbone name")
    parser.add_argument("--live", action="store_true",
                        help="Disable feature cache and extract features live")
    parser.add_argument("--feature-dir", type=str, default="data/features/dinov2_vitb14",
                        help="Directory containing cached features")
    parser.add_argument("--coreset-cache-dir", type=str, default="data/coreset_cache",
                        help="Directory for cached coreset files")

    args = parser.parse_args()

    # Load configs
    config = load_yaml(args.config)
    task_config = load_yaml(args.tasks)

    # Apply CLI overrides
    if args.budget is not None:
        config["coreset"]["global_budget"] = args.budget
    if args.backbone is not None:
        config["backbone"]["name"] = args.backbone

    # Run pipeline
    pipeline = MemoryADPipeline(
        config=config,
        task_config=task_config,
        output_dir=args.output,
        use_cache=not args.live,
        feature_dir=args.feature_dir,
        coreset_cache_dir=args.coreset_cache_dir,
    )

    results = pipeline.run()

    print(f"\nExperiment complete. Results saved to: {args.output}/results.json")


if __name__ == "__main__":
    main()
