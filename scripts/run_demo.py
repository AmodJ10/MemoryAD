"""Run the tiny MemoryAD demo configuration."""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_experiment import load_yaml
from src.pipeline import MemoryADPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MemoryAD tiny demo")
    parser.add_argument("--config", default="configs/demo_default.yaml")
    parser.add_argument("--tasks", default="configs/demo_tasks.yaml")
    parser.add_argument("--output", default="results/demo_judge")
    parser.add_argument("--feature-dir", default="demo/demo_features/dinov2_vitb14")
    parser.add_argument("--coreset-cache-dir", default="demo/demo_coreset_cache")
    parser.add_argument("--live", action="store_true", help="Extract features live from demo dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_yaml(args.config)
    task_config = load_yaml(args.tasks)

    pipeline = MemoryADPipeline(
        config=config,
        task_config=task_config,
        output_dir=args.output,
        use_cache=not args.live,
        feature_dir=args.feature_dir,
        coreset_cache_dir=args.coreset_cache_dir,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
