# MemoryAD

**Architecture-Agnostic Continual Anomaly Detection via Adaptive Coreset Replay and Foundation Features**

## Overview

MemoryAD tackles the problem of **continual anomaly detection** in industrial visual inspection. When new product categories arrive on a production line, the system must learn to detect anomalies in them **without forgetting** how to inspect previous products.

Our approach:
1. **Frozen DINOv2** extracts rich patch-level features from normal images
2. An **adaptive coreset manager** maintains a fixed-budget memory bank that grows incrementally
3. **k-NN scoring** flags anomalies by measuring distance to stored normal patches
4. **Zero training, zero forgetting** — no gradient updates, no catastrophic forgetting

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download MVTec AD to data/mvtec_ad/
# https://www.mvtec.com/company/research/datasets/mvtec-ad

# 3. Run the main experiment (5 tasks × 3 categories)
python scripts/run_experiment.py --config configs/default.yaml --tasks configs/mvtec_5task.yaml --output results/E1_mvtec_5task

# 4. Run with different budget
python scripts/run_experiment.py --config configs/default.yaml --tasks configs/mvtec_5task.yaml --budget 5000 --output results/E3_budget_5k
```

## Project Structure

```
memoryad/
├── configs/              # YAML configs for experiments
├── src/
│   ├── backbones/        # Feature extractors (DINOv2, CLIP, WideResNet)
│   ├── coreset/          # Greedy selection + adaptive coreset manager
│   ├── scoring/          # k-NN anomaly scoring
│   ├── evaluation/       # AUROC, forgetting rate, CIL metrics
│   ├── data_utils/       # Dataset loaders (MVTec AD, VisA)
│   └── pipeline.py       # Main experiment orchestrator
├── scripts/              # CLI entry points
├── data/                 # Datasets (download separately)
└── results/              # Experiment outputs
```

## Experiments

| # | Experiment | Command |
|---|-----------|---------|
| E1 | Main result (MVTec 5-task) | `python scripts/run_experiment.py --tasks configs/mvtec_5task.yaml` |
| E2 | VisA generalization | `python scripts/run_experiment.py --tasks configs/visa_4task.yaml` |
| E3 | Budget ablation | Run E1 with `--budget 1000/5000/20000` |
| E4 | Backbone ablation | Run E1 with `--backbone dinov2_vitl14/wide_resnet50` |
| E5 | Scalability (15 tasks) | `python scripts/run_experiment.py --tasks configs/mvtec_15task.yaml` |
