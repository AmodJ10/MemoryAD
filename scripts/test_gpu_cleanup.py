"""Quick test: verify 3 RD4AD baselines run sequentially without GPU OOM."""
import sys, time, gc
sys.path.insert(0, ".")
import torch

TASKS = [{"categories": ["bottle", "cable"]}, {"categories": ["carpet", "grid"]}]
ALL_CATS = ["bottle", "cable", "carpet", "grid"]
config = {
    "backbone": {"name": "dinov2_vitb14", "layers": [7, 11], "input_size": 518, "batch_size": 4},
    "coreset": {"global_budget": 5000, "strategy": "proportional"},
    "scoring": {"k": 9, "image_score_method": "max"},
}

def gpu_mb():
    return torch.cuda.memory_allocated() / 1024**2

print(f"GPU before: {gpu_mb():.0f} MB")

# --- EWC ---
print("\n=== EWC (5 epochs) ===")
from src.baselines.ewc_baseline import EWCBaseline
b = EWCBaseline(config, ewc_lambda=5000, epochs=5)
print(f"  GPU after init: {gpu_mb():.0f} MB")
r = b.run("data/mvtec_ad", TASKS, ALL_CATS)
print(f"  EWC mean I-AUROC: {r['mean_auroc']:.4f}")
del b
gc.collect()
torch.cuda.empty_cache()
print(f"  GPU after cleanup: {gpu_mb():.0f} MB")

# --- LwF ---
print("\n=== LwF (5 epochs) ===")
from src.baselines.lwf_baseline import LwFBaseline
b = LwFBaseline(config, alpha=1.0, epochs=5)
print(f"  GPU after init: {gpu_mb():.0f} MB")
r = b.run("data/mvtec_ad", TASKS, ALL_CATS)
print(f"  LwF mean I-AUROC: {r['mean_auroc']:.4f}")
del b
gc.collect()
torch.cuda.empty_cache()
print(f"  GPU after cleanup: {gpu_mb():.0f} MB")

# --- Replay ---
print("\n=== Replay (5 epochs) ===")
from src.baselines.replay_baseline import ReplayBaseline
b = ReplayBaseline(config, buffer_per_category=10, epochs=5)
print(f"  GPU after init: {gpu_mb():.0f} MB")
r = b.run("data/mvtec_ad", TASKS, ALL_CATS)
print(f"  Replay mean I-AUROC: {r['mean_auroc']:.4f}")
del b
gc.collect()
torch.cuda.empty_cache()
print(f"  GPU after cleanup: {gpu_mb():.0f} MB")

print("\n=== ALL 3 BASELINES PASSED ===")
