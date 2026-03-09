"""Worker script for running RD4AD baselines (full 5-task, 15-category).

Called by run_baselines_full.py. Runs a single baseline method and prints
RESULT_JSON:<json> as the last line for the parent to parse.
"""
import sys, json, time
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

TASKS = [
    {"categories": ["bottle", "cable", "capsule"]},
    {"categories": ["carpet", "grid", "hazelnut"]},
    {"categories": ["leather", "metal_nut", "pill"]},
    {"categories": ["screw", "tile", "toothbrush"]},
    {"categories": ["transistor", "wood", "zipper"]},
]
ALL_CATS = [c for t in TASKS for c in t["categories"]]

method = sys.argv[1]
epochs = int(sys.argv[2])

config = {
    "backbone": {"name": "dinov2_vitb14", "layers": [7, 11], "input_size": 518, "batch_size": 4},
    "coreset": {"global_budget": 10000, "strategy": "proportional"},
    "scoring": {"k": 9, "image_score_method": "max"},
}

t0 = time.time()
print(f"[Worker] Starting {method.upper()} with {epochs} epochs, 5 tasks, 15 categories...")

if method == "ewc":
    from src.baselines.ewc_baseline import EWCBaseline
    b = EWCBaseline(config, ewc_lambda=5000, epochs=epochs)
    r = b.run("data/mvtec_ad", TASKS, ALL_CATS)
elif method == "lwf":
    from src.baselines.lwf_baseline import LwFBaseline
    b = LwFBaseline(config, alpha=1.0, epochs=epochs)
    r = b.run("data/mvtec_ad", TASKS, ALL_CATS)
elif method == "replay":
    from src.baselines.replay_baseline import ReplayBaseline
    b = ReplayBaseline(config, buffer_per_category=10, epochs=epochs)
    r = b.run("data/mvtec_ad", TASKS, ALL_CATS)
else:
    print(f"Unknown method: {method}")
    sys.exit(1)

elapsed = time.time() - t0
print(f"[Worker] {method.upper()} completed in {elapsed:.0f}s")
print("RESULT_JSON:" + json.dumps({"auroc_matrix": r["auroc_matrix"], "mean_auroc": r["mean_auroc"]}))
