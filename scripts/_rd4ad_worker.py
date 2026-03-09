"""Worker script for running RD4AD baselines in a subprocess.

Called by test_baselines.py. Runs a single baseline method and prints
RESULT_JSON:<value> as the last line for the parent to parse.

All stdout is flushed immediately so the parent sees live progress.
"""
import sys, json, time
sys.path.insert(0, ".")

# Force unbuffered output so parent process sees live progress
sys.stdout.reconfigure(line_buffering=True)

TASKS = [{"categories": ["bottle", "cable"]}, {"categories": ["carpet", "grid"]}]
ALL_CATS = ["bottle", "cable", "carpet", "grid"]

method = sys.argv[1]
epochs = int(sys.argv[2])

config = {
    "backbone": {"name": "dinov2_vitb14", "layers": [7, 11], "input_size": 518, "batch_size": 4},
    "coreset": {"global_budget": 5000, "strategy": "proportional"},
    "scoring": {"k": 9, "image_score_method": "max"},
}

t0 = time.time()
print(f"[Worker] Starting {method.upper()} with {epochs} epochs...")

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
print("RESULT_JSON:" + json.dumps(r["mean_auroc"]))
