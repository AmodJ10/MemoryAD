"""
Train full MemoryAD on MVTec-15 (cached features), export a CPU bundle,
create a compact 120-image mini test set, and build a polished visual demo.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import MemoryADPipeline
from scripts.infer_with_bundle import infer_images

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def resolve_mask(gt_root: Path, defect_type: str, image_name: str) -> Path | None:
    base = gt_root / defect_type / image_name
    if base.exists():
        return base

    png = base.with_suffix(".png")
    if png.exists():
        return png

    stem_mask = (gt_root / defect_type / Path(image_name).stem).with_name(f"{Path(image_name).stem}_mask.png")
    if stem_mask.exists():
        return stem_mask

    return None


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_mini_testset(
    source_root: Path,
    target_root: Path,
    categories: list[str],
    good_per_cat: int = 4,
    bad_per_cat: int = 4,
) -> dict:
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    summary = {"categories": {}, "totals": {"images": 0, "good": 0, "anomalous": 0}}

    for cat in categories:
        cat_src = source_root / cat
        test_good_dir = cat_src / "test" / "good"
        defect_dirs = sorted([d for d in (cat_src / "test").iterdir() if d.is_dir() and d.name != "good"])
        if not defect_dirs:
            raise RuntimeError(f"No anomaly defect folders for category: {cat}")

        chosen_defect = defect_dirs[0]
        good_imgs = sorted([p for p in test_good_dir.iterdir() if p.suffix.lower() in VALID_EXTS])[:good_per_cat]
        bad_imgs = sorted([p for p in chosen_defect.iterdir() if p.suffix.lower() in VALID_EXTS])[:bad_per_cat]

        for p in good_imgs:
            copy_file(p, target_root / cat / "test" / "good" / p.name)

        for p in bad_imgs:
            copy_file(p, target_root / cat / "test" / chosen_defect.name / p.name)
            mask = resolve_mask(cat_src / "ground_truth", chosen_defect.name, p.name)
            if mask is not None:
                copy_file(mask, target_root / cat / "ground_truth" / chosen_defect.name / mask.name)

        cat_good = len(good_imgs)
        cat_bad = len(bad_imgs)
        summary["categories"][cat] = {
            "defect_type": chosen_defect.name,
            "good": cat_good,
            "anomalous": cat_bad,
            "total": cat_good + cat_bad,
        }
        summary["totals"]["images"] += cat_good + cat_bad
        summary["totals"]["good"] += cat_good
        summary["totals"]["anomalous"] += cat_bad

    return summary


def export_bundle(
    pipeline: MemoryADPipeline,
    config: dict,
    task_cfg: dict,
    output_root: Path,
    training_results: dict,
) -> Path:
    bundle_dir = output_root / "model_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    coreset = pipeline.manager.get_global_coreset().astype(np.float32)
    np.save(bundle_dir / "coreset_global.npy", coreset)

    metadata = {
        "name": "MemoryAD MVTec Full CPU Bundle",
        "dataset": task_cfg["dataset"],
        "categories": pipeline.all_categories,
        "spatial_dims": list(pipeline.spatial_dims),
        "feature_dim": int(coreset.shape[1]),
        "coreset_size": int(coreset.shape[0]),
        "backbone": config["backbone"],
        "scoring": config["scoring"],
        "coreset": config["coreset"],
        "training_summary": {
            "final_mean_auroc": float(training_results.get("final_mean_auroc", 0.0)),
            "avg_incremental_auroc": float(training_results.get("avg_incremental_auroc", 0.0)),
            "forgetting_rate": float(training_results.get("forgetting_rate", 0.0)),
            "total_time_seconds": float(training_results.get("total_time_seconds", 0.0)),
        },
    }

    with (bundle_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    save_yaml(bundle_dir / "config_runtime.yaml", config)
    save_yaml(bundle_dir / "tasks_runtime.yaml", task_cfg)

    return bundle_dir


def build_visual_report(infer_dir: Path, report_dir: Path, title: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    pred_path = infer_dir / "predictions.json"
    with pred_path.open("r", encoding="utf-8") as f:
        preds = json.load(f)

    top = preds[:24]

    cards = []
    for idx, row in enumerate(top, start=1):
        rel_overlay = Path(row["overlay_path"]).relative_to(report_dir.parent)
        cards.append(
            f"""
            <article class=\"card\">
              <div class=\"meta\">#{idx} | score: {row['anomaly_score']:.6f}</div>
              <img src=\"../{str(rel_overlay).replace('\\\\', '/')}\" alt=\"overlay\" />
              <div class=\"path\">{row['image_path']}</div>
            </article>
            """
        )

    html = f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    :root {{
      --bg1: #0b132b;
      --bg2: #1c2541;
      --panel: #f9fafb;
      --ink: #111827;
      --muted: #4b5563;
      --accent: #00b4d8;
      --accent2: #fb8500;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background: radial-gradient(1200px 700px at 15% -10%, #3a506b 0%, transparent 60%),
                  radial-gradient(900px 550px at 100% 0%, #5bc0be 0%, transparent 50%),
                  linear-gradient(145deg, var(--bg1), var(--bg2));
      min-height: 100vh;
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 28px 18px 42px; }}
    .hero {{
      background: linear-gradient(120deg, rgba(0,180,216,0.22), rgba(251,133,0,0.20));
      border: 1px solid rgba(255,255,255,0.2);
      border-radius: 16px;
      color: #ecfeff;
      padding: 18px 20px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.25);
      margin-bottom: 20px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 1.8rem; letter-spacing: 0.2px; }}
    .subtitle {{ margin: 0; opacity: 0.94; font-size: 0.98rem; }}
    .grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    }}
    .card {{
      background: var(--panel);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 10px 22px rgba(0,0,0,0.18);
      transform: translateY(0);
      transition: transform 160ms ease, box-shadow 160ms ease;
    }}
    .card:hover {{ transform: translateY(-3px); box-shadow: 0 16px 28px rgba(0,0,0,0.22); }}
    .meta {{
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      color: white;
      padding: 8px 10px;
      font-weight: 700;
      font-size: 0.88rem;
    }}
    .card img {{ width: 100%; display: block; aspect-ratio: 1 / 1; object-fit: cover; }}
    .path {{ padding: 9px 10px 12px; font-size: 0.78rem; color: var(--muted); word-break: break-all; }}
  </style>
</head>
<body>
  <main class=\"wrap\">
    <section class=\"hero\">
      <h1>{title}</h1>
      <p class=\"subtitle\">Top anomaly-ranked overlays from the exported CPU bundle on the 120-image mini MVTec test pack.</p>
    </section>
    <section class=\"grid\">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""

    with (report_dir / "index.html").open("w", encoding="utf-8") as f:
        f.write(html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full MVTec model, export bundle, and build visual demo")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tasks", default="configs/mvtec_15task.yaml")
    parser.add_argument("--feature-dir", default="data/features/dinov2_vitb14")
    parser.add_argument("--dataset-root", default="data/mvtec_ad")
    parser.add_argument("--output-root", default="exports/mvtec_full_cpu")
    parser.add_argument("--mini-size", type=int, default=120, help="Target mini test set size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(Path(args.config))
    task_cfg = load_yaml(Path(args.tasks))

    cfg["backbone"]["use_fp16"] = False
    cfg["backbone"]["batch_size"] = max(1, int(cfg["backbone"].get("batch_size", 4)))

    print("[1/5] Training full MVTec-15 run using cached features...")
    pipeline = MemoryADPipeline(
        config=cfg,
        task_config=task_cfg,
        output_dir=str(output_root / "training_results"),
        use_cache=True,
        feature_dir=args.feature_dir,
        coreset_cache_dir=str(output_root / "coreset_cache"),
    )
    results = pipeline.run()

    print("[2/5] Exporting portable CPU model bundle...")
    bundle_dir = export_bundle(
        pipeline=pipeline,
        config=cfg,
        task_cfg=task_cfg,
        output_root=output_root,
        training_results=results,
    )

    print("[3/5] Building compact 120-image mini MVTec test pack...")
    categories = [t["categories"][0] for t in task_cfg["tasks"]]
    per_cat = max(1, args.mini_size // max(1, len(categories)))
    good_per_cat = per_cat // 2
    bad_per_cat = per_cat - good_per_cat

    mini_root = output_root / "mini_testset_120"
    mini_summary = build_mini_testset(
        source_root=Path(args.dataset_root),
        target_root=mini_root,
        categories=categories,
        good_per_cat=good_per_cat,
        bad_per_cat=bad_per_cat,
    )

    with (output_root / "mini_testset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(mini_summary, f, indent=2)

    print("[4/5] Running bundle inference on the mini test pack...")
    images = sorted(
        [
            p
            for p in mini_root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in VALID_EXTS
            and "test" in p.parts
            and "ground_truth" not in p.parts
        ]
    )
    infer_dir = output_root / "demo_inference"
    infer_images(bundle_dir=bundle_dir, image_paths=images, output_dir=infer_dir)

    print("[5/5] Generating polished visual HTML demo...")
    report_dir = output_root / "visual_demo"
    build_visual_report(
        infer_dir=infer_dir,
        report_dir=report_dir,
        title="MemoryAD CPU Demo - MVTec Full Bundle",
    )

    summary = {
        "bundle_dir": str(bundle_dir).replace("\\", "/"),
        "mini_testset_dir": str(mini_root).replace("\\", "/"),
        "demo_report": str((report_dir / "index.html")).replace("\\", "/"),
        "training_results": str((output_root / "training_results" / "results.json")).replace("\\", "/"),
        "inference_csv": str((infer_dir / "predictions.csv")).replace("\\", "/"),
        "mini_testset_summary": mini_summary["totals"],
    }

    with (output_root / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
