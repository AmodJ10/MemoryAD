"""
MemoryAD — Custom Demo Server.

A lightweight Flask server that powers the judge demo interface.
Loads the exported coreset bundle and provides endpoints for:
  - /              → serves the demo frontend
  - /api/metrics   → returns original project metrics
  - /api/infer     → single-image anomaly inference
  - /api/compare   → two-image comparison
  - /api/sample-images → lists available test images for quick testing

Usage:
    python scripts/custom_demo_server.py

    # Or with custom paths:
    python scripts/custom_demo_server.py --bundle custom_demo/model_bundle --port 5000
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import math
import time
from pathlib import Path
from typing import Any

import sys
import argparse
import base64
import io
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import flask
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backbones.dinov2 import DINOv2Extractor
from src.scoring.knn_scorer import KNNScorer
from src.coreset.adaptive_manager import AdaptiveCoresetManager

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}


class DemoEngine:
    """
    Self-contained inference engine.
    Loads the coreset from disk, initialises a frozen DINOv2 extractor
    and a KNN scorer, and exposes a simple `infer(PIL.Image) -> dict` API.
    """

    def __init__(self, bundle_dir: Path):
        self.bundle_dir = bundle_dir

        # Load metadata
        with (bundle_dir / "metadata.json").open("r") as f:
            self.metadata: dict[str, Any] = json.load(f)

        # Load coreset
        coreset = np.load(bundle_dir / "coreset_global.npy").astype(np.float32)

        backbone_cfg = self.metadata["backbone"]
        scoring_cfg = self.metadata["scoring"]
        self.spatial_dims = tuple(self.metadata["spatial_dims"])
        self.input_size = int(backbone_cfg.get("input_size", 518))

        # Image preprocessing
        self.transform = T.Compose([
            T.Resize(
                (self.input_size, self.input_size),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Feature extractor (frozen, no training)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading DINOv2 on {device}...")
        self.extractor = DINOv2Extractor(
            model_name=backbone_cfg["name"],
            layers=backbone_cfg["layers"],
            aggregation=backbone_cfg.get("feature_aggregation", "concat"),
            use_fp16=device == "cuda",
            device=device,
        )

        # KNN scorer
        self.scorer = KNNScorer(
            k=int(scoring_cfg.get("k", 9)),
            image_score_method=scoring_cfg.get("image_score_method", "max"),
            top_k_percent=float(scoring_cfg.get("top_k_percent", 1.0)),
        )
        self.scorer.fit(coreset)
        self._coreset = coreset  # keep reference for re-fitting

        print(f"  Coreset loaded: {coreset.shape[0]} patches, dim={coreset.shape[1]}")
        print(f"  Spatial dims: {self.spatial_dims}")

    def get_hyperparams(self) -> dict[str, Any]:
        """Return current scoring hyperparameters."""
        return {
            "k": self.scorer.k,
            "image_score_method": self.scorer.image_score_method,
            "top_k_percent": self.scorer.top_k_percent,
        }

    def update_hyperparams(
        self,
        k: int | None = None,
        image_score_method: str | None = None,
        top_k_percent: float | None = None,
    ) -> dict[str, Any]:
        """Update scoring hyperparameters and rebuild the scorer."""
        new_k = k if k is not None else self.scorer.k
        new_method = image_score_method if image_score_method is not None else self.scorer.image_score_method
        new_top_k = top_k_percent if top_k_percent is not None else self.scorer.top_k_percent

        # Validate
        new_k = max(1, min(50, int(new_k)))
        if new_method not in ("max", "top_k_mean"):
            new_method = "max"
        new_top_k = max(0.01, min(1.0, float(new_top_k)))

        self.scorer = KNNScorer(
            k=new_k,
            image_score_method=new_method,
            top_k_percent=new_top_k,
        )
        self.scorer.fit(self._coreset)
        return self.get_hyperparams()

    def infer(self, pil_img: Image.Image) -> dict[str, Any]:
        """Run inference on a single PIL image."""
        pil_img = pil_img.convert("RGB")
        w, h = pil_img.size

        t0 = time.perf_counter()

        # Extract features
        tensor = self.transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            feats = self.extractor.extract(tensor).cpu().numpy()

        # Score
        image_scores, anomaly_maps = self.scorer.score_batch(feats, self.spatial_dims)
        score = float(image_scores[0])
        amap = anomaly_maps[0]  # [H_patch, W_patch]

        # Resize anomaly map to original image size
        amap_resized = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Generate heatmap overlay
        overlay = self._make_overlay(np.array(pil_img), amap_resized)

        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0

        return {
            "score_raw": round(score, 6),
            "inference_ms": round(inference_ms, 1),
            "scoring_k": self.scorer.k,
            "score_method": self.scorer.image_score_method,
            "top_k_percent": self.scorer.top_k_percent,
            "dimensions": f"{w}×{h}",
            "heatmap_data_uri": self._to_data_uri(overlay),
            "heatmap_mean": round(float(np.mean(amap_resized)), 4),
            "heatmap_max": round(float(np.max(amap_resized)), 4),
        }

    @staticmethod
    def _make_overlay(
        image_rgb: np.ndarray, score_map: np.ndarray, alpha: float = 0.45
    ) -> np.ndarray:
        """Blend a turbo colourmap heatmap onto the original image."""
        norm = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
        heat = cv2.applyColorMap(np.uint8(255 * norm), cv2.COLORMAP_TURBO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(image_rgb, 1.0 - alpha, heat, alpha, 0)

    @staticmethod
    def _to_data_uri(image_rgb: np.ndarray) -> str:
        """Encode an RGB numpy array as a PNG data URI."""
        pil = Image.fromarray(image_rgb)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        payload = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{payload}"


def create_app(
    bundle_dir: Path,
    html_path: Path,
    test_data_dir: Path | None = None,
) -> Flask:
    """Create and configure the Flask app."""
    app = Flask(__name__)
    engine = DemoEngine(bundle_dir=bundle_dir)

    # ── Middleware ──────────────────────────────────────────

    @app.after_request
    def _cors(resp):
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @app.route("/api/<path:_>", methods=["OPTIONS"])
    def _preflight(_):
        return ("", 204)

    # ── Routes ─────────────────────────────────────────────

    @app.get("/")
    def index():
        return send_file(html_path)

    @app.get("/api/metrics")
    def metrics():
        summary = engine.metadata.get("training_summary", {})
        per_cat = engine.metadata.get("per_category_results", {})
        coreset = engine.metadata.get("coreset_stats", {})
        return jsonify({
            "ok": True,
            "backbone": engine.metadata["backbone"]["name"],
            "coreset_size": engine.metadata.get("coreset_size", 0),
            "final_mean_auroc": summary.get("final_mean_auroc"),
            "avg_incremental_auroc": summary.get("avg_incremental_auroc"),
            "forgetting_rate": summary.get("forgetting_rate"),
            "forward_transfer": summary.get("forward_transfer"),
            "per_category": per_cat,
            "coreset_stats": coreset,
        })

    @app.get("/api/training-data")
    def training_data():
        """Return the full training trajectory for the training replay visualization."""
        # Task structure for MVTec 5-task
        tasks = [
            {"id": 1, "categories": ["bottle", "cable", "capsule"]},
            {"id": 2, "categories": ["carpet", "grid", "hazelnut"]},
            {"id": 3, "categories": ["leather", "metal_nut", "pill"]},
            {"id": 4, "categories": ["screw", "tile", "toothbrush"]},
            {"id": 5, "categories": ["transistor", "wood", "zipper"]},
        ]
        all_categories = [c for t in tasks for c in t["categories"]]

        # ── Toy Training Endpoints ──────────────────────────────
    
    @app.route("/api/toy_train/init", methods=["POST", "OPTIONS"])
    def toy_train_init():
        """Reset the model to an untrained state."""
        # Create a fresh coreset manager
        engine.coreset_manager = AdaptiveCoresetManager(
            global_budget=10000,
            strategy="proportional",
            selection_method="greedy",
        )
        engine._coreset = None
        
        # Reset the scorer to un-fitted state
        engine.scorer = KNNScorer(
            k=engine.scorer.k,
            image_score_method=engine.scorer.image_score_method,
            top_k_percent=engine.scorer.top_k_percent,
        )
        
        # Clear the running AUROC matrix
        engine.toy_auroc_matrix = []
        engine.toy_learned_categories = []
        
        return jsonify({"ok": True, "message": "Model reset successfully."})

    @app.route("/api/toy_train/task/<int:task_id>", methods=["POST", "OPTIONS"])
    def toy_train_task(task_id: int):
        """Train the model dynamically on a specific task."""
        if not hasattr(engine, "coreset_manager"):
            return jsonify({"ok": False, "error": "Model not initialized. Call /init first."}), 400
            
        tasks = [
            {"id": 1, "categories": ["bottle", "cable", "capsule"]},
            {"id": 2, "categories": ["carpet", "grid", "hazelnut"]},
            {"id": 3, "categories": ["leather", "metal_nut", "pill"]},
            {"id": 4, "categories": ["screw", "tile", "toothbrush"]},
            {"id": 5, "categories": ["transistor", "wood", "zipper"]},
        ]
        
        if task_id < 1 or task_id > len(tasks):
            return jsonify({"ok": False, "error": "Invalid task ID"}), 400
            
        task_info = tasks[task_id - 1]
        task_cats = task_info["categories"]
        
        toy_train_dir = engine.bundle_dir.parent / "toy_train_data"
        test_dir = engine.bundle_dir.parent / "test_data"
        
        if not toy_train_dir.exists():
            return jsonify({"ok": False, "error": "toy_train_data directory not found"}), 500
            
        # 1. Extract features for training images
        for cat in task_cats:
            cat_dir = toy_train_dir / cat
            if not cat_dir.exists():
                continue
                
            for img_path in cat_dir.glob("*.*"):
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                    
                pil_img = Image.open(img_path).convert("RGB")
                tensor = engine.transform(pil_img).unsqueeze(0)
                
                with torch.no_grad():
                    feats = engine.extractor.extract(tensor).cpu().numpy()
                
                # Reshape from [1, P, D] to [P, D]
                feats = feats.reshape(-1, feats.shape[-1])
                engine.coreset_manager.add_features(feats)
                
            engine.toy_learned_categories.append(cat)
            
        # 2. Refit scorer
        engine._coreset = engine.coreset_manager.get_coreset()
        engine.scorer.fit(engine._coreset)
        
        # 3. Evaluate on ALL learned categories so far
        all_categories = [c for t in tasks for c in t["categories"]]
        
        cat_scores = []
        cat_labels = []
        
        row_aurocs = []
        
        for cat in all_categories:
            if cat not in engine.toy_learned_categories:
                row_aurocs.append(None)
                continue
                
            cat_test_dir = test_dir / cat / "test"
            if not cat_test_dir.exists():
                row_aurocs.append(None)
                continue
                
            cat_scores_local = []
            cat_labels_local = []
            
            # Predict for all images in this category's test set
            for split_dir in cat_test_dir.iterdir():
                if not split_dir.is_dir():
                    continue
                    
                is_anomaly = (split_dir.name != "good")
                
                for img_path in split_dir.glob("*.*"):
                    if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                        continue
                        
                    pil_img = Image.open(img_path).convert("RGB")
                    tensor = engine.transform(pil_img).unsqueeze(0)
                    
                    with torch.no_grad():
                        feats = engine.extractor.extract(tensor).cpu().numpy()
                        
                    image_scores, _ = engine.scorer.score_batch(feats, engine.spatial_dims)
                    cat_scores_local.append(float(image_scores[0]))
                    cat_labels_local.append(1 if is_anomaly else 0)
                    
            if len(set(cat_labels_local)) > 1:
                auroc = roc_auc_score(cat_labels_local, cat_scores_local)
                row_aurocs.append(float(auroc))
            else:
                row_aurocs.append(None)
                
        if not hasattr(engine, "toy_auroc_matrix"):
            engine.toy_auroc_matrix = []
            
        engine.toy_auroc_matrix.append(row_aurocs)
        
        return jsonify({
            "ok": True,
            "task_id": task_id,
            "coreset_size": engine.coreset_manager.get_coreset().shape[0],
            "auroc_row": row_aurocs,
            "auroc_matrix": engine.toy_auroc_matrix,
        })

        # Read auroc_matrix from the results stored in metadata
        # The matrix is [num_tasks x num_categories], NaN for unseen categories
        summary = engine.metadata.get("training_summary", {})
        coreset = engine.metadata.get("coreset_stats", {})

        # Try to load full results.json for the auroc_matrix
        auroc_matrix = None
        results_path = engine.bundle_dir.parent / "model_bundle" / "metadata.json"
        try:
            with (engine.bundle_dir / "metadata.json").open("r") as f:
                meta = json.load(f)
            if "auroc_matrix" in meta:
                auroc_matrix = meta["auroc_matrix"]
        except Exception:
            pass

        # If not in metadata, try the original results.json
        if auroc_matrix is None:
            for candidate in [
                engine.bundle_dir.parent.parent / "results" / "E1_mvtec_5task" / "results.json",
                Path(__file__).resolve().parents[1] / "results" / "E1_mvtec_5task" / "results.json",
            ]:
                if candidate.exists():
                    try:
                        with candidate.open("r") as f:
                            results = json.load(f)
                        auroc_matrix = results.get("auroc_matrix")
                        break
                    except Exception:
                        pass

        # Replace NaN with None so jsonify works correctly
        if auroc_matrix:
            auroc_matrix = [[None if math.isnan(x) else x for x in row] for row in auroc_matrix]

        return jsonify({
            "ok": True,
            "tasks": tasks,
            "all_categories": all_categories,
            "auroc_matrix": auroc_matrix,
            "coreset_stats": coreset,
            "total_time_seconds": summary.get("total_time_seconds", 0),
            "final_mean_auroc": summary.get("final_mean_auroc"),
            "avg_incremental_auroc": summary.get("avg_incremental_auroc"),
            "forgetting_rate": summary.get("forgetting_rate"),
            "forward_transfer": summary.get("forward_transfer"),
        })

    @app.post("/api/infer")
    def infer():
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "Missing file field 'image'"}), 400
        f = request.files["image"]
        if not f.filename:
            return jsonify({"ok": False, "error": "Empty filename"}), 400
        try:
            pil = Image.open(f.stream)
            result = engine.infer(pil)
            return jsonify({"ok": True, "result": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.post("/api/compare")
    def compare():
        if "image_a" not in request.files or "image_b" not in request.files:
            return jsonify({"ok": False, "error": "Missing image_a or image_b"}), 400
        try:
            ra = engine.infer(Image.open(request.files["image_a"].stream))
            rb = engine.infer(Image.open(request.files["image_b"].stream))
            delta = ra["score_raw"] - rb["score_raw"]
            higher = "A" if delta > 0 else ("B" if delta < 0 else "Equal")
            return jsonify({
                "ok": True,
                "result_a": ra,
                "result_b": rb,
                "delta": round(abs(delta), 6),
                "higher_risk": higher,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.get("/api/sample-images")
    def sample_images():
        """List test images available in the demo test_data folder."""
        if test_data_dir is None or not test_data_dir.exists():
            return jsonify({"ok": True, "images": []})

        images = []
        for p in sorted(test_data_dir.rglob("*")):
            if (
                p.is_file()
                and p.suffix.lower() in VALID_IMAGE_EXTS
                and "test" in p.parts
                and "ground_truth" not in p.parts
            ):
                rel = p.relative_to(test_data_dir)
                parts = rel.parts
                cat = parts[0] if len(parts) > 0 else "unknown"
                label = "good" if "good" in parts else "anomalous"
                images.append({
                    "path": str(rel).replace("\\", "/"),
                    "category": cat,
                    "label": label,
                })
        return jsonify({"ok": True, "images": images})

    @app.get("/api/hyperparams")
    def get_hyperparams():
        return jsonify({"ok": True, "params": engine.get_hyperparams()})

    @app.post("/api/hyperparams")
    def set_hyperparams():
        data = request.get_json(silent=True) or {}
        try:
            updated = engine.update_hyperparams(
                k=data.get("k"),
                image_score_method=data.get("image_score_method"),
                top_k_percent=data.get("top_k_percent"),
            )
            return jsonify({"ok": True, "params": updated})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.get("/test-image/<path:rel_path>")
    def serve_test_image(rel_path: str):
        """Serve a test image file from the test_data directory."""
        if test_data_dir is None or not test_data_dir.exists():
            return ("Test data not found", 404)
        return send_from_directory(str(test_data_dir), rel_path)

    return app


def main() -> None:
    import argparse

    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="MemoryAD Custom Demo Server")
    parser.add_argument(
        "--bundle",
        default=str(repo_root / "custom_demo" / "model_bundle"),
        help="Path to the model bundle directory",
    )
    parser.add_argument(
        "--html",
        default=str(repo_root / "custom_demo.html"),
        help="Path to the demo HTML file",
    )
    parser.add_argument(
        "--test-data",
        default=str(repo_root / "custom_demo" / "test_data"),
        help="Path to the curated test data folder",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle).resolve()
    html_path = Path(args.html).resolve()
    test_data_dir = Path(args.test_data).resolve()

    if not bundle_dir.exists():
        print(f"ERROR: Bundle not found at {bundle_dir}")
        print("Run: python scripts/create_custom_demo.py first")
        sys.exit(1)
    if not html_path.exists():
        print(f"ERROR: HTML not found at {html_path}")
        sys.exit(1)

    print("=" * 60)
    print("MemoryAD — Custom Demo Server")
    print("=" * 60)
    print(f"  Bundle:    {bundle_dir}")
    print(f"  HTML:      {html_path}")
    print(f"  Test data: {test_data_dir}")
    print()

    app = create_app(
        bundle_dir=bundle_dir,
        html_path=html_path,
        test_data_dir=test_data_dir if test_data_dir.exists() else None,
    )

    print(f"\n  Server running at http://{args.host}:{args.port}")
    print("  Press Ctrl+C to stop.\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
