"""
Interactive MemoryAD demo server.

Features:
- Serves memoryad_demo.html
- Single-image inference endpoint
- Two-image comparison endpoint
- Uses exported CPU bundle (no retraining)
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backbones.dinov2 import DINOv2Extractor
from src.scoring.knn_scorer import KNNScorer


@dataclass
class Calibration:
    score_min: float
    score_max: float
    p50: float
    p85: float


class MemoryADDemoEngine:
    def __init__(self, bundle_dir: Path):
        self.bundle_dir = bundle_dir
        self.metadata = self._load_metadata()
        self.coreset = np.load(self.bundle_dir / "coreset_global.npy").astype(np.float32)

        backbone_cfg = self.metadata["backbone"]
        scoring_cfg = self.metadata["scoring"]

        self.spatial_dims = tuple(self.metadata["spatial_dims"])
        self.input_size = int(backbone_cfg.get("input_size", 518))

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.extractor = DINOv2Extractor(
            model_name=backbone_cfg["name"],
            layers=backbone_cfg["layers"],
            aggregation=backbone_cfg.get("feature_aggregation", "concat"),
            use_fp16=False,
            device="cpu",
        )

        self.scorer = KNNScorer(
            k=int(scoring_cfg.get("k", 9)),
            image_score_method=scoring_cfg.get("image_score_method", "max"),
            top_k_percent=float(scoring_cfg.get("top_k_percent", 1.0)),
        )
        self.scorer.fit(self.coreset)

        self.calibration = self._load_calibration()

    def _load_metadata(self) -> dict[str, Any]:
        with (self.bundle_dir / "metadata.json").open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_calibration(self) -> Calibration:
        candidate = self.bundle_dir.parent / "demo_inference_clean" / "predictions.csv"
        scores: list[float] = []

        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        scores.append(float(row["anomaly_score"]))
                    except Exception:
                        continue

        if not scores:
            # Fallback conservative defaults
            return Calibration(score_min=0.0, score_max=5000.0, p50=1200.0, p85=2500.0)

        arr = np.array(scores, dtype=np.float32)
        return Calibration(
            score_min=float(np.min(arr)),
            score_max=float(np.max(arr)),
            p50=float(np.percentile(arr, 50)),
            p85=float(np.percentile(arr, 85)),
        )

    def _risk_level(self, score: float) -> str:
        if score >= self.calibration.p85:
            return "High"
        if score >= self.calibration.p50:
            return "Moderate"
        return "Low"

    def _normalize_score(self, score: float) -> float:
        den = max(1e-6, self.calibration.score_max - self.calibration.score_min)
        return float(np.clip((score - self.calibration.score_min) / den * 100.0, 0.0, 100.0))

    @staticmethod
    def _overlay(image_rgb: np.ndarray, score_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        score_norm = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
        heat = cv2.applyColorMap(np.uint8(255 * score_norm), cv2.COLORMAP_TURBO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(image_rgb, 1.0 - alpha, heat, alpha, 0)

    @staticmethod
    def _to_data_uri(image_rgb: np.ndarray) -> str:
        pil = Image.fromarray(image_rgb)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        payload = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{payload}"

    def infer_pil(self, pil_img: Image.Image) -> dict[str, Any]:
        pil_img = pil_img.convert("RGB")
        w, h = pil_img.size

        t0 = time.perf_counter()
        image_tensor = self.transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            feats = self.extractor.extract(image_tensor).cpu().numpy()

        image_scores, anomaly_maps = self.scorer.score_batch(feats, self.spatial_dims)
        score = float(image_scores[0])
        amap = anomaly_maps[0]

        amap_resized = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
        overlay = self._overlay(np.array(pil_img), amap_resized)

        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        return {
            "score_raw": score,
            "score_norm_100": round(self._normalize_score(score), 2),
            "risk_level": self._risk_level(score),
            "inference_ms": round(infer_ms, 2),
            "scoring_k": int(self.metadata.get("scoring", {}).get("k", 0)),
            "width": w,
            "height": h,
            "heatmap_mean": float(np.mean(amap_resized)),
            "heatmap_max": float(np.max(amap_resized)),
            "overlay_data_uri": self._to_data_uri(overlay),
        }


def create_app(bundle_dir: Path, html_path: Path) -> Flask:
    app = Flask(__name__)
    engine = MemoryADDemoEngine(bundle_dir=bundle_dir)

    @app.after_request
    def add_headers(resp):
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @app.route("/api/<path:_subpath>", methods=["OPTIONS"])
    def handle_preflight(_subpath: str):
        return ("", 204)

    @app.get("/")
    def root_page():
        return send_file(html_path)

    @app.get("/api/health")
    def health():
        summary = dict(engine.metadata.get("training_summary", {}))
        if "forward_transfer" not in summary:
            results_path = bundle_dir.parent / "training_results" / "results.json"
            if results_path.exists():
                try:
                    with results_path.open("r", encoding="utf-8") as f:
                        results = json.load(f)
                    if "forward_transfer" in results:
                        summary["forward_transfer"] = float(results["forward_transfer"])
                except Exception:
                    pass

        return jsonify(
            {
                "ok": True,
                "bundle": str(bundle_dir).replace("\\", "/"),
                "backbone": engine.metadata.get("backbone", {}).get("name", "unknown"),
                "coreset_size": engine.metadata.get("coreset_size", 0),
                "final_mean_auroc": summary.get("final_mean_auroc", None),
                "avg_incremental_auroc": summary.get("avg_incremental_auroc", None),
                "forgetting_rate": summary.get("forgetting_rate", None),
                "forward_transfer": summary.get("forward_transfer", None),
                "calibration": {
                    "score_min": engine.calibration.score_min,
                    "score_max": engine.calibration.score_max,
                    "p50": engine.calibration.p50,
                    "p85": engine.calibration.p85,
                },
            }
        )

    @app.post("/api/infer")
    def infer_one():
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "Missing file field: image"}), 400
        f = request.files["image"]
        if not f.filename:
            return jsonify({"ok": False, "error": "Empty filename"}), 400

        try:
            pil = Image.open(f.stream)
            out = engine.infer_pil(pil)
            return jsonify({"ok": True, "result": out})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.post("/api/compare")
    def compare_two():
        if "image_a" not in request.files or "image_b" not in request.files:
            return jsonify({"ok": False, "error": "Missing fields: image_a and image_b"}), 400

        fa = request.files["image_a"]
        fb = request.files["image_b"]
        if not fa.filename or not fb.filename:
            return jsonify({"ok": False, "error": "Both files must be selected"}), 400

        try:
            ra = engine.infer_pil(Image.open(fa.stream))
            rb = engine.infer_pil(Image.open(fb.stream))

            delta = float(ra["score_raw"] - rb["score_raw"])
            higher = "A" if delta > 0 else "B" if delta < 0 else "Equal"

            return jsonify(
                {
                    "ok": True,
                    "result_a": ra,
                    "result_b": rb,
                    "delta_raw": delta,
                    "delta_abs": abs(delta),
                    "higher_risk": higher,
                }
            )
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    return app


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_bundle = repo_root / "exports" / "mvtec_full_cpu" / "model_bundle"
    html_path = repo_root / "memoryad_demo.html"

    bundle_dir = Path(os.environ.get("MEMORYAD_BUNDLE_DIR", str(default_bundle))).resolve()
    port = int(os.environ.get("MEMORYAD_DEMO_PORT", "8000"))

    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle dir not found: {bundle_dir}")
    if not html_path.exists():
        raise FileNotFoundError(f"Demo HTML not found: {html_path}")

    app = create_app(bundle_dir=bundle_dir, html_path=html_path)
    print(f"MemoryAD demo server running at http://127.0.0.1:{port}")
    print(f"Using bundle: {bundle_dir}")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
