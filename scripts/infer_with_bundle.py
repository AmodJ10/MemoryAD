"""
Run MemoryAD inference from an exported bundle on raw images.

This script is CPU-first and designed for portability:
- Loads coreset + metadata from a bundle directory
- Extracts DINOv2 features from input images
- Computes anomaly scores and heatmap overlays
- Exports CSV + JSON + rendered overlays
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backbones.dinov2 import DINOv2Extractor
from src.scoring.knn_scorer import KNNScorer

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}


def _collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in VALID_EXTS:
            return [input_path]
        return []

    images = []
    for p in input_path.rglob("*"):
        if (
            p.is_file()
            and p.suffix.lower() in VALID_EXTS
            and "ground_truth" not in p.parts
        ):
            images.append(p)
    return sorted(images)


def _make_overlay(image_rgb: np.ndarray, score_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    score_norm = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
    heat = cv2.applyColorMap(np.uint8(255 * score_norm), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, heat, alpha, 0)
    return overlay


def _build_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def infer_images(
    bundle_dir: Path,
    image_paths: Iterable[Path],
    output_dir: Path,
) -> list[dict]:
    bundle_dir = Path(bundle_dir)
    output_dir = Path(output_dir)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    with (bundle_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    coreset = np.load(bundle_dir / "coreset_global.npy")

    backbone_cfg = metadata["backbone"]
    scoring_cfg = metadata["scoring"]
    spatial_dims = tuple(metadata["spatial_dims"])
    input_size = int(backbone_cfg.get("input_size", 518))

    extractor = DINOv2Extractor(
        model_name=backbone_cfg["name"],
        layers=backbone_cfg["layers"],
        aggregation=backbone_cfg.get("feature_aggregation", "concat"),
        use_fp16=False,
        device="cpu",
    )

    scorer = KNNScorer(
        k=int(scoring_cfg.get("k", 9)),
        image_score_method=scoring_cfg.get("image_score_method", "max"),
        top_k_percent=float(scoring_cfg.get("top_k_percent", 1.0)),
    )
    scorer.fit(coreset.astype(np.float32))

    img_transform = _build_transform(input_size)

    records = []

    for img_path in tqdm(list(image_paths), desc="Inference"):
        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size
        image_tensor = img_transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            feats = extractor.extract(image_tensor).cpu().numpy()  # [1, P, D]

        image_scores, anomaly_maps = scorer.score_batch(feats, spatial_dims)
        score = float(image_scores[0])
        amap = anomaly_maps[0]

        amap_resized = cv2.resize(amap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        img_np = np.array(pil_img)
        overlay = _make_overlay(img_np, amap_resized)

        out_name = f"{img_path.stem}_overlay.png"
        out_path = overlays_dir / out_name
        Image.fromarray(overlay).save(out_path)

        record = {
            "image_path": str(img_path).replace("\\", "/"),
            "overlay_path": str(out_path).replace("\\", "/"),
            "anomaly_score": score,
            "width": orig_w,
            "height": orig_h,
        }
        records.append(record)

    records.sort(key=lambda r: r["anomaly_score"], reverse=True)

    with (output_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "overlay_path", "anomaly_score", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(records)

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference from MemoryAD bundle")
    parser.add_argument("--bundle-dir", required=True, help="Directory with coreset_global.npy and metadata.json")
    parser.add_argument("--input", required=True, help="Input image path or folder")
    parser.add_argument("--output-dir", default="exports/inference_output", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    images = _collect_images(input_path)
    if not images:
        raise RuntimeError(f"No images found at: {input_path}")

    records = infer_images(
        bundle_dir=Path(args.bundle_dir),
        image_paths=images,
        output_dir=Path(args.output_dir),
    )

    print(f"Inference done on {len(records)} images.")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
