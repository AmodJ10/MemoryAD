"""
Generate qualitative anomaly heatmaps for selected MVTec AD samples.
Produces a grid of: Original Image | Ground Truth Mask | Predicted Heatmap
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils.dataset import AnomalyDataset
from src.data_utils.feature_cache import FeatureCache
from src.scoring.knn_scorer import KNNScorer
from src.coreset.greedy_coreset import greedy_coreset_selection
from torchvision import transforms
import torch
import cv2

# Hardcoded good examples for visualization
SAMPLES_TO_VIZ = [
    {"category": "bottle", "defect": "broken_large", "idx": 0},
    {"category": "hazelnut", "defect": "crack", "idx": 0},
    {"category": "metal_nut", "defect": "scratch", "idx": 1},
    {"category": "pill", "defect": "contamination", "idx": 0},
    {"category": "toothbrush", "defect": "defective", "idx": 0},
]
BUDGET = 5000  # Smaller budget is fine for visualization demo
K = 9
FEATURE_DIR = "data/features/dinov2_vitb14"

def make_heatmap(image_np, score_map):
    # Normalize score_map
    score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * score_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    alpha = 0.5
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    return overlay

def main():
    print("Generating qualitative heatmaps...")
    cache = FeatureCache(FEATURE_DIR)
    
    scorer = KNNScorer(k=K)
    
    fig, axes = plt.subplots(len(SAMPLES_TO_VIZ), 3, figsize=(9, 3 * len(SAMPLES_TO_VIZ)))
    if len(SAMPLES_TO_VIZ) == 1:
        axes = [axes]
        
    for i, spec in enumerate(SAMPLES_TO_VIZ):
        cat = spec["category"]
        defect = spec["defect"]
        target_idx = spec["idx"]
        
        print(f"Processing {cat} - {defect} (idx={target_idx})")
        
        train_feats = cache.load_train_features(cat)
        coreset = greedy_coreset_selection(train_feats, budget=BUDGET)
        scorer.fit(coreset)
        
        # Load dataset
        dataset = AnomalyDataset(
            root="data/mvtec_ad",
            category=cat,
            split="test",
            input_size=518,
            mask_size=518,
        )
        
        # Find the image
        img_np = None
        mask_np = None
        test_idx_in_cache = -1
        current_idx_for_defect = 0
        
        for j in range(len(dataset)):
            sample = dataset[j]
            if sample["label"] == 1: # anomalous
                # Wait, AnomalyDataset just returns (image, mask, label) but doesn't expose defect names.
                # Actually, let's just pick the 'target_idx'-th anomaly.
                if current_idx_for_defect == target_idx:
                    img_tensor = sample["image"] # [3, 518, 518]
                    # Convert to numpy for matplotlib. Images are normalized, we need to denormalize.
                    # Standard imagenet denorm:
                    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
                    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
                    img_np = img_tensor.permute(1, 2, 0).numpy()
                    img_np = np.clip(img_np * std + mean, 0, 1)
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    mask_np = sample["mask"].squeeze().numpy() * 255
                    test_idx_in_cache = j
                    break
                current_idx_for_defect += 1
                
        if test_idx_in_cache == -1:
            print("WARNING: Sample not found")
            continue
            
        test_feats, _ = cache.load_test_data(cat)
        target_feat = test_feats[test_idx_in_cache:test_idx_in_cache+1] # [1, H*W, D]
        
        _, patch_scores = scorer.score_batch(target_feat, cache.spatial_dims)
        
        H, W = cache.spatial_dims
        score_map = patch_scores[0].reshape(1, 1, H, W)
        score_map_tensor = torch.from_numpy(score_map)
        score_map_up = torch.nn.functional.interpolate(
            score_map_tensor, size=(518, 518), mode="bilinear", align_corners=False
        ).numpy()[0, 0]
        
        overlay_img = make_heatmap(img_np, score_map_up)
        
        ax_img, ax_mask, ax_heat = axes[i]
        
        ax_img.imshow(img_np)
        ax_img.set_title(f"Image ({cat})")
        ax_img.axis("off")
        
        ax_mask.imshow(mask_np, cmap="gray")
        ax_mask.set_title("Ground Truth Mask")
        ax_mask.axis("off")
        
        ax_heat.imshow(overlay_img)
        ax_heat.set_title("Predicted Anomaly Map")
        ax_heat.axis("off")

    plt.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    out_path = "paper/figures/f7_qualitative_heatmaps.pdf"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
