"""
DINOv2 feature extractor for MemoryAD.
Extracts patch-level features from intermediate layers of a frozen DINOv2 ViT.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class DINOv2Extractor:
    """
    Extracts patch-level features from a frozen DINOv2 Vision Transformer.

    DINOv2 ViT-B/14 produces 37×37 = 1369 patch tokens from a 518×518 input.
    We extract from intermediate layers (e.g. [7, 11]) and concatenate/average
    to get rich mid-to-high-level features.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        layers: List[int] = [7, 11],
        aggregation: str = "concat",
        use_fp16: bool = True,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.layers = layers
        self.aggregation = aggregation
        self.use_fp16 = use_fp16
        self.device = device

        # Load model
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        self.model.to(device)

        if use_fp16:
            self.model.half()

        # Determine feature dimension
        if "vitb" in model_name:
            self.embed_dim = 768
        elif "vitl" in model_name:
            self.embed_dim = 1024
        elif "vits" in model_name:
            self.embed_dim = 384
        elif "vitg" in model_name:
            self.embed_dim = 1536
        else:
            self.embed_dim = 768

        if aggregation == "concat":
            self.feature_dim = self.embed_dim * len(layers)
        else:
            self.feature_dim = self.embed_dim

        # Storage for hook outputs
        self._features = {}
        self._hooks = []

        # Register hooks on target layers
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on intermediate transformer blocks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        for layer_idx in self.layers:
            hook = self.model.blocks[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        """Create a hook function that stores the output of a layer."""
        def hook_fn(module, input, output):
            self._features[layer_idx] = output
        return hook_fn

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features from a batch of images.

        Args:
            images: [B, 3, H, W] tensor, normalised with ImageNet stats.

        Returns:
            features: [B, num_patches, feature_dim] tensor.
                       For ViT-B/14 at 518×518: num_patches = 37*37 = 1369
                       feature_dim = 768 (average) or 1536 (concat of 2 layers)
        """
        self._features = {}

        if self.use_fp16:
            images = images.half()

        images = images.to(self.device)

        # Forward pass (triggers hooks)
        _ = self.model(images)

        # Collect and aggregate features from target layers
        layer_features = []
        for layer_idx in self.layers:
            feat = self._features[layer_idx]
            # DINOv2 output shape: [B, 1 + num_patches, embed_dim]
            # Remove the [CLS] token (index 0)
            feat = feat[:, 1:, :]  # [B, num_patches, embed_dim]
            layer_features.append(feat.float())  # Convert back to float32

        if self.aggregation == "concat":
            features = torch.cat(layer_features, dim=-1)  # [B, P, D*num_layers]
        elif self.aggregation == "average":
            features = torch.stack(layer_features, dim=0).mean(dim=0)  # [B, P, D]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return features

    def get_spatial_dims(self, input_size: int = 518) -> tuple:
        """
        Get the spatial dimensions (H, W) of the patch grid.
        For ViT-B/14: input_size 518 → 518 // 14 = 37 → (37, 37)
        """
        patch_size = 14
        h = w = input_size // patch_size
        return h, w

    def __repr__(self):
        return (
            f"DINOv2Extractor(model={self.model_name}, layers={self.layers}, "
            f"aggregation={self.aggregation}, feature_dim={self.feature_dim})"
        )
