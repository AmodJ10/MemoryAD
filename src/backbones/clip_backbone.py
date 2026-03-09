"""
CLIP ViT-L/14 feature extractor for MemoryAD.
Extracts patch-level features from intermediate layers of a frozen CLIP vision encoder.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List


class CLIPExtractor:
    """
    Extracts patch-level features from a frozen CLIP ViT-L/14 vision encoder.
    Uses the open_clip library for flexible model loading.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        layers: List[int] = [18, 23],
        aggregation: str = "concat",
        use_fp16: bool = True,
        device: str = "cuda",
    ):
        import open_clip

        self.model_name = model_name
        self.layers = layers
        self.aggregation = aggregation
        self.use_fp16 = use_fp16
        self.device = device

        # Load CLIP model
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.visual
        self.model.eval()
        self.model.to(device)

        if use_fp16:
            self.model.half()

        # Determine feature dimension
        if "ViT-L" in model_name:
            self.embed_dim = 1024
        elif "ViT-B" in model_name:
            self.embed_dim = 768
        else:
            self.embed_dim = 768

        if aggregation == "concat":
            self.feature_dim = self.embed_dim * len(layers)
        else:
            self.feature_dim = self.embed_dim

        # Hook storage
        self._features = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on CLIP vision transformer blocks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        # CLIP's visual transformer blocks
        transformer = self.model.transformer
        for layer_idx in self.layers:
            hook = transformer.resblocks[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            self._features[layer_idx] = output
        return hook_fn

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features from a batch of images.

        Args:
            images: [B, 3, H, W] tensor, normalised for CLIP.

        Returns:
            features: [B, num_patches, feature_dim] tensor.
        """
        self._features = {}

        if self.use_fp16:
            images = images.half()
        images = images.to(self.device)

        # Forward pass
        _ = self.model(images)

        # Collect features
        layer_features = []
        for layer_idx in self.layers:
            feat = self._features[layer_idx]
            # open_clip may output [B, 1+P, D] (batch-first) or [1+P, B, D] (seq-first)
            # Detect format: if dim0 matches batch size, it's batch-first
            if feat.shape[0] == images.shape[0]:
                # Batch-first: [B, 1+P, D] — already correct layout
                feat = feat[:, 1:, :]  # Remove CLS token
            else:
                # Sequence-first: [1+P, B, D]
                feat = feat.permute(1, 0, 2)  # -> [B, 1+P, D]
                feat = feat[:, 1:, :]  # Remove CLS token
            layer_features.append(feat.float())

        if self.aggregation == "concat":
            features = torch.cat(layer_features, dim=-1)
        elif self.aggregation == "average":
            features = torch.stack(layer_features, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return features

    def get_spatial_dims(self, input_size: int = 224) -> tuple:
        """CLIP ViT-L/14: 224 // 14 = 16 → (16, 16)"""
        h = w = input_size // 14
        return h, w

    def __repr__(self):
        return (
            f"CLIPExtractor(model={self.model_name}, layers={self.layers}, "
            f"aggregation={self.aggregation}, feature_dim={self.feature_dim})"
        )
