"""
WideResNet-50 feature extractor for MemoryAD.
This is the PatchCore default backbone — used as a baseline comparison.
Extracts features from intermediate conv layers of a frozen WideResNet-50.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


class WideResNetExtractor:
    """
    Extracts patch-level features from a frozen WideResNet-50 (ImageNet pretrained).
    Uses features from layers 2 and 3 (standard PatchCore configuration).
    """

    def __init__(
        self,
        layers: List[str] = ["layer2", "layer3"],
        aggregation: str = "concat",
        use_fp16: bool = True,
        device: str = "cuda",
    ):
        self.layer_names = layers
        self.aggregation = aggregation
        self.use_fp16 = use_fp16
        self.device = device

        # Load pretrained model
        self.model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model.to(device)

        if use_fp16:
            self.model.half()

        # Feature dimensions per layer
        self._layer_dims = {
            "layer1": 256,
            "layer2": 512,
            "layer3": 1024,
            "layer4": 2048,
        }

        if aggregation == "concat":
            self.feature_dim = sum(self._layer_dims[l] for l in layers)
        else:
            # For averaging, all layers must have the same dim — project to common dim
            self.feature_dim = max(self._layer_dims[l] for l in layers)

        # Hook storage
        self._features = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        for layer_name in self.layer_names:
            layer = getattr(self.model, layer_name)
            hook = layer.register_forward_hook(self._make_hook(layer_name))
            self._hooks.append(hook)

    def _make_hook(self, layer_name: str):
        def hook_fn(module, input, output):
            self._features[layer_name] = output
        return hook_fn

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features from a batch of images.

        Args:
            images: [B, 3, H, W] tensor, normalised with ImageNet stats.

        Returns:
            features: [B, num_patches, feature_dim] tensor.
        """
        self._features = {}

        if self.use_fp16:
            images = images.half()
        images = images.to(self.device)

        _ = self.model(images)

        # Collect features from target layers
        layer_features = []
        target_h = None

        for layer_name in self.layer_names:
            feat = self._features[layer_name].float()  # [B, C, H_l, W_l]
            B, C, H_l, W_l = feat.shape

            if target_h is None:
                target_h = H_l  # Use first layer's spatial dims as reference

            # Upsample/downsample to match spatial dims if needed
            if H_l != target_h:
                feat = F.interpolate(feat, size=(target_h, target_h), mode="bilinear", align_corners=False)

            # Reshape from [B, C, H, W] → [B, H*W, C]
            feat = feat.permute(0, 2, 3, 1).reshape(B, -1, C)
            layer_features.append(feat)

        if self.aggregation == "concat":
            features = torch.cat(layer_features, dim=-1)  # [B, H*W, C1+C2]
        elif self.aggregation == "average":
            # Pad smaller features to max dim, then average
            max_dim = max(f.shape[-1] for f in layer_features)
            padded = []
            for f in layer_features:
                if f.shape[-1] < max_dim:
                    f = F.pad(f, (0, max_dim - f.shape[-1]))
                padded.append(f)
            features = torch.stack(padded, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return features

    def get_spatial_dims(self, input_size: int = 224) -> tuple:
        """WideResNet layer2: input_size // 16 spatial dim."""
        h = w = input_size // 16
        return h, w

    def __repr__(self):
        return (
            f"WideResNetExtractor(layers={self.layer_names}, "
            f"aggregation={self.aggregation}, feature_dim={self.feature_dim})"
        )
