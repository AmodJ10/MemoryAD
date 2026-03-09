"""
RD4AD — Reverse Distillation for Anomaly Detection.

A lightweight trainable AD model used as the backbone for CIL baselines
(EWC, LwF, Replay). Architecture:
  - Encoder: Frozen WideResNet-50 (layers 1-3)
  - Decoder: Trainable reverse decoder that reconstructs encoder features
  - Anomaly score: MSE between encoder and decoder features

Reference: Deng & Li, "Anomaly Detection via Reverse Distillation from
One-Class Embedding", CVPR 2022.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from typing import Optional


class DecoderBlock(nn.Module):
    """Single block of the reverse decoder."""

    def __init__(self, in_channels: int, out_channels: int, upsample: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = upsample

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class ReverseDecoder(nn.Module):
    """
    Reverse decoder that reconstructs encoder features.
    Takes the deepest encoder feature map and progressively upsamples
    to match intermediate encoder feature maps.
    """

    def __init__(self):
        super().__init__()
        # WideResNet-50 channel sizes: layer1=256, layer2=512, layer3=1024
        self.block3 = DecoderBlock(1024, 512, upsample=True)
        self.block2 = DecoderBlock(512, 256, upsample=True)
        self.block1 = DecoderBlock(256, 256, upsample=False)  # Match layer1 size

        # Projection heads to match encoder dimensions
        self.proj3 = nn.Conv2d(1024, 1024, 1)  # layer3 output
        self.proj2 = nn.Conv2d(512, 512, 1)    # layer2 output
        self.proj1 = nn.Conv2d(256, 256, 1)    # layer1 output

    def forward(self, x):
        """
        Args:
            x: deepest encoder feature [B, 1024, H, W]
        Returns:
            list of decoded features matching encoder layers 3, 2, 1
        """
        d3 = self.proj3(x)                # [B, 1024, H, W]
        d2 = self.block3(x)               # [B, 512, 2H, 2W]
        d2_proj = self.proj2(d2)
        d1 = self.block2(d2)              # [B, 256, 4H, 4W]
        d1_proj = self.proj1(d1)

        return [d1_proj, d2_proj, d3]


class RD4AD(nn.Module):
    """
    Reverse Distillation for Anomaly Detection.

    Encoder is frozen; decoder is trained to reconstruct encoder features.
    At test time, anomaly score = MSE between encoder and decoder outputs.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

        # Frozen encoder (WideResNet-50)
        backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1),
            backbone.layer2,
            backbone.layer3,
        ])
        for param in self.encoder_layers.parameters():
            param.requires_grad = False
        self.encoder_layers.eval()

        # Trainable decoder
        self.decoder = ReverseDecoder()

        self.to(device)

    def encode(self, x: torch.Tensor):
        """Extract multi-scale encoder features."""
        features = []
        with torch.no_grad():
            self.encoder_layers.eval()
            h = x
            for layer in self.encoder_layers:
                h = layer(h)
                features.append(h)
        return features  # [layer1_out, layer2_out, layer3_out]

    def decode(self, deepest_feature: torch.Tensor):
        """Reconstruct encoder features from deepest layer."""
        return self.decoder(deepest_feature)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            encoder_features: list of 3 feature maps
            decoder_features: list of 3 reconstructed feature maps
        """
        enc_feats = self.encode(x)
        dec_feats = self.decode(enc_feats[-1])  # Use deepest layer
        return enc_feats, dec_feats

    def compute_loss(self, enc_feats, dec_feats):
        """Reconstruction loss: MSE between encoder and decoder features."""
        loss = 0
        for enc, dec in zip(enc_feats, dec_feats):
            # Align spatial dims if needed
            if enc.shape != dec.shape:
                dec = F.interpolate(dec, size=enc.shape[2:], mode="bilinear", align_corners=False)
            loss = loss + F.mse_loss(dec, enc.detach())
        return loss

    def compute_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pixel-level anomaly map as MSE between encoder/decoder features."""
        with torch.amp.autocast("cuda"):
            enc_feats, dec_feats = self(x)
            target_size = enc_feats[0].shape[2:]
            amap = torch.zeros(
                x.size(0), target_size[0], target_size[1],
                device=self.device,
            )
            for enc, dec in zip(enc_feats, dec_feats):
                if dec.shape != enc.shape:
                    dec = F.interpolate(dec, size=enc.shape[2:], mode="bilinear", align_corners=False)
                diff = (enc - dec) ** 2
                diff = diff.mean(dim=1)
                diff = F.interpolate(
                    diff.unsqueeze(1), size=target_size,
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
                amap += diff
        return amap.float()

    def train_on_loader(
        self,
        train_loader: DataLoader,
        epochs: int = 50,
        lr: float = 5e-4,
        extra_loss_fn=None,
    ):
        """
        Train the decoder on normal images with mixed-precision.

        Args:
            train_loader: DataLoader yielding batches with 'image' key.
            epochs: Number of training epochs.
            lr: Learning rate.
            extra_loss_fn: Optional additional loss (for EWC/LwF).
                          Callable(model, enc_feats, dec_feats) -> loss_tensor.
        """
        import time as _time
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        scaler = torch.amp.GradScaler("cuda")
        self.decoder.train()
        self.encoder_layers.eval()
        t_start = _time.time()

        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            for batch in train_loader:
                images = batch["image"].to(self.device)

                # Mixed-precision forward pass
                with torch.amp.autocast("cuda"):
                    enc_feats, dec_feats = self(images)
                    loss = self.compute_loss(enc_feats, dec_feats)

                    if extra_loss_fn is not None:
                        extra = extra_loss_fn(self, enc_feats, dec_feats)
                        loss = loss + extra

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                n_batches += 1

            avg = total_loss / max(n_batches, 1)
            elapsed = _time.time() - t_start
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg:.4f}, Time: {elapsed:.0f}s", flush=True)

        self.decoder.eval()

    def evaluate(
        self,
        test_loader: DataLoader,
    ) -> float:
        """Evaluate I-AUROC on a test loader."""
        import time as _time
        from ..evaluation.metrics import compute_auroc

        self.eval()
        all_scores, all_labels = [], []
        t0 = _time.time()
        n_images = 0

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].numpy()
                amap = self.compute_anomaly_map(images)  # [B, H, W]
                scores = amap.view(amap.size(0), -1).max(dim=1)[0].cpu().numpy()
                all_scores.extend(scores.tolist())
                all_labels.extend(labels.tolist())
                n_images += len(labels)

        auroc = compute_auroc(np.array(all_labels), np.array(all_scores))
        print(f"      Evaluated {n_images} images in {_time.time()-t0:.1f}s", flush=True)
        return auroc
