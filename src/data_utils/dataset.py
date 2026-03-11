"""
Dataset utilities for MVTec AD and VisA.
Handles loading images, ground truth masks, and organising by category.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AnomalyDataset(Dataset):
    """
    Generic anomaly detection dataset loader.
    Supports MVTec AD and VisA directory structures.
    
    MVTec AD structure:
        root/category/train/good/*.png
        root/category/test/good/*.png
        root/category/test/defect_type/*.png
        root/category/ground_truth/defect_type/*.png
    
    VisA structure:
        root/category/train/good/*.JPG
        root/category/test/good/*.JPG
        root/category/test/bad/*.JPG
        root/category/ground_truth/bad/*.png
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        dataset_type: str = "mvtec_ad",
        input_size: int = 518,
        mask_size: int = 518,
    ):
        self.root = Path(root)
        self.category = category
        self.split = split
        self.dataset_type = dataset_type
        self.input_size = input_size
        self.mask_size = mask_size

        # Build file lists
        self.image_paths = []
        self.mask_paths = []
        self.labels = []  # 0 = normal, 1 = anomalous

        self._load_file_list()

        # Image transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((mask_size, mask_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _load_file_list(self):
        """Scan directory structure and build lists of image/mask paths."""
        category_dir = self.root / self.category

        if self.split == "train":
            # Training data is always the 'good' subdirectory
            good_dir = category_dir / "train" / "good"
            if good_dir.exists():
                for img_path in sorted(good_dir.glob("*")):
                    if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif"):
                        self.image_paths.append(str(img_path))
                        self.mask_paths.append(None)
                        self.labels.append(0)

        elif self.split == "test":
            test_dir = category_dir / "test"
            gt_dir = category_dir / "ground_truth"

            if not test_dir.exists():
                return

            for defect_type_dir in sorted(test_dir.iterdir()):
                if not defect_type_dir.is_dir():
                    continue

                defect_type = defect_type_dir.name
                is_normal = defect_type == "good"

                for img_path in sorted(defect_type_dir.glob("*")):
                    if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif"):
                        continue

                    self.image_paths.append(str(img_path))
                    self.labels.append(0 if is_normal else 1)

                    if is_normal:
                        self.mask_paths.append(None)
                    else:
                        # Try to find corresponding ground truth mask
                        mask_path = gt_dir / defect_type / img_path.name
                        
                        # Handle VisA (images might be .JPG but masks are .png)
                        if not mask_path.exists():
                            mask_path = mask_path.with_suffix(".png")
                            
                        # Handle MVTec (masks have _mask.png suffix)
                        if not mask_path.exists():
                            mask_path = mask_path.with_name(f"{img_path.stem}_mask.png")
                            
                        if mask_path.exists():
                            self.mask_paths.append(str(mask_path))
                        else:
                            self.mask_paths.append(None)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.img_transform(image)

        # Load mask (if available)
        if self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()  # Binarise
        else:
            mask = torch.zeros(1, self.mask_size, self.mask_size)

        label = self.labels[idx]

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "path": self.image_paths[idx],
            "category": self.category,
        }


def get_category_dataloaders(
    root: str,
    category: str,
    dataset_type: str = "mvtec_ad",
    input_size: int = 518,
    batch_size: int = 4,
    num_workers: int = 0,
):
    """Create train and test data loaders for a single category.
    
    Note: num_workers defaults to 0 (main process) for Windows compatibility.
    On Linux, you can safely set num_workers=4 for faster loading.
    """

    train_dataset = AnomalyDataset(
        root=root,
        category=category,
        split="train",
        dataset_type=dataset_type,
        input_size=input_size,
    )

    test_dataset = AnomalyDataset(
        root=root,
        category=category,
        split="test",
        dataset_type=dataset_type,
        input_size=input_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
