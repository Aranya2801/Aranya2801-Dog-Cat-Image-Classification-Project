"""
DogCat Vision — Dataset & DataLoader
======================================
Custom PyTorch Dataset with:
  - Albumentations augmentation pipeline
  - Automatic train/val/test splitting
  - Class balancing support
  - Memory-efficient image loading
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Callable
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


CLASS_NAMES = ["cat", "dog"]
CLASS_TO_IDX = {"cat": 0, "dog": 1}


# ─── Augmentation Pipelines ──────────────────────────────────────────────────

def get_train_transforms(input_size: int = 380) -> A.Compose:
    """
    Heavy augmentation pipeline for training.
    Designed for maximum generalization.
    """
    return A.Compose([
        A.RandomResizedCrop(height=input_size, width=input_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=15, p=0.4),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3),
        A.GridDistortion(p=0.1),
        A.CoarseDropout(
            max_holes=8, max_height=input_size // 10, max_width=input_size // 10,
            min_holes=1, fill_value=0, p=0.3
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(input_size: int = 380) -> A.Compose:
    """Minimal augmentation for validation/test — only resize and normalize."""
    return A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_inference_transforms(input_size: int = 380) -> A.Compose:
    """Transforms for single-image inference."""
    return get_val_transforms(input_size)


# ─── Dataset Class ────────────────────────────────────────────────────────────

class DogCatDataset(Dataset):
    """
    PyTorch Dataset for Dog vs Cat classification.

    Expects directory structure:
        root/
            cats/  (or cat/)
                img1.jpg, img2.jpg, ...
            dogs/  (or dog/)
                img1.jpg, img2.jpg, ...
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[A.Compose] = None,
        split: str = "train",
        return_path: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.return_path = return_path

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}. Check dataset structure.")

    def _load_samples(self):
        """Scan directory and build sample list."""
        # Support both "cats/dogs" and "cat/dog" folder names
        cat_dirs = ["cats", "cat"]
        dog_dirs = ["dogs", "dog"]

        for cat_dir in cat_dirs:
            cat_path = self.root_dir / cat_dir
            if cat_path.exists():
                for img_path in cat_path.iterdir():
                    if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                        self.samples.append((img_path, 0))  # 0 = cat
                break

        for dog_dir in dog_dirs:
            dog_path = self.root_dir / dog_dir
            if dog_path.exists():
                for img_path in dog_path.iterdir():
                    if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                        self.samples.append((img_path, 1))  # 1 = dog
                break

        # Shuffle for balanced batches
        import random
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            # Return a random valid sample on corrupt image
            return self.__getitem__((idx + 1) % len(self))

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.return_path:
            return image, label, str(img_path)
        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        labels = [s[1] for s in self.samples]
        class_counts = torch.bincount(torch.tensor(labels))
        weights = 1.0 / class_counts.float()
        return weights

    def get_sample_weights(self) -> List[float]:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return [class_weights[label].item() for _, label in self.samples]


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    input_size: int = 380,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        data_dir: Root data directory containing train/, val/, test/
        batch_size: Batch size
        input_size: Input image size (square)
        num_workers: Number of DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
        use_sampler: Use WeightedRandomSampler for class balance

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    train_dataset = DogCatDataset(
        root_dir=str(data_dir / "train"),
        transform=get_train_transforms(input_size),
        split="train",
    )
    val_dataset = DogCatDataset(
        root_dir=str(data_dir / "val"),
        transform=get_val_transforms(input_size),
        split="val",
    )
    test_dataset = DogCatDataset(
        root_dir=str(data_dir / "test"),
        transform=get_val_transforms(input_size),
        split="test",
        return_path=True,  # Return paths for error analysis
    )

    # Weighted sampler for balanced training
    train_sampler = None
    if use_sampler:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Can use larger batch for eval
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
