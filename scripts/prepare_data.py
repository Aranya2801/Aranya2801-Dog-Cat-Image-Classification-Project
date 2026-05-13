"""
DogCat Vision — Data Preparation Script
=========================================
Splits raw dataset into train/val/test and organizes folder structure.

Usage:
    python scripts/prepare_data.py --data-dir data
    python scripts/prepare_data.py --data-dir data --train 0.8 --val 0.1
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from rich.console import Console
from rich.progress import track

console = Console()


def prepare_data(
    data_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    random.seed(seed)
    raw_dir = data_dir / "raw"
    out_dir = data_dir / "processed"

    console.rule("[bold cyan]📦 DogCat Vision — Data Preparation")

    # Detect structure
    class_dirs = {}
    for cls, aliases in [("cats", ["cats", "cat"]), ("dogs", ["dogs", "dog"])]:
        for alias in aliases:
            p = raw_dir / alias
            if p.exists():
                class_dirs[cls] = p
                break
        # Fallback: Kaggle flat structure
        if cls not in class_dirs:
            train_flat = raw_dir / "train"
            if train_flat.exists():
                prefix = "cat" if cls == "cats" else "dog"
                imgs = list(train_flat.glob(f"{prefix}.*.jpg"))
                if imgs:
                    class_dirs[cls] = ("flat", train_flat, prefix)

    if not class_dirs:
        console.print("[red]❌ No images found in data/raw/. Run download_dataset.py first.[/]")
        sys.exit(1)

    # Create output splits
    for split in ["train", "val", "test"]:
        for cls in ["cats", "dogs"]:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    # Process each class
    for cls, src in class_dirs.items():
        if isinstance(src, tuple):
            _, flat_dir, prefix = src
            images = list(flat_dir.glob(f"{prefix}.*.jpg"))
        else:
            images = (
                list(src.glob("*.jpg")) +
                list(src.glob("*.jpeg")) +
                list(src.glob("*.png"))
            )

        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        console.print(f"\n[cyan]{cls.upper()}:[/] {n:,} images total")
        for split_name, split_imgs in splits.items():
            dest_dir = out_dir / split_name / cls
            for i, img_path in enumerate(track(split_imgs, description=f"  → {split_name:5s}")):
                ext = img_path.suffix
                prefix_name = "cat" if cls == "cats" else "dog"
                shutil.copy2(str(img_path), str(dest_dir / f"{prefix_name}_{i:05d}{ext}"))
            console.print(f"    {split_name}: {len(split_imgs):,} images ✓")

    # Summary
    console.rule("[bold green]✅ Data Preparation Complete")
    for split in ["train", "val", "test"]:
        cats = len(list((out_dir / split / "cats").glob("*")))
        dogs = len(list((out_dir / split / "dogs").glob("*")))
        console.print(f"  {split:6s}: 🐱 {cats:,} cats + 🐶 {dogs:,} dogs = {cats+dogs:,} total")

    console.print(f"\n[bold]Processed data:[/] {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train",    type=float, default=0.8)
    parser.add_argument("--val",      type=float, default=0.1)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    assert args.train + args.val < 1.0, "train + val ratios must be < 1.0"
    prepare_data(args.data_dir, args.train, args.val, args.seed)


if __name__ == "__main__":
    main()
