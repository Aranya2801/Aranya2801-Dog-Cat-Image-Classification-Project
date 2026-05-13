"""
DogCat Vision — Dataset Download Script
=========================================
Downloads and prepares the Microsoft Cats vs Dogs dataset.

Usage:
    python scripts/download_dataset.py --method kaggle
    python scripts/download_dataset.py --method huggingface
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


def download_from_huggingface(data_dir: Path):
    """Download using HuggingFace datasets (no account needed)."""
    console.print("[bold cyan]Downloading from HuggingFace...[/]")
    from datasets import load_dataset

    dataset = load_dataset("microsoft/cats_vs_dogs")
    console.print(f"✅ Dataset loaded: {len(dataset['train'])} images")

    # Save images
    cats_dir = data_dir / "raw" / "cats"
    dogs_dir = data_dir / "raw" / "dogs"
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    cat_count, dog_count = 0, 0
    for item in track(dataset["train"], description="Saving images..."):
        label = item["labels"]  # 0=cat, 1=dog
        image = item["image"]
        if label == 0:
            image.save(str(cats_dir / f"cat_{cat_count:05d}.jpg"))
            cat_count += 1
        else:
            image.save(str(dogs_dir / f"dog_{dog_count:05d}.jpg"))
            dog_count += 1

    console.print(f"✅ Saved: {cat_count} cats, {dog_count} dogs")
    return cats_dir, dogs_dir


def download_from_kaggle(data_dir: Path):
    """Download using Kaggle API."""
    console.print("[bold cyan]Downloading from Kaggle...[/]")
    console.print("Requires ~/.kaggle/kaggle.json — get it at https://www.kaggle.com/settings > API")

    import subprocess
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "kaggle", "competitions", "download", "-c", "dogs-vs-cats",
        "-p", str(raw_dir)
    ], check=True)

    subprocess.run(["unzip", "-q", str(raw_dir / "dogs-vs-cats.zip"), "-d", str(raw_dir)], check=True)
    subprocess.run(["unzip", "-q", str(raw_dir / "train.zip"), "-d", str(raw_dir)], check=True)

    console.print("✅ Kaggle dataset downloaded and extracted")


def split_dataset(raw_dir: Path, output_dir: Path, train_ratio=0.8, val_ratio=0.1):
    """Split into train/val/test sets."""
    console.print("[bold cyan]Splitting dataset...[/]")

    for split in ["train", "val", "test"]:
        for cls in ["cats", "dogs"]:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls_name, label in [("cats", "cat"), ("dogs", "dog")]:
        src_dir = raw_dir / cls_name
        if not src_dir.exists():
            # Try Kaggle's flat structure: cat.0.jpg
            images = list((raw_dir / "train").glob(f"{label}.*.jpg"))
        else:
            images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.png"))

        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            for i, img in enumerate(track(split_images, description=f"  {split_name}/{cls_name}...")):
                dest = output_dir / split_name / cls_name / f"{label}_{i:05d}{img.suffix}"
                shutil.copy2(str(img), str(dest))

    console.print("✅ Dataset split complete")
    console.print(f"  Train: {len(list((output_dir/'train').rglob('*.jpg')))} images")
    console.print(f"  Val:   {len(list((output_dir/'val').rglob('*.jpg')))} images")
    console.print(f"  Test:  {len(list((output_dir/'test').rglob('*.jpg')))} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["kaggle", "huggingface"], default="huggingface")
    parser.add_argument("--data-dir", default="data", type=Path)
    args = parser.parse_args()

    console.rule("[bold]🐾 DogCat Vision — Dataset Setup")

    if args.method == "huggingface":
        download_from_huggingface(args.data_dir)
    elif args.method == "kaggle":
        download_from_kaggle(args.data_dir)

    split_dataset(
        raw_dir=args.data_dir / "raw",
        output_dir=args.data_dir / "processed",
    )

    console.rule("[bold green]✅ Dataset ready!")
    console.print(f"Data location: {args.data_dir.resolve()}")


if __name__ == "__main__":
    main()
