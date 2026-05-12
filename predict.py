"""
DogCat Vision — Inference Script
==================================
Run predictions on single images or entire folders.

Usage:
    # Single image
    python src/predict.py --image my_dog.jpg

    # Batch folder
    python src/predict.py --folder my_images/ --output results/

    # With Grad-CAM visualization
    python src/predict.py --image my_dog.jpg --visualize

    # From URL
    python src/predict.py --url https://example.com/dog.jpg
"""

import sys
import io
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.efficientnet import DogCatClassifier
from src.utils.gradcam import GradCAM
from src.utils.dataset import get_inference_transforms

console = Console()

CLASS_NAMES = {0: "🐱 Cat", 1: "🐶 Dog"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "checkpoints/best_model.pth"


def load_model() -> DogCatClassifier:
    if Path(CHECKPOINT).exists():
        model = DogCatClassifier.load_checkpoint(CHECKPOINT, device=DEVICE)
    else:
        console.print("[yellow]⚠ No checkpoint found. Using pretrained weights only.[/]")
        model = DogCatClassifier(pretrained=True)
        model.to(DEVICE)
    model.eval()
    return model


def preprocess(image: Image.Image, input_size: int = 380) -> torch.Tensor:
    transform = get_inference_transforms(input_size)
    image_np = np.array(image.convert("RGB"))
    return transform(image=image_np)["image"].unsqueeze(0).to(DEVICE)


@torch.no_grad()
def predict_image(model: DogCatClassifier, image: Image.Image) -> dict:
    import time
    tensor = preprocess(image)
    t0 = time.perf_counter()
    logits = model(tensor)
    elapsed = (time.perf_counter() - t0) * 1000

    probs = torch.softmax(logits, dim=1)[0]
    pred_class = probs.argmax().item()
    confidence = probs[pred_class].item()

    return {
        "class": pred_class,
        "label": CLASS_NAMES[pred_class],
        "confidence": confidence,
        "dog_prob": probs[1].item(),
        "cat_prob": probs[0].item(),
        "inference_ms": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="DogCat Vision — Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--folder", type=str, help="Path to folder of images")
    group.add_argument("--url", type=str, help="URL of an image")

    parser.add_argument("--output", type=str, default="results", help="Output directory for batch")
    parser.add_argument("--visualize", action="store_true", help="Generate Grad-CAM visualizations")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    args = parser.parse_args()

    console.rule("[bold cyan]🐾 DogCat Vision Inference")
    model = load_model()

    if args.image:
        image = Image.open(args.image)
        result = predict_image(model, image)

        panel_color = "orange1" if result["class"] == 1 else "medium_purple1"
        console.print(Panel(
            f"[bold]{result['label']}[/bold]\n"
            f"Confidence: [bold]{result['confidence']:.1%}[/bold]\n"
            f"Dog: {result['dog_prob']:.4f} | Cat: {result['cat_prob']:.4f}\n"
            f"Inference: {result['inference_ms']:.1f}ms",
            title="🔮 Prediction",
            border_style=panel_color,
        ))

        if args.visualize:
            cam = GradCAM(model)
            tensor = preprocess(image)
            tensor.requires_grad_(True)
            heatmap = cam.generate(tensor, target_class=result["class"])
            out_path = Path(args.image).stem + "_gradcam.png"
            cam.save_overlay(np.array(image.convert("RGB")), heatmap, out_path)
            console.print(f"[green]Grad-CAM saved: {out_path}[/]")

    elif args.folder:
        folder = Path(args.folder)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)

        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
        if not images:
            console.print("[red]No images found in folder[/]")
            return

        table = Table(title=f"Batch Results — {folder.name}")
        table.add_column("File", style="cyan")
        table.add_column("Prediction", style="bold")
        table.add_column("Confidence")
        table.add_column("Time (ms)")

        dogs, cats = 0, 0
        for img_path in images:
            image = Image.open(img_path)
            result = predict_image(model, image)
            if result["class"] == 1: dogs += 1
            else: cats += 1
            color = "orange1" if result["class"] == 1 else "medium_purple1"
            table.add_row(
                img_path.name,
                f"[{color}]{result['label']}[/]",
                f"{result['confidence']:.1%}",
                f"{result['inference_ms']:.1f}",
            )

        console.print(table)
        console.print(f"\n🐶 Dogs: {dogs} | 🐱 Cats: {cats} | Total: {len(images)}")

    elif args.url:
        resp = requests.get(args.url, timeout=10)
        image = Image.open(io.BytesIO(resp.content))
        result = predict_image(model, image)
        console.print(f"Result: [bold]{result['label']}[/bold] ({result['confidence']:.1%})")


if __name__ == "__main__":
    main()
