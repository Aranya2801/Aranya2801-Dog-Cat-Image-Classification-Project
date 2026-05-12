"""
DogCat Vision — Evaluation Script
====================================
Comprehensive model evaluation with:
  - Accuracy, Precision, Recall, F1, AUC-ROC
  - Confusion matrix visualization
  - Per-class metrics
  - Error analysis (worst predictions)
  - Grad-CAM on misclassified samples

Usage:
    python src/evaluate.py --checkpoint checkpoints/best_model.pth
    python src/evaluate.py --checkpoint checkpoints/best_model.pth --generate-cam
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.cuda.amp import autocast
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    auc, precision_recall_curve
)
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.efficientnet import DogCatClassifier
from src.utils.dataset import get_dataloaders
from src.utils.gradcam import GradCAM

console = Console()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Cat", "Dog"]


def evaluate(
    model: DogCatClassifier,
    test_loader,
    generate_cam: bool = False,
    output_dir: Path = Path("evaluation"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    all_images, all_paths = [], []
    total_time = 0.0
    import time

    console.print("[cyan]Running evaluation...[/]")
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, labels, paths = batch
                all_paths.extend(paths)
            else:
                images, labels = batch
                paths = [""] * len(labels)

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            t0 = time.perf_counter()
            with autocast():
                logits = model(images)
            total_time += (time.perf_counter() - t0) * 1000

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_images.extend(images.cpu())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    n = len(all_labels)
    avg_ms = total_time / n

    # ── Print metrics ────────────────────────────────────────────
    table = Table(title="📊 Evaluation Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    accuracy = (all_preds == all_labels).mean() * 100
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)

    table.add_row("Accuracy", f"[bold green]{accuracy:.2f}%[/]")
    table.add_row("Cat Precision", f"{report['Cat']['precision']:.4f}")
    table.add_row("Cat Recall",    f"{report['Cat']['recall']:.4f}")
    table.add_row("Cat F1",        f"{report['Cat']['f1-score']:.4f}")
    table.add_row("Dog Precision", f"{report['Dog']['precision']:.4f}")
    table.add_row("Dog Recall",    f"{report['Dog']['recall']:.4f}")
    table.add_row("Dog F1",        f"{report['Dog']['f1-score']:.4f}")
    table.add_row("Macro F1",      f"{report['macro avg']['f1-score']:.4f}")
    table.add_row("Avg Inference", f"{avg_ms:.2f} ms/image")
    table.add_row("Total Images",  str(n))
    console.print(table)

    # ── Confusion matrix ─────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0],
                cbar_kws={"shrink": 0.8})
    axes[0].set_title("Confusion Matrix (Counts)", color="white", fontsize=13, pad=10)
    axes[0].set_xlabel("Predicted", color="white")
    axes[0].set_ylabel("Actual", color="white")
    axes[0].tick_params(colors="white")

    # Normalized
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1],
                cbar_kws={"shrink": 0.8})
    axes[1].set_title("Confusion Matrix (Normalized)", color="white", fontsize=13, pad=10)
    axes[1].set_xlabel("Predicted", color="white")
    axes[1].set_ylabel("Actual", color="white")
    axes[1].tick_params(colors="white")

    plt.suptitle(f"EfficientNet-B4 Evaluation | Accuracy: {accuracy:.2f}%",
                 color="white", fontsize=14, y=1.02)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    console.print(f"[green]Confusion matrix saved: {cm_path}[/]")

    # ── ROC + PR curves ───────────────────────────────────────────
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall_vals, precision_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")

    axes[0].plot(fpr, tpr, color="#38bdf8", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
    axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1.02])
    axes[0].set_xlabel("False Positive Rate", color="white")
    axes[0].set_ylabel("True Positive Rate", color="white")
    axes[0].set_title("ROC Curve", color="white", fontsize=13)
    axes[0].legend(loc="lower right", facecolor="#1a1a2e", labelcolor="white")
    axes[0].set_facecolor("#111827"); axes[0].tick_params(colors="white")
    for sp in axes[0].spines.values(): sp.set_color("#1f2d4a")

    axes[1].plot(recall_vals, precision_vals, color="#a78bfa", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    axes[1].set_xlabel("Recall", color="white")
    axes[1].set_ylabel("Precision", color="white")
    axes[1].set_title("Precision-Recall Curve", color="white", fontsize=13)
    axes[1].legend(facecolor="#1a1a2e", labelcolor="white")
    axes[1].set_facecolor("#111827"); axes[1].tick_params(colors="white")
    for sp in axes[1].spines.values(): sp.set_color("#1f2d4a")

    plt.tight_layout()
    curves_path = output_dir / "roc_pr_curves.png"
    plt.savefig(curves_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    console.print(f"[green]ROC/PR curves saved: {curves_path}[/]")
    console.print(f"[bold]ROC-AUC: {roc_auc:.4f}  |  PR-AUC: {pr_auc:.4f}[/]")

    # ── Error analysis ────────────────────────────────────────────
    wrong_idx = np.where(all_preds != all_labels)[0]
    console.print(f"\n[yellow]Misclassified: {len(wrong_idx)} / {n} ({len(wrong_idx)/n:.1%})[/]")

    if generate_cam and len(wrong_idx) > 0:
        cam = GradCAM(model)
        sample_wrong = wrong_idx[:6]
        fig, axes = plt.subplots(2, 6, figsize=(20, 7))
        fig.patch.set_facecolor("#0d1117")
        fig.suptitle("Misclassified Examples + Grad-CAM", color="white", fontsize=13)

        for i, idx in enumerate(sample_wrong):
            img_tensor = all_images[idx].unsqueeze(0).to(DEVICE)
            img_tensor.requires_grad_(True)

            # Denormalize for display
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            img_display = (all_images[idx] * std + mean).clamp(0,1).permute(1,2,0).numpy()
            img_display = (img_display * 255).astype(np.uint8)

            heatmap = cam.generate(img_tensor, target_class=int(all_preds[idx]))
            overlay = cam.overlay(img_display, heatmap)

            actual = CLASS_NAMES[int(all_labels[idx])]
            predicted = CLASS_NAMES[int(all_preds[idx])]
            conf = all_probs[idx] if all_preds[idx] == 1 else 1 - all_probs[idx]

            axes[0][i].imshow(img_display)
            axes[0][i].set_title(f"Actual: {actual}", color="#4ade80", fontsize=9)
            axes[0][i].axis("off")

            axes[1][i].imshow(overlay)
            axes[1][i].set_title(f"Pred: {predicted} ({conf:.0%})", color="#f87171", fontsize=9)
            axes[1][i].axis("off")

        plt.tight_layout()
        err_path = output_dir / "error_analysis_gradcam.png"
        plt.savefig(err_path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        cam.remove_hooks()
        console.print(f"[green]Error analysis Grad-CAM saved: {err_path}[/]")

    console.rule(f"[bold green]✅ Evaluation complete — Accuracy: {accuracy:.2f}%")
    return {"accuracy": accuracy, "roc_auc": roc_auc, "pr_auc": pr_auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data", default="data/processed/test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--generate-cam", action="store_true")
    parser.add_argument("--output", default="evaluation")
    args = parser.parse_args()

    model = DogCatClassifier.load_checkpoint(args.checkpoint, device=DEVICE)
    _, _, test_loader = get_dataloaders(
        data_dir=str(Path(args.data).parent),
        batch_size=args.batch_size,
    )
    evaluate(model, test_loader, generate_cam=args.generate_cam, output_dir=Path(args.output))


if __name__ == "__main__":
    main()
