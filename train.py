"""
DogCat Vision — Advanced Training Pipeline
============================================
Production-grade training with:
  - Mixed precision training (AMP)
  - Gradient clipping
  - Cosine annealing + warmup scheduler
  - Label smoothing
  - MixUp augmentation
  - MLflow experiment tracking
  - Early stopping
  - Best model checkpointing
  - TensorBoard logging

Usage:
    python src/train.py
    python src/train.py --config configs/efficientnet_b4.yaml
    python src/train.py --epochs 20 --batch-size 32 --lr 1e-4
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import mlflow
import mlflow.pytorch
import yaml
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.efficientnet import DogCatClassifier, LabelSmoothingCrossEntropy, MixUpAugmentation
from src.utils.dataset import DogCatDataset, get_dataloaders
from src.utils.metrics import compute_metrics
from src.utils.augmentations import get_train_transforms, get_val_transforms

console = Console()

# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Model
    "model_name": "efficientnet_b4",
    "pretrained": True,
    "dropout_rate": 0.4,
    "num_classes": 2,

    # Training
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": 3,
    "gradient_clip_norm": 1.0,

    # Data
    "data_dir": "data/processed",
    "input_size": 380,
    "num_workers": 4,

    # Augmentation
    "use_mixup": True,
    "mixup_alpha": 0.4,
    "label_smoothing": 0.1,

    # Checkpointing
    "checkpoint_dir": "checkpoints",
    "save_every_n_epochs": 5,

    # Early Stopping
    "early_stopping_patience": 7,

    # Logging
    "mlflow_experiment": "dogcat-vision",
    "tensorboard_dir": "runs",

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mixed_precision": True,
}


# ─── Trainer Class ────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training lifecycle manager for DogCat Vision.
    Implements a two-phase training strategy:
      Phase 1: Freeze backbone, train classifier head (5 epochs)
      Phase 2: Unfreeze all, fine-tune end-to-end (remaining epochs)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config["device"])
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metrics tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        # Setup components
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_logging()

    def _setup_model(self):
        """Initialize model with two-phase transfer learning strategy."""
        self.model = DogCatClassifier(
            model_name=self.config["model_name"],
            num_classes=self.config["num_classes"],
            pretrained=self.config["pretrained"],
            dropout_rate=self.config["dropout_rate"],
            freeze_backbone=True,  # Phase 1: freeze backbone
        ).to(self.device)

        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=self.config["label_smoothing"]
        )
        self.mixup = MixUpAugmentation(alpha=self.config["mixup_alpha"])
        self.scaler = GradScaler(enabled=self.config["mixed_precision"])

    def _setup_data(self):
        """Setup DataLoaders with augmentation pipeline."""
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            data_dir=self.config["data_dir"],
            batch_size=self.config["batch_size"],
            input_size=self.config["input_size"],
            num_workers=self.config["num_workers"],
        )
        logger.info(
            f"📦 Data loaded | "
            f"Train: {len(self.train_loader.dataset):,} | "
            f"Val: {len(self.val_loader.dataset):,} | "
            f"Test: {len(self.test_loader.dataset):,}"
        )

    def _setup_optimizer(self):
        """AdamW optimizer with differential learning rates."""
        # Phase 1: Only train classifier head
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

    def _setup_scheduler(self):
        """Cosine annealing with linear warmup."""
        total_steps = self.config["epochs"] * len(self.train_loader)
        warmup_steps = self.config["warmup_epochs"] * len(self.train_loader)

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config["learning_rate"],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
        )

    def _setup_logging(self):
        """Setup MLflow and TensorBoard."""
        mlflow.set_experiment(self.config["mlflow_experiment"])
        self.tb_writer = SummaryWriter(
            log_dir=f"{self.config['tensorboard_dir']}/run_{int(time.time())}"
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch with AMP and MixUp."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Apply MixUp augmentation
            if self.config["use_mixup"]:
                images, labels_a, labels_b, lam = self.mixup(images, labels)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with automatic mixed precision
            with autocast(enabled=self.config["mixed_precision"]):
                logits = self.model(images)
                if self.config["use_mixup"]:
                    loss = MixUpAugmentation.mixup_criterion(
                        self.criterion, logits, labels_a, labels_b, lam
                    )
                else:
                    loss = self.criterion(logits, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["gradient_clip_norm"]
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels if not self.config["use_mixup"] else labels_a).sum().item()

        return {
            "loss": total_loss / len(self.train_loader),
            "accuracy": 100.0 * correct / total,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.config["mixed_precision"]):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)

            total_loss += loss.item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        metrics = compute_metrics(all_labels, all_preds, all_probs)
        metrics["loss"] = total_loss / len(self.val_loader)
        return metrics

    def _switch_to_phase2(self):
        """
        Phase 2: Unfreeze backbone and fine-tune end-to-end
        with a lower learning rate for the backbone.
        """
        logger.info("🔄 Switching to Phase 2: Full fine-tuning")
        self.model.unfreeze_backbone()

        # Differential learning rates: backbone gets 10x lower LR
        self.optimizer = optim.AdamW([
            {"params": self.model.backbone.parameters(), "lr": self.config["learning_rate"] / 10},
            {"params": self.model.classifier.parameters(), "lr": self.config["learning_rate"]},
        ], weight_decay=self.config["weight_decay"])

    def train(self):
        """Full training loop with two-phase strategy."""
        console.rule("[bold cyan]🐾 DogCat Vision Training Started")

        with mlflow.start_run():
            mlflow.log_params(self.config)
            phase2_started = False

            for epoch in range(1, self.config["epochs"] + 1):
                # Switch to Phase 2 at epoch 6
                if epoch == 6 and not phase2_started:
                    self._switch_to_phase2()
                    phase2_started = True

                epoch_start = time.time()
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate()
                epoch_time = time.time() - epoch_start

                # Update history
                self.history["train_loss"].append(train_metrics["loss"])
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["train_acc"].append(train_metrics["accuracy"])
                self.history["val_acc"].append(val_metrics["accuracy"])

                # Log to MLflow
                mlflow.log_metrics({
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_acc": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                }, step=epoch)

                # Log to TensorBoard
                self.tb_writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
                self.tb_writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                self.tb_writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
                self.tb_writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)

                # Print epoch summary
                self._print_epoch_summary(epoch, train_metrics, val_metrics, epoch_time)

                # Checkpointing
                is_best = val_metrics["accuracy"] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics["accuracy"]
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.model.save_checkpoint(
                        str(self.checkpoint_dir / "best_model.pth"),
                        epoch=epoch,
                        optimizer_state=self.optimizer.state_dict(),
                        metrics=val_metrics,
                    )
                    logger.info(f"🏆 New best model! Val Acc: {self.best_val_acc:.2f}%")
                else:
                    self.patience_counter += 1

                if epoch % self.config["save_every_n_epochs"] == 0:
                    self.model.save_checkpoint(
                        str(self.checkpoint_dir / f"epoch_{epoch:03d}.pth"),
                        epoch=epoch,
                        metrics=val_metrics,
                    )

                # Early stopping
                if self.patience_counter >= self.config["early_stopping_patience"]:
                    logger.info(
                        f"⏹️  Early stopping at epoch {epoch}. "
                        f"Best was epoch {self.best_epoch} ({self.best_val_acc:.2f}%)"
                    )
                    break

            # Log final model
            mlflow.pytorch.log_model(self.model, "dogcat_efficientnet_b4")
            self._print_final_summary()
            self.tb_writer.close()

    def _print_epoch_summary(
        self,
        epoch: int,
        train_m: Dict,
        val_m: Dict,
        elapsed: float,
    ):
        table = Table(title=f"Epoch {epoch}/{self.config['epochs']}")
        table.add_column("Metric", style="cyan")
        table.add_column("Train", style="yellow")
        table.add_column("Val", style="green")
        table.add_row("Loss", f"{train_m['loss']:.4f}", f"{val_m['loss']:.4f}")
        table.add_row("Accuracy", f"{train_m['accuracy']:.2f}%", f"{val_m['accuracy']:.2f}%")
        table.add_row("F1", "—", f"{val_m.get('f1', 0):.4f}")
        table.add_row("Time", f"{elapsed:.1f}s", "—")
        console.print(table)

    def _print_final_summary(self):
        console.rule("[bold green]✅ Training Complete")
        console.print(f"[bold]Best Validation Accuracy:[/bold] {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        console.print(f"[bold]Best model saved to:[/bold] {self.checkpoint_dir}/best_model.pth")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DogCat Vision Training")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()

    # Load YAML config if provided
    if args.config:
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
        config.update(yaml_config)

    # CLI overrides
    if args.epochs: config["epochs"] = args.epochs
    if args.batch_size: config["batch_size"] = args.batch_size
    if args.lr: config["learning_rate"] = args.lr
    if args.device: config["device"] = args.device
    if args.data_dir: config["data_dir"] = args.data_dir

    logger.info(f"🚀 Starting training with config: {config}")
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
