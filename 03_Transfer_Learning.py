# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 🧠 Notebook 03 — EfficientNet-B4 Transfer Learning
# **DogCat Vision | Two-Phase Fine-Tuning**
#
# This notebook walks through the complete fine-tuning pipeline:
# - Phase 1: Freeze backbone, train custom head (5 epochs)
# - Phase 2: Unfreeze all, fine-tune end-to-end (15 epochs)
# - Mixed-precision training, MixUp augmentation, label smoothing
# - MLflow experiment tracking

# +
import sys
import time
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from src.models.efficientnet import DogCatClassifier, LabelSmoothingCrossEntropy, MixUpAugmentation
from src.utils.dataset import get_dataloaders

console = Console()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {DEVICE.upper()}")
print(f"🔥 PyTorch: {torch.__version__}")
if DEVICE == "cuda":
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# -

# ## 1. Load Data

# +
BATCH_SIZE = 32
INPUT_SIZE = 380
DATA_DIR   = "../data/processed"

train_loader, val_loader, test_loader = get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    input_size=INPUT_SIZE,
    num_workers=4,
)
print(f"📦 Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,} | Test: {len(test_loader.dataset):,}")
# -

# ## 2. Visualize Augmented Samples

# +
images, labels = next(iter(train_loader))
CLASS_NAMES = {0: "🐱 Cat", 1: "🐶 Dog"}
CLASS_COLORS = {0: "#a78bfa", 1: "#f97316"}

mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

fig, axes = plt.subplots(2, 8, figsize=(22, 6))
fig.patch.set_facecolor("#0d1117")
for i in range(16):
    ax = axes[i // 8, i % 8]
    img = (images[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    ax.imshow(img)
    label = labels[i].item()
    ax.set_title(CLASS_NAMES[label], color=CLASS_COLORS[label], fontsize=8, pad=3)
    ax.axis("off")

plt.suptitle("Augmented Training Batch (Albumentations Pipeline)", color="white", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("../assets/images/augmented_batch.png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
plt.show()
# -

# ## 3. Model Instantiation

# +
model = DogCatClassifier(
    model_name="efficientnet_b4",
    pretrained=True,
    dropout_rate=0.4,
    freeze_backbone=True,  # Phase 1 starts frozen
)
model = model.to(DEVICE)

total_params     = model.count_parameters(trainable_only=False)
trainable_params = model.count_parameters(trainable_only=True)
frozen_params    = total_params - trainable_params

print(f"📐 Total parameters:     {total_params:>12,}")
print(f"🔓 Trainable parameters: {trainable_params:>12,}  ({trainable_params/total_params*100:.1f}%)")
print(f"🔒 Frozen parameters:    {frozen_params:>12,}  ({frozen_params/total_params*100:.1f}%)")
print(f"\n📊 Classifier head architecture:")
print(model.classifier)
# -

# ## 4. Training Setup

# +
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
mixup = MixUpAugmentation(alpha=0.4)
scaler = GradScaler()

# Phase 1 optimizer (classifier head only)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-5,
)

history = {
    "train_loss": [], "val_loss": [],
    "train_acc":  [], "val_acc":  [],
    "lr": []
}

def train_one_epoch(model, loader, optimizer, criterion, mixup, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images, la, lb, lam = mixup(images, labels)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(images)
            loss = MixUpAugmentation.mixup_criterion(criterion, logits, la, lb, lam)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()
        _, pred = logits.max(1)
        total += labels.size(0); correct += pred.eq(la).sum().item()
    return total_loss / len(loader), 100. * correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        _, pred = logits.max(1)
        total += labels.size(0); correct += pred.eq(labels).sum().item()
    return total_loss / len(loader), 100. * correct / total

print("✅ Training utilities ready")
# -

# ## 5. Phase 1: Train Classifier Head (5 epochs)

# +
PHASE1_EPOCHS = 5
print(f"🔒 Phase 1: Training classifier head for {PHASE1_EPOCHS} epochs")
print(f"   Trainable params: {model.count_parameters():,}\n")

mlflow.set_experiment("dogcat-efficientnet-b4")

with mlflow.start_run(run_name="phase1_head_training"):
    mlflow.log_params({
        "model": "efficientnet_b4",
        "phase": 1,
        "epochs": PHASE1_EPOCHS,
        "lr": 1e-3,
        "batch_size": BATCH_SIZE,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.4,
    })

    best_val_acc = 0.0
    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, mixup, scaler, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        mlflow.log_metrics({
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
        }, step=epoch)

        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)
        star = "⭐" if is_best else "  "

        print(f"  {star} Epoch {epoch:02d}/{PHASE1_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.2f}% | {elapsed:.0f}s")

print(f"\n✅ Phase 1 complete | Best Val Acc: {best_val_acc:.2f}%")
# -

# ## 6. Phase 2: Full Fine-Tuning (15 epochs)

# +
PHASE2_EPOCHS = 15
model.unfreeze_backbone()
print(f"🔓 Phase 2: Full fine-tuning for {PHASE2_EPOCHS} epochs")
print(f"   Trainable params: {model.count_parameters():,}\n")

# Differential LRs: backbone gets 10x smaller LR
optimizer2 = optim.AdamW([
    {"params": model.backbone.parameters(),   "lr": 1e-5},
    {"params": model.classifier.parameters(), "lr": 1e-4},
], weight_decay=1e-5)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=PHASE2_EPOCHS, eta_min=1e-7)

with mlflow.start_run(run_name="phase2_full_finetune"):
    mlflow.log_params({
        "model": "efficientnet_b4", "phase": 2,
        "epochs": PHASE2_EPOCHS, "backbone_lr": 1e-5, "head_lr": 1e-4,
    })

    best_val_acc_p2 = 0.0
    for epoch in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer2, criterion, mixup, scaler, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer2.param_groups[1]["lr"])

        mlflow.log_metrics({
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
        }, step=PHASE1_EPOCHS + epoch)

        is_best = val_acc > best_val_acc_p2
        best_val_acc_p2 = max(best_val_acc_p2, val_acc)
        star = "⭐" if is_best else "  "
        print(f"  {star} Epoch {epoch:02d}/{PHASE2_EPOCHS} | "
              f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
              f"Val: {val_loss:.4f}/{val_acc:.2f}% | {elapsed:.0f}s")

    # Save best model
    model.save_checkpoint("../checkpoints/best_model.pth", epoch=PHASE1_EPOCHS + PHASE2_EPOCHS,
                          metrics={"val_acc": best_val_acc_p2})
    mlflow.pytorch.log_model(model, "efficientnet_b4_final")

print(f"\n🏆 Phase 2 complete | Best Val Acc: {best_val_acc_p2:.2f}%")
# -

# ## 7. Training Curves

# +
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("#0d1117")
all_epochs = list(range(1, len(history["train_loss"]) + 1))
phase_boundary = PHASE1_EPOCHS + 0.5

for ax in axes: ax.set_facecolor("#111827"); [s.set_color("#1f2d4a") for s in ax.spines.values()]

# Loss
axes[0].plot(all_epochs, history["train_loss"], color="#38bdf8", lw=2, label="Train Loss")
axes[0].plot(all_epochs, history["val_loss"],   color="#f97316", lw=2, label="Val Loss")
axes[0].axvline(phase_boundary, color="white", linestyle="--", alpha=0.4, label="Phase 2 start")
axes[0].set_title("Loss Curves", color="white", fontsize=12)
axes[0].set_xlabel("Epoch", color="white"); axes[0].tick_params(colors="white")
axes[0].legend(facecolor="#1a1a2e", labelcolor="white")

# Accuracy
axes[1].plot(all_epochs, history["train_acc"], color="#38bdf8", lw=2, label="Train Acc")
axes[1].plot(all_epochs, history["val_acc"],   color="#4ade80", lw=2, label="Val Acc")
axes[1].axvline(phase_boundary, color="white", linestyle="--", alpha=0.4)
axes[1].set_title("Accuracy Curves", color="white", fontsize=12)
axes[1].set_xlabel("Epoch", color="white"); axes[1].tick_params(colors="white")
axes[1].legend(facecolor="#1a1a2e", labelcolor="white")
axes[1].set_ylim([80, 100])

# Learning Rate
axes[2].semilogy(all_epochs, history["lr"], color="#a78bfa", lw=2)
axes[2].axvline(phase_boundary, color="white", linestyle="--", alpha=0.4)
axes[2].set_title("Learning Rate Schedule", color="white", fontsize=12)
axes[2].set_xlabel("Epoch", color="white"); axes[2].tick_params(colors="white")

plt.suptitle("EfficientNet-B4 Training Progress — Two-Phase Fine-Tuning",
             color="white", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("../assets/images/training_curves.png", dpi=140, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print(f"\n🏆 Final best validation accuracy: {max(history['val_acc']):.2f}%")
# -
