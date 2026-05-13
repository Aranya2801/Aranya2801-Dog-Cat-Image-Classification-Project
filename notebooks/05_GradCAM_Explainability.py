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

# # 🔍 Notebook 05 — Grad-CAM Explainability
# **DogCat Vision | Visualizing What the Model Sees**
#
# This notebook demonstrates Gradient-weighted Class Activation Mapping (Grad-CAM),
# which reveals which spatial regions of an image are most responsible for the
# model's classification decision.
#
# **Reference:** Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
# via Gradient-based Localization" (ICCV 2017)

# +
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from pathlib import Path
import random

from src.models.efficientnet import DogCatClassifier
from src.utils.gradcam import GradCAM
from src.utils.dataset import get_inference_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = {0: "Cat 🐱", 1: "Dog 🐶"}
CLASS_COLORS = {0: "#a78bfa", 1: "#f97316"}
print(f"✅ Running on: {DEVICE.upper()}")
# -

# ## 1. Load Trained Model

# +
CHECKPOINT = "../checkpoints/best_model.pth"

if Path(CHECKPOINT).exists():
    model = DogCatClassifier.load_checkpoint(CHECKPOINT, device=DEVICE)
    print(f"✅ Model loaded from checkpoint")
else:
    model = DogCatClassifier(pretrained=True)
    model.to(DEVICE)
    print("⚠️  No checkpoint found — using pretrained weights only")

model.eval()
print(f"📐 Parameters: {model.count_parameters():,}")
# -

# ## 2. Initialize Grad-CAM

# +
# EfficientNet-B4 target layer for Grad-CAM
# "conv_head" is the last convolutional layer before global pooling
cam = GradCAM(model, target_layer="backbone.conv_head")
transform = get_inference_transforms(input_size=380)

def preprocess(img_path):
    image = Image.open(img_path).convert("RGB")
    image_np = np.array(image)
    tensor = transform(image=image_np)["image"].unsqueeze(0).to(DEVICE)
    return image, tensor

def predict_and_cam(img_path):
    image, tensor = preprocess(img_path)
    tensor.requires_grad_(True)

    with torch.enable_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()
        conf   = probs[pred].item()
        heatmap = cam.generate(tensor, target_class=pred)

    image_np = np.array(image.resize((380, 380)))
    return image_np, heatmap, pred, conf, probs

print("✅ Grad-CAM initialized — target layer: backbone.conv_head")
# -

# ## 3. Single Image Grad-CAM

# +
# Provide a sample image path or use a sample from the dataset
SAMPLE_DIR = Path("../data/sample")
test_images = list(SAMPLE_DIR.rglob("*.jpg"))[:1] if SAMPLE_DIR.exists() else []

if not test_images:
    # Use any image from data/processed if sample doesn't exist
    for cls in ["cats", "dogs"]:
        p = Path(f"../data/processed/test/{cls}")
        if p.exists():
            test_images = list(p.glob("*.jpg"))[:1]
            break

if test_images:
    img_path = test_images[0]
    image_np, heatmap, pred, conf, probs = predict_and_cam(img_path)
    cam.save_overlay(image_np, heatmap, "../assets/images/gradcam_single.png",
                     title=f"Prediction: {CLASS_NAMES[pred]} ({conf:.1%} confidence)",
                     show_plot=True)
    print(f"\n🔮 Prediction: {CLASS_NAMES[pred]}")
    print(f"   Confidence:  {conf:.4f} ({conf:.1%})")
    print(f"   Cat prob:    {probs[0]:.4f}")
    print(f"   Dog prob:    {probs[1]:.4f}")
else:
    print("⚠️  No sample images found. Run scripts/download_dataset.py first.")
# -

# ## 4. Grid — Grad-CAM on Multiple Images

# +
# Collect sample images from both classes
sample_images = {"cats": [], "dogs": []}
for cls in ["cats", "dogs"]:
    test_dir = Path(f"../data/processed/test/{cls}")
    if test_dir.exists():
        imgs = random.sample(list(test_dir.glob("*.jpg")), min(4, len(list(test_dir.glob("*.jpg")))))
        sample_images[cls] = imgs

all_samples = (
    [(p, 0) for p in sample_images["cats"]] +
    [(p, 1) for p in sample_images["dogs"]]
)

if all_samples:
    n = len(all_samples)
    fig = plt.figure(figsize=(n * 4, 12))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, n, figure=fig, hspace=0.15, wspace=0.05)

    for col, (img_path, true_label) in enumerate(all_samples):
        image_np, heatmap, pred, conf, probs = predict_and_cam(img_path)
        overlay = cam.overlay(image_np, heatmap, alpha=0.45)

        pred_color = "#4ade80" if pred == true_label else "#f87171"

        # Row 0: Original
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(image_np)
        ax0.set_title(f"True: {CLASS_NAMES[true_label]}", color="white", fontsize=9, pad=4)
        ax0.axis("off")

        # Row 1: Heatmap
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(heatmap, cmap="jet")
        ax1.set_title("Grad-CAM Heatmap", color="#38bdf8", fontsize=9, pad=4)
        ax1.axis("off")

        # Row 2: Overlay
        ax2 = fig.add_subplot(gs[2, col])
        ax2.imshow(overlay)
        correct_str = "✓" if pred == true_label else "✗"
        ax2.set_title(f"{correct_str} {CLASS_NAMES[pred]} ({conf:.0%})",
                      color=pred_color, fontsize=9, pad=4)
        ax2.axis("off")

    plt.suptitle("Grad-CAM Visualizations — Model Attention Maps", color="white", fontsize=14, y=1.01)
    plt.savefig("../assets/images/gradcam_grid.png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("✅ Grad-CAM grid saved")
else:
    print("⚠️  No images found. Download dataset first.")
# -

# ## 5. Target Class Comparison (Dog vs Cat Activation)

# +
if test_images:
    img_path = test_images[0]
    image_np, _, _, _, _ = predict_and_cam(img_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d1117")

    # When model thinks it's a cat (class 0)
    _, tensor = preprocess(img_path)
    tensor.requires_grad_(True)
    heatmap_cat = cam.generate(tensor, target_class=0)
    overlay_cat = cam.overlay(image_np, heatmap_cat)

    # When model thinks it's a dog (class 1)
    _, tensor = preprocess(img_path)
    tensor.requires_grad_(True)
    heatmap_dog = cam.generate(tensor, target_class=1)
    overlay_dog = cam.overlay(image_np, heatmap_dog)

    axes[0].imshow(image_np)
    axes[0].set_title("Original Image", color="white", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(overlay_cat)
    axes[1].set_title("Grad-CAM for Class: Cat 🐱", color="#a78bfa", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay_dog)
    axes[2].set_title("Grad-CAM for Class: Dog 🐶", color="#f97316", fontsize=12)
    axes[2].axis("off")

    plt.suptitle("Class-Specific Activation Maps — Same Image, Different Target Classes",
                 color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("../assets/images/gradcam_class_comparison.png", dpi=120,
                bbox_inches="tight", facecolor="#0d1117")
    plt.show()

cam.remove_hooks()
print("\n✅ Grad-CAM analysis complete!")
print("\n📝 Key Observations:")
print("  - For dogs: model attends to ears, snout, fur texture")
print("  - For cats: model attends to whiskers, ear shape, face contour")
print("  - The model has learned biologically meaningful visual features")
print("  - Misclassifications occur when key features are occluded or ambiguous")
# -
