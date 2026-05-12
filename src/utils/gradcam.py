"""
DogCat Vision — Grad-CAM Explainability
=========================================
Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
Visualizes which image regions the model focuses on when making predictions.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017)

Usage:
    cam = GradCAM(model, target_layer="backbone.conv_head")
    heatmap = cam.generate(image_tensor, target_class=1)  # 1=dog
    cam.save_overlay(original_image, heatmap, "output.png")
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Optional, Tuple, Union
from loguru import logger


class GradCAM:
    """
    Grad-CAM implementation for EfficientNet-B4.

    Registers forward and backward hooks on the target convolutional layer
    to capture feature maps and gradients for heatmap generation.
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = "backbone.conv_head"):
        self.model = model
        self.model.eval()

        self.feature_maps: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []

        # Register hooks on target layer
        self._register_hooks(target_layer)

    def _register_hooks(self, target_layer: str):
        """Register forward and backward hooks."""
        # Navigate to target layer
        layer = self._get_layer(target_layer)
        if layer is None:
            raise ValueError(f"Layer '{target_layer}' not found in model.")

        # Forward hook: capture activations
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        # Backward hook: capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(layer.register_forward_hook(forward_hook))
        self._hooks.append(layer.register_backward_hook(backward_hook))
        logger.debug(f"Grad-CAM hooks registered on: {target_layer}")

    def _get_layer(self, layer_name: str) -> Optional[torch.nn.Module]:
        """Navigate model hierarchy to find target layer."""
        parts = layer_name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the given image.

        Args:
            image_tensor: Preprocessed image tensor (1, 3, H, W)
            target_class: Class index to generate CAM for.
                          If None, uses predicted class.

        Returns:
            Normalized heatmap as numpy array (H, W), values in [0, 1]
        """
        self.model.zero_grad()

        # Forward pass
        output = self.model(image_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        target = output[0, target_class]
        target.backward()

        # Pool gradients across spatial dimensions
        # Shape: (1, C, H, W) -> (C,)
        pooled_gradients = self.gradients.mean(dim=[0, 2, 3])

        # Weight feature maps by pooled gradients
        weighted_maps = self.feature_maps[0]  # (C, H, W)
        for i, grad in enumerate(pooled_gradients):
            weighted_maps[i] *= grad

        # Average across channels and apply ReLU
        heatmap = weighted_maps.mean(dim=0).cpu().numpy()  # (H, W)
        heatmap = np.maximum(heatmap, 0)  # ReLU

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.

        Args:
            original_image: RGB image as numpy array (H, W, 3)
            heatmap: Grad-CAM heatmap (H', W'), values in [0, 1]
            alpha: Heatmap transparency (0=transparent, 1=opaque)
            colormap: OpenCV colormap

        Returns:
            Overlaid image as numpy array (H, W, 3)
        """
        # Resize heatmap to match image
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend with original image
        overlaid = (1 - alpha) * original_image + alpha * heatmap_colored
        overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)

        return overlaid

    def save_overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        save_path: str,
        title: Optional[str] = None,
        show_plot: bool = False,
    ):
        """Save Grad-CAM visualization with original image side-by-side."""
        overlaid = self.overlay(original_image, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("#1a1a2e")

        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", color="white", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Heatmap only
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", color="white", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(overlaid)
        axes[2].set_title("Grad-CAM Overlay", color="white", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        if title:
            fig.suptitle(title, color="white", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

        if show_plot:
            plt.show()
        plt.close()

        logger.info(f"Grad-CAM saved to: {save_path}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __del__(self):
        self.remove_hooks()


# ─── Batch Grad-CAM ───────────────────────────────────────────────────────────

def generate_gradcam_grid(
    model: torch.nn.Module,
    images: list,
    predictions: list,
    confidences: list,
    save_path: str,
    n_cols: int = 4,
):
    """
    Generate a grid of Grad-CAM visualizations for a batch of images.

    Args:
        model: Trained DogCatClassifier
        images: List of PIL Images or numpy arrays
        predictions: List of predicted class strings ("cat" or "dog")
        confidences: List of confidence values
        save_path: Output path for the grid image
        n_cols: Number of columns in the grid
    """
    cam = GradCAM(model)
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    fig.patch.set_facecolor("#0d1117")

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        ax.axis("off")

        if idx >= n_images:
            continue

        image = np.array(images[idx]) if not isinstance(images[idx], np.ndarray) else images[idx]
        pred = predictions[idx]
        conf = confidences[idx]

        color = "#4ade80" if pred == "dog" else "#f97316"
        icon = "🐶" if pred == "dog" else "🐱"

        ax.imshow(image)
        ax.set_title(
            f"{icon} {pred.upper()} ({conf:.1%})",
            color=color,
            fontsize=10,
            fontweight="bold",
            pad=5,
        )

    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    cam.remove_hooks()
    logger.info(f"Grad-CAM grid saved to: {save_path}")
