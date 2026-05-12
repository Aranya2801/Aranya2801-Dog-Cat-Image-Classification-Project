"""
DogCat Vision — Model Export Script
======================================
Export trained model to ONNX and TorchScript for production deployment.

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --checkpoint checkpoints/best_model.pth --format all
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.efficientnet import DogCatClassifier

console = Console()
DEVICE = "cpu"  # Export on CPU for portability


def benchmark_inference(model, input_size=(380, 380), n_runs=100, warmup=10):
    """Benchmark model inference speed."""
    dummy = torch.randn(1, 3, *input_size).to(DEVICE)
    model.eval()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy)

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms":   round(float(times.mean()), 2),
        "std_ms":    round(float(times.std()), 2),
        "p50_ms":    round(float(np.percentile(times, 50)), 2),
        "p95_ms":    round(float(np.percentile(times, 95)), 2),
        "p99_ms":    round(float(np.percentile(times, 99)), 2),
        "min_ms":    round(float(times.min()), 2),
        "max_ms":    round(float(times.max()), 2),
        "fps":       round(1000 / float(times.mean()), 1),
    }


def export_onnx(model, output_path, input_size=(380, 380)):
    """Export to ONNX format."""
    console.print("[cyan]Exporting to ONNX...[/]")
    model.eval()
    model.export_onnx(output_path, input_size=input_size)

    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        dummy = np.random.randn(1, 3, *input_size).astype(np.float32)
        output = sess.run(None, {"image": dummy})
        console.print(f"  ✅ ONNX verified | Output shape: {output[0].shape}")
        console.print(f"  📦 File size: {Path(output_path).stat().st_size / 1e6:.1f} MB")
    except ImportError:
        console.print("  ⚠️  onnxruntime not installed — skipping verification")
    except Exception as e:
        console.print(f"  ❌ ONNX verification failed: {e}")


def export_torchscript(model, output_path):
    """Export to TorchScript format."""
    console.print("[cyan]Exporting to TorchScript...[/]")
    model.eval()
    model.export_torchscript(output_path)
    console.print(f"  📦 File size: {Path(output_path).stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="DogCat Vision Model Export")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--output-dir", default="exports")
    parser.add_argument("--format", choices=["onnx", "torchscript", "all"], default="all")
    parser.add_argument("--input-size", type=int, default=380)
    parser.add_argument("--benchmark", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_size = (args.input_size, args.input_size)

    console.rule("[bold cyan]🐾 DogCat Vision — Model Export")

    # Load model
    if Path(args.checkpoint).exists():
        model = DogCatClassifier.load_checkpoint(args.checkpoint, device=DEVICE)
    else:
        console.print("[yellow]⚠️  No checkpoint found — using pretrained weights[/]")
        model = DogCatClassifier(pretrained=True)
        model.to(DEVICE)

    model.eval()
    console.print(f"✅ Model loaded | Parameters: {model.count_parameters():,}")

    # Benchmark PyTorch
    if args.benchmark:
        console.print("\n[bold]📊 PyTorch Inference Benchmark (CPU, 100 runs)[/]")
        stats = benchmark_inference(model, input_size)
        for key, val in stats.items():
            console.print(f"  {key:12s}: {val}")

    # Export
    if args.format in ["onnx", "all"]:
        export_onnx(model, str(output_dir / "dogcat_efficientnet_b4.onnx"), input_size)

    if args.format in ["torchscript", "all"]:
        export_torchscript(model, str(output_dir / "dogcat_efficientnet_b4.pt"))

    console.rule("[bold green]✅ Export Complete")
    console.print(f"Exported models: {output_dir.resolve()}")
    console.print("\n📌 Deployment Notes:")
    console.print("  ONNX:         Use with onnxruntime (Python, C++, Java, .NET, iOS, Android)")
    console.print("  TorchScript:  Use with libtorch (C++ production, mobile)")


if __name__ == "__main__":
    main()
