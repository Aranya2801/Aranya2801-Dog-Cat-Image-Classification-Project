"""
DogCat Vision — Metrics Computation
"""
from typing import List, Optional, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


def compute_metrics(
    labels: List[int],
    predictions: List[int],
    probabilities: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics."""
    labels = np.array(labels)
    predictions = np.array(predictions)

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)) * 100,
        "precision": float(precision_score(labels, predictions, average="binary", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="binary", zero_division=0)),
        "f1": float(f1_score(labels, predictions, average="binary", zero_division=0)),
    }

    if probabilities is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        except Exception:
            metrics["roc_auc"] = 0.0

    return metrics
