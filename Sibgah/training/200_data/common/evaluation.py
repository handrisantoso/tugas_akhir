"""
Shared evaluation and plotting utilities for all gesture classification models.
All plots are saved to disk (non-interactive backend) so scripts can run headless.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: str,
    filename: str = "confusion_matrix.png",
    title: str = "Confusion Matrix (Test Set)",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


def plot_history(
    history,
    output_dir: str,
    filename: str = "training_history.png",
) -> None:
    """Plot Keras training history: loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    if "val_accuracy" in history.history:
        axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: str,
    filename: str = "classification_report.txt",
) -> None:
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        f.write(report)
    print(f"  Saved: {path}")


def save_metrics_summary(
    metrics: dict,
    output_dir: str,
    filename: str = "metrics_summary.txt",
) -> None:
    """Write a simple key: value metrics file."""
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"  Saved: {path}")
