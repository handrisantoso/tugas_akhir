"""
Create evaluation plots across models:
  1) accuracy_f1_comparison.png
  2) confusion_matrix_test_all_models.png
  3) confusion_matrix_val_all_models.png

Inputs are read from each model's output directory.
"""

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(__file__)
EVAL_DIR = os.path.join(BASE_DIR, "eval")
os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_OUTPUT_DIRS = {
    "Random Forest": os.path.join(BASE_DIR, "random_forest", "output"),
    "1D CNN": os.path.join(BASE_DIR, "cnn_1d", "output"),
    "CNN-LSTM": os.path.join(BASE_DIR, "cnn_lstm", "output"),
}

METRIC_KEYS = [
    "val_accuracy",
    "test_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
]


def parse_metrics(metrics_path: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
    return metrics


def load_all_metrics() -> Dict[str, Dict[str, float]]:
    all_metrics: Dict[str, Dict[str, float]] = {}
    for model_name, output_dir in MODEL_OUTPUT_DIRS.items():
        metrics_path = os.path.join(output_dir, "metrics_summary.txt")
        if not os.path.exists(metrics_path):
            print(f"WARNING: metrics missing for {model_name}: {metrics_path}")
            continue
        model_metrics = parse_metrics(metrics_path)
        all_metrics[model_name] = {k: model_metrics.get(k, 0.0) for k in METRIC_KEYS}
    return all_metrics


def plot_accuracy_f1(all_metrics: Dict[str, Dict[str, float]]) -> None:
    if not all_metrics:
        print("No metrics available to plot.")
        return

    models = list(all_metrics.keys())
    labels = ["Val Acc", "Test Acc", "Macro F1", "Weighted F1"]
    x = np.arange(len(labels))
    width = 0.8 / len(models)
    colors = plt.cm.Set2.colors

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        values = [
            all_metrics[model]["val_accuracy"],
            all_metrics[model]["test_accuracy"],
            all_metrics[model]["test_macro_f1"],
            all_metrics[model]["test_weighted_f1"],
        ]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=colors[i % len(colors)],
            label=model,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_title("Model Evaluation: Accuracy and F1")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    plt.tight_layout()

    out_path = os.path.join(EVAL_DIR, "accuracy_f1_comparison.png")
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved: {out_path}")


def plot_confusion_montage(filename: str, output_filename: str, title: str) -> None:
    model_items = list(MODEL_OUTPUT_DIRS.items())
    fig, axes = plt.subplots(1, len(model_items), figsize=(5 * len(model_items), 4))
    if len(model_items) == 1:
        axes = [axes]

    for ax, (model_name, output_dir) in zip(axes, model_items):
        cm_path = os.path.join(output_dir, filename)
        if not os.path.exists(cm_path):
            ax.text(0.5, 0.5, "Not found", ha="center", va="center")
            ax.set_axis_off()
            ax.set_title(model_name)
            continue

        img = plt.imread(cm_path)
        ax.imshow(img)
        ax.set_title(model_name)
        ax.set_axis_off()

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(EVAL_DIR, output_filename)
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    all_metrics = load_all_metrics()
    plot_accuracy_f1(all_metrics)
    plot_confusion_montage(
        filename="confusion_matrix.png",
        output_filename="confusion_matrix_test_all_models.png",
        title="Confusion Matrix (Test) - All Models",
    )
    plot_confusion_montage(
        filename="confusion_matrix_val.png",
        output_filename="confusion_matrix_val_all_models.png",
        title="Confusion Matrix (Validation) - All Models",
    )
    print("Done.")


if __name__ == "__main__":
    main()
