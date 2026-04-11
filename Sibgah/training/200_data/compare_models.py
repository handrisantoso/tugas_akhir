"""
Cross-model comparison.
Reads metrics_summary.txt from each model's output folder and produces:
  - compare_models.png  — grouped bar chart of val/test accuracy + test F1
  - compare_models.txt  — summary table
Saved next to this script (training/).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(__file__)
OUTPUT_FILE_PNG = os.path.join(BASE_DIR, "compare_models.png")
OUTPUT_FILE_TXT = os.path.join(BASE_DIR, "compare_models.txt")

MODEL_DIRS = {
    "Random Forest": os.path.join(BASE_DIR, "random_forest", "output", "metrics_summary.txt"),
    "1D CNN":        os.path.join(BASE_DIR, "cnn_1d",       "output", "metrics_summary.txt"),
    "CNN-LSTM":      os.path.join(BASE_DIR, "cnn_lstm",     "output", "metrics_summary.txt"),
}

METRICS_OF_INTEREST = [
    ("val_accuracy",      "Val Accuracy"),
    ("test_accuracy",     "Test Accuracy"),
    ("test_macro_f1",     "Test Macro F1"),
    ("test_weighted_f1",  "Test Weighted F1"),
]


def parse_metrics(path: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, _, val = line.partition(":")
                try:
                    metrics[key.strip()] = float(val.strip())
                except ValueError:
                    pass
    return metrics


def main() -> None:
    all_metrics: dict[str, dict[str, float]] = {}
    for model_name, path in MODEL_DIRS.items():
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping {model_name}")
            continue
        all_metrics[model_name] = parse_metrics(path)

    if not all_metrics:
        print("No metrics found. Run each model's train.py first.")
        return

    models = list(all_metrics.keys())
    n_metrics = len(METRICS_OF_INTEREST)
    n_models = len(models)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.Set2.colors

    for i, model in enumerate(models):
        values = [
            all_metrics[model].get(key, 0.0)
            for key, _ in METRICS_OF_INTEREST
        ]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=colors[i], edgecolor="black", linewidth=0.5)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRICS_OF_INTEREST])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Gesture Classification")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_PNG, dpi=120)
    plt.close()
    print(f"Saved: {OUTPUT_FILE_PNG}")

    # Text summary table
    col_w = 18
    header = f"{'Model':<{col_w}}" + "".join(f"{lbl:>{col_w}}" for _, lbl in METRICS_OF_INTEREST)
    rows = []
    for model in models:
        row = f"{model:<{col_w}}"
        for key, _ in METRICS_OF_INTEREST:
            val = all_metrics[model].get(key, float("nan"))
            row += f"{val:>{col_w}.4f}"
        rows.append(row)

    table = "\n".join([header, "-" * len(header)] + rows)
    print("\n" + table + "\n")
    with open(OUTPUT_FILE_TXT, "w") as f:
        f.write(table + "\n")
    print(f"Saved: {OUTPUT_FILE_TXT}")


if __name__ == "__main__":
    main()
