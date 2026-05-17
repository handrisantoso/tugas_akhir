"""Utility to export LOSO results to CSV and JSON for research reproducibility."""
import json
import csv
import os
import datetime
import numpy as np
from config import GESTURE_NAMES, MODELS_DIR


def _native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(i) for i in obj]
    return obj


def export_loso_report(result, output_dir=None, model_type="unknown"):
    """Export LOSO cross-validation results to JSON and CSV files.

    Args:
        result: dict returned by evaluate_model_loso()
        output_dir: directory to save files. Defaults to MODELS_DIR/loso_reports/
        model_type: string label for the model (e.g., "rf", "mlp", "cnn1d")

    Returns:
        str: path to the generated JSON report
    """
    if output_dir is None:
        output_dir = os.path.join(MODELS_DIR, "loso_reports")
    os.makedirs(output_dir, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(output_dir, f"loso_{model_type}_{ts}")

    # ── JSON: full nested result ───────────────────────────────────
    with open(f"{base}.json", "w") as f:
        json.dump(_native(result), f, indent=2)

    # ── CSV: per-fold summary ───────────────────────────────────────
    with open(f"{base}_folds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "subject_held_out", "train_n", "test_n",
                    "accuracy", "precision", "recall", "f1"])
        for fl in result["per_fold"]:
            w.writerow([
                fl["fold"],
                fl["subject_held_out"],
                fl["train_n"],
                fl["test_n"],
                f"{fl['accuracy']:.4f}",
                f"{fl['precision']:.4f}",
                f"{fl['recall']:.4f}",
                f"{fl['f1']:.4f}",
            ])

    # ── CSV: aggregated confusion matrix ───────────────────────────
    with open(f"{base}_confusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + GESTURE_NAMES)
        for i, row in enumerate(result["confusion_agg"]):
            w.writerow([GESTURE_NAMES[i]] + row.tolist())

    # ── CSV: aggregated metrics ─────────────────────────────────────
    agg = result["aggregated"]
    with open(f"{base}_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std"])
        for key in ["accuracy", "precision", "recall", "f1"]:
            w.writerow([key, f"{agg[f'{key}_mean']:.4f}", f"{agg[f'{key}_std']:.4f}"])

    print(f"LOSO report saved to {output_dir}/")
    print(f"  - {os.path.basename(base)}.json     (full results)")
    print(f"  - {os.path.basename(base)}_folds.csv     (per-fold)")
    print(f"  - {os.path.basename(base)}_confusion.csv (confusion matrix)")
    print(f"  - {os.path.basename(base)}_metrics.csv   (aggregated metrics)")

    return f"{base}.json"