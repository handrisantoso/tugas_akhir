"""
Phase 1: Random Forest gesture classifier.

Features: mean, std, min, max, median, RMS per IMU channel (12 channels × 6 = 72 features).
Outputs saved to random_forest/output/:
  - confusion_matrix.png        (test set)
  - confusion_matrix_val.png    (validation set)
  - classification_report.txt   (test set)
  - feature_importance.png      (top 20 features)
  - metrics_summary.txt
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.preprocessing import (
    load_file_list, split_file_list, get_class_names,
    extract_windows, fit_scaler, apply_scaler, compute_class_weights,
    FEATURES,
)
from common.evaluation import (
    plot_confusion_matrix, save_classification_report, save_metrics_summary,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "cleaned_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Compute window-level statistical features per channel.
    X: (N, window_size, 12)  ->  output: (N, 72)
    Stats: mean, std, min, max, median, RMS
    """
    parts = []
    for c in range(X.shape[2]):
        ch = X[:, :, c]
        parts.append(ch.mean(axis=1))
        parts.append(ch.std(axis=1))
        parts.append(ch.min(axis=1))
        parts.append(ch.max(axis=1))
        parts.append(np.median(ch, axis=1))
        parts.append(np.sqrt((ch ** 2).mean(axis=1)))
    return np.column_stack(parts)


def plot_feature_importance(clf, output_dir: str) -> None:
    stats = ["mean", "std", "min", "max", "median", "rms"]
    feature_names = [f"{stat}_{feat}" for feat in FEATURES for stat in stats]

    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(20), importances[top_idx], color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(20))
    ax.set_xticklabels([feature_names[i] for i in top_idx], rotation=45, ha="right", fontsize=9)
    ax.set_title("Random Forest — Top 20 Feature Importances")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


def main() -> None:
    print("=" * 50)
    print("  Phase 1: Random Forest")
    print("=" * 50)

    file_list = load_file_list(DATA_DIR)
    class_names = get_class_names(file_list)
    print(f"\nClasses: {class_names}")

    train_files, val_files, test_files = split_file_list(file_list)
    print(f"Files  — train: {len(train_files)}  val: {len(val_files)}  test: {len(test_files)}")

    print("\nExtracting windows...")
    X_train, y_train = extract_windows(train_files, class_names)
    X_val,   y_val   = extract_windows(val_files,   class_names)
    X_test,  y_test  = extract_windows(test_files,  class_names)
    print(f"Windows — train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")

    scaler = fit_scaler(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_val   = apply_scaler(X_val,   scaler)
    X_test  = apply_scaler(X_test,  scaler)

    F_train = extract_features(X_train)
    F_val   = extract_features(X_val)
    F_test  = extract_features(X_test)

    class_weights = compute_class_weights(y_train, len(class_names))

    print(f"\nTraining RandomForestClassifier (n_estimators=20, max_depth=10)...")
    clf = RandomForestClassifier(
        n_estimators=20,
        max_depth=10,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(F_train, y_train)

    y_val_pred  = clf.predict(F_val)
    y_test_pred = clf.predict(F_test)

    val_acc          = accuracy_score(y_val,  y_val_pred)
    test_acc         = accuracy_score(y_test, y_test_pred)
    test_f1_macro    = f1_score(y_test, y_test_pred, average="macro")
    test_f1_weighted = f1_score(y_test, y_test_pred, average="weighted")

    print(f"\nValidation accuracy : {val_acc:.4f}")
    print(f"Test accuracy       : {test_acc:.4f}")
    print(f"Test macro F1       : {test_f1_macro:.4f}")
    print(f"Test weighted F1    : {test_f1_weighted:.4f}\n")

    print("Saving outputs...")
    plot_confusion_matrix(y_test, y_test_pred, class_names, OUTPUT_DIR)
    plot_confusion_matrix(
        y_val, y_val_pred, class_names, OUTPUT_DIR,
        filename="confusion_matrix_val.png",
        title="Confusion Matrix (Validation Set)",
    )
    save_classification_report(y_test, y_test_pred, class_names, OUTPUT_DIR)
    plot_feature_importance(clf, OUTPUT_DIR)
    save_metrics_summary(
        {
            "model": "RandomForest",
            "val_accuracy": f"{val_acc:.4f}",
            "test_accuracy": f"{test_acc:.4f}",
            "test_macro_f1": f"{test_f1_macro:.4f}",
            "test_weighted_f1": f"{test_f1_weighted:.4f}",
            "n_estimators": 20,
            "max_depth": 10,
            "train_windows": len(X_train),
            "val_windows": len(X_val),
            "test_windows": len(X_test),
        },
        OUTPUT_DIR,
    )
    import joblib
    joblib.dump(clf,    os.path.join(OUTPUT_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
