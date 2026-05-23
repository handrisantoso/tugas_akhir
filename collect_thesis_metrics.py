"""
Gesture Glove — Thesis Metrics Collector (BAB 4-5)
Trains RF, MLP, CNN1D and collects all metrics needed for the thesis:
- Accuracy, F1-score, Precision, Recall per class
- Confusion matrices
- Training curves (loss/accuracy)
- Inference latency (PC-side)
- Model size / memory usage
- Classification reports
All figures saved to thesis_figures/ directory.
"""

import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GESTURE_NAMES, DEFAULT_CSV, WINDOW_SIZE, NUM_AXES, MODELS_DIR
from recorder import GestureRecorder
from models import (
    train_keras_model, train_random_forest, extract_features,
    fit_scaler, build_mlp, build_cnn1d, split_dataset,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_RF_TREES, DEFAULT_RF_DEPTH
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Plot style
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
    "font.family": "serif", "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3
})
COLORS = {"RF": "#2ecc71", "MLP": "#3498db", "CNN1D": "#e74c3c"}


def load_data():
    print("=" * 60)
    print("  GESTURE GLOVE — THESIS METRICS COLLECTOR")
    print("=" * 60)
    csv_path = DEFAULT_CSV
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "gesture_dataset.csv")
    print(f"\n[1] Loading: {csv_path}")
    X, y, _subject_ids = GestureRecorder.load_dataset(csv_path)
    print(f"    {len(X)} windows, {X.shape[1]} features/window")
    unique, counts = np.unique(y, return_counts=True)
    for idx, cnt in zip(unique, counts):
        print(f"    {GESTURE_NAMES[idx]:15s}: {cnt}")
    return X, y


def train_all_models(X, y):
    """Train RF, MLP, CNN1D with standard 70/20/10 split and return results."""
    results = {}

    # ── Random Forest ──
    print("\n[2] Training Random Forest...")
    t0 = time.time()
    rf_res = train_random_forest(X, y, n_estimators=DEFAULT_RF_TREES, max_depth=DEFAULT_RF_DEPTH)
    rf_res["train_time"] = time.time() - t0
    rf_res["model_type"] = "RF"
    results["RF"] = rf_res
    print(f"    RF Accuracy: {rf_res['accuracy']:.4f} ({rf_res['train_time']:.1f}s)")

    # ── MLP ──
    print("\n[3] Training MLP...")
    t0 = time.time()
    mlp_res = train_keras_model("mlp", X, y, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE)
    mlp_res["train_time"] = time.time() - t0
    mlp_res["model_type"] = "MLP"
    results["MLP"] = mlp_res
    print(f"    MLP Accuracy: {mlp_res['accuracy']:.4f} ({mlp_res['train_time']:.1f}s)")

    # ── CNN1D ──
    print("\n[4] Training CNN1D...")
    t0 = time.time()
    cnn_res = train_keras_model("cnn1d", X, y, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE)
    cnn_res["train_time"] = time.time() - t0
    cnn_res["model_type"] = "CNN1D"
    results["CNN1D"] = cnn_res
    print(f"    CNN1D Accuracy: {cnn_res['accuracy']:.4f} ({cnn_res['train_time']:.1f}s)")

    return results


def measure_latency(results, X, y, n_runs=100):
    """Measure PC-side inference latency for each model.
    
    Times the FULL pipeline per sample: feature extraction + scaling + prediction.
    Uses X_test from the actual 70/20/10 split (not arbitrary X[:n_runs]).
    """
    print("\n[5] Measuring Inference Latency...")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    # Reproduce the same 70/20/10 split used during training
    _, _, X_test_raw, _, _, _ = split_dataset(X, y)
    n_samples = min(n_runs, len(X_test_raw))
    print(f"    Using {n_samples} test samples from split (total test={len(X_test_raw)})")

    for name, res in results.items():
        model = res["model"]
        scaler = res["scaler"]
        latencies = []

        for i in range(n_samples):
            sample = X_test_raw[i:i+1]

            if name == "RF":
                t0 = time.perf_counter()
                feat = extract_features(sample)
                scaled = scaler.transform(feat)
                model.predict(scaled)
                latencies.append((time.perf_counter() - t0) * 1000)
            elif name == "MLP":
                t0 = time.perf_counter()
                feat = extract_features(sample)
                scaled = scaler.transform(feat)
                model.predict(scaled, verbose=0)
                latencies.append((time.perf_counter() - t0) * 1000)
            elif name == "CNN1D":
                t0 = time.perf_counter()
                flat = sample.flatten().reshape(1, -1)
                scaled = scaler.transform(flat).reshape(1, WINDOW_SIZE, NUM_AXES)
                model.predict(scaled, verbose=0)
                latencies.append((time.perf_counter() - t0) * 1000)

        res["latency_mean_ms"] = np.mean(latencies)
        res["latency_std_ms"] = np.std(latencies)
        res["latency_p95_ms"] = np.percentile(latencies, 95)
        res["latencies"] = latencies
        fps = 1000.0 / res["latency_mean_ms"] if res["latency_mean_ms"] > 0 else 0
        res["fps"] = fps
        print(f"    {name}: {res['latency_mean_ms']:.2f} +/- {res['latency_std_ms']:.2f} ms "
              f"(p95={res['latency_p95_ms']:.2f}ms, {fps:.1f} FPS)")


def get_model_sizes(results):
    """Get model file sizes and ESP32 flash usage."""
    print("\n[6] Collecting Model Sizes...")
    size_map = {
        "RF":   ("rf_model.joblib", "rf_model_data.h"),
        "MLP":  ("mlp_model.keras", "mlp_model.tflite"),
        "CNN1D":("cnn1d_model.keras", "cnn1d_model.tflite"),
    }
    for name, (pc_file, esp_file) in size_map.items():
        pc_path = os.path.join(MODELS_DIR, pc_file)
        esp_path = os.path.join(MODELS_DIR, esp_file)
        results[name]["pc_model_kb"] = os.path.getsize(pc_path) / 1024 if os.path.exists(pc_path) else 0
        results[name]["esp_model_kb"] = os.path.getsize(esp_path) / 1024 if os.path.exists(esp_path) else 0
        print(f"    {name}: PC={results[name]['pc_model_kb']:.1f}KB, ESP32={results[name]['esp_model_kb']:.1f}KB")

    # Tensor arena sizes from firmware
    results["RF"]["arena_kb"] = 0  # Pure C, no arena
    results["MLP"]["arena_kb"] = 100
    results["CNN1D"]["arena_kb"] = 220


def compute_extra_metrics(results):
    """Compute F1, Precision, Recall (macro/weighted)."""
    print("\n[7] Computing Detailed Metrics...")
    for name, res in results.items():
        yt, yp = res["y_test"], res["y_pred"]
        res["f1_macro"] = f1_score(yt, yp, average="macro")
        res["f1_weighted"] = f1_score(yt, yp, average="weighted")
        res["precision_macro"] = precision_score(yt, yp, average="macro")
        res["recall_macro"] = recall_score(yt, yp, average="macro")
        print(f"    {name}: F1(macro)={res['f1_macro']:.4f}, "
              f"Prec={res['precision_macro']:.4f}, Rec={res['recall_macro']:.4f}")


# ═══════════════════════════════════════════════════
#  FIGURE GENERATORS
# ═══════════════════════════════════════════════════

def plot_confusion_matrices(results):
    """Fig 4.x — Confusion matrices for all 3 models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, res) in zip(axes, results.items()):
        cm = res["confusion"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=GESTURE_NAMES, yticklabels=GESTURE_NAMES,
                    cbar=False, linewidths=0.5, square=True)
        ax.set_title(f"{name} (Acc={res['accuracy']:.1%})", fontweight="bold", fontsize=13)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.tick_params(labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
    fig.suptitle("Confusion Matrix — Perbandingan Model", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_training_curves(results):
    """Fig 4.x — Training loss/accuracy curves for MLP and CNN1D."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col, name in enumerate(["MLP", "CNN1D"]):
        hist = results[name].get("history", {})
        if not hist:
            continue
        # Loss
        ax = axes[0, col]
        ax.plot(hist.get("loss", []), label="Train Loss", color=COLORS[name], linewidth=2)
        ax.plot(hist.get("val_loss", []), label="Val Loss", color=COLORS[name], linewidth=2, linestyle="--")
        ax.set_title(f"{name} — Loss", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend()
        # Accuracy
        ax = axes[1, col]
        ax.plot(hist.get("accuracy", []), label="Train Acc", color=COLORS[name], linewidth=2)
        ax.plot(hist.get("val_accuracy", []), label="Val Acc", color=COLORS[name], linewidth=2, linestyle="--")
        ax.set_title(f"{name} — Accuracy", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.legend()
    fig.suptitle("Grafik Training — MLP vs CNN1D", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, "training_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_latency_chart(results):
    """Fig 4.x — Inference latency comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    names = list(results.keys())
    means = [results[n]["latency_mean_ms"] for n in names]
    stds = [results[n]["latency_std_ms"] for n in names]
    colors = [COLORS[n] for n in names]

    bars = ax1.bar(names, means, yerr=stds, color=colors, capsize=8, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.2f}ms", ha="center", fontweight="bold", fontsize=11)
    ax1.set_title("Inference Latency (ms)", fontweight="bold", fontsize=13)
    ax1.set_ylabel("Latency (ms)")

    # Box plot
    data = [results[n]["latencies"] for n in names]
    bp = ax2.boxplot(data, labels=names, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax2.set_title("Latency Distribution", fontweight="bold", fontsize=13)
    ax2.set_ylabel("Latency (ms)")

    fig.suptitle("Perbandingan Inference Latency", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "latency_chart.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_model_comparison(results):
    """Fig 4.x — Radar/bar chart comparing all metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    names = list(results.keys())
    colors = [COLORS[n] for n in names]

    # Accuracy + F1
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.3
    accs = [results[n]["accuracy"] for n in names]
    f1s = [results[n]["f1_macro"] for n in names]
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color=colors, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, f1s, w, label="F1 (macro)", color=colors, alpha=0.6, edgecolor="black", linewidth=0.5)
    for bar, val in zip(b1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.2%}", ha="center", fontsize=9)
    for bar, val in zip(b2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.2%}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12); ax.set_title("Accuracy & F1-Score", fontweight="bold")
    ax.legend()

    # Model Size
    ax = axes[1]
    esp_sizes = [results[n]["esp_model_kb"] for n in names]
    bars = ax.bar(names, esp_sizes, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, esp_sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}KB", ha="center", fontweight="bold")
    ax.set_title("ESP32 Flash Usage (KB)", fontweight="bold")
    ax.set_ylabel("Size (KB)")

    # FPS
    ax = axes[2]
    fps_vals = [results[n]["fps"] for n in names]
    bars = ax.bar(names, fps_vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, fps_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.0f}", ha="center", fontweight="bold")
    ax.set_title("Inference Rate (FPS)", fontweight="bold")
    ax.set_ylabel("FPS")

    fig.suptitle("Perbandingan Keseluruhan Model", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "model_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_per_class_f1(results):
    """Fig 4.x — Per-class F1 scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(GESTURE_NAMES))
    w = 0.25
    for i, (name, res) in enumerate(results.items()):
        report = res["report"]
        f1s = [report[g]["f1-score"] for g in GESTURE_NAMES]
        bars = ax.bar(x + i * w, f1s, w, label=name, color=COLORS[name], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", fontsize=8, rotation=0)
    ax.set_xticks(x + w); ax.set_xticklabels(GESTURE_NAMES, fontsize=10)
    ax.set_ylim(0, 1.15); ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score per Gesture per Model", fontweight="bold", fontsize=14)
    ax.legend(fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "per_class_f1.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def save_summary_json(results):
    """Save all numeric results to JSON for easy reference."""
    summary = {}
    for name, res in results.items():
        summary[name] = {
            "accuracy": round(res["accuracy"], 4),
            "f1_macro": round(res["f1_macro"], 4),
            "f1_weighted": round(res["f1_weighted"], 4),
            "precision_macro": round(res["precision_macro"], 4),
            "recall_macro": round(res["recall_macro"], 4),
            "val_accuracy": round(res.get("val_accuracy", 0), 4),
            "latency_mean_ms": round(res["latency_mean_ms"], 2),
            "latency_std_ms": round(res["latency_std_ms"], 2),
            "latency_p95_ms": round(res["latency_p95_ms"], 2),
            "fps": round(res["fps"], 1),
            "pc_model_kb": round(res["pc_model_kb"], 1),
            "esp_model_kb": round(res["esp_model_kb"], 1),
            "arena_kb": res["arena_kb"],
            "train_time_s": round(res["train_time"], 1),
            "train_samples": res["train_samples"],
            "test_samples": res["test_samples"],
            "per_class": {g: {
                "precision": round(res["report"][g]["precision"], 4),
                "recall": round(res["report"][g]["recall"], 4),
                "f1": round(res["report"][g]["f1-score"], 4),
                "support": int(res["report"][g]["support"]),
            } for g in GESTURE_NAMES},
            "confusion_matrix": res["confusion"].tolist(),
        }
    path = os.path.join(OUT_DIR, "thesis_metrics.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved: {path}")
    return summary


def print_latex_table(summary):
    """Print comparison table in a format ready for the thesis."""
    print("\n" + "=" * 70)
    print("  TABLE: Model Comparison (copy to thesis)")
    print("=" * 70)
    header = f"{'Metric':<25} {'RF':>12} {'MLP':>12} {'CNN1D':>12}"
    print(header)
    print("-" * 70)
    rows = [
        ("Accuracy", "accuracy", "{:.2%}"),
        ("F1-Score (macro)", "f1_macro", "{:.2%}"),
        ("Precision (macro)", "precision_macro", "{:.2%}"),
        ("Recall (macro)", "recall_macro", "{:.2%}"),
        ("Latency (ms)", "latency_mean_ms", "{:.2f}"),
        ("FPS", "fps", "{:.1f}"),
        ("ESP32 Flash (KB)", "esp_model_kb", "{:.0f}"),
        ("Arena (KB)", "arena_kb", "{:.0f}"),
        ("Training Time (s)", "train_time_s", "{:.1f}"),
    ]
    for label, key, fmt in rows:
        vals = [fmt.format(summary[m][key]) for m in ["RF", "MLP", "CNN1D"]]
        print(f"{label:<25} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")
    print("=" * 70)


def main():
    X, y = load_data()
    results = train_all_models(X, y)
    measure_latency(results, X, y)
    get_model_sizes(results)
    compute_extra_metrics(results)

    print("\n[8] Generating Figures...")
    plot_confusion_matrices(results)
    plot_training_curves(results)
    plot_latency_chart(results)
    plot_model_comparison(results)
    plot_per_class_f1(results)

    print("\n[9] Saving Summary...")
    summary = save_summary_json(results)
    print_latex_table(summary)

    print(f"\n[OK] All figures saved to: {OUT_DIR}")
    print("   Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"     • {f}")


if __name__ == "__main__":
    main()
