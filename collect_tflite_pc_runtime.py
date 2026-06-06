"""
Gesture Glove — TFLite PC Runtime (Thesis Comparison)

Perbandingan precision model di PC:
  Platform   Format      Precision
  ─────────────────────────────────────────
  PC         Keras .keras  float32  ← collect_thesis_metrics.py
  PC         TFLite .tflite int8    ← program ini
  ESP32      TFLite Micro  int8

Menjalankan int8 TFLite models (sama seperti di ESP32) di PC untuk
membandingkan akurasi dan latency dengan float32 Keras models.
"""

import os, sys, time, json, platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GESTURE_NAMES, DEFAULT_CSV, WINDOW_SIZE, NUM_AXES, MODELS_DIR
from recorder import GestureRecorder
from models import extract_features
from sklearn.model_selection import train_test_split

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
    "font.family": "serif", "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3
})

# ── TensorFlow / TFLite imports ─────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# ── Model mapping ───────────────────────────────────────────────
MODELS = {
    "MLP_float32":  {"keras": "mlp_model.keras",  "tflite": None},
    "MLP_int8":     {"keras": None,               "tflite": "mlp_model.tflite"},
    "CNN1D_float32":{"keras": "cnn1d_model.keras", "tflite": None},
    "CNN1D_int8":   {"keras": None,               "tflite": "cnn1d_model.tflite"},
    "RF":           {"keras": None,                "tflite": None},  # sklearn .joblib
}


def load_data():
    print("=" * 60)
    print("  GESTURE GLOVE — TFLITE PC RUNTIME (INT8 vs FLOAT32)")
    print("=" * 60)
    csv_path = DEFAULT_CSV
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "gesture_dataset.csv")
    print(f"\n[1] Loading: {csv_path}")
    X, y, _ = GestureRecorder.load_dataset(csv_path)
    print(f"    {len(X)} windows, {X.shape[1]} features/window")
    return X, y


def get_test_split(X, y, scaler):
    """Same 70/20/10 split as collect_thesis_metrics.py."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    val_size = 0.6667  # 20% of 30% = ~20% of total
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    # Fit scaler on training set
    X_train_feat = extract_features(X_train)
    scaler.fit(X_train_feat)

    _, _, X_test_raw, _, _, _ = split_dataset(X, y)
    return X_test_raw, y_test


def load_keras_model(name):
    path = os.path.join(MODELS_DIR, MODELS[name]["keras"])
    return tf.keras.models.load_model(path)


def load_tflite_interpreter(tflite_path):
    """Load int8 TFLite model and return interpreter + quantization params."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Quantization parameters (for int8 models)
    input_scale, input_zero  = input_details["quantization"]
    output_scale, output_zero = output_details["quantization"]

    return interpreter, input_scale, input_zero, output_scale, output_zero


def dequantize(output, scale, zero):
    """Convert int8 output back to float32 probabilities."""
    return (output.astype(np.float32) - zero) * scale


def predict_float32(model, X_sample, model_name):
    """Predict using float32 Keras model."""
    if "CNN1D" in model_name:
        X = X_sample.reshape(1, WINDOW_SIZE, NUM_AXES)
        prob = model.predict(X, verbose=0)[0]
    else:
        X = X_sample.reshape(1, -1)
        prob = model.predict(X, verbose=0)[0]
    return np.argmax(prob), prob


def predict_int8(interpreter, input_scale, input_zero, output_scale, output_zero,
                 X_sample, model_name):
    """Predict using int8 TFLite model."""
    if "CNN1D" in model_name:
        # CNN1D input: (1, 250, 6) → reshape from flat features
        X_float = X_sample.reshape(WINDOW_SIZE, NUM_AXES)
        X_int8 = np.round(X_float / input_scale + input_zero).astype(np.int8)
        X_int8 = np.expand_dims(X_int8, axis=0)
    else:
        # MLP/others: (1, 1500) flat features
        X_float = X_sample.reshape(1, -1)
        X_int8 = np.round(X_float / input_scale + input_zero).astype(np.int8)

    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], X_int8)
    interpreter.invoke()
    output_int8 = interpreter.get_tensor(
        interpreter.get_output_details()[0]["index"]
    )[0]
    prob = dequantize(output_int8, output_scale, output_zero)
    return np.argmax(prob), prob


def benchmark_inference(X_test, y_test, results, n_runs=100, n_warmup=10):
    """Measure inference latency for all model variants."""
    print("\n[2] Benchmarking Models...")

    print("-" * 55)
    print(f"  PC Spec:       {platform.processor()} | {platform.machine()}")
    print(f"  TF Version:    {tf.__version__}")
    print(f"  Samples:       {n_runs} | Warmup: {n_warmup}")
    print("-" * 55)

    n_samples = min(n_runs, len(X_test))

    for name, res in results.items():
        model  = res["model"]
        scaler = res["scaler"]

        # Pre-process for THIS model type
        if res["type"] == "rf":
            # RF: extract features (36), then scale
            feat_all   = extract_features(X_test[:n_samples])
            scaled_all = scaler.transform(feat_all)
        elif res["input"] == "windows":
            # CNN1D: scale flat windows (1500) using cnn1d_scaler, then reshape to (n, 250, 6)
            flat_all   = X_test[:n_samples].reshape(n_samples, -1)
            scaled_all = scaler.transform(flat_all).reshape(n_samples, WINDOW_SIZE, NUM_AXES)
        else:
            # MLP: scale features (36) using mlp_scaler
            feat_all   = extract_features(X_test[:n_samples])
            scaled_all = scaler.transform(feat_all)

        # Warmup
        print(f"  {name}: warming up ({n_warmup} calls)...")
        for i in range(n_warmup):
            if res["type"] == "keras":
                predict_float32(model, scaled_all[i], name)
            elif res["type"] == "tflite":
                predict_int8(
                    model, res["in_scale"], res["in_zero"],
                    res["out_scale"], res["out_zero"],
                    scaled_all[i], name
                )
            else:
                # RF: sklearn model, returns probabilities directly
                model.predict_proba(scaled_all[i:i+1])

        # Benchmark
        latencies = []
        correct = 0
        all_probs = []

        t0_total = time.perf_counter()
        for i in range(n_samples):
            t0 = time.perf_counter()

            if res["type"] == "keras":
                pred, prob = predict_float32(model, scaled_all[i], name)
            elif res["type"] == "tflite":
                pred, prob = predict_int8(
                    model, res["in_scale"], res["in_zero"],
                    res["out_scale"], res["out_zero"],
                    scaled_all[i], name
                )
            else:
                # RF: sklearn model
                prob = model.predict_proba(scaled_all[i:i+1])[0]
                pred = np.argmax(prob)

            latencies.append((time.perf_counter() - t0) * 1000)
            if pred == y_test[i]:
                correct += 1
            all_probs.append(prob)

        elapsed = time.perf_counter() - t0_total

        res["latency_mean_ms"] = np.mean(latencies)
        res["latency_std_ms"]  = np.std(latencies)
        res["latency_min_ms"]  = np.min(latencies)
        res["latency_max_ms"]  = np.max(latencies)
        res["latency_p95_ms"]  = np.percentile(latencies, 95)
        res["latencies"]       = latencies
        res["fps"]             = n_samples / elapsed if elapsed > 0 else 0
        res["accuracy"]        = correct / n_samples
        res["y_pred"]          = np.array([
            np.argmax(p) for p in all_probs
        ])
        res["y_true"]          = y_test[:n_samples]
        res["all_probs"]       = all_probs

        print(f"  {name}: acc={res['accuracy']:.4f} | "
              f"mean={res['latency_mean_ms']:.2f}ms "
              f"std={res['latency_std_ms']:.2f}ms "
              f"p95={res['latency_p95_ms']:.2f}ms fps={res['fps']:.1f}")


def compute_metrics(results):
    """Compute F1, precision, recall per model."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    for name, res in results.items():
        yt, yp = res["y_true"], res["y_pred"]
        res["f1_macro"]         = f1_score(yt, yp, average="macro")
        res["f1_weighted"]      = f1_score(yt, yp, average="weighted")
        res["precision_macro"]  = precision_score(yt, yp, average="macro", zero_division=0)
        res["recall_macro"]     = recall_score(yt, yp, average="macro", zero_division=0)
        print(f"  {name}: F1={res['f1_macro']:.4f}  Prec={res['precision_macro']:.4f}  Rec={res['recall_macro']:.4f}")


# ═══════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════

COLORS = {
    "RF": "#2ecc71",
    "MLP_float32": "#3498db", "MLP_int8": "#85c1e9",
    "CNN1D_float32": "#e74c3c", "CNN1D_int8": "#f1948a",
}
LINESTYLE = {"float32": "-", "int8": "--"}


def plot_latency_comparison(results):
    """Fig: Latency comparison — all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = list(results.keys())
    means = [results[n]["latency_mean_ms"] for n in names]
    stds  = [results[n]["latency_std_ms"]  for n in names]
    colors = [COLORS.get(n, "#95a5a6") for n in names]

    bars = ax1.bar(names, means, yerr=stds, color=colors, capsize=6,
                   edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.2f}ms", ha="center", fontweight="bold", fontsize=10)
    ax1.set_title("Inference Latency (ms)", fontweight="bold", fontsize=13)
    ax1.set_ylabel("Latency (ms)")
    ax1.tick_params(axis="x", rotation=20)

    data = [results[n]["latencies"] for n in names]
    bp = ax2.boxplot(data, tick_labels=names, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax2.set_title("Latency Distribution", fontweight="bold", fontsize=13)
    ax2.set_ylabel("Latency (ms)")
    ax2.tick_params(axis="x", rotation=20)

    fig.suptitle("Perbandingan Latency Semua Model (PC)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "tflite_latency_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_accuracy_comparison(results):
    """Fig: Accuracy/F1 comparison — all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    f1s   = [results[n]["f1_macro"] for n in names]
    colors = [COLORS.get(n, "#95a5a6") for n in names]

    x = np.arange(len(names))
    w = 0.35
    b1 = ax1.bar(x - w/2, accs, w, label="Accuracy", color=colors,
                 edgecolor="black", linewidth=0.5)
    b2 = ax1.bar(x + w/2, f1s,  w, label="F1 (macro)", color=colors,
                 alpha=0.6, edgecolor="black", linewidth=0.5)
    for bar, val in zip(b1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.2%}", ha="center", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylim(0, 1.15); ax1.set_title("Accuracy & F1-Score", fontweight="bold")
    ax1.legend(); ax1.tick_params(axis="x", rotation=20)

    means = [results[n]["latency_mean_ms"] for n in names]
    bars = ax2.bar(names, means, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.2f}ms", ha="center", fontsize=10)
    ax2.set_title("Latency (ms)", fontweight="bold")
    ax2.tick_params(axis="x", rotation=20)

    fig.suptitle("Perbandingan Semua Model: Keras float32 vs TFLite int8 vs RF", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "tflite_accuracy_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_confusion_all(results):
    """Fig: Confusion matrices for all models."""
    from sklearn.metrics import confusion_matrix as cm_func

    n_models = len(results)
    fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models + 2, 8))

    for col, (name, res) in enumerate(results.items()):
        cm = cm_func(res["y_true"], res["y_pred"])

        ax = axes[0, col] if n_models > 1 else axes[0]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=GESTURE_NAMES, yticklabels=GESTURE_NAMES,
                    cbar=False, linewidths=0.5, square=True)
        precision = res["accuracy"]
        ax.set_title(f"{name}\n(Acc={precision:.1%})", fontweight="bold", fontsize=11)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.tick_params(labelsize=7)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        # Per-class F1 bar
        ax2 = axes[1, col] if n_models > 1 else axes[1]
        from sklearn.metrics import classification_report
        report = classification_report(
            res["y_true"], res["y_pred"],
            labels=range(len(GESTURE_NAMES)),
            target_names=GESTURE_NAMES,
            output_dict=True, zero_division=0
        )
        f1s = [report[g]["f1-score"] for g in GESTURE_NAMES]
        x = np.arange(len(GESTURE_NAMES))
        ax2.bar(x, f1s, color=COLORS.get(name, "#95a5a6"), edgecolor="black", linewidth=0.5)
        ax2.set_xticks(x); ax2.set_xticklabels(GESTURE_NAMES, fontsize=7)
        ax2.set_ylim(0, 1.1); ax2.set_ylabel("F1-Score")
        ax2.tick_params(axis="x", rotation=35)

    if n_models == 1:
        axes[0].set_visible(False)
        axes[1].set_visible(False)
        fig.delaxes(axes[0])
        fig.delaxes(axes[1])

    fig.suptitle("Confusion Matrix & Per-Class F1: float32 vs int8", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT_DIR, "tflite_confusion_all.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_float_vs_int_bar(results):
    """Fig: Side-by-side float32 vs int8 for MLP/CNN1D, plus RF standalone."""
    architectures = ["MLP", "CNN1D"]
    all_model_keys = ["RF", "MLP_float32", "MLP_int8", "CNN1D_float32", "CNN1D_int8"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    metrics  = ["accuracy", "f1_macro", "latency_mean_ms"]
    titles   = ["Accuracy", "F1-Score (macro)", "Latency Mean (ms)"]
    formats  = ["{:.2%}", "{:.4f}", "{:.2f}ms"]

    for row, arch in enumerate(architectures):
        float_name = f"{arch}_float32"
        int_name   = f"{arch}_int8"

        for col, (metric, title, fmt) in enumerate(zip(metrics, titles, formats)):
            ax = axes[row, col]
            names  = [float_name, int_name]
            vals   = [results[float_name][metric], results[int_name][metric]]
            colors = [COLORS[float_name], COLORS[int_name]]

            bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, vals):
                y_offset = 0.02 if metric != "latency_mean_ms" else max(vals) * 0.05
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                        fmt.format(val), ha="center", fontweight="bold", fontsize=10)

            ax.set_title(f"{arch} — {title}", fontweight="bold")
            if metric == "latency_mean_ms":
                ax.set_ylabel("Latency (ms)")
            elif metric == "f1_macro":
                ax.set_ylim(0, 1.1)
            else:
                ax.set_ylim(0, 1.12)

    # RF standalone: accuracy + F1 + latency
    ax = axes[0, 2]
    names = ["RF"]
    vals  = [results["RF"]["accuracy"]]
    ax.bar(names, vals, color=[COLORS["RF"]], edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.12)
    ax.set_title(f"RF — Accuracy", fontweight="bold")
    for bar, val in zip(ax.patches, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                "{:.2%}".format(val), ha="center", fontweight="bold")

    ax = axes[1, 2]
    names = ["RF"]
    vals  = [results["RF"]["f1_macro"]]
    ax.bar(names, vals, color=[COLORS["RF"]], edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"RF — F1-Score (macro)", fontweight="bold")
    for bar, val in zip(ax.patches, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                "{:.4f}".format(val), ha="center", fontweight="bold")

    fig.suptitle("Perbandingan: Keras float32 vs TFLite int8 vs RF",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "tflite_float_vs_int_bar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def save_json(results):
    """Save summary JSON."""
    summary = {}
    for name, res in results.items():
        summary[name] = {
            "accuracy":        round(res["accuracy"], 4),
            "f1_macro":        round(res["f1_macro"], 4),
            "f1_weighted":     round(res["f1_weighted"], 4),
            "precision_macro": round(res["precision_macro"], 4),
            "recall_macro":    round(res["recall_macro"], 4),
            "latency_mean_ms": round(res["latency_mean_ms"], 2),
            "latency_std_ms":  round(res["latency_std_ms"], 2),
            "latency_min_ms":  round(res["latency_min_ms"], 2),
            "latency_max_ms":  round(res["latency_max_ms"], 2),
            "latency_p95_ms":  round(res["latency_p95_ms"], 2),
            "fps":             round(res["fps"], 1),
            "per_class": {}
        }
        # Per-class metrics
        from sklearn.metrics import classification_report
        report = classification_report(
            res["y_true"], res["y_pred"],
            labels=range(len(GESTURE_NAMES)),
            target_names=GESTURE_NAMES,
            output_dict=True, zero_division=0
        )
        for g in GESTURE_NAMES:
            summary[name]["per_class"][g] = {
                "precision": round(report[g]["precision"], 4),
                "recall":    round(report[g]["recall"], 4),
                "f1":        round(report[g]["f1-score"], 4),
                "support":   int(report[g]["support"]),
            }

    path = os.path.join(OUT_DIR, "tflite_pc_runtime.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved: {path}")
    return summary


def print_latex_table(summary):
    """Print LaTeX-ready table."""
    print("\n" + "=" * 80)
    print("  TABLE: TFLite PC Runtime - RF + Keras float32 + TFLite int8 (Thesis)")
    print("=" * 80)
    rows = [
        ("Accuracy",          "accuracy",        "{:.2%}"),
        ("F1-Score (macro)",  "f1_macro",         "{:.4f}"),
        ("Precision (macro)", "precision_macro",  "{:.4f}"),
        ("Recall (macro)",    "recall_macro",     "{:.4f}"),
        ("Latency mean (ms)", "latency_mean_ms",  "{:.3f}"),
        ("Latency std (ms)",  "latency_std_ms",   "{:.3f}"),
        ("Latency P95 (ms)",  "latency_p95_ms",   "{:.3f}"),
        ("FPS",               "fps",              "{:.1f}"),
    ]
    keys = ["RF", "MLP_float32", "MLP_int8", "CNN1D_float32", "CNN1D_int8"]
    print(f"{'Metric':<22} {'RF':>8} {'MLP_f32':>8} {'MLP_i8':>8} {'CNN1D_f32':>10} {'CNN1D_i8':>10}")
    print("-" * 80)
    for label, key, fmt in rows:
        vals = [fmt.format(summary[k].get(key, 0)) for k in keys]
        print(f"{label:<22} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>10} {vals[4]:>10}")

    print("  Accuracy drop (float32 -> int8):")
    for arch in ["MLP", "CNN1D"]:
        f32 = summary[f"{arch}_float32"]["accuracy"]
        i8  = summary[f"{arch}_int8"]["accuracy"]
        drop = (f32 - i8) * 100
        sign = "+" if drop >= 0 else ""
        print(f"    {arch}: {sign}{drop:.2f}% (f32={f32:.2%}, i8={i8:.2%})")

    print("=" * 80)


def main():
    import joblib
    X, y = load_data()

    # 70/20/10 split (same as collect_thesis_metrics.py)
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.6667, random_state=42, stratify=y_temp
    )
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    results = {}

    print("\n[2] Loading Models...")

    # ── Random Forest ───────────────────────────────────────
    print("  Loading RF (sklearn .joblib)...")
    rf_scaler = joblib.load(os.path.join(MODELS_DIR, "rf_scaler.joblib"))
    rf_model  = joblib.load(os.path.join(MODELS_DIR, "rf_model.joblib"))
    results["RF"] = {
        "model": rf_model, "scaler": rf_scaler,
        "type": "rf", "input": "features",
    }

    # MLP float32 + its pre-trained scaler (36 features)
    print("  Loading MLP float32 (Keras)...")
    scaler_mlp = joblib.load(os.path.join(MODELS_DIR, "mlp_scaler.joblib"))
    results["MLP_float32"] = {
        "model": load_keras_model("MLP_float32"),
        "scaler": scaler_mlp,
        "type": "keras",
        "input": "features",
    }

    # MLP int8 + same scaler
    print("  Loading MLP int8 (TFLite)...")
    interp_mlp, in_s_mlp, in_z_mlp, out_s_mlp, out_z_mlp = load_tflite_interpreter(
        os.path.join(MODELS_DIR, "mlp_model.tflite")
    )
    results["MLP_int8"] = {
        "model": interp_mlp,
        "scaler": scaler_mlp,
        "type": "tflite",
        "input": "features",
        "in_scale": in_s_mlp, "in_zero": in_z_mlp,
        "out_scale": out_s_mlp, "out_zero": out_z_mlp,
    }

    # CNN1D float32 + its pre-trained scaler (1500 features, flat windows)
    print("  Loading CNN1D float32 (Keras)...")
    scaler_cnn = joblib.load(os.path.join(MODELS_DIR, "cnn1d_scaler.joblib"))
    results["CNN1D_float32"] = {
        "model": load_keras_model("CNN1D_float32"),
        "scaler": scaler_cnn,
        "type": "keras",
        "input": "windows",
    }

    # CNN1D int8 + same scaler
    print("  Loading CNN1D int8 (TFLite)...")
    interp_cnn, in_s_cnn, in_z_cnn, out_s_cnn, out_z_cnn = load_tflite_interpreter(
        os.path.join(MODELS_DIR, "cnn1d_model.tflite")
    )
    results["CNN1D_int8"] = {
        "model": interp_cnn,
        "scaler": scaler_cnn,
        "type": "tflite",
        "input": "windows",
        "in_scale": in_s_cnn, "in_zero": in_z_cnn,
        "out_scale": out_s_cnn, "out_zero": out_z_cnn,
    }

    print(f"  Models loaded: {list(results.keys())}")

    # Benchmark
    benchmark_inference(X_test, y_test, results, n_runs=100, n_warmup=10)
    compute_metrics(results)

    # Figures
    print("\n[3] Generating Figures...")
    plot_latency_comparison(results)
    plot_accuracy_comparison(results)
    plot_confusion_all(results)
    plot_float_vs_int_bar(results)

    # Save results
    print("\n[4] Saving Results...")
    summary = save_json(results)
    print_latex_table(summary)

    print(f"\n[OK] TFLite PC runtime complete!")
    print(f"    Figures: {OUT_DIR}")
    print(f"    JSON:    {os.path.join(OUT_DIR, 'tflite_pc_runtime.json')}")


if __name__ == "__main__":
    main()