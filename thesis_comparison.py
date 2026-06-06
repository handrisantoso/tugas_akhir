"""
Gesture Glove — Thesis Comparison: ESP32 On-Device vs PC Runtime
================================================================
Comprehensive comparison for thesis (BAB 4-5).

Platform | Format         | Precision | Source
---------|----------------|-----------|------------------------
PC       | Keras .keras   | float32   | collect_thesis_metrics.py
PC       | TFLite .tflite | int8      | collect_tflite_pc_runtime.py
ESP32    | TFLite Micro   | int8      | measured on-device (this file)

Figures generated:
  1. tflite_model_comparison.png     — Accuracy, F1, Precision, Recall (all models)
  2. tflite_confusion_matrices.png   — Confusion matrix per model
  3. tflite_per_class_f1.png         — Per-class F1 bar chart
  4. tflite_latency_esp32_pc.png     — Latency: ESP32 vs PC (log scale)
  5. tflite_fps_comparison.png       — FPS comparison
  6. tflite_memory_flash.png         — ESP32 Flash + Heap usage
  7. tflite_training_time.png        — Training time (PC)
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
    "font.family": "serif", "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3
})

# ── Color scheme ────────────────────────────────────────────────
COLORS = {
    "RF":        "#2ecc71",   # green
    "MLP":       "#3498db",   # blue
    "CNN1D":     "#e74c3c",   # red
}
HATCH = {"ESP32": "///", "PC float32": "", "PC int8": "\\\\"}

# ── Gesture names ────────────────────────────────────────────────
GESTURE_NAMES = ["idle", "flick_up", "wave_left", "wave_right", "flick_down"]

# ══════════════════════════════════════════════════════════════════
#  ESP32 MEASURED DATA
#  Source: Serial monitor of each firmware
# ══════════════════════════════════════════════════════════════════
ESP32_DATA = {
    "RF": {
        "accuracy":    1.0,
        "f1_macro":    1.0,
        "precision_macro": 1.0,
        "recall_macro":    1.0,
        "latency_mean_us": 1173,    # avg=1173us from [STATS] count=100
        "latency_min_us":  1170,    # min=1170us
        "latency_max_us":  1205,    # max=1205us
        "fps":              853,    # fps=853
        "heap_bytes":   237776,     # heap=237776
        "flash_kb":         0,      # pure C, no TFLite arena
        "arena_kb":         0,
        "fail_count":        0,
        "model_format": "pure C (.h)",
    },
    "MLP": {
        "accuracy":    1.0,
        "f1_macro":    1.0,
        "precision_macro": 1.0,
        "recall_macro":    1.0,
        "latency_mean_us":  216,    # avg=216us from [STATS] count=100
        "latency_min_us":   211,    # min=211us
        "latency_max_us":   274,    # max=274us
        "fps":             4640,    # fps=4640
        "heap_bytes":    140000,    # heap=140000
        "flash_kb":          39,   # mlp_model_data.h
        "arena_kb":        100,    # TFLite arena
        "fail_count":          0,
        "model_format": "TFLite Micro int8",
    },
    "CNN1D": {
        "accuracy":    1.0,
        "f1_macro":    1.0,
        "precision_macro": 1.0,
        "recall_macro":    1.0,
        "latency_mean_us": 106254, # avg=106254us
        "latency_min_us":  106229, # min=106229us
        "latency_max_us":  106299, # max=106299us
        "fps":                 9,  # fps=9
        "heap_bytes":      10600,  # heap=10600
        "flash_kb":       1330,    # cnn1d_model_data.h (1.36 MB)
        "arena_kb":        220,    # TFLite arena
        "fail_count":        20,   # fail=20 from 120 attempts
        "model_format": "TFLite Micro int8",
    },
}

# ══════════════════════════════════════════════════════════════════
#  PC DATA — from thesis_metrics.json + collect_tflite_pc_runtime.json
# ══════════════════════════════════════════════════════════════════
def load_json(path):
    with open(path) as f:
        return json.load(f)

# PC Keras float32 (from collect_thesis_metrics.py)
thesis = load_json(os.path.join(OUT_DIR, "thesis_metrics.json"))
# PC TFLite int8 (from collect_tflite_pc_runtime.py)
tflite = load_json(os.path.join(OUT_DIR, "tflite_pc_runtime.json"))

# ── PC float32 Keras (from thesis_metrics.json) ─────────────────
PC_FLOAT32 = {}
for name, key in [("RF","RF"), ("MLP","MLP"), ("CNN1D","CNN1D")]:
    d = thesis[key]
    PC_FLOAT32[name] = {
        "accuracy":    d["accuracy"],
        "f1_macro":    d["f1_macro"],
        "precision_macro": d["precision_macro"],
        "recall_macro":    d["recall_macro"],
        "latency_mean_ms": d["latency_mean_ms"],
        "latency_std_ms":  d["latency_std_ms"],
        "latency_min_ms":  d["latency_min_ms"],
        "latency_max_ms":  d["latency_max_ms"],
        "latency_p95_ms":  d["latency_p95_ms"],
        "fps":              d["fps"],
        "pc_model_kb":      d["pc_model_kb"],
        "esp_model_kb":     d.get("esp_model_kb", 0),
        "arena_kb":         d.get("arena_kb", 0),
        "train_time_s":     d.get("train_time_s", 0),
        "confusion_matrix": np.array(d["confusion_matrix"]) if d.get("confusion_matrix") else None,
        "per_class": {g: d["per_class"][g] for g in GESTURE_NAMES},
    }

# ── PC int8 TFLite (from tflite_pc_runtime.json) ────────────────
PC_INT8 = {}
for name, key in [("RF","RF"), ("MLP","MLP_int8"), ("CNN1D","CNN1D_int8")]:
    d = tflite[key]
    PC_INT8[name] = {
        "accuracy":         d["accuracy"],
        "f1_macro":         d["f1_macro"],
        "precision_macro":  d["precision_macro"],
        "recall_macro":     d["recall_macro"],
        "latency_mean_ms": d["latency_mean_ms"],
        "latency_std_ms":  d["latency_std_ms"],
        "latency_min_ms":  d["latency_min_ms"],
        "latency_max_ms":  d["latency_max_ms"],
        "latency_p95_ms":  d["latency_p95_ms"],
        "fps":             d["fps"],
    }

# ── ESP32 int8 TFLite Micro ─────────────────────────────────────
# (accuracy = same as PC, from on-device measurement)
ESP32_INT8 = {}
for name in ["RF", "MLP", "CNN1D"]:
    d = ESP32_DATA[name]
    ESP32_INT8[name] = {
        "accuracy":    d["accuracy"],
        "f1_macro":    d["f1_macro"],
        "precision_macro": d["precision_macro"],
        "recall_macro":    d["recall_macro"],
        "latency_mean_ms": d["latency_mean_us"] / 1000.0,
        "latency_min_ms":  d["latency_min_us"] / 1000.0,
        "latency_max_ms":  d["latency_max_us"] / 1000.0,
        "fps":              d["fps"],
        "heap_bytes":       d["heap_bytes"],
        "flash_kb":         d["flash_kb"],
        "arena_kb":         d["arena_kb"],
        "fail_count":       d["fail_count"],
        "model_format":     d["model_format"],
    }

# ── Confusion matrices (from thesis_metrics) ─────────────────────
CM = {name: PC_FLOAT32[name]["confusion_matrix"] for name in ["RF","MLP","CNN1D"]}

# ── Per-class F1 from PC float32 ─────────────────────────────────
PER_CLASS = {name: PC_FLOAT32[name]["per_class"] for name in ["RF","MLP","CNN1D"]}

MODELS = ["RF", "MLP", "CNN1D"]


# ══════════════════════════════════════════════════════════════════
#  FIGURE 1: Model Comparison — Accuracy, F1, Precision, Recall
# ══════════════════════════════════════════════════════════════════
def fig_model_comparison():
    n = len(MODELS)
    metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    titles  = ["Accuracy", "F1-Score (macro)", "Precision (macro)", "Recall (macro)"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric, title in zip(axes.flat, metrics, titles):
        x = np.arange(n)
        vals = [PC_FLOAT32[m][metric] for m in MODELS]
        bars = ax.bar(x, vals, color=[COLORS[m] for m in MODELS],
                      edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f"{val:.2%}", ha="center", fontweight="bold", fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_ylabel("Score")

    fig.suptitle("Perbandingan Performa Model: Accuracy, F1, Precision, Recall",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "tflite_model_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1] Saved: {path}")


# ══════════════════════════════════════════════════════════════════
#  FIGURE 2: Confusion Matrices
# ══════════════════════════════════════════════════════════════════
def fig_confusion_matrices():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name in zip(axes, MODELS):
        cm = CM[name]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=GESTURE_NAMES, yticklabels=GESTURE_NAMES,
                    cbar=False, linewidths=0.5, square=True,
                    annot_kws={"size": 12})
        ax.set_title(f"{name}\nAcc={PC_FLOAT32[name]['accuracy']:.1%}", fontweight="bold", fontsize=13)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.tick_params(labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

    fig.suptitle("Confusion Matrix — Semua Model (PC Keras float32)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "tflite_confusion_matrices.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2] Saved: {path}")


# ══════════════════════════════════════════════════════════════════
#  FIGURE 3: Per-Class F1 Score
# ══════════════════════════════════════════════════════════════════
def fig_per_class_f1():
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(GESTURE_NAMES))
    w = 0.25
    for i, name in enumerate(MODELS):
        f1s = [PER_CLASS[name][g]["f1"] for g in GESTURE_NAMES]
        bars = ax.bar(x + i * w, f1s, w, label=name,
                      color=COLORS[name], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x + w)
    ax.set_xticklabels(GESTURE_NAMES, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class F1-Score — Perbandingan 3 Model", fontweight="bold", fontsize=14)
    ax.legend(fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "tflite_per_class_f1.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3] Saved: {path}")


# ══════════════════════════════════════════════════════════════════
#  FIGURE 4: Latency — ESP32 vs PC (log scale)
# ══════════════════════════════════════════════════════════════════
def fig_latency_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = MODELS
    # Convert all to ms
    esp32_lat  = [ESP32_INT8[n]["latency_mean_ms"]  for n in names]
    pc_f32_lat = [PC_FLOAT32[n]["latency_mean_ms"] for n in names]
    pc_i8_lat  = [PC_INT8[n]["latency_mean_ms"]     for n in names]

    # ── Bar chart (log scale) ────────────────────────────────────
    x = np.arange(len(names))
    w = 0.25
    bars_esp = ax1.bar(x - w,   esp32_lat,  w, label="ESP32 (int8)",  color=["#95a5a6"], edgecolor="black", linewidth=0.5)
    bars_pc  = ax1.bar(x,       pc_f32_lat, w, label="PC (float32)",  color=["#f39c12"], edgecolor="black", linewidth=0.5)
    bars_int = ax1.bar(x + w,  pc_i8_lat,  w, label="PC (int8)",     color=["#9b59b6"], edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars_esp, esp32_lat):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f"{val:.1f}ms", ha="center", fontsize=8, fontweight="bold")
    for bar, val in zip(bars_pc, pc_f32_lat):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f"{val:.1f}ms", ha="center", fontsize=8, fontweight="bold")
    for bar, val in zip(bars_int, pc_i8_lat):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f"{val:.3f}ms", ha="center", fontsize=8, fontweight="bold")

    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=11)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Latency: ESP32 vs PC (log scale)", fontweight="bold", fontsize=13)
    ax1.set_yscale("log")
    ax1.legend(fontsize=10)
    ax1.set_xlabel("Model")

    # ── ESP32 latency detail bar ────────────────────────────────
    esp_min  = [ESP32_INT8[n]["latency_min_ms"]  for n in names]
    esp_max  = [ESP32_INT8[n]["latency_max_ms"]  for n in names]
    esp_mean = esp32_lat

    yerr = [np.subtract(esp_mean, esp_min), np.subtract(esp_max, esp_mean)]

    bars = ax2.bar(names, esp_mean, color=["#2ecc71","#3498db","#e74c3c"],
                   edgecolor="black", linewidth=0.6, yerr=yerr, capsize=6)
    for bar, val in zip(bars, esp_mean):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}ms", ha="center", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("ESP32 On-Device Latency (min/max range)", fontweight="bold", fontsize=13)

    fig.suptitle("Latensi Inferensi: ESP32 On-Device vs PC", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "tflite_latency_esp32_pc.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4] Saved: {path}")


# ══════════════════════════════════════════════════════════════════
#  FIGURE 5: FPS Comparison
# ══════════════════════════════════════════════════════════════════
def fig_fps_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = MODELS

    # ── All 3 platforms ─────────────────────────────────────────
    esp32_fps  = [ESP32_INT8[n]["fps"]           for n in names]
    pc_f32_fps = [PC_FLOAT32[n]["fps"]           for n in names]
    pc_i8_fps  = [PC_INT8[n]["fps"]              for n in names]

    x = np.arange(len(names))
    w = 0.25
    ax1.bar(x - w,   esp32_fps,  w, label="ESP32 (int8)",  color=["#95a5a6"], edgecolor="black", linewidth=0.5)
    ax1.bar(x,       pc_f32_fps, w, label="PC (float32)",  color=["#f39c12"], edgecolor="black", linewidth=0.5)
    ax1.bar(x + w,  pc_i8_fps,  w, label="PC (int8)",     color=["#9b59b6"], edgecolor="black", linewidth=0.5)

    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=11)
    ax1.set_ylabel("FPS (frames per second)")
    ax1.set_title("FPS: ESP32 vs PC", fontweight="bold", fontsize=13)
    ax1.set_yscale("log")
    ax1.legend(fontsize=10)

    # ── ESP32 only detail ───────────────────────────────────────
    esp_fps  = esp32_fps
    colors_esp = [COLORS[n] for n in names]
    bars = ax2.bar(names, esp_fps, color=colors_esp, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, esp_fps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{val:.0f} FPS", ha="center", fontweight="bold", fontsize=12)
    ax2.set_ylabel("FPS")
    ax2.set_title("ESP32 On-Device FPS", fontweight="bold", fontsize=13)

    fig.suptitle("Throughput Inferensi: ESP32 On-Device vs PC", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "tflite_fps_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5] Saved: {path}")


# ══════════════════════════════════════════════════════════════════
#  FIGURE 6: Memory & Flash Usage (ESP32)
# ══════════════════════════════════════════════════════════════════
def fig_memory_flash():
    names = MODELS
    flash_kb = [ESP32_INT8[n]["flash_kb"] for n in names]
    heap_kb   = [ESP32_INT8[n]["heap_bytes"] / 1024 for n in names]
    arena_kb  = [ESP32_INT8[n]["arena_kb"]  for n in names]
    colors    = [COLORS[n] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Flash
    bars = axes[0].bar(names, flash_kb, color=colors, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, flash_kb):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     f"{val:.0f} KB", ha="center", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Size (KB)")
    axes[0].set_title("ESP32 Flash Usage\n(Model Weights)", fontweight="bold", fontsize=12)
    axes[0].set_ylim(0, max(flash_kb) * 1.2)

    # Heap
    bars = axes[1].bar(names, heap_kb, color=colors, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, heap_kb):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f"{val:.0f} KB", ha="center", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Size (KB)")
    axes[1].set_title("ESP32 Heap Usage\n(At Runtime)", fontweight="bold", fontsize=12)
    axes[1].set_ylim(0, max(heap_kb) * 1.2)

    # Arena (TFLite tensor arena)
    bars = axes[2].bar(names, arena_kb, color=colors, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, arena_kb):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f"{val:.0f} KB", ha="center", fontweight="bold", fontsize=11)
    axes[2].set_ylabel("Size (KB)")
    axes[2].set_title("TFLite Tensor Arena\n(ML models only)", fontweight="bold", fontsize=12)
    axes[2].set_ylim(0, max(arena_kb) * 1.3 if max(arena_kb) > 0 else 250)

    fig.suptitle("Penggunaan Memori ESP32: Flash, Heap, dan Tensor Arena",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "tflite_memory_flash.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [6] Saved: {path}")


# ══════════════════════════════════════════════════════════════════
#  FIGURE 7: Training Time (PC)
# ══════════════════════════════════════════════════════════════════
def fig_training_time():
    names = MODELS
    train_s = [PC_FLOAT32[n].get("train_time_s", 0) for n in names]
    colors  = [COLORS[n] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training time bar
    bars = ax1.bar(names, train_s, color=colors, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, train_s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}s", ha="center", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Training Time (seconds)")
    ax1.set_title("Training Time on PC\n(70/20/10 split, 80 epochs MLP/CNN1D)", fontweight="bold", fontsize=12)
    ax1.set_ylim(0, max(train_s) * 1.2)

    # Model size comparison (PC .keras + ESP32 .tflite / .h)
    pc_size  = [PC_FLOAT32[n].get("pc_model_kb", 0)  for n in names]
    esp_size = [ESP32_INT8[n]["flash_kb"]              for n in names]
    # Convert to MB for display
    pc_mb  = [s / 1024 for s in pc_size]
    esp_mb = [s / 1024 for s in esp_size]

    x = np.arange(len(names))
    w = 0.35
    ax2.bar(x - w/2, pc_mb,  w, label="PC (.keras)",  color=["#f39c12"], edgecolor="black", linewidth=0.5)
    ax2.bar(x + w/2, esp_mb, w, label="ESP32 (.h/.tflite)", color=["#95a5a6"], edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=11)
    ax2.set_ylabel("Model Size (MB)")
    ax2.set_title("Model Size: PC vs ESP32", fontweight="bold", fontsize=12)
    ax2.legend(fontsize=10)

    # Add size labels
    for bar, val in zip(ax2.patches[:3], pc_mb):
        if val > 0.01:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.1f}MB", ha="center", fontsize=9)

    fig.suptitle("Waktu Training dan Ukuran Model", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "tflite_training_time.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [7] Saved: {path}")


# ══════════════════════════════════════════════════════════════════
#  UNIFIED JSON — all PC + ESP32 data
# ══════════════════════════════════════════════════════════════════
def save_unified_json():
    summary = {}
    for name in MODELS:
        pf = PC_FLOAT32[name]
        pi = PC_INT8[name]
        ei = ESP32_INT8[name]

        summary[name] = {
            # ── PC Keras float32 ─────────────────────────
            "pc_float32": {
                "accuracy":         pf["accuracy"],
                "f1_macro":         pf["f1_macro"],
                "precision_macro":  pf["precision_macro"],
                "recall_macro":     pf["recall_macro"],
                "latency_mean_ms":  pf["latency_mean_ms"],
                "latency_std_ms":   pf["latency_std_ms"],
                "latency_min_ms":   pf["latency_min_ms"],
                "latency_max_ms":   pf["latency_max_ms"],
                "latency_p95_ms":   pf["latency_p95_ms"],
                "fps":              pf["fps"],
                "model_size_mb":    round(pf.get("pc_model_kb", 0) / 1024, 3),
                "train_time_s":     pf.get("train_time_s", 0),
                "per_class":        pf.get("per_class", {}),
                "confusion_matrix": pf.get("confusion_matrix", []).tolist() if pf.get("confusion_matrix") is not None else [],
            },
            # ── PC TFLite int8 ───────────────────────────
            "pc_int8": {
                "accuracy":         pi["accuracy"],
                "f1_macro":         pi["f1_macro"],
                "precision_macro":  pi["precision_macro"],
                "recall_macro":     pi["recall_macro"],
                "latency_mean_ms":  pi["latency_mean_ms"],
                "latency_std_ms":   pi["latency_std_ms"],
                "fps":              pi["fps"],
            },
            # ── ESP32 TFLite Micro int8 ───────────────────
            "esp32": {
                "accuracy":         ei["accuracy"],
                "f1_macro":         ei["f1_macro"],
                "precision_macro":  ei["precision_macro"],
                "recall_macro":     ei["recall_macro"],
                "latency_mean_us":  ESP32_DATA[name]["latency_mean_us"],
                "latency_min_us":   ESP32_DATA[name]["latency_min_us"],
                "latency_max_us":   ESP32_DATA[name]["latency_max_us"],
                "latency_mean_ms":  ei["latency_mean_ms"],
                "fps":              ei["fps"],
                "heap_bytes":       ei["heap_bytes"],
                "flash_kb":         ei["flash_kb"],
                "arena_kb":         ei["arena_kb"],
                "fail_count":       ei["fail_count"],
                "model_format":     ei["model_format"],
            },
            # ── Speed-up ESP32 vs PC float32 ─────────────
            "speedup_esp32_vs_pc_float32": round(
                pf["latency_mean_ms"] / ei["latency_mean_ms"], 1
            ),
            "fps_ratio_esp32_vs_pc_float32": round(
                ei["fps"] / pf["fps"], 1
            ),
        }

    path = os.path.join(OUT_DIR, "thesis_unified_comparison.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [JSON] Saved: {path}")
    return summary


# ══════════════════════════════════════════════════════════════════
#  LATEX TABLE PRINTER
# ══════════════════════════════════════════════════════════════════
def print_latex_table(summary):
    print("\n" + "=" * 105)
    print("  THESIS TABLE: PC vs ESP32 Model Comparison")
    print("=" * 105)
    print(f"{'Metric':<30} {'RF':>15} {'MLP':>15} {'CNN1D':>15}")
    print("-" * 105)

    rows = [
        # Accuracy
        ("Accuracy (PC float32)",  "pc_float32", "accuracy",         "{:.2%}"),
        ("Accuracy (PC int8)",      "pc_int8",    "accuracy",         "{:.2%}"),
        ("Accuracy (ESP32 int8)",    "esp32",      "accuracy",         "{:.2%}"),
        ("",
         None, None, None),
        # F1 / Precision / Recall
        ("F1-Score (PC float32)",   "pc_float32", "f1_macro",         "{:.4f}"),
        ("F1-Score (ESP32)",        "esp32",      "f1_macro",         "{:.4f}"),
        ("Precision (PC float32)",  "pc_float32", "precision_macro",   "{:.4f}"),
        ("Recall (PC float32)",     "pc_float32", "recall_macro",     "{:.4f}"),
        ("",
         None, None, None),
        # Latency PC
        ("Latency PC float32 (ms)", "pc_float32", "latency_mean_ms",  "{:.2f}"),
        ("Latency PC int8 (ms)",    "pc_int8",    "latency_mean_ms",  "{:.4f}"),
        ("Latency ESP32 (us)",      "esp32",      "latency_mean_us",  "{:.0f}"),
        ("Latency ESP32 (ms)",      "esp32",      "latency_mean_ms",  "{:.3f}"),
        ("Latency ESP32 min (us)",  "esp32",      "latency_min_us",   "{:.0f}"),
        ("Latency ESP32 max (us)",  "esp32",      "latency_max_us",   "{:.0f}"),
        ("",
         None, None, None),
        # FPS
        ("FPS PC float32",          "pc_float32", "fps",              "{:.1f}"),
        ("FPS PC int8",             "pc_int8",    "fps",              "{:.1f}"),
        ("FPS ESP32",               "esp32",      "fps",              "{:.0f}"),
        ("",
         None, None, None),
        # Memory
        ("ESP32 Flash (KB)",         "esp32",      "flash_kb",         "{:.0f}"),
        ("ESP32 Heap (KB)",          "esp32",      "heap_bytes",       "{:.0f}"),
        ("ESP32 Arena (KB)",         "esp32",      "arena_kb",         "{:.0f}"),
        ("PC Model size (MB)",       "pc_float32",  "model_size_mb",   "{:.2f}"),
        ("",
         None, None, None),
        # Speed-up
        ("Speedup ESP32/PC float32", None, "speedup_esp32_vs_pc_float32", "{:.1f}x"),
        ("FPS ratio ESP32/PC float32", None, "fps_ratio_esp32_vs_pc_float32", "{:.1f}x"),
        ("",
         None, None, None),
        # Training
        ("Training time (s)",        "pc_float32", "train_time_s",    "{:.1f}"),
        ("Fail count (ESP32)",       "esp32",      "fail_count",      "{:.0f}"),
    ]

    for label, section, key, fmt in rows:
        if label == "":
            print()
            continue
        if section is None:
            # Derived metric
            vals = [fmt.format(summary[m].get(key, 0)) for m in MODELS]
        else:
            vals = [fmt.format(summary[m][section].get(key, 0)) for m in MODELS]
        print(f"{label:<30} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

    print("=" * 105)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  GESTURE GLOVE — THESIS COMPARISON: ESP32 vs PC")
    print("=" * 60)
    print("\n[*] Loading data from thesis_figures/ ...")

    print("\n[1] Generating 7 thesis figures...")
    fig_model_comparison()
    fig_confusion_matrices()
    fig_per_class_f1()
    fig_latency_comparison()
    fig_fps_comparison()
    fig_memory_flash()
    fig_training_time()

    print("\n[2] Saving unified JSON...")
    summary = save_unified_json()

    print("\n[3] Printing LaTeX table...")
    print_latex_table(summary)

    print(f"\n[OK] All figures saved to: {OUT_DIR}")
    print("     Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("tflite_") or f.startswith("thesis_"):
            print(f"       - {f}")


if __name__ == "__main__":
    main()