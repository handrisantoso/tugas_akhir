"""
Flowchart Generator for Figure 3.1.1 - Metode Penelitian
Gesture Glove ML Pipeline

Creates three separate flowchart figures:
  Phase 1: Pengumpulan Data (portrait)
  Phase 2: Pelatihan Model  (landscape)
  Phase 3: Deployment & Inferensi Real-Time (landscape)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ─── Colour palette ───────────────────────────────────────────────────────────
C_DARK       = "#0d1117"
C_LIGHT      = "#f0f6fc"
C_PHASE1_BG  = "#0d2137"
C_PHASE2_BG  = "#1a0d37"
C_PHASE3_BG  = "#0d3717"
C_ACCENT1    = "#58a6ff"
C_ACCENT2    = "#bc8cff"
C_ACCENT3    = "#56d364"
C_MIDPOINT   = "#ff7b72"
C_DECISION   = "#ffa657"
C_GRAY_BOX   = "#8b949e"
C_WARN       = "#d29922"
C_ARROW      = "#8b949e"

# ─── Shared drawing helpers ───────────────────────────────────────────────────

def rounded_box(ax, x, y, w, h, text, *, fill="#161b22", edge=C_ACCENT1,
                lw=1.8, fs=9, text_color=C_LIGHT, radius=0.18, bold=False):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0.08,rounding_size={radius}",
                         linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=text_color, zorder=4,
            fontfamily="monospace", fontweight=weight,
            multialignment="center")


def diamond(ax, cx, cy, w, h, text, *, fill="#21262d", edge=C_DECISION,
            lw=1.8, fs=8.5, text_color=C_LIGHT):
    dx, dy = w/2, h/2
    xs = [cx, cx+dx, cx, cx-dx, cx]
    ys = [cy+dy, cy, cy-dy, cy, cy+dy]
    poly = plt.Polygon(list(zip(xs, ys)), closed=True,
                       linewidth=lw, edgecolor=edge,
                       facecolor=fill, zorder=3)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=text_color, zorder=4,
            fontfamily="monospace", fontweight="bold",
            multialignment="center")


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.4, head_w=0.18, head_l=0.16):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=f"-|>", color=color,
                    linewidth=lw,
                    mutation_scale=15,
                    shrinkA=0, shrinkB=0,
                ), zorder=2)


def label_arrow(ax, x, y, text, *, color=C_GRAY_BOX, fs=7.5):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=color, fontfamily="monospace", zorder=5)


def footer(ax, text, *, y=-0.06):
    ax.text(0.5, y, text, ha="center", va="top", fontsize=7.5,
            color=C_GRAY_BOX, fontfamily="monospace", transform=ax.transAxes)


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — PENGUMPULAN DATA (portrait: 6.5 × 9)
# ════════════════════════════════════════════════════════════════════════════
def draw_phase1(outpath):
    fig, ax = plt.subplots(figsize=(6.5, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.5)
    ax.axis("off")
    fig.patch.set_facecolor(C_PHASE1_BG)
    ax.set_facecolor(C_PHASE1_BG)

    # ── title ─────────────────────────────────────────────────────────────────
    ax.text(5, 13.05, "GAMBAR 3.1  FASE 1 — PENGUMPULAN DATA",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C_ACCENT1, fontfamily="monospace")
    ax.text(5, 12.60, "Gesture Glove | XIAO ESP32-S3 + MPU6050",
            ha="center", va="center", fontsize=8.5, color=C_GRAY_BOX,
            fontfamily="monospace")

    # ── box positions (y) ──────────────────────────────────────────────────────
    Y = {
        "sensor": 11.3,
        "imu":    10.1,
        "window":  8.7,
        "d1":      7.4,
        "csv":     6.1,
        "check":   4.6,
        "label":   3.2,
        "repeat":  2.1,
        "done":    0.7,
    }

    # ── boxes ──────────────────────────────────────────────────────────────────
    rounded_box(ax, 5, Y["sensor"], 5.0, 0.60,
                "🎮 Subjek menggunakan Gesture Glove",
                fill="#112240", edge=C_ACCENT1, fs=9.5, bold=True)

    rounded_box(ax, 5, Y["imu"], 5.2, 0.65,
                "Sensor MPU6050 → akuisisi 6-axis data\n(Accel X/Y/Z + Gyro X/Y/Z @ 100 Hz)",
                fill="#161b22", edge=C_ACCENT1, fs=8.5)

    arrow(ax, 5, Y["sensor"]-0.3, 5, Y["imu"]+0.33)
    label_arrow(ax, 5.7, 10.7, "100 Hz")

    rounded_box(ax, 5, Y["window"], 5.4, 0.65,
                "Sliding Window 2.5 detik\n(250 sampel per window)",
                fill="#161b22", edge=C_ACCENT1, fs=8.5)

    arrow(ax, 5, Y["imu"]-0.33, 5, Y["window"]+0.33)

    # Decision: enough samples?
    diamond(ax, 5, Y["d1"], 2.8, 0.90,
            "> 200\nsampel?", fs=8)

    arrow(ax, 5, Y["window"]-0.33, 5, Y["d1"]+0.45)
    label_arrow(ax, 6.3, 8.0, "Ya")

    rounded_box(ax, 5, Y["csv"], 5.2, 0.75,
                "Simpan ke CSV\n(dataset/gesture_name.csv)",
                fill="#161b22", edge=C_ACCENT2, fs=8.5)

    arrow(ax, 5, Y["d1"]-0.45, 5, Y["csv"]+0.38)

    rounded_box(ax, 5, Y["check"], 5.4, 0.70,
                "Beri label gesture\n(flick_up, wave_left, …)",
                fill="#161b22", edge=C_ACCENT2, fs=8.5)

    arrow(ax, 5, Y["csv"]-0.38, 5, Y["check"]+0.35)

    # Decision: another gesture?
    diamond(ax, 5, Y["label"], 2.8, 0.90,
            "Gestur\nlain?", fs=8.5)

    arrow(ax, 5, Y["check"]-0.35, 5, Y["label"]+0.45)
    label_arrow(ax, 6.3, 3.8, "Ya")

    rounded_box(ax, 5, Y["repeat"], 4.6, 0.60,
                "Ulangi untuk setiap gesture",
                fill="#0d2137", edge=C_DECISION, fs=8.5,
                text_color=C_DECISION)

    arrow(ax, 3.5, Y["label"]-0.45, 2.0, Y["repeat"])
    label_arrow(ax, 2.0, 4.2, "Tidak")

    rounded_box(ax, 5, Y["done"], 5.0, 0.60,
                "✅ Dataset lengkap\n(600 sampel × 5 gesture = 3000 baris)",
                fill="#112240", edge=C_ACCENT1, fs=8.5, bold=True)

    arrow(ax, 7.5, Y["label"]-0.45, 7.5, Y["done"]+0.3)
    label_arrow(ax, 7.9, 4.5, "Ya,\nsemua done")

    # ── horizontal divider ─────────────────────────────────────────────────────
    ax.axhline(0.02, color=C_ACCENT1, linewidth=0.8, alpha=0.5)

    footer(ax, "Fase 1 dari 3  |  Pengumpulan Data  |  Output: dataset CSV",
           y=0.0)

    # ── phase colour strip (left edge) ─────────────────────────────────────────
    ax.add_patch(plt.Rectangle((0, 0), 0.12, 13.5,
                               facecolor=C_ACCENT1, alpha=0.35, zorder=1))

    fig.savefig(outpath, dpi=180, bbox_inches="tight",
                facecolor=C_PHASE1_BG)
    plt.close()
    print("  -> Saved " + outpath)


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — PELATIHAN MODEL (landscape: 13 × 7)
# ════════════════════════════════════════════════════════════════════════════
def draw_phase2(outpath):
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 9.5)
    ax.axis("off")
    fig.patch.set_facecolor(C_PHASE2_BG)
    ax.set_facecolor(C_PHASE2_BG)

    # ── title ─────────────────────────────────────────────────────────────────
    ax.text(9.5, 9.1, "GAMBAR 3.2  FASE 2 — PELATIHAN MODEL",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C_ACCENT2, fontfamily="monospace")
    ax.text(9.5, 8.70, "3 Model: Random Forest · MLP · CNN1D  |  Python (sklearn / TensorFlow)",
            ha="center", va="center", fontsize=8.5, color=C_GRAY_BOX,
            fontfamily="monospace")

    # ── vertical layout helpers ────────────────────────────────────────────────
    #  (x_center, y_top of node body, half-height)
    nodes = {
        "csv":          (2.2,  7.5),
        "split":        (2.2,  6.4),
        "train_label":  (2.2,  5.5),
        "scaler":       (2.2,  4.5),

        # branch point after scaler
        "rf_path":     (6.5,  7.5),
        "mlp_path":    (9.5,  7.5),
        "cnn_path":   (12.5,  7.5),

        # features
        "feat_rf":     (6.5,  6.3),
        "feat_mlp":    (9.5,  6.3),
        "raw_cnn":    (12.5,  6.3),

        # model train
        "train_rf":    (6.5,  5.1),
        "train_mlp":   (9.5,  5.1),
        "train_cnn":  (12.5,  5.1),

        # eval
        "eval_rf":     (6.5,  3.8),
        "eval_mlp":    (9.5,  3.8),
        "eval_cnn":   (12.5,  3.8),

        # header export
        "exp_rf":      (6.5,  2.4),
        "exp_mlp":     (9.5,  2.4),
        "exp_cnn":    (12.5,  2.4),
    }

    # ── SOURCE block ───────────────────────────────────────────────────────────
    rounded_box(ax, nodes["csv"][0], nodes["csv"][1], 3.0, 0.65,
                "📂 Dataset CSV\n(3000 baris × 6 fitur)",
                fill="#1a1033", edge=C_ACCENT2, fs=8.5)

    rounded_box(ax, nodes["split"][0], nodes["split"][1], 3.0, 0.65,
                "Train / Test Split\n(80% / 20%, stratified)",
                fill="#1a1033", edge=C_ACCENT2, fs=8.5)
    arrow(ax, nodes["csv"][0], nodes["csv"][1]-0.33,
               nodes["csv"][0], nodes["split"][1]+0.33)

    rounded_box(ax, nodes["train_label"][0], nodes["train_label"][1], 3.0, 0.65,
                "🎯 Label Encoding\n(idle, flick_up, wave_left …)",
                fill="#1a1033", edge=C_ACCENT2, fs=8.5)
    arrow(ax, nodes["split"][0], nodes["split"][1]-0.33,
               nodes["split"][0], nodes["train_label"][1]+0.33)

    rounded_box(ax, nodes["scaler"][0], nodes["scaler"][1], 3.0, 0.65,
                "📊 StandardScaler (fit on train)\n→ mean & std di-export",
                fill="#1a1033", edge=C_ACCENT2, fs=8.5)
    arrow(ax, nodes["train_label"][0], nodes["train_label"][1]-0.33,
               nodes["train_label"][0], nodes["scaler"][1]+0.33)

    # ── branch arrows ─────────────────────────────────────────────────────────
    feat_keys = {"RF": "feat_rf", "MLP": "feat_mlp", "CNN1D": "raw_cnn"}
    for bx, lbl in [(6.5, "RF"), (9.5, "MLP"), (12.5, "CNN1D")]:
        key = feat_keys[lbl]
        arrow(ax, nodes["scaler"][0]+1.5, nodes["scaler"][1]-0.33,
                   bx, nodes[key][1]+0.33)
        label_arrow(ax, (nodes["scaler"][0]+bx)/2 + 0.4,
                    nodes["scaler"][1]-0.8, lbl)

    # ── FEATURES ─────────────────────────────────────────────────────────────
    rounded_box(ax, nodes["feat_rf"][0], nodes["feat_rf"][1], 2.4, 0.65,
                "📐 Ekstraksi Fitur\n(36 statistik)",
                fill="#161b22", edge=C_GRAY_BOX, fs=8.5)

    rounded_box(ax, nodes["feat_mlp"][0], nodes["feat_mlp"][1], 2.4, 0.65,
                "📐 Ekstraksi Fitur\n(36 statistik)",
                fill="#161b22", edge=C_GRAY_BOX, fs=8.5)

    rounded_box(ax, nodes["raw_cnn"][0], nodes["raw_cnn"][1], 2.4, 0.65,
                "📐 Raw Window\n(1500 poin time-series)",
                fill="#161b22", edge=C_GRAY_BOX, fs=8.5)

    for lbl in ["rf", "mlp"]:
        n = nodes[f"feat_{lbl}"][1]
        arrow(ax, nodes[f"{lbl}_path"][0], n+0.33,
                   nodes[f"{lbl}_path"][0], n+0.33 - 0.001)  # no-op, already placed above

    # ── MODEL TRAIN ───────────────────────────────────────────────────────────
    fill_rf = "#1a3020" if True else "#0d2137"
    rounded_box(ax, nodes["train_rf"][0], nodes["train_rf"][1], 2.4, 0.65,
                "🌲 Random Forest\n(100 tree, max_depth=10)",
                fill="#0d2137", edge=C_ACCENT3, fs=8.5)
    arrow(ax, nodes["feat_rf"][0], nodes["feat_rf"][1]-0.33,
               nodes["train_rf"][0], nodes["train_rf"][1]+0.33)

    rounded_box(ax, nodes["train_mlp"][0], nodes["train_mlp"][1], 2.4, 0.65,
                "🧠 MLP Neural Network\n(64→32, ReLU, Adam)",
                fill="#0d2137", edge=C_ACCENT3, fs=8.5)
    arrow(ax, nodes["feat_mlp"][0], nodes["feat_mlp"][1]-0.33,
               nodes["train_mlp"][0], nodes["train_mlp"][1]+0.33)

    rounded_box(ax, nodes["train_cnn"][0], nodes["train_cnn"][1], 2.4, 0.65,
                "🔁 CNN1D ConvNet\n(Conv1D→MaxPool→Dense)",
                fill="#0d2137", edge=C_ACCENT3, fs=8.5)
    arrow(ax, nodes["raw_cnn"][0], nodes["raw_cnn"][1]-0.33,
               nodes["train_cnn"][0], nodes["train_cnn"][1]+0.33)

    # ── EVALUASI ──────────────────────────────────────────────────────────────
    rounded_box(ax, nodes["eval_rf"][0], nodes["eval_rf"][1], 2.4, 0.65,
                "📈 Evaluasi: Akurasi, F1, Confusion Matrix\n& LOSO Cross-Validation",
                fill="#161b22", edge=C_MIDPOINT, fs=7.8)
    arrow(ax, nodes["train_rf"][0], nodes["train_rf"][1]-0.33,
               nodes["eval_rf"][0], nodes["eval_rf"][1]+0.33)

    rounded_box(ax, nodes["eval_mlp"][0], nodes["eval_mlp"][1], 2.4, 0.65,
                "📈 Evaluasi: Akurasi, F1, Confusion Matrix\n& LOSO Cross-Validation",
                fill="#161b22", edge=C_MIDPOINT, fs=7.8)
    arrow(ax, nodes["train_mlp"][0], nodes["train_mlp"][1]-0.33,
               nodes["eval_mlp"][0], nodes["eval_mlp"][1]+0.33)

    rounded_box(ax, nodes["eval_cnn"][0], nodes["eval_cnn"][1], 2.4, 0.65,
                "📈 Evaluasi: Akurasi, F1, Confusion Matrix\n& LOSO Cross-Validation",
                fill="#161b22", edge=C_MIDPOINT, fs=7.8)
    arrow(ax, nodes["train_cnn"][0], nodes["train_cnn"][1]-0.33,
               nodes["eval_cnn"][0], nodes["eval_cnn"][1]+0.33)

    # ── HEADER EXPORT ─────────────────────────────────────────────────────────
    rounded_box(ax, nodes["exp_rf"][0], nodes["exp_rf"][1], 2.4, 0.65,
                "📦 Export: rf_model_data.h\n(m2cgen → pure C)",
                fill="#0d2137", edge=C_ACCENT3, fs=8.5)
    arrow(ax, nodes["eval_rf"][0], nodes["eval_rf"][1]-0.33,
               nodes["exp_rf"][0], nodes["exp_rf"][1]+0.33)

    rounded_box(ax, nodes["exp_mlp"][0], nodes["exp_mlp"][1], 2.4, 0.65,
                "📦 Export: mlp_model_data.h\n(TFLite Micro → int8 quantized)",
                fill="#0d2137", edge=C_ACCENT3, fs=8.5)
    arrow(ax, nodes["eval_mlp"][0], nodes["eval_mlp"][1]-0.33,
               nodes["exp_mlp"][0], nodes["exp_mlp"][1]+0.33)

    rounded_box(ax, nodes["exp_cnn"][0], nodes["exp_cnn"][1], 2.4, 0.65,
                "📦 Export: cnn1d_model_data.h\n(TFLite Micro → int8 quantized)",
                fill="#0d2137", edge=C_ACCENT3, fs=8.5)
    arrow(ax, nodes["eval_cnn"][0], nodes["eval_cnn"][1]-0.33,
               nodes["exp_cnn"][0], nodes["exp_cnn"][1]+0.33)

    # ── decision: best model? ─────────────────────────────────────────────────
    diamond(ax, 15.5, 3.4, 2.0, 0.80,
            "Model\nOK?", fs=8)
    for (ex, ey) in [(nodes["exp_rf"][0], nodes["exp_rf"][1]),
                     (nodes["exp_mlp"][0], nodes["exp_mlp"][1]),
                     (nodes["exp_cnn"][0], nodes["exp_cnn"][1])]:
        arrow(ax, ex+1.2, ey, 14.5, 3.8)

    label_arrow(ax, 15.8, 4.3, "Ya")

    rounded_box(ax, 16.8, 2.3, 2.0, 0.70,
                "✅ Model siap\ndi-deploy",
                fill="#1a3020", edge=C_ACCENT3, fs=8, bold=True)

    arrow(ax, 15.5, 3.0, 16.8, 2.65)

    label_arrow(ax, 17.1, 4.0, "Tidak →\nhyperparameter\ntuning")

    # ── separators between branches ───────────────────────────────────────────
    for x in [7.8, 10.8]:
        ax.axvline(x, color=C_GRAY_BOX, linewidth=0.4, alpha=0.25, ymin=0.02, ymax=0.92)

    # ── branch labels ─────────────────────────────────────────────────────────
    ax.text(6.5, 8.0, "Random Forest", ha="center", fontsize=9,
            color=C_ACCENT3, fontweight="bold", fontfamily="monospace")
    ax.text(9.5, 8.0, "MLP", ha="center", fontsize=9,
            color=C_ACCENT3, fontweight="bold", fontfamily="monospace")
    ax.text(12.5, 8.0, "CNN1D", ha="center", fontsize=9,
            color=C_ACCENT3, fontweight="bold", fontfamily="monospace")

    # ── phase colour strip ────────────────────────────────────────────────────
    ax.add_patch(plt.Rectangle((0, 0), 0.12, 9.5,
                               facecolor=C_ACCENT2, alpha=0.35, zorder=1))

    footer(ax, "Fase 2 dari 3  |  Pelatihan Model  |  Output: .keras, .h (ESP32-ready)",
           y=-0.01)

    fig.savefig(outpath, dpi=180, bbox_inches="tight",
                facecolor=C_PHASE2_BG)
    plt.close()
    print("  -> Saved " + outpath)


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — DEPLOYMENT & INFERENSI REAL-TIME (landscape: 13 × 7.5)
# ════════════════════════════════════════════════════════════════════════════
def draw_phase3(outpath):
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor(C_PHASE3_BG)
    ax.set_facecolor(C_PHASE3_BG)

    # ── title ─────────────────────────────────────────────────────────────────
    ax.text(9.5, 9.62, "GAMBAR 3.3  FASE 3 — DEPLOYMENT & INFERENSI REAL-TIME",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C_ACCENT3, fontfamily="monospace")
    ax.text(9.5, 9.20, "Dual-Inference: ESP32 (device) + Python (PC debug) | USB HID Keyboard",
            ha="center", va="center", fontsize=8.5, color=C_GRAY_BOX,
            fontfamily="monospace")

    # ─── left column: ESP32 side ───────────────────────────────────────────────
    cx_esp = 4.5   # center x of ESP32 column

    rounded_box(ax, cx_esp, 8.5, 6.0, 0.65,
                "⚡ XIAO ESP32-S3\n(flashed dengan model .h)",
                fill="#0d3a1a", edge=C_ACCENT3, fs=8.5, bold=True)

    rounded_box(ax, cx_esp, 7.4, 5.8, 0.65,
                "IMU MPU6050\nakuisisi real-time @ 100 Hz",
                fill="#161b22", edge=C_ACCENT3, fs=8.5)

    arrow(ax, cx_esp, 8.5-0.33, cx_esp, 7.74)

    rounded_box(ax, cx_esp, 6.4, 5.8, 0.65,
                "Sliding Window 2.5 detik\n(250 sampel, overlapping 50%)",
                fill="#161b22", edge=C_ACCENT3, fs=8.5)

    arrow(ax, cx_esp, 7.07, cx_esp, 6.73)

    rounded_box(ax, cx_esp, 5.3, 5.8, 0.70,
                "Normalisasi (StandardScaler)\n+ Inference model (.h / TFLite)",
                fill="#161b22", edge=C_ACCENT3, fs=8.5)

    arrow(ax, cx_esp, 5.98, cx_esp, 5.65)

    rounded_box(ax, cx_esp, 4.2, 5.8, 0.70,
                "🎯 Prediksi Gesture\n→ Mapping ke Keyboard HID",
                fill="#161b22", edge=C_MIDPOINT, fs=8.5, bold=True)

    arrow(ax, cx_esp, 4.93, cx_esp, 4.55)

    rounded_box(ax, cx_esp, 3.1, 5.8, 0.65,
                "USB HID Keyboard Event\n(Arrow Keys, PgUp/PgDn …)",
                fill="#0d3a1a", edge=C_ACCENT3, fs=8.5)

    arrow(ax, cx_esp, 3.42, cx_esp, 2.78)

    rounded_box(ax, cx_esp, 1.9, 5.8, 0.65,
                "✅ PC menerima input\ntanpa driver tambahan",
                fill="#0d3a1a", edge=C_ACCENT3, fs=8.5)

    arrow(ax, cx_esp, 2.27, cx_esp, 1.58)

    # ─── right column: PC Python side ─────────────────────────────────────────
    cx_pc = 13.5   # center x of PC column

    rounded_box(ax, cx_pc, 8.5, 5.0, 0.65,
                "💻 Python GUI (gui.py)\nLive Debug Tab",
                fill="#0d2137", edge=C_ACCENT1, fs=8.5, bold=True)

    rounded_box(ax, cx_pc, 7.4, 5.0, 0.65,
                "Terima data serial\n(baud 115200)",
                fill="#161b22", edge=C_ACCENT1, fs=8.5)

    arrow(ax, cx_pc, 8.5-0.33, cx_pc, 7.74)

    rounded_box(ax, cx_pc, 6.4, 5.0, 0.65,
                "Normalisasi\n+ PC-side inference (.keras)",
                fill="#161b22", edge=C_ACCENT1, fs=8.5)

    arrow(ax, cx_pc, 7.07, cx_pc, 6.73)

    rounded_box(ax, cx_pc, 5.3, 5.0, 0.70,
                "📊 Visualisasi real-time\n(Grafik gesture + confidence)",
                fill="#161b22", edge=C_ACCENT1, fs=8.5)

    arrow(ax, cx_pc, 5.98, cx_pc, 5.65)

    rounded_box(ax, cx_pc, 4.2, 5.0, 0.70,
                "🔄 Bandingkan prediksi:\nESP32 vs PC (side-by-side)",
                fill="#161b22", edge=C_ACCENT1, fs=8.5)

    arrow(ax, cx_pc, 4.93, cx_pc, 4.55)

    rounded_box(ax, cx_pc, 3.1, 5.0, 0.65,
                "📈 Logging & Metrik\n(Akurasi, Latency, FPS)",
                fill="#161b22", edge=C_ACCENT1, fs=8.5)

    arrow(ax, cx_pc, 3.42, cx_pc, 2.78)

    rounded_box(ax, cx_pc, 1.9, 5.0, 0.65,
                "📋 Simpan log ke CSV\n(untuk analisis thesis)",
                fill="#0d2137", edge=C_ACCENT1, fs=8.5)

    arrow(ax, cx_pc, 2.27, cx_pc, 1.58)

    # ─── middle: serial / USB bridge ───────────────────────────────────────────
    # vertical dashed "cable" between ESP and PC
    for y in np.arange(1.3, 7.5, 0.25):
        ax.plot([cx_esp + 3.0, cx_esp + 3.4], [y, y],
                color=C_GRAY_BOX, linewidth=0.5, alpha=0.4)

    ax.text(9.5, 0.88, "🔌 Serial USB\n(115200 baud)", ha="center",
            fontsize=7.5, color=C_GRAY_BOX, fontfamily="monospace")

    # ─── bottom: loop arrow ────────────────────────────────────────────────────
    # Gesture loop label
    ax.annotate("", xy=(cx_esp - 0.3, 1.55),
                xytext=(cx_esp - 3.0, 1.55),
                arrowprops=dict(arrowstyle="-|>", color=C_ACCENT3,
                                linewidth=1.4, mutation_scale=14,
                                connectionstyle="arc3,rad=0",
                                shrinkA=0, shrinkB=0),
                zorder=3)
    ax.text(cx_esp - 1.8, 1.35, "Loop (100 Hz)", ha="center",
            fontsize=7.5, color=C_ACCENT3, fontfamily="monospace")

    # ─── vertical divider ─────────────────────────────────────────────────────
    ax.axvline(8.5, color=C_GRAY_BOX, linewidth=0.6, alpha=0.3, ymin=0.02, ymax=0.93)

    # column labels
    ax.text(cx_esp, 0.5, "🖥 ESP32 (Device Side)", ha="center",
            fontsize=9, color=C_ACCENT3, fontweight="bold",
            fontfamily="monospace")
    ax.text(cx_pc,  0.5, "💻 PC (Python Debug Side)", ha="center",
            fontsize=9, color=C_ACCENT1, fontweight="bold",
            fontfamily="monospace")

    # ─── phase colour strip ────────────────────────────────────────────────────
    ax.add_patch(plt.Rectangle((0, 0), 0.12, 10,
                               facecolor=C_ACCENT3, alpha=0.35, zorder=1))

    footer(ax, "Fase 3 dari 3  |  Deployment & Inferensi Real-Time  |  USB HID → PC Keyboard Input",
           y=-0.01)

    fig.savefig(outpath, dpi=180, bbox_inches="tight",
                facecolor=C_PHASE3_BG)
    plt.close()
    print("  -> Saved " + outpath)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    outdir = "thesis_figures"
    import os
    os.makedirs(outdir, exist_ok=True)

    print("\nGenerating flowchart figures …")
    draw_phase1(f"{outdir}/methodology_phase1_data_collection.png")
    draw_phase2(f"{outdir}/methodology_phase2_model_training.png")
    draw_phase3(f"{outdir}/methodology_phase3_deployment.png")
    print("\nAll done!")