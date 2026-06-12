"""
Generate thesis figures: Feature Extraction Visualization (2 separate files)
  - feature_extraction_signal.png  : annotated single-axis signal
  - feature_extraction_vector.png  : full 36-feature bar chart
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from recorder import GestureRecorder
from models import extract_features
from config import (
    WINDOW_SIZE, NUM_AXES, AXIS_NAMES, AXIS_COLORS,
    DEFAULT_CSV, PROJECT_ROOT
)

OUT_DIR = os.path.join(PROJECT_ROOT, "thesis_figures")

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f9fa",
    "font.family":       "serif",
    "font.size":         12,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Load dataset ──────────────────────────────────────────────────────
print("Loading dataset...")
X, y, _ = GestureRecorder.load_dataset(DEFAULT_CSV)

# Flick Up (class 1), sample ke-3 untuk variasi
idxs       = np.where(y == 1)[0]
window_flat = X[idxs[2]]                             # (1500,)
window      = window_flat.reshape(WINDOW_SIZE, NUM_AXES)  # (250, 6)

# Demo axis: Gyro X (index 3) — paling jelas untuk flick_up
DEMO_AXIS  = 3
axis_label = AXIS_NAMES[DEMO_AXIS]
axis_color = AXIS_COLORS[DEMO_AXIS]
sig = window[:, DEMO_AXIS]
t   = np.arange(WINDOW_SIZE) / 100.0  # seconds

# ── Hitung 6 fitur untuk sumbu demo ──────────────────────────────────
mean_val      = np.mean(sig)
std_val       = np.std(sig)
max_val       = np.max(sig)
min_val       = np.min(sig)
rms_val       = np.sqrt(np.mean(sig ** 2))
mean_centered = sig - mean_val
cross_mask    = np.diff(np.sign(mean_centered)) != 0
mcr_val       = np.sum(cross_mask) / (WINDOW_SIZE - 1)
cross_times   = t[:-1][cross_mask]

# ── Ekstrak vektor 36 fitur ───────────────────────────────────────────
feat_vec   = extract_features(window_flat.reshape(1, -1))[0]   # (36,)
FEAT_NAMES = ["Mean", "Std", "Max", "Min", "RMS", "MCR"]

yrange = max_val - min_val
pad    = yrange * 0.09

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Annotated Signal
# ═══════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(11, 5))

# Sinyal mentah
ax1.plot(t, sig, color=axis_color, linewidth=1.8, label=axis_label, zorder=3)

# Shading ±std
ax1.fill_between(t, mean_val - std_val, mean_val + std_val,
                 color="#3B82F6", alpha=0.13,
                 label=f"$\\pm$Std ({std_val:.2f})", zorder=1)

# Garis horizontal
ax1.axhline(mean_val, color="#3B82F6", linestyle="--", linewidth=1.6,
            label=f"Mean ({mean_val:.2f})", zorder=2)
ax1.axhline(max_val,  color="#EF4444", linestyle=":",  linewidth=1.6,
            label=f"Max ({max_val:.2f})",  zorder=2)
ax1.axhline(min_val,  color="#22C55E", linestyle=":",  linewidth=1.6,
            label=f"Min ({min_val:.2f})",  zorder=2)
ax1.axhline(rms_val,  color="#F59E0B", linestyle="-.", linewidth=1.6,
            label=f"RMS ({rms_val:.2f})",  zorder=2)

# Titik MCR (max 18 untuk keterbacaan)
shown = cross_times[:18]
ax1.scatter(shown, np.interp(shown, t, sig),
            color="#A855F7", s=35, zorder=5,
            label=f"MCR crossings (rate={mcr_val:.3f})")

# ── Anotasi panah ────────────────────────────────────────────────────
ax1.annotate(f"Max = {max_val:.2f}",
             xy=(t[np.argmax(sig)], max_val),
             xytext=(t[np.argmax(sig)] + 0.18, max_val + pad),
             fontsize=9.5, color="#EF4444",
             arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1.3))

ax1.annotate(f"Min = {min_val:.2f}",
             xy=(t[np.argmin(sig)], min_val),
             xytext=(t[np.argmin(sig)] + 0.18, min_val - pad),
             fontsize=9.5, color="#22C55E",
             arrowprops=dict(arrowstyle="->", color="#22C55E", lw=1.3))

ax1.annotate(f"Mean = {mean_val:.2f}",
             xy=(0.08, mean_val),
             xytext=(0.08, mean_val + pad * 1.8),
             fontsize=9.5, color="#3B82F6",
             arrowprops=dict(arrowstyle="->", color="#3B82F6", lw=1.3))

# Bracket 2σ di sisi kanan
bx = t[-1] - 0.08
ax1.annotate("", xy=(bx, mean_val - std_val), xytext=(bx, mean_val + std_val),
             arrowprops=dict(arrowstyle="<->", color="#3B82F6", lw=1.6))
ax1.text(bx - 0.12, mean_val, f"2$\\sigma$={2*std_val:.2f}",
         fontsize=9, color="#3B82F6", va="center", ha="right")

ax1.annotate(f"RMS = {rms_val:.2f}",
             xy=(1.9, rms_val),
             xytext=(1.55, rms_val - pad * 2.2),
             fontsize=9.5, color="#F59E0B",
             arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=1.3))

ax1.set_xlabel("Waktu (s)", fontsize=12)
ax1.set_ylabel(f"Nilai Sensor ({axis_label})", fontsize=12)
ax1.set_title(
    f"Ekstraksi 6 Fitur Statistik dari Satu Window\n"
    f"Gesture: Flick Up — Sumbu: {axis_label}  (250 sampel @ 100 Hz = 2,5 s)",
    fontsize=12, fontweight="bold"
)
ax1.legend(fontsize=9.5, loc="upper right", framealpha=0.92, ncol=2, columnspacing=1.2)
ax1.set_xlim(0, t[-1] + 0.02)

fig1.tight_layout()
path1 = os.path.join(OUT_DIR, "feature_extraction_signal.png")
fig1.savefig(path1, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {path1}")
plt.close(fig1)

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 2 — 36-Feature Bar Chart
# ═══════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(13, 5))

bar_colors = [AXIS_COLORS[a] for a in range(NUM_AXES) for _ in range(6)]
x_pos = np.arange(36)
ax2.bar(x_pos, feat_vec, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.85)
ax2.axhline(0, color="#6B7280", linewidth=0.8)

# Separator antar grup sumbu + label grup
y_top = ax2.get_ylim()[1]
y_bot = ax2.get_ylim()[0]
y_lbl = y_top + (y_top - y_bot) * 0.04

SHORT = ["ax", "ay", "az", "gx", "gy", "gz"]
for a in range(NUM_AXES):
    if a > 0:
        ax2.axvline(a * 6 - 0.5, color="#9CA3AF", linewidth=1.0, linestyle="--", alpha=0.55)
    mid = a * 6 + 2.5
    ax2.text(mid, y_lbl, SHORT[a], ha="center", va="bottom",
             fontsize=11, color=AXIS_COLORS[a], fontweight="bold")

# x-axis: label fitur per bar
ax2.set_xticks(x_pos)
ax2.set_xticklabels(FEAT_NAMES * NUM_AXES, fontsize=8.5, rotation=45, ha="right")

ax2.set_ylabel("Nilai Fitur (sebelum StandardScaler)", fontsize=12)
ax2.set_xlim(-0.6, 35.6)
ax2.set_ylim(y_bot - (y_top - y_bot) * 0.02, y_top + (y_top - y_bot) * 0.14)
ax2.set_title(
    "Vektor 36 Fitur Hasil extract_features()  —  Input RF dan MLP\n"
    "Gesture: Flick Up  (6 fitur × 6 sumbu = 36 dimensi)",
    fontsize=12, fontweight="bold"
)

legend_patches = [mpatches.Patch(color=AXIS_COLORS[i], label=AXIS_NAMES[i])
                  for i in range(NUM_AXES)]
ax2.legend(handles=legend_patches, fontsize=9.5, loc="upper right",
           framealpha=0.92, ncol=3, columnspacing=1.0)

fig2.tight_layout()
path2 = os.path.join(OUT_DIR, "feature_extraction_vector.png")
fig2.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {path2}")
plt.close(fig2)
