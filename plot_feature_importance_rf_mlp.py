"""
Plot Feature Importance — RF (Gini) vs MLP (Permutation Importance).
Trains both models on 100% data with the same 36-feature pipeline,
then compares which features each model relies on most.

Output:
  - thesis_figures/feature_importance_rf_mlp.png
"""
import os, sys, json, warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from config import (
    GESTURE_NAMES, WINDOW_SIZE, NUM_AXES, AXIS_NAMES, AXIS_COLORS,
    DEFAULT_CSV, PROJECT_ROOT, NUM_CLASSES,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_RF_TREES, DEFAULT_RF_DEPTH,
)
from recorder import GestureRecorder
from models import extract_features, fit_scaler, build_mlp

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f9fa",
    "font.family":       "serif",
    "font.size":         11,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

OUT_DIR = os.path.join(PROJECT_ROOT, "thesis_figures")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42

# ── Feature names (6 axes × 6 stats = 36) ─────────────────────────
STAT_NAMES = ["Mean", "Std", "Max", "Min", "RMS", "MCR"]
SHORT_AXES = ["ax", "ay", "az", "gx", "gy", "gz"]
FEATURE_NAMES_36 = [f"{ax}_{st}" for ax in SHORT_AXES for st in STAT_NAMES]

# ── Seed ───────────────────────────────────────────────────────────
def _seed_everything():
    import random, tensorflow as tf
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ── Load dataset ───────────────────────────────────────────────────
print("Loading dataset ...")
csv_path = DEFAULT_CSV
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(__file__), "gesture_dataset.csv")
X, y, subj = GestureRecorder.load_dataset(csv_path)
print(f"  Samples: {len(y)},  Classes: {NUM_CLASSES},  Subjects: {len(np.unique(subj))}")

# ── Extract 36-dim features ────────────────────────────────────────
X_feat = extract_features(X)
scaler = fit_scaler(X_feat)
X_scaled = scaler.transform(X_feat)

_seed_everything()

# ═══════════════════════════════════════════════════════════════════
#  1. RF — Gini Importance (built-in) + Permutation Importance
# ═══════════════════════════════════════════════════════════════════
print("\n[1/2] Training RF ...")
rf = RandomForestClassifier(
    n_estimators=DEFAULT_RF_TREES, max_depth=DEFAULT_RF_DEPTH,
    random_state=SEED, n_jobs=-1,
)
rf.fit(X_scaled, y)
rf_gini = rf.feature_importances_

print("  Computing RF permutation importance ...")
rf_perm = permutation_importance(rf, X_scaled, y, n_repeats=10,
                                  random_state=SEED, n_jobs=-1)
rf_perm_mean = rf_perm.importances_mean

print(f"  RF training accuracy: {rf.score(X_scaled, y)*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════
#  2. MLP — Permutation Importance
# ═══════════════════════════════════════════════════════════════════
print("\n[2/2] Training MLP ...")
from tensorflow import keras as K

mlp = build_mlp(input_dim=X_scaled.shape[1])
early = K.callbacks.EarlyStopping(monitor="loss", patience=10,
                                   restore_best_weights=True, verbose=0)
mlp.fit(X_scaled, y, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
        verbose=0, callbacks=[early])

mlp_train_acc = np.mean(np.argmax(mlp.predict(X_scaled, verbose=0), axis=1) == y)
print(f"  MLP training accuracy: {mlp_train_acc*100:.2f}%")

# sklearn permutation_importance needs a scorer-compatible estimator
# We wrap the Keras model
from sklearn.base import BaseEstimator, ClassifierMixin

class _KerasWrapper(BaseEstimator, ClassifierMixin):
    """Minimal sklearn wrapper for Keras model (predict only)."""
    def __init__(self, model):
        self.model = model
        self.classes_ = np.arange(NUM_CLASSES)
    def fit(self, X, y): return self
    def predict(self, X):
        return np.argmax(self.model.predict(X, verbose=0), axis=1)
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

mlp_wrapper = _KerasWrapper(mlp)

print("  Computing MLP permutation importance ...")
mlp_perm = permutation_importance(mlp_wrapper, X_scaled, y, n_repeats=10,
                                   random_state=SEED, n_jobs=1)  # n_jobs=1 for Keras
mlp_perm_mean = mlp_perm.importances_mean

K.backend.clear_session()

# ═══════════════════════════════════════════════════════════════════
#  PLOT — 2×2 panels
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# ── Helper: axis color per feature index ──────────────────────────
def _axis_color(feat_idx):
    return AXIS_COLORS[feat_idx // 6]

# ── Panel (0,0): RF Gini Importance ──────────────────────────────
ax = axes[0, 0]
sort_idx = np.argsort(rf_gini)[::-1]
colors = [_axis_color(i) for i in sort_idx]
ax.barh(range(36), rf_gini[sort_idx], color=colors,
        edgecolor="white", linewidth=0.4)
ax.set_yticks(range(36))
ax.set_yticklabels([FEATURE_NAMES_36[i] for i in sort_idx], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Importance (Gini)", fontsize=11)
ax.set_title("RF — Gini Importance", fontsize=13, fontweight="bold")
for rank in range(36):
    val = rf_gini[sort_idx[rank]]
    ax.text(val + 0.001, rank, f"{val:.3f}", va="center", fontsize=6.5,
            fontweight="bold", color="#1F2937")

# ── Panel (0,1): RF Permutation Importance ────────────────────────
ax = axes[0, 1]
sort_idx = np.argsort(rf_perm_mean)[::-1]
colors = [_axis_color(i) for i in sort_idx]
ax.barh(range(36), rf_perm_mean[sort_idx], color=colors,
        edgecolor="white", linewidth=0.4)
ax.set_yticks(range(36))
ax.set_yticklabels([FEATURE_NAMES_36[i] for i in sort_idx], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Mean Accuracy Drop", fontsize=11)
ax.set_title("RF — Permutation Importance", fontsize=13, fontweight="bold")
for rank in range(36):
    val = rf_perm_mean[sort_idx[rank]]
    ax.text(val + 0.0005, rank, f"{val:.4f}", va="center", fontsize=6.5,
            fontweight="bold", color="#1F2937")

# ── Panel (1,0): MLP Permutation Importance ───────────────────────
ax = axes[1, 0]
sort_idx = np.argsort(mlp_perm_mean)[::-1]
colors = [_axis_color(i) for i in sort_idx]
ax.barh(range(36), mlp_perm_mean[sort_idx], color=colors,
        edgecolor="white", linewidth=0.4)
ax.set_yticks(range(36))
ax.set_yticklabels([FEATURE_NAMES_36[i] for i in sort_idx], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Mean Accuracy Drop", fontsize=11)
ax.set_title("MLP — Permutation Importance", fontsize=13, fontweight="bold")
for rank in range(36):
    val = mlp_perm_mean[sort_idx[rank]]
    ax.text(val + 0.0005, rank, f"{val:.4f}", va="center", fontsize=6.5,
            fontweight="bold", color="#1F2937")

# ── Panel (1,1): Side-by-side Top-10 comparison ──────────────────
ax = axes[1, 1]
# Combine top features from both models
rf_top10 = set(np.argsort(rf_perm_mean)[::-1][:10])
mlp_top10 = set(np.argsort(mlp_perm_mean)[::-1][:10])
combined = sorted(rf_top10 | mlp_top10, key=lambda i: rf_perm_mean[i] + mlp_perm_mean[i], reverse=True)

y_pos = np.arange(len(combined))
bar_h = 0.35
ax.barh(y_pos - bar_h/2, [rf_perm_mean[i] for i in combined], bar_h,
        color="#22C55E", edgecolor="white", linewidth=0.4, label="RF", alpha=0.85)
ax.barh(y_pos + bar_h/2, [mlp_perm_mean[i] for i in combined], bar_h,
        color="#3B82F6", edgecolor="white", linewidth=0.4, label="MLP", alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels([FEATURE_NAMES_36[i] for i in combined], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Mean Accuracy Drop", fontsize=11)
ax.set_title("Perbandingan Top Fitur — RF vs MLP\n(Permutation Importance)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

# ── Axis legend (shared) ─────────────────────────────────────────
legend_patches = [mpatches.Patch(color=AXIS_COLORS[i], label=AXIS_NAMES[i])
                  for i in range(NUM_AXES)]
for ax_i in [axes[0, 0], axes[0, 1], axes[1, 0]]:
    ax_i.legend(handles=legend_patches, fontsize=7, loc="lower right",
                framealpha=0.9)

# ── Save ──────────────────────────────────────────────────────────
fig.suptitle("Feature Importance — RF vs MLP (36 Fitur Statistik)",
             fontsize=15, fontweight="bold", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.97))

out_path = os.path.join(OUT_DIR, "feature_importance_rf_mlp.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\n[OK] Saved: {out_path}")

# ── Summary table ─────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  Top-10 Features per Model (Permutation Importance)")
print("  " + "-" * 68)
print(f"  {'Rank':<6} {'RF Feature':<16} {'RF Score':>10}   {'MLP Feature':<16} {'MLP Score':>10}")
print("  " + "-" * 68)
rf_sorted = np.argsort(rf_perm_mean)[::-1]
mlp_sorted = np.argsort(mlp_perm_mean)[::-1]
for rank in range(10):
    ri, mi = rf_sorted[rank], mlp_sorted[rank]
    print(f"  {rank+1:<6} {FEATURE_NAMES_36[ri]:<16} {rf_perm_mean[ri]:>10.4f}   "
          f"{FEATURE_NAMES_36[mi]:<16} {mlp_perm_mean[mi]:>10.4f}")
print("=" * 72)
