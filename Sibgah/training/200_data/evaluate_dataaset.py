import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "cleaned_data")
IMU_FEATURES = ["ax1", "ay1", "az1", "gx1", "gy1", "gz1",
                "ax2", "ay2", "az2", "gx2", "gy2", "gz2"]


# ── helpers ────────────────────────────────────────────────────────────────────

def load_dataset(data_dir: str) -> dict[str, list[pd.DataFrame]]:
    """Return {label: [df, ...]} for every CSV found under data_dir/<label>/."""
    dataset: dict[str, list[pd.DataFrame]] = {}
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        frames = []
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(label_dir, fname))
            frames.append(df)
        dataset[label] = frames
    return dataset


def sample_lengths(frames: list[pd.DataFrame]) -> list[int]:
    return [len(df) for df in frames]


# ── analysis functions ─────────────────────────────────────────────────────────

def print_overview(dataset: dict[str, list[pd.DataFrame]]) -> None:
    print("=" * 55)
    print("DATASET OVERVIEW")
    print("=" * 55)
    total = 0
    for label, frames in dataset.items():
        lengths = sample_lengths(frames)
        total += len(frames)
        print(f"  {label:<8}  samples={len(frames):>4}  "
              f"rows/sample: min={min(lengths):>4}  "
              f"avg={np.mean(lengths):>6.1f}  max={max(lengths):>4}")
    print(f"  {'TOTAL':<8}  samples={total:>4}")
    print()


def check_missing_values(dataset: dict[str, list[pd.DataFrame]]) -> None:
    print("=" * 55)
    print("MISSING VALUES CHECK")
    print("=" * 55)
    issues_found = False
    for label, frames in dataset.items():
        for i, df in enumerate(frames):
            n_missing = df[IMU_FEATURES].isnull().sum().sum()
            if n_missing > 0:
                print(f"  [{label}] sample #{i+1} has {n_missing} missing value(s)")
                issues_found = True
    if not issues_found:
        print("  No missing values found.")
    print()


def print_feature_stats(dataset: dict[str, list[pd.DataFrame]]) -> None:
    print("=" * 55)
    print("PER-LABEL FEATURE STATISTICS (mean ± std)")
    print("=" * 55)
    for label, frames in dataset.items():
        all_data = pd.concat(frames)[IMU_FEATURES]
        print(f"\n  [{label}]")
        for feat in IMU_FEATURES:
            m, s = all_data[feat].mean(), all_data[feat].std()
            mn, mx = all_data[feat].min(), all_data[feat].max()
            print(f"    {feat:<6}  mean={m:>10.1f}  std={s:>9.1f}  "
                  f"min={mn:>10.1f}  max={mx:>10.1f}")
    print()


# ── plot functions ─────────────────────────────────────────────────────────────

def plot_class_distribution(dataset: dict[str, list[pd.DataFrame]]) -> None:
    labels = list(dataset.keys())
    counts = [len(v) for v in dataset.values()]
    colors = plt.cm.tab10.colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, padding=3)
    ax.set_title("Sample Count per Class")
    ax.set_xlabel("Gesture")
    ax.set_ylabel("Number of Samples")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "eval_class_distribution.png"), dpi=120)
    plt.show()


def plot_sample_length_distribution(dataset: dict[str, list[pd.DataFrame]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, frames in dataset.items():
        lengths = sample_lengths(frames)
        ax.hist(lengths, bins=30, alpha=0.6, label=label)
    ax.set_title("Distribution of Sample Lengths (rows per recording)")
    ax.set_xlabel("Number of Rows")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "eval_sample_lengths.png"), dpi=120)
    plt.show()


def plot_feature_boxplots(dataset: dict[str, list[pd.DataFrame]]) -> None:
    """One subplot per IMU feature; box per label."""
    all_data_per_label: dict[str, pd.DataFrame] = {
        label: pd.concat(frames)[IMU_FEATURES].reset_index(drop=True)
        for label, frames in dataset.items()
    }
    labels = list(all_data_per_label.keys())
    n_features = len(IMU_FEATURES)

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, feat in enumerate(IMU_FEATURES):
        ax = axes[i]
        data_to_plot = [all_data_per_label[lbl][feat].values for lbl in labels]
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showfliers=False)
        colors = plt.cm.tab10.colors[:len(labels)]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(feat)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Feature Distribution per Gesture (no outliers shown)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "eval_feature_boxplots.png"), dpi=120)
    plt.show()


def plot_sample_overlay(dataset: dict[str, list[pd.DataFrame]],
                        feature: str = "ax1",
                        n_samples: int = 5) -> None:
    """Overlay a few time-series samples per class for a given feature."""
    labels = list(dataset.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 3), sharey=True)

    for ax, label in zip(axes, labels):
        frames = dataset[label][:n_samples]
        for df in frames:
            ax.plot(df["timestamp"], df[feature], alpha=0.6, linewidth=0.9)
        ax.set_title(label)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel(feature)
    fig.suptitle(f"Sample overlay – {feature} (first {n_samples} samples per class)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", f"eval_overlay_{feature}.png"), dpi=120)
    plt.show()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\nLoading dataset from: {DATA_DIR}\n")
    dataset = load_dataset(DATA_DIR)

    print_overview(dataset)
    check_missing_values(dataset)
    print_feature_stats(dataset)

    print("Generating plots …")
    plot_class_distribution(dataset)
    plot_sample_length_distribution(dataset)
    plot_feature_boxplots(dataset)
    plot_sample_overlay(dataset, feature="ax1")
    plot_sample_overlay(dataset, feature="gx1")
    print("Done. Plots saved next to the cleaned_data folder.")


if __name__ == "__main__":
    main()
