#!/usr/bin/env python3
"""
PCA Visualisation of Text vs Image Embeddings (2-D)
====================================================

For each of the three embedding models (Google, Jina, Qwen) this script:
  1. Loads the first 10 text embeddings and first 10 image embeddings.
  2. Fits a single PCA to ALL 20 vectors together (so the axes are shared).
  3. Plots text points in one colour and image points in another.
  4. Draws a dashed line between the text and image point for every book
     that appears in BOTH sets (same book_id).
  5. Annotates every point with a short book label.
  6. Saves the figure to  eval/pca_embeddings/<model>_pca.png

Usage
-----
Run from the repository root:
    python eval/pca_embeddings/visualize_pca.py

Or point to different files with optional flags:
    python eval/pca_embeddings/visualize_pca.py --n 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from sklearn.decomposition import PCA

# ── Colour palette ──────────────────────────────────────────────────────────
TEXT_COLOR  = "#1a6fc4"   # rich blue     – text embeddings
IMAGE_COLOR = "#d94f00"   # burnt orange  – image embeddings
LINE_COLOR  = "#999999"   # mid-grey      – connecting lines
BG_COLOR    = "#f7f7fa"   # off-white     – axes background
FIG_BG      = "#ffffff"   # pure white    – figure background
GRID_COLOR  = "#ddddee"   # subtle grid

# ── Model definitions ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent   # repo root

MODELS = {
    "Google": {
        "text_file":  ROOT / "embeddings_text"  / "text_embeddings_google.json",
        "image_file": ROOT / "embeddings_image" / "image_embeddings_google.json",
        "accent":     "#1a6fc4",   # blue title
    },
    "Jina": {
        "text_file":  ROOT / "embeddings_text"  / "text_embeddings_jina.json",
        "image_file": ROOT / "embeddings_image" / "image_embeddings_jina.json",
        "accent":     "#c0185a",   # deep pink title
    },
    "Qwen": {
        "text_file":  ROOT / "embeddings_text"  / "text_embeddings_qwen.json",
        "image_file": ROOT / "embeddings_image" / "image_embeddings_qwen.json",
        "accent":     "#a33a00",   # dark orange title
    },
}

OUTPUT_DIR = Path(__file__).resolve().parent   # eval/pca_embeddings/


# ── Helpers ───────────────────────────────────────────────────────────────────

def _short_label(book_id: str, max_len: int = 18) -> str:
    """Create a readable label from a book_id."""
    label = (
        book_id
        .replace("isbn13_", "")
        .replace("isbn10_", "")
        .replace("oclc_", "oclc:")
        .replace("book_1_", "")
        .replace("_", " ")
    )
    return label[:max_len] + "…" if len(label) > max_len else label


def _load_n(filepath: Path, n: int) -> list[dict]:
    """Load first *n* records from a JSON embedding file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:n]


# ── Main plot routine ─────────────────────────────────────────────────────────

def plot_model(model_name: str, cfg: dict, n: int, output_dir: Path) -> None:
    text_records  = _load_n(cfg["text_file"],  n)
    image_records = _load_n(cfg["image_file"], n)

    # ---- build id → record maps ----
    text_map  = {r["book_id"]: r for r in text_records}
    image_map = {r["book_id"]: r for r in image_records}

    text_ids  = list(text_map.keys())
    image_ids = list(image_map.keys())

    # ---- stack ALL vectors for joint PCA ----
    all_vecs = (
        [text_map[bid]["embedding"]  for bid in text_ids]
        + [image_map[bid]["embedding"] for bid in image_ids]
    )
    X = np.array(all_vecs, dtype=np.float32)

    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X)

    nt = len(text_ids)
    text_pts  = X2d[:nt]          # shape (n, 2)
    image_pts = X2d[nt:]          # shape (n, 2)

    var = pca.explained_variance_ratio_

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors="#333333")
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_color("#333333")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.8)

    # ── Dashed connectors for shared books ────────────────────────────────────
    shared_ids = set(text_ids) & set(image_ids)
    for bid in shared_ids:
        ti = text_ids.index(bid)
        ii = image_ids.index(bid)
        tx, ty = text_pts[ti]
        ix, iy = image_pts[ii]
        ax.plot(
            [tx, ix], [ty, iy],
            color=LINE_COLOR, linewidth=1.0,
            linestyle="--", alpha=0.55, zorder=1
        )

    # ── Scatter: image first (lower z), text on top ───────────────────────────
    ax.scatter(
        image_pts[:, 0], image_pts[:, 1],
        s=110, c=IMAGE_COLOR, edgecolors="white", linewidths=0.6,
        alpha=0.90, zorder=3, label="Image embedding"
    )
    ax.scatter(
        text_pts[:, 0], text_pts[:, 1],
        s=110, c=TEXT_COLOR, edgecolors="white", linewidths=0.6,
        alpha=0.90, zorder=4, label="Text embedding"
    )

    # ── Annotations ───────────────────────────────────────────────────────────
    label_kw = dict(
        fontsize=7, color="#111111",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white", edgecolor="#bbbbbb",
            alpha=0.85, linewidth=0.5
        )
    )
    for i, bid in enumerate(text_ids):
        ax.annotate(
            _short_label(bid), (text_pts[i, 0], text_pts[i, 1]),
            xytext=(5, 5), textcoords="offset points",
            ha="left", va="bottom", zorder=5, **label_kw
        )
    for i, bid in enumerate(image_ids):
        ax.annotate(
            _short_label(bid), (image_pts[i, 0], image_pts[i, 1]),
            xytext=(5, -10), textcoords="offset points",
            ha="left", va="top", zorder=5, **label_kw
        )

    # ── Legend & title ────────────────────────────────────────────────────────
    dash_handle = mlines.Line2D(
        [], [], color=LINE_COLOR, linewidth=1.2, linestyle="--",
        label="Same book (text <-> image)"
    )
    legend = ax.legend(
        handles=[
            ax.scatter([], [], s=80, c=TEXT_COLOR,  edgecolors="#333333", linewidths=0.6, label="Text embedding"),
            ax.scatter([], [], s=80, c=IMAGE_COLOR, edgecolors="#333333", linewidths=0.6, label="Image embedding"),
            dash_handle,
        ],
        facecolor="white", edgecolor="#cccccc",
        labelcolor="#111111", fontsize=9, loc="best"
    )

    ax.set_title(
        f"{model_name} Embeddings — 2D PCA (first {n} books each)\n"
        f"Explained variance: PC1={var[0]:.1%}  PC2={var[1]:.1%}",
    )
    ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)", fontsize=9, color="#333333")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)", fontsize=9, color="#333333")

    # ── Tighten axis limits to reduce whitespace / visual spread ──────────────
    x_all = np.concatenate([text_pts[:, 0], image_pts[:, 0]])
    y_all = np.concatenate([text_pts[:, 1], image_pts[:, 1]])
    x_pad = (x_all.max() - x_all.min()) * 0.18
    y_pad = (y_all.max() - y_all.min()) * 0.18
    ax.set_xlim(x_all.min() - x_pad, x_all.max() + x_pad)
    ax.set_ylim(y_all.min() - y_pad, y_all.max() + y_pad)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = output_dir / f"{model_name.lower()}_pca.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {out_path.relative_to(ROOT)}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PCA visualisation of text vs image embeddings.")
    parser.add_argument("-n", "--n", type=int, default=10,
                        help="Number of entries to load from each embedding file (default: 10).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, cfg in MODELS.items():
        print(f"Processing {model_name}...")
        if not cfg["text_file"].exists():
            print(f"  WARNING: text file not found → {cfg['text_file']}. Skipping.")
            continue
        if not cfg["image_file"].exists():
            print(f"  WARNING: image file not found → {cfg['image_file']}. Skipping.")
            continue
        plot_model(model_name, cfg, args.n, OUTPUT_DIR)

    print("\nDone! All plots saved to eval/pca_embeddings/")


if __name__ == "__main__":
    main()
