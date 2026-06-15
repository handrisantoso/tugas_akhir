#!/usr/bin/env python3
"""
Library Chatbot — Image Embedding Generation (Qwen3-VL-Embedding, local)

Generates embeddings for book cover IMAGES using the
Qwen/Qwen3-VL-Embedding-2B (or -8B) model loaded locally via
sentence-transformers.  No API key required — model weights are
downloaded automatically from HuggingFace on first run.

Output
------
JSON file at `embeddings_image/image_embeddings_qwen.json` with the
SAME record shape as the Google / OpenRouter versions so that
`vector_db/create_vector_db_multimodal.py` can ingest it without
changes (point IMAGE_EMBEDDINGS_FILE at the new path).

Install (conda-forge first, then pip for qwen-vl-utils)
-------
    conda install -c conda-forge "transformers>=4.57" sentence-transformers pillow
    pip install qwen-vl-utils

The Google version of this script (`create_image_embeddings_google.py`)
is intentionally left untouched.
"""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed.  conda install -c conda-forge numpy")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed.  conda install -c conda-forge pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not installed.  conda install -c conda-forge tqdm")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed.")
    print("       conda install -c conda-forge sentence-transformers")
    sys.exit(1)


# ======================================================================
# Configuration
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE  = PROJECT_ROOT / "chunking"       / "book_chunks_image.json"
OUTPUT_DIR  = PROJECT_ROOT / "embeddings_image"
OUTPUT_FILE = OUTPUT_DIR   / "image_embeddings_qwen.json"
STATS_FILE  = OUTPUT_DIR   / "image_embedding_statistics_qwen.txt"

# Switch to "Qwen/Qwen3-VL-Embedding-8B" if you have the VRAM for it.
EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"

# Mirror the preprocessing used in the Google version.
MAX_IMAGE_DIMENSION = 512
JPEG_QUALITY        = 85
MAX_IMAGE_SIZE_MB   = 4
SUPPORTED_FORMATS   = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# How often (chunks processed) to flush results to disk.
SAVE_INTERVAL = 25


# ======================================================================
# Helpers
# ======================================================================

def _load_model(model_name: str) -> SentenceTransformer:
    """Load the Qwen embedding model.

    On first call the weights (~4-5 GB for 2B) are downloaded from
    HuggingFace and cached in ~/.cache/huggingface/hub automatically.
    Subsequent runs load from the local cache instantly.
    """
    print(f"Loading model: {model_name}")
    print("  (first run downloads ~4-5 GB; subsequent runs use cache)")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print(f"  model loaded — embedding dim: {model.get_sentence_embedding_dimension()}")
    return model


def _preprocess_image(image_path: Path) -> Optional[Image.Image]:
    """Load → RGB → thumbnail to MAX_IMAGE_DIMENSION.

    Mirrors the preprocessing used in `create_image_embeddings_google.py`
    so both scripts operate on the same effective image content.

    Returns the PIL Image ready for encoding, or None if the file
    cannot be processed.
    """
    if not image_path.exists():
        return None

    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        return None

    size_mb = image_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return None

    try:
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > MAX_IMAGE_DIMENSION:
            img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)
        return img
    except Exception:
        return None


def _embed_image(model: SentenceTransformer, pil_image: Image.Image) -> List[float]:
    """Return a normalised embedding vector (plain Python list) for one image.

    sentence-transformers accepts PIL Image objects directly for
    Qwen3-VL-Embedding.
    """
    embeddings = model.encode(
        [pil_image],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,   # cosine-ready; mirrors Qwen's recommended usage
        convert_to_numpy=True,
    )
    return embeddings[0].tolist()


# ======================================================================
# Main pipeline
# ======================================================================

def main() -> int:
    print(f"Image embedder (Qwen local, model={EMBEDDING_MODEL})")
    print(f"  input:  {INPUT_FILE}")
    print(f"  output: {OUTPUT_FILE}")

    if not INPUT_FILE.exists():
        print(f"ERROR: input not found: {INPUT_FILE}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    model = _load_model(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()

    # ---- Load chunks ----
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    chunks = [c for c in chunks if c.get("chunk_type") == "image"]
    total = len(chunks)
    print(f"  image chunks loaded: {total:,}")
    if total == 0:
        print("Nothing to do.")
        return 0

    # Quick disk check
    existing_on_disk = sum(
        1 for c in chunks if Path(c.get("image_path", "")).exists()
    )
    print(f"  images found on disk: {existing_on_disk:,}/{total:,}")

    # ---- Resume if an output already exists ----
    embedded: List[Dict[str, Any]] = []
    done_book_ids: set[str] = set()
    if OUTPUT_FILE.exists():
        try:
            embedded = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
            done_book_ids = {c.get("book_id") for c in embedded if c.get("book_id")}
            print(f"  resuming: {len(done_book_ids):,} already embedded")
        except Exception as e:
            print(f"  could not read existing output ({e}); starting fresh")
            embedded = []
            done_book_ids = set()

    pending = [c for c in chunks if c.get("book_id") not in done_book_ids]
    print(f"  to embed: {len(pending):,}")

    # ---- Stats ----
    stats: Dict[str, Any] = {
        "started":         datetime.now(),
        "total_chunks":    total,
        "already_done":    len(done_book_ids),
        "processed":       0,
        "failed":          0,
        "skipped_missing": 0,
        "skipped_format":  0,
        "skipped_size":    0,
        "embedding_dim":   embedding_dim,
    }

    # ---- Embed ----
    try:
        with tqdm(total=len(pending), desc="image embeddings") as pbar:
            for i, chunk in enumerate(pending):
                book_id        = chunk.get("book_id") or f"unknown_{i}"
                image_path_str = chunk.get("image_path", "")
                image_path     = Path(image_path_str)

                # Pre-flight checks (mirroring Google version)
                if not image_path.exists():
                    stats["skipped_missing"] += 1
                    pbar.update(1)
                    continue
                if image_path.suffix.lower() not in SUPPORTED_FORMATS:
                    stats["skipped_format"] += 1
                    pbar.update(1)
                    continue
                if image_path.stat().st_size / (1024 * 1024) > MAX_IMAGE_SIZE_MB:
                    stats["skipped_size"] += 1
                    pbar.update(1)
                    continue

                pil_image = _preprocess_image(image_path)
                if pil_image is None:
                    stats["skipped_format"] += 1
                    pbar.update(1)
                    continue

                try:
                    embedding = _embed_image(model, pil_image)
                except Exception as e:
                    stats["failed"] += 1
                    pbar.write(f"  failed book_id={book_id}: {e}")
                    pbar.update(1)
                    continue

                # Dimension sanity check
                if len(embedding) != embedding_dim:
                    pbar.write(
                        f"  WARNING: dim mismatch for {book_id}: "
                        f"got {len(embedding)}, expected {embedding_dim}"
                    )

                embedded.append({
                    "book_id":    book_id,
                    "chunk_type": "image",
                    "image_path": image_path_str,
                    "embedding":  embedding,
                    "metadata": {
                        **chunk.get("metadata", {}),
                        "embedding_model": EMBEDDING_MODEL,
                        "embedding_dim":   len(embedding),
                        "preprocessing": {
                            "max_dim": MAX_IMAGE_DIMENSION,
                            "format":  "PIL→RGB→thumbnail",
                            "quality": JPEG_QUALITY,
                        },
                        "processed_at": datetime.now().isoformat(),
                    },
                })
                stats["processed"] += 1

                if (i + 1) % SAVE_INTERVAL == 0:
                    OUTPUT_FILE.write_text(
                        json.dumps(embedded, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    pbar.write(f"  saved progress: {len(embedded):,}/{total:,}")

                pbar.update(1)

    except KeyboardInterrupt:
        print("\nInterrupted — saving partial results...")

    # ---- Final save ----
    OUTPUT_FILE.write_text(
        json.dumps(embedded, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ---- Stats report ----
    stats["ended"] = datetime.now()
    duration = stats["ended"] - stats["started"]

    report = [
        "LIBRARY CHATBOT — IMAGE EMBEDDING STATISTICS (Qwen local)",
        "=" * 60,
        f"Date:       {stats['started'].strftime('%Y-%m-%d %H:%M:%S')}",
        f"Duration:   {duration}",
        f"Model:      {EMBEDDING_MODEL}",
        f"Output:     {OUTPUT_FILE}",
        "",
        "PROCESSING SUMMARY:",
        f"  Total image chunks:     {stats['total_chunks']:,}",
        f"  Already done (resume):  {stats['already_done']:,}",
        f"  Newly embedded:         {stats['processed']:,}",
        f"  Failed:                 {stats['failed']:,}",
        f"  Skipped (missing):      {stats['skipped_missing']:,}",
        f"  Skipped (bad format):   {stats['skipped_format']:,}",
        f"  Skipped (too large):    {stats['skipped_size']:,}",
        f"  Embedding dim:          {stats['embedding_dim']}",
        "",
    ]
    if embedded:
        sample = np.array(embedded[0]["embedding"])
        report += [
            "SAMPLE EMBEDDING:",
            f"  mean={sample.mean():.6f}  std={sample.std():.6f}",
            f"  min={sample.min():.6f}  max={sample.max():.6f}",
        ]

    STATS_FILE.write_text("\n".join(report), encoding="utf-8")
    print()
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
