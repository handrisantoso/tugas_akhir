#!/usr/bin/env python3
"""
Library Chatbot — Text Embedding Generation (Jina CLIP v2, local)

Generates embeddings for book TEXT chunks using the
jinaai/jina-clip-v2 model loaded locally via sentence-transformers.
No API key required — model weights are downloaded automatically
from HuggingFace on first run.

Output
------
JSON file at `embeddings_text/text_embeddings_jina.json` with the
SAME record shape as the Google / Qwen versions so that
`vector_db/create_vector_db_multimodal.py` can ingest it without
changes (point TEXT_EMBEDDINGS_FILE at the new path).

Install
-------
    conda install -c conda-forge sentence-transformers pillow
    pip install einops

Jina CLIP v2 is much smaller than Qwen (~1.5 GB vs ~4-5 GB).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed.  conda install -c conda-forge numpy")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not installed.  conda install -c conda-forge tqdm")
    sys.exit(1)

try:
    from transformers import AutoModel
except ImportError:
    print("ERROR: transformers not installed.")
    print("       conda install -c conda-forge transformers")
    sys.exit(1)

# ---- Compatibility patch for Jina CLIP v2 + transformers >= 4.49 ----
# Jina's custom code imports `clip_loss` from transformers, but it was
# removed in newer versions.  Re-add it so the import succeeds.
try:
    import torch
    import torch.nn as nn
    import transformers.models.clip.modeling_clip as _clip_module
    if not hasattr(_clip_module, "clip_loss"):
        def _clip_loss(similarity: torch.Tensor) -> torch.Tensor:
            caption_loss = nn.functional.cross_entropy(
                similarity, torch.arange(len(similarity), device=similarity.device))
            image_loss = nn.functional.cross_entropy(
                similarity.t(), torch.arange(len(similarity), device=similarity.device))
            return (caption_loss + image_loss) / 2.0
        _clip_module.clip_loss = _clip_loss
except Exception:
    pass  # best-effort; if it fails the original error will surface later


# ======================================================================
# Configuration
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE  = PROJECT_ROOT / "chunking"       / "book_chunks_text.json"
OUTPUT_DIR  = PROJECT_ROOT / "embeddings_text"
OUTPUT_FILE = OUTPUT_DIR   / "text_embeddings_jina.json"
STATS_FILE  = OUTPUT_DIR   / "text_embedding_statistics_jina.txt"

EMBEDDING_MODEL = "jinaai/jina-clip-v2"
EMBEDDING_DIM   = 1024  # Jina CLIP v2 output dimension

# Using AutoModel.encode_text() — same interface as encode_image().
# No asymmetric prompts: stored text and query text both go through
# the raw text encoder, keeping them in the same vector space.

# Batch size for encode_text() calls.
BATCH_SIZE = 16

# How often (chunks processed) to flush results to disk.
SAVE_INTERVAL = 50


# ======================================================================
# Helpers
# ======================================================================

def _load_model(model_name: str):
    """Load Jina CLIP v2 via AutoModel and move to GPU if available."""
    print(f"Loading model: {model_name}")
    print("  (first run downloads ~1.5 GB; subsequent runs use cache)")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"  model loaded on GPU — embedding dim: {EMBEDDING_DIM}")
    else:
        print(f"  model loaded on CPU — embedding dim: {EMBEDDING_DIM}")
    return model


def _embed_texts(model, texts: List[str]) -> List[List[float]]:
    """Return normalised embedding vectors via AutoModel.encode_text().

    No prompts are used — both stored and query embeddings go through
    the same raw text encoder, keeping them in the same vector space.
    """
    with torch.no_grad():
        embeddings = model.encode_text(texts)
    if hasattr(embeddings, "cpu"):
        embeddings = embeddings.cpu()
    arr = np.array(embeddings).astype(np.float32)
    # L2-normalise each row
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    arr = arr / norms
    return arr.tolist()


# ======================================================================
# Main pipeline
# ======================================================================

def main() -> int:
    print(f"Text embedder (Jina CLIP v2 local, model={EMBEDDING_MODEL})")
    print(f"  input:  {INPUT_FILE}")
    print(f"  output: {OUTPUT_FILE}")

    if not INPUT_FILE.exists():
        print(f"ERROR: input not found: {INPUT_FILE}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    model = _load_model(EMBEDDING_MODEL)
    embedding_dim = EMBEDDING_DIM

    # ---- Load chunks ----
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    chunks = [c for c in chunks if c.get("chunk_type") == "text"]
    total = len(chunks)
    print(f"  text chunks loaded: {total:,}")
    if total == 0:
        print("Nothing to do.")
        return 0

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
        "started":      datetime.now(),
        "total_chunks": total,
        "already_done": len(done_book_ids),
        "processed":    0,
        "failed":       0,
        "skipped_empty": 0,
        "embedding_dim": embedding_dim,
        "total_chars":  0,
    }

    # ---- Embed ----
    try:
        with tqdm(total=len(pending), desc="text embeddings") as pbar:
            for i, chunk in enumerate(pending):
                book_id = chunk.get("book_id") or f"unknown_{i}"
                text    = chunk.get("text") or ""

                if not text.strip():
                    stats["skipped_empty"] += 1
                    pbar.write(f"  empty text for book_id={book_id}, skipping")
                    pbar.update(1)
                    continue

                try:
                    [embedding] = _embed_texts(model, [text])
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
                    "chunk_type": "text",
                    "text":       chunk.get("text", ""),
                    "embedding":  embedding,
                    "metadata": {
                        **chunk.get("metadata", {}),
                        "embedding_model":  EMBEDDING_MODEL,
                        "embedding_dim":    len(embedding),
                        "input_char_count": len(text),
                        "processed_at":     datetime.now().isoformat(),
                    },
                })
                stats["processed"]   += 1
                stats["total_chars"] += len(text)

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
    avg_chars = (stats["total_chars"] / stats["processed"]) if stats["processed"] else 0

    report = [
        "LIBRARY CHATBOT — TEXT EMBEDDING STATISTICS (Jina CLIP v2 local)",
        "=" * 60,
        f"Date:       {stats['started'].strftime('%Y-%m-%d %H:%M:%S')}",
        f"Duration:   {duration}",
        f"Model:      {EMBEDDING_MODEL}",
        f"Output:     {OUTPUT_FILE}",
        "",
        "PROCESSING SUMMARY:",
        f"  Total text chunks:      {stats['total_chunks']:,}",
        f"  Already done (resume):  {stats['already_done']:,}",
        f"  Newly embedded:         {stats['processed']:,}",
        f"  Failed:                 {stats['failed']:,}",
        f"  Skipped (empty text):   {stats['skipped_empty']:,}",
        f"  Avg input length:       {avg_chars:.0f} chars",
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
