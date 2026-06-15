#!/usr/bin/env python3
"""
Library Chatbot — Image Embedding Generation (Jina CLIP v2, local)

Generates embeddings for book cover IMAGES using the
jinaai/jina-clip-v2 model via transformers AutoModel.

NOTE: We use AutoModel (not SentenceTransformer) because the
sentence-transformers wrapper for jina-clip-v2 only registers the
'text' modality.  AutoModel exposes encode_image() directly.

The model is moved to GPU automatically if CUDA is available.

Images do NOT use a retrieval prompt — Jina's asymmetric prompts only
apply to text (passages use "document", queries use "retrieval.query").

Output
------
JSON file at `embeddings_image/image_embeddings_jina.json` with the
SAME record shape as the Google / Qwen versions.

Install
-------
    conda install -c conda-forge transformers pillow
    pip install einops timm
"""

from __future__ import annotations

import json
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

# ---- Compatibility patch for Jina CLIP v2 + transformers >= 4.49 ----
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
    pass

try:
    from transformers import AutoModel
except ImportError:
    print("ERROR: transformers not installed.  conda install -c conda-forge transformers")
    sys.exit(1)


# ======================================================================
# Configuration
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE  = PROJECT_ROOT / "chunking"       / "book_chunks_image.json"
OUTPUT_DIR  = PROJECT_ROOT / "embeddings_image"
OUTPUT_FILE = OUTPUT_DIR   / "image_embeddings_jina.json"
STATS_FILE  = OUTPUT_DIR   / "image_embedding_statistics_jina.txt"

EMBEDDING_MODEL = "jinaai/jina-clip-v2"
EMBEDDING_DIM   = 1024  # Jina CLIP v2 default output dimension

MAX_IMAGE_DIMENSION = 512
JPEG_QUALITY        = 85
MAX_IMAGE_SIZE_MB   = 4
SUPPORTED_FORMATS   = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

SAVE_INTERVAL = 25


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
        print("  WARNING: CPU mode is very slow. Enable CUDA for fast embedding.")
    return model


def _preprocess_image(image_path: Path) -> Optional[Image.Image]:
    """Load → RGB → thumbnail to MAX_IMAGE_DIMENSION."""
    if not image_path.exists():
        return None
    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        return None
    if image_path.stat().st_size / (1024 * 1024) > MAX_IMAGE_SIZE_MB:
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > MAX_IMAGE_DIMENSION:
            img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)
        return img
    except Exception:
        return None


def _embed_image(model, pil_image: Image.Image) -> List[float]:
    """Embed one image via AutoModel.encode_image() — normalised output."""
    with torch.no_grad():
        embedding = model.encode_image([pil_image])
    if hasattr(embedding, "cpu"):
        embedding = embedding.cpu()
    arr = np.array(embedding).flatten().astype(np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


# ======================================================================
# Main pipeline
# ======================================================================

def main() -> int:
    print(f"Image embedder (Jina CLIP v2 local, model={EMBEDDING_MODEL})")
    print(f"  input:  {INPUT_FILE}")
    print(f"  output: {OUTPUT_FILE}")

    if not INPUT_FILE.exists():
        print(f"ERROR: input not found: {INPUT_FILE}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = _load_model(EMBEDDING_MODEL)

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    chunks = [c for c in chunks if c.get("chunk_type") == "image"]
    total = len(chunks)
    print(f"  image chunks loaded: {total:,}")
    if total == 0:
        print("Nothing to do.")
        return 0

    existing_on_disk = sum(
        1 for c in chunks if Path(c.get("image_path", "")).exists()
    )
    print(f"  images found on disk: {existing_on_disk:,}/{total:,}")

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

    stats: Dict[str, Any] = {
        "started":         datetime.now(),
        "total_chunks":    total,
        "already_done":    len(done_book_ids),
        "processed":       0,
        "failed":          0,
        "skipped_missing": 0,
        "skipped_format":  0,
        "skipped_size":    0,
        "embedding_dim":   EMBEDDING_DIM,
    }

    try:
        with tqdm(total=len(pending), desc="image embeddings") as pbar:
            for i, chunk in enumerate(pending):
                book_id        = chunk.get("book_id") or f"unknown_{i}"
                image_path_str = chunk.get("image_path", "")
                image_path     = Path(image_path_str)

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

                if len(embedding) != EMBEDDING_DIM:
                    pbar.write(
                        f"  WARNING: dim mismatch for {book_id}: "
                        f"got {len(embedding)}, expected {EMBEDDING_DIM}"
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

    OUTPUT_FILE.write_text(
        json.dumps(embedded, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stats["ended"] = datetime.now()
    duration = stats["ended"] - stats["started"]

    report = [
        "LIBRARY CHATBOT — IMAGE EMBEDDING STATISTICS (Jina CLIP v2 local)",
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
