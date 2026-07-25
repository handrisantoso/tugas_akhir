#!/usr/bin/env python3
"""
Library Chatbot — Image Embedding Generation (Google GenAI SDK)

Generates 3072-dim embeddings for book cover images using the Google
GenAI Python SDK with `gemini-embedding-2`. This script is intentionally
aligned with the runtime query path in `RAG/rag_engine.py` so ingest and
query share the same:
  - SDK / provider (google.genai)
  - Model id (gemini-embedding-2)
  - Image preprocessing (RGB → thumbnail to 512 px → JPEG q=85 → typed
    Part)

Output
------
JSON file at `embeddings_image/image_embeddings_google.json` with the
same record shape as the OpenRouter version, so downstream
`vector_db/create_vector_db_multimodal.py` can ingest it without
changes (point its IMAGE_EMBEDDINGS_FILE at the new path).

Quotas honoured (per the project's gemini-embedding-2 plan):
  - 100 requests / minute
  - 30,000 tokens / minute
  - 1,000 requests / day

The OpenRouter version of this script (`create_image_embeddings_openrouter.py`)
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
    print("ERROR: numpy not installed. pip install numpy")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed. pip install Pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not installed. pip install tqdm")
    sys.exit(1)

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai not installed. pip install google-genai")
    sys.exit(1)

# Local sibling import (script lives in embedding/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _google_rate_limit import GeminiRateLimiter, estimate_image_tokens  # noqa: E402


# ======================================================================
# Configuration
# ======================================================================

# All paths are resolved relative to the project root (parent of this
# script's directory) so the script works regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "chunking" / "book_chunks_image.json"
OUTPUT_DIR = PROJECT_ROOT / "embeddings_image"
OUTPUT_FILE = OUTPUT_DIR / "image_embeddings_google.json"
STATS_FILE = OUTPUT_DIR / "image_embedding_statistics_google.txt"

# .env search order: project root first, then embedding/ local
ENV_CANDIDATES = [
    PROJECT_ROOT / ".env",
    Path(__file__).resolve().parent / ".env",
]

# Must match RAG/config.py and RAG/rag_engine.py so query and ingest
# live in the same vector space.
EMBEDDING_MODEL = "gemini-embedding-2"

# Mirror the runtime preprocessing in `_preprocess_and_embed_image`.
MAX_IMAGE_DIMENSION = 512
JPEG_QUALITY = 85
MAX_IMAGE_SIZE_MB = 4  # safety cap before resize
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

SAVE_INTERVAL = 25


# ======================================================================
# Helpers
# ======================================================================

def _load_api_key() -> str:
    """Load GOOGLE_API_KEY from project-root .env, falling back to embedding/.env."""
    try:
        from dotenv import load_dotenv
        for env_file in ENV_CANDIDATES:
            if env_file.exists():
                load_dotenv(env_file, override=False)
    except ImportError:
        for env_file in ENV_CANDIDATES:
            if not env_file.exists():
                continue
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Set it in the project-root .env or in the environment."
        )
    return key


def _preprocess_image(image_path: Path) -> Optional[bytes]:
    """Load → RGB → thumbnail to MAX_IMAGE_DIMENSION → JPEG q=JPEG_QUALITY.

    Returns the JPEG bytes, or None if the file can't be processed.
    Mirrors `RAG/rag_engine.py::_preprocess_and_embed_image` exactly.
    """
    if not image_path.exists():
        return None

    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        return None

    size_mb = image_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return None

    img = Image.open(image_path).convert("RGB")
    if max(img.size) > MAX_IMAGE_DIMENSION:
        img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _embed_image_bytes(client: "genai.Client", limiter: GeminiRateLimiter,
                       jpeg_bytes: bytes) -> List[float]:
    """Send JPEG bytes through gemini-embedding-2 via the SDK's typed Part."""
    limiter.wait_if_needed(estimate_image_tokens(len(jpeg_bytes)))

    part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=part,
    )
    embedding = list(result.embeddings[0].values)
    # Token usage isn't always returned for embed_content; record the
    # estimate, which keeps the limiter conservative.
    limiter.record(estimate_image_tokens(len(jpeg_bytes)))
    return embedding


# ======================================================================
# Main pipeline
# ======================================================================

def main() -> int:
    print(f"Image embedder (Google GenAI SDK, model={EMBEDDING_MODEL})")
    print(f"  input:  {INPUT_FILE}")
    print(f"  output: {OUTPUT_FILE}")

    if not INPUT_FILE.exists():
        print(f"ERROR: input not found: {INPUT_FILE}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)
    limiter = GeminiRateLimiter()

    # ---- Load chunks ----
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    chunks = [c for c in chunks if c.get("chunk_type") == "image"]
    total = len(chunks)
    print(f"  image chunks loaded: {total:,}")
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
        "started": datetime.now(),
        "total_chunks": total,
        "already_done": len(done_book_ids),
        "processed": 0,
        "failed": 0,
        "skipped_missing": 0,
        "skipped_format": 0,
        "skipped_size": 0,
        "embedding_dim": None,
    }

    # ---- Embed ----
    try:
        with tqdm(total=len(pending), desc="image embeddings") as pbar:
            for i, chunk in enumerate(pending):
                book_id = chunk.get("book_id") or f"unknown_{i}"
                image_path_str = chunk.get("image_path", "")
                image_path = Path(image_path_str)

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

                jpeg = _preprocess_image(image_path)
                if jpeg is None:
                    stats["skipped_format"] += 1
                    pbar.update(1)
                    continue

                try:
                    embedding = _embed_image_bytes(client, limiter, jpeg)
                except Exception as e:
                    stats["failed"] += 1
                    pbar.write(f"  failed book_id={book_id}: {e}")
                    pbar.update(1)
                    continue

                if stats["embedding_dim"] is None:
                    stats["embedding_dim"] = len(embedding)
                elif len(embedding) != stats["embedding_dim"]:
                    pbar.write(
                        f"  WARNING: dim mismatch for {book_id}: "
                        f"got {len(embedding)}, expected {stats['embedding_dim']}"
                    )

                embedded.append({
                    "book_id": book_id,
                    "chunk_type": "image",
                    "image_path": image_path_str,
                    "embedding": embedding,
                    "metadata": {
                        **chunk.get("metadata", {}),
                        "embedding_model": EMBEDDING_MODEL,
                        "embedding_dim": len(embedding),
                        "preprocessing": {
                            "max_dim": MAX_IMAGE_DIMENSION,
                            "format": "JPEG",
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

    rl = limiter.stats()
    report = [
        "LIBRARY CHATBOT — IMAGE EMBEDDING STATISTICS (Google GenAI SDK)",
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
        "RATE-LIMIT WINDOW (final):",
        f"  Requests in last 60s:   {rl['requests_last_60s']} / {rl['rpm_cap']}",
        f"  Tokens   in last 60s:   {rl['tokens_last_60s']:,} / {rl['tpm_cap']:,}",
        f"  Requests today:         {rl['requests_today']} / {rl['rpd_cap']}",
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
