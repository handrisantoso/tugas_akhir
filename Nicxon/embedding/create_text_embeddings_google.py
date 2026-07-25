#!/usr/bin/env python3
"""
Library Chatbot — Text Embedding Generation (Google GenAI SDK)

Generates 3072-dim embeddings for book TEXT chunks using the Google
GenAI Python SDK with `gemini-embedding-2`. Aligned with the runtime
query path in `RAG/rag_engine.py` so query and ingest share the same
SDK, model id, and embedding space.

Output
------
JSON file at `embeddings_text/text_embeddings_google.json` with the
same record shape as the OpenRouter version.

Quotas honoured (per the project's gemini-embedding-2 plan):
  - 100 requests / minute
  - 30,000 tokens / minute
  - 1,000 requests / day

The OpenRouter version of this script (`create_text_embeddings_openrouter.py`)
is intentionally left untouched.
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
    print("ERROR: numpy not installed. pip install numpy")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not installed. pip install tqdm")
    sys.exit(1)

try:
    from google import genai
except ImportError:
    print("ERROR: google-genai not installed. pip install google-genai")
    sys.exit(1)

# Local sibling import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _google_rate_limit import GeminiRateLimiter, estimate_text_tokens  # noqa: E402


# ======================================================================
# Configuration
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "chunking" / "book_chunks_text.json"
OUTPUT_DIR = PROJECT_ROOT / "embeddings_text"
OUTPUT_FILE = OUTPUT_DIR / "text_embeddings_google.json"
STATS_FILE = OUTPUT_DIR / "text_embedding_statistics_google.txt"

# .env search order: project root first, then embedding/ local
ENV_CANDIDATES = [
    PROJECT_ROOT / ".env",
    Path(__file__).resolve().parent / ".env",
]

EMBEDDING_MODEL = "gemini-embedding-2"

# Hard input cap for gemini-embedding-2 is 2048 tokens. We truncate at
# ~7000 chars (estimate_text_tokens uses chars/3.5, so 7000 → ~2000
# tokens with headroom) to avoid hard 400s on long chunks.
MAX_INPUT_CHARS = 7000

SAVE_INTERVAL = 50


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


def _embed_text(client: "genai.Client", limiter: GeminiRateLimiter,
                text: str) -> List[float]:
    estimated = estimate_text_tokens(text)
    limiter.wait_if_needed(estimated)

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    embedding = list(response.embeddings[0].values)
    limiter.record(estimated)
    return embedding


# ======================================================================
# Main pipeline
# ======================================================================

def main() -> int:
    print(f"Text embedder (Google GenAI SDK, model={EMBEDDING_MODEL})")
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

    chunks = [c for c in chunks if c.get("chunk_type") == "text"]
    total = len(chunks)
    print(f"  text chunks loaded: {total:,}")
    if total == 0:
        print("Nothing to do.")
        return 0

    # ---- Resume ----
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
        "truncated": 0,
        "embedding_dim": None,
        "total_chars": 0,
    }

    # ---- Embed ----
    try:
        with tqdm(total=len(pending), desc="text embeddings") as pbar:
            for i, chunk in enumerate(pending):
                book_id = chunk.get("book_id") or f"unknown_{i}"
                text = chunk.get("text") or ""

                if not text.strip():
                    stats["failed"] += 1
                    pbar.write(f"  empty text for book_id={book_id}, skipping")
                    pbar.update(1)
                    continue

                truncated_flag = False
                if len(text) > MAX_INPUT_CHARS:
                    text = text[:MAX_INPUT_CHARS]
                    truncated_flag = True
                    stats["truncated"] += 1

                try:
                    embedding = _embed_text(client, limiter, text)
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
                    "chunk_type": "text",
                    "text": chunk.get("text", ""),
                    "embedding": embedding,
                    "metadata": {
                        **chunk.get("metadata", {}),
                        "embedding_model": EMBEDDING_MODEL,
                        "embedding_dim": len(embedding),
                        "input_truncated": truncated_flag,
                        "input_char_count": len(text),
                        "processed_at": datetime.now().isoformat(),
                    },
                })
                stats["processed"] += 1
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

    rl = limiter.stats()
    report = [
        "LIBRARY CHATBOT — TEXT EMBEDDING STATISTICS (Google GenAI SDK)",
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
        f"  Truncated to {MAX_INPUT_CHARS} chars:  {stats['truncated']:,}",
        f"  Avg input length:       {avg_chars:.0f} chars",
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
