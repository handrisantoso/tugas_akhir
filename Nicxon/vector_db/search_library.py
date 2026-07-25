#!/usr/bin/env python3
"""
Library retrieval tester for the multimodal ChromaDB.

Goal
----
A small CLI that mirrors the exact embedding path used by the live RAG
engine (`RAG/rag_engine.py`) so you can probe what each collection
returns for a given query, without going through any of the LLM steps.

It:
  - reads GOOGLE_API_KEY from the environment (.env in this folder, the
    project root, or RAG/.env — first one wins),
Supports two embedding backends, selected with --model:
  gemini  — Google GenAI gemini-embedding-2  (requires GOOGLE_API_KEY)
  qwen    — Local Qwen/Qwen3-VL-Embedding-2B (no API key, ~4 GB download)

Usage
-----
    # Gemini backend (default DB: chroma_db_multimodal_google)
    python vector_db/search_library.py --model gemini --text "books about birds"

    # Qwen backend (default DB: chroma_db_multimodal_qwen)
    python vector_db/search_library.py --model qwen --text "books about birds"

    # Override DB path explicitly
    python vector_db/search_library.py --model qwen --db-path vector_db/my_db --text "..."

    # Image query
    python vector_db/search_library.py --model qwen --image path/to/cover.png

    # Combined
    python vector_db/search_library.py --model gemini --text "..." --image path/to/cover.png -k 10

Run from the project root so the default DB paths resolve correctly.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# Config — DB paths per model (used as argparse defaults)
# ----------------------------------------------------------------------

DB_PATH_BY_MODEL = {
    "gemini": "vector_db/chroma_db_multimodal_google",
    "qwen":   "vector_db/chroma_db_multimodal_qwen",
    "jina":   "vector_db/chroma_db_multimodal_jina",
}

TEXT_COLLECTION  = "library_books_text"
IMAGE_COLLECTION = "library_books_image"
MAX_IMAGE_DIM    = 512  # mirrors RAGConfig.MAX_UPLOAD_IMAGE_DIMENSION

# Model names per backend
GEMINI_MODEL_NAME = "gemini-embedding-2"
QWEN_MODEL_NAME   = "Qwen/Qwen3-VL-Embedding-2B"
JINA_MODEL_NAME   = "jinaai/jina-embeddings-v5-omni-small"


# ----------------------------------------------------------------------
# Environment / API key loading
# ----------------------------------------------------------------------

def _load_env_files() -> None:
    """Load GOOGLE_API_KEY from any nearby .env file.

    Search order (first existing wins): script dir, parent (project root),
    parent's RAG folder. dotenv is optional — if it isn't installed we
    fall back to a tiny manual parser.
    """
    here = Path(__file__).resolve().parent
    candidates = [here / ".env", here.parent / ".env", here.parent / "RAG" / ".env",
                  here.parent / "embedding" / ".env"]

    try:
        from dotenv import load_dotenv  # type: ignore
        for path in candidates:
            if path.exists():
                load_dotenv(path, override=False)
        return
    except ImportError:
        pass

    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


# ----------------------------------------------------------------------
# Gemini embedding helpers
# ----------------------------------------------------------------------

def _gemini_embed_text(client, text: str) -> List[float]:
    """Embed a text query via Gemini."""
    response = client.models.embed_content(
        model=GEMINI_MODEL_NAME,
        contents=text,
    )
    return list(response.embeddings[0].values)


def _gemini_embed_image(client, image_path: Path) -> List[float]:
    """Embed an image via Gemini.

    Preprocessing: RGB → thumbnail 512px (LANCZOS) → JPEG q=85 → typed Part.
    Mirrors `_preprocess_and_embed_image` in rag_engine.py exactly.
    """
    from PIL import Image
    from google.genai import types

    img = Image.open(image_path).convert("RGB")
    if max(img.size) > MAX_IMAGE_DIM:
        img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    jpeg_bytes = buf.getvalue()

    part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
    response = client.models.embed_content(
        model=GEMINI_MODEL_NAME,
        contents=part,
    )
    return list(response.embeddings[0].values)


# ----------------------------------------------------------------------
# Qwen embedding helpers
# ----------------------------------------------------------------------

def _load_qwen_model():
    """Load the Qwen sentence-transformers model (cached after first run)."""
    return _load_local_model(QWEN_MODEL_NAME)


def _load_local_model(model_name: str):
    """Load any sentence-transformers model by name (works for Qwen, Jina, etc.)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.",
              "  conda install -c conda-forge sentence-transformers", file=sys.stderr)
        sys.exit(1)

    # Jina CLIP v2 compat: clip_loss was removed in transformers >= 4.49
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

    print(f"Loading model: {model_name}  (uses local cache if already downloaded)")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print(f"  model ready — dim={model.get_sentence_embedding_dimension()}")
    return model



def _qwen_embed_text(model, text: str) -> List[float]:
    """Embed a text query via Qwen (local)."""
    embeddings = model.encode(
        [text],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings[0].tolist()


def _qwen_embed_image(model, image_path: Path) -> List[float]:
    """Embed an image via Qwen (local).

    Preprocessing mirrors create_image_embeddings_qwen.py:
    RGB → thumbnail 512px (LANCZOS) — PIL Image passed directly to model.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    if max(img.size) > MAX_IMAGE_DIM:
        img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)

    embeddings = model.encode(
        [img],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings[0].tolist()


# ----------------------------------------------------------------------
# Title extraction — handles both collections' shapes
# ----------------------------------------------------------------------

_TITLE_RE = re.compile(r"TITLE:\s*([^\n]+)")


def _title_from_record(metadata: Optional[Dict[str, Any]],
                       document: Optional[str]) -> str:
    """Best-effort title resolution.

    Image collection records carry `title` directly. Text collection
    records embed it inside the document body as `TITLE: <value>`. Some
    records may also stash it in metadata.
    """
    if isinstance(metadata, dict) and metadata.get("title"):
        return str(metadata["title"]).strip()
    if document:
        m = _TITLE_RE.search(document)
        if m:
            return m.group(1).strip()
        # Fall back to the first non-empty line of the document.
        for line in document.split("\n"):
            line = line.strip()
            if line:
                return line[:120]
    if isinstance(metadata, dict) and metadata.get("book_id"):
        return f"<no title — book_id={metadata['book_id']}>"
    return "<unknown title>"


# ----------------------------------------------------------------------
# Search + print
# ----------------------------------------------------------------------

def _query_and_print(label: str, collection, embedding: List[float],
                     k: int) -> None:
    print(f"\n[{label}]  collection={collection.name}  count={collection.count():,}")
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    docs  = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    if not docs and not metas:
        print("  (no results)")
        return

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        title = _title_from_record(meta, doc)
        sim   = 1 - dist
        bid   = (meta or {}).get("book_id", "?")
        print(f"  {rank:>2}. {title}    [sim={sim:.3f}, book_id={bid}]")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the multimodal ChromaDB with a text query and/or a "
            "cover image. Supports --model gemini (Google API), "
            "--model qwen (local Qwen3-VL-Embedding), or "
            "--model jina (local Jina CLIP v2)."
        )
    )
    parser.add_argument(
        "--model", "-m",
        choices=["gemini", "qwen", "jina"],
        default="gemini",
        help="Embedding backend: 'gemini' (default), 'qwen', or 'jina'.",
    )
    parser.add_argument("--text",  "-t", default=None,
                        help="Text query.")
    parser.add_argument("--image", "-i", default=None,
                        help="Path to a cover image (jpg/png/webp/...).")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Top-K results per collection (default 5).")
    parser.add_argument("--db-path", default=None,
                        help=(
                            "ChromaDB path. Defaults to "
                            f"{DB_PATH_BY_MODEL['gemini']} for gemini, "
                            f"{DB_PATH_BY_MODEL['qwen']} for qwen, or "
                            f"{DB_PATH_BY_MODEL['jina']} for jina."
                        ))
    args = parser.parse_args()

    if not args.text and not args.image:
        parser.error("Provide --text and/or --image.")

    if args.image and not Path(args.image).exists():
        parser.error(f"--image not found: {args.image}")

    # Resolve DB path
    db_path = args.db_path or DB_PATH_BY_MODEL[args.model]
    if not Path(db_path).exists():
        print(f"ERROR: ChromaDB not found at {db_path}. "
              "Run from the project root, or check --db-path.", file=sys.stderr)
        return 1

    model_labels = {'gemini': 'Google GenAI', 'qwen': 'local Qwen', 'jina': 'local Jina CLIP v2'}
    print(f"Model:   {args.model}  ({model_labels[args.model]})")
    print(f"DB path: {db_path}")

    # ChromaDB
    try:
        import chromadb
    except ImportError:
        print("ERROR: chromadb not installed. pip install chromadb", file=sys.stderr)
        return 1

    chroma    = chromadb.PersistentClient(path=db_path)
    text_col  = chroma.get_collection(name=TEXT_COLLECTION)
    image_col = chroma.get_collection(name=IMAGE_COLLECTION)

    # ---- Set up embedding backend ----
    if args.model == "gemini":
        _load_env_files()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: GOOGLE_API_KEY not set in env or any .env file.",
                  file=sys.stderr)
            return 1
        try:
            from google import genai
        except ImportError:
            print("ERROR: google-genai not installed. pip install google-genai",
                  file=sys.stderr)
            return 1
        client = genai.Client(api_key=api_key)
        embed_text  = lambda text:  _gemini_embed_text(client, text)
        embed_image = lambda path:  _gemini_embed_image(client, path)

    elif args.model == "qwen":
        qwen_model  = _load_qwen_model()
        embed_text  = lambda text:  _qwen_embed_text(qwen_model, text)
        embed_image = lambda path:  _qwen_embed_image(qwen_model, path)

    else:  # jina
        # Jina Embeddings v5 Omni Small — Qwen3-based, no clip_loss patch needed.
        # SentenceTransformer with modality='vision' supports text + image in
        # one shared vector space.  encode_query() for queries, encode_document()
        # for stored passages/images.
        from sentence_transformers import SentenceTransformer as _ST
        import numpy as _np
        print(f"Loading model: {JINA_MODEL_NAME}  (uses local cache if already downloaded)")
        jina_model = _ST(
            JINA_MODEL_NAME,
            trust_remote_code=True,
            model_kwargs={"modality": "vision", "default_task": "retrieval"},
        )
        print(f"  model ready — dim={jina_model.get_sentence_embedding_dimension()}")

        def _jina_embed_text(text: str) -> List[float]:
            # encode() with task='retrieval.query' — works for text and images.
            emb = jina_model.encode(
                [text],
                task="retrieval",
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return emb[0].tolist()

        def _jina_embed_image(path: Path) -> List[float]:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            if max(img.size) > MAX_IMAGE_DIM:
                img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)
            # encode() with task='retrieval.query' for image search queries
            emb = jina_model.encode(
                [img],
                task="retrieval",
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return emb[0].tolist()

        embed_text  = _jina_embed_text
        embed_image = _jina_embed_image

    # --- Text branch ---
    if args.text:
        print(f"\n=== Text query: {args.text!r} ===")
        try:
            t_emb = embed_text(args.text)
        except Exception as e:
            print(f"  text embedding failed: {e}")
        else:
            print(f"  text embedding dim={len(t_emb)}")
            _query_and_print("text->TEXT",  text_col,  t_emb, args.top_k)
            _query_and_print("text->IMAGE", image_col, t_emb, args.top_k)

    # --- Image branch ---
    if args.image:
        print(f"\n=== Image query: {args.image} ===")
        try:
            i_emb = embed_image(Path(args.image))
        except Exception as e:
            print(f"  image embedding failed: {e}")
        else:
            print(f"  image embedding dim={len(i_emb)}")
            _query_and_print("image->TEXT",  text_col,  i_emb, args.top_k)
            _query_and_print("image->IMAGE", image_col, i_emb, args.top_k)

    return 0


if __name__ == "__main__":
    sys.exit(main())
