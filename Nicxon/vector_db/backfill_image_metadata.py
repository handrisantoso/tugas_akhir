#!/usr/bin/env python3
"""
Backfill metadata on the `library_books_image` ChromaDB collection.

Why this exists
---------------
The image collection was populated by `create_vector_db_multimodal.py`
with a minimal metadata schema (`book_id`, `chunk_type`, `title`,
`authors`, `image_path`). The query path in `RAG/rag_engine.py` builds
metadata filters (language, publish_year, page_count, format,
has_description, has_subjects, has_toc, format, ...) from the text
collection and only applies them to text queries — image queries
silently bypass every filter.

This one-shot script copies the filterable fields from each book's
text chunk metadata into the matching image record so that the same
`where` clause can be applied to both collections. It does NOT touch
embeddings — only metadata is mutated, so the paid OpenRouter
embeddings produced earlier are preserved as-is.

Notes
-----
- The text→image join uses `where={"book_id": <bid>}` (the canonical
  join key), not the `text_{book_id}` ID pattern, which is unreliable
  whenever a book has more than one text chunk.
- ChromaDB `update()` REPLACES the metadata dict for an ID; this script
  merges new fields onto the existing dict before writing back, so
  current image-side fields (image_path, title, authors, ...) survive.
- The script is idempotent: re-running it just overwrites previously
  copied fields with the same values.
- Run from the project root, e.g.:
      python vector_db/backfill_image_metadata.py
      python vector_db/backfill_image_metadata.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
except ImportError:
    print("ERROR: chromadb is not installed. Run: pip install chromadb")
    sys.exit(1)


# ----------------------------------------------------------------------
# Configuration — must match the rest of the project
# ----------------------------------------------------------------------

DB_PATH = "vector_db/chroma_db_multimodal_jina"
TEXT_COLLECTION_NAME = "library_books_text"
IMAGE_COLLECTION_NAME = "library_books_image"

# Fields to copy from a book's text metadata onto its image metadata.
# These are the keys the runtime where-clause builder in
# `RAG/rag_engine.py::_perform_vector_search` actually uses.
COPYABLE_FIELDS = (
    "language",
    "publish_year",
    "page_count",
    "format",
    "has_description",
    "has_subjects",
    "has_toc",
    "has_cover_image",
    "field_count",
    "author",
    "title",
    "subjects_text",
)

# Existing image-side fields we never want to clobber, even if the
# text record happens to have a same-named key.
IMAGE_PROTECTED_FIELDS = (
    "book_id",
    "chunk_type",
    "image_path",
    "authors",
)

UPDATE_BATCH_SIZE = 200


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def build_text_metadata_index(
    text_collection: "chromadb.api.models.Collection.Collection",
) -> Dict[str, Dict[str, Any]]:
    """Return a {book_id: text_metadata} map.

    If a book has multiple text chunks, the first one wins. That's fine
    here because all of the COPYABLE_FIELDS are book-level (language,
    publish year, etc.), not chunk-level.
    """
    print(f"Indexing text collection '{TEXT_COLLECTION_NAME}'...")

    # `get()` with no ids returns every record. We only need metadata.
    text_records = text_collection.get(include=["metadatas"])
    text_ids: List[str] = text_records.get("ids") or []
    text_metas: List[Dict[str, Any]] = text_records.get("metadatas") or []

    index: Dict[str, Dict[str, Any]] = {}
    skipped_no_book_id = 0

    for tid, meta in zip(text_ids, text_metas):
        if not isinstance(meta, dict):
            continue
        bid = meta.get("book_id")
        if not bid:
            skipped_no_book_id += 1
            continue
        # First-write-wins for duplicate book_ids
        if bid not in index:
            index[bid] = meta

    print(
        f"  Indexed {len(index):,} unique book_ids "
        f"(from {len(text_ids):,} text records, "
        f"{skipped_no_book_id:,} without book_id)."
    )
    return index


def merged_metadata(
    image_meta: Dict[str, Any],
    text_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge text fields onto the existing image metadata dict.

    Rules:
    - Start from the existing image metadata (so image_path, etc. survive).
    - Overlay COPYABLE_FIELDS from text_meta when present and non-None.
    - Never overwrite IMAGE_PROTECTED_FIELDS even if text has them.
    """
    merged: Dict[str, Any] = dict(image_meta or {})

    if not text_meta:
        return merged

    for field in COPYABLE_FIELDS:
        if field in IMAGE_PROTECTED_FIELDS:
            continue
        value = text_meta.get(field)
        # Skip None / "" so we don't replace a real value with nothing.
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        merged[field] = value

    return merged


def metadata_changed(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    """Cheap structural diff so we only push records that actually changed."""
    if before.keys() != after.keys():
        return True
    for k in after:
        if before.get(k) != after.get(k):
            return True
    return False


def chunked(seq: List[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy filterable book-level fields from `library_books_text` "
            "metadata onto matching `library_books_image` records."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change but do not write to ChromaDB.",
    )
    parser.add_argument(
        "--db-path",
        default=DB_PATH,
        help=f"Path to the ChromaDB store (default: {DB_PATH}).",
    )
    args = parser.parse_args()

    db_path = args.db_path
    if not Path(db_path).exists():
        print(f"ERROR: ChromaDB path not found: {db_path}")
        print("Run this script from the project root.")
        return 1

    print(f"Opening ChromaDB at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)

    try:
        text_collection = client.get_collection(name=TEXT_COLLECTION_NAME)
        image_collection = client.get_collection(name=IMAGE_COLLECTION_NAME)
    except Exception as e:
        print(f"ERROR: could not open collections: {e}")
        return 1

    print(
        f"  text:  {text_collection.count():,} records | "
        f"image: {image_collection.count():,} records"
    )

    # 1. Build {book_id: text_meta} lookup once.
    text_index = build_text_metadata_index(text_collection)

    # 2. Pull every image record (id + current metadata).
    print(f"\nReading image collection '{IMAGE_COLLECTION_NAME}'...")
    image_records = image_collection.get(include=["metadatas"])
    image_ids: List[str] = image_records.get("ids") or []
    image_metas: List[Dict[str, Any]] = image_records.get("metadatas") or []
    print(f"  Loaded {len(image_ids):,} image records.")

    # 3. Compute merged metadata per record, collect those that changed.
    pending_ids: List[str] = []
    pending_metas: List[Dict[str, Any]] = []

    matched = 0
    unmatched = 0
    no_change = 0

    for img_id, img_meta in zip(image_ids, image_metas):
        if not isinstance(img_meta, dict):
            img_meta = {}
        bid = img_meta.get("book_id")

        text_meta = text_index.get(bid) if bid else None
        if text_meta is None:
            unmatched += 1
        else:
            matched += 1

        new_meta = merged_metadata(img_meta, text_meta)

        if not metadata_changed(img_meta, new_meta):
            no_change += 1
            continue

        pending_ids.append(img_id)
        pending_metas.append(new_meta)

    print(
        "\nPlan:\n"
        f"  matched to text:    {matched:,}\n"
        f"  unmatched book_id:  {unmatched:,}\n"
        f"  already up-to-date: {no_change:,}\n"
        f"  to be updated:      {len(pending_ids):,}"
    )

    if not pending_ids:
        print("\nNothing to write. Done.")
        return 0

    # 4. Show a sample diff so the operator can sanity-check.
    print("\nSample of pending updates (up to 3):")
    for i, (iid, new_meta) in enumerate(
        zip(pending_ids[:3], pending_metas[:3]), start=1
    ):
        old_meta = image_metas[image_ids.index(iid)] or {}
        added = {
            k: v for k, v in new_meta.items() if old_meta.get(k) != v
        }
        print(f"  {i}. id={iid}")
        print(f"     book_id={new_meta.get('book_id', '?')}")
        print(f"     adds/updates: {added}")

    if args.dry_run:
        print("\n--dry-run set: no changes written.")
        return 0

    # 5. Push updates in batches.
    print(f"\nWriting {len(pending_ids):,} updates in batches of {UPDATE_BATCH_SIZE}...")
    written = 0
    for ids_batch, metas_batch in zip(
        chunked(pending_ids, UPDATE_BATCH_SIZE),
        chunked(pending_metas, UPDATE_BATCH_SIZE),
    ):
        try:
            image_collection.update(ids=ids_batch, metadatas=metas_batch)
            written += len(ids_batch)
            print(f"  wrote {written:,}/{len(pending_ids):,}")
        except Exception as e:
            print(f"  ERROR writing batch starting at {ids_batch[0]}: {e}")
            print("  Aborting; partial updates have been persisted.")
            return 1

    print(f"\nDone. Updated {written:,} image records.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
