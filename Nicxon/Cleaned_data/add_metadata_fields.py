"""
add_metadata_fields.py
──────────────────────
Adds / normalises metadata fields on every book entry in 'books_english_only.json'.

Fields handled:
  has_toc          – True if 'table_of_contents' exists, else False
  has_cover_image  – True if 'cover_image_path' exists, else False
  has_desc         – True if 'description' exists (after flattening), else False
  subjects         – Ensures every book has a 'subjects' list (empty [] if missing)
  description      – Flattens {"type":…, "value":…} dicts into plain strings;
                     leaves existing strings untouched
"""

import json
import os

# ── Configuration ──────────────────────────────────────────────────────────
JSON_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "books_english_only.json")


def flatten_description(book: dict) -> None:
    """
    If 'description' is an OpenLibrary-style dict like
        {"type": "/type/text", "value": "actual text…"}
    replace it with just the string value.
    """
    desc = book.get("description")
    if isinstance(desc, dict):
        book["description"] = desc.get("value", "")


def main():
    # Load
    print(f"Loading: {JSON_FILE}")
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        books = json.load(f)

    total = len(books)
    print(f"Total books: {total}")

    updated_count = 0
    stats = {
        "has_toc": 0,
        "has_cover_image": 0,
        "has_desc": 0,
        "has_subjects": 0,
        "desc_flattened": 0,
        "subjects_added": 0,
    }

    for book in books:
        changed = False

        # ── Flatten description dict → string ──────────────────────────
        if isinstance(book.get("description"), dict):
            flatten_description(book)
            stats["desc_flattened"] += 1
            changed = True

        # ── Ensure 'subjects' list exists ──────────────────────────────
        if "subjects" not in book:
            book["subjects"] = []
            stats["subjects_added"] += 1
            changed = True

        # ── has_toc ────────────────────────────────────────────────────
        should_have_toc = bool(book.get("table_of_contents"))
        if book.get("has_toc") != should_have_toc:
            book["has_toc"] = should_have_toc
            changed = True
        if should_have_toc:
            stats["has_toc"] += 1

        # ── has_cover_image ────────────────────────────────────────────
        should_have_cover = bool(book.get("cover_image_path"))
        if book.get("has_cover_image") != should_have_cover:
            book["has_cover_image"] = should_have_cover
            changed = True
        if should_have_cover:
            stats["has_cover_image"] += 1

        # ── has_desc ───────────────────────────────────────────────────
        desc = book.get("description")
        should_have_desc = bool(desc and isinstance(desc, str) and desc.strip())
        if book.get("has_desc") != should_have_desc:
            book["has_desc"] = should_have_desc
            changed = True
        if should_have_desc:
            stats["has_desc"] += 1

        # ── Track subjects ─────────────────────────────────────────────
        if book.get("subjects"):
            stats["has_subjects"] += 1

        if changed:
            updated_count += 1

    # Save
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(books, f, indent=2, ensure_ascii=False)

    print(f"\nDone!  ({updated_count} books modified)")
    print(f"  Books with TOC:              {stats['has_toc']}")
    print(f"  Books with cover image:      {stats['has_cover_image']}")
    print(f"  Books with description:      {stats['has_desc']}")
    print(f"  Books with subjects:         {stats['has_subjects']}")
    print(f"  Descriptions flattened:      {stats['desc_flattened']}")
    print(f"  Empty subjects list added:   {stats['subjects_added']}")


if __name__ == "__main__":
    main()
