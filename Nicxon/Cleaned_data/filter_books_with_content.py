"""
Filter books_english_only.json to keep only books that have at least one of:
  - Cover image (has_cover_image == True)
  - Description (has_desc == True)
  - Table of contents (has_toc == True)

Outputs: books_with_content.json (in the same folder)
"""

import json
import os

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "books_english_only.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "books_with_content.json")


def has_content(book: dict) -> bool:
    """Return True if the book has a cover, description, or table of contents."""
    return (
        book.get("has_cover_image", False)
        or book.get("has_desc", False)
        or book.get("has_toc", False)
    )


def main():
    print(f"Loading books from: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        books = json.load(f)

    total = len(books)
    print(f"Total books loaded: {total:,}")

    filtered = [book for book in books if has_content(book)]
    kept = len(filtered)

    print(f"\nBooks with cover, description, or TOC: {kept:,}")
    print(f"Books removed (no content):            {total - kept:,}")

    # --- Breakdown ---
    with_cover = sum(1 for b in filtered if b.get("has_cover_image", False))
    with_desc  = sum(1 for b in filtered if b.get("has_desc", False))
    with_toc   = sum(1 for b in filtered if b.get("has_toc", False))
    with_all   = sum(
        1 for b in filtered
        if b.get("has_cover_image", False)
        and b.get("has_desc", False)
        and b.get("has_toc", False)
    )

    print(f"\n--- Breakdown (among kept books) ---")
    print(f"  With cover image : {with_cover:,}")
    print(f"  With description : {with_desc:,}")
    print(f"  With TOC         : {with_toc:,}")
    print(f"  With all three   : {with_all:,}")

    # --- Write output ---
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"\nFiltered books saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
