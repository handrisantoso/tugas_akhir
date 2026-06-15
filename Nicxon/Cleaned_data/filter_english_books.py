"""
Filter books_updated_cleaned.json to keep only English-language books.

Criteria:
- Book must have a "languages" field
- At least one language entry must have {"key": "english"}
- Books with no languages field or only non-English languages are removed

Output: books_english_only.json in the same directory
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "books_updated_cleaned.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "books_english_only.json")


def is_english(book: dict) -> bool:
    """Check if a book has English as one of its languages."""
    languages = book.get("languages")
    if not languages:
        return False
    return any(
        lang.get("key", "").lower() == "english"
        for lang in languages
        if isinstance(lang, dict)
    )


def main():
    print(f"Loading books from: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        books = json.load(f)

    total = len(books)
    print(f"Total books loaded: {total:,}")

    english_books = [book for book in books if is_english(book)]
    filtered_out = total - len(english_books)

    print(f"English books kept: {len(english_books):,}")
    print(f"Non-English / no-language books removed: {filtered_out:,}")

    print(f"\nSaving to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(english_books, f, indent=2, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    main()
