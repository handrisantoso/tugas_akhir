import json
import os
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, "books_english_only.json")

def main():
    print(f"Loading {JSON_PATH} ...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        books = json.load(f)

    total = len(books)
    print(f"Total entries: {total}\n")

    field_counts = Counter()

    for book in books:
        for key in book:
            field_counts[key] += 1

    # Sort by count descending
    sorted_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)

    # Print table
    max_name_len = max(len(name) for name, _ in sorted_fields)
    header_field = "Field".ljust(max_name_len)
    header_count = "Count".rjust(8)
    header_pct = "%".rjust(7)
    print(f"  {header_field}  {header_count}  {header_pct}")
    print(f"  {'─' * max_name_len}  {'─' * 8}  {'─' * 7}")

    for name, count in sorted_fields:
        pct = (count / total) * 100
        print(f"  {name.ljust(max_name_len)}  {str(count).rjust(8)}  {pct:6.2f}%")

    print(f"\n  Total unique fields: {len(sorted_fields)}")

if __name__ == "__main__":
    main()
