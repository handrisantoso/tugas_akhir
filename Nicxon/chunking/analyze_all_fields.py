#!/usr/bin/env python3
"""
Script to analyze all fields across all book records
"""
import json
from collections import defaultdict

def analyze_all_fields():
    """Analyze all fields across all book records"""
    try:
        print("Loading and analyzing all book records...")
        
        with open('cleaned_data/books_with_content.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total books: {len(data)}")
        
        # Collect all unique fields and their frequency
        field_stats = defaultdict(int)
        field_samples = {}
        
        for book in data:
            for field in book.keys():
                field_stats[field] += 1
                if field not in field_samples:
                    value = book[field]
                    if isinstance(value, list) and value:
                        field_samples[field] = str(value[0])[:100]
                    elif isinstance(value, str):
                        field_samples[field] = value[:100]
                    else:
                        field_samples[field] = str(value)[:100]
        
        print(f"\nUnique fields found: {len(field_stats)}")
        print("\nField Analysis (sorted by frequency):")
        print("="*90)
        
        for field, count in sorted(field_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(data)) * 100
            sample = field_samples[field]
            if len(sample) == 100:
                sample += "..."
            print(f"{field:25s} | {count:5d} books ({percentage:5.1f}%) | {sample}")
        
        # Find a record with many fields for better chunking understanding
        max_fields = 0
        richest_book = None
        for book in data:
            if len(book.keys()) > max_fields:
                max_fields = len(book.keys())
                richest_book = book
        
        print(f"\nRichest record has {max_fields} fields:")
        print("="*90)
        print(json.dumps(richest_book, indent=2, ensure_ascii=False)[:1500] + "...")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_all_fields() 