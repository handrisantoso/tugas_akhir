#!/usr/bin/env python3
"""
Script to examine the structure of cleaned book data
"""
import json
import sys

def examine_data_structure():
    """Examine the structure of the cleaned book data"""
    try:
        print("Loading cleaned book data...")
        
        # Load the cleaned data with proper encoding
        with open('cleaned_data/books_updated_cleaned.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total books: {len(data)}")
        
        if data:
            # Get the fields from the first record
            sample_book = data[0]
            print(f"\nFields available ({len(sample_book.keys())}):")
            for i, field in enumerate(sorted(sample_book.keys()), 1):
                value = sample_book.get(field)
                value_type = type(value).__name__
                if isinstance(value, list) and value:
                    sample_val = str(value[0])[:50] + "..." if len(str(value[0])) > 50 else str(value[0])
                    print(f"{i:2d}. {field:25s} ({value_type}, {len(value)} items) - {sample_val}")
                elif isinstance(value, str):
                    sample_val = value[:50] + "..." if len(value) > 50 else value
                    print(f"{i:2d}. {field:25s} ({value_type}) - {sample_val}")
                else:
                    print(f"{i:2d}. {field:25s} ({value_type}) - {value}")
            
            print("\n" + "="*80)
            print("SAMPLE BOOK RECORD:")
            print("="*80)
            print(json.dumps(sample_book, indent=2, ensure_ascii=False)[:2000] + "...")
            
        else:
            print("No data found in the file!")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    examine_data_structure() 