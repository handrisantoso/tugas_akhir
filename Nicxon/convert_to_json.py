#!/usr/bin/env python3
"""
Script to safely convert readable_books.txt to proper JSON format.
The input file contains tab-separated values where the last column is JSON data.
"""

import json
import sys
from typing import List, Dict, Any, Optional


def parse_line(line: str, line_number: int) -> Optional[Dict[Any, Any]]:
    """
    Parse a single line and extract the JSON portion.
    
    Args:
        line: The input line to parse
        line_number: Line number for error reporting
        
    Returns:
        Parsed JSON object or None if parsing fails
    """
    line = line.strip()
    
    # Skip empty lines and comments
    if not line or line.startswith('#'):
        return None
    
    try:
        # Split by tabs - expecting at least 5 fields
        # Format: /type/edition	book_id	revision	timestamp	{json_data}
        parts = line.split('\t')
        
        if len(parts) < 5:
            print(f"Warning: Line {line_number} has insufficient fields ({len(parts)}), skipping")
            return None
        
        # The JSON data is everything from the 5th field onwards (in case JSON contains tabs)
        json_str = '\t'.join(parts[4:])
        
        # Attempt to parse the JSON
        json_obj = json.loads(json_str)
        
        # Optionally, add metadata from the tab-separated fields
        metadata = {
            'record_type': parts[0],
            'book_id': parts[1],
            'revision': int(parts[2]) if parts[2].isdigit() else parts[2],
            'last_modified': parts[3]
        }
        
        # Add metadata to the JSON object (optional - comment out if not needed)
        json_obj['_metadata'] = metadata
        
        return json_obj
        
    except json.JSONDecodeError as e:
        print(f"Warning: JSON parse error on line {line_number}: {e}")
        print(f"  Problematic JSON: {json_str[:100]}{'...' if len(json_str) > 100 else ''}")
        return None
    except Exception as e:
        print(f"Warning: Unexpected error on line {line_number}: {e}")
        return None


def convert_file_to_json(input_file: str, output_file: str = None, include_metadata: bool = True) -> bool:
    """
    Convert the tab-separated file to proper JSON format.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file (if None, prints to stdout)
        include_metadata: Whether to include tab-separated metadata in JSON objects
        
    Returns:
        True if conversion successful, False otherwise
    """
    books: List[Dict[Any, Any]] = []
    total_lines = 0
    processed_lines = 0
    error_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                total_lines += 1
                
                result = parse_line(line, line_number)
                if result is not None:
                    books.append(result)
                    processed_lines += 1
                elif line.strip() and not line.strip().startswith('#'):
                    error_count += 1
        
        print(f"Processing complete:")
        print(f"  Total lines: {total_lines}")
        print(f"  Successfully processed: {processed_lines}")
        print(f"  Errors/skipped: {error_count}")
        print(f"  Comments/empty lines: {total_lines - processed_lines - error_count}")
        
        # Create the final JSON structure
        output_data = {
            "metadata": {
                "total_books": len(books),
                "source_file": input_file,
                "conversion_stats": {
                    "total_lines": total_lines,
                    "processed_lines": processed_lines,
                    "error_count": error_count
                }
            },
            "books": books
        }
        
        # Output the JSON
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"JSON written to: {output_file}")
        else:
            print(json.dumps(output_data, indent=2, ensure_ascii=False))
            
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def convert_minimal_json(input_file: str, output_file: str = None) -> bool:
    """
    Convert to minimal JSON format (just the book objects in an array).
    
    Args:
        input_file: Path to input file
        output_file: Path to output file (if None, prints to stdout)
        
    Returns:
        True if conversion successful, False otherwise
    """
    books: List[Dict[Any, Any]] = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                try:
                    parts = line.split('\t')
                    if len(parts) >= 5:
                        json_str = '\t'.join(parts[4:])
                        json_obj = json.loads(json_str)
                        books.append(json_obj)
                except (json.JSONDecodeError, Exception):
                    # Silently skip problematic lines in minimal mode
                    continue
        
        # Output just the array of books
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(books, f, indent=2, ensure_ascii=False)
            print(f"Minimal JSON written to: {output_file}")
        else:
            print(json.dumps(books, indent=2, ensure_ascii=False))
            
        print(f"Converted {len(books)} books to JSON format")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python convert_to_json.py <input_file> [output_file] [--minimal]")
        print("  input_file: Path to the readable_books.txt file")
        print("  output_file: Optional output file (if not provided, prints to stdout)")
        print("  --minimal: Create minimal JSON (just array of books, no metadata)")
        return
    
    input_file = sys.argv[1]
    output_file = None
    minimal = False
    
    # Parse additional arguments
    for arg in sys.argv[2:]:
        if arg == '--minimal':
            minimal = True
        elif not arg.startswith('--'):
            output_file = arg
    
    # Perform conversion
    if minimal:
        success = convert_minimal_json(input_file, output_file)
    else:
        success = convert_file_to_json(input_file, output_file)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main() 