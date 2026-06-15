#!/usr/bin/env python3
"""
Filter readable books from OpenLibrary data
Based on Internet Archive integration and readability indicators
"""

import json
import csv
from urllib.parse import quote

def extract_readable_books(input_file='sampled_data.txt', output_file='readable_books.txt'):
    """Extract books that are readable online based on various indicators."""
    
    print('📚 Extracting readable books from sampled data...')
    print('=' * 60)
    
    readable_books = []
    total_processed = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            total_processed += 1
            
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    json_data = json.loads(parts[4])
                    
                    # Check for readability indicators
                    is_readable = False
                    readability_info = {}
                    
                    # 1. Check for OCAID (most reliable indicator)
                    if 'ocaid' in json_data:
                        is_readable = True
                        readability_info['source'] = 'Internet Archive (OCAID)'
                        readability_info['ocaid'] = json_data['ocaid']
                        readability_info['archive_url'] = f"https://archive.org/details/{json_data['ocaid']}"
                    
                    # 2. Check source_records for Internet Archive
                    elif 'source_records' in json_data:
                        for source in json_data['source_records']:
                            if isinstance(source, str) and source.startswith('ia:'):
                                is_readable = True
                                ia_id = source[3:]  # Remove 'ia:' prefix
                                readability_info['source'] = 'Internet Archive (Source)'
                                readability_info['ia_id'] = ia_id
                                readability_info['archive_url'] = f"https://archive.org/details/{ia_id}"
                                break
                    
                    # 3. Check for other IA indicators
                    elif 'ia_box_id' in json_data or 'ia_loaded_id' in json_data:
                        is_readable = True
                        readability_info['source'] = 'Internet Archive (Metadata)'
                        if 'ia_loaded_id' in json_data:
                            readability_info['ia_loaded_id'] = json_data['ia_loaded_id']
                            readability_info['archive_url'] = f"https://archive.org/details/{json_data['ia_loaded_id']}"
                    
                    if is_readable:
                        book_info = {
                            'ol_key': json_data.get('key', ''),
                            'title': json_data.get('title', 'Unknown Title'),
                            'subtitle': json_data.get('subtitle', ''),
                            'authors': [author.get('key', '') for author in json_data.get('authors', [])],
                            'publish_date': json_data.get('publish_date', ''),
                            'publishers': json_data.get('publishers', []),
                            'isbn_13': json_data.get('isbn_13', []),
                            'isbn_10': json_data.get('isbn_10', []),
                            'number_of_pages': json_data.get('number_of_pages', ''),
                            'languages': [lang.get('key', '') for lang in json_data.get('languages', [])],
                            'readability': readability_info,
                            'full_json': json_data  # Keep full data for reference
                        }
                        readable_books.append(book_info)
            
            except (json.JSONDecodeError, IndexError, KeyError) as e:
                continue
    
    # Write readable books to file
    print(f'✅ Found {len(readable_books)} readable books out of {total_processed} total records')
    print(f'📊 Readability rate: {(len(readable_books)/total_processed)*100:.1f}%')
    
    # Write detailed output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# READABLE BOOKS FROM OPENLIBRARY DATA\n")
        f.write(f"# Found {len(readable_books)} readable books\n")
        f.write("# Format: Original tab-separated line\n\n")
        
        for book in readable_books:
            # Write original line format
            parts = [
                "/type/edition",
                book['ol_key'],
                str(book['full_json'].get('revision', '')),
                book['full_json'].get('last_modified', {}).get('value', ''),
                json.dumps(book['full_json'])
            ]
            f.write('\t'.join(parts) + '\n')
    
    # Also create a CSV summary for easy viewing
    csv_file = output_file.replace('.txt', '_summary.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'OpenLibrary_Key', 'Title', 'Subtitle', 'Authors', 'Publish_Date', 
            'Publishers', 'ISBN_13', 'Pages', 'Readability_Source', 'Archive_URL'
        ])
        
        for book in readable_books:
            full_title = book['title']
            if book['subtitle']:
                full_title += f": {book['subtitle']}"
            
            writer.writerow([
                book['ol_key'],
                full_title,
                book['subtitle'],
                '; '.join(book['authors']) if book['authors'] else '',
                book['publish_date'],
                '; '.join(book['publishers']) if book['publishers'] else '',
                '; '.join(book['isbn_13']) if book['isbn_13'] else '',
                book['number_of_pages'],
                book['readability']['source'],
                book['readability'].get('archive_url', '')
            ])
    
    print(f'💾 Readable books saved to: {output_file}')
    print(f'📄 CSV summary saved to: {csv_file}')
    
    # Print sample of readable books
    print(f'\n📖 SAMPLE OF READABLE BOOKS:')
    for i, book in enumerate(readable_books[:5]):
        print(f'\n{i+1}. "{book["title"]}"')
        if book['subtitle']:
            print(f'   Subtitle: {book["subtitle"]}')
        print(f'   OpenLibrary: https://openlibrary.org{book["ol_key"]}')
        print(f'   Readable at: {book["readability"].get("archive_url", "N/A")}')
        print(f'   Source: {book["readability"]["source"]}')
    
    if len(readable_books) > 5:
        print(f'\n... and {len(readable_books) - 5} more readable books!')
    
    return readable_books

def get_readability_stats(readable_books):
    """Generate statistics about readable books."""
    
    print(f'\n📊 READABILITY STATISTICS:')
    print('=' * 40)
    
    # Count by source
    source_counts = {}
    for book in readable_books:
        source = book['readability']['source']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print('By readability source:')
    for source, count in source_counts.items():
        percentage = (count / len(readable_books)) * 100
        print(f'  {source}: {count} books ({percentage:.1f}%)')
    
    # Count by publication decade
    decade_counts = {}
    for book in readable_books:
        pub_date = book['publish_date']
        if pub_date and pub_date.strip():
            try:
                # Extract year from various date formats
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', pub_date)
                if year_match:
                    year = int(year_match.group())
                    decade = (year // 10) * 10
                    decade_counts[decade] = decade_counts.get(decade, 0) + 1
            except:
                pass
    
    if decade_counts:
        print('\nBy publication decade:')
        for decade in sorted(decade_counts.keys()):
            count = decade_counts[decade]
            percentage = (count / len(readable_books)) * 100
            print(f'  {decade}s: {count} books ({percentage:.1f}%)')

if __name__ == "__main__":
    readable_books = extract_readable_books()
    get_readability_stats(readable_books)
    
    print(f'\n🎯 CONCLUSION:')
    print(f'  Your OpenLibrary dataset contains books that ARE readable online!')
    print(f'  Look for books with "ocaid" field or Internet Archive source records.')
    print(f'  These can be read directly at archive.org using the provided URLs.') 