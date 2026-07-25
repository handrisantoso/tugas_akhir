#!/usr/bin/env python3

import json
import os
import time
from collections import defaultdict
from datetime import datetime


class BookChunker:
    def __init__(self):
        self.stats = {
            'total_books': 0,
            'text_chunks_created': 0,
            'image_chunks_created': 0,
            'books_without_cover': 0,
            'field_inclusion_counts': defaultdict(int),
            'chunk_lengths': [],
            'processing_start': None,
            'processing_end': None
        }

    def generate_book_id(self, book, index):
        isbn_13 = book.get('isbn_13')
        if isbn_13 and isinstance(isbn_13, list) and isbn_13[0]:
            return f"isbn13_{isbn_13[0]}"

        isbn_10 = book.get('isbn_10')
        if isbn_10 and isinstance(isbn_10, list) and isbn_10[0]:
            return f"isbn10_{isbn_10[0]}"

        oclc = book.get('oclc_numbers') or book.get('oclc_number')
        if oclc and isinstance(oclc, list) and oclc[0]:
            return f"oclc_{oclc[0]}"

        title = book.get('title', 'unknown')
        safe_title = "".join(c if c.isalnum() else '_' for c in title[:50]).strip('_')
        return f"book_{index}_{safe_title}"

    def safe_get_text(self, value):
        if not value:
            return ""

        if isinstance(value, str):
            return value.strip()
        elif isinstance(value, list):
            if not value:
                return ""
            if all(isinstance(item, str) for item in value):
                return ", ".join(value)
            elif all(isinstance(item, dict) for item in value):
                texts = []
                for item in value:
                    if 'name' in item:
                        texts.append(item['name'])
                    elif 'value' in item:
                        texts.append(str(item['value']))
                    else:
                        texts.append(str(item))
                return ", ".join(texts)
            else:
                return ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            if 'value' in value:
                return str(value['value'])
            elif 'name' in value:
                return str(value['name'])
            elif 'key' in value:
                key = value['key']
                if key == 'english':
                    return 'English'
                elif key == 'spanish':
                    return 'Spanish'
                elif key == 'french':
                    return 'French'
                elif key == 'german':
                    return 'German'
                else:
                    return key.replace('_', ' ').title()
            else:
                return str(value)
        else:
            return str(value)

    def format_table_of_contents(self, toc):
        if not toc:
            return ""

        if isinstance(toc, str):
            return toc.strip()

        if isinstance(toc, list):
            contents = []
            for item in toc:
                if isinstance(item, dict) and 'title' in item:
                    title = item['title']
                    if 'pagenum' in item:
                        contents.append(f"{title} (p. {item['pagenum']})")
                    else:
                        contents.append(title)
                elif isinstance(item, str):
                    contents.append(item)
            return "; ".join(contents) if contents else ""

        return ""

    def create_text_chunk(self, book, book_id):
        chunk_parts = []
        metadata = {
            'book_id': book_id,
            'chunk_type': 'text',
            'has_cover_image': book.get('has_cover_image', False),
            'has_description': book.get('has_desc', False),
            'has_toc': book.get('has_toc', False),
            'cover_image_path': book.get('cover_image_path', ''),
            'field_count': len(book.keys())
        }

        title = self.safe_get_text(book.get('title', ''))
        subtitle = self.safe_get_text(book.get('subtitle', ''))
        full_title = self.safe_get_text(book.get('full_title', ''))

        if full_title and full_title != title:
            title_text = f"TITLE: {full_title}"
        elif subtitle:
            title_text = f"TITLE: {title}: {subtitle}"
        else:
            title_text = f"TITLE: {title}"

        chunk_parts.append(title_text)
        self.stats['field_inclusion_counts']['title'] += 1
        metadata['title'] = title

        authors = self.safe_get_text(book.get('authors', ''))
        if authors:
            chunk_parts.append(f"AUTHOR(S): {authors}")
            self.stats['field_inclusion_counts']['authors'] += 1

        description = self.safe_get_text(book.get('description', ''))
        if description:
            chunk_parts.append(f"DESCRIPTION: {description}")
            self.stats['field_inclusion_counts']['description'] += 1

        topics = []

        subjects = self.safe_get_text(book.get('subjects', ''))
        if subjects:
            topics.append(f"Subjects: {subjects}")
            self.stats['field_inclusion_counts']['subjects'] += 1

        genres = self.safe_get_text(book.get('genres', ''))
        if genres:
            topics.append(f"Genres: {genres}")
            self.stats['field_inclusion_counts']['genres'] += 1

        subject_people = self.safe_get_text(book.get('subject_people', ''))
        if subject_people:
            topics.append(f"People: {subject_people}")
            self.stats['field_inclusion_counts']['subject_people'] += 1

        subject_places = self.safe_get_text(book.get('subject_places', '') or book.get('subject_place', ''))
        if subject_places:
            topics.append(f"Places: {subject_places}")
            self.stats['field_inclusion_counts']['subject_places'] += 1

        subject_times = self.safe_get_text(book.get('subject_times', '') or book.get('subject_time', ''))
        if subject_times:
            topics.append(f"Time periods: {subject_times}")
            self.stats['field_inclusion_counts']['subject_times'] += 1

        if topics:
            chunk_parts.append(f"SUBJECTS & TOPICS: {'; '.join(topics)}")

        pub_parts = []

        publishers = self.safe_get_text(book.get('publishers', ''))
        if publishers:
            pub_parts.append(f"Publisher: {publishers}")
            self.stats['field_inclusion_counts']['publishers'] += 1

        publish_date = self.safe_get_text(book.get('publish_date', ''))
        if publish_date:
            pub_parts.append(f"Published: {publish_date}")
            self.stats['field_inclusion_counts']['publish_date'] += 1

        publish_places = self.safe_get_text(book.get('publish_places', ''))
        if publish_places:
            pub_parts.append(f"Place: {publish_places}")
            self.stats['field_inclusion_counts']['publish_places'] += 1

        publish_country = self.safe_get_text(book.get('publish_country', ''))
        if publish_country:
            pub_parts.append(f"Country: {publish_country}")
            self.stats['field_inclusion_counts']['publish_country'] += 1

        copyright_date = self.safe_get_text(book.get('copyright_date', ''))
        if copyright_date and copyright_date != publish_date:
            pub_parts.append(f"Copyright: {copyright_date}")
            self.stats['field_inclusion_counts']['copyright_date'] += 1

        if pub_parts:
            chunk_parts.append(f"PUBLICATION: {'; '.join(pub_parts)}")

        format_parts = []

        physical_format = self.safe_get_text(book.get('physical_format', ''))
        if physical_format:
            format_parts.append(f"Format: {physical_format}")
            self.stats['field_inclusion_counts']['physical_format'] += 1

        pages = self.safe_get_text(book.get('number_of_pages', ''))
        if pages:
            format_parts.append(f"Pages: {pages}")
            self.stats['field_inclusion_counts']['number_of_pages'] += 1

        dimensions = self.safe_get_text(book.get('physical_dimensions', ''))
        if dimensions:
            format_parts.append(f"Dimensions: {dimensions}")
            self.stats['field_inclusion_counts']['physical_dimensions'] += 1

        weight = self.safe_get_text(book.get('weight', ''))
        if weight:
            format_parts.append(f"Weight: {weight}")
            self.stats['field_inclusion_counts']['weight'] += 1

        if format_parts:
            chunk_parts.append(f"FORMAT & DETAILS: {'; '.join(format_parts)}")

        languages = self.safe_get_text(book.get('languages', ''))
        if languages:
            chunk_parts.append(f"LANGUAGE: {languages}")
            self.stats['field_inclusion_counts']['languages'] += 1

        series = self.safe_get_text(book.get('series', ''))
        if series:
            chunk_parts.append(f"SERIES: {series}")
            self.stats['field_inclusion_counts']['series'] += 1

        content_parts = []

        first_sentence = self.safe_get_text(book.get('first_sentence', ''))
        if first_sentence:
            content_parts.append(f"Opening: {first_sentence}")
            self.stats['field_inclusion_counts']['first_sentence'] += 1

        toc = book.get('table_of_contents')
        if toc:
            toc_text = self.format_table_of_contents(toc)
            if toc_text:
                content_parts.append(f"Contents: {toc_text}")
                self.stats['field_inclusion_counts']['table_of_contents'] += 1

        if content_parts:
            chunk_parts.append(f"CONTENT PREVIEW: {'; '.join(content_parts)}")

        class_parts = []

        dewey = self.safe_get_text(book.get('dewey_decimal_class', ''))
        if dewey:
            class_parts.append(f"Dewey Decimal: {dewey}")
            self.stats['field_inclusion_counts']['dewey_decimal_class'] += 1

        lexile = self.safe_get_text(book.get('lexile', ''))
        if lexile:
            class_parts.append(f"Reading Level: {lexile}")
            self.stats['field_inclusion_counts']['lexile'] += 1

        if class_parts:
            chunk_parts.append(f"CLASSIFICATION: {'; '.join(class_parts)}")

        id_parts = []

        isbn_13 = self.safe_get_text(book.get('isbn_13', ''))
        if isbn_13:
            id_parts.append(f"ISBN-13: {isbn_13}")
            self.stats['field_inclusion_counts']['isbn_13'] += 1

        isbn_10 = self.safe_get_text(book.get('isbn_10', ''))
        if isbn_10:
            id_parts.append(f"ISBN-10: {isbn_10}")
            self.stats['field_inclusion_counts']['isbn_10'] += 1

        oclc = self.safe_get_text(book.get('oclc_numbers', '') or book.get('oclc_number', ''))
        if oclc:
            id_parts.append(f"OCLC: {oclc}")
            self.stats['field_inclusion_counts']['oclc_numbers'] += 1

        if id_parts:
            chunk_parts.append(f"IDENTIFIERS: {'; '.join(id_parts)}")

        notes = self.safe_get_text(book.get('notes', ''))
        if notes:
            chunk_parts.append(f"NOTES: {notes}")
            self.stats['field_inclusion_counts']['notes'] += 1

        chunk_text = "\n\n".join(chunk_parts)
        self.stats['chunk_lengths'].append(len(chunk_text))

        return {
            'chunk_type': 'text',
            'book_id': book_id,
            'text': chunk_text,
            'metadata': metadata
        }

    def create_image_chunk(self, book, book_id):
        if not book.get('has_cover_image', False):
            return None

        image_path = book.get('cover_image_path', '')
        if not image_path:
            return None

        if not os.path.exists(image_path):
            print(f"  ⚠ Cover image not found: {image_path}")
            return None

        title = self.safe_get_text(book.get('title', ''))
        authors = self.safe_get_text(book.get('authors', ''))

        return {
            'chunk_type': 'image',
            'book_id': book_id,
            'image_path': image_path,
            'metadata': {
                'book_id': book_id,
                'chunk_type': 'image',
                'title': title,
                'authors': authors,
            }
        }

    def process_all_books(self, input_file, text_output_file, image_output_file):
        print("Starting chunking process (Approach A: Separate Embeddings)...")
        print(f"  Input:        {input_file}")
        print(f"  Text output:  {text_output_file}")
        print(f"  Image output: {image_output_file}")

        self.stats['processing_start'] = datetime.now()

        print("\nLoading book data...")
        with open(input_file, 'r', encoding='utf-8') as f:
            books = json.load(f)

        self.stats['total_books'] = len(books)
        print(f"Loaded {len(books):,} books")

        print("Creating chunks...")
        text_chunks = []
        image_chunks = []

        for i, book in enumerate(books):
            if i % 1000 == 0:
                print(f"  Processed {i:,}/{len(books):,} books...")

            book_id = self.generate_book_id(book, i)

            text_chunk = self.create_text_chunk(book, book_id)
            text_chunks.append(text_chunk)
            self.stats['text_chunks_created'] += 1

            image_chunk = self.create_image_chunk(book, book_id)
            if image_chunk:
                image_chunks.append(image_chunk)
                self.stats['image_chunks_created'] += 1
            else:
                self.stats['books_without_cover'] += 1

        self.stats['processing_end'] = datetime.now()

        print(f"\nSaving {len(text_chunks):,} text chunks...")
        with open(text_output_file, 'w', encoding='utf-8') as f:
            json.dump(text_chunks, f, indent=2, ensure_ascii=False)
        print(f"  → {text_output_file}")

        print(f"Saving {len(image_chunks):,} image chunks...")
        with open(image_output_file, 'w', encoding='utf-8') as f:
            json.dump(image_chunks, f, indent=2, ensure_ascii=False)
        print(f"  → {image_output_file}")

        return text_chunks, image_chunks

    def generate_statistics_report(self, output_file):
        report = []
        report.append("LIBRARY CHATBOT CHUNKING STATISTICS (Approach A: Separate Embeddings)")
        report.append("=" * 65)
        report.append(f"Processing Date: {self.stats['processing_start'].strftime('%Y-%m-%d %H:%M:%S')}")

        duration = self.stats['processing_end'] - self.stats['processing_start']
        report.append(f"Processing Time: {duration}")
        report.append("")

        report.append("PROCESSING SUMMARY:")
        report.append(f"  Total books processed:   {self.stats['total_books']:,}")
        report.append(f"  Text chunks created:     {self.stats['text_chunks_created']:,}")
        report.append(f"  Image chunks created:    {self.stats['image_chunks_created']:,}")
        report.append(f"  Books without cover:     {self.stats['books_without_cover']:,}")

        if self.stats['total_books'] > 0:
            cover_pct = (self.stats['image_chunks_created'] / self.stats['total_books']) * 100
            report.append(f"  Cover image coverage:    {cover_pct:.1f}%")
        report.append("")

        report.append("TEXT CHUNK LENGTH STATISTICS:")
        if self.stats['chunk_lengths']:
            avg_length = sum(self.stats['chunk_lengths']) / len(self.stats['chunk_lengths'])
            min_length = min(self.stats['chunk_lengths'])
            max_length = max(self.stats['chunk_lengths'])

            report.append(f"  Average chunk length: {avg_length:.0f} characters")
            report.append(f"  Minimum chunk length: {min_length:,} characters")
            report.append(f"  Maximum chunk length: {max_length:,} characters")
        report.append("")

        report.append("FIELD INCLUSION STATISTICS:")
        report.append("(How many text chunks included each field)")
        total_chunks = self.stats['text_chunks_created']

        for field, count in sorted(self.stats['field_inclusion_counts'].items(),
                                   key=lambda x: x[1], reverse=True):
            percentage = (count / total_chunks) * 100 if total_chunks > 0 else 0
            report.append(f"  {field:25s}: {count:5d} chunks ({percentage:5.1f}%)")

        report.append("")
        report.append("CHUNKING PROCESS COMPLETE ✅")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print('\n'.join(report))

    def create_sample_chunks(self, text_chunks, image_chunks, output_file, num_samples=10):
        samples = []
        samples.append("LIBRARY CHATBOT CHUNK SAMPLES (Approach A: Separate Embeddings)")
        samples.append("=" * 65)
        samples.append(f"Showing up to {num_samples} sample text chunks + matching image chunks")
        samples.append("")

        for i in range(min(num_samples, len(text_chunks))):
            text_chunk = text_chunks[i]
            book_id = text_chunk['book_id']

            samples.append(f"{'=' * 80}")
            samples.append(f"SAMPLE #{i + 1} — Book ID: {book_id}")
            samples.append(f"{'=' * 80}")

            samples.append("")
            samples.append("[TEXT CHUNK]")
            samples.append("-" * 40)
            samples.append(text_chunk['text'])
            samples.append("")
            samples.append(f"Metadata: {text_chunk['metadata']}")

            matching_image = next((ic for ic in image_chunks if ic['book_id'] == book_id), None)
            samples.append("")
            if matching_image:
                samples.append("[IMAGE CHUNK]")
                samples.append("-" * 40)
                samples.append(f"Image path: {matching_image['image_path']}")
                samples.append(f"Metadata:   {matching_image['metadata']}")
            else:
                samples.append("[IMAGE CHUNK] — No cover image available")

            samples.append("")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(samples))

        print(f"Sample chunks saved to {output_file}")


def main():
    chunker = BookChunker()

    text_chunks, image_chunks = chunker.process_all_books(
        'cleaned_data/books_with_content.json',
        'chunking/book_chunks_text.json',
        'chunking/book_chunks_image.json'
    )

    chunker.generate_statistics_report('chunking/chunk_statistics.txt')

    chunker.create_sample_chunks(text_chunks, image_chunks, 'chunking/sample_chunks.txt')

    print("\n✅ Chunking process completed successfully!")
    print("\nOutput files created:")
    print("  - chunking/book_chunks_text.json    (text chunks for text embedding)")
    print("  - chunking/book_chunks_image.json   (image chunks for image embedding)")
    print("  - chunking/chunk_statistics.txt      (statistics report)")
    print("  - chunking/sample_chunks.txt         (quality review samples)")


if __name__ == "__main__":
    main()