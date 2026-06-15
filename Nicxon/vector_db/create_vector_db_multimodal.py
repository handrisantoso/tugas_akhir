#!/usr/bin/env python3
"""
Library Chatbot — Multimodal Vector Database Setup
Creates and populates ChromaDB with BOTH text and image embeddings
from the OpenRouter embedding pipeline (Approach A).

Two collections:
  - library_books_text   → text embeddings   (318 chunks)
  - library_books_image  → image embeddings  (48 chunks)

Both are linked by book_id for cross-modal retrieval.

Query workflow:
  1. User sends a text query
  2. Generate a query embedding using the SAME model (gemini-embedding-2-preview)
  3. Search the text collection for semantic matches
  4. Optionally search the image collection for visual matches
  5. Combine results by book_id
"""

import json
import os
import sys
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import chromadb
    from chromadb.config import Settings
    import numpy as np
except ImportError:
    print("❌ Error: Required libraries not installed")
    print("Please install: pip install chromadb numpy tqdm")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("❌ Error: requests package not installed")
    print("Please install: pip install requests")
    sys.exit(1)


# ======================================================================
# Configuration
# ======================================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Input files (relative to project root)
TEXT_EMBEDDINGS_FILE = "embeddings_text/text_embeddings_jina.json"
IMAGE_EMBEDDINGS_FILE = "embeddings_image/image_embeddings_jina.json"

# Database
DB_PATH = "vector_db/chroma_db_multimodal_jina"
TEXT_COLLECTION_NAME = "library_books_text"
IMAGE_COLLECTION_NAME = "library_books_image"


# ======================================================================
# Multimodal Vector Database
# ======================================================================

class MultimodalLibraryVectorDB:
    def __init__(self, db_path: str = DB_PATH,
                 openrouter_api_key: str = None,
                 embedding_model: str = "google/gemini-embedding-2-preview"):
        """Initialize the multimodal vector database.

        Args:
            db_path: Path where ChromaDB stores its data on disk.
            openrouter_api_key: OpenRouter API key for generating query
                embeddings at search time.
            embedding_model: Must match the model used to generate the
                stored text/image embeddings.
        """
        self.db_path = db_path
        self.client = None
        self.text_collection = None
        self.image_collection = None
        self.openrouter_api_key = openrouter_api_key
        self.embedding_model = embedding_model

        # Statistics
        self.stats = {
            'text_total': 0,
            'text_processed': 0,
            'text_failed': 0,
            'image_total': 0,
            'image_processed': 0,
            'image_failed': 0,
            'processing_start': None,
            'processing_end': None,
        }

        self.setup_database()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def setup_database(self):
        """Initialize ChromaDB client and both collections."""
        try:
            print("🗄️  Setting up ChromaDB (multimodal)...")

            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # --- Text collection ---
            self.text_collection = self._get_or_create_collection(
                TEXT_COLLECTION_NAME,
                "Text embeddings for semantic book search"
            )

            # --- Image collection ---
            self.image_collection = self._get_or_create_collection(
                IMAGE_COLLECTION_NAME,
                "Image embeddings for visual book search"
            )

            print(f"✅ Database ready at {self.db_path}")

        except Exception as e:
            print(f"❌ Error setting up database: {e}")
            raise

    def _get_or_create_collection(self, name: str, description: str):
        """Get an existing collection or create a new one.

        Both collections are created with `hnsw:space=cosine`. The
        previous default was Chroma's squared-L2 distance, which —
        because the stored embeddings are unit-normalised — produced a
        displayed similarity of `2*cos - 1` rather than the cosine
        itself. Switching to cosine makes the `1 - distance` formula
        used in `RAG/rag_engine.py` yield the actual cosine similarity,
        so UI numbers match the conventional [-1, 1] cosine range and
        moderately-related hits no longer look like 0.1.

        Note: ranking is unchanged. Squared L2 and cosine are
        monotonically related on unit vectors, so the same books
        surface in the same order — only the magnitude of the score
        shifts.
        """
        # Chroma fixes the distance space at collection-creation time;
        # there is no way to mutate it later. Keeping the metadata in a
        # named local makes both create paths use the same value.
        creation_metadata = {
            "description": description,
            "hnsw:space": "cosine",
        }
        try:
            collection = self.client.get_collection(name)
            existing = collection.count()
            print(f"  📁 Collection '{name}' exists with {existing:,} documents")

            if existing > 0:
                reset = input(f"  🔄 Reset '{name}'? (y/n): ").strip().lower()
                if reset == 'y':
                    self.client.delete_collection(name)
                    collection = self.client.create_collection(
                        name=name,
                        embedding_function=None,
                        metadata=creation_metadata,
                    )
                    print(f"  ✅ Collection '{name}' reset (cosine space)")
                else:
                    print(f"  📁 Keeping existing '{name}'")

            return collection

        except Exception:
            collection = self.client.create_collection(
                name=name,
                embedding_function=None,
                metadata=creation_metadata,
            )
            print(f"  ✅ New collection '{name}' created (cosine space)")
            return collection

    # ------------------------------------------------------------------
    # Query embedding generation (for search time)
    # ------------------------------------------------------------------

    def generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate a query embedding via OpenRouter using the same model
        that produced the stored embeddings."""
        if not self.openrouter_api_key:
            raise Exception("OpenRouter API key required for query embeddings")

        url = f"{OPENROUTER_BASE_URL}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.embedding_model,
            "input": query_text,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            body = resp.json()

            data = body.get("data", [])
            if data and "embedding" in data[0]:
                return data[0]["embedding"]

            raise Exception(f"Unexpected response: {list(body.keys())}")

        except Exception as e:
            print(f"❌ Error generating query embedding: {e}")
            raise

    # ------------------------------------------------------------------
    # Metadata extraction (from text chunks)
    # ------------------------------------------------------------------

    # Comprehensive language mapping (same as original create_vector_db.py)
    LANGUAGE_MAP = {
        'english': 'English', 'eng': 'English',
        'spanish': 'Spanish', 'spa': 'Spanish',
        'french': 'French', 'fre': 'French', 'fra': 'French',
        'german': 'German', 'ger': 'German', 'deu': 'German',
        'hindi': 'Hindi', 'hin': 'Hindi',
        'italian': 'Italian', 'ita': 'Italian',
        'portuguese': 'Portuguese', 'por': 'Portuguese',
        'russian': 'Russian', 'rus': 'Russian',
        'chinese': 'Chinese', 'chi': 'Chinese', 'zho': 'Chinese',
        'japanese': 'Japanese', 'jpn': 'Japanese',
        'korean': 'Korean', 'kor': 'Korean',
        'arabic': 'Arabic', 'ara': 'Arabic',
        'dutch': 'Dutch', 'dut': 'Dutch', 'nld': 'Dutch',
        'swedish': 'Swedish', 'swe': 'Swedish',
        'norwegian': 'Norwegian', 'nor': 'Norwegian',
        'danish': 'Danish', 'dan': 'Danish',
        'finnish': 'Finnish', 'fin': 'Finnish',
        'polish': 'Polish', 'pol': 'Polish',
        'czech': 'Czech', 'cze': 'Czech', 'ces': 'Czech',
        'hungarian': 'Hungarian', 'hun': 'Hungarian',
        'greek': 'Greek', 'gre': 'Greek', 'ell': 'Greek',
        'hebrew': 'Hebrew', 'heb': 'Hebrew',
        'turkish': 'Turkish', 'tur': 'Turkish',
        'thai': 'Thai', 'tha': 'Thai',
        'vietnamese': 'Vietnamese', 'vie': 'Vietnamese',
        'indonesian': 'Indonesian', 'ind': 'Indonesian',
        'malay': 'Malay', 'may': 'Malay', 'msa': 'Malay',
        'persian': 'Persian', 'per': 'Persian', 'fas': 'Persian',
        'latin': 'Latin', 'lat': 'Latin',
        'multiple': 'Multiple Languages', 'multilingual': 'Multiple Languages',
        'undetermined': 'Undetermined', 'und': 'Undetermined',
    }

    def extract_text_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalise metadata from a text chunk for ChromaDB filtering."""
        text = chunk.get('text', '')
        orig_meta = chunk.get('metadata', {})

        metadata = {
            'book_id': chunk.get('book_id', ''),
            'chunk_type': 'text',
            'has_description': orig_meta.get('has_description', False),
            'has_toc': orig_meta.get('has_toc', False),
            'has_cover_image': orig_meta.get('has_cover_image', False),
            'field_count': orig_meta.get('field_count', 0),
        }

        # Extract language
        if 'LANGUAGE:' in text:
            lang_match = re.search(r'LANGUAGE:\s*([^\n]+)', text)
            if lang_match:
                raw = lang_match.group(1).strip()
                if raw.startswith('{') and 'key' in raw:
                    key_match = re.search(r"'key':\s*'([^']+)'", raw)
                    if key_match:
                        lang_key = key_match.group(1).lower().strip()
                        metadata['language'] = self.LANGUAGE_MAP.get(lang_key, lang_key.title())
                else:
                    clean = raw.lower().strip()
                    metadata['language'] = self.LANGUAGE_MAP.get(clean, raw.title())

        # Extract publication year
        pub_match = re.search(r'Published:\s*(\d{4})', text)
        if pub_match:
            metadata['publish_year'] = int(pub_match.group(1))

        # Extract page count
        pages_match = re.search(r'Pages:\s*(\d+)', text)
        if pages_match:
            metadata['page_count'] = int(pages_match.group(1))

        # Extract format
        format_match = re.search(r'Format:\s*([^;,\n]+)', text)
        if format_match:
            metadata['format'] = format_match.group(1).strip()

        # Extract subjects
        subjects_match = re.search(r'SUBJECTS & TOPICS:\s*([^\n]+)', text)
        if subjects_match:
            metadata['subjects_text'] = subjects_match.group(1)

        # Extract author
        author_match = re.search(r'AUTHOR\(S\):\s*([^\n]+)', text)
        if author_match:
            metadata['author'] = author_match.group(1).strip()

        # Extract title
        title_match = re.search(r'TITLE:\s*([^\n]+)', text)
        if title_match:
            metadata['title'] = title_match.group(1).strip()

        return metadata

    def extract_image_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from an image chunk for ChromaDB filtering."""
        orig_meta = chunk.get('metadata', {})

        metadata = {
            'book_id': chunk.get('book_id', ''),
            'chunk_type': 'image',
            'title': orig_meta.get('title', ''),
            'authors': orig_meta.get('authors', ''),
            'image_path': chunk.get('image_path', ''),
        }

        return metadata

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_embeddings(self, filepath: str, label: str) -> List[Dict[str, Any]]:
        """Load embeddings from a JSON file."""
        print(f"📖 Loading {label} from {filepath}...")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✅ Loaded {len(data):,} {label}")
        return data

    # ------------------------------------------------------------------
    # Database population
    # ------------------------------------------------------------------

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all metadata values are ChromaDB-compatible types."""
        clean = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean

    def populate_text_collection(self, text_embeddings: List[Dict[str, Any]],
                                  batch_size: int = 100):
        """Populate the text collection."""
        print(f"\n🚀 Populating text collection with {len(text_embeddings):,} chunks...")

        total_batches = (len(text_embeddings) + batch_size - 1) // batch_size

        with tqdm(total=len(text_embeddings), desc="Adding text embeddings") as pbar:

            for batch_idx in range(total_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(text_embeddings))
                batch = text_embeddings[start:end]

                ids = []
                documents = []
                embeddings_list = []
                metadatas = []

                for chunk in batch:
                    try:
                        book_id = chunk.get('book_id', f'unknown_{start + len(ids)}')
                        ids.append(f"text_{book_id}")
                        documents.append(chunk['text'])
                        embeddings_list.append(chunk['embedding'])

                        meta = self.extract_text_metadata(chunk)
                        metadatas.append(self._clean_metadata(meta))
                        self.stats['text_processed'] += 1

                    except Exception as e:
                        print(f"  ⚠️  Error processing text chunk {chunk.get('book_id', '?')}: {e}")
                        self.stats['text_failed'] += 1

                if ids:
                    try:
                        self.text_collection.add(
                            ids=ids,
                            embeddings=embeddings_list,
                            documents=documents,
                            metadatas=metadatas
                        )
                    except Exception as e:
                        print(f"  ❌ Error adding text batch {batch_idx}: {e}")
                        self.stats['text_failed'] += len(batch)

                pbar.update(len(batch))

        print(f"✅ Text collection: {self.text_collection.count():,} documents")

    def populate_image_collection(self, image_embeddings: List[Dict[str, Any]],
                                   batch_size: int = 50):
        """Populate the image collection."""
        print(f"\n🚀 Populating image collection with {len(image_embeddings):,} chunks...")

        total_batches = (len(image_embeddings) + batch_size - 1) // batch_size

        with tqdm(total=len(image_embeddings), desc="Adding image embeddings") as pbar:

            for batch_idx in range(total_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(image_embeddings))
                batch = image_embeddings[start:end]

                ids = []
                documents = []
                embeddings_list = []
                metadatas = []

                for chunk in batch:
                    try:
                        book_id = chunk.get('book_id', f'unknown_{start + len(ids)}')
                        ids.append(f"image_{book_id}")

                        # ChromaDB requires a document string; use title as placeholder
                        meta_title = chunk.get('metadata', {}).get('title', book_id)
                        documents.append(f"[Cover Image] {meta_title}")

                        embeddings_list.append(chunk['embedding'])

                        meta = self.extract_image_metadata(chunk)
                        metadatas.append(self._clean_metadata(meta))
                        self.stats['image_processed'] += 1

                    except Exception as e:
                        print(f"  ⚠️  Error processing image chunk {chunk.get('book_id', '?')}: {e}")
                        self.stats['image_failed'] += 1

                if ids:
                    try:
                        self.image_collection.add(
                            ids=ids,
                            embeddings=embeddings_list,
                            documents=documents,
                            metadatas=metadatas
                        )
                    except Exception as e:
                        print(f"  ❌ Error adding image batch {batch_idx}: {e}")
                        self.stats['image_failed'] += len(batch)

                pbar.update(len(batch))

        print(f"✅ Image collection: {self.image_collection.count():,} documents")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_text(self, query: str, n_results: int = 5,
                    where: Dict = None) -> Dict[str, Any]:
        """Search the text collection with a text query."""
        query_embedding = self.generate_query_embedding(query)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ['documents', 'metadatas', 'distances'],
        }
        if where:
            kwargs["where"] = where

        return self.text_collection.query(**kwargs)

    def search_image(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search the image collection with a text query.

        Because text and images share the same embedding space
        (gemini-embedding-2-preview), a plain text query can retrieve
        visually relevant book covers.
        """
        query_embedding = self.generate_query_embedding(query)

        return self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'],
        )

    def search_multimodal(self, query: str, n_text: int = 5,
                          n_image: int = 3) -> Dict[str, Any]:
        """Combined cross-modal search: text + image results merged by book_id."""
        text_results = self.search_text(query, n_results=n_text)
        image_results = self.search_image(query, n_results=n_image)

        # Merge by book_id
        combined = {}

        if text_results['documents'] and text_results['documents'][0]:
            for doc, meta, dist in zip(
                text_results['documents'][0],
                text_results['metadatas'][0],
                text_results['distances'][0]
            ):
                bid = meta.get('book_id', '')
                combined[bid] = {
                    'book_id': bid,
                    'title': meta.get('title', 'Unknown'),
                    'text_distance': dist,
                    'text_similarity': 1 - dist,
                    'text_snippet': doc[:200],
                    'text_metadata': meta,
                    'image_distance': None,
                    'image_similarity': None,
                    'image_path': None,
                }

        if image_results['documents'] and image_results['documents'][0]:
            for doc, meta, dist in zip(
                image_results['documents'][0],
                image_results['metadatas'][0],
                image_results['distances'][0]
            ):
                bid = meta.get('book_id', '')
                if bid in combined:
                    combined[bid]['image_distance'] = dist
                    combined[bid]['image_similarity'] = 1 - dist
                    combined[bid]['image_path'] = meta.get('image_path', '')
                else:
                    combined[bid] = {
                        'book_id': bid,
                        'title': meta.get('title', 'Unknown'),
                        'text_distance': None,
                        'text_similarity': None,
                        'text_snippet': None,
                        'text_metadata': None,
                        'image_distance': dist,
                        'image_similarity': 1 - dist,
                        'image_path': meta.get('image_path', ''),
                    }

        return combined

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def test_search(self, test_queries: List[str] = None, n_results: int = 5):
        """Test the search functionality across both collections."""

        if not test_queries:
            test_queries = [
                "mystery novels",
                "space exploration science fiction",
                "romance books",
                "books about history",
                "fantasy adventure stories"
            ]

        print(f"\n🔍 Testing search with {len(test_queries)} queries...")

        if not self.openrouter_api_key:
            print("⚠️  Skipping — OpenRouter API key required for query embeddings")
            return

        for query in test_queries:
            print(f"\n📝 Query: '{query}'")

            try:
                # --- Text search ---
                text_results = self.search_text(query, n_results=n_results)

                if text_results['documents'] and text_results['documents'][0]:
                    print(f"  📚 Text results ({len(text_results['documents'][0])}):")
                    for i, (doc, meta, dist) in enumerate(zip(
                        text_results['documents'][0],
                        text_results['metadatas'][0],
                        text_results['distances'][0]
                    ), 1):
                        title = meta.get('title', 'Unknown')
                        print(f"    {i}. {title}  (similarity: {1-dist:.3f})")
                        if i >= 3:
                            break

                # --- Image search ---
                image_results = self.search_image(query, n_results=3)

                if image_results['documents'] and image_results['documents'][0]:
                    print(f"  🖼️  Image results ({len(image_results['documents'][0])}):")
                    for i, (doc, meta, dist) in enumerate(zip(
                        image_results['documents'][0],
                        image_results['metadatas'][0],
                        image_results['distances'][0]
                    ), 1):
                        title = meta.get('title', 'Unknown')
                        print(f"    {i}. {title}  (similarity: {1-dist:.3f})")
                        if i >= 3:
                            break

            except Exception as e:
                print(f"  ❌ Search error: {e}")

    def test_filtered_search(self):
        """Test search with metadata filters."""

        print(f"\n🎯 Testing filtered search...")

        if not self.openrouter_api_key:
            print("⚠️  Skipping — OpenRouter API key required")
            return

        filters = [
            {
                "description": "English books only",
                "where": {"language": "English"},
                "query": "adventure stories"
            },
            {
                "description": "Books with descriptions",
                "where": {"has_description": True},
                "query": "science fiction"
            },
        ]

        for test in filters:
            print(f"\n🔍 Filter: {test['description']}")
            print(f"📝 Query: '{test['query']}'")

            try:
                results = self.search_text(test['query'], n_results=3, where=test['where'])

                if results['documents'] and results['documents'][0]:
                    print(f"✅ Found {len(results['documents'][0])} results:")
                    for i, (doc, meta) in enumerate(
                        zip(results['documents'][0], results['metadatas'][0]), 1
                    ):
                        title = meta.get('title', 'Unknown')
                        lang = meta.get('language', '?')
                        print(f"  {i}. {title} ({lang})")
                else:
                    print("❌ No results found")

            except Exception as e:
                print(f"❌ Filtered search error: {e}")

    def test_multimodal_search(self):
        """Test the combined cross-modal search."""

        print(f"\n🌐 Testing cross-modal (multimodal) search...")

        if not self.openrouter_api_key:
            print("⚠️  Skipping — OpenRouter API key required")
            return

        query = "fantasy adventure with magic"
        print(f"📝 Query: '{query}'")

        try:
            combined = self.search_multimodal(query)

            if combined:
                print(f"✅ Found {len(combined)} unique books across both modalities:")
                for i, (bid, info) in enumerate(combined.items(), 1):
                    title = info['title']
                    text_sim = f"{info['text_similarity']:.3f}" if info['text_similarity'] else "—"
                    img_sim = f"{info['image_similarity']:.3f}" if info['image_similarity'] else "—"
                    has_img = "yes" if info['image_path'] else "no"
                    print(f"  {i}. {title}")
                    print(f"     Text sim: {text_sim} | Image sim: {img_sim} | Cover: {has_img}")
            else:
                print("❌ No results found")

        except Exception as e:
            print(f"❌ Multimodal search error: {e}")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def generate_statistics(self):
        """Generate and print database statistics."""

        print(f"\n📊 DATABASE STATISTICS")
        print("=" * 55)

        text_count = self.text_collection.count()
        image_count = self.image_collection.count()
        print(f"Text collection:   {text_count:,} documents")
        print(f"Image collection:  {image_count:,} documents")

        # Database size on disk
        if os.path.exists(self.db_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.db_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            print(f"Database size:     {total_size / (1024*1024):.1f} MB")

        # Processing stats
        if self.stats['processing_start'] and self.stats['processing_end']:
            duration = self.stats['processing_end'] - self.stats['processing_start']
            print(f"Processing time:   {duration}")

        print(f"\nText:  {self.stats['text_processed']:,} ok, {self.stats['text_failed']:,} failed")
        print(f"Image: {self.stats['image_processed']:,} ok, {self.stats['image_failed']:,} failed")

        # Metadata analysis from text collection
        if text_count > 0:
            sample = self.text_collection.get(
                limit=min(100, text_count),
                include=['metadatas']
            )

            languages = {}
            has_desc = 0
            has_cover = 0

            for meta in sample['metadatas']:
                lang = meta.get('language', 'Unknown')
                languages[lang] = languages.get(lang, 0) + 1
                if meta.get('has_description'):
                    has_desc += 1
                if meta.get('has_cover_image'):
                    has_cover += 1

            n = len(sample['metadatas'])
            print(f"\nSample analysis ({n} docs):")
            print(f"  With description:  {has_desc} ({has_desc/n*100:.0f}%)")
            print(f"  With cover image:  {has_cover} ({has_cover/n*100:.0f}%)")

            print(f"\n  Top languages:")
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {lang}: {count}")


# ======================================================================
# Main
# ======================================================================

def main():
    """Main entry point — interactive CLI."""

    print("🗄️  Library Chatbot — Multimodal Vector Database Creator")
    print("=" * 55)
    print("Ingests BOTH text and image embeddings from OpenRouter.")
    print(f"  Text file:  {TEXT_EMBEDDINGS_FILE}")
    print(f"  Image file: {IMAGE_EMBEDDINGS_FILE}")
    print()

    # ---- API key (for search testing) ----
    api_key = input("🔑 Enter OpenRouter API key (for search tests, or press Enter to skip): ").strip()
    if not api_key:
        print("⚠️  No API key — search tests will be skipped")
        api_key = None

    # ---- Initialize ----
    try:
        db = MultimodalLibraryVectorDB(
            openrouter_api_key=api_key,
            embedding_model="google/gemini-embedding-2-preview"
        )
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    # ---- Load embeddings ----
    if not os.path.exists(TEXT_EMBEDDINGS_FILE):
        print(f"❌ Text embeddings not found: {TEXT_EMBEDDINGS_FILE}")
        return
    if not os.path.exists(IMAGE_EMBEDDINGS_FILE):
        print(f"❌ Image embeddings not found: {IMAGE_EMBEDDINGS_FILE}")
        return

    try:
        text_embeddings = db.load_embeddings(TEXT_EMBEDDINGS_FILE, "text embeddings")
        image_embeddings = db.load_embeddings(IMAGE_EMBEDDINGS_FILE, "image embeddings")
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        return

    db.stats['text_total'] = len(text_embeddings)
    db.stats['image_total'] = len(image_embeddings)

    # ---- Populate ----
    db.stats['processing_start'] = datetime.now()

    try:
        db.populate_text_collection(text_embeddings)
        db.populate_image_collection(image_embeddings)
    except Exception as e:
        print(f"❌ Failed to populate database: {e}")
        return

    db.stats['processing_end'] = datetime.now()

    # ---- Test ----
    print(f"\n{'='*55}")
    print("🧪 TESTING DATABASE FUNCTIONALITY")
    print(f"{'='*55}")

    db.test_search()
    db.test_filtered_search()
    db.test_multimodal_search()
    db.generate_statistics()

    print(f"\n🎉 Multimodal vector database setup complete!")
    print(f"📁 Database: {db.db_path}")
    print(f"📚 Text docs:  {db.text_collection.count():,}")
    print(f"🖼️  Image docs: {db.image_collection.count():,}")
    print(f"\n🚀 Ready for multimodal semantic search and RAG!")


if __name__ == "__main__":
    main()
