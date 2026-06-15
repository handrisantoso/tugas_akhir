#!/usr/bin/env python3
"""
Library Chatbot — Text Embedding Generation (OpenRouter)
Generates embeddings for book TEXT chunks using OpenRouter's embedding API.

Part of Approach A (Separate Embeddings):
  - This script handles TEXT embeddings  (book_chunks_text.json)
  - A companion script handles IMAGE embeddings (book_chunks_image.json)
  - Both are linked by a shared book_id in the vector DB

Uses the openai package with OpenRouter's base URL for compatibility.
"""

import json
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import sys
from tqdm import tqdm

# Add the parent directory to path to import from other folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openai import OpenAI
except ImportError:
    print("❌ Error: openai package not installed")
    print("Please install: pip install openai")
    sys.exit(1)

# ======================================================================
# Configuration
# ======================================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# File paths (relative to project root)
INPUT_FILE = "chunking/book_chunks_text.json"
OUTPUT_FILE = "embedding/text_embeddings_openrouter.json"
STATS_FILE = "embedding/text_embedding_statistics.txt"


# ======================================================================
# Embedding Generator
# ======================================================================

class TextEmbeddingGenerator:
    def __init__(self, api_key: str = None, model_name: str = "openai/text-embedding-3-small"):
        """Initialize the text embedding generator.

        Args:
            api_key: OpenRouter API key.
            model_name: Model identifier on OpenRouter. Common options:
                - "openai/text-embedding-3-small"  (1536 dims, cheap & fast)
                - "openai/text-embedding-3-large"  (3072 dims, higher quality)
                - "openai/text-embedding-ada-002"   (1536 dims, legacy)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_dim = None  # Detected from test call

        # Statistics
        self.stats = {
            'total_chunks': 0,
            'processed_chunks': 0,
            'failed_chunks': 0,
            'skipped_chunks': 0,
            'processing_start': None,
            'processing_end': None,
            'api_calls': 0,
            'total_tokens': 0,
            'avg_chunk_length': 0,
            'embedding_dimensions': 0,
            'rate_limit_hits': 0
        }

        # Rate limiting
        self.last_api_call = 0
        self.min_delay = 0.01  # 10ms between API calls

        if api_key:
            self.configure_api(api_key)

    # ------------------------------------------------------------------
    # API setup
    # ------------------------------------------------------------------

    def configure_api(self, api_key: str):
        """Configure the OpenRouter API using the openai package."""
        try:
            self.api_key = api_key
            self.client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=api_key,
            )
            print(f"✅ OpenRouter API configured (model: {self.model_name})")

            # Verify with a test call
            self.test_api_connection()

        except Exception as e:
            print(f"❌ Error configuring API: {e}")
            raise

    def test_api_connection(self):
        """Test the API connection with a simple embedding."""
        try:
            print("🧪 Testing API connection...")

            # First, try via the OpenAI SDK
            embedding = self._embed_via_sdk("test connection")

            if embedding is None:
                # Fallback: try a raw HTTP request (some models like
                # gemini-embedding-2-preview may not parse correctly
                # through the OpenAI SDK)
                print("🔄 SDK returned no data — trying raw HTTP fallback...")
                embedding = self._embed_via_http("test connection")

            if embedding:
                self.embedding_dim = len(embedding)
                # Remember which path worked so we use it later
                print(f"✅ API connection successful — Embedding dimension: {self.embedding_dim}")
            else:
                raise Exception("No embedding data received from either SDK or HTTP fallback")

        except Exception as e:
            print(f"❌ API connection test failed: {e}")
            raise

    def _embed_via_sdk(self, text: str) -> Optional[List[float]]:
        """Try generating an embedding via the OpenAI SDK."""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )

            if response and response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                if response.usage:
                    print(f"📊 Tokens used: {response.usage.total_tokens}")
                self._use_http_fallback = False
                return embedding

            # Debug: show what we actually got back
            print(f"⚠️  SDK response had no data. response.data = {response.data}")
            return None

        except Exception as e:
            print(f"⚠️  SDK call failed: {e}")
            return None

    def _embed_via_http(self, text: str) -> Optional[List[float]]:
        """Fallback: call the OpenRouter embeddings API directly via HTTP.
        Some models (e.g. google/gemini-embedding-2-preview) may return
        data that the OpenAI SDK cannot parse, but the raw JSON is fine.
        """
        import requests

        url = f"{OPENROUTER_BASE_URL}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": text,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            body = resp.json()

            # Standard OpenAI-compatible format
            data = body.get("data", [])
            if data and "embedding" in data[0]:
                self._use_http_fallback = True
                usage = body.get("usage", {})
                if usage:
                    print(f"📊 Tokens used: {usage.get('total_tokens', '?')}")
                return data[0]["embedding"]

            # Debug: print the raw response so we can see what went wrong
            print(f"⚠️  Unexpected HTTP response structure:")
            # Truncate to avoid flooding the console
            print(f"    Keys: {list(body.keys())}")
            if "error" in body:
                print(f"    Error: {body['error']}")
            return None

        except Exception as e:
            print(f"⚠️  HTTP fallback failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def rate_limit_delay(self):
        """Enforce minimum delay between API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call

        if time_since_last_call < self.min_delay:
            time.sleep(self.min_delay - time_since_last_call)

        self.last_api_call = time.time()

    def generate_single_embedding(self, text: str, retry_count: int = 3) -> Optional[List[float]]:
        """Generate embedding for a single text string with retry logic.

        Automatically uses the raw HTTP fallback if the SDK didn't work
        during the initial connection test.
        """
        use_http = getattr(self, '_use_http_fallback', False)

        for attempt in range(retry_count):
            try:
                self.rate_limit_delay()

                # Pick the path that worked during test_api_connection
                if use_http:
                    embedding = self._embed_via_http(text)
                else:
                    embedding = self._embed_via_sdk(text)
                    # If SDK returns None, try HTTP as last resort
                    if embedding is None:
                        embedding = self._embed_via_http(text)

                self.stats['api_calls'] += 1

                if embedding:
                    # Rough token tracking (exact count is printed by helpers)
                    self.stats['total_tokens'] += len(text.split())

                    # Sanity check
                    if self.embedding_dim and len(embedding) != self.embedding_dim:
                        raise Exception(f"Unexpected embedding dimension: {len(embedding)}")

                    return embedding
                else:
                    raise Exception("No embedding returned")

            except Exception as e:
                error_msg = str(e)
                print(f"⚠️  Attempt {attempt + 1} failed: {error_msg}")

                if "429" in error_msg or "rate" in error_msg.lower():
                    self.stats['rate_limit_hits'] += 1
                    wait_time = (attempt + 1) * 5
                    print(f"💤 Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif attempt < retry_count - 1:
                    time.sleep(1)
                else:
                    print(f"❌ Failed after {retry_count} attempts")
                    raise

        return None

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def load_chunks(self, input_file: str) -> List[Dict[str, Any]]:
        """Load book text chunks from JSON file."""
        print(f"📖 Loading text chunks from {input_file}...")

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            # Validate chunk structure
            if chunks and 'chunk_type' in chunks[0]:
                text_chunks = [c for c in chunks if c.get('chunk_type') == 'text']
                if len(text_chunks) < len(chunks):
                    print(f"⚠️  Filtered to {len(text_chunks)} text chunks "
                          f"(skipped {len(chunks) - len(text_chunks)} non-text chunks)")
                    chunks = text_chunks

            self.stats['total_chunks'] = len(chunks)

            # Stats
            chunk_lengths = [len(chunk['text']) for chunk in chunks]
            self.stats['avg_chunk_length'] = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0

            print(f"✅ Loaded {len(chunks):,} text chunks")
            print(f"📊 Average chunk length: {self.stats['avg_chunk_length']:.0f} characters")

            return chunks

        except Exception as e:
            print(f"❌ Error loading chunks: {e}")
            raise

    def save_embeddings(self, embedded_chunks: List[Dict[str, Any]], output_file: str):
        """Save embeddings to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")
            raise

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    def process_chunks(self, chunks: List[Dict[str, Any]], output_file: str,
                       resume_from: int = 0, save_interval: int = 100):
        """Process all text chunks and generate embeddings.

        Each output entry preserves the book_id for linking with image
        embeddings in the vector DB.
        """
        print(f"\n🚀 Starting text embedding generation for {len(chunks):,} chunks...")
        print(f"📝 Output file: {output_file}")

        if resume_from > 0:
            print(f"🔄 Resuming from chunk {resume_from}")

        self.stats['processing_start'] = datetime.now()

        # Load existing results when resuming
        embedded_chunks = []
        if resume_from > 0 and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    embedded_chunks = json.load(f)
                print(f"📁 Loaded {len(embedded_chunks):,} existing embeddings")
            except Exception:
                print("⚠️  Could not load existing embeddings, starting fresh")
                embedded_chunks = []

        # Process with progress bar
        with tqdm(total=len(chunks), initial=resume_from,
                  desc="Generating text embeddings") as pbar:

            for i, chunk in enumerate(chunks[resume_from:], start=resume_from):

                book_id = chunk.get('book_id', f'unknown_{i}')

                try:
                    embedding = self.generate_single_embedding(chunk['text'])

                    if embedding:
                        embedded_chunk = {
                            'book_id': book_id,
                            'chunk_type': 'text',
                            'text': chunk['text'],
                            'embedding': embedding,
                            'metadata': {
                                **chunk.get('metadata', {}),
                                'embedding_model': self.model_name,
                                'embedding_dim': len(embedding),
                                'processed_at': datetime.now().isoformat()
                            }
                        }

                        embedded_chunks.append(embedded_chunk)
                        self.stats['processed_chunks'] += 1

                        # Periodic save
                        if (i + 1) % save_interval == 0:
                            self.save_embeddings(embedded_chunks, output_file)
                            print(f"\n💾 Progress saved: {i + 1}/{len(chunks)} chunks")
                    else:
                        print(f"❌ No embedding returned for book_id={book_id}")
                        self.stats['failed_chunks'] += 1

                except Exception as e:
                    print(f"❌ Error on chunk {i} (book_id={book_id}): {e}")
                    self.stats['failed_chunks'] += 1

                pbar.update(1)
                processed_so_far = i - resume_from + 1
                if processed_so_far > 0:
                    success_rate = (self.stats['processed_chunks'] / processed_so_far) * 100
                    pbar.set_description(f"Text embeddings ({success_rate:.1f}% success)")

        # Final save
        self.save_embeddings(embedded_chunks, output_file)
        self.stats['processing_end'] = datetime.now()

        print(f"\n✅ Text embedding generation completed!")
        print(f"📊 Processed: {self.stats['processed_chunks']:,}/{self.stats['total_chunks']:,}")
        print(f"❌ Failed: {self.stats['failed_chunks']:,}")

        return embedded_chunks

    # ------------------------------------------------------------------
    # Reporting & validation
    # ------------------------------------------------------------------

    def generate_statistics_report(self, output_file: str = None):
        """Generate detailed statistics report."""
        output_file = output_file or STATS_FILE

        duration = self.stats['processing_end'] - self.stats['processing_start']

        report = []
        report.append("LIBRARY CHATBOT — TEXT EMBEDDING STATISTICS (OpenRouter)")
        report.append("=" * 60)
        report.append(f"Processing Date: {self.stats['processing_start'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Processing Time: {duration}")
        report.append("")

        report.append("PROCESSING SUMMARY:")
        report.append(f"  Total text chunks:        {self.stats['total_chunks']:,}")
        report.append(f"  Successfully embedded:    {self.stats['processed_chunks']:,}")
        report.append(f"  Failed:                   {self.stats['failed_chunks']:,}")

        if self.stats['total_chunks'] > 0:
            success_rate = (self.stats['processed_chunks'] / self.stats['total_chunks']) * 100
            report.append(f"  Success rate:             {success_rate:.1f}%")
        report.append("")

        report.append("API USAGE:")
        report.append(f"  Total API calls:          {self.stats['api_calls']:,}")
        report.append(f"  Rate limit hits:          {self.stats['rate_limit_hits']:,}")
        report.append(f"  Total tokens processed:   {self.stats['total_tokens']:,}")

        if self.stats['api_calls'] > 0:
            avg_tokens = self.stats['total_tokens'] / self.stats['api_calls']
            report.append(f"  Avg tokens per call:      {avg_tokens:.1f}")
        report.append("")

        report.append("EMBEDDING DETAILS:")
        report.append(f"  Model:                    {self.model_name}")
        report.append(f"  Dimensions:               {self.embedding_dim}")
        report.append(f"  Avg chunk length:         {self.stats['avg_chunk_length']:.0f} chars")
        report.append("")
        report.append("TEXT EMBEDDING GENERATION COMPLETE ✅")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print('\n'.join(report))

    def validate_embeddings(self, embedded_chunks: List[Dict[str, Any]]):
        """Validate the generated embeddings."""
        print("\n🔍 Validating text embeddings...")

        if not embedded_chunks:
            print("❌ No embeddings to validate")
            return False

        # Check consistent dimensions
        embedding_dims = [len(c['embedding']) for c in embedded_chunks]
        unique_dims = set(embedding_dims)

        if len(unique_dims) == 1:
            print(f"✅ All embeddings have consistent dimension: {list(unique_dims)[0]}")
        else:
            print(f"⚠️  Inconsistent dimensions found: {unique_dims}")

        # Null check
        null_count = sum(1 for c in embedded_chunks if not c['embedding'])
        print(f"📊 Null embeddings: {null_count}")

        # Check book_id presence
        missing_ids = sum(1 for c in embedded_chunks if not c.get('book_id'))
        if missing_ids:
            print(f"⚠️  {missing_ids} chunks missing book_id (needed to link with image embeddings)")
        else:
            print(f"✅ All {len(embedded_chunks):,} chunks have a book_id for cross-modal linking")

        # Sample stats
        sample = np.array(embedded_chunks[0]['embedding'])
        print(f"📊 Sample embedding stats:")
        print(f"   Mean: {np.mean(sample):.6f}")
        print(f"   Std:  {np.std(sample):.6f}")
        print(f"   Min:  {np.min(sample):.6f}")
        print(f"   Max:  {np.max(sample):.6f}")

        return True


# ======================================================================
# Main
# ======================================================================

def main():
    """Main entry point — interactive CLI."""

    print("🚀 Library Chatbot — Text Embedding Generator (OpenRouter)")
    print("=" * 60)
    print("Part of Approach A: this generates TEXT embeddings only.")
    print("Image embeddings are handled by a separate companion script.")
    print()

    # ---- API key ----
    api_key = input("🔑 Enter your OpenRouter API key: ").strip()
    if not api_key:
        print("❌ API key is required")
        return

    # ---- Model selection ----
    print("\n📋 Available embedding models on OpenRouter:")
    print("  1. openai/text-embedding-3-small      (1536 dims, cheap & fast)")
    print("  2. openai/text-embedding-3-large      (3072 dims, higher quality)")
    print("  3. openai/text-embedding-ada-002       (1536 dims, legacy)")
    print("  4. google/gemini-embedding-001         (3072 dims, Google)")
    print("  5. google/gemini-embedding-2-preview   (3072 dims, Google latest)")
    print("  6. Custom model ID")

    model_choice = input("\n🔧 Select model (1-6, default: 1): ").strip()

    model_map = {
        "1": "openai/text-embedding-3-small",
        "2": "openai/text-embedding-3-large",
        "3": "openai/text-embedding-ada-002",
        "4": "google/gemini-embedding-001",
        "5": "google/gemini-embedding-2-preview",
        "":  "openai/text-embedding-3-small",
    }

    if model_choice == "6":
        model_name = input("Enter model ID: ").strip()
    else:
        model_name = model_map.get(model_choice, "openai/text-embedding-3-small")

    print(f"\n📌 Using model: {model_name}")

    # ---- Initialise ----
    try:
        generator = TextEmbeddingGenerator(api_key, model_name=model_name)
    except Exception as e:
        print(f"❌ Failed to initialise: {e}")
        return

    # ---- Load chunks ----
    try:
        chunks = generator.load_chunks(INPUT_FILE)
    except Exception as e:
        print(f"❌ Failed to load chunks: {e}")
        return

    # ---- Resume handling ----
    resume_from = 0
    if os.path.exists(OUTPUT_FILE):
        resume_choice = input(
            "📁 Existing embeddings found. Resume from where left off? (y/n): "
        ).strip().lower()
        if resume_choice == 'y':
            try:
                with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                resume_from = len(existing)
                print(f"🔄 Will resume from chunk {resume_from}")
            except Exception:
                print("⚠️  Could not read existing file, starting fresh")

    # ---- Generate embeddings ----
    try:
        embedded_chunks = generator.process_chunks(chunks, OUTPUT_FILE, resume_from)

        generator.validate_embeddings(embedded_chunks)
        generator.generate_statistics_report()

        print(f"\n🎉 Successfully generated text embeddings for {len(embedded_chunks):,} chunks!")
        print(f"📁 Output:     {OUTPUT_FILE}")
        print(f"📊 Statistics: {STATS_FILE}")
        print(f"\n💡 Next step: run the image embedding script for cover images")

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted — partial results have been saved and can be resumed")
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")


if __name__ == "__main__":
    main()
