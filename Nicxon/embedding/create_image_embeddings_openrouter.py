#!/usr/bin/env python3
"""
Library Chatbot — Image Embedding Generation (OpenRouter)
Generates embeddings for book cover images using OpenRouter's embedding API.

Part of Approach A (Separate Embeddings):
  - Companion script handles TEXT embeddings  (via OpenRouter)
  - This script handles IMAGE embeddings      (via OpenRouter)
  - Both linked by book_id in the vector DB

Requires:
  - An OpenRouter API key
  - pip install openai requests Pillow numpy tqdm
"""

import json
import time
import base64
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

try:
    import numpy as np
except ImportError:
    print("❌ Error: numpy package not installed")
    print("Please install: pip install numpy")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("❌ Error: requests package not installed")
    print("Please install: pip install requests")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("❌ Error: Pillow package not installed")
    print("Please install: pip install Pillow")
    sys.exit(1)

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
INPUT_FILE = "chunking/book_chunks_image.json"
OUTPUT_FILE = "embedding/image_embeddings_openrouter.json"
STATS_FILE = "embedding/image_embedding_statistics_openrouter.txt"

# Image processing limits
MAX_IMAGE_SIZE_MB = 4          # OpenRouter/Gemini limit is ~4MB for inline data
MAX_IMAGE_DIMENSION = 512      # Resize if larger (saves tokens & latency)
SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'}

# ======================================================================
# Image Embedding Generator
# ======================================================================

class ImageEmbeddingGeneratorOpenRouter:
    def __init__(self, api_key: str = None, model_name: str = "google/gemini-embedding-2-preview"):
        """Initialize the image embedding generator.

        Args:
            api_key: OpenRouter API key.
            model_name: Multimodal embedding model on OpenRouter.
                Defaults to "google/gemini-embedding-2-preview".
        """
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_dim = None
        self._use_http_fallback = False

        self.stats = {
            'total_chunks': 0,
            'processed_chunks': 0,
            'failed_chunks': 0,
            'skipped_missing': 0,
            'skipped_format': 0,
            'skipped_size': 0,
            'processing_start': None,
            'processing_end': None,
            'api_calls': 0,
            'rate_limit_hits': 0,
            'total_tokens': 0,
        }

        # Rate limiting
        self.last_api_call = 0
        self.min_delay = 0.5  # 500ms between calls — image embeddings are heavy

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
        """Test the API connection with a tiny synthetic image to verify
        the model accepts image input and to detect embedding dimensions."""
        print("🧪 Testing API connection with a test image...")

        try:
            # Create a tiny 8x8 test image
            test_img = Image.new('RGB', (8, 8), color=(128, 128, 128))
            
            import io
            buffer = io.BytesIO()
            test_img.save(buffer, format="PNG")
            b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            b64_data_url = f"data:image/png;base64,{b64_data}"

            # First, try via the OpenAI SDK
            embedding = self._embed_via_sdk(b64_data_url)

            if embedding is None:
                # Fallback: try raw HTTP request
                print("🔄 SDK returned no data — trying raw HTTP fallback...")
                embedding = self._embed_via_http(b64_data_url)

            if embedding:
                self.embedding_dim = len(embedding)
                print(f"✅ API connection successful — Embedding dimension: {self.embedding_dim}")
            else:
                raise Exception("No embedding data received for test image from either SDK or HTTP fallback")

        except Exception as e:
            print(f"❌ API connection test failed: {e}")
            raise

    def _embed_via_sdk(self, b64_data_url: str) -> Optional[List[float]]:
        """Try generating embedding via the OpenAI SDK."""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=b64_data_url
            )

            if response and response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                self._use_http_fallback = False
                
                # Capture token usage if available
                if getattr(response, 'usage', None):
                    tokens = response.usage.total_tokens
                    self.stats['total_tokens'] += tokens
                    print(f"   📊 Tokens used: {tokens}")
                    
                return embedding

            print(f"⚠️  SDK response had no data. response.data = {getattr(response, 'data', None)}")
            return None

        except Exception as e:
            print(f"⚠️  SDK call failed: {e}")
            return None

    def _embed_via_http(self, b64_data_url: str) -> Optional[List[float]]:
        """Call OpenRouter embeddings API directly via HTTP."""
        url = f"{OPENROUTER_BASE_URL}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": b64_data_url,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=45)
            
            if resp.status_code == 429:
                raise Exception("429 rate limit")
                
            resp.raise_for_status()
            body = resp.json()

            data = body.get("data", [])
            if data and "embedding" in data[0]:
                self._use_http_fallback = True
                
                # Capture token usage if available
                usage = body.get("usage", {})
                if usage:
                    tokens = usage.get("total_tokens", 0)
                    self.stats['total_tokens'] += tokens
                    print(f"   📊 Tokens used: {tokens}")
                    
                return data[0]["embedding"]

            print(f"⚠️  Unexpected HTTP response structure. Keys: {list(body.keys())}")
            if "error" in body:
                print(f"    Error: {body['error']}")
            return None

        except Exception as e:
            print(f"⚠️  HTTP fallback failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Image processing helpers
    # ------------------------------------------------------------------

    def _load_and_prepare_image(self, image_path: str) -> Optional[Image.Image]:
        """Load an image, validate it, and resize if necessary."""
        if not os.path.exists(image_path):
            return None

        ext = os.path.splitext(image_path)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            self.stats['skipped_format'] += 1
            return None

        # Check file size
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if file_size_mb > MAX_IMAGE_SIZE_MB:
            self.stats['skipped_size'] += 1
            print(f"  ⚠️  Image too large ({file_size_mb:.1f}MB): {os.path.basename(image_path)}")
            return None

        try:
            img = Image.open(image_path).convert('RGB')

            # Resize if too large (preserves aspect ratio)
            if max(img.size) > MAX_IMAGE_DIMENSION:
                img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)

            return img

        except Exception as e:
            print(f"  ⚠️  Could not load image: {e}")
            return None

    def _image_to_base64_data_url(self, image: Image.Image, ext: str) -> str:
        """Convert a PIL Image to a Base64 Data URL string.
        
        Always encodes as JPEG regardless of original format to minimize
        base64 payload size and reduce API token consumption.
        """
        import io

        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{b64_data}"

    # ------------------------------------------------------------------
    # Single image embedding with retry
    # ------------------------------------------------------------------

    def rate_limit_delay(self):
        """Enforce minimum delay between API calls."""
        current_time = time.time()
        elapsed = current_time - self.last_api_call

        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        self.last_api_call = time.time()

    def generate_single_embedding(self, image_path: str, retry_count: int = 3) -> Optional[List[float]]:
        """Generate embedding for a single image with retry logic."""

        image = self._load_and_prepare_image(image_path)
        if image is None:
            self.stats['skipped_missing'] += 1
            return None

        ext = os.path.splitext(image_path)[1]
        b64_data_url = self._image_to_base64_data_url(image, ext)

        use_http = getattr(self, '_use_http_fallback', False)

        for attempt in range(retry_count):
            try:
                self.rate_limit_delay()
                
                if use_http:
                    embedding = self._embed_via_http(b64_data_url)
                else:
                    embedding = self._embed_via_sdk(b64_data_url)
                    if embedding is None:
                        embedding = self._embed_via_http(b64_data_url)

                self.stats['api_calls'] += 1

                if embedding:
                    if self.embedding_dim and len(embedding) != self.embedding_dim:
                        raise Exception(f"Unexpected dimension: {len(embedding)}")
                    return embedding
                else:
                    raise Exception("No embedding returned")

            except Exception as e:
                error_msg = str(e)
                print(f"  ⚠️  Attempt {attempt + 1} failed: {error_msg}")

                if "429" in error_msg or "rate" in error_msg.lower():
                    self.stats['rate_limit_hits'] += 1
                    wait_time = (attempt + 1) * 10  # longer wait for images
                    print(f"  💤 Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif attempt < retry_count - 1:
                    time.sleep(2)
                else:
                    print(f"  ❌ Failed after {retry_count} attempts")
                    return None

        return None

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def load_chunks(self, input_file: str) -> List[Dict[str, Any]]:
        """Load image chunks from JSON file."""
        print(f"📖 Loading image chunks from {input_file}...")

        with open(input_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        # Filter to image chunks only (safety check)
        image_chunks = [c for c in chunks if c.get('chunk_type') == 'image']
        if len(image_chunks) < len(chunks):
            print(f"⚠️  Filtered to {len(image_chunks)} image chunks "
                  f"(skipped {len(chunks) - len(image_chunks)} non-image entries)")
            chunks = image_chunks

        self.stats['total_chunks'] = len(chunks)
        print(f"✅ Loaded {len(chunks):,} image chunks")

        # Quick check: how many images actually exist on disk?
        existing = sum(1 for c in chunks if os.path.exists(c.get('image_path', '')))
        print(f"📊 Images found on disk: {existing:,}/{len(chunks):,}")

        return chunks

    def save_embeddings(self, embedded_chunks: List[Dict[str, Any]], output_file: str):
        """Save embeddings to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving: {e}")
            raise

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    def process_chunks(self, chunks: List[Dict[str, Any]], output_file: str,
                       existing_book_ids: set = None, save_interval: int = 50):
        """Process all image chunks and generate embeddings.

        Filters out chunks that are already present in existing_book_ids.
        """
        existing_book_ids = existing_book_ids or set()
        
        # Filter chunks that are already embedded
        chunks_to_process = [c for c in chunks if c.get('book_id') not in existing_book_ids]

        print(f"\n🚀 Starting image embedding generation...")
        print(f"📊 Total image chunks: {len(chunks):,}")
        print(f"📁 Already embedded:    {len(existing_book_ids):,}")
        print(f"🔄 To be processed:    {len(chunks_to_process):,}")
        print(f"📝 Output file:        {output_file}")

        self.stats['processing_start'] = datetime.now()

        # Load existing results when resuming
        embedded_chunks = []
        if existing_book_ids and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    embedded_chunks = json.load(f)
                print(f"📁 Loaded {len(embedded_chunks):,} existing image embeddings")
            except Exception:
                print("⚠️  Could not load existing embeddings, starting fresh")
                embedded_chunks = []
                chunks_to_process = chunks

        # Process with progress bar
        with tqdm(total=len(chunks_to_process), desc="Generating image embeddings") as pbar:

            for i, chunk in enumerate(chunks_to_process):

                book_id = chunk.get('book_id', f'unknown_{i}')
                image_path = chunk.get('image_path', '')

                try:
                    embedding = self.generate_single_embedding(image_path)

                    if embedding:
                        embedded_chunk = {
                            'book_id': book_id,
                            'chunk_type': 'image',
                            'image_path': image_path,
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
                            print(f"\n💾 Progress saved: {len(embedded_chunks)}/{len(chunks)} images")
                    else:
                        self.stats['failed_chunks'] += 1

                except Exception as e:
                    print(f"  ❌ Error on image {i} (book_id={book_id}): {e}")
                    self.stats['failed_chunks'] += 1

                pbar.update(1)
                processed_so_far = i + 1
                if processed_so_far > 0:
                    success_rate = (self.stats['processed_chunks'] / processed_so_far) * 100
                    pbar.set_description(f"Image embeddings ({success_rate:.1f}% success)")

        # Final save
        self.save_embeddings(embedded_chunks, output_file)
        self.stats['processing_end'] = datetime.now()

        # Update stats to reflect the pre-existing ones
        self.stats['processed_chunks'] += len(existing_book_ids)

        print(f"\n✅ Image embedding generation completed!")
        print(f"📊 Processed: {self.stats['processed_chunks']:,}/{self.stats['total_chunks']:,}")
        print(f"❌ Failed: {self.stats['failed_chunks']:,}")
        print(f"⏭️  Skipped (missing file):  {self.stats['skipped_missing']:,}")
        print(f"⏭️  Skipped (bad format):    {self.stats['skipped_format']:,}")
        print(f"⏭️  Skipped (too large):     {self.stats['skipped_size']:,}")

        return embedded_chunks

    # ------------------------------------------------------------------
    # Reporting & validation
    # ------------------------------------------------------------------

    def generate_statistics_report(self, output_file: str = None):
        """Generate detailed statistics report."""
        output_file = output_file or STATS_FILE

        duration = self.stats['processing_end'] - self.stats['processing_start']

        report = []
        report.append("LIBRARY CHATBOT — IMAGE EMBEDDING STATISTICS (OpenRouter)")
        report.append("=" * 60)
        report.append(f"Processing Date: {self.stats['processing_start'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Processing Time: {duration}")
        report.append("")

        report.append("PROCESSING SUMMARY:")
        report.append(f"  Total image chunks:       {self.stats['total_chunks']:,}")
        report.append(f"  Successfully embedded:    {self.stats['processed_chunks']:,}")
        report.append(f"  Failed:                   {self.stats['failed_chunks']:,}")
        report.append(f"  Skipped (missing file):   {self.stats['skipped_missing']:,}")
        report.append(f"  Skipped (bad format):     {self.stats['skipped_format']:,}")
        report.append(f"  Skipped (too large):      {self.stats['skipped_size']:,}")

        if self.stats['total_chunks'] > 0:
            success_rate = (self.stats['processed_chunks'] / self.stats['total_chunks']) * 100
            report.append(f"  Success rate:             {success_rate:.1f}%")
        report.append("")

        report.append("API USAGE:")
        report.append(f"  Total API calls:          {self.stats['api_calls']:,}")
        report.append(f"  Rate limit hits:          {self.stats['rate_limit_hits']:,}")
        report.append(f"  Total tokens processed:   {self.stats['total_tokens']:,}")
        report.append("")

        report.append("EMBEDDING DETAILS:")
        report.append(f"  Model:                    {self.model_name}")
        report.append(f"  Dimensions:               {self.embedding_dim}")
        report.append("")
        report.append("IMAGE EMBEDDING GENERATION COMPLETE ✅")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print('\n'.join(report))

    def validate_embeddings(self, embedded_chunks: List[Dict[str, Any]]):
        """Validate the generated image embeddings."""
        print("\n🔍 Validating image embeddings...")

        if not embedded_chunks:
            print("❌ No embeddings to validate")
            return False

        # Dimension consistency
        dims = set(len(c['embedding']) for c in embedded_chunks)
        if len(dims) == 1:
            print(f"✅ All embeddings have consistent dimension: {list(dims)[0]}")
        else:
            print(f"⚠️  Inconsistent dimensions: {dims}")

        # Null check
        null_count = sum(1 for c in embedded_chunks if not c['embedding'])
        print(f"📊 Null embeddings: {null_count}")

        # book_id check
        missing_ids = sum(1 for c in embedded_chunks if not c.get('book_id'))
        if missing_ids:
            print(f"⚠️  {missing_ids} chunks missing book_id")
        else:
            print(f"✅ All {len(embedded_chunks):,} image embeddings have book_id for linking")

        # Sample stats
        sample = np.array(embedded_chunks[0]['embedding'])
        print(f"📊 Sample image embedding stats:")
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

    print("🖼️  Library Chatbot — Image Embedding Generator (OpenRouter)")
    print("=" * 60)
    print("Part of Approach A: this generates IMAGE embeddings only.")
    print("Text embeddings are handled by the companion script.")
    print()

    # ---- API key ----
    api_key = input("🔑 Enter your OpenRouter API key: ").strip()
    if not api_key:
        print("❌ API key is required")
        return

    # ---- Model selection ----
    print("\n📋 Available OpenRouter multimodal embedding models:")
    print("  1. google/gemini-embedding-2-preview    (3072 dims, multimodal latest)")
    print("  2. Custom model name")

    model_choice = input("\n🔧 Select model (1-2, default: 1): ").strip()

    if model_choice == "2":
        model_name = input("Enter model name: ").strip()
    else:
        model_name = "google/gemini-embedding-2-preview"

    print(f"\n📌 Using model: {model_name}")

    # ---- Initialise ----
    try:
        generator = ImageEmbeddingGeneratorOpenRouter(api_key, model_name=model_name)
    except Exception as e:
        print(f"❌ Failed to initialise: {e}")
        return

    # ---- Load chunks ----
    try:
        chunks = generator.load_chunks(INPUT_FILE)
    except Exception as e:
        print(f"❌ Failed to load chunks: {e}")
        return

    if not chunks:
        print("⚠️  No image chunks to process")
        return

    # ---- Resume handling ----
    existing_book_ids = set()
    if os.path.exists(OUTPUT_FILE):
        resume_choice = input(
            "📁 Existing image embeddings found. Resume? (y/n): "
        ).strip().lower()
        if resume_choice == 'y':
            try:
                with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                existing_book_ids = {c.get('book_id') for c in existing if c.get('book_id')}
                print(f"🔄 Resuming: {len(existing_book_ids)} already embedded, will skip them")
            except Exception:
                print("⚠️  Could not read existing file, starting fresh")

    # ---- Generate embeddings ----
    try:
        embedded_chunks = generator.process_chunks(chunks, OUTPUT_FILE, existing_book_ids)

        if embedded_chunks:
            generator.validate_embeddings(embedded_chunks)
            generator.generate_statistics_report()

            print(f"\n🎉 Successfully generated image embeddings for {len(embedded_chunks):,} images!")
            print(f"📁 Output:     {OUTPUT_FILE}")
            print(f"📊 Statistics: {STATS_FILE}")
            print(f"\n💡 Both text and image embeddings are now ready for your vector DB.")
            print(f"   They share the same book_id for cross-modal linking.")
        else:
            print("\n⚠️ No embeddings were generated.")

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted — partial results have been saved and can be resumed")
    except Exception as e:
        print(f"❌ Image embedding generation failed: {e}")

if __name__ == "__main__":
    main()
