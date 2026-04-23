#!/usr/bin/env python3
"""
Library Chatbot Embedding Generation Script
Generates embeddings for book text chunks using Google's text-embedding-004 model
"""

import json
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import os
import sys
from tqdm import tqdm

# Add the parent directory to path to import from other folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from google import genai
except ImportError:
    print("❌ Error: google-genai not installed")
    print("Please install: pip install google-genai")
    sys.exit(1)

class BookEmbeddingGenerator:
    def __init__(self, api_key: str = None):
        """Initialize the embedding generator"""
        self.api_key = api_key
        self.model_name = "gemini-embedding-001"
        self.embedding_dim = 768
        
        # Statistics tracking
        self.stats = {
            'total_chunks': 0,
            'processed_chunks': 0,
            'failed_chunks': 0,
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
        self.min_delay = 0.1  # Minimum delay between API calls (100ms)
        
        if api_key:
            self.configure_api(api_key)
    
    def configure_api(self, api_key: str):
        """Configure the Google AI API"""
        try:
            self.client = genai.Client(api_key=api_key)
            print("✅ Google AI API configured successfully")
            
            # Test the API connection
            self.test_api_connection()
            
        except Exception as e:
            print(f"❌ Error configuring API: {e}")
            raise
    
    def test_api_connection(self):
        """Test the API connection with a simple embedding"""
        try:
            print("🧪 Testing API connection...")
            test_result = self.client.models.embed_content(
                model=self.model_name,
                contents="test connection"
            )
            
            if test_result and test_result.embeddings:
                embedding_dim = len(test_result.embeddings[0].values)
                print(f"✅ API connection successful - Embedding dimension: {embedding_dim}")
                self.embedding_dim = embedding_dim
            else:
                raise Exception("Invalid API response format")
                
        except Exception as e:
            print(f"❌ API connection test failed: {e}")
            raise
    
    def rate_limit_delay(self):
        """Implement rate limiting to avoid API quota issues"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_delay:
            sleep_time = self.min_delay - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
    
    def generate_single_embedding(self, text: str, retry_count: int = 3) -> List[float]:
        """Generate embedding for a single text chunk with retry logic"""
        
        for attempt in range(retry_count):
            try:
                # Rate limiting
                self.rate_limit_delay()
                
                # Make API call
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                
                self.stats['api_calls'] += 1
                self.stats['total_tokens'] += len(text.split())
                
                if result and result.embeddings:
                    embedding = result.embeddings[0].values
                    
                    # Validate embedding
                    if len(embedding) == self.embedding_dim:
                        return embedding
                    else:
                        raise Exception(f"Unexpected embedding dimension: {len(embedding)}")
                else:
                    raise Exception("Invalid API response")
                    
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                
                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    self.stats['rate_limit_hits'] += 1
                    wait_time = (attempt + 1) * 5  # Exponential backoff
                    print(f"💤 Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif attempt < retry_count - 1:
                    time.sleep(1)  # Brief pause before retry
                else:
                    print(f"❌ Failed to generate embedding after {retry_count} attempts")
                    raise
        
        return None
    
    def load_chunks(self, input_file: str) -> List[Dict[str, Any]]:
        """Load book chunks from JSON file"""
        print(f"📖 Loading chunks from {input_file}...")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            self.stats['total_chunks'] = len(chunks)
            
            # Calculate average chunk length
            chunk_lengths = [len(chunk['text']) for chunk in chunks]
            self.stats['avg_chunk_length'] = sum(chunk_lengths) / len(chunk_lengths)
            
            print(f"✅ Loaded {len(chunks)} chunks")
            print(f"📊 Average chunk length: {self.stats['avg_chunk_length']:.0f} characters")
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error loading chunks: {e}")
            raise
    
    def process_chunks(self, chunks: List[Dict[str, Any]], output_file: str, 
                      resume_from: int = 0, save_interval: int = 100):
        """Process all chunks and generate embeddings"""
        
        print(f"🚀 Starting embedding generation for {len(chunks)} chunks...")
        print(f"📝 Output file: {output_file}")
        
        if resume_from > 0:
            print(f"🔄 Resuming from chunk {resume_from}")
        
        self.stats['processing_start'] = datetime.now()
        
        # Prepare output structure
        embedded_chunks = []
        
        # Load existing results if resuming
        if resume_from > 0 and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    embedded_chunks = json.load(f)
                print(f"📁 Loaded {len(embedded_chunks)} existing embeddings")
            except:
                print("⚠️  Could not load existing embeddings, starting fresh")
                embedded_chunks = []
        
        # Process chunks with progress bar
        with tqdm(total=len(chunks), initial=resume_from, desc="Generating embeddings") as pbar:
            
            for i, chunk in enumerate(chunks[resume_from:], start=resume_from):
                
                try:
                    # Generate embedding for the text only
                    embedding = self.generate_single_embedding(chunk['text'])
                    
                    if embedding:
                        # Create embedded chunk
                        embedded_chunk = {
                            'chunk_id': i,
                            'text': chunk['text'],
                            'embedding': embedding,
                            'metadata': {
                                **chunk['metadata'],
                                'embedding_model': self.model_name,
                                'embedding_dim': len(embedding),
                                'processed_at': datetime.now().isoformat()
                            }
                        }
                        
                        embedded_chunks.append(embedded_chunk)
                        self.stats['processed_chunks'] += 1
                        
                        # Periodic saving
                        if (i + 1) % save_interval == 0:
                            self.save_embeddings(embedded_chunks, output_file)
                            print(f"\n💾 Saved progress: {i + 1}/{len(chunks)} chunks processed")
                    
                    else:
                        print(f"❌ Failed to generate embedding for chunk {i}")
                        self.stats['failed_chunks'] += 1
                
                except Exception as e:
                    print(f"❌ Error processing chunk {i}: {e}")
                    self.stats['failed_chunks'] += 1
                
                pbar.update(1)
                
                # Update progress description
                success_rate = (self.stats['processed_chunks'] / (i + 1)) * 100
                pbar.set_description(f"Embeddings ({success_rate:.1f}% success)")
        
        # Final save
        self.save_embeddings(embedded_chunks, output_file)
        self.stats['processing_end'] = datetime.now()
        
        print(f"\n✅ Embedding generation completed!")
        print(f"📊 Processed: {self.stats['processed_chunks']}/{self.stats['total_chunks']}")
        print(f"❌ Failed: {self.stats['failed_chunks']}")
        
        return embedded_chunks
    
    def save_embeddings(self, embedded_chunks: List[Dict[str, Any]], output_file: str):
        """Save embeddings to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")
            raise
    
    def generate_statistics_report(self, output_file: str = "embedding/embedding_statistics.txt"):
        """Generate detailed statistics report"""
        
        duration = self.stats['processing_end'] - self.stats['processing_start']
        
        report = []
        report.append("LIBRARY CHATBOT EMBEDDING STATISTICS")
        report.append("=" * 50)
        report.append(f"Processing Date: {self.stats['processing_start'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Processing Time: {duration}")
        report.append("")
        
        report.append("PROCESSING SUMMARY:")
        report.append(f"Total chunks: {self.stats['total_chunks']:,}")
        report.append(f"Successfully processed: {self.stats['processed_chunks']:,}")
        report.append(f"Failed: {self.stats['failed_chunks']:,}")
        
        if self.stats['total_chunks'] > 0:
            success_rate = (self.stats['processed_chunks'] / self.stats['total_chunks']) * 100
            report.append(f"Success rate: {success_rate:.1f}%")
        
        report.append("")
        
        report.append("API USAGE STATISTICS:")
        report.append(f"Total API calls: {self.stats['api_calls']:,}")
        report.append(f"Rate limit hits: {self.stats['rate_limit_hits']:,}")
        report.append(f"Total tokens processed: {self.stats['total_tokens']:,}")
        
        if self.stats['api_calls'] > 0:
            avg_tokens_per_call = self.stats['total_tokens'] / self.stats['api_calls']
            report.append(f"Average tokens per call: {avg_tokens_per_call:.1f}")
        
        report.append("")
        
        report.append("EMBEDDING DETAILS:")
        report.append(f"Model used: {self.model_name}")
        report.append(f"Embedding dimensions: {self.embedding_dim}")
        report.append(f"Average chunk length: {self.stats['avg_chunk_length']:.0f} characters")
        
        report.append("")
        report.append("EMBEDDING GENERATION COMPLETE ✅")
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))
    
    def validate_embeddings(self, embedded_chunks: List[Dict[str, Any]]):
        """Validate the generated embeddings"""
        print("\n🔍 Validating embeddings...")
        
        if not embedded_chunks:
            print("❌ No embeddings to validate")
            return False
        
        # Check dimensions
        embedding_dims = [len(chunk['embedding']) for chunk in embedded_chunks]
        unique_dims = set(embedding_dims)
        
        if len(unique_dims) == 1:
            print(f"✅ All embeddings have consistent dimension: {list(unique_dims)[0]}")
        else:
            print(f"⚠️  Inconsistent embedding dimensions found: {unique_dims}")
        
        # Check for null embeddings
        null_count = sum(1 for chunk in embedded_chunks if not chunk['embedding'])
        print(f"📊 Null embeddings: {null_count}")
        
        # Sample embedding statistics
        sample_embedding = embedded_chunks[0]['embedding']
        embedding_array = np.array(sample_embedding)
        
        print(f"📊 Sample embedding stats:")
        print(f"   Mean: {np.mean(embedding_array):.4f}")
        print(f"   Std:  {np.std(embedding_array):.4f}")
        print(f"   Min:  {np.min(embedding_array):.4f}")
        print(f"   Max:  {np.max(embedding_array):.4f}")
        
        return True

def main():
    """Main function to run the embedding generation"""
    
    print("🚀 Library Chatbot Embedding Generator")
    print("=" * 50)
    
    # Check for API key
    api_key = input("🔑 Please enter your Google AI Studio API key: ").strip()
    
    if not api_key:
        print("❌ API key is required to proceed")
        return
    
    # Initialize generator
    try:
        generator = BookEmbeddingGenerator(api_key)
    except Exception as e:
        print(f"❌ Failed to initialize embedding generator: {e}")
        return
    
    # Load chunks
    input_file = "chunking/book_chunks.json"
    output_file = "embedding/book_embeddings_2.json"
    
    try:
        chunks = generator.load_chunks(input_file)
    except Exception as e:
        print(f"❌ Failed to load chunks: {e}")
        return
    
    # Ask about resume
    resume_from = 0
    if os.path.exists(output_file):
        resume_choice = input("📁 Existing embeddings found. Resume from where left off? (y/n): ").strip().lower()
        if resume_choice == 'y':
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                resume_from = len(existing)
                print(f"🔄 Will resume from chunk {resume_from}")
            except:
                print("⚠️  Could not read existing file, starting fresh")
    
    # Generate embeddings
    try:
        embedded_chunks = generator.process_chunks(chunks, output_file, resume_from)
        
        # Validate results
        generator.validate_embeddings(embedded_chunks)
        
        # Generate statistics
        generator.generate_statistics_report()
        
        print(f"\n🎉 Successfully generated embeddings for {len(embedded_chunks)} chunks!")
        print(f"📁 Output saved to: {output_file}")
        print(f"📊 Statistics saved to: embedding/embedding_statistics.txt")
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
        print("📁 Partial results have been saved and can be resumed")
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")

if __name__ == "__main__":
    main() 