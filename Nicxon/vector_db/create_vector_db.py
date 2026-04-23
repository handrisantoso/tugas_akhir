#!/usr/bin/env python3
"""
Library Chatbot Vector Database Setup
Creates and populates ChromaDB with book embeddings for semantic search
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
    from google import genai
except ImportError:
    print("❌ Error: Required libraries not installed")
    print("Please install: pip install -r vector_db/requirements.txt")
    print("Also ensure Google GenAI is installed: pip install google-genai")
    sys.exit(1)

class LibraryVectorDB:
    def __init__(self, db_path: str = "vector_db/chroma_db", google_api_key: str = None):
        """Initialize the vector database"""
        self.db_path = db_path
        self.client = None
        self.collection = None
        self.collection_name = "library_books"
        self.google_api_key = google_api_key
        
        # Statistics
        self.stats = {
            'total_embeddings': 0,
            'processed_embeddings': 0,
            'failed_embeddings': 0,
            'processing_start': None,
            'processing_end': None,
            'metadata_extracted': 0,
            'database_size_mb': 0
        }
        
        # Configure Google API if provided
        if google_api_key:
            self.configure_google_api(google_api_key)
        
        self.setup_database()
    
    def configure_google_api(self, api_key: str):
        """Configure Google Generative AI API for query embeddings"""
        try:
            self.genai_client = genai.Client(api_key=api_key)
            print("✅ Google AI API configured for query embeddings")
        except Exception as e:
            print(f"⚠️  Warning: Could not configure Google API: {e}")
            print("Search functionality will be limited")
    
    def generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for search query using Google API"""
        if not self.google_api_key:
            raise Exception("Google API key required for query embeddings")
        
        try:
            result = self.genai_client.models.embed_content(
                model="gemini-embedding-001",
                contents=query_text
            )
            
            if result and result.embeddings:
                return result.embeddings[0].values
            else:
                raise Exception("Invalid API response for query embedding")
                
        except Exception as e:
            print(f"❌ Error generating query embedding: {e}")
            raise
    
    def setup_database(self):
        """Initialize ChromaDB client and collection"""
        try:
            print("🗄️  Setting up ChromaDB...")
            
            # Create persistent ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection with no default embedding function
            try:
                self.collection = self.client.get_collection(self.collection_name)
                existing_count = self.collection.count()
                print(f"📁 Found existing collection with {existing_count} documents")
                
                # Ask if user wants to reset
                if existing_count > 0:
                    reset = input("🔄 Reset existing collection? (y/n): ").strip().lower()
                    if reset == 'y':
                        self.client.delete_collection(self.collection_name)
                        self.collection = self.client.create_collection(
                            name=self.collection_name,
                            embedding_function=None,  # We provide our own embeddings
                            metadata={"description": "Library book embeddings for semantic search"}
                        )
                        print("✅ Collection reset successfully")
                    else:
                        print("📁 Using existing collection")
                        
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,  # We provide our own embeddings
                    metadata={"description": "Library book embeddings for semantic search"}
                )
                print("✅ New collection created")
            
        except Exception as e:
            print(f"❌ Error setting up database: {e}")
            raise
    
    def extract_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize metadata for efficient filtering"""
        text = chunk.get('text', '')
        original_metadata = chunk.get('metadata', {})
        
        # Initialize metadata
        metadata = {
            'chunk_id': chunk.get('chunk_id', 0),
            'has_description': original_metadata.get('has_description', False),
            'has_subjects': original_metadata.get('has_subjects', False),
            'has_toc': original_metadata.get('has_toc', False),
            'field_count': original_metadata.get('field_count', 0)
        }
        
        # Comprehensive language mapping
        language_mapping = {
            'english': 'English',
            'eng': 'English',
            'spanish': 'Spanish', 
            'spa': 'Spanish',
            'french': 'French',
            'fre': 'French',
            'fra': 'French',
            'german': 'German',
            'ger': 'German',
            'deu': 'German',
            'hindi': 'Hindi',
            'hin': 'Hindi',
            'italian': 'Italian',
            'ita': 'Italian',
            'portuguese': 'Portuguese',
            'por': 'Portuguese',
            'russian': 'Russian',
            'rus': 'Russian',
            'chinese': 'Chinese',
            'chi': 'Chinese',
            'zho': 'Chinese',
            'japanese': 'Japanese',
            'jpn': 'Japanese',
            'korean': 'Korean',
            'kor': 'Korean',
            'arabic': 'Arabic',
            'ara': 'Arabic',
            'dutch': 'Dutch',
            'dut': 'Dutch',
            'nld': 'Dutch',
            'swedish': 'Swedish',
            'swe': 'Swedish',
            'norwegian': 'Norwegian',
            'nor': 'Norwegian',
            'danish': 'Danish',
            'dan': 'Danish',
            'finnish': 'Finnish',
            'fin': 'Finnish',
            'polish': 'Polish',
            'pol': 'Polish',
            'czech': 'Czech',
            'cze': 'Czech',
            'ces': 'Czech',
            'hungarian': 'Hungarian',
            'hun': 'Hungarian',
            'greek': 'Greek',
            'gre': 'Greek',
            'ell': 'Greek',
            'hebrew': 'Hebrew',
            'heb': 'Hebrew',
            'turkish': 'Turkish',
            'tur': 'Turkish',
            'thai': 'Thai',
            'tha': 'Thai',
            'vietnamese': 'Vietnamese',
            'vie': 'Vietnamese',
            'indonesian': 'Indonesian',
            'ind': 'Indonesian',
            'malay': 'Malay',
            'may': 'Malay',
            'msa': 'Malay',
            'swahili': 'Swahili',
            'swa': 'Swahili',
            'afrikaans': 'Afrikaans',
            'afr': 'Afrikaans',
            'bulgarian': 'Bulgarian',
            'bul': 'Bulgarian',
            'croatian': 'Croatian',
            'hrv': 'Croatian',
            'serbian': 'Serbian',
            'srp': 'Serbian',
            'slovenian': 'Slovenian',
            'slv': 'Slovenian',
            'slovak': 'Slovak',
            'slo': 'Slovak',
            'slk': 'Slovak',
            'romanian': 'Romanian',
            'rum': 'Romanian',
            'ron': 'Romanian',
            'ukrainian': 'Ukrainian',
            'ukr': 'Ukrainian',
            'belarusian': 'Belarusian',
            'bel': 'Belarusian',
            'lithuanian': 'Lithuanian',
            'lit': 'Lithuanian',
            'latvian': 'Latvian',
            'lav': 'Latvian',
            'estonian': 'Estonian',
            'est': 'Estonian',
            'icelandic': 'Icelandic',
            'ice': 'Icelandic',
            'isl': 'Icelandic',
            'irish': 'Irish',
            'gle': 'Irish',
            'welsh': 'Welsh',
            'wel': 'Welsh',
            'cym': 'Welsh',
            'scots': 'Scots',
            'gla': 'Scottish Gaelic',
            'basque': 'Basque',
            'baq': 'Basque',
            'eus': 'Basque',
            'catalan': 'Catalan',
            'cat': 'Catalan',
            'galician': 'Galician',
            'glg': 'Galician',
            'urdu': 'Urdu',
            'urd': 'Urdu',
            'bengali': 'Bengali',
            'ben': 'Bengali',
            'gujarati': 'Gujarati',
            'guj': 'Gujarati',
            'marathi': 'Marathi',
            'mar': 'Marathi',
            'punjabi': 'Punjabi',
            'pan': 'Punjabi',
            'tamil': 'Tamil',
            'tam': 'Tamil',
            'telugu': 'Telugu',
            'tel': 'Telugu',
            'kannada': 'Kannada',
            'kan': 'Kannada',
            'malayalam': 'Malayalam',
            'mal': 'Malayalam',
            'sinhalese': 'Sinhalese',
            'sin': 'Sinhalese',
            'nepali': 'Nepali',
            'nep': 'Nepali',
            'tibetan': 'Tibetan',
            'tib': 'Tibetan',
            'bod': 'Tibetan',
            'burmese': 'Burmese',
            'bur': 'Burmese',
            'mya': 'Burmese',
            'khmer': 'Khmer',
            'khm': 'Khmer',
            'lao': 'Lao',
            'mongolian': 'Mongolian',
            'mon': 'Mongolian',
            'persian': 'Persian',
            'per': 'Persian',
            'fas': 'Persian',
            'pashto': 'Pashto',
            'pus': 'Pashto',
            'kurdish': 'Kurdish',
            'kur': 'Kurdish',
            'armenian': 'Armenian',
            'arm': 'Armenian',
            'hye': 'Armenian',
            'georgian': 'Georgian',
            'geo': 'Georgian',
            'kat': 'Georgian',
            'azerbaijani': 'Azerbaijani',
            'aze': 'Azerbaijani',
            'kazakh': 'Kazakh',
            'kaz': 'Kazakh',
            'kyrgyz': 'Kyrgyz',
            'kir': 'Kyrgyz',
            'tajik': 'Tajik',
            'tgk': 'Tajik',
            'turkmen': 'Turkmen',
            'tuk': 'Turkmen',
            'uzbek': 'Uzbek',
            'uzb': 'Uzbek',
            'albanian': 'Albanian',
            'alb': 'Albanian',
            'sqi': 'Albanian',
            'macedonian': 'Macedonian',
            'mac': 'Macedonian',
            'mkd': 'Macedonian',
            'maltese': 'Maltese',
            'mlt': 'Maltese',
            'latin': 'Latin',
            'lat': 'Latin',
            'esperanto': 'Esperanto',
            'epo': 'Esperanto',
            'yiddish': 'Yiddish',
            'yid': 'Yiddish',
            'amharic': 'Amharic',
            'amh': 'Amharic',
            'hausa': 'Hausa',
            'hau': 'Hausa',
            'yoruba': 'Yoruba',
            'yor': 'Yoruba',
            'igbo': 'Igbo',
            'ibo': 'Igbo',
            'zulu': 'Zulu',
            'zul': 'Zulu',
            'xhosa': 'Xhosa',
            'xho': 'Xhosa',
            'sesotho': 'Sesotho',
            'sot': 'Sesotho',
            'setswana': 'Setswana',
            'tsn': 'Setswana',
            'shona': 'Shona',
            'sna': 'Shona',
            'ndebele': 'Ndebele',
            'nde': 'Ndebele',
            'somali': 'Somali',
            'som': 'Somali',
            'oromo': 'Oromo',
            'orm': 'Oromo',
            'tigrinya': 'Tigrinya',
            'tir': 'Tigrinya',
            'kinyarwanda': 'Kinyarwanda',
            'kin': 'Kinyarwanda',
            'kirundi': 'Kirundi',
            'run': 'Kirundi',
            'luganda': 'Luganda',
            'lug': 'Luganda',
            'wolof': 'Wolof',
            'wol': 'Wolof',
            'bambara': 'Bambara',
            'bam': 'Bambara',
            'fulah': 'Fulah',
            'ful': 'Fulah',
            'lingala': 'Lingala',
            'lin': 'Lingala',
            'kikongo': 'Kikongo',
            'kon': 'Kikongo',
            'chichewa': 'Chichewa',
            'nya': 'Chichewa',
            'malagasy': 'Malagasy',
            'mlg': 'Malagasy',
            'multiple': 'Multiple Languages',
            'multilingual': 'Multiple Languages',
            'undetermined': 'Undetermined',
            'und': 'Undetermined',
            'no_linguistic': 'Non-linguistic',
            'zxx': 'Non-linguistic'
        }
        
        # Extract language (convert to simple string)
        if 'LANGUAGE:' in text:
            lang_match = re.search(r'LANGUAGE:\s*([^\n]+)', text)
            if lang_match:
                language_raw = lang_match.group(1).strip()
                
                # Clean up language format and extract key
                if language_raw.startswith('{') and 'key' in language_raw:
                    # Extract from format like "{'key': 'english'}"
                    key_match = re.search(r"'key':\s*'([^']+)'", language_raw)
                    if key_match:
                        lang_key = key_match.group(1).lower().strip()
                        # Look up in comprehensive mapping
                        language = language_mapping.get(lang_key, lang_key.title())
                    else:
                        language = 'Unknown'
                else:
                    # Handle direct language strings
                    lang_clean = language_raw.lower().strip()
                    language = language_mapping.get(lang_clean, language_raw.title())
                
                metadata['language'] = language
        
        # Extract publication year
        if 'PUBLICATION:' in text:
            pub_match = re.search(r'Published:\s*(\d{4})', text)
            if pub_match:
                metadata['publish_year'] = int(pub_match.group(1))
        
        # Extract page count
        if 'Pages:' in text:
            pages_match = re.search(r'Pages:\s*(\d+)', text)
            if pages_match:
                metadata['page_count'] = int(pages_match.group(1))
        
        # Extract format
        if 'Format:' in text:
            format_match = re.search(r'Format:\s*([^;,\n]+)', text)
            if format_match:
                metadata['format'] = format_match.group(1).strip()
        
        # Extract genres from subjects
        if 'SUBJECTS & TOPICS:' in text:
            subjects_match = re.search(r'SUBJECTS & TOPICS:\s*([^\n]+)', text)
            if subjects_match:
                subjects_text = subjects_match.group(1)
                metadata['subjects_text'] = subjects_text
                
                # Extract specific genres
                genres = []
                genre_patterns = [
                    r'mystery', r'detective', r'crime', r'thriller',
                    r'fantasy', r'magic', r'science fiction', r'sci-fi',
                    r'romance', r'love story', r'horror', r'scary'
                ]
                
                for pattern in genre_patterns:
                    if re.search(pattern, subjects_text, re.IGNORECASE):
                        genre_name = pattern.replace('r\'', '').replace('\'', '')
                        if genre_name not in genres:
                            genres.append(genre_name)
                
                if genres:
                    metadata['genres'] = ', '.join(genres)
        
        # Extract author
        if 'AUTHOR(S):' in text:
            author_match = re.search(r'AUTHOR\(S\):\s*([^\n]+)', text)
            if author_match:
                metadata['author'] = author_match.group(1).strip()
        
        # Extract title
        if 'TITLE:' in text:
            title_match = re.search(r'TITLE:\s*([^\n]+)', text)
            if title_match:
                metadata['title'] = title_match.group(1).strip()
        
        return metadata
    
    def load_embeddings(self, embeddings_file: str) -> List[Dict[str, Any]]:
        """Load embeddings from JSON file"""
        print(f"📖 Loading embeddings from {embeddings_file}...")
        
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embeddings = json.load(f)
            
            self.stats['total_embeddings'] = len(embeddings)
            print(f"✅ Loaded {len(embeddings)} embeddings")
            
            return embeddings
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            raise
    
    def populate_database(self, embeddings: List[Dict[str, Any]], batch_size: int = 100):
        """Populate the vector database with embeddings and metadata"""
        
        print(f"🚀 Populating database with {len(embeddings)} embeddings...")
        self.stats['processing_start'] = datetime.now()
        
        # Process in batches for efficiency
        total_batches = (len(embeddings) + batch_size - 1) // batch_size
        
        with tqdm(total=len(embeddings), desc="Adding to database") as pbar:
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(embeddings))
                batch = embeddings[start_idx:end_idx]
                
                # Prepare batch data
                ids = []
                documents = []
                embeddings_list = []
                metadatas = []
                
                for chunk in batch:
                    try:
                        # Generate unique ID
                        chunk_id = chunk.get('chunk_id', start_idx + len(ids))
                        ids.append(f"book_{chunk_id}")
                        
                        # Get document text
                        documents.append(chunk['text'])
                        
                        # Get embedding vector
                        embeddings_list.append(chunk['embedding'])
                        
                        # Extract and prepare metadata
                        metadata = self.extract_metadata(chunk)
                        
                        # Ensure all metadata values are strings, numbers, or booleans for ChromaDB
                        clean_metadata = {}
                        for key, value in metadata.items():
                            if isinstance(value, (str, int, float, bool)):
                                clean_metadata[key] = value
                            else:
                                clean_metadata[key] = str(value)
                        
                        metadatas.append(clean_metadata)
                        self.stats['processed_embeddings'] += 1
                        self.stats['metadata_extracted'] += 1
                        
                    except Exception as e:
                        print(f"⚠️  Error processing chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                        self.stats['failed_embeddings'] += 1
                        continue
                
                # Add batch to collection
                if ids:  # Only add if we have valid data
                    try:
                        self.collection.add(
                            ids=ids,
                            embeddings=embeddings_list,
                            documents=documents,
                            metadatas=metadatas
                        )
                        
                    except Exception as e:
                        print(f"❌ Error adding batch {batch_idx}: {e}")
                        self.stats['failed_embeddings'] += len(batch)
                
                pbar.update(len(batch))
                
                # Update progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"\n💾 Processed {batch_idx + 1}/{total_batches} batches")
        
        self.stats['processing_end'] = datetime.now()
        
        # Get final count
        final_count = self.collection.count()
        print(f"\n✅ Database population completed!")
        print(f"📊 Documents in database: {final_count}")
        print(f"✅ Successfully processed: {self.stats['processed_embeddings']}")
        print(f"❌ Failed: {self.stats['failed_embeddings']}")
    
    def test_search(self, test_queries: List[str] = None, n_results: int = 5):
        """Test the search functionality"""
        
        if not test_queries:
            test_queries = [
                "mystery novels",
                "space exploration science fiction",
                "romance books",
                "books about history",
                "fantasy adventure stories"
            ]
        
        print(f"\n🔍 Testing search functionality with {len(test_queries)} queries...")
        
        if not self.google_api_key:
            print("⚠️  Skipping search tests - Google API key required for query embeddings")
            return
        
        for query in test_queries:
            print(f"\n📝 Query: '{query}'")
            
            try:
                # Generate query embedding using Google API
                query_embedding = self.generate_query_embedding(query)
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],  # Use our generated embedding
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances']
                )
                
                if results['documents'] and results['documents'][0]:
                    print(f"✅ Found {len(results['documents'][0])} results:")
                    
                    for i, (doc, metadata, distance) in enumerate(
                        zip(results['documents'][0], 
                            results['metadatas'][0], 
                            results['distances'][0]), 1
                    ):
                        # Extract title for display
                        title_match = re.search(r'TITLE:\s*([^\n]+)', doc)
                        title = title_match.group(1) if title_match else "Unknown Title"
                        
                        print(f"  {i}. {title}")
                        print(f"     Language: {metadata.get('language', 'Unknown')}")
                        print(f"     Similarity: {1-distance:.3f}")
                        
                        if i >= 3:  # Show only top 3 for testing
                            break
                else:
                    print("❌ No results found")
                    
            except Exception as e:
                print(f"❌ Search error: {e}")
    
    def test_filtered_search(self):
        """Test search with metadata filtering"""
        
        print(f"\n🎯 Testing filtered search...")
        
        if not self.google_api_key:
            print("⚠️  Skipping filtered search tests - Google API key required for query embeddings")
            return
        
        test_filters = [
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
            {
                "description": "Books under 300 pages",
                "where": {"page_count": {"$lt": 300}},
                "query": "mystery novels"
            }
        ]
        
        for test in test_filters:
            print(f"\n🔍 Filter: {test['description']}")
            print(f"📝 Query: '{test['query']}'")
            
            try:
                # Generate query embedding using Google API
                query_embedding = self.generate_query_embedding(test['query'])
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],  # Use our generated embedding
                    where=test['where'],
                    n_results=3,
                    include=['documents', 'metadatas', 'distances']
                )
                
                if results['documents'] and results['documents'][0]:
                    print(f"✅ Found {len(results['documents'][0])} filtered results")
                    
                    for i, (doc, metadata) in enumerate(
                        zip(results['documents'][0], results['metadatas'][0]), 1
                    ):
                        title_match = re.search(r'TITLE:\s*([^\n]+)', doc)
                        title = title_match.group(1) if title_match else "Unknown Title"
                        print(f"  {i}. {title} ({metadata.get('language', 'Unknown')})")
                else:
                    print("❌ No filtered results found")
                    
            except Exception as e:
                print(f"❌ Filtered search error: {e}")
    
    def generate_statistics(self):
        """Generate database statistics"""
        
        print(f"\n📊 Generating database statistics...")
        
        try:
            total_docs = self.collection.count()
            
            # Get sample of metadata for analysis
            sample_results = self.collection.get(
                limit=min(100, total_docs),
                include=['metadatas']
            )
            
            # Analyze metadata
            languages = {}
            formats = {}
            years = {}
            has_desc_count = 0
            
            for metadata in sample_results['metadatas']:
                # Language distribution
                lang = metadata.get('language', 'Unknown')
                languages[lang] = languages.get(lang, 0) + 1
                
                # Format distribution
                fmt = metadata.get('format', 'Unknown')
                formats[fmt] = formats.get(fmt, 0) + 1
                
                # Year distribution
                year = metadata.get('publish_year')
                if year:
                    decade = (year // 10) * 10
                    years[f"{decade}s"] = years.get(f"{decade}s", 0) + 1
                
                # Description availability
                if metadata.get('has_description'):
                    has_desc_count += 1
            
            # Calculate database size
            if os.path.exists(self.db_path):
                db_size = sum(os.path.getsize(os.path.join(self.db_path, f)) 
                             for f in os.listdir(self.db_path) 
                             if os.path.isfile(os.path.join(self.db_path, f)))
                self.stats['database_size_mb'] = db_size / (1024 * 1024)
            
            # Print statistics
            duration = self.stats['processing_end'] - self.stats['processing_start']
            
            print(f"\n📋 DATABASE STATISTICS")
            print(f"{'='*50}")
            print(f"Total documents: {total_docs:,}")
            print(f"Database size: {self.stats['database_size_mb']:.1f} MB")
            print(f"Processing time: {duration}")
            print(f"Success rate: {(self.stats['processed_embeddings']/self.stats['total_embeddings']*100):.1f}%")
            
            print(f"\n📚 SAMPLE ANALYSIS (from {len(sample_results['metadatas'])} docs):")
            
            print(f"\nLanguages:")
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {lang}: {count}")
            
            print(f"\nFormats:")
            for fmt, count in sorted(formats.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {fmt}: {count}")
            
            print(f"\nPublication decades:")
            for decade, count in sorted(years.items())[-5:]:
                print(f"  {decade}: {count}")
            
            print(f"\nContent richness:")
            print(f"  Books with descriptions: {has_desc_count}")
            print(f"  Percentage: {(has_desc_count/len(sample_results['metadatas'])*100):.1f}%")
            
        except Exception as e:
            print(f"❌ Error generating statistics: {e}")

def main():
    """Main function to create and populate the vector database"""
    
    print("🗄️  Library Chatbot Vector Database Creator")
    print("=" * 50)
    
    # Get Google API key for query embeddings
    google_api_key = input("🔑 Enter your Google AI Studio API key (for search functionality): ").strip()
    if not google_api_key:
        print("⚠️  Warning: No API key provided. Search functionality will be limited.")
        google_api_key = None
    
    # Initialize database
    try:
        db = LibraryVectorDB(google_api_key=google_api_key)
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return
    
    # Load embeddings
    embeddings_file = "embedding/book_embeddings_openrouter.json"
    
    if not os.path.exists(embeddings_file):
        print(f"❌ Embeddings file not found: {embeddings_file}")
        print("Please run the embedding generation first")
        return
    
    try:
        embeddings = db.load_embeddings(embeddings_file)
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        return
    
    # Check if database is already populated
    current_count = db.collection.count()
    if current_count > 0:
        print(f"📁 Database already contains {current_count} documents")
        repopulate = input("🔄 Re-populate database? (y/n): ").strip().lower()
        if repopulate != 'y':
            print("✅ Using existing database")
            # Skip to testing
            db.test_search()
            db.test_filtered_search()
            db.generate_statistics()
            return
    
    # Populate database
    try:
        db.populate_database(embeddings)
    except Exception as e:
        print(f"❌ Failed to populate database: {e}")
        return
    
    # Test functionality
    print(f"\n{'='*50}")
    print("🧪 TESTING DATABASE FUNCTIONALITY")
    print(f"{'='*50}")
    
    db.test_search()
    db.test_filtered_search()
    db.generate_statistics()
    
    print(f"\n🎉 Vector database setup completed successfully!")
    print(f"📁 Database location: {db.db_path}")
    print(f"📊 Total documents: {db.collection.count():,}")
    print(f"\n🚀 Ready for semantic search and RAG implementation!")

if __name__ == "__main__":
    main() 