#!/usr/bin/env python3
"""
Library Search Utility
Interactive search tool for the library chatbot vector database
"""

import json
import os
import sys
import re
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("❌ Error: ChromaDB not installed")
    print("Please install: pip install -r vector_db/requirements.txt")
    sys.exit(1)

class LibrarySearcher:
    def __init__(self, db_path: str = "vector_db/chroma_db"):
        """Initialize the search interface"""
        self.db_path = db_path
        self.client = None
        self.collection = None
        self.collection_name = "library_books"
        
        self.connect_to_database()
    
    def connect_to_database(self):
        """Connect to existing ChromaDB"""
        try:
            print("🔗 Connecting to vector database...")
            
            if not os.path.exists(self.db_path):
                print(f"❌ Database not found at {self.db_path}")
                print("Please run 'python vector_db/create_vector_db.py' first")
                sys.exit(1)
            
            # Connect to persistent client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            
            # Get collection
            self.collection = self.client.get_collection(self.collection_name)
            doc_count = self.collection.count()
            
            print(f"✅ Connected to database with {doc_count:,} books")
            
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            sys.exit(1)
    
    def search_books(self, query: str, n_results: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for books using semantic similarity"""
        try:
            print(f"🔍 Searching for: '{query}'")
            
            # Perform search
            search_params = {
                'query_texts': [query],
                'n_results': n_results,
                'include': ['documents', 'metadatas', 'distances']
            }
            
            # Add filters if provided
            if filters:
                search_params['where'] = filters
                print(f"🎯 Using filters: {filters}")
            
            results = self.collection.query(**search_params)
            
            # Process results
            processed_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(results['documents'][0], 
                        results['metadatas'][0], 
                        results['distances'][0])
                ):
                    # Extract key information
                    title_match = re.search(r'TITLE:\s*([^\n]+)', doc)
                    title = title_match.group(1) if title_match else "Unknown Title"
                    
                    author_match = re.search(r'AUTHOR\(S\):\s*([^\n]+)', doc)
                    author = author_match.group(1) if author_match else "Unknown Author"
                    
                    desc_match = re.search(r'DESCRIPTION:\s*([^\n]+)', doc)
                    description = desc_match.group(1) if desc_match else "No description available"
                    
                    similarity_score = 1 - distance  # Convert distance to similarity
                    
                    processed_results.append({
                        'rank': i + 1,
                        'title': title,
                        'author': author,
                        'description': description,
                        'similarity': similarity_score,
                        'language': metadata.get('language', 'Unknown'),
                        'publish_year': metadata.get('publish_year'),
                        'page_count': metadata.get('page_count'),
                        'format': metadata.get('format'),
                        'has_description': metadata.get('has_description', False),
                        'full_text': doc,
                        'metadata': metadata
                    })
            
            return processed_results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def display_results(self, results: List[Dict[str, Any]], detailed: bool = False):
        """Display search results in a user-friendly format"""
        
        if not results:
            print("📭 No results found")
            return
        
        print(f"\n📚 Found {len(results)} results:")
        print("=" * 80)
        
        for result in results:
            print(f"\n{result['rank']}. {result['title']}")
            print(f"   Author: {result['author']}")
            print(f"   Language: {result['language']}")
            
            if result['publish_year']:
                print(f"   Published: {result['publish_year']}")
            
            if result['page_count']:
                print(f"   Pages: {result['page_count']}")
            
            if result['format']:
                print(f"   Format: {result['format']}")
            
            print(f"   Similarity: {result['similarity']:.3f}")
            
            if detailed:
                # Show description
                description = result['description']
                if len(description) > 200:
                    description = description[:200] + "..."
                print(f"   Description: {description}")
                
                # Show subjects if available
                subjects_match = re.search(r'SUBJECTS & TOPICS:\s*([^\n]+)', result['full_text'])
                if subjects_match:
                    subjects = subjects_match.group(1)
                    if len(subjects) > 150:
                        subjects = subjects[:150] + "..."
                    print(f"   Topics: {subjects}")
            
            print("-" * 60)
    
    def advanced_search_interface(self):
        """Interactive advanced search with filters"""
        print("\n🎯 Advanced Search Interface")
        print("=" * 40)
        
        # Get search query
        query = input("📝 Enter search query: ").strip()
        if not query:
            return
        
        # Get optional filters
        filters = {}
        
        # Language filter
        language = input("🌍 Filter by language (e.g., English, Spanish) [Enter to skip]: ").strip()
        if language:
            filters['language'] = language
        
        # Year filter
        year_input = input("📅 Filter by publication year (e.g., 2000, >1990, <2010) [Enter to skip]: ").strip()
        if year_input:
            if year_input.startswith('>'):
                filters['publish_year'] = {"$gt": int(year_input[1:])}
            elif year_input.startswith('<'):
                filters['publish_year'] = {"$lt": int(year_input[1:])}
            elif year_input.isdigit():
                filters['publish_year'] = int(year_input)
        
        # Page count filter
        pages_input = input("📄 Filter by page count (e.g., >300, <200) [Enter to skip]: ").strip()
        if pages_input:
            if pages_input.startswith('>'):
                filters['page_count'] = {"$gt": int(pages_input[1:])}
            elif pages_input.startswith('<'):
                filters['page_count'] = {"$lt": int(pages_input[1:])}
        
        # Has description filter
        desc_filter = input("📝 Only books with descriptions? (y/n) [Enter to skip]: ").strip().lower()
        if desc_filter == 'y':
            filters['has_description'] = True
        
        # Number of results
        try:
            n_results = int(input("🔢 Number of results (default 10): ").strip() or "10")
        except ValueError:
            n_results = 10
        
        # Detailed output
        detailed = input("📋 Show detailed results? (y/n) [default n]: ").strip().lower() == 'y'
        
        # Perform search
        results = self.search_books(query, n_results, filters if filters else None)
        self.display_results(results, detailed)
    
    def quick_search_interface(self):
        """Simple quick search interface"""
        print("\n🔍 Quick Search")
        print("=" * 30)
        
        query = input("📝 Search for books: ").strip()
        if not query:
            return
        
        results = self.search_books(query, n_results=5)
        self.display_results(results)
    
    def example_searches(self):
        """Run example searches to demonstrate capabilities"""
        print("\n🎪 Example Searches")
        print("=" * 40)
        
        examples = [
            {
                'query': 'mystery detective novels',
                'description': 'Mystery and detective fiction'
            },
            {
                'query': 'space science fiction adventure',
                'description': 'Science fiction with space themes'
            },
            {
                'query': 'historical romance',
                'description': 'Romance novels with historical settings'
            },
            {
                'query': 'children fantasy magic',
                'description': 'Fantasy books for children'
            }
        ]
        
        for example in examples:
            print(f"\n🔍 Example: {example['description']}")
            print(f"Query: '{example['query']}'")
            
            results = self.search_books(example['query'], n_results=3)
            self.display_results(results[:3])  # Show top 3 results
            
            input("\nPress Enter to continue...")
    
    def database_stats(self):
        """Show database statistics"""
        print("\n📊 Database Statistics")
        print("=" * 40)
        
        try:
            total_docs = self.collection.count()
            print(f"Total books: {total_docs:,}")
            
            # Get sample for analysis
            sample = self.collection.get(limit=min(100, total_docs), include=['metadatas'])
            
            if sample['metadatas']:
                # Language distribution
                languages = {}
                formats = {}
                years = []
                desc_count = 0
                
                for metadata in sample['metadatas']:
                    # Languages
                    lang = metadata.get('language', 'Unknown')
                    languages[lang] = languages.get(lang, 0) + 1
                    
                    # Formats
                    fmt = metadata.get('format', 'Unknown')
                    if fmt != 'Unknown':
                        formats[fmt] = formats.get(fmt, 0) + 1
                    
                    # Years
                    year = metadata.get('publish_year')
                    if year:
                        years.append(year)
                    
                    # Descriptions
                    if metadata.get('has_description'):
                        desc_count += 1
                
                # Display statistics
                print(f"\n📚 Sample Analysis (from {len(sample['metadatas'])} books):")
                
                print(f"\nTop Languages:")
                for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {lang}: {count}")
                
                if formats:
                    print(f"\nFormats:")
                    for fmt, count in sorted(formats.items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"  {fmt}: {count}")
                
                if years:
                    print(f"\nPublication Years:")
                    print(f"  Range: {min(years)} - {max(years)}")
                    print(f"  Average: {sum(years) / len(years):.0f}")
                
                print(f"\nContent Quality:")
                desc_percentage = (desc_count / len(sample['metadatas'])) * 100
                print(f"  Books with descriptions: {desc_percentage:.1f}%")
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")

def main():
    """Main interactive interface"""
    print("🚀 Library Chatbot Search Interface")
    print("=" * 50)
    
    # Initialize searcher
    try:
        searcher = LibrarySearcher()
    except Exception as e:
        print(f"❌ Failed to initialize searcher: {e}")
        return
    
    while True:
        print("\n🎮 Choose an option:")
        print("1. Quick Search")
        print("2. Advanced Search (with filters)")
        print("3. Example Searches")
        print("4. Database Statistics")
        print("5. Exit")
        
        choice = input("\n📝 Enter choice (1-5): ").strip()
        
        if choice == '1':
            searcher.quick_search_interface()
        elif choice == '2':
            searcher.advanced_search_interface()
        elif choice == '3':
            searcher.example_searches()
        elif choice == '4':
            searcher.database_stats()
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main() 