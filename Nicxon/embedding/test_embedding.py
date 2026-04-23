#!/usr/bin/env python3
"""
Test script for embedding generation
Tests the API connection and generates embeddings for a small sample
"""

import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import google.generativeai as genai
except ImportError:
    print("❌ Error: google-generativeai not installed")
    print("Please install: pip install google-generativeai")
    sys.exit(1)

def test_embedding_api(api_key: str):
    """Test the embedding API with sample text"""
    print("🧪 Testing Google AI Studio Embedding API...")
    
    try:
        # Configure API
        genai.configure(api_key=api_key)
        
        # Test embedding
        test_texts = [
            "This is a test book about space exploration",
            "A mystery novel set in Victorian England",
            "Romance story with happy ending"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n📝 Test {i}: '{text[:50]}...'")
            
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text
            )
            
            if result and 'embedding' in result:
                embedding = result['embedding']
                print(f"✅ Success! Embedding dimension: {len(embedding)}")
                print(f"📊 Sample values: {embedding[:5]} ... {embedding[-5:]}")
            else:
                print("❌ Failed: Invalid response format")
                return False
        
        print("\n🎉 All tests passed! API is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_with_real_chunks(api_key: str, num_samples: int = 3):
    """Test with actual book chunks"""
    print(f"\n📚 Testing with {num_samples} real book chunks...")
    
    try:
        # Load sample chunks
        with open("chunking/book_chunks.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"📖 Loaded {len(chunks)} total chunks")
        
        # Test with first few chunks
        genai.configure(api_key=api_key)
        
        for i in range(min(num_samples, len(chunks))):
            chunk = chunks[i]
            text = chunk['text']
            
            print(f"\n📝 Testing chunk {i+1}:")
            print(f"📄 Text preview: {text[:100]}...")
            print(f"📊 Text length: {len(text)} characters")
            
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text
            )
            
            if result and 'embedding' in result:
                embedding = result['embedding']
                print(f"✅ Embedding generated: {len(embedding)} dimensions")
                
                # Show metadata
                metadata = chunk.get('metadata', {})
                print(f"📋 Has description: {metadata.get('has_description', False)}")
                print(f"📋 Field count: {metadata.get('field_count', 0)}")
            else:
                print("❌ Failed to generate embedding")
                return False
        
        print("\n🎉 Real chunk testing successful!")
        return True
        
    except FileNotFoundError:
        print("❌ Could not find chunking/book_chunks.json")
        print("Please run the chunking process first")
        return False
    except Exception as e:
        print(f"❌ Error testing real chunks: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Library Chatbot Embedding Test")
    print("=" * 40)
    
    # Get API key
    api_key = input("🔑 Enter your Google AI Studio API key: ").strip()
    
    if not api_key:
        print("❌ API key required")
        return
    
    # Run tests
    print("\n" + "="*50)
    if not test_embedding_api(api_key):
        print("❌ Basic API test failed - check your API key")
        return
    
    print("\n" + "="*50)
    if not test_with_real_chunks(api_key):
        print("❌ Real chunk test failed")
        return
    
    print("\n" + "="*50)
    print("🎉 All tests passed!")
    print("✅ Your API key and setup are working correctly")
    print("🚀 Ready to run the full embedding generation!")
    print("\nNext step: Run 'python embedding/create_embeddings.py'")

if __name__ == "__main__":
    main() 