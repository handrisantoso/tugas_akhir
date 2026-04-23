#!/usr/bin/env python3
"""
ChromaDB Installation Test
Tests if ChromaDB is properly installed and working
"""

import sys
import os

def test_chromadb_installation():
    """Test ChromaDB installation and basic functionality"""
    print("🧪 Testing ChromaDB Installation")
    print("=" * 40)
    
    # Test import
    try:
        import chromadb
        from chromadb.config import Settings
        print("✅ ChromaDB imported successfully")
        print(f"📦 ChromaDB version: {chromadb.__version__}")
    except ImportError as e:
        print("❌ ChromaDB import failed")
        print(f"Error: {e}")
        print("\n🔧 To fix this, run:")
        print("pip install chromadb")
        return False
    
    # Test basic functionality
    try:
        print("\n🔬 Testing basic functionality...")
        
        # Use in-memory client to avoid Windows file locking issues
        client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create collection
        collection = client.create_collection("test")
        print("✅ Collection created successfully")
        
        # Add test documents
        test_embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        test_documents = ["Hello world", "Goodbye world"]
        test_ids = ["doc1", "doc2"]
        
        collection.add(
            embeddings=test_embeddings,
            documents=test_documents,
            ids=test_ids
        )
        print("✅ Documents added successfully")
        
        # Test query
        results = collection.query(
            query_embeddings=[[1.1, 2.1, 3.1]],
            n_results=1
        )
        
        if results and results['documents']:
            print("✅ Query executed successfully")
            print(f"📄 Found document: {results['documents'][0][0]}")
        else:
            print("⚠️  Query returned no results")
        
        # Test count
        count = collection.count()
        print(f"✅ Collection count: {count} documents")
        
        print("\n🎉 All tests passed! ChromaDB is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB functionality test failed: {e}")
        print("\n🔧 Possible solutions:")
        print("1. Try reinstalling: pip uninstall chromadb && pip install chromadb")
        print("2. Check if you have sufficient disk space")
        print("3. Check if you have write permissions in the current directory")
        return False

def test_numpy_compatibility():
    """Test NumPy compatibility"""
    print("\n🔢 Testing NumPy compatibility...")
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
        
        # Test array operations
        test_array = np.array([1.0, 2.0, 3.0])
        test_result = np.dot(test_array, test_array)
        print(f"✅ NumPy operations working (dot product: {test_result})")
        
        return True
        
    except ImportError:
        print("❌ NumPy not installed")
        print("🔧 Install with: pip install numpy")
        return False
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
        return False

def test_file_permissions():
    """Test file system permissions"""
    print("\n📁 Testing file system permissions...")
    
    try:
        # Test write permissions
        test_dir = "vector_db/test_temp"
        os.makedirs(test_dir, exist_ok=True)
        
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        # Clean up
        os.remove(test_file)
        os.rmdir(test_dir)
        
        print("✅ File system permissions OK")
        return True
        
    except Exception as e:
        print(f"❌ File system test failed: {e}")
        print("🔧 Check if you have write permissions in the current directory")
        return False

def main():
    """Run all tests"""
    print("🚀 Library Chatbot Vector Database Compatibility Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test ChromaDB installation
    if not test_chromadb_installation():
        all_passed = False
    
    # Test NumPy compatibility
    if not test_numpy_compatibility():
        all_passed = False
    
    # Test file permissions
    if not test_file_permissions():
        all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 All tests passed! Your system is ready for vector database setup.")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r vector_db/requirements.txt")
        print("2. Create vector database: python vector_db/create_vector_db.py")
        print("3. Test search functionality: python vector_db/search_library.py")
    else:
        print("❌ Some tests failed. Please fix the issues above before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 