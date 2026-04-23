# Library Chatbot Vector Database

This module creates and manages a **ChromaDB vector database** for semantic search of your library's book collection. It uses the embeddings generated from your 8,992 processed books to enable intelligent book discovery.

## 🎯 **Why ChromaDB?**

- **Perfect Scale**: Optimized for your 8,992 book collection
- **Metadata Filtering**: Advanced filtering by language, year, genre, etc.
- **Local Storage**: No external dependencies or costs
- **Easy Setup**: Simple installation and configuration
- **Production Ready**: Scalable to larger collections

## 📁 **Files Overview**

```
vector_db/
├── requirements.txt          # Dependencies
├── test_chroma.py           # Installation test
├── create_vector_db.py      # Main database creator
├── search_library.py        # Interactive search tool
├── vector_db_process_log.txt # Process documentation
└── README.md               # This file
```

## 🚀 **Quick Start**

### 1. Install Dependencies

```bash
pip install -r vector_db/requirements.txt
```

### 2. Test Installation

```bash
python vector_db/test_chroma.py
```

### 3. Create Vector Database

```bash
python vector_db/create_vector_db.py
```

### 4. Test Search Functionality

```bash
python vector_db/search_library.py
```

## 📋 **Detailed Setup Instructions**

### Prerequisites

- ✅ Book embeddings generated (from `embedding/book_embeddings.json`)
- ✅ Python 3.7+ installed
- ✅ Sufficient disk space (~50-100MB for database)

### Step 1: Validate System Compatibility

```bash
python vector_db/test_chroma.py
```

**Expected Output:**

```
🎉 All tests passed! Your system is ready for vector database setup.
```

### Step 2: Create the Vector Database

```bash
python vector_db/create_vector_db.py
```

**What This Does:**

- Loads 8,992 book embeddings from JSON file
- Creates persistent ChromaDB collection
- Extracts and indexes metadata for filtering
- Populates database with embeddings and text
- Runs comprehensive tests and generates statistics
- **Processing Time**: 5-10 minutes

**Expected Output:**

```
🎉 Vector database setup completed successfully!
📁 Database location: vector_db/chroma_db
📊 Total documents: 8,992
🚀 Ready for semantic search and RAG implementation!
```

### Step 3: Test Search Capabilities

```bash
python vector_db/search_library.py
```

**Interactive Menu:**

1. **Quick Search** - Simple text-based search
2. **Advanced Search** - Search with filters (language, year, etc.)
3. **Example Searches** - Pre-configured demos
4. **Database Statistics** - Collection analytics

## 🔍 **Search Capabilities**

### Basic Semantic Search

```python
# Find books similar to query
results = searcher.search_books("mystery detective novels", n_results=10)
```

### Advanced Filtered Search

```python
# Find English mystery books under 300 pages
filters = {
    "language": "English",
    "page_count": {"$lt": 300}
}
results = searcher.search_books("mystery novels", filters=filters)
```

### Available Filters

- **language**: `"English"`, `"Spanish"`, `"French"`, `"German"`, `"Italian"`, `"Portuguese"`, `"Russian"`, `"Chinese"`, `"Japanese"`, `"Arabic"`, `"Hindi"`, and 100+ other languages
- **publish_year**: `2000`, `{"$gt": 1990}`, `{"$lt": 2010}`
- **page_count**: `{"$lt": 300}`, `{"$gt": 500}`
- **has_description**: `True` (only books with descriptions)
- **format**: `"Paperback"`, `"Hardcover"`, etc.

## 📊 **Database Schema**

### Collection: `library_books`

- **IDs**: `book_0`, `book_1`, ..., `book_8991`
- **Embeddings**: 768-dimensional vectors (Google text-embedding-004)
- **Documents**: Full text chunks with book information
- **Metadata**: Structured data for filtering

### Metadata Fields

```json
{
  "chunk_id": 0,
  "title": "Book Title",
  "author": "Author Name",
  "language": "English",
  "publish_year": 2020,
  "page_count": 250,
  "format": "Paperback",
  "has_description": true,
  "has_subjects": true,
  "field_count": 15
}
```

## 🎯 **Example Searches**

### 1. Genre-Based Search

```
Query: "mystery detective novels"
Results: Crime fiction, detective stories, mystery thrillers
```

### 2. Topic Search

```
Query: "space exploration science fiction"
Results: Sci-fi books about space travel, astronomy, future tech
```

### 3. Filtered Search

```
Query: "romance books"
Filters: language="English", publish_year>2000
Results: Modern English romance novels
```

## 📈 **Performance Characteristics**

- **Search Speed**: ~50-100ms per query
- **Database Size**: ~50-100MB for 8,992 books
- **Memory Usage**: ~200-500MB during operation
- **Scalability**: Can handle 10K-100K documents efficiently

## 🔧 **Troubleshooting**

### Common Issues

**1. "ChromaDB import failed"**

```bash
pip install chromadb>=0.4.0
```

**2. "Database not found"**

- Run `python vector_db/create_vector_db.py` first
- Check that `embedding/book_embeddings.json` exists

**3. "No results found"**

- Try simpler queries ("fantasy", "mystery")
- Check database population was successful
- Verify embeddings were created correctly

**4. "Permission denied"**

- Ensure write permissions in current directory
- Run as administrator if necessary

### Performance Optimization

**For Faster Searches:**

- Use specific language filters
- Limit result count (n_results=5 instead of 50)
- Use boolean filters rather than range queries when possible

**For Memory Efficiency:**

- Close search interface when not in use
- Restart if running many consecutive searches

## 🔄 **Database Maintenance**

### Reset Database

```bash
# The script will ask if you want to reset existing data
python vector_db/create_vector_db.py
```

### Update Embeddings

```bash
# Regenerate embeddings first, then recreate database
python embedding/create_embeddings.py
python vector_db/create_vector_db.py
```

### Backup Database

```bash
# Copy the entire database folder
cp -r vector_db/chroma_db vector_db/chroma_db_backup
```

## 🚀 **Next Steps**

After successful vector database setup:

1. **✅ Semantic Search**: Working and tested
2. **🔄 RAG Integration**: Connect to Gemini Flash 2.0
3. **🎨 Chat Interface**: Build conversational UI
4. **📱 Web Application**: Create user-friendly frontend
5. **🔧 Advanced Features**: Query expansion, conversation memory

## 📞 **Support**

If you encounter issues:

1. Run `python vector_db/test_chroma.py` for diagnostics
2. Check `vector_db/vector_db_process_log.txt` for documentation
3. Verify all prerequisites are met
4. Try reinstalling dependencies

---

🎉 **Ready to power your library chatbot with intelligent semantic search!**
