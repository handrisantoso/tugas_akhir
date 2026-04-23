# Library Chatbot Embedding Generation

This folder contains scripts to generate embeddings for your book catalog chunks using Google's text-embedding-004 model.

## 📁 Files Overview

- **`create_embeddings.py`** - Main script to generate embeddings for all 8,992 book chunks
- **`test_embedding.py`** - Test script to verify API setup with sample chunks
- **`config.py`** - Configuration settings and parameters
- **`requirements.txt`** - Required Python packages
- **`embedding_process_log.txt`** - Process documentation and notes

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r embedding/requirements.txt
```

### 2. Test Your Setup (Recommended)

```bash
python embedding/test_embedding.py
```

This will:

- Test your API key with sample texts
- Verify embedding generation with real book chunks
- Confirm everything is working before processing all 8,992 chunks

### 3. Generate All Embeddings

```bash
python embedding/create_embeddings.py
```

You'll be prompted to enter your Google AI Studio API key.

## 📊 What It Does

1. **Loads Book Chunks**: Reads your processed chunks from `chunking/book_chunks.json`
2. **Generates Embeddings**: Creates 768-dimensional embeddings for each book's text using Google's text-embedding-004
3. **Saves Progress**: Automatically saves every 100 chunks (resume capability)
4. **Creates Statistics**: Generates detailed processing reports

## 📁 Output Files

After running, you'll get:

- **`book_embeddings.json`** - Main output with embeddings for all chunks
- **`embedding_statistics.txt`** - Detailed processing statistics
- **Updated `embedding_process_log.txt`** - Complete process log

## ⚙️ Features

- **Resume Capability**: If interrupted, you can resume from where you left off
- **Rate Limiting**: Respects API quotas with automatic delays
- **Error Handling**: Retries failed requests with exponential backoff
- **Progress Tracking**: Real-time progress bar and statistics
- **Validation**: Automatic validation of embedding dimensions and quality

## 🔧 Configuration

You can modify settings in `config.py`:

- **Rate limiting**: Adjust delays between API calls
- **Batch sizes**: Change how often progress is saved
- **File paths**: Customize input/output locations
- **Error handling**: Set retry attempts and failure thresholds

## 📈 Expected Processing Time

For 8,992 chunks with rate limiting:

- **Estimated time**: 20-30 minutes
- **API calls**: ~9,000 requests
- **Output size**: ~30-50 MB (depending on JSON formatting)

## 🛡️ Error Handling

The script handles:

- **Rate limiting**: Automatic backoff when hitting API limits
- **Network errors**: Retries with exponential delay
- **Invalid responses**: Validation and error reporting
- **Interruptions**: Save progress and resume capability

## 🎯 Next Steps

After generating embeddings, you'll be ready for:

1. **Vector Database Setup** (Chroma, FAISS, or Pinecone)
2. **Semantic Search Implementation**
3. **RAG System Integration**
4. **Chatbot Interface Development**

## 📞 Troubleshooting

### Common Issues:

**API Key Error**: Make sure your Google AI Studio API key is valid

- Get key from: https://ai.google.dev

**Rate Limiting**: If you hit limits frequently, increase delays in `config.py`

**Memory Issues**: For very large datasets, the script processes in batches

**Resume Not Working**: Check if `book_embeddings.json` exists and is valid JSON

## 💡 Tips

- **Run test first**: Always use `test_embedding.py` before the full process
- **Monitor progress**: The script shows real-time success rates
- **Check statistics**: Review the statistics file for quality metrics
- **Backup embeddings**: The output file is valuable - consider backing it up

---

Ready to create embeddings for your 8,992 book chunks! 🚀
