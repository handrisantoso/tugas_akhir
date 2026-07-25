#!/usr/bin/env python3
"""
Configuration settings for the embedding generation process
"""

# Google AI Studio / Gemini API Settings
EMBEDDING_MODEL = "models/text-embedding-004"
EXPECTED_EMBEDDING_DIM = 768

# Rate limiting settings (to avoid quota issues)
MIN_DELAY_BETWEEN_CALLS = 0.1  # seconds (100ms)
RETRY_ATTEMPTS = 3
RATE_LIMIT_BACKOFF_BASE = 5  # seconds

# Processing settings
SAVE_INTERVAL = 100  # Save progress every N chunks
PROGRESS_UPDATE_INTERVAL = 10  # Update progress bar every N chunks

# File paths
INPUT_FILE = "chunking/book_chunks.json"
OUTPUT_FILE = "embedding/book_embeddings.json"
STATISTICS_FILE = "embedding/embedding_statistics.txt"
LOG_FILE = "embedding/embedding_process_log.txt"

# Validation settings
ENABLE_EMBEDDING_VALIDATION = True
ENABLE_API_CONNECTION_TEST = True

# Error handling
MAX_FAILED_CHUNKS_BEFORE_ABORT = 100  # Stop if too many failures
ENABLE_RESUME_CAPABILITY = True

# Memory management (for large datasets)
CHUNK_BATCH_SIZE = 1000  # Process chunks in batches of this size
MEMORY_LIMIT_MB = 2048  # Rough memory limit for embeddings in memory 
