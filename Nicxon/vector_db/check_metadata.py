import chromadb

client = chromadb.PersistentClient(path="vector_db/chroma_db_multimodal_2")
collection = client.get_collection(name="library_books_image")

# Peek at the first 10 items
peek_results = collection.peek()

# Inspect the metadatas list
print(peek_results["metadatas"])