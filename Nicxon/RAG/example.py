from pathlib import Path
import torch
import chainlit
from typing import List, Dict, Any
import os

# LangChain imports

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Transformers imports
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- CONFIGURATION ---
FOLDER_PATH = "documents"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
MAX_NEW_TOKENS = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- DOCUMENT LOADING WITH LANGCHAIN ---
def load_documents():
    """Load documents from folder using LangChain loaders."""
    # Create loaders for different file types
    loaders = {
        "**/*.txt": TextLoader,
        "**/*.pdf": PyMuPDFLoader,
        "**/*.docx": Docx2txtLoader,
        "**/*.xlsx": UnstructuredExcelLoader,
        "**/*.xls": UnstructuredExcelLoader,
    }
    
    all_docs = []
    
    # Ensure the documents directory exists
    if not os.path.exists(FOLDER_PATH):
        os.makedirs(FOLDER_PATH)
        print(f"Created documents directory at {FOLDER_PATH}")
        print(f"Please add some documents to {FOLDER_PATH} before running again")
        return all_docs
    
    for glob_pattern, loader_cls in loaders.items():
        try:
            loader = DirectoryLoader(
                FOLDER_PATH, 
                glob=glob_pattern, 
                loader_cls=loader_cls,
                show_progress=True,
                use_multithreading=True
            )
            docs = loader.load()
            print(f"Loaded {len(docs)} documents from {glob_pattern}")
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {glob_pattern}: {e}")
    
    if not all_docs:
        print(f"No documents found in {FOLDER_PATH}. Please add some documents.")
    
    return all_docs


# --- CHUNKING WITH LANGCHAIN ---
def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks using LangChain's RecursiveCharacterTextSplitter."""
    if not documents:
        print("No documents to chunk.")
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
    return chunked_docs


# --- VECTOR STORE WITH LANGCHAIN ---
def create_vector_store(documents: List[Document]):
    """Create a FAISS vector store from documents using LangChain."""
    if not documents:
        print("No documents to create vector store from.")
        # Create a dummy document to prevent errors
        dummy_doc = Document(page_content="This is a placeholder document.", metadata={"source": "placeholder"})
        documents = [dummy_doc]
    
    try:
        # Initialize HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Test the embeddings with a sample text
        test_embedding = embeddings.embed_query("Test embedding")
        print(f"Embedding dimension: {len(test_embedding)}")
        
        # Create vector store
        vector_store = FAISS.from_documents(documents, embeddings)
        print(f"Created vector store with {len(documents)} documents")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise


# --- LLM PIPELINE ---
class LLMPipeline:
    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading base model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
            
        self.model.to(device)
        self.model.eval()
        
        # Create pipeline for generation
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device if device.type == "cuda" else -1
        )
        
    def format_prompt(self, context: str, query: str) -> str:
        return f"<|im_start|>user\n{context}\n\nUser question: {query}<|im_end|>\n<|im_start|>assistant\n"
    
    def generate(self, context: str, query: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        prompt = self.format_prompt(context, query)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
                outputs = self.pipe(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    return_full_text=False
                )
        
        response = outputs[0]['generated_text']
        # Clean up the response
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
            
        return response.strip()


# --- INITIALIZE PIPELINES ---
print("[*] Loading documents...")
raw_docs = load_documents()

print("[*] Chunking documents...")
chunked_docs = chunk_documents(raw_docs)

print("[*] Creating vector store...")
try:
    vector_store = create_vector_store(chunked_docs)
    print("[✓] RAG pipeline ready.")
except Exception as e:
    print(f"[!] Failed to create vector store: {e}")
    print("[!] Will continue with LLM only (no RAG)")
    vector_store = None

print("[*] Initializing LLM pipeline...")
llm_pipeline = LLMPipeline(MODEL_NAME)


# --- CHAINLIT INFERENCE ---
@chainlit.on_message
async def main(message: chainlit.Message):
    torch.manual_seed(123)
    query = message.content

    # Show thinking process to user
    thinking_msg = await chainlit.Message(content="Processing your query...").send()
    
    if vector_store:
        # Update thinking message - use the correct API
        thinking_msg.content = "Searching through documents..."
        await thinking_msg.update()
        
        # Retrieve relevant chunks
        try:
            retrieved_docs = vector_store.similarity_search_with_score(query, k=TOP_K)
            
            # Format context with source information
            context_parts = []
            sources = set()
            
            for doc, score in retrieved_docs:
                source = doc.metadata.get("source", "Unknown source")
                source_name = Path(source).name
                sources.add(source_name)
                
                content = doc.page_content
                context_parts.append(f"Source: {source_name}\n{content}")
            
            context = "\n\n".join(context_parts)
        except Exception as e:
            print(f"Error during retrieval: {e}")
            context = "I couldn't retrieve relevant information from the documents."
            sources = set()
    else:
        # If no vector store, just use the query directly
        context = "No document context available."
        sources = set()
    
    # Update thinking message
    thinking_msg.content = "Generating response..."
    await thinking_msg.update()
    
    # Generate response
    response = llm_pipeline.generate(context, query)
    
    # Show sources used if any
    final_response = response
    if sources:
        source_text = "Sources: " + ", ".join(sources)
        final_response = f"{response}\n\n{source_text}"
    
    # Send final response
    await chainlit.Message(content=final_response).send()
    
    # Remove thinking message
    await thinking_msg.remove()
