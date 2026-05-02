import os
import json
import requests
import pandas as pd
import chainlit as cl
import chromadb
import  time

from typing import Any, List
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.embeddings import BaseEmbedding

# ==========================================
# 1. CUSTOM OPENROUTER EMBEDDING CLASS
# ==========================================
class OpenRouterEmbedding(BaseEmbedding):
    """Custom class to force LlamaIndex to use OpenRouter's Embedding API."""
    api_key: str
    model_name: str

    def __init__(self, api_key: str, model_name: str, **kwargs: Any):
        super().__init__(api_key=api_key, model_name=model_name, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "OpenRouterEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embeddings([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        import time
        embeddings = []
        
        for text in texts:
            safe_text = text if text.strip() else "empty document"
            success = False
            max_retries = 5
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model_name,
                            "input": safe_text, 
                            "encoding_format": "float"
                        },
                        timeout=30
                    )
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    wait = 5 * (attempt + 1)
                    print(f"Connection error: {type(e).__name__}. Waiting {wait}s before retry ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                
                if response.status_code == 200:
                    resp_json = response.json()
                    
                    if "error" in resp_json:
                        print(f"API error: {resp_json['error'].get('message', 'Unknown')}. Retrying... ({attempt+1}/{max_retries})")
                        time.sleep(2)
                        continue
                    
                    data = resp_json.get("data", [])
                    
                    if data:
                        embeddings.append(data[0]["embedding"])
                        success = True
                        break
                    else:
                        print(f"Warning: OpenRouter returned empty data. Retrying... ({attempt+1}/{max_retries})")
                        time.sleep(2) 
                elif response.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"Rate limited. Pausing for {wait} seconds...")
                    time.sleep(wait)
                else:
                    print(f"API Error {response.status_code}. Retrying... ({attempt+1}/{max_retries})")
                    time.sleep(2)
            
            if not success:
                dim = len(embeddings[0]) if embeddings else 768
                print(f"Failed to embed document: '{safe_text[:40]}...'. Skipping and injecting blank {dim}-dim vector.")
                embeddings.append([0.0] * dim)
                
        return embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)\

# ==========================================
# 2. ENVIRONMENT & LLM SETUP
# ==========================================
os.environ["OPENROUTER_API_KEY"] = ""

# The Chatbot
Settings.llm = OpenAILike(
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    model="google/gemini-2.5-flash", 
    is_chat_model=True,
    max_tokens=1000,
)

Settings.embed_model = OpenRouterEmbedding(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model_name="qwen/qwen3-embedding-8b" 
)

# ==========================================
# 3. SYSTEM PROMPTS (DUAL-PERSONA)
# ==========================================
PROMPTS = {
    "customer": """
    You are a polite, helpful customer service assistant for our Shopee store.
    Answer the user's question using ONLY the provided context. 
    Pay close attention to the [LIVE INVENTORY STATUS] for accurate pricing and stock.
    DO NOT discuss internal analytics, negative review diagnostics, or store policies not listed.
    If the context does not contain the answer, politely say you don't have that information.
    """,
    
    "owner": """
    You are an internal Product Manager and Diagnostic Assistant for this Shopee store.
    Your goal is to analyze the provided context—especially customer reviews and feedback—to 
    identify product weaknesses, summarize complaints, and suggest actionable business improvements.
    You have full access to all positive and negative feedback in the context.
    """
}

# ==========================================
# 4. DATA PREPARATION & DATABASE FUNCTIONS
# ==========================================
def load_json_to_df(filename):
    filepath = f"data_exports/{filename}.json"
    if not os.path.exists(filepath):
        return pd.DataFrame()
    with open(filepath, 'r', encoding='utf-8') as f:
        return pd.DataFrame(json.load(f))

def load_live_inventory():
    """Loads Bucket B into a fast dictionary for real-time interception."""
    try:
        with open("data_exports/bucket_B_models.json", "r", encoding='utf-8') as f:
            models_data = json.load(f)
            return {str(item["item_id"]): item for item in models_data}
    except FileNotFoundError:
        print("Warning: Live inventory data (Bucket B) not found.")
        return {}

def initialize_database():
    """Creates documents from JSON and pushes them to ChromaDB in safe batches."""
    print("Checking database status...")
    db_path = "./chroma_db"
    
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection("shopee_knowledge_base")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # If the collection already has documents, we don't need to rebuild it
    if chroma_collection.count() > 0:
        print(f"Database loaded! Found {chroma_collection.count()} embedded documents.")
        return VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
    print("Database is empty. Parsing JSON files...")
    df_catalog = load_json_to_df("bucket_A_catalog")
    df_comments = load_json_to_df("bucket_C_comments")
    
    documents = []
    
    # Global Store Doc
    shop_doc = Document(
        text="This is our official Shopee store. We sell a variety of products. Ask about our stock, prices, or reviews.",
        metadata={"type": "shop_info", "item_id": "GLOBAL"}
    )
    shop_doc.excluded_embed_metadata_keys = ["type", "item_id"]
    documents.append(shop_doc)
    
    # Product Docs (Bucket A)
    if not df_catalog.empty:
        for _, row in df_catalog.iterrows():
            item_id = str(row.get('item_id', ''))
            name = str(row.get('item_name', 'Unknown'))
            desc = str(row.get('description', '')).replace('\n', ' ')
            doc = Document(
                text=f"Product Name: {name}. Description: {desc}",
                metadata={"type": "product_info", "item_id": item_id, "item_name": name}
            )
            doc.excluded_embed_metadata_keys = ["type", "item_id"]
            documents.append(doc)
            
    # Review Docs (Bucket C)
    if not df_comments.empty:
        id_to_name = dict(zip(df_catalog['item_id'].astype(str), df_catalog['item_name']))
        for _, row in df_comments.iterrows():
            item_id = str(row.get('item_id', ''))
            item_name = id_to_name.get(item_id, "this product")
            comment_text = str(row.get('comment', '')).replace('\n', ' ').strip()
            rating = float(row.get('rating_star', 0))
            if not comment_text: continue
            
            sentiment = "positive" if rating >= 4 else ("neutral" if rating == 3 else "negative")
            doc = Document(
                text=f"Review for '{item_name}' ({rating}/5 stars): '{comment_text}'",
                metadata={"type": "review", "item_id": item_id, "sentiment": sentiment}
            )
            doc.excluded_embed_metadata_keys = ["type", "item_id", "sentiment"]
            documents.append(doc)
            
    print(f"Ready to embed {len(documents)} documents. Starting paced ingestion...")
    
    # --- NEW PACED INGESTION LOGIC ---
    index = None
    batch_size = 50 
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(documents) + batch_size - 1) // batch_size
        print(f"Embedding batch {batch_num}/{total_batches}... (Items {i} to {i + len(batch)})")
        
        if index is None:
            # The very first batch initializes the database
            index = VectorStoreIndex.from_documents(batch, storage_context=storage_context)
        else:
            # Subsequent batches are inserted one by one
            for doc in batch:
                index.insert(doc)
        
        # Pause between batches to avoid rate limits / connection resets
        if i + batch_size < len(documents):
            time.sleep(2)
            
    print("Database successfully built and saved!")
    return index

# Initialize global engine resources
LIVE_INVENTORY = load_live_inventory()
INDEX = initialize_database()
RETRIEVER = INDEX.as_retriever(similarity_top_k=20) # Fetch the top 5 most relevant chunks

# ==========================================
# 5. CHAINLIT USER INTERFACE & LOGIC
# ==========================================
@cl.on_chat_start
async def start_chat():
    cl.user_session.set("mode", "customer")
    
    actions = [
        cl.Action(name="switch_mode", payload={"value": "customer"}, label="👤 Customer Mode"),
        cl.Action(name="switch_mode", payload={"value": "owner"}, label="🛠️ Shop Owner Mode")
    ]
    
    await cl.Message(
        content="Welcome to the Shopee AI Assistant! Please select your mode to begin:", 
        actions=actions
    ).send()

@cl.action_callback("switch_mode")
async def on_action(action: cl.Action):
    selected_mode = action.payload.get("value")
    
    cl.user_session.set("mode", selected_mode)
    mode_name = "Customer" if selected_mode == "customer" else "Shop Owner"
    
    await cl.Message(content=f"Switched to **{mode_name} Mode**. How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):
    mode = cl.user_session.get("mode", "customer")
    
    # 1. Retrieve static documents from ChromaDB
    retrieved_nodes = RETRIEVER.retrieve(message.content)
    
    # 2. The Interceptor: Build context and inject live data
    context_builder = "Here is the retrieved information from the database:\n\n"
    
    for node in retrieved_nodes:
        # Filter: If in customer mode, DO NOT inject negative reviews
        if mode == "customer" and node.metadata.get("sentiment") == "negative":
            continue 
            
        context_builder += f"- {node.text}\n"
        
        # Live Data Injection
        item_id = str(node.metadata.get("item_id", ""))
        if item_id and item_id in LIVE_INVENTORY:
            live_data = LIVE_INVENTORY[item_id]
            variations = live_data.get("models", [])
            
            context_builder += "  [LIVE INVENTORY STATUS]:\n"
            for var in variations:
                sku_name = var.get("model_sku", "Default")
                stock = var.get("model_stock_info", [{}])[0].get("current_stock", 0) if isinstance(var.get("model_stock_info"), list) else var.get("model_stock_info", {}).get("current_stock", 0)
                price = var.get("price_info", [{}])[0].get("current_price", 0) if isinstance(var.get("price_info"), list) else var.get("price_info", {}).get("current_price", 0)
                context_builder += f"   * Variation: {sku_name} | Stock: {stock} | Price: Rp {price}\n"
        context_builder += "\n"

    # 3. Assemble the final prompt
    system_prompt = PROMPTS[mode]
    final_prompt = f"{system_prompt}\n\nUser Question: {message.content}\n\nContext:\n{context_builder}"
    
    # 4. Stream the response back to the UI
    msg = cl.Message(content="")
    response = Settings.llm.stream_complete(final_prompt)
    
    for token in response:
        await msg.stream_token(token.delta)
        
    await msg.send()