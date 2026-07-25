"""
Configuration settings for Library Chatbot RAG System
"""
import os
from typing import Dict, Any

class RAGConfig:
    """Configuration class for RAG system settings"""
    
    # API Configuration
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    LLM_MODEL = "gemini-3.1-flash-lite"
    
    # ------------------------------------------------------------------
    # Embedding backend selection
    # ------------------------------------------------------------------
    # Set EMBEDDING_BACKEND env var to "google" (default) or "qwen"
    # before launching the app to choose which embedding model & vector
    # database to use.  LLM calls always go through Google Gemini.
    EMBEDDING_BACKEND = os.getenv('EMBEDDING_BACKEND', 'google').lower()
    
    BACKEND_PROFILES = {
        'google': {
            'embedding_model': 'gemini-embedding-2',
            'embedding_dim': 3072,
            'vector_db_path': 'vector_db/chroma_db_multimodal_google',
            'query_embed_type': 'gemini_api',    # uses Google GenAI SDK
            'display_name': 'Google gemini-embedding-2',
        },
        'qwen': {
            'embedding_model': 'Qwen/Qwen3-VL-Embedding-2B',
            'embedding_dim': 2048,
            'vector_db_path': 'vector_db/chroma_db_multimodal_qwen',
            'query_embed_type': 'qwen_local',    # runs locally via sentence-transformers
            'display_name': 'Qwen3-VL-Embedding-2B (local)',
        },
        'jina': {
            'embedding_model': 'jinaai/jina-embeddings-v5-omni-small',
            'embedding_dim': 1024,
            'vector_db_path': 'vector_db/chroma_db_multimodal_jina',
            'query_embed_type': 'jina_local',    # SentenceTransformer: encode_query + encode_document
            'display_name': 'Jina Embeddings v5 Omni Small (local)',
        },
    }
    
    # Resolve active profile
    _profile = BACKEND_PROFILES.get(EMBEDDING_BACKEND)
    if _profile is None:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND='{EMBEDDING_BACKEND}'. "
            f"Valid values: {list(BACKEND_PROFILES.keys())}"
        )
    
    EMBEDDING_MODEL = _profile['embedding_model']
    EMBEDDING_DIM = _profile['embedding_dim']
    VECTOR_DB_PATH = _profile['vector_db_path']
    QUERY_EMBED_TYPE = _profile['query_embed_type']
    EMBEDDING_DISPLAY_NAME = _profile['display_name']
    
    # Database Configuration
    TEXT_COLLECTION_NAME = "library_books_text"
    IMAGE_COLLECTION_NAME = "library_books_image"
    
    # Search Configuration
    DEFAULT_SEARCH_LIMIT = 10
    MAX_SEARCH_LIMIT = 50
    # Cosine similarity floor for keeping a candidate. The collections
    # are now built with `hnsw:space=cosine`, so similarity =
    # `1 - distance` is true cosine in [-1, 1]. 0.4 is a conservative
    # cross-modal floor — same-modal relevant hits typically score
    # above 0.6, cross-modal relevant hits often sit in 0.4–0.7. Lower
    # this if you'd rather see weak matches than empty results.
    # NOTE: not currently enforced anywhere in the engine; reserved for
    # future filtering. Tune before wiring up.
    SIMILARITY_THRESHOLD = 0.4

    # Multimodal Search Configuration
    TEXT_SEARCH_WEIGHT = 0.7   # Weight for text similarity in hybrid scoring
    IMAGE_SEARCH_WEIGHT = 0.3  # Weight for image similarity in hybrid scoring
    MAX_UPLOAD_IMAGE_DIMENSION = 512  # Max px for uploaded query images (same as embedding pipeline)
    
    # Response Configuration
    MAX_CONTEXT_LENGTH = 8000
    MAX_CONVERSATION_HISTORY = 5
    
    # Intent Classification
    INTENT_CONFIDENCE_THRESHOLD = 0.7
    
    # Supported Languages for filtering (matches comprehensive language mapping in vector_db)
    SUPPORTED_LANGUAGES = [
        # Major European Languages
        'english', 'spanish', 'french', 'german', 'italian', 'portuguese', 'russian',
        'dutch', 'swedish', 'norwegian', 'danish', 'finnish', 'polish', 'czech', 
        'hungarian', 'greek', 'turkish', 'romanian', 'ukrainian', 'bulgarian',
        'croatian', 'serbian', 'slovenian', 'slovak', 'belarusian', 'lithuanian',
        'latvian', 'estonian', 'icelandic', 'irish', 'welsh', 'scots', 'scottish gaelic',
        'basque', 'catalan', 'galician', 'albanian', 'macedonian', 'maltese',
        
        # Asian Languages  
        'chinese', 'japanese', 'korean', 'hindi', 'arabic', 'thai', 'vietnamese',
        'indonesian', 'malay', 'bengali', 'gujarati', 'marathi', 'punjabi',
        'tamil', 'telugu', 'kannada', 'malayalam', 'sinhalese', 'nepali',
        'tibetan', 'burmese', 'khmer', 'lao', 'mongolian', 'persian', 'pashto',
        'kurdish', 'armenian', 'georgian', 'azerbaijani', 'kazakh', 'kyrgyz',
        'tajik', 'turkmen', 'uzbek', 'urdu',
        
        # African Languages
        'swahili', 'afrikaans', 'amharic', 'hausa', 'yoruba', 'igbo', 'zulu',
        'xhosa', 'sesotho', 'setswana', 'shona', 'ndebele', 'somali', 'oromo',
        'tigrinya', 'kinyarwanda', 'kirundi', 'luganda', 'wolof',
        
        # Classical/Constructed Languages  
        'latin', 'esperanto', 'yiddish', 'hebrew'
    ]
    
    # Response Templates
    SYSTEM_PROMPTS = {
        'search_mode_classifier': """You are classifying what type of search to perform for a library chatbot.

The user has uploaded an IMAGE and also written this text message: "{user_message}"

Classify the search mode as exactly one of:
- IMAGE: The text is empty, generic ("find books like this", "similar books", "what is this"), or only refers to the uploaded image. Use the image to find visually similar books.
- TEXT: The text contains specific search terms, topics, genres, authors, or keywords that should drive the search. The image is supplementary context.
- HYBRID: The text adds meaningful search context beyond just "find similar" AND the image should also influence results.

Respond with ONLY one word: IMAGE, TEXT, or HYBRID""",

        'intent_classifier': """You are an intent classifier for a library chatbot. Analyze the user's message and classify it into one of these categories:

SEARCH - User wants to find books (keywords: "find", "recommend", "looking for", "about", "book on", etc.)
CLARIFICATION - User is asking for more details about a previous search or book
GENERAL - General conversation, greetings, or library-related questions
FOLLOWUP - User wants to continue previous search with modifications

Respond with ONLY the intent category (SEARCH/CLARIFICATION/GENERAL/FOLLOWUP) and confidence (0.0-1.0).
Format: INTENT|CONFIDENCE

User message: {user_message}""",

        'query_processor': """Extract search terms and filters from this library search request:

User: {user_message}

Extract:
1. SEARCH_TERMS: Main keywords for semantic search
2. LANGUAGE_FILTER: Specific language mentioned (or none)
3. YEAR_FILTER: Publication year/range mentioned (e.g., "1990", "after 1990", "1990-2000", "before 2000") (or none)
4. GENRE_FILTER: Specific genres mentioned (or none)
5. FORMAT_FILTER: Physical format mentioned (e.g., "paperback", "hardcover", "ebook") (or none)
6. PAGE_COUNT_FILTER: Page count mentioned (e.g., "under 300", "over 500", "less than 200") (or none)
7. CONTENT_FILTER: Content richness mentioned (e.g., "with descriptions", "detailed books", "books with subjects") (or none)

Respond in this exact format:
SEARCH_TERMS: [terms]
LANGUAGE_FILTER: [language or none]
YEAR_FILTER: [year/range or none]
GENRE_FILTER: [genres or none]
FORMAT_FILTER: [format or none]
PAGE_COUNT_FILTER: [page count constraint or none]
CONTENT_FILTER: [content requirement or none]""",

        'post_retrieval_filter': """You are a librarian evaluating search results. Given the user's query and search results, rank and filter the most relevant books.

User Query: {user_query}
Search Results: {search_results}

Rank books by:
1. Match to user intent
2. Match to search terms
3. Availability of supporting metadata

Prefer books whose descriptions,
subjects, and contents directly support
answering the user's question.

Return the top {max_results} most relevant books with brief explanations of why each is relevant.
Format as a numbered list with book titles and relevance explanations.""",

        'response_generator': """You are a library assistant answering questions about books.

You must base your response ONLY on information present in the retrieved book records.

User Query: {user_query}
Relevant Books: {relevant_books}
Conversation Context: {conversation_context}

Rules:

1. Treat the retrieved records as the source of truth.
2. Do not add facts from your own knowledge.
3. Do not infer details that are not explicitly supported by the retrieved records.
4. If information is missing, clearly state that the retrieved record does not provide that information.
5. When describing a book, prioritize:

   * Description
   * Subjects & Topics
   * Table of Contents
   * Publication Information
6. If a conclusion is inferred from subjects or table of contents rather than directly stated, use language such as:

   * "The record suggests..."
   * "Based on the listed subjects..."
   * "The table of contents indicates..."
7. Do not invent plot details, author background, themes, awards, popularity, or reviews.

Response style:

* Friendly and concise.
* Prefer accuracy over completeness.
* When uncertain, acknowledge uncertainty.
"""
    }

    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        
        if not cls.GOOGLE_API_KEY:
            issues.append("GOOGLE_API_KEY environment variable not set")
        
        if not os.path.exists(cls.VECTOR_DB_PATH):
            issues.append(f"Vector database not found at {cls.VECTOR_DB_PATH}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'config_summary': {
                'embedding_backend': cls.EMBEDDING_BACKEND,
                'embedding_display_name': cls.EMBEDDING_DISPLAY_NAME,
                'embedding_model': cls.EMBEDDING_MODEL,
                'llm_model': cls.LLM_MODEL,
                'vector_db_path': cls.VECTOR_DB_PATH,
                'text_collection': cls.TEXT_COLLECTION_NAME,
                'image_collection': cls.IMAGE_COLLECTION_NAME
            }
        } 