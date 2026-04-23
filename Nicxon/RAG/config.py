"""
Configuration settings for Library Chatbot RAG System
"""
import os
from typing import Dict, Any

class RAGConfig:
    """Configuration class for RAG system settings"""
    
    # API Configuration
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    EMBEDDING_MODEL = "gemini-embedding-001"
    LLM_MODEL = "gemini-2.5-flash"
    
    # Database Configuration
    VECTOR_DB_PATH = "vector_db/chroma_db"
    COLLECTION_NAME = "library_books"
    
    # Search Configuration
    DEFAULT_SEARCH_LIMIT = 10
    MAX_SEARCH_LIMIT = 50
    SIMILARITY_THRESHOLD = 0.1
    
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

Evaluate each book for:
1. Relevance to user's specific request
2. Content quality and completeness
3. Appropriateness for the user's needs

Return the top {max_results} most relevant books with brief explanations of why each is relevant.
Format as a numbered list with book titles and relevance explanations.""",

        'response_generator': """You are a knowledgeable and friendly library assistant. Generate a helpful response based on the user's query and the relevant books found.

User Query: {user_query}
Relevant Books: {relevant_books}
Conversation Context: {conversation_context}

Guidelines:
1. Be conversational and helpful like a real librarian
2. Highlight the most relevant books first
3. Include publication details, ISBN, and acquisition information when available
4. Offer follow-up suggestions or related searches
5. If no perfect matches, suggest alternatives or broader searches

Provide your thinking process in a THINKING section, then your response in a RESPONSE section.

Format:
THINKING: [Your reasoning process about the query and results]
RESPONSE: [Your helpful response to the user]"""
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
                'embedding_model': cls.EMBEDDING_MODEL,
                'llm_model': cls.LLM_MODEL,
                'vector_db_path': cls.VECTOR_DB_PATH,
                'collection_name': cls.COLLECTION_NAME
            }
        } 