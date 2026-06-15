"""
Conversation Manager for Library Chatbot RAG System
Handles context memory for current chat session
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from config import RAGConfig

class ConversationTurn:
    """Represents a single turn in the conversation"""
    
    def __init__(self, user_message: str, assistant_response: str, 
                 search_results: Optional[List[Dict]] = None,
                 intent: Optional[str] = None,
                 thinking_process: Optional[str] = None,
                 image_embedding: Optional[List[float]] = None,
                 image_mime: Optional[str] = None,
                 image_jpeg: Optional[bytes] = None,
                 search_mode: Optional[str] = None):
        self.user_message = user_message
        self.assistant_response = assistant_response
        self.search_results = search_results or []
        self.intent = intent
        self.thinking_process = thinking_process
        # Optional image context — populated when the turn was driven by
        # an uploaded cover image. Storing the *embedding* (not the raw
        # bytes) lets follow-up turns refine the search with new filters
        # without paying a second embedding API call. The resized JPEG
        # bytes are also cached so follow-ups can keep showing the
        # cover to the LLM without re-uploading or re-preprocessing.
        self.image_embedding = image_embedding
        self.image_mime = image_mime
        self.image_jpeg = image_jpeg
        self.search_mode = search_mode
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for context building"""
        return {
            'user_message': self.user_message,
            'assistant_response': self.assistant_response,
            'intent': self.intent,
            'timestamp': self.timestamp.isoformat(),
            'has_search_results': len(self.search_results) > 0,
            'has_image_context': self.image_embedding is not None,
            'search_mode': self.search_mode,
        }

class ConversationManager:
    """Manages conversation context and memory"""
    
    def __init__(self, max_history: int = RAGConfig.MAX_CONVERSATION_HISTORY):
        self.conversation_history: List[ConversationTurn] = []
        self.max_history = max_history
        self.current_topic = None
        self.last_search_results = []
        # Last image-driven search context, retained even if the original
        # turn falls out of `conversation_history` due to max-history
        # truncation. Followups can re-issue the vector search using
        # this embedding without re-embedding the image, and re-show
        # the resized JPEG to multimodal LLM calls without re-uploading.
        self.last_image_embedding: Optional[List[float]] = None
        self.last_image_mime: Optional[str] = None
        self.last_image_jpeg: Optional[bytes] = None
        self.last_search_mode: Optional[str] = None
    
    def add_turn(self, user_message: str, assistant_response: str,
                 search_results: Optional[List[Dict]] = None,
                 intent: Optional[str] = None,
                 thinking_process: Optional[str] = None,
                 image_embedding: Optional[List[float]] = None,
                 image_mime: Optional[str] = None,
                 image_jpeg: Optional[bytes] = None,
                 search_mode: Optional[str] = None):
        """Add a new conversation turn"""
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            search_results=search_results,
            intent=intent,
            thinking_process=thinking_process,
            image_embedding=image_embedding,
            image_mime=image_mime,
            image_jpeg=image_jpeg,
            search_mode=search_mode,
        )
        
        self.conversation_history.append(turn)
        
        # Update last search results if this turn had search
        if search_results:
            self.last_search_results = search_results

        # Update last image context if this turn carried one. We only
        # overwrite when a new embedding is supplied so that a no-image
        # followup turn doesn't wipe out the cached embedding the user
        # may want to refine against.
        if image_embedding is not None:
            self.last_image_embedding = image_embedding
            self.last_image_mime = image_mime
            self.last_image_jpeg = image_jpeg
            self.last_search_mode = search_mode
        
        # Maintain history limit (but don't include thinking in stored context)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def get_context_summary(self) -> str:
        """Get a summary of recent conversation for context"""
        if not self.conversation_history:
            return "No previous conversation."
        
        context_parts = []
        
        # Add recent conversation turns (without thinking process)
        for turn in self.conversation_history[-3:]:  # Last 3 turns
            context_parts.append(f"User: {turn.user_message}")
            # Only include the response part, not thinking
            response = turn.assistant_response.split("RESPONSE:")[-1].strip() if "RESPONSE:" in turn.assistant_response else turn.assistant_response
            context_parts.append(f"Assistant: {response[:200]}...")  # Truncate for brevity
        
        return "\n".join(context_parts)
    
    def get_last_search_context(self) -> Optional[List[Dict]]:
        """Get last search results for clarification questions"""
        return self.last_search_results if self.last_search_results else None

    def get_last_image_context(self) -> Optional[Dict[str, Any]]:
        """Return the cached image-search context from a prior turn.

        Returns a dict with `embedding`, `mime`, `jpeg`, `search_mode`,
        or None when no image-driven turn is on record. Used by
        follow-up handlers so the user can refine an image-driven
        search with new filters without re-uploading the image, and so
        the resized JPEG can still be sent to multimodal LLM calls
        during followups.
        """
        if not self.last_image_embedding:
            return None
        return {
            'embedding': self.last_image_embedding,
            'mime': self.last_image_mime,
            'jpeg': self.last_image_jpeg,
            'search_mode': self.last_search_mode,
        }

    def clear_image_context(self):
        """Drop the cached image-search context.

        Useful when the user explicitly starts a new topic — the engine
        does not call this automatically because most follow-ups should
        keep the context.
        """
        self.last_image_embedding = None
        self.last_image_mime = None
        self.last_image_jpeg = None
        self.last_search_mode = None
    
    def is_followup_likely(self, user_message: str) -> bool:
        """Determine if this is likely a follow-up to previous search"""
        if not self.conversation_history:
            return False
        
        # Check if last turn had search results
        last_turn = self.conversation_history[-1]
        if not last_turn.search_results:
            return False
        
        # Simple heuristics for follow-up detection
        followup_indicators = [
            'more about', 'tell me about', 'what about', 'how about',
            'similar', 'like that', 'others', 'more', 'also',
            'first one', 'second one', 'that book', 'this book'
        ]
        
        user_lower = user_message.lower()
        return any(indicator in user_lower for indicator in followup_indicators)
    
    def extract_topic_evolution(self) -> List[str]:
        """Extract how the conversation topic has evolved"""
        topics = []
        for turn in self.conversation_history:
            if turn.intent == 'SEARCH':
                # Extract main topic from user message
                # Simple extraction - could be enhanced with NLP
                topics.append(turn.user_message[:50] + "...")
        return topics
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation"""
        if not self.conversation_history:
            return {'total_turns': 0}
        
        intents = [turn.intent for turn in self.conversation_history if turn.intent]
        search_turns = len([t for t in self.conversation_history if t.search_results])
        
        return {
            'total_turns': len(self.conversation_history),
            'search_turns': search_turns,
            'intents': intents,
            'duration_minutes': (
                self.conversation_history[-1].timestamp - 
                self.conversation_history[0].timestamp
            ).total_seconds() / 60,
            'has_recent_search': len(self.last_search_results) > 0
        }
    
    def clear_conversation(self):
        """Clear conversation history (for new session)"""
        self.conversation_history.clear()
        self.last_search_results.clear()
        self.current_topic = None
        # Drop any cached image-driven search context too — a cleared
        # session should not leak prior image embeddings into followups.
        self.last_image_embedding = None
        self.last_image_mime = None
        self.last_image_jpeg = None
        self.last_search_mode = None
    
    def format_for_llm_context(self, include_last_n: int = 2) -> str:
        """Format conversation context for LLM prompt (excluding thinking)"""
        if not self.conversation_history:
            return "This is the start of a new conversation."
        
        recent_turns = self.conversation_history[-include_last_n:]
        formatted_context = []
        
        for turn in recent_turns:
            formatted_context.append(f"Previous User Query: {turn.user_message}")
            # Extract only the response part, not the thinking
            if "RESPONSE:" in turn.assistant_response:
                response_only = turn.assistant_response.split("RESPONSE:")[-1].strip()
            else:
                response_only = turn.assistant_response
            formatted_context.append(f"Previous Assistant Response: {response_only}")
        
        return "\n".join(formatted_context) 