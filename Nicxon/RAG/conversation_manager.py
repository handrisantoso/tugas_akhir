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
                 thinking_process: Optional[str] = None):
        self.user_message = user_message
        self.assistant_response = assistant_response
        self.search_results = search_results or []
        self.intent = intent
        self.thinking_process = thinking_process
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for context building"""
        return {
            'user_message': self.user_message,
            'assistant_response': self.assistant_response,
            'intent': self.intent,
            'timestamp': self.timestamp.isoformat(),
            'has_search_results': len(self.search_results) > 0
        }

class ConversationManager:
    """Manages conversation context and memory"""
    
    def __init__(self, max_history: int = RAGConfig.MAX_CONVERSATION_HISTORY):
        self.conversation_history: List[ConversationTurn] = []
        self.max_history = max_history
        self.current_topic = None
        self.last_search_results = []
    
    def add_turn(self, user_message: str, assistant_response: str,
                 search_results: Optional[List[Dict]] = None,
                 intent: Optional[str] = None,
                 thinking_process: Optional[str] = None):
        """Add a new conversation turn"""
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            search_results=search_results,
            intent=intent,
            thinking_process=thinking_process
        )
        
        self.conversation_history.append(turn)
        
        # Update last search results if this turn had search
        if search_results:
            self.last_search_results = search_results
        
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