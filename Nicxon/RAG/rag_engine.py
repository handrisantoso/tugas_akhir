"""
Core RAG Engine for Library Chatbot
Handles intent classification, search, filtering, and response generation
"""
import os
import json
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
import chromadb
from google import genai
from config import RAGConfig
from conversation_manager import ConversationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LibraryRAGEngine:
    """Main RAG engine for library book search and recommendations"""
    
    def __init__(self):
        self.config = RAGConfig()
        self.conversation_manager = ConversationManager()
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all RAG components"""
        try:
            # Configure Google AI
            if not self.config.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            self.genai_client = genai.Client(api_key=self.config.GOOGLE_API_KEY)
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(path=self.config.VECTOR_DB_PATH)
            self.collection = self.chroma_client.get_collection(name=self.config.COLLECTION_NAME)
            
            logger.info("RAG Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            raise
    
    async def process_query(self, user_message: str) -> Dict[str, Any]:
        """Main entry point for processing user queries"""
        try:
            # Step 1: Intent Classification
            intent, confidence = await self._classify_intent(user_message)
            logger.info(f"Intent classified: {intent} (confidence: {confidence})")
            
            # Step 2: Process based on intent
            if intent == "SEARCH":
                return await self._handle_search_intent(user_message, intent)
            elif intent == "FOLLOWUP":
                return await self._handle_followup_intent(user_message, intent)
            elif intent == "CLARIFICATION":
                return await self._handle_clarification_intent(user_message, intent)
            else:  # GENERAL
                return await self._handle_general_intent(user_message, intent)
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._create_error_response(str(e))
    
    async def _classify_intent(self, user_message: str) -> Tuple[str, float]:
        """Classify user intent using LLM"""
        prompt = self.config.SYSTEM_PROMPTS['intent_classifier'].format(
            user_message=user_message
        )
        
        try:
            response = await self.genai_client.aio.models.generate_content(
                model=self.config.LLM_MODEL,
                contents=prompt
            )
            result = response.text.strip()
            
            # Parse result (format: INTENT|CONFIDENCE)
            if '|' in result:
                intent, confidence_str = result.split('|', 1)
                confidence = float(confidence_str)
            else:
                intent = result
                confidence = 0.8  # Default confidence
            
            return intent.strip(), confidence
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return "SEARCH", 0.5  # Default to search
    
    async def _handle_search_intent(self, user_message: str, intent: str) -> Dict[str, Any]:
        """Handle search queries"""
        # Step 1: Extract search parameters
        search_params = await self._extract_search_parameters(user_message)
        
        # Step 2: Perform vector search
        search_results = await self._perform_vector_search(
            search_params['search_terms'], 
            search_params
        )
        
        # Step 3: Post-retrieval filtering
        filtered_results = await self._post_retrieval_filter(
            user_message, 
            search_results
        )
        
        # Step 4: Generate response
        response_data = await self._generate_response(
            user_message, 
            filtered_results, 
            intent
        )
        
        # Step 5: Update conversation memory
        self.conversation_manager.add_turn(
            user_message=user_message,
            assistant_response=response_data['full_response'],
            search_results=filtered_results,
            intent=intent,
            thinking_process=response_data['thinking']
        )
        
        return response_data
    
    async def _handle_followup_intent(self, user_message: str, intent: str) -> Dict[str, Any]:
        """Handle follow-up questions using previous search context"""
        last_results = self.conversation_manager.get_last_search_context()
        
        if not last_results:
            # No previous search, treat as new search
            return await self._handle_search_intent(user_message, "SEARCH")
        
        # Use LLM to filter previous results based on follow-up
        filtered_results = await self._post_retrieval_filter(
            user_message, 
            last_results
        )
        
        response_data = await self._generate_response(
            user_message, 
            filtered_results, 
            intent
        )
        
        self.conversation_manager.add_turn(
            user_message=user_message,
            assistant_response=response_data['full_response'],
            search_results=filtered_results,
            intent=intent,
            thinking_process=response_data['thinking']
        )
        
        return response_data
    
    async def _handle_clarification_intent(self, user_message: str, intent: str) -> Dict[str, Any]:
        """Handle clarification requests about specific books"""
        # Similar to follow-up but focused on explaining/elaborating
        return await self._handle_followup_intent(user_message, intent)
    
    async def _handle_general_intent(self, user_message: str, intent: str) -> Dict[str, Any]:
        """Handle general conversation and library questions"""
        thinking = f"This is a general library question that doesn't require book search. I should respond helpfully about library services or general information."
        
        prompt = f"""You are a friendly library assistant. Respond to this general question or comment:

User: {user_message}

Provide a helpful, conversational response about library services, general book information, or engage in friendly conversation as appropriate."""

        try:
            response = await self.genai_client.aio.models.generate_content(
                model=self.config.LLM_MODEL,
                contents=prompt
            )
            assistant_response = response.text.strip()
            
            full_response = f"THINKING: {thinking}\nRESPONSE: {assistant_response}"
            
            self.conversation_manager.add_turn(
                user_message=user_message,
                assistant_response=full_response,
                intent=intent,
                thinking_process=thinking
            )
            
            return {
                'thinking': thinking,
                'response': assistant_response,
                'full_response': full_response,
                'search_results': [],
                'intent': intent
            }
            
        except Exception as e:
            logger.error(f"General response generation failed: {e}")
            return self._create_error_response(str(e))
    
    async def _extract_search_parameters(self, user_message: str) -> Dict[str, Any]:
        """Extract search terms and filters from user message"""
        prompt = self.config.SYSTEM_PROMPTS['query_processor'].format(
            user_message=user_message
        )
        
        try:
            response = await self.genai_client.aio.models.generate_content(
                model=self.config.LLM_MODEL,
                contents=prompt
            )
            result = response.text.strip()
            
            # Parse the structured response
            params = {
                'search_terms': '',
                'language_filter': None,
                'year_filter': None,
                'genre_filter': None,
                'format_filter': None,
                'page_count_filter': None,
                'content_filter': None
            }
            
            for line in result.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    # Clean up the value - remove brackets, quotes, and common artifacts
                    if value:
                        value = value.strip('[](){}""\'\'`')
                        value = value.replace('[', '').replace(']', '')
                        value = value.replace('(', '').replace(')', '')
                        value = value.strip()
                    
                    if value.lower() in ['none', '', 'none mentioned', 'not mentioned', 'no', 'na', 'n/a']:
                        value = None
                    
                    if 'search_terms' in key:
                        params['search_terms'] = value
                    elif 'language' in key:
                        params['language_filter'] = value
                    elif 'year' in key:
                        params['year_filter'] = value
                    elif 'genre' in key:
                        params['genre_filter'] = value
                    elif 'format' in key:
                        # Only keep actual format values, ignore generic terms
                        if value and value.lower() not in ['books', 'book', 'literature', 'text']:
                            params['format_filter'] = value
                        else:
                            params['format_filter'] = None
                    elif 'page_count' in key or 'page count' in key:
                        params['page_count_filter'] = value
                    elif 'content' in key:
                        params['content_filter'] = value
            
            # Log the extracted parameters
            logger.info(f"Extracted search parameters: {params}")
            return params
            
        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
            return {
                'search_terms': user_message, 
                'language_filter': None,
                'year_filter': None, 
                'genre_filter': None, 
                'format_filter': None,
                'page_count_filter': None,
                'content_filter': None
            }
    
    async def _perform_vector_search(self, query_text: str, search_params: Dict) -> List[Dict]:
        """Perform semantic search using ChromaDB with advanced filtering"""
        try:
            # Generate embedding for the query
            embedding_response = await self.genai_client.aio.models.embed_content(
                model=self.config.EMBEDDING_MODEL,
                contents=query_text
            )
            query_embedding = embedding_response.embeddings[0].values
            
            # Build comprehensive where clause for filtering
            where_clause = {}
            
            # Language filter
            if search_params.get('language_filter'):
                language = search_params['language_filter'].title()  # Use title case to match database format
                where_clause['language'] = language
            
            # Year filter - support ranges
            if search_params.get('year_filter'):
                year_filter = search_params['year_filter']
                if isinstance(year_filter, str):
                    # Parse different year formats
                    if '-' in year_filter:  # Range like "1990-2000"
                        start_year, end_year = year_filter.split('-')
                        where_clause['publish_year'] = {
                            "$gte": int(start_year),
                            "$lte": int(end_year)
                        }
                    elif 'after' in year_filter.lower() or '>' in year_filter:
                        # Extract year from "after 1990" or "> 1990"
                        import re
                        year_match = re.search(r'\d{4}', year_filter)
                        if year_match:
                            where_clause['publish_year'] = {"$gt": int(year_match.group())}
                    elif 'before' in year_filter.lower() or '<' in year_filter:
                        # Extract year from "before 2000" or "< 2000"
                        import re
                        year_match = re.search(r'\d{4}', year_filter)
                        if year_match:
                            where_clause['publish_year'] = {"$lt": int(year_match.group())}
                    else:
                        # Try to parse as single year
                        import re
                        year_match = re.search(r'\d{4}', year_filter)
                        if year_match:
                            where_clause['publish_year'] = int(year_match.group())
            
            # Page count filter
            if search_params.get('page_count_filter'):
                page_filter = search_params['page_count_filter']
                if isinstance(page_filter, str):
                    import re
                    # Parse "under 300", "less than 300", "< 300"
                    if 'under' in page_filter.lower() or 'less than' in page_filter.lower() or '<' in page_filter:
                        page_match = re.search(r'\d+', page_filter)
                        if page_match:
                            where_clause['page_count'] = {"$lt": int(page_match.group())}
                    # Parse "over 500", "more than 500", "> 500"
                    elif 'over' in page_filter.lower() or 'more than' in page_filter.lower() or '>' in page_filter:
                        page_match = re.search(r'\d+', page_filter)
                        if page_match:
                            where_clause['page_count'] = {"$gt": int(page_match.group())}
                    # Parse exact number
                    else:
                        page_match = re.search(r'\d+', page_filter)
                        if page_match:
                            where_clause['page_count'] = int(page_match.group())
            
            # Content richness filters
            if search_params.get('content_filter'):
                content_filter = search_params['content_filter'].lower()
                if 'description' in content_filter or 'detailed' in content_filter:
                    where_clause['has_description'] = True
                elif 'subjects' in content_filter or 'topics' in content_filter:
                    where_clause['has_subjects'] = True
            
            # Format filter
            if search_params.get('format_filter'):
                format_val = search_params['format_filter']
                if isinstance(format_val, str) and format_val.strip():
                    # Normalize format values
                    format_val = format_val.lower().strip()
                    if 'paperback' in format_val or 'paper' in format_val or 'soft' in format_val:
                        where_clause['format'] = 'paperback'
                    elif 'hardcover' in format_val or 'hard' in format_val or 'bound' in format_val:
                        where_clause['format'] = 'hardcover'
                    elif 'ebook' in format_val or 'digital' in format_val or 'electronic' in format_val:
                        where_clause['format'] = 'ebook'
                    # Only add if it's a recognized format, skip generic terms
                    elif format_val not in ['books', 'book', 'literature', 'text', 'reading']:
                        where_clause['format'] = format_val
            
            # Log the filters being applied
            if where_clause:
                logger.info(f"Applying filters: {where_clause}")
            
            # Perform search with filters
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=self.config.DEFAULT_SEARCH_LIMIT,
                    where=where_clause if where_clause else None,
                    include=['documents', 'metadatas', 'distances']
                )
            except Exception as db_error:
                logger.error(f"ChromaDB query failed: {db_error}")
                # Retry without filters if filtering fails
                logger.info("Retrying search without filters...")
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=self.config.DEFAULT_SEARCH_LIMIT,
                    include=['documents', 'metadatas', 'distances']
                )
            
            # Format results
            formatted_results = []
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'document': doc,
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'chunk_id': results['ids'][0][i] if 'ids' in results else f"result_{i}",
                    'applied_filters': where_clause  # Track what filters were used
                })
            
            logger.info(f"Vector search returned {len(formatted_results)} results with filters: {where_clause}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _post_retrieval_filter(self, user_query: str, search_results: List[Dict]) -> List[Dict]:
        """Use LLM to filter and rank search results"""
        if not search_results:
            return []
        
        # Prepare search results for LLM evaluation
        results_text = ""
        for i, result in enumerate(search_results[:10]):  # Limit for prompt size
            doc = result['document']
            metadata = result.get('metadata', {})
            similarity = result.get('similarity_score', 0)
            
            # Extract title from document (first line usually contains title)
            title_line = doc.split('\n')[0] if doc else "Unknown Title"
            
            results_text += f"\n{i+1}. TITLE: {title_line}\n"
            results_text += f"   SIMILARITY: {similarity:.3f}\n"
            results_text += f"   CONTENT: {doc[:400]}...\n"
            
            # Add metadata if available
            if metadata.get('language'):
                results_text += f"   LANGUAGE: {metadata['language']}\n"
            if metadata.get('publish_year'):
                results_text += f"   YEAR: {metadata['publish_year']}\n"
            
            results_text += "\n---\n"
        
        prompt = self.config.SYSTEM_PROMPTS['post_retrieval_filter'].format(
            user_query=user_query,
            search_results=results_text,
            max_results=min(5, len(search_results))
        )
        
        try:
            response = await self.genai_client.aio.models.generate_content(
                model=self.config.LLM_MODEL,
                contents=prompt
            )
            filter_response = response.text.strip()
            
            # Parse LLM response to get ranked book numbers
            ranked_results = []
            
            # Simple parsing: look for numbered lists in response
            lines = filter_response.split('\n')
            book_numbers = []
            
            for line in lines:
                # Look for patterns like "1." or "1)" or mentions of book numbers
                import re
                matches = re.findall(r'\b(\d+)\b', line)
                for match in matches:
                    book_num = int(match)
                    if 1 <= book_num <= len(search_results) and book_num not in book_numbers:
                        book_numbers.append(book_num)
                        
                        # Add explanation if found in the same line
                        explanation = line.strip()
                        if explanation:
                            # Store explanation in metadata for later use
                            search_results[book_num - 1]['llm_explanation'] = explanation
            
            # Reorder results based on LLM ranking
            for book_num in book_numbers[:5]:  # Top 5
                if book_num <= len(search_results):
                    ranked_results.append(search_results[book_num - 1])
            
            # If parsing failed or incomplete, fall back to similarity ranking
            if len(ranked_results) < 3:
                logger.warning("LLM filtering parsing incomplete, falling back to similarity ranking")
                ranked_results = sorted(search_results[:5], 
                                      key=lambda x: x.get('similarity_score', 0), 
                                      reverse=True)
            
            # Add LLM filtering metadata
            for result in ranked_results:
                result['filtered_by_llm'] = True
                result['filter_timestamp'] = logger.handlers[0].format(
                    logging.LogRecord('filter', logging.INFO, '', 0, '', (), None)
                ).split(' - ')[0] if logger.handlers else "unknown"
            
            logger.info(f"Post-retrieval filtering: {len(ranked_results)} books selected from {len(search_results)}")
            return ranked_results[:5]
            
        except Exception as e:
            logger.error(f"Post-retrieval filtering failed: {e}")
            # Fallback to similarity-based ranking
            fallback_results = sorted(search_results[:5], 
                                    key=lambda x: x.get('similarity_score', 0), 
                                    reverse=True)
            
            for result in fallback_results:
                result['filtered_by_llm'] = False
                result['fallback_reason'] = str(e)
            
            return fallback_results
    
    async def _generate_response(self, user_query: str, relevant_books: List[Dict], intent: str) -> Dict[str, Any]:
        """Generate final response using LLM"""
        # Prepare book information
        books_text = ""
        for book in relevant_books:
            books_text += f"- {book['document']}\n"
        
        # Get conversation context
        conversation_context = self.conversation_manager.format_for_llm_context()
        
        prompt = self.config.SYSTEM_PROMPTS['response_generator'].format(
            user_query=user_query,
            relevant_books=books_text,
            conversation_context=conversation_context
        )
        
        try:
            response = await self.genai_client.aio.models.generate_content(
                model=self.config.LLM_MODEL,
                contents=prompt
            )
            full_response = response.text.strip()
            
            # Parse thinking and response sections
            if "THINKING:" in full_response and "RESPONSE:" in full_response:
                thinking = full_response.split("THINKING:")[1].split("RESPONSE:")[0].strip()
                response_text = full_response.split("RESPONSE:")[-1].strip()
            else:
                thinking = "Processing the user's query and searching for relevant books."
                response_text = full_response
            
            return {
                'thinking': thinking,
                'response': response_text,
                'full_response': full_response,
                'search_results': relevant_books,
                'intent': intent
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._create_error_response(str(e))
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        thinking = f"An error occurred while processing the request: {error_message}"
        response = "I apologize, but I encountered an issue while searching for books. Please try rephrasing your question or contact the librarian for assistance."
        
        return {
            'thinking': thinking,
            'response': response,
            'full_response': f"THINKING: {thinking}\nRESPONSE: {response}",
            'search_results': [],
            'intent': 'ERROR'
        }
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get current conversation statistics"""
        return self.conversation_manager.get_conversation_stats()
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_manager.clear_conversation()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and configuration"""
        config_status = self.config.validate_config()
        conversation_stats = self.get_conversation_stats()
        
        return {
            'config_valid': config_status['valid'],
            'config_issues': config_status.get('issues', []),
            'conversation_stats': conversation_stats,
            'database_collection': self.config.COLLECTION_NAME,
            'models_used': {
                'embedding': self.config.EMBEDDING_MODEL,
                'llm': self.config.LLM_MODEL
            }
        } 