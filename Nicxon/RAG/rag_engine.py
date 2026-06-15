"""
Core RAG Engine for Library Chatbot
Handles intent classification, search, filtering, and response generation
"""
import asyncio
import os
import io
import json
import logging
import re
from datetime import datetime
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
            # Configure Google AI — always needed for LLM calls
            if not self.config.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            self.genai_client = genai.Client(api_key=self.config.GOOGLE_API_KEY)
            
            # ----------------------------------------------------------
            # Embedding backend setup
            # ----------------------------------------------------------
            self.qwen_model = None  # sentence-transformers model (qwen)
            self.jina_model = None  # AutoModel (jina)
            
            if self.config.QUERY_EMBED_TYPE == 'qwen_local':
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    raise ImportError(
                        "sentence-transformers is required for the "
                        f"{self.config.EMBEDDING_BACKEND} embedding backend.  "
                        "Install with:\n"
                        "  conda install -c conda-forge sentence-transformers"
                    )
                
                logger.info(
                    f"Loading embedding model: {self.config.EMBEDDING_MODEL} ..."
                )
                self.qwen_model = SentenceTransformer(
                    self.config.EMBEDDING_MODEL, trust_remote_code=True
                )
                logger.info(
                    f"Model loaded — dim="
                    f"{self.qwen_model.get_sentence_embedding_dimension()}"
                )
            
            elif self.config.QUERY_EMBED_TYPE == 'jina_local':
                # Jina CLIP v2 compat patch
                try:
                    import torch
                    import torch.nn as nn
                    import transformers.models.clip.modeling_clip as _clip_module
                    if not hasattr(_clip_module, "clip_loss"):
                        def _clip_loss(similarity: torch.Tensor) -> torch.Tensor:
                            caption_loss = nn.functional.cross_entropy(
                                similarity,
                                torch.arange(len(similarity), device=similarity.device))
                            image_loss = nn.functional.cross_entropy(
                                similarity.t(),
                                torch.arange(len(similarity), device=similarity.device))
                            return (caption_loss + image_loss) / 2.0
                        _clip_module.clip_loss = _clip_loss
                except Exception:
                    pass
                
                try:
                    from transformers import AutoModel as _AutoModel
                except ImportError:
                    raise ImportError(
                        "transformers is required for the Jina backend."
                    )
                
                import torch as _t
                logger.info(
                    f"Loading Jina model: {self.config.EMBEDDING_MODEL} ..."
                )
                # Single AutoModel: encode_text() for text, encode_image() for images
                self.jina_model = _AutoModel.from_pretrained(
                    self.config.EMBEDDING_MODEL, trust_remote_code=True
                )
                if _t.cuda.is_available():
                    self.jina_model = self.jina_model.to("cuda")
                    logger.info(f"Jina model loaded on GPU — dim={self.config.EMBEDDING_DIM}")
                else:
                    logger.info(f"Jina model loaded on CPU — dim={self.config.EMBEDDING_DIM}")
            
            # Initialize ChromaDB — two collections for text and image embeddings
            self.chroma_client = chromadb.PersistentClient(path=self.config.VECTOR_DB_PATH)
            self.text_collection = self.chroma_client.get_collection(
                name=self.config.TEXT_COLLECTION_NAME
            )
            self.image_collection = self.chroma_client.get_collection(
                name=self.config.IMAGE_COLLECTION_NAME
            )
            
            logger.info(
                f"RAG Engine initialized — "
                f"backend: {self.config.EMBEDDING_BACKEND}, "
                f"embedding: {self.config.EMBEDDING_DISPLAY_NAME}, "
                f"text: {self.text_collection.count()} docs, "
                f"image: {self.image_collection.count()} docs"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            raise
    
    async def process_query(self, user_message: str,
                             image_bytes: bytes = None,
                             mime_type: str = None) -> Dict[str, Any]:
        """Main entry point for processing user queries.

        Args:
            user_message: Text from the user.
            image_bytes: Optional raw bytes of an uploaded image.
            mime_type: MIME type of the uploaded image (e.g. 'image/jpeg').
        """
        try:
            has_image = image_bytes is not None

            # Step 1: Intent Classification.
            # An attached image is the strongest possible signal that the
            # user wants to *search* — only `_handle_search_intent` knows
            # how to consume the image; FOLLOWUP / CLARIFICATION / GENERAL
            # all silently drop it. The LLM-driven classifier only sees
            # the text, so vague text accompanying an upload (empty, "?",
            # "what is this", "any thoughts") is routinely classified
            # GENERAL and the image is thrown away. Short-circuit here.
            if has_image:
                intent, confidence = "SEARCH", 1.0
                logger.info(
                    "Intent forced to SEARCH because an image was attached "
                    "(text='%s', %d bytes, mime=%s)",
                    (user_message or '')[:60], len(image_bytes), mime_type
                )
            else:
                intent, confidence = await self._classify_intent(user_message)
                logger.info(f"Intent classified: {intent} (confidence: {confidence})")
            
            # Step 2: Process based on intent
            if intent == "SEARCH":
                return await self._handle_search_intent(user_message, intent, image_bytes, mime_type)
            elif intent == "FOLLOWUP":
                return await self._handle_followup_intent(
                    user_message, intent,
                    image_bytes=image_bytes, mime_type=mime_type
                )
            elif intent == "CLARIFICATION":
                return await self._handle_clarification_intent(
                    user_message, intent,
                    image_bytes=image_bytes, mime_type=mime_type
                )
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
    
    async def _handle_search_intent(self, user_message: str, intent: str,
                                     image_bytes: bytes = None,
                                     mime_type: str = None) -> Dict[str, Any]:
        """Handle search queries, optionally with an uploaded cover image."""
        # Step 1: Determine search mode (TEXT / IMAGE / HYBRID)
        has_image = image_bytes is not None
        search_mode = await self._classify_search_mode(user_message, has_image)
        logger.info(f"Search mode: {search_mode}")

        # Step 2: Embed uploaded image if provided. The helper now
        # also returns the resized JPEG bytes so we can pass them as
        # context to downstream multimodal LLM calls (rerank + response
        # generator) without re-running PIL or another resize pass.
        image_embedding: Optional[List[float]] = None
        image_jpeg: Optional[bytes] = None
        if has_image:
            preprocessed = await self._preprocess_and_embed_image(image_bytes, mime_type)
            if preprocessed is None:
                logger.warning("Image embedding failed; falling back to TEXT mode")
                search_mode = "TEXT"
            else:
                image_embedding, image_jpeg = preprocessed

        # Step 3: Extract search parameters from text
        search_params = await self._extract_search_parameters(user_message)

        # Step 4: Perform vector search
        search_results = await self._perform_vector_search(
            search_params['search_terms'],
            search_params,
            image_embedding=image_embedding,
            search_mode=search_mode,
        )
        
        # Step 5: Post-retrieval filtering. When an image is in play,
        # let the rerank LLM see it so visual cues (genre, audience,
        # tone signalled by the cover) can influence ordering.
        filtered_results = await self._post_retrieval_filter(
            user_message,
            search_results,
            image_jpeg=image_jpeg,
        )
        
        # Step 6: Generate response. The response LLM also sees the
        # image when one was uploaded — this is what closes the bug
        # where the assistant said "the user has not provided a
        # specific search query" while clearly displaying an image.
        response_data = await self._generate_response(
            user_message,
            filtered_results,
            intent,
            image_jpeg=image_jpeg,
        )
        response_data['search_mode'] = search_mode  # Surface to UI
        
        # Step 7: Update conversation memory. Persist the image
        # embedding AND the JPEG bytes so a follow-up turn can refine
        # the search with new filters at zero embedding cost AND keep
        # showing the image to the LLM for context.
        self.conversation_manager.add_turn(
            user_message=user_message,
            assistant_response=response_data['full_response'],
            search_results=filtered_results,
            intent=intent,
            thinking_process=response_data['thinking'],
            image_embedding=image_embedding,
            image_mime=mime_type if image_embedding is not None else None,
            image_jpeg=image_jpeg,
            search_mode=search_mode,
        )
        
        return response_data
    
    async def _handle_followup_intent(self, user_message: str, intent: str,
                                       image_bytes: bytes = None,
                                       mime_type: str = None) -> Dict[str, Any]:
        """Handle follow-up questions using previous search context.

        Three cases, in order of preference:

        1. The user attached a *new* image with their followup. Treat it
           as a fresh search — the new image is the dominant signal.
        2. A previous turn embedded an image and the user is now adding
           text-based filters ("only English", "before 1980"). Re-run
           the vector search using the cached image embedding plus the
           freshly-extracted filters. This is the path the previous
           implementation lacked entirely; followups could only re-rank
           the cached top-5, never expand or constrain the candidate
           pool against the original image.
        3. No image context at all. Re-rank the cached previous results
           with the LLM (the original behaviour).
        """
        # Case 1: a fresh image trumps the cached context.
        if image_bytes is not None:
            return await self._handle_search_intent(
                user_message, "SEARCH",
                image_bytes=image_bytes, mime_type=mime_type
            )

        # Case 2: previous turn carried an image. Refine against it.
        prior_image = self.conversation_manager.get_last_image_context()
        if prior_image and prior_image.get('embedding'):
            search_params = await self._extract_search_parameters(user_message)
            # Decide search mode for the refinement: if the followup text
            # is substantive, blend it with the cached image; otherwise
            # stay in the prior mode (typically IMAGE).
            has_substantive_text = bool(
                (user_message or '').strip()
                and len(user_message.strip()) >= 10
            )
            refined_mode = (
                'HYBRID' if has_substantive_text else (prior_image.get('search_mode') or 'IMAGE')
            )
            logger.info(
                "Followup refining cached image search "
                f"(mode={refined_mode}, filters={ {k: v for k, v in search_params.items() if v} })"
            )

            search_results = await self._perform_vector_search(
                search_params['search_terms'],
                search_params,
                image_embedding=prior_image['embedding'],
                search_mode=refined_mode,
            )
            # Reuse the cached JPEG bytes from the originating turn so
            # the rerank and response LLMs still see the cover during
            # the followup. No re-embedding, no re-resize, no extra
            # bytes from the user.
            cached_jpeg = prior_image.get('jpeg')
            filtered_results = await self._post_retrieval_filter(
                user_message, search_results,
                image_jpeg=cached_jpeg,
            )
            response_data = await self._generate_response(
                user_message, filtered_results, intent,
                image_jpeg=cached_jpeg,
            )
            response_data['search_mode'] = refined_mode

            # Persist the image context forward so the next followup
            # can keep refining without losing context.
            self.conversation_manager.add_turn(
                user_message=user_message,
                assistant_response=response_data['full_response'],
                search_results=filtered_results,
                intent=intent,
                thinking_process=response_data['thinking'],
                image_embedding=prior_image['embedding'],
                image_mime=prior_image.get('mime'),
                image_jpeg=cached_jpeg,
                search_mode=refined_mode,
            )
            return response_data

        # Case 3: no image context anywhere. Original cached-rerank path.
        last_results = self.conversation_manager.get_last_search_context()
        if not last_results:
            # No previous search either, treat as new search
            return await self._handle_search_intent(user_message, "SEARCH")

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
    
    async def _handle_clarification_intent(self, user_message: str, intent: str,
                                            image_bytes: bytes = None,
                                            mime_type: str = None) -> Dict[str, Any]:
        """Handle clarification requests about specific books"""
        # Similar to follow-up but focused on explaining/elaborating
        return await self._handle_followup_intent(
            user_message, intent,
            image_bytes=image_bytes, mime_type=mime_type
        )
    
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


    async def _perform_vector_search(self, query_text: str, search_params: Dict,
                                      image_embedding: List[float] = None,
                                      search_mode: str = "TEXT") -> List[Dict]:
        """Perform semantic search using ChromaDB.

        Routes to the appropriate collection(s) based on search_mode:
        - TEXT (default): query text_collection with a text embedding
        - IMAGE: query image_collection with an image embedding, look up full
                 text chunks by book_id for LLM context (Option A)
        - HYBRID: query both, merge by book_id with weighted scoring
        """
        try:
            # ------------------------------------------------------------------
            # 1. Generate text query embedding (skipped for pure IMAGE mode)
            # ------------------------------------------------------------------
            text_embedding = None
            if search_mode in ("TEXT", "HYBRID") and query_text:
                text_embedding = await self._embed_text(query_text)

            # ------------------------------------------------------------------
            # 2. Build where clause (applies to text collection only)
            # ------------------------------------------------------------------
            where_clause = {}

            if search_params.get('language_filter'):
                where_clause['language'] = search_params['language_filter'].title()

            if search_params.get('year_filter'):
                year_filter = search_params['year_filter']
                if isinstance(year_filter, str):
                    if '-' in year_filter:
                        start_year, end_year = year_filter.split('-')
                        where_clause['publish_year'] = {
                            "$gte": int(start_year), "$lte": int(end_year)
                        }
                    elif 'after' in year_filter.lower() or '>' in year_filter:
                        year_match = re.search(r'\d{4}', year_filter)
                        if year_match:
                            where_clause['publish_year'] = {"$gt": int(year_match.group())}
                    elif 'before' in year_filter.lower() or '<' in year_filter:
                        year_match = re.search(r'\d{4}', year_filter)
                        if year_match:
                            where_clause['publish_year'] = {"$lt": int(year_match.group())}
                    else:
                        year_match = re.search(r'\d{4}', year_filter)
                        if year_match:
                            where_clause['publish_year'] = int(year_match.group())

            if search_params.get('page_count_filter'):
                page_filter = search_params['page_count_filter']
                if isinstance(page_filter, str):
                    if 'under' in page_filter.lower() or 'less than' in page_filter.lower() or '<' in page_filter:
                        page_match = re.search(r'\d+', page_filter)
                        if page_match:
                            where_clause['page_count'] = {"$lt": int(page_match.group())}
                    elif 'over' in page_filter.lower() or 'more than' in page_filter.lower() or '>' in page_filter:
                        page_match = re.search(r'\d+', page_filter)
                        if page_match:
                            where_clause['page_count'] = {"$gt": int(page_match.group())}
                    else:
                        page_match = re.search(r'\d+', page_filter)
                        if page_match:
                            where_clause['page_count'] = int(page_match.group())

            if search_params.get('content_filter'):
                content_filter = search_params['content_filter'].lower()
                if 'description' in content_filter or 'detailed' in content_filter:
                    where_clause['has_description'] = True
                elif 'subjects' in content_filter or 'topics' in content_filter:
                    where_clause['has_subjects'] = True

            if search_params.get('format_filter'):
                format_val = search_params['format_filter']
                if isinstance(format_val, str) and format_val.strip():
                    format_val = format_val.lower().strip()
                    if 'paperback' in format_val or 'paper' in format_val or 'soft' in format_val:
                        where_clause['format'] = 'paperback'
                    elif 'hardcover' in format_val or 'hard' in format_val or 'bound' in format_val:
                        where_clause['format'] = 'hardcover'
                    elif 'ebook' in format_val or 'digital' in format_val or 'electronic' in format_val:
                        where_clause['format'] = 'ebook'
                    elif format_val not in ['books', 'book', 'literature', 'text', 'reading']:
                        where_clause['format'] = format_val

            if where_clause:
                logger.info(f"Applying filters: {where_clause}")

            # ------------------------------------------------------------------
            # 3. Route to the appropriate search strategy
            # ------------------------------------------------------------------
            if search_mode == "IMAGE":
                if image_embedding is None:
                    logger.warning("IMAGE mode but no image_embedding; falling back to TEXT")
                    search_mode = "TEXT"
                else:
                    return await self._search_image_only(image_embedding, search_mode, where_clause)

            if search_mode == "HYBRID":
                if text_embedding is not None and image_embedding is not None:
                    return await self._search_hybrid(
                        text_embedding, image_embedding, where_clause, search_mode
                    )
                logger.warning("HYBRID mode missing an embedding; falling back to TEXT")

            # Default: TEXT
            return await self._search_text_only(text_embedding, where_clause)

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Search mode classification
    # ------------------------------------------------------------------

    async def _classify_search_mode(self, user_message: str, has_image: bool) -> str:
        """Determine whether to use TEXT, IMAGE, or HYBRID search.

        Rules (fast-path before LLM call):
        - No image → always TEXT
        - Image + empty/very short text → IMAGE
        - Image + generic similarity phrase → IMAGE
        - Image + substantive query → ask LLM (HYBRID or TEXT)
        """
        if not has_image:
            return "TEXT"

        if not user_message or len(user_message.strip()) < 10:
            return "IMAGE"

        generic_phrases = [
            'find books like this', 'similar books', 'books like this',
            'find similar', 'what is this', 'what book is this',
            'identify this', 'books like it', 'look for this',
        ]
        msg_lower = user_message.lower().strip()
        if any(phrase in msg_lower for phrase in generic_phrases):
            return "IMAGE"

        prompt = self.config.SYSTEM_PROMPTS['search_mode_classifier'].format(
            user_message=user_message
        )
        try:
            response = await self.genai_client.aio.models.generate_content(
                model=self.config.LLM_MODEL,
                contents=prompt
            )
            mode = response.text.strip().upper().split()[0]
            if mode in ("TEXT", "IMAGE", "HYBRID"):
                return mode
        except Exception as e:
            logger.warning(f"Search mode classification failed: {e}")

        return "HYBRID"  # Safe default when image + substantial text

    # ------------------------------------------------------------------
    # Embedding dispatch (text & image) — backend-aware
    # ------------------------------------------------------------------

    async def _embed_text(self, text: str) -> List[float]:
        """Generate a text embedding using the active backend.

        - gemini_api: calls Google GenAI async API
        - qwen_local: runs sentence-transformers model.encode() in a thread
        - jina_local: runs AutoModel.encode_text() in a thread
        """
        if self.config.QUERY_EMBED_TYPE == 'qwen_local':
            # Qwen's encode() is synchronous — offload to a thread
            def _encode():
                emb = self.qwen_model.encode(
                    [text],
                    batch_size=1,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                return emb[0].tolist()
            return await asyncio.to_thread(_encode)
        elif self.config.QUERY_EMBED_TYPE == 'jina_local':
            # Text: AutoModel.encode_text() — no prompt, matches stored embeddings.
            import numpy as _np
            import torch as _t
            def _encode_jina_text():
                with _t.no_grad():
                    emb = self.jina_model.encode_text([text])
                if hasattr(emb, "cpu"):
                    emb = emb.cpu()
                arr = _np.array(emb).flatten().astype(_np.float32)
                norm = _np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                return arr.tolist()
            return await asyncio.to_thread(_encode_jina_text)
        else:
            # Default: Gemini API
            response = await self.genai_client.aio.models.embed_content(
                model=self.config.EMBEDDING_MODEL,
                contents=text,
            )
            return list(response.embeddings[0].values)

    # ------------------------------------------------------------------
    # Image preprocessing and embedding
    # ------------------------------------------------------------------

    async def _preprocess_and_embed_image(self, image_bytes: bytes,
                                           mime_type: str = "image/jpeg"
                                           ) -> Optional[Tuple[List[float], bytes]]:
        """Resize an uploaded image and generate its embedding.

        Preprocessing is deliberately identical to the image-embedding
        ingest pipeline so query and ingest land in the same input
        distribution:
        - Convert to RGB
        - Thumbnail to MAX_UPLOAD_IMAGE_DIMENSION px (preserves aspect ratio)
        - Re-encode as JPEG quality 85 (minimises token cost)

        Embedding is then routed to the active backend:
        - gemini_api: sends a typed Part to the Google GenAI API
        - qwen_local: passes the PIL Image directly to model.encode()

        Returns:
            (embedding, jpeg_bytes) so callers can reuse the resized
            JPEG bytes for downstream multimodal LLM calls without
            re-doing the resize/re-encode work. Returns None on any
            failure (PIL error, embedding API error, etc.).
        """
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            max_dim = self.config.MAX_UPLOAD_IMAGE_DIMENSION
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=85)
            jpeg_bytes = buf.getvalue()
            logger.info(f"Image preprocessed: {img.size}, {len(jpeg_bytes) / 1024:.1f} KB")

            # --- Embed with active backend ---
            if self.config.QUERY_EMBED_TYPE == 'qwen_local':
                # Qwen accepts PIL Image objects directly
                pil_for_embed = img  # already RGB + resized

                def _encode_img():
                    emb = self.qwen_model.encode(
                        [pil_for_embed],
                        batch_size=1,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                    return emb[0].tolist()

                embedding = await asyncio.to_thread(_encode_img)
            elif self.config.QUERY_EMBED_TYPE == 'jina_local':
                # Image: AutoModel.encode_image() — no prompt, same as stored embeddings.
                import numpy as _np
                import torch as _t
                pil_for_embed = img

                def _encode_img_jina():
                    with _t.no_grad():
                        emb = self.jina_model.encode_image([pil_for_embed])
                    if hasattr(emb, "cpu"):
                        emb = emb.cpu()
                    arr = _np.array(emb).flatten().astype(_np.float32)
                    norm = _np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    return arr.tolist()

                embedding = await asyncio.to_thread(_encode_img_jina)
            else:
                # Default: Gemini API
                from google.genai import types
                part = types.Part.from_bytes(data=jpeg_bytes, mime_type='image/jpeg')
                result = await self.genai_client.aio.models.embed_content(
                    model=self.config.EMBEDDING_MODEL,
                    contents=part
                )
                embedding = list(result.embeddings[0].values)

            logger.info(f"Image embedded: {len(embedding)} dims")
            return embedding, jpeg_bytes

        except Exception as e:
            logger.error(f"Image preprocessing/embedding failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Search helpers (called by _perform_vector_search)
    # ------------------------------------------------------------------

    def _fetch_text_chunks_by_book_ids(self, book_ids: List[str]) -> Dict[str, Dict]:
        """Look up text chunks for a list of book_ids.

        Joins on the canonical `book_id` metadata field via a `where`
        clause instead of the prior `text_{book_id}` id pattern. The id
        pattern silently misses whenever the text collection holds more
        than one chunk per book (chunk-suffixed ids) or whenever chunk
        ids collided at ingest, leaving the LLM with a `[Cover Image]`
        placeholder instead of real text context.

        Returns:
            {book_id: {'document': str, 'metadata': dict}}.
            If a book has multiple text chunks, the first one returned
            wins — fine for surfacing book-level context to the LLM.
        """
        out: Dict[str, Dict] = {}
        unique_ids = list(dict.fromkeys(bid for bid in book_ids if bid))
        if not unique_ids:
            return out

        # ChromaDB accepts `$in` with any number of values. Use the
        # equality shorthand for a single id for slightly cheaper paths
        # in older Chroma builds.
        where = (
            {"book_id": unique_ids[0]}
            if len(unique_ids) == 1
            else {"book_id": {"$in": unique_ids}}
        )

        try:
            lookup = self.text_collection.get(
                where=where,
                include=['documents', 'metadatas']
            )
        except Exception as e:
            logger.warning(f"Text lookup by book_id failed: {e}")
            return out

        docs = lookup.get('documents') or []
        metas = lookup.get('metadatas') or []
        for doc, meta in zip(docs, metas):
            if not isinstance(meta, dict):
                continue
            bid = meta.get('book_id', '')
            if not bid or bid in out:
                # First chunk per book wins
                continue
            out[bid] = {'document': doc, 'metadata': meta}

        return out

    def _fetch_image_paths_by_book_ids(self, book_ids: List[str]) -> Dict[str, str]:
        """Look up cover image paths for a list of book_ids.

        Joins on the canonical `book_id` metadata field in
        `library_books_image`. Used to attach `cover_image_path` to
        text-mode results so the UI can render covers — the text
        collection's metadata does not carry `cover_image_path`, only a
        `has_cover_image` boolean.

        Returns:
            {book_id: image_path} for books that have a non-empty cover
            path on disk. Books without an image record, or whose record
            is missing `image_path`, are simply absent from the map.
        """
        out: Dict[str, str] = {}
        unique_ids = list(dict.fromkeys(bid for bid in book_ids if bid))
        if not unique_ids:
            return out

        where = (
            {"book_id": unique_ids[0]}
            if len(unique_ids) == 1
            else {"book_id": {"$in": unique_ids}}
        )

        try:
            lookup = self.image_collection.get(
                where=where,
                include=['metadatas']
            )
        except Exception as e:
            logger.warning(f"Image-path lookup by book_id failed: {e}")
            return out

        metas = lookup.get('metadatas') or []
        for meta in metas:
            if not isinstance(meta, dict):
                continue
            bid = meta.get('book_id', '')
            path = meta.get('image_path', '')
            if not bid or not path or bid in out:
                continue
            out[bid] = path

        return out

    def _attach_cover_image_paths(self, results: List[Dict]) -> None:
        """Mutate `results` in place, adding `cover_image_path` to each
        result's `metadata` when the book has an entry in
        `library_books_image`.

        The Chainlit UI in `chainlit_app.py::show_search_details` reads
        `metadata.get('cover_image_path', '')` when `has_cover_image` is
        truthy. Without this enrichment the field is never populated for
        text-mode results and covers are silently skipped.
        """
        if not results:
            return

        # Fast path: image-mode and image-only hybrid results already
        # carry `image_path` from `library_books_image` — promote it to
        # `cover_image_path` without an extra DB lookup.
        missing: List[Dict] = []
        for r in results:
            meta = r.get('metadata')
            if not isinstance(meta, dict):
                continue
            if meta.get('cover_image_path'):
                continue
            local_path = meta.get('image_path')
            if local_path:
                new_meta = dict(meta)
                new_meta['cover_image_path'] = local_path
                new_meta.setdefault('has_cover_image', True)
                r['metadata'] = new_meta
                continue
            missing.append(r)

        if not missing:
            return

        # Slow path: text-mode results need a join into
        # `library_books_image` to find the on-disk cover path.
        book_ids = [
            r['metadata'].get('book_id', '') for r in missing
        ]
        path_by_book = self._fetch_image_paths_by_book_ids(book_ids)
        if not path_by_book:
            return

        for r in missing:
            meta = r['metadata']
            bid = meta.get('book_id', '')
            path = path_by_book.get(bid)
            if path:
                # Mutate a shallow copy so we don't disturb the
                # underlying ChromaDB-returned dict (some Chroma builds
                # share metadata references across calls).
                new_meta = dict(meta)
                new_meta['cover_image_path'] = path
                # Helpful for the UI's `has_cover_image` check too.
                new_meta.setdefault('has_cover_image', True)
                r['metadata'] = new_meta

    async def _search_text_only(self, text_embedding: List[float],
                                 where_clause: Dict) -> List[Dict]:
        """Query the text collection and return formatted results."""
        try:
            results = self.text_collection.query(
                query_embeddings=[text_embedding],
                n_results=self.config.DEFAULT_SEARCH_LIMIT,
                where=where_clause if where_clause else None,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as db_error:
            logger.error(f"Text collection query failed: {db_error}")
            logger.info("Retrying without filters...")
            results = self.text_collection.query(
                query_embeddings=[text_embedding],
                n_results=self.config.DEFAULT_SEARCH_LIMIT,
                include=['documents', 'metadatas', 'distances']
            )

        formatted = []
        for i, doc in enumerate(results['documents'][0]):
            text_sim = 1 - results['distances'][0][i]
            formatted.append({
                'document': doc,
                'metadata': results['metadatas'][0][i],
                'similarity_score': text_sim,
                'text_similarity': text_sim,
                'image_similarity': None,
                'chunk_id': results['ids'][0][i] if 'ids' in results else f"result_{i}",
                'applied_filters': where_clause,
                'search_mode': 'TEXT',
            })

        logger.info(f"Text search returned {len(formatted)} results")
        # Attach cover_image_path so the UI can render covers — text
        # collection metadata only stores `has_cover_image`, not the path.
        self._attach_cover_image_paths(formatted)
        return formatted

    async def _search_image_only(self, image_embedding: List[float],
                                  search_mode: str,
                                  where_clause: Optional[Dict] = None) -> List[Dict]:
        """Query the image collection, then look up full text chunks by book_id (Option A).

        `where_clause` is an optional ChromaDB metadata filter. The image
        collection only carries the canonical `book_id`, `chunk_type`,
        `image_path`, `title`, `authors` fields by default — additional
        filterable fields (language, publish_year, page_count, format,
        ...) are populated by `vector_db/backfill_image_metadata.py`.
        Until that backfill is run, applying the filter would erase
        every result, so we transparently retry without the filter on
        failure or empty match.
        """
        applied_where = where_clause if where_clause else None
        try:
            img_results = self.image_collection.query(
                query_embeddings=[image_embedding],
                n_results=self.config.DEFAULT_SEARCH_LIMIT,
                where=applied_where,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            logger.warning(
                f"Image collection query with filters failed ({e}); "
                f"retrying without filters"
            )
            applied_where = None
            try:
                img_results = self.image_collection.query(
                    query_embeddings=[image_embedding],
                    n_results=self.config.DEFAULT_SEARCH_LIMIT,
                    include=['documents', 'metadatas', 'distances']
                )
            except Exception as e2:
                logger.error(f"Image collection query failed: {e2}")
                return []

        # If a filter was applied but matched nothing, retry unfiltered.
        # The image collection's metadata schema is sparser than the
        # text collection's, so a strict filter that the text branch
        # would still satisfy can wipe out all image hits.
        if applied_where is not None and (
            not img_results.get('documents')
            or not img_results['documents'][0]
        ):
            logger.info(
                "Image filter matched zero hits; retrying without filter"
            )
            applied_where = None
            try:
                img_results = self.image_collection.query(
                    query_embeddings=[image_embedding],
                    n_results=self.config.DEFAULT_SEARCH_LIMIT,
                    include=['documents', 'metadatas', 'distances']
                )
            except Exception as e:
                logger.error(f"Image collection query failed: {e}")
                return []

        if not img_results['documents'] or not img_results['documents'][0]:
            return []

        # Look up full text chunks from text_collection by book_id metadata.
        # Joining via `where={"book_id": ...}` is robust to the actual id
        # scheme in `library_books_text` (which may suffix chunk indices
        # or hold multiple chunks per book), unlike the prior
        # `text_{book_id}` id pattern which silently missed.
        book_ids = [meta.get('book_id', '') for meta in img_results['metadatas'][0]]
        text_by_book = self._fetch_text_chunks_by_book_ids(book_ids)

        formatted = []
        for img_meta, img_dist in zip(
            img_results['metadatas'][0], img_results['distances'][0]
        ):
            book_id = img_meta.get('book_id', '')
            img_sim = 1 - img_dist
            text_data = text_by_book.get(book_id, {})
            formatted.append({
                'document': text_data.get(
                    'document', f"[Cover Image] {img_meta.get('title', book_id)}"
                ),
                'metadata': text_data.get('metadata', img_meta),
                'similarity_score': img_sim,
                'text_similarity': None,
                'image_similarity': img_sim,
                'chunk_id': f"image_{book_id}",
                'applied_filters': applied_where or {},
                'search_mode': search_mode,
            })

        logger.info(f"Image search returned {len(formatted)} results")
        # Image-mode results come from `library_books_image`, whose
        # metadata stores `image_path` rather than the `cover_image_path`
        # key the UI looks for. Attach it so covers render.
        self._attach_cover_image_paths(formatted)
        return formatted

    async def _search_hybrid(self, text_embedding: List[float],
                              image_embedding: List[float],
                              where_clause: Dict,
                              search_mode: str) -> List[Dict]:
        """Query both collections and merge results by book_id.

        Scoring strategy (safe — no penalty for books without image embeddings):
        - In both:  combined = TEXT_WEIGHT * text_sim + IMAGE_WEIGHT * img_sim
        - Text only: combined = text_sim  (full score, no penalty)
        - Image only: combined = IMAGE_WEIGHT * img_sim
        """
        tw = self.config.TEXT_SEARCH_WEIGHT
        iw = self.config.IMAGE_SEARCH_WEIGHT

        # --- Text query ---
        try:
            txt_r = self.text_collection.query(
                query_embeddings=[text_embedding],
                n_results=self.config.DEFAULT_SEARCH_LIMIT,
                where=where_clause if where_clause else None,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception:
            txt_r = self.text_collection.query(
                query_embeddings=[text_embedding],
                n_results=self.config.DEFAULT_SEARCH_LIMIT,
                include=['documents', 'metadatas', 'distances']
            )

        # --- Image query ---
        # Apply the same metadata filter as the text branch so HYBRID
        # mode actually constrains both halves of the merge. Until the
        # image-metadata backfill has been run, the image collection
        # may not carry every filterable field — fall back gracefully
        # by retrying without filters on failure or empty match.
        applied_where_image = where_clause if where_clause else None
        try:
            img_r = self.image_collection.query(
                query_embeddings=[image_embedding],
                n_results=self.config.DEFAULT_SEARCH_LIMIT,
                where=applied_where_image,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            logger.warning(
                f"Hybrid image query with filters failed ({e}); "
                f"retrying without filters"
            )
            applied_where_image = None
            try:
                img_r = self.image_collection.query(
                    query_embeddings=[image_embedding],
                    n_results=self.config.DEFAULT_SEARCH_LIMIT,
                    include=['documents', 'metadatas', 'distances']
                )
            except Exception as e2:
                logger.warning(f"Hybrid image query failed: {e2}")
                img_r = {'documents': [[]], 'metadatas': [[]],
                         'distances': [[]], 'ids': [[]]}

        # Retry unfiltered if a filter erased every image hit.
        if applied_where_image is not None and (
            not img_r.get('documents') or not img_r['documents'][0]
        ):
            logger.info(
                "Hybrid image filter matched zero hits; retrying without filter"
            )
            applied_where_image = None
            try:
                img_r = self.image_collection.query(
                    query_embeddings=[image_embedding],
                    n_results=self.config.DEFAULT_SEARCH_LIMIT,
                    include=['documents', 'metadatas', 'distances']
                )
            except Exception as e:
                logger.warning(f"Hybrid image query failed: {e}")
                img_r = {'documents': [[]], 'metadatas': [[]],
                         'distances': [[]], 'ids': [[]]}

        # --- Merge by book_id ---
        merged: Dict[str, Dict] = {}

        if txt_r['documents'] and txt_r['documents'][0]:
            for doc, meta, dist in zip(
                txt_r['documents'][0], txt_r['metadatas'][0], txt_r['distances'][0]
            ):
                bid = meta.get('book_id', '')
                ts = 1 - dist
                merged[bid] = {
                    'document': doc, 'metadata': meta,
                    'text_similarity': ts, 'image_similarity': None,
                    'combined_score': ts,  # no penalty if no image match
                }

        image_only_ids: List[str] = []
        if img_r['documents'] and img_r['documents'][0]:
            for img_meta, img_dist in zip(img_r['metadatas'][0], img_r['distances'][0]):
                bid = img_meta.get('book_id', '')
                img_sim = 1 - img_dist
                if bid in merged:
                    ts = merged[bid]['text_similarity']
                    merged[bid]['image_similarity'] = img_sim
                    merged[bid]['combined_score'] = tw * ts + iw * img_sim
                else:
                    image_only_ids.append(bid)
                    merged[bid] = {
                        'document': None, 'metadata': img_meta,
                        'text_similarity': None, 'image_similarity': img_sim,
                        'combined_score': iw * img_sim,
                    }

        # Look up text chunks for image-only results. Joins via
        # `where={"book_id": ...}` (canonical key) instead of the brittle
        # `text_{book_id}` id pattern that silently misses for any book
        # with chunk-suffixed ids or multiple text chunks.
        if image_only_ids:
            text_by_book = self._fetch_text_chunks_by_book_ids(image_only_ids)
            for bid, text_data in text_by_book.items():
                if bid in merged:
                    merged[bid]['document'] = text_data['document']
                    merged[bid]['metadata'] = text_data['metadata']

        # --- Sort and format ---
        sorted_items = sorted(
            merged.values(), key=lambda x: x['combined_score'], reverse=True
        )

        formatted = []
        for item in sorted_items[:self.config.DEFAULT_SEARCH_LIMIT]:
            bid = item['metadata'].get('book_id', '')
            formatted.append({
                'document': item['document'] or f"[Cover] {item['metadata'].get('title', bid)}",
                'metadata': item['metadata'],
                'similarity_score': item['combined_score'],
                'text_similarity': item['text_similarity'],
                'image_similarity': item['image_similarity'],
                'chunk_id': f"hybrid_{bid}",
                'applied_filters': where_clause,
                'search_mode': search_mode,
            })

        logger.info(f"Hybrid search returned {len(formatted)} merged results")
        # Hybrid results may be text-only, image-only, or both. Image-only
        # entries already carry `image_path`; text-only entries do not.
        # Normalise so the UI can render covers regardless of branch.
        self._attach_cover_image_paths(formatted)
        return formatted
    
    # ------------------------------------------------------------------
    # Multimodal LLM helper
    # ------------------------------------------------------------------

    async def _llm_generate(self, prompt: str,
                             image_jpeg: Optional[bytes] = None) -> str:
        """Run a `generate_content` call with optional image context.

        When `image_jpeg` is provided the call is multimodal — the
        image is sent as a typed `Part` alongside the text prompt so
        the LLM can ground its reasoning in the cover. On any error
        (corrupt bytes, transient API hiccup), we retry text-only so
        the conversation never silently fails because the image
        couldn't be sent. The caller still gets a usable response.

        Returns the raw `.text` of the LLM response (stripped).
        """
        contents: Any
        if image_jpeg:
            try:
                from google.genai import types
                image_part = types.Part.from_bytes(
                    data=image_jpeg, mime_type='image/jpeg'
                )
                # Image first, then prompt — typical multimodal layout
                # for Gemini and lets the LLM "look" before "thinking".
                contents = [image_part, prompt]
                response = await self.genai_client.aio.models.generate_content(
                    model=self.config.LLM_MODEL,
                    contents=contents,
                )
                return response.text.strip()
            except Exception as e:
                logger.warning(
                    f"Multimodal LLM call failed ({e}); retrying text-only"
                )

        # Text-only path (default, and the fallback when multimodal fails)
        response = await self.genai_client.aio.models.generate_content(
            model=self.config.LLM_MODEL,
            contents=prompt,
        )
        return response.text.strip()

    async def _post_retrieval_filter(self, user_query: str,
                                      search_results: List[Dict],
                                      image_jpeg: Optional[bytes] = None
                                      ) -> List[Dict]:
        """Use LLM to filter and rank search results.

        When `image_jpeg` is supplied the rerank LLM also sees the
        uploaded cover, so visual cues (subject portrait vs. abstract
        art, audience signalling, tone) can influence the ordering.
        """
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
            filter_response = await self._llm_generate(prompt, image_jpeg=image_jpeg)
            
            # Parse LLM response to get ranked book numbers.
            #
            # The previous implementation used `re.findall(r'\b(\d+)\b', line)`
            # which matches ANY integer on the line — page counts, years,
            # ISBNs, similarity scores like "0.872", references like
            # "1990s", etc. — and treats whichever number appears first as
            # a book index. That made the rerank essentially noise whenever
            # the parser didn't fall through to similarity ordering.
            #
            # The pattern below is anchored to start-of-line and only
            # matches a leading numbered-list marker like "1.", "2)",
            # "3:", "4]" with optional surrounding whitespace. Each line
            # contributes at most one book index, so explanations no
            # longer get clobbered by trailing digits in the same line.
            ranked_results = []
            list_marker_re = re.compile(r'^\s*(\d+)\s*[.):\]]')

            lines = filter_response.split('\n')
            book_numbers: List[int] = []

            for line in lines:
                m = list_marker_re.match(line)
                if not m:
                    continue
                book_num = int(m.group(1))
                if not (1 <= book_num <= len(search_results)):
                    continue
                if book_num in book_numbers:
                    continue

                book_numbers.append(book_num)

                # Capture explanation only on the first line that mentions
                # this book — later mentions (e.g. cross-references)
                # would otherwise overwrite the primary justification.
                explanation = line.strip()
                if explanation:
                    search_results[book_num - 1].setdefault(
                        'llm_explanation', explanation
                    )
            
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
            
            # Add LLM filtering metadata. Stamp wall-clock time directly
            # rather than reaching into `logger.handlers[0].format(...)`
            # to extract a date string — that path crashes when the root
            # handler has no formatter, depends on the format string
            # containing ' - ', and pollutes log output via a synthetic
            # LogRecord just to get a timestamp.
            stamp = datetime.now().isoformat()
            for result in ranked_results:
                result['filtered_by_llm'] = True
                result['filter_timestamp'] = stamp
            
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
    
    async def _generate_response(self, user_query: str,
                                  relevant_books: List[Dict],
                                  intent: str,
                                  image_jpeg: Optional[bytes] = None
                                  ) -> Dict[str, Any]:
        """Generate final response using LLM.

        When `image_jpeg` is supplied the response LLM also sees the
        uploaded cover. This is what lets the assistant acknowledge
        and reason about the image — without it, the LLM only saw
        the user's text (often empty for image-only queries) and the
        retrieved books, leading to thinking like "the user has not
        provided a specific search query" while the user clearly had.
        """
        # Prepare book information
        books_text = ""
        for book in relevant_books:
            books_text += f"- {book['document']}\n"
        
        # Get conversation context
        conversation_context = self.conversation_manager.format_for_llm_context()
        
        # When an image is in play, surface that fact in the prompt so
        # the LLM is reliably aware of it even on the text-only retry
        # path (where the image bytes themselves don't reach the API).
        prompt_user_query = user_query
        if image_jpeg:
            note = (
                "[The user attached a book cover image with this message. "
                "The candidate books below were retrieved by visual and/or "
                "text similarity. Acknowledge and use the cover when "
                "relevant.]"
            )
            prompt_user_query = (
                f"{note}\n\nUser text: {user_query!s}"
                if user_query else f"{note}\n\nUser text: (empty)"
            )

        prompt = self.config.SYSTEM_PROMPTS['response_generator'].format(
            user_query=prompt_user_query,
            relevant_books=books_text,
            conversation_context=conversation_context
        )
        
        try:
            full_response = await self._llm_generate(prompt, image_jpeg=image_jpeg)
            
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
            'text_collection': self.config.TEXT_COLLECTION_NAME,
            'image_collection': self.config.IMAGE_COLLECTION_NAME,
            'text_doc_count': self.text_collection.count(),
            'image_doc_count': self.image_collection.count(),
            'models_used': {
                'embedding': self.config.EMBEDDING_MODEL,
                'llm': self.config.LLM_MODEL
            }
        } 