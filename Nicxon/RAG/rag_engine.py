import asyncio
import os
import io
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import chromadb
from google import genai
from config import RAGConfig
from conversation_manager import ConversationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root = parent of this file's directory (RAG/ -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_image_path(raw: str) -> str:
    """Turn a stored image path into an absolute path usable by the OS.

    Stored paths are relative to the project root (e.g.
    ``Cleaned_data/cover_images/foo.png``) so they work on any machine
    or Docker container regardless of mount point.  Absolute legacy paths
    are returned unchanged.
    """
    if not raw:
        return raw
    p = Path(raw)
    if p.is_absolute():
        return raw
    return str(_PROJECT_ROOT / p)

class LibraryRAGEngine:
    
    def __init__(self):
        self.config = RAGConfig()
        self.conversation_manager = ConversationManager()
        self._initialize_components()
    
    def _initialize_components(self):
        try:
            if not self.config.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            self.genai_client = genai.Client(api_key=self.config.GOOGLE_API_KEY)
            
            self.qwen_model = None
            self.jina_model = None
            
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
                try:
                    from sentence_transformers import SentenceTransformer as _ST
                except ImportError:
                    raise ImportError(
                        "sentence-transformers is required for the Jina v5 backend. "
                        "conda install -c conda-forge sentence-transformers"
                    )

                logger.info(
                    f"Loading Jina model: {self.config.EMBEDDING_MODEL} ..."
                )
                self.jina_model = _ST(
                    self.config.EMBEDDING_MODEL,
                    trust_remote_code=True,
                    model_kwargs={"modality": "vision", "default_task": "retrieval"},
                )
                logger.info(
                    f"Jina model loaded — dim="
                    f"{self.jina_model.get_sentence_embedding_dimension()}"
                )
            
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
        try:
            has_image = image_bytes is not None

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
            else:
                return await self._handle_general_intent(user_message, intent)
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._create_error_response(str(e))
    
    async def _classify_intent(self, user_message: str) -> Tuple[str, float]:
        prompt = self.config.SYSTEM_PROMPTS['intent_classifier'].format(
            user_message=user_message
        )
        
        try:
            response = await self.genai_client.aio.models.generate_content(
                model=self.config.LLM_MODEL,
                contents=prompt
            )
            result = response.text.strip()
            
            if '|' in result:
                intent, confidence_str = result.split('|', 1)
                confidence = float(confidence_str)
            else:
                intent = result
                confidence = 0.8
            
            return intent.strip(), confidence
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return "SEARCH", 0.5
    
    async def _handle_search_intent(self, user_message: str, intent: str,
                                     image_bytes: bytes = None,
                                     mime_type: str = None) -> Dict[str, Any]:
        has_image = image_bytes is not None
        search_mode = await self._classify_search_mode(user_message, has_image)
        logger.info(f"Search mode: {search_mode}")

        image_embedding: Optional[List[float]] = None
        image_jpeg: Optional[bytes] = None
        if has_image:
            preprocessed = await self._preprocess_and_embed_image(image_bytes, mime_type)
            if preprocessed is None:
                logger.warning("Image embedding failed; falling back to TEXT mode")
                search_mode = "TEXT"
            else:
                image_embedding, image_jpeg = preprocessed

        search_params = await self._extract_search_parameters(user_message)

        search_results = await self._perform_vector_search(
            search_params['search_terms'],
            search_params,
            image_embedding=image_embedding,
            search_mode=search_mode,
        )
        
        filtered_results = await self._post_retrieval_filter(
            user_message,
            search_results,
            image_jpeg=image_jpeg,
        )
        
        response_data = await self._generate_response(
            user_message,
            filtered_results,
            intent,
            image_jpeg=image_jpeg,
        )
        response_data['search_mode'] = search_mode
        
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
        if image_bytes is not None:
            return await self._handle_search_intent(
                user_message, "SEARCH",
                image_bytes=image_bytes, mime_type=mime_type
            )

        prior_image = self.conversation_manager.get_last_image_context()
        if prior_image and prior_image.get('embedding'):
            search_params = await self._extract_search_parameters(user_message)
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

        last_results = self.conversation_manager.get_last_search_context()
        if not last_results:
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
        return await self._handle_followup_intent(
            user_message, intent,
            image_bytes=image_bytes, mime_type=mime_type
        )
    
    async def _handle_general_intent(self, user_message: str, intent: str) -> Dict[str, Any]:
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
        prompt = self.config.SYSTEM_PROMPTS['query_processor'].format(
            user_message=user_message
        )
        
        try:
            response = await self.genai_client.aio.models.generate_content(
                model=self.config.LLM_MODEL,
                contents=prompt
            )
            result = response.text.strip()
            
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
                        if value and value.lower() not in ['books', 'book', 'literature', 'text']:
                            params['format_filter'] = value
                        else:
                            params['format_filter'] = None
                    elif 'page_count' in key or 'page count' in key:
                        params['page_count_filter'] = value
                    elif 'content' in key:
                        params['content_filter'] = value
            
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
        try:
            text_embedding = None
            if search_mode in ("TEXT", "HYBRID") and query_text:
                text_embedding = await self._embed_text(query_text)

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

            return await self._search_text_only(text_embedding, where_clause)

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _classify_search_mode(self, user_message: str, has_image: bool) -> str:
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

        return "HYBRID"

    async def _embed_text(self, text: str) -> List[float]:
        if self.config.QUERY_EMBED_TYPE == 'qwen_local':
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
            def _encode_jina_text():
                emb = self.jina_model.encode(
                    [text],
                    task="retrieval",
                    batch_size=1,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                return emb[0].tolist()
            return await asyncio.to_thread(_encode_jina_text)
        else:
            response = await self.genai_client.aio.models.embed_content(
                model=self.config.EMBEDDING_MODEL,
                contents=text,
            )
            return list(response.embeddings[0].values)

    async def _preprocess_and_embed_image(self, image_bytes: bytes,
                                           mime_type: str = "image/jpeg"
                                           ) -> Optional[Tuple[List[float], bytes]]:
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

            if self.config.QUERY_EMBED_TYPE == 'qwen_local':
                pil_for_embed = img

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
                pil_for_embed = img

                def _encode_img_jina():
                    emb = self.jina_model.encode(
                        [pil_for_embed],
                        task="retrieval",
                        batch_size=1,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                    return emb[0].tolist()

                embedding = await asyncio.to_thread(_encode_img_jina)
            else:
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

    def _fetch_text_chunks_by_book_ids(self, book_ids: List[str]) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        unique_ids = list(dict.fromkeys(bid for bid in book_ids if bid))
        if not unique_ids:
            return out

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
                continue
            out[bid] = {'document': doc, 'metadata': meta}

        return out

    def _fetch_image_paths_by_book_ids(self, book_ids: List[str]) -> Dict[str, str]:
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
        if not results:
            return

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
                new_meta['cover_image_path'] = _resolve_image_path(local_path)
                new_meta.setdefault('has_cover_image', True)
                r['metadata'] = new_meta
                continue
            missing.append(r)

        if not missing:
            return

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
                new_meta = dict(meta)
                new_meta['cover_image_path'] = _resolve_image_path(path)
                new_meta.setdefault('has_cover_image', True)
                r['metadata'] = new_meta

    async def _search_text_only(self, text_embedding: List[float],
                                 where_clause: Dict) -> List[Dict]:
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
        self._attach_cover_image_paths(formatted)
        return formatted

    async def _search_image_only(self, image_embedding: List[float],
                                  search_mode: str,
                                  where_clause: Optional[Dict] = None) -> List[Dict]:
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
        self._attach_cover_image_paths(formatted)
        return formatted

    async def _search_hybrid(self, text_embedding: List[float],
                              image_embedding: List[float],
                              where_clause: Dict,
                              search_mode: str) -> List[Dict]:
        tw = self.config.TEXT_SEARCH_WEIGHT
        iw = self.config.IMAGE_SEARCH_WEIGHT

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
                    'combined_score': ts,
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

        if image_only_ids:
            text_by_book = self._fetch_text_chunks_by_book_ids(image_only_ids)
            for bid, text_data in text_by_book.items():
                if bid in merged:
                    merged[bid]['document'] = text_data['document']
                    merged[bid]['metadata'] = text_data['metadata']

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
        self._attach_cover_image_paths(formatted)
        return formatted
    
    async def _llm_generate(self, prompt: str,
                             image_jpeg: Optional[bytes] = None) -> str:
        contents: Any
        if image_jpeg:
            try:
                from google.genai import types
                image_part = types.Part.from_bytes(
                    data=image_jpeg, mime_type='image/jpeg'
                )
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

        response = await self.genai_client.aio.models.generate_content(
            model=self.config.LLM_MODEL,
            contents=prompt,
        )
        return response.text.strip()

    async def _post_retrieval_filter(self, user_query: str,
                                      search_results: List[Dict],
                                      image_jpeg: Optional[bytes] = None
                                      ) -> List[Dict]:
        if not search_results:
            return []
        
        results_text = ""
        for i, result in enumerate(search_results[:10]):
            doc = result['document']
            metadata = result.get('metadata', {})
            similarity = result.get('similarity_score', 0)
            
            title_line = doc.split('\n')[0] if doc else "Unknown Title"
            
            results_text += f"\n{i+1}. TITLE: {title_line}\n"
            results_text += f"   SIMILARITY: {similarity:.3f}\n"
            results_text += f"   CONTENT: {doc[:400]}...\n"
            
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

                explanation = line.strip()
                if explanation:
                    search_results[book_num - 1].setdefault(
                        'llm_explanation', explanation
                    )
            
            for book_num in book_numbers[:5]:
                if book_num <= len(search_results):
                    ranked_results.append(search_results[book_num - 1])
            
            if len(ranked_results) < 3:
                logger.warning("LLM filtering parsing incomplete, falling back to similarity ranking")
                ranked_results = sorted(search_results[:5], 
                                      key=lambda x: x.get('similarity_score', 0), 
                                      reverse=True)
            
            stamp = datetime.now().isoformat()
            for result in ranked_results:
                result['filtered_by_llm'] = True
                result['filter_timestamp'] = stamp
            
            logger.info(f"Post-retrieval filtering: {len(ranked_results)} books selected from {len(search_results)}")
            return ranked_results[:5]
            
        except Exception as e:
            logger.error(f"Post-retrieval filtering failed: {e}")
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
        books_text = ""
        for book in relevant_books:
            books_text += f"- {book['document']}\n"
        
        conversation_context = self.conversation_manager.format_for_llm_context()
        
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

            return {
                'thinking': '',
                'response': full_response,
                'full_response': full_response,
                'search_results': relevant_books,
                'intent': intent
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._create_error_response(str(e))
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        response = "I apologize, but I encountered an issue while searching for books. Please try rephrasing your question or contact the librarian for assistance."
        
        return {
            'thinking': '',
            'response': response,
            'full_response': response,
            'search_results': [],
            'intent': 'ERROR'
        }
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        return self.conversation_manager.get_conversation_stats()
    
    def clear_conversation(self):
        self.conversation_manager.clear_conversation()
    
    def get_system_status(self) -> Dict[str, Any]:
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