from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset import EvalQuery
from retrieval_metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


EMBEDDING_MODEL_DEFAULT = "gemini-embedding-2"
QWEN_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
JINA_MODEL_NAME = "jinaai/jina-embeddings-v5-omni-small"
MAX_IMAGE_DIM = 512
JPEG_QUALITY = 85
TOP_K = 10

_GEMINI_PREFIXES = ("gemini-", "text-embedding-", "embedding-", "models/")


def _is_gemini(model_id: str) -> bool:
    return any(model_id.startswith(p) for p in _GEMINI_PREFIXES)


@dataclass
class QueryResult:
    query_id: str
    search_mode: str
    retrieved_text_book_ids: List[str] = field(default_factory=list)
    retrieved_text_similarities: List[float] = field(default_factory=list)
    retrieved_image_book_ids: List[str] = field(default_factory=list)
    retrieved_image_similarities: List[float] = field(default_factory=list)
    primary_book_ids: List[str] = field(default_factory=list)
    primary_similarities: List[float] = field(default_factory=list)
    embedding_latency_ms: float = 0.0
    query_latency_ms: float = 0.0
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "search_mode": self.search_mode,
            "retrieved_text_book_ids": "|".join(self.retrieved_text_book_ids),
            "retrieved_text_similarities": ",".join(
                f"{s:.4f}" for s in self.retrieved_text_similarities
            ),
            "retrieved_image_book_ids": "|".join(self.retrieved_image_book_ids),
            "retrieved_image_similarities": ",".join(
                f"{s:.4f}" for s in self.retrieved_image_similarities
            ),
            "primary_book_ids": "|".join(self.primary_book_ids),
            "primary_similarities": ",".join(f"{s:.4f}" for s in self.primary_similarities),
            "embedding_latency_ms": round(self.embedding_latency_ms, 2),
            "query_latency_ms": round(self.query_latency_ms, 2),
            "error": self.error or "",
        }


def _embed_text(client, model_id: str, text: str) -> List[float]:
    resp = client.models.embed_content(model=model_id, contents=text)
    return list(resp.embeddings[0].values)


def _embed_image(client, model_id: str, image_path: Path) -> List[float]:
    from PIL import Image
    from google.genai import types

    img = Image.open(image_path).convert("RGB")
    if max(img.size) > MAX_IMAGE_DIM:
        img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
    resp = client.models.embed_content(model=model_id, contents=part)
    return list(resp.embeddings[0].values)


_LOCAL_MODEL_CACHE: Dict[str, Any] = {}


def _load_local_model(model_id: str):
    if model_id in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[model_id]

    try:
        import torch
        import torch.nn as nn
        import transformers.models.clip.modeling_clip as _cm
        if not hasattr(_cm, "clip_loss"):
            def _cl(s: torch.Tensor) -> torch.Tensor:
                cap = nn.functional.cross_entropy(s, torch.arange(len(s), device=s.device))
                img_l = nn.functional.cross_entropy(s.t(), torch.arange(len(s), device=s.device))
                return (cap + img_l) / 2.0
            _cm.clip_loss = _cl
    except Exception:
        pass

    from sentence_transformers import SentenceTransformer
    print(f"  [local] loading model: {model_id}")
    kwargs = {"trust_remote_code": True}
    if "jina" in model_id.lower():
        kwargs["model_kwargs"] = {"modality": "vision", "default_task": "retrieval"}
    model = SentenceTransformer(model_id, **kwargs)
    print(f"  [local] model ready — dim={model.get_embedding_dimension() or model.get_sentence_embedding_dimension()}")
    _LOCAL_MODEL_CACHE[model_id] = model
    return model


def _embed_text_local(model_id: str, text: str) -> List[float]:
    import numpy as np
    model = _load_local_model(model_id)
    if "jina" in model_id.lower():
        emb = model.encode(
            [text],
            task="retrieval",
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    else:
        emb = model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    return emb[0].tolist()


def _embed_image_local(model_id: str, image_path: Path) -> List[float]:
    import numpy as np
    from PIL import Image
    model = _load_local_model(model_id)
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > MAX_IMAGE_DIM:
        img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)
    if "jina" in model_id.lower():
        emb = model.encode(
            [img],
            task="retrieval",
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    else:
        emb = model.encode(
            [img],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    return emb[0].tolist()


def _decide_search_mode(query: EvalQuery) -> str:
    if query.expected_search_mode:
        return query.expected_search_mode

    if not query.has_image:
        return "TEXT"
    if not query.has_text or len(query.query_text.strip()) < 10:
        return "IMAGE"

    generic_phrases = (
        "find books like this", "similar books", "books like this",
        "find similar", "what is this", "what book is this",
        "identify this", "books like it", "look for this",
    )
    msg = query.query_text.lower().strip()
    if any(p in msg for p in generic_phrases):
        return "IMAGE"

    return "HYBRID"


_TITLE_RE = re.compile(r"TITLE:\s*([^\n]+)")


def _title_from_record(metadata: Optional[dict], document: Optional[str]) -> str:
    if isinstance(metadata, dict) and metadata.get("title"):
        return str(metadata["title"]).strip()
    if document:
        m = _TITLE_RE.search(document)
        if m:
            return m.group(1).strip()
    if isinstance(metadata, dict) and metadata.get("book_id"):
        return f"<no title — {metadata['book_id']}>"
    return "<unknown>"


def _query_collection(collection, embedding: List[float]):
    return collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )


def _format_results(raw: dict) -> tuple[List[str], List[float], List[dict]]:
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    book_ids: List[str] = []
    sims: List[float] = []
    metadatas: List[dict] = []
    for meta, dist in zip(metas, dists):
        bid = (meta or {}).get("book_id", "")
        book_ids.append(bid)
        sims.append(1.0 - float(dist))
        metadatas.append(meta or {})
    return book_ids, sims, metadatas


def run_query(client, model_id: str,
              text_collection, image_collection,
              query: EvalQuery) -> QueryResult:
    result = QueryResult(query_id=query.query_id, search_mode=_decide_search_mode(query))
    use_local = not _is_gemini(model_id)

    text_emb: Optional[List[float]] = None
    image_emb: Optional[List[float]] = None

    embed_start = time.perf_counter()
    try:
        if result.search_mode in ("TEXT", "HYBRID") and query.has_text:
            if use_local:
                text_emb = _embed_text_local(model_id, query.query_text)
            else:
                text_emb = _embed_text(client, model_id, query.query_text)
        if result.search_mode in ("IMAGE", "HYBRID") and query.has_image:
            if use_local:
                image_emb = _embed_image_local(model_id, query.resolved_image_path)
            else:
                image_emb = _embed_image(client, model_id, query.resolved_image_path)
    except Exception as e:
        result.error = f"embedding: {e}"
        return result
    finally:
        result.embedding_latency_ms = (time.perf_counter() - embed_start) * 1000.0

    query_start = time.perf_counter()
    try:
        if text_emb is not None:
            tr = _query_collection(text_collection, text_emb)
            tids, tsims, _ = _format_results(tr)
            result.retrieved_text_book_ids = tids
            result.retrieved_text_similarities = tsims

        if image_emb is not None:
            ir = _query_collection(image_collection, image_emb)
            iids, isims, _ = _format_results(ir)
            result.retrieved_image_book_ids = iids
            result.retrieved_image_similarities = isims
    except Exception as e:
        result.error = f"query: {e}"
        return result
    finally:
        result.query_latency_ms = (time.perf_counter() - query_start) * 1000.0

    if result.search_mode == "TEXT":
        result.primary_book_ids = result.retrieved_text_book_ids
        result.primary_similarities = result.retrieved_text_similarities
    elif result.search_mode == "IMAGE":
        result.primary_book_ids = result.retrieved_image_book_ids
        result.primary_similarities = result.retrieved_image_similarities
    else:
        TW, IW = 0.7, 0.3
        merged: Dict[str, Dict[str, float]] = {}
        for bid, sim in zip(result.retrieved_text_book_ids,
                             result.retrieved_text_similarities):
            if not bid:
                continue
            merged[bid] = {"text_sim": sim, "img_sim": None, "score": sim}
        for bid, sim in zip(result.retrieved_image_book_ids,
                             result.retrieved_image_similarities):
            if not bid:
                continue
            entry = merged.get(bid)
            if entry is None:
                merged[bid] = {"text_sim": None, "img_sim": sim, "score": IW * sim}
            else:
                entry["img_sim"] = sim
                entry["score"] = TW * entry["text_sim"] + IW * sim
        ranked = sorted(merged.items(), key=lambda kv: kv[1]["score"], reverse=True)
        result.primary_book_ids = [bid for bid, _ in ranked][:TOP_K]
        result.primary_similarities = [d["score"] for _, d in ranked][:TOP_K]

    return result


def metrics_for_query(query: EvalQuery, result: QueryResult) -> Dict[str, Any]:
    expected = query.expected_book_ids
    primary = result.primary_book_ids

    row: Dict[str, Any] = {
        "query_id": query.query_id,
        "query_text": (query.query_text or "")[:200],
        "image_path": query.image_path or "",
        "expected_book_ids": "|".join(expected),
        "search_mode": result.search_mode,
        "primary_book_ids": "|".join(primary),
        "primary_similarities": ",".join(f"{s:.4f}" for s in result.primary_similarities),
        "first_hit_rank": next(
            (i + 1 for i, b in enumerate(primary) if b in set(expected)),
            None,
        ) or 0,
        "hit@1": hit_rate_at_k(primary, expected, 1),
        "hit@5": hit_rate_at_k(primary, expected, 5),
        "hit@10": hit_rate_at_k(primary, expected, 10),
        "recall@1": recall_at_k(primary, expected, 1),
        "recall@5": recall_at_k(primary, expected, 5),
        "recall@10": recall_at_k(primary, expected, 10),
        "mrr": reciprocal_rank(primary, expected),
        "ndcg@10": ndcg_at_k(primary, expected, 10),
        "embedding_latency_ms": round(result.embedding_latency_ms, 2),
        "query_latency_ms": round(result.query_latency_ms, 2),
        "latency_ms": round(result.embedding_latency_ms + result.query_latency_ms, 2),
        "error": result.error or "",
    }
    return row
