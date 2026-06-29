import asyncio
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, List
import chromadb
import httpx
import requests
import chainlit as cl
from dotenv import load_dotenv
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OWNER_PASSWORD     = os.getenv("OWNER_PASSWORD", "")
CHROMA_PATH        = "./chroma_db"
DATA_PATH          = "./data_exports"

if not OPENROUTER_API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY is not set in your .env file.")
if not OWNER_PASSWORD:
    raise EnvironmentError(
        "OWNER_PASSWORD is not set in your .env file. "
        "Set a strong value — owner mode unlocks internal diagnostic data."
    )


# ===========================================================================
# 1. EMBEDDING
# ===========================================================================
class OpenRouterEmbedding(BaseEmbedding):
    api_key:    str
    model_name: str

    def __init__(self, api_key: str, model_name: str, **kwargs: Any):
        super().__init__(api_key=api_key, model_name=model_name, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "OpenRouterEmbedding"

    # ── sync interface ──────────────────────────────────────────────────────
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_with_retry(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_with_retry(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [
            self._embed_with_retry(text.strip() or "empty document")
            for text in texts
        ]

    def _embed_with_retry(self, text: str, max_retries: int = 5) -> List[float]:
        last_error: str = ""
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    url="https://openrouter.ai/api/v1/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={
                        "model":           self.model_name,
                        "input":           text,
                        "encoding_format": "float",
                    },
                    timeout=30,
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = f"{type(e).__name__}: {e}"
                wait = 5 * (attempt + 1)
                print(f"  [Embed] Connection error ({type(e).__name__}). Retry {attempt+1}/{max_retries} in {wait}s.")
                time.sleep(wait)
                continue

            if resp.status_code == 200:
                body = resp.json()
                if "error" in body:
                    last_error = f"API error: {body['error'].get('message')}"
                    print(f"  [Embed] {last_error}. Retrying...")
                    time.sleep(2)
                    continue
                data = body.get("data", [])
                if data:
                    return data[0]["embedding"]
                last_error = "200 OK but empty data array"
                time.sleep(2)
            elif resp.status_code == 429:
                last_error = "rate limited (429)"
                wait = 5 * (attempt + 1)
                print(f"  [Embed] Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                print(f"  [Embed] {last_error}. Retrying...")
                time.sleep(2)

        raise RuntimeError(
            f"Embedding failed after {max_retries} attempts. Last error: {last_error}"
        )

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._aembed_with_retry(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._aembed_with_retry(text)

    async def _aembed_with_retry(self, text: str, max_retries: int = 5) -> List[float]:
        """Async sibling of _embed_with_retry — non-blocking for use under Chainlit."""
        last_error: str = ""
        async with httpx.AsyncClient(timeout=30) as client:
            for attempt in range(max_retries):
                try:
                    resp = await client.post(
                        url="https://openrouter.ai/api/v1/embeddings",
                        headers={"Authorization": f"Bearer {self.api_key}",
                                 "Content-Type": "application/json"},
                        json={
                            "model":           self.model_name,
                            "input":           text,
                            "encoding_format": "float",
                        },
                    )
                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    last_error = f"{type(e).__name__}: {e}"
                    wait = 5 * (attempt + 1)
                    print(f"  [Embed-async] Connection error. Retry {attempt+1}/{max_retries} in {wait}s.")
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code == 200:
                    body = resp.json()
                    if "error" in body:
                        last_error = f"API error: {body['error'].get('message')}"
                        await asyncio.sleep(2)
                        continue
                    data = body.get("data", [])
                    if data:
                        return data[0]["embedding"]
                    last_error = "200 OK but empty data array"
                    await asyncio.sleep(2)
                elif resp.status_code == 429:
                    last_error = "rate limited (429)"
                    wait = 5 * (attempt + 1)
                    await asyncio.sleep(wait)
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    await asyncio.sleep(2)

        raise RuntimeError(
            f"Async embedding failed after {max_retries} attempts. Last error: {last_error}"
        )


# ===========================================================================
# 2. LLM & EMBEDDING SETUP
# ===========================================================================
Settings.llm = OpenAILike(
    api_key=OPENROUTER_API_KEY,
    api_base="https://openrouter.ai/api/v1",
    model="google/gemini-2.5-flash",
    is_chat_model=True,
    max_tokens=1500,
)

Settings.embed_model = OpenRouterEmbedding(
    api_key=OPENROUTER_API_KEY,
    model_name="google/gemini-embedding-001",
)


# ===========================================================================
# 3. SYSTEM PROMPTS
# ===========================================================================
SYSTEM_PROMPTS = {
    "customer": (
        "You are a polite, helpful customer service assistant for our Shopee store.\n"
        "**Always respond in Bahasa Indonesia by default.** Match the user's language only "
        "if they clearly write in another language for several turns.\n"
        "Answer the user's question using ONLY the provided context.\n\n"

        "── Price & Stock rules ──\n"
        "Pay close attention to [LIVE INVENTORY STATUS] blocks for real-time price and stock.\n"
        "When a product has variants with different stocks, mention the in-stock variants by name.\n"
        "Even if some variants are out of stock, ALWAYS provide the price if it appears in the context.\n"
        "If the customer asks about a specific product by name and ALL its variants have 0 stock, "
        "you may acknowledge the product exists but inform them it is currently out of stock.\n\n"

        "── Ranking rules ──\n"
        "When the context contains a ranked list (e.g. 'MOST EXPENSIVE IN-STOCK PRODUCTS', "
        "'CHEAPEST IN-STOCK PRODUCTS', 'HIGHEST-RATED PRODUCTS'), trust that list — "
        "it is the full sorted ranking for the whole shop. Do NOT try to re-rank "
        "from individual product entries; those are only a similarity sample.\n"
        "For general ranking questions like 'paling mahal' or 'paling murah', "
        "ONLY consider products that are currently in stock.\n\n"

        "── Information boundaries ──\n"
        "You may discuss product names, descriptions, prices, stock, variants, "
        "ratings, and customer reviews.\n"
        "Do NOT reveal or discuss: units sold (sales count), page view counts, "
        "like counts, shop health, violations, diagnostics, quality scores, "
        "or any internal operational data. These are NOT publicly available.\n"
        "If the customer asks for information you don't have or that is internal, "
        "politely say you don't have that information.\n"
        "Do NOT provide information about the shop owner, supplier, financials, "
        "discount vouchers, or physical address."
    ),
    "owner": (
        "You are an internal Product Manager and Diagnostic Assistant for this Shopee store.\n"
        "**Respond in Bahasa Indonesia by default.** Match the user's language if they switch.\n"
        "You have FULL access to all data including: sales counts, view counts, like counts, "
        "diagnostics, violations, quality scores, category recommendations, and shop performance.\n"
        "Analyze the provided context — reviews, diagnostics, violation data — to identify "
        "product weaknesses, recurring complaints, and actionable improvements.\n"
        "When the context contains a ranked list (e.g. 'BEST-SELLING PRODUCTS'), trust "
        "that list as the full sorted ranking for the whole shop.\n"
        "For general rankings (most expensive, cheapest), consider ALL products "
        "regardless of stock status.\n"
        "Be specific, data-driven, and structured."
    ),
}


# ===========================================================================
# 4. DATA HELPERS
# ===========================================================================
def _extract_variant_stock(variant: dict) -> int:
    siv2 = variant.get("stock_info_v2")
    if isinstance(siv2, dict):
        summary = siv2.get("summary_info") or {}
        if "total_available_stock" in summary:
            return int(summary.get("total_available_stock") or 0)
        seller_stock = siv2.get("seller_stock") or []
        if isinstance(seller_stock, list):
            return sum(int(s.get("stock") or 0) for s in seller_stock if isinstance(s, dict))
    return 0


def _extract_variant_price(variant: dict) -> float:
    pi = variant.get("price_info") or []
    if isinstance(pi, list) and pi and isinstance(pi[0], dict):
        return float(pi[0].get("current_price") or 0)
    if isinstance(pi, dict):
        return float(pi.get("current_price") or 0)
    return 0.0


def _variant_label(variant: dict) -> str:
    return (variant.get("model_name") or variant.get("model_sku") or "Default").strip() or "Default"


def _load_json(filename: str) -> list:
    path = os.path.join(DATA_PATH, f"{filename}.json")
    if not os.path.exists(path):
        print(f"  [Data] Warning: {path} not found. Skipping.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_live_inventory() -> dict:
    raw = _load_json("bucket_B_models")
    if not raw:
        print("  [Data] Warning: Bucket B not found. Live inventory unavailable.")
        return {}
    inventory = {str(item["item_id"]): item for item in raw}
    print(f"  [Data] Loaded {len(inventory):,} live inventory entries.")
    return inventory


def _compute_total_stock(live_item: dict) -> int:
    return sum(
        _extract_variant_stock(v)
        for v in (live_item.get("models", []) or [])
    )


def _compute_out_of_stock_ids(live_inventory: dict) -> set:
    oos = set()
    for item_id, data in live_inventory.items():
        if _compute_total_stock(data) == 0:
            oos.add(item_id)
    print(f"  [Data] {len(oos):,} product(s) are fully out of stock.")
    return oos


def _build_price_snapshot(models_raw: list) -> dict:
    snapshot: dict[str, dict] = {}
    for item in models_raw:
        item_id = str(item["item_id"])
        prices  = [_extract_variant_price(m) for m in item.get("models", [])]
        prices  = [p for p in prices if p > 0]
        if prices:
            snapshot[item_id] = {"min": min(prices), "max": max(prices)}
    return snapshot


def _build_extra_info_lookup(extra_raw: list) -> dict:
    return {str(item.get("item_id", "")): item for item in extra_raw}


def _build_category_lookup(categories_raw: list) -> dict:
    lookup: dict[str, str] = {}
    for cat in categories_raw:
        cat_id = str(cat.get("category_id", "")).strip()
        name   = (cat.get("display_category_name")
                  or cat.get("category_name")
                  or cat.get("name", "")).strip()
        if cat_id and name:
            lookup[cat_id] = name
    if lookup:
        print(f"  [Data] Loaded {len(lookup)} category id → name mappings.")
    return lookup


def _build_review_aggregates(comments_raw: list, id_to_name: dict) -> tuple[dict, dict]:
    by_item: dict[str, list] = defaultdict(list)
    for row in comments_raw:
        item_id = str(row.get("item_id", ""))
        text    = str(row.get("comment", "")).strip()
        rating  = float(row.get("rating_star", 0))
        if text:
            by_item[item_id].append({"text": text, "rating": rating})

    summaries: dict[str, str] = {}
    for item_id, reviews in by_item.items():
        name    = id_to_name.get(item_id, "this product")
        ratings = [r["rating"] for r in reviews]
        avg     = sum(ratings) / len(ratings)
        total   = len(ratings)
        pos     = sum(1 for r in ratings if r >= 4)
        neg     = sum(1 for r in ratings if r <= 2)

        pos_snippets = [r["text"][:120] for r in reviews if r["rating"] >= 4][:3]
        neg_snippets = [r["text"][:120] for r in reviews if r["rating"] <= 2][:3]

        lines = [
            f"Review summary for '{name}': {total} reviews, avg {avg:.1f}/5 stars.",
            f"Breakdown: {pos} positive, {total - pos - neg} neutral, {neg} negative.",
        ]
        if pos_snippets:
            lines.append("Common praise: " + " | ".join(pos_snippets))
        if neg_snippets:
            lines.append("Common complaints: " + " | ".join(neg_snippets))

        summaries[item_id] = "\n".join(lines)

    return summaries, by_item


# ===========================================================================
# 5. CHROMADB — DUAL COLLECTION SETUP
# ===========================================================================
def _build_public_aggregate_docs(
    records: list[dict],
    top_n: int = 20,
) -> list[Document]:
    docs: list[Document] = []

    def add(header: str, rows: list, line_fmt, doc_type: str) -> None:
        if not rows:
            return
        lines = [f"{i}. {line_fmt(r)}" for i, r in enumerate(rows, 1)]
        docs.append(Document(
            text=header + "\n" + "\n".join(lines),
            metadata={"type": doc_type, "item_id": "GLOBAL"},
        ))

    # 1. Most expensive IN-STOCK
    in_stock_expensive = sorted(
        [r for r in records if r["price_max"] > 0 and r["total_stock"] > 0],
        key=lambda r: r["price_max"], reverse=True,
    )[:top_n]
    add(
        "MOST EXPENSIVE IN-STOCK PRODUCTS — barang paling mahal, harga termahal, "
        "harga paling tinggi, produk termahal, most expensive.\n"
        "Only products currently in stock are listed. "
        "Ranked by maximum variant price (highest first):",
        in_stock_expensive,
        lambda r: f"{r['name']} (category: {r['category_name']}) — "
                  f"Rp {r['price_max']:,.0f} ({r['total_stock']} units in stock)",
        "aggregate_expensive_in_stock",
    )

    # 2. Cheapest IN-STOCK
    in_stock_cheap = sorted(
        [r for r in records if r["price_min"] > 0 and r["total_stock"] > 0],
        key=lambda r: r["price_min"],
    )[:top_n]
    add(
        "CHEAPEST IN-STOCK PRODUCTS — barang paling murah, harga termurah, "
        "harga paling rendah, produk termurah, cheapest.\n"
        "Only products currently in stock are listed. "
        "Ranked by minimum variant price (lowest first):",
        in_stock_cheap,
        lambda r: f"{r['name']} (category: {r['category_name']}) — "
                  f"Rp {r['price_min']:,.0f} ({r['total_stock']} units in stock)",
        "aggregate_cheap_in_stock",
    )

    # 3. Highest rated
    MIN_REVIEWS = 5
    rated = sorted(
        [r for r in records if r["comment_count"] >= MIN_REVIEWS],
        key=lambda r: (r["rating_star"], r["comment_count"]),
        reverse=True,
    )[:top_n]
    add(
        f"HIGHEST-RATED PRODUCTS — rating tertinggi, rating paling bagus, "
        f"produk dengan bintang terbaik (minimum {MIN_REVIEWS} reviews).\n"
        "Ranked by average star rating, ties broken by review count:",
        rated,
        lambda r: f"{r['name']} (category: {r['category_name']}) — "
                  f"{r['rating_star']:.2f}/5 ({r['comment_count']} reviews)",
        "aggregate_highest_rated",
    )

    # 4. Most reviewed
    reviewed = sorted(
        [r for r in records if r["comment_count"] > 0],
        key=lambda r: r["comment_count"], reverse=True,
    )[:top_n]
    add(
        "MOST-REVIEWED PRODUCTS — ulasan terbanyak, paling banyak ulasan, "
        "paling banyak rating, paling banyak komentar, most reviews.\n"
        "Ranked by number of customer reviews (highest first):",
        reviewed,
        lambda r: f"{r['name']} (category: {r['category_name']}) — "
                  f"{r['comment_count']} reviews ({r['rating_star']:.1f}/5 avg)",
        "aggregate_most_reviewed",
    )

    return docs


def _build_private_aggregate_docs(
    records: list[dict],
    top_n: int = 20,
) -> list[Document]:
    docs: list[Document] = []

    def add(header: str, rows: list, line_fmt, doc_type: str) -> None:
        if not rows:
            return
        lines = [f"{i}. {line_fmt(r)}" for i, r in enumerate(rows, 1)]
        docs.append(Document(
            text=header + "\n" + "\n".join(lines),
            metadata={"type": doc_type, "item_id": "GLOBAL"},
        ))

    # 1. Best sellers 
    sold = sorted(
        [r for r in records if r["sale"] > 0],
        key=lambda r: r["sale"], reverse=True,
    )[:top_n]
    add(
        "BEST-SELLING PRODUCTS — paling laku, paling banyak terjual, "
        "paling banyak dibeli, produk paling laris, best sellers.\n"
        "Ranked by total units sold (highest first):",
        sold,
        lambda r: f"{r['name']} (category: {r['category_name']}) — "
                  f"{r['sale']} units sold",
        "aggregate_bestsellers",
    )

    # 2. Most viewed (owner-only metric)
    viewed = sorted(
        [r for r in records if r["views"] > 0],
        key=lambda r: r["views"], reverse=True,
    )[:top_n]
    add(
        "MOST-VIEWED PRODUCTS — paling banyak dilihat, paling sering dilihat, "
        "produk populer dilihat, most viewed.\n"
        "Ranked by total page views (highest first):",
        viewed,
        lambda r: f"{r['name']} (category: {r['category_name']}) — "
                  f"{r['views']} views",
        "aggregate_most_viewed",
    )

    # 3. Most liked (owner-only metric)
    liked = sorted(
        [r for r in records if r["likes"] > 0],
        key=lambda r: r["likes"], reverse=True,
    )[:top_n]
    add(
        "MOST-LIKED PRODUCTS — paling banyak disukai, paling banyak like, "
        "produk paling disukai pelanggan, most liked.\n"
        "Ranked by total likes (highest first):",
        liked,
        lambda r: f"{r['name']} (category: {r['category_name']}) — "
                  f"{r['likes']} likes",
        "aggregate_most_liked",
    )

    # 4. Most expensive ALL products (including out-of-stock), owner only
    expensive = sorted(
        [r for r in records if r["price_max"] > 0],
        key=lambda r: r["price_max"], reverse=True,
    )[:top_n]
    add(
        "MOST EXPENSIVE PRODUCTS (ALL, including out-of-stock) — barang paling mahal "
        "termasuk yang stoknya habis, harga termahal semua produk.\n"
        "Ranked by maximum variant price (highest first), regardless of stock:",
        expensive,
        lambda r: f"{r['name']} (category: {r['category_name']}) — "
                  f"Rp {r['price_max']:,.0f} (stock: {r['total_stock']})",
        "aggregate_most_expensive_all",
    )

    # 5. Cheapest ALL products (including out-of-stock), owner only
    cheap = sorted(
        [r for r in records if r["price_min"] > 0],
        key=lambda r: r["price_min"],
    )[:top_n]
    add(
        "CHEAPEST PRODUCTS (ALL, including out-of-stock) — barang paling murah "
        "termasuk yang stoknya habis, harga termurah semua produk.\n"
        "Ranked by minimum variant price (lowest first), regardless of stock:",
        cheap,
        lambda r: f"{r['name']} (category: {r['category_name']}) — "
                  f"Rp {r['price_min']:,.0f} (stock: {r['total_stock']})",
        "aggregate_cheapest_all",
    )

    return docs


def _batch_ingest(docs: list, storage_ctx: StorageContext, label: str) -> VectorStoreIndex:
    index      = None
    batch_size = 50
    total      = (len(docs) + batch_size - 1) // batch_size

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        n     = (i // batch_size) + 1
        print(f"  [{label}] Batch {n}/{total} ({len(batch)} docs)...")

        if index is None:
            index = VectorStoreIndex.from_documents(batch, storage_context=storage_ctx)
        else:
            for doc in batch:
                index.insert(doc)

        if i + batch_size < len(docs):
            time.sleep(10)

    return index


def initialize_databases() -> tuple[VectorStoreIndex, VectorStoreIndex]:
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    pub_col  = client.get_or_create_collection("public_kb")
    priv_col = client.get_or_create_collection("private_kb")

    pub_store  = ChromaVectorStore(chroma_collection=pub_col)
    priv_store = ChromaVectorStore(chroma_collection=priv_col)

    if pub_col.count() > 0 and priv_col.count() > 0:
        print(f"  [DB] Loaded — public_kb: {pub_col.count():,} | private_kb: {priv_col.count():,}")
        return (
            VectorStoreIndex.from_vector_store(pub_store),
            VectorStoreIndex.from_vector_store(priv_store),
        )

    print("  [DB] Collections empty. Building from JSON exports...")

    shop_info_raw      = _load_json("bucket_A_shop_info")
    shop_rating_raw    = _load_json("bucket_A_shop_rating")
    status_counts_raw  = _load_json("bucket_A_item_status_counts")
    categories_raw     = _load_json("bucket_A_categories")
    catalog_raw        = _load_json("bucket_A_catalog")
    extra_info_raw     = _load_json("bucket_A_extra_info")
    models_raw         = _load_json("bucket_B_models")
    comments_raw       = _load_json("bucket_C_comments")
    diagnostics_raw    = _load_json("bucket_C_diagnostics")
    low_quality_raw    = _load_json("bucket_C_low_quality_items")
    violations_raw     = _load_json("bucket_C_violations")
    cat_rec_raw        = _load_json("bucket_C_category_recommend")
    shop_perf_raw      = _load_json("bucket_C_shop_performance")

    id_to_name     = {str(i["item_id"]): i.get("item_name", "Unknown") for i in catalog_raw}
    price_snapshot = _build_price_snapshot(models_raw)
    extra_lookup   = _build_extra_info_lookup(extra_info_raw)
    cat_lookup     = _build_category_lookup(categories_raw)
    review_summaries, reviews_by_item = _build_review_aggregates(comments_raw, id_to_name)

    # Map item_id → category_name
    id_to_category: dict[str, str] = {}
    for item in catalog_raw:
        item_id = str(item.get("item_id", ""))
        cat_id  = str(item.get("category_id", "")).strip()
        id_to_category[item_id] = cat_lookup.get(cat_id, "")

    # Build per-product records
    records: list[dict] = []
    for item_id, name in id_to_name.items():
        extra = extra_lookup.get(item_id, {})
        snap  = price_snapshot.get(item_id)
        live  = LIVE_INVENTORY.get(item_id)
        total_stock = _compute_total_stock(live) if live else 0

        records.append({
            "item_id":       item_id,
            "name":          name,
            "category_name": id_to_category.get(item_id, ""),
            "sale":          extra.get("sale") or 0,
            "views":         extra.get("views") or 0,
            "likes":         extra.get("likes") or 0,
            "rating_star":   extra.get("rating_star") or 0,
            "comment_count": extra.get("comment_count") or 0,
            "price_min":     snap["min"] if snap else 0,
            "price_max":     snap["max"] if snap else 0,
            "total_stock":   total_stock,
        })

    # PUBLIC documents
    public_docs: list[Document] = []

    # 1. Shop info
    shop_lines = []
    if shop_info_raw:
        shop      = shop_info_raw[0]
        shop_name = shop.get("shop_name") or "our Shopee store"
        region    = shop.get("region", "")
        status    = shop.get("status", "")
        create_ts = shop.get("create_time") or shop.get("created_at")

        shop_lines.extend([
            f"Our shop name is '{shop_name}'.",
            f"This is our official Shopee store, also known as '{shop_name}'.",
            f"You are chatting with the assistant for the Shopee shop '{shop_name}'.",
        ])
        if region:
            _REGION_NAMES: dict[str, str] = {
                "ID": "Indonesia", "SG": "Singapore", "MY": "Malaysia",
                "TH": "Thailand",  "PH": "Philippines", "VN": "Vietnam",
                "TW": "Taiwan",    "BR": "Brazil",      "MX": "Mexico",
                "CO": "Colombia",  "CL": "Chile",       "PL": "Poland",
                "FR": "France",    "ES": "Spain",       "IN": "India",
            }
            region_display = _REGION_NAMES.get(region.upper(), region)
            shop_lines.append(
                f"The shop is located in {region_display} (region code: {region})."
            )
        if status:
            shop_lines.append(f"Shop status: {status}.")
        if create_ts:
            try:
                join_dt      = datetime.fromtimestamp(int(create_ts), tz=timezone.utc)
                join_year    = join_dt.year
                now_year     = datetime.now(tz=timezone.utc).year
                years_active = now_year - join_year
                shop_lines.append(
                    f"The shop joined Shopee in {join_year} "
                    f"({years_active} tahun lalu / {years_active} years ago). "
                    f"Join date: {join_dt.strftime('%d %B %Y')}."
                )
            except (ValueError, OSError, OverflowError):
                shop_lines.append(f"Shop creation timestamp: {create_ts}.")
    else:
        shop_lines.append(
            "This is our official Shopee store. Ask about products, prices, stock, or reviews."
        )

    if status_counts_raw:
        for entry in status_counts_raw:
            if str(entry.get("status", "")).upper() == "NORMAL":
                count = entry.get("count", 0)
                shop_lines.append(
                    f"The shop currently has {count} active (NORMAL) products listed."
                )
                break
        total_entry = next(
            (e for e in status_counts_raw if str(e.get("status", "")).upper() == "TOTAL"),
            None,
        )
        if total_entry:
            shop_lines.append(
                f"Total products across all statuses (including unlisted): "
                f"{total_entry.get('count', 0)}."
            )

    if shop_rating_raw:
        sr = shop_rating_raw[0]
        rating  = sr.get("computed_rating_star", "")
        reviews = sr.get("computed_total_reviews", "")
        if rating:
            shop_lines.append(f"Shop rating (penilaian toko): {rating} stars.")
        if reviews:
            shop_lines.append(f"Total shop reviews (total penilaian): {reviews}.")

    public_docs.append(Document(
        text=" ".join(shop_lines),
        metadata={"type": "shop_info", "item_id": "GLOBAL"},
    ))

    # 2. Product catalog
    for item in catalog_raw:
        item_id  = str(item.get("item_id", ""))
        name     = item.get("item_name", "Unknown")
        desc     = str(item.get("description", "")).replace("\n", " ").strip()
        snap     = price_snapshot.get(item_id)
        cat_name = id_to_category.get(item_id, "")

        price_hint = (
            f"Approximate price range: Rp {snap['min']:,.0f} – Rp {snap['max']:,.0f} "
            "(exact real-time values shown below). "
            if snap else ""
        )

        category_hint = f"Category: {cat_name}. " if cat_name else ""

        extra      = extra_lookup.get(item_id, {})
        avg_rating = extra.get("rating_star")
        rating_hint = f"Average rating: {avg_rating}/5. " if avg_rating else ""

        public_docs.append(Document(
            text=f"Product: {name}. {category_hint}{price_hint}{rating_hint}"
                 f"Description: {desc}",
            metadata={
                "type": "product_info",
                "item_id": item_id,
                "item_name": name,
                "category_name": cat_name,
            },
        ))

    for item in models_raw:
        item_id    = str(item.get("item_id", ""))
        name       = id_to_name.get(item_id, "Unknown")
        variations = item.get("variations", [])
        if not variations:
            continue
        var_parts = []
        for tier in variations:
            tier_name = tier.get("name", "")
            options   = [o.get("name", "") for o in tier.get("option_list", []) if o.get("name")]
            if tier_name and options:
                var_parts.append(f"{tier_name}: {', '.join(options)}")
        if var_parts:
            public_docs.append(Document(
                text=f"Available variations for '{name}': {'; '.join(var_parts)}.",
                metadata={"type": "variation_info", "item_id": item_id, "item_name": name},
            ))

    for item_id, summary_text in review_summaries.items():
        public_docs.append(Document(
            text=summary_text,
            metadata={
                "type":      "review_summary",
                "item_id":   item_id,
                "item_name": id_to_name.get(item_id, "Unknown"),
            },
        ))

    public_agg = _build_public_aggregate_docs(records)
    if public_agg:
        print(f"  [DB] Built {len(public_agg)} public ranking doc(s).")
        public_docs.extend(public_agg)

    # PRIVATE documents
    private_docs: list[Document] = []

    # 0. Product catalog WITH full internal metrics (sales, views, likes)
    for item in catalog_raw:
        item_id  = str(item.get("item_id", ""))
        name     = item.get("item_name", "Unknown")
        extra    = extra_lookup.get(item_id, {})
        cat_name = id_to_category.get(item_id, "")
        snap     = price_snapshot.get(item_id)

        sold   = extra.get("sale")
        views  = extra.get("views")
        likes  = extra.get("likes")

        metrics = []
        if sold is not None:
            metrics.append(f"Units sold: {sold}")
        if views is not None:
            metrics.append(f"Page views: {views}")
        if likes is not None:
            metrics.append(f"Likes: {likes}")

        if metrics:
            price_hint = (
                f"Price range: Rp {snap['min']:,.0f} – Rp {snap['max']:,.0f}. "
                if snap else ""
            )
            category_hint = f"Category: {cat_name}. " if cat_name else ""
            private_docs.append(Document(
                text=f"Internal metrics for '{name}': {category_hint}{price_hint}"
                     f"{'. '.join(metrics)}.",
                metadata={
                    "type": "product_metrics",
                    "item_id": item_id,
                    "item_name": name,
                    "category_name": cat_name,
                },
            ))

    # 1. PRIVATE aggregate ranking docs
    private_agg = _build_private_aggregate_docs(records)
    if private_agg:
        print(f"  [DB] Built {len(private_agg)} private ranking doc(s).")
        private_docs.extend(private_agg)

    # 2. Individual reviews
    for item_id, reviews in reviews_by_item.items():
        name = id_to_name.get(item_id, "Unknown")
        for r in reviews:
            rating    = r["rating"]
            sentiment = "positive" if rating >= 4 else ("neutral" if rating == 3 else "negative")
            private_docs.append(Document(
                text=f"Review for '{name}' ({rating}/5 stars): \"{r['text']}\"",
                metadata={"type": "individual_review", "item_id": item_id, "sentiment": sentiment},
            ))

    # 3. Content diagnosis (quality level + issues + suggestions, per product)
    for item in diagnostics_raw:
        item_id = str(item.get("item_id", ""))
        name    = id_to_name.get(item_id, "Unknown")
        private_docs.append(Document(
            text=f"Content diagnosis for '{name}': {json.dumps(item, ensure_ascii=False)}",
            metadata={"type": "diagnostic", "item_id": item_id},
        ))

    # 4. Low/medium quality items sweep
    if low_quality_raw:
        low_lines = [
            f"- [{r.get('_quality_level', '?')}] "
            f"{id_to_name.get(str(r.get('item_id', '')), str(r.get('item_id', '')))}"
            for r in low_quality_raw
        ]
        low_count = sum(1 for r in low_quality_raw if r.get("_quality_level") == "LOW")
        med_count = sum(1 for r in low_quality_raw if r.get("_quality_level") == "MEDIUM")
        private_docs.append(Document(
            text=(
                f"Products with LOW or MEDIUM content quality (need attention). "
                f"Total: {len(low_quality_raw)} products ({low_count} LOW, {med_count} MEDIUM).\n"
                + "\n".join(low_lines)
            ),
            metadata={"type": "low_quality_summary", "item_id": "GLOBAL"},
        ))

    # 5. Violation info (per product)
    for item in violations_raw:
        item_id = str(item.get("item_id", ""))
        name    = id_to_name.get(item_id, "Unknown")
        private_docs.append(Document(
            text=f"Violation info for '{name}': {json.dumps(item, ensure_ascii=False)}",
            metadata={"type": "violation", "item_id": item_id},
        ))

    # 6. Category recommendations (per product)
    for rec in cat_rec_raw:
        item_id    = str(rec.get("item_id", ""))
        name       = rec.get("item_name", id_to_name.get(item_id, "Unknown"))
        current    = rec.get("current_category_id", "unknown")
        suggested  = rec.get("recommended_categories", [])
        if not suggested:
            continue
        top         = suggested[0]
        top_cat     = top.get("category_name", top.get("category_id", "?"))
        confidence  = top.get("confidence", "?")
        private_docs.append(Document(
            text=(
                f"Category recommendation for '{name}': "
                f"currently in category {current}. "
                f"Shopee suggests category '{top_cat}' (confidence: {confidence}). "
                f"Miscategorisation reduces search visibility."
            ),
            metadata={"type": "category_recommend", "item_id": item_id},
        ))

    # 7. Shop performance KPIs
    if shop_perf_raw:
        perf = shop_perf_raw[0]
        private_docs.append(Document(
            text=f"Shop performance metrics: {json.dumps(perf, ensure_ascii=False)}",
            metadata={"type": "shop_performance", "item_id": "GLOBAL"},
        ))

    # ── Ingest ──────────────────────────────────────────────────────────────
    print(f"\n  Embedding {len(public_docs):,} public documents...")
    pub_ctx = StorageContext.from_defaults(vector_store=pub_store)
    pub_idx = _batch_ingest(public_docs, pub_ctx, "PUBLIC")

    print(f"\n  Embedding {len(private_docs):,} private documents...")
    priv_ctx = StorageContext.from_defaults(vector_store=priv_store)
    priv_idx = _batch_ingest(private_docs, priv_ctx, "PRIVATE")

    print("\n  [DB] Both collections built and persisted.\n")
    return pub_idx, priv_idx

print("[Init] Loading live inventory...")
LIVE_INVENTORY = _load_live_inventory()
OUT_OF_STOCK_IDS = _compute_out_of_stock_ids(LIVE_INVENTORY)

print("[Init] Initializing ChromaDB...")
PUBLIC_INDEX, PRIVATE_INDEX = initialize_databases()
PUBLIC_RETRIEVER  = PUBLIC_INDEX.as_retriever(similarity_top_k=15)
PRIVATE_RETRIEVER = PRIVATE_INDEX.as_retriever(similarity_top_k=12)

print("[Init] Ready.\n")


# ===========================================================================
# 7. CONTEXT BUILDER
# ===========================================================================
def _retrieve_nodes(query: str, mode: str) -> list:
    nodes = list(PUBLIC_RETRIEVER.retrieve(query))

    if mode == "owner":
        seen = {n.node_id for n in nodes}
        for n in PRIVATE_RETRIEVER.retrieve(query):
            if n.node_id not in seen:
                nodes.append(n)
                seen.add(n.node_id)

    return nodes


def _inventory_block_for_item(item_id: str) -> str:
    if item_id not in LIVE_INVENTORY:
        return ""

    live     = LIVE_INVENTORY[item_id]
    variants = live.get("models", []) or []

    rendered       = []
    in_stock_count = 0
    for v in variants:
        stock = _extract_variant_stock(v)
        price = _extract_variant_price(v)
        rendered.append((_variant_label(v), stock, price))
        if stock > 0:
            in_stock_count += 1

    inv_lines = ["  [LIVE INVENTORY STATUS]:"]
    if rendered:
        if in_stock_count == 0:
            inv_lines.append(
                f"    (All {len(rendered)} variant(s) currently out of stock"
                " — price still available below.)"
            )
        else:
            inv_lines.append(
                f"    ({in_stock_count} of {len(rendered)} variant(s) currently in stock.)"
            )
        for label, stock, price in rendered:
            state = "in stock" if stock > 0 else "out of stock"
            inv_lines.append(
                f"    * {label} | Stock: {stock} units ({state}) | Price: Rp {price:,.0f}"
            )
    else:
        inv_lines.append("    (No variant data available for this item.)")

    return "\n".join(inv_lines)


def _nodes_to_context(nodes: list, mode: str) -> str:
    if not nodes:
        return "No relevant information found in the knowledge base."

    lines: list[str]   = []
    injected: set[str] = set()

    for node in nodes:
        lines.append(f"- {node.text}")

        item_id = str(node.metadata.get("item_id", ""))
        if item_id and item_id not in injected:
            block = _inventory_block_for_item(item_id)
            if block:
                injected.add(item_id)
                lines.append(block)

        lines.append("")

    return "\n".join(lines).strip()


def build_context(query: str, mode: str) -> str:
    return _nodes_to_context(_retrieve_nodes(query, mode), mode)


# ===========================================================================
# 8. RETRIEVAL + GENERATION
# ===========================================================================
def _build_chat_messages(query: str, mode: str, full_context: str) -> list[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPTS[mode]),
        ChatMessage(
            role=MessageRole.USER,
            content=f"Context:\n{full_context}\n\nQuestion: {query}",
        ),
    ]


def retrieve_and_generate(query: str, mode: str) -> dict:
    nodes = _retrieve_nodes(query, mode)
    seen_inv: set[str] = set()
    ragas_contexts: list[str] = []
    for n in nodes:
        entry    = n.text
        item_id  = str(n.metadata.get("item_id", ""))
        if item_id and item_id not in seen_inv:
            block = _inventory_block_for_item(item_id)
            if block:
                seen_inv.add(item_id)
                entry = entry + "\n" + block
        ragas_contexts.append(entry)

    full_context = _nodes_to_context(nodes, mode)

    chat_messages = _build_chat_messages(query, mode, full_context)

    response = Settings.llm.chat(chat_messages)
    answer   = response.message.content

    return {
        "answer":   answer,
        "contexts": ragas_contexts,
        "context":  full_context,
        "mode":     mode,
    }


# ===========================================================================
# 9. CHAINLIT UI
# ===========================================================================
@cl.on_chat_start
async def start_chat():
    cl.user_session.set("mode", "customer")
    cl.user_session.set("owner_authenticated", False)

    actions = [
        cl.Action(name="switch_mode", payload={"value": "customer"}, label="👤 Customer Mode"),
        cl.Action(name="switch_mode", payload={"value": "owner"},    label="🛠️ Shop Owner Mode"),
    ]
    await cl.Message(
        content="Welcome to the Shopee AI Assistant! Please select your mode to begin:",
        actions=actions,
    ).send()


@cl.action_callback("switch_mode")
async def on_mode_switch(action: cl.Action):
    selected = action.payload.get("value")

    if selected == "customer":
        cl.user_session.set("mode", "customer")
        cl.user_session.set("owner_authenticated", False)
        await cl.Message(content="Switched to **Customer Mode**. How can I help you today?").send()
        return

    res = await cl.AskUserMessage(
        content="Enter the owner password to access Shop Owner Mode:",
        timeout=30,
    ).send()

    if res and res.get("output", "").strip() == OWNER_PASSWORD:
        cl.user_session.set("mode", "owner")
        cl.user_session.set("owner_authenticated", True)
        await cl.Message(
            content="Access granted. Welcome, Shop Owner!\n"
                    "Ask me about diagnostics, reviews, violations, or product performance."
        ).send()
    else:
        await cl.Message(content="Incorrect password. Remaining in Customer Mode.").send()


@cl.on_message
async def on_message(message: cl.Message):
    mode = cl.user_session.get("mode", "customer")

    if mode == "owner" and not cl.user_session.get("owner_authenticated", False):
        mode = "customer"
        cl.user_session.set("mode", "customer")

    try:
        full_context  = build_context(message.content, mode)
        chat_messages = _build_chat_messages(message.content, mode, full_context)
    except Exception as e:  # noqa: BLE001
        await cl.Message(content=f"Sorry — I couldn't build a response: {e}").send()
        return

    response_msg = cl.Message(content="")
    try:
        for token in Settings.llm.stream_chat(chat_messages):
            await response_msg.stream_token(token.delta)
        await response_msg.send()
    except Exception as e:  # noqa: BLE001
        if not response_msg.content:
            await cl.Message(content=f"The model call failed: {e}").send()
        else:
            await response_msg.stream_token(f"\n\nResponse cut short: {e}")
            await response_msg.send()