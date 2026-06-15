"""
Pure-Python retrieval metrics. No dependency on numpy or any framework.

All functions take:
  retrieved: List[str]          — book_ids returned by the system, ordered by rank (rank 1 first)
  expected:  Iterable[str]      — book_ids the system was supposed to return

A retrieved id is "correct" iff it appears in expected. The metrics here are
designed for the common "any expected id at top-K is good" semantics that fits
the library RAG use-case (one or a small number of correct answers per query).
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional


def _rank_of_first_hit(retrieved: List[str], expected_set: set[str]) -> Optional[int]:
    for i, bid in enumerate(retrieved, start=1):
        if bid in expected_set:
            return i
    return None


def hit_rate_at_k(retrieved: List[str], expected: Iterable[str], k: int) -> int:
    """1 if any expected id is in retrieved[:k] else 0."""
    expected_set = set(expected)
    return int(any(bid in expected_set for bid in retrieved[:k]))


def recall_at_k(retrieved: List[str], expected: Iterable[str], k: int) -> float:
    """|hits ∩ expected| / |expected| over the top-K. Falls back to 0 when expected is empty."""
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    hits = sum(1 for bid in retrieved[:k] if bid in expected_set)
    return hits / len(expected_set)


def reciprocal_rank(retrieved: List[str], expected: Iterable[str]) -> float:
    """1/rank of the first correct hit, 0 if none found."""
    rank = _rank_of_first_hit(retrieved, set(expected))
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(retrieved: List[str], expected: Iterable[str], k: int) -> float:
    """Binary-relevance NDCG@K.

    With multiple expected ids treated as equally relevant (gain = 1 per hit)
    and IDCG normalised against placing min(|expected|, K) hits at the top.
    """
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    dcg = 0.0
    for i, bid in enumerate(retrieved[:k], start=1):
        if bid in expected_set:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def aggregate_per_query(per_query_results: List[dict]) -> dict:
    """Aggregate the per-query metrics into a summary dict.

    `per_query_results` is a list of dicts; each must contain at least
    `recall@1`, `recall@5`, `recall@10`, `mrr`, `hit@10`, `ndcg@10`,
    `latency_ms`. Any missing key is treated as 0 for that query.
    """
    if not per_query_results:
        return {"n_queries": 0}

    keys = ("recall@1", "recall@5", "recall@10", "mrr",
            "hit@1", "hit@5", "hit@10", "ndcg@10", "latency_ms")
    sums = {k: 0.0 for k in keys}
    for row in per_query_results:
        for k in keys:
            sums[k] += float(row.get(k, 0.0) or 0.0)

    n = len(per_query_results)
    return {
        "n_queries": n,
        "mean_recall@1": sums["recall@1"] / n,
        "mean_recall@5": sums["recall@5"] / n,
        "mean_recall@10": sums["recall@10"] / n,
        "mrr": sums["mrr"] / n,
        "hit_rate@1": sums["hit@1"] / n,
        "hit_rate@5": sums["hit@5"] / n,
        "hit_rate@10": sums["hit@10"] / n,
        "ndcg@10": sums["ndcg@10"] / n,
        "mean_latency_ms": sums["latency_ms"] / n,
    }
