"""
Embedding retrieval evaluation.

This script evaluates whether the vector retriever returns the expected
document IDs for a small manually-labelled embedding test set.

Metrics:
  - hit@k: at least one expected doc_id appears in top-k
  - recall@k: retrieved expected doc_ids / total expected doc_ids
  - precision@k: retrieved expected doc_ids / k
  - mrr@k: reciprocal rank of the first expected doc_id
  - ndcg@k: ranking quality with binary relevance
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

os.chdir(PROJECT_DIR)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

load_dotenv(PROJECT_DIR / ".env")

EMBEDDING_MODEL_LABEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "google/gemini-embedding-001",
)


def load_test_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not samples:
        sys.exit(f"[Embedding Eval] Test set is empty: {path}")

    valid = []
    for idx, sample in enumerate(samples, 1):
        expected = sample.get("expected_doc_ids") or []
        if not expected:
            print(
                f"[Embedding Eval] Skipping sample {idx}: "
                "expected_doc_ids is empty."
            )
            continue
        valid.append(sample)

    if not valid:
        sys.exit("[Embedding Eval] No samples with expected_doc_ids. Aborting.")
    return valid


def node_metadata(node) -> dict:
    return dict(getattr(node, "metadata", {}) or {})


def node_text(node) -> str:
    return str(getattr(node, "text", "") or "")


def node_doc_id(node) -> str:
    metadata = node_metadata(node)
    return str(metadata.get("doc_id") or getattr(node, "node_id", ""))


def build_retrievers(top_k: int):
    import rag_hybrid

    public_retriever = rag_hybrid.PUBLIC_INDEX.as_retriever(similarity_top_k=top_k)
    private_retriever = rag_hybrid.PRIVATE_INDEX.as_retriever(similarity_top_k=top_k)
    return public_retriever, private_retriever


def expected_scopes(sample: dict) -> set[str]:
    expected_doc_ids = [str(x) for x in sample.get("expected_doc_ids", [])]
    scopes = set()
    if any(doc_id.startswith("public:") for doc_id in expected_doc_ids):
        scopes.add("public")
    if any(doc_id.startswith("private:") for doc_id in expected_doc_ids):
        scopes.add("private")
    if not scopes:
        scopes.add("public")
        if sample.get("mode") == "owner":
            scopes.add("private")
    return scopes


def retrieve_nodes(sample: dict, top_k: int) -> tuple[list, str]:
    query = sample["question"]
    public_retriever, private_retriever = build_retrievers(top_k)
    scopes = expected_scopes(sample)
    nodes = []
    source_labels = []

    if "public" in scopes:
        nodes.extend(list(public_retriever.retrieve(query)))
        source_labels.append("public")

    if "private" in scopes:
        seen = {node_doc_id(n) or getattr(n, "node_id", "") for n in nodes}
        for node in private_retriever.retrieve(query):
            key = node_doc_id(node) or getattr(node, "node_id", "")
            if key not in seen:
                nodes.append(node)
                seen.add(key)
        source_labels.append("private")

    return nodes[:top_k], "+".join(source_labels)


def dcg(binary_relevance: list[int]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(binary_relevance))


def score_sample(sample: dict, nodes: list, top_k: int) -> dict:
    expected_doc_ids = [str(x) for x in sample.get("expected_doc_ids", [])]
    expected_item_ids = {str(x) for x in sample.get("expected_item_ids", [])}
    expected_doc_types = {str(x) for x in sample.get("expected_doc_types", [])}
    expected_set = set(expected_doc_ids)

    retrieved_doc_ids = [node_doc_id(n) for n in nodes]
    retrieved_metadatas = [node_metadata(n) for n in nodes]
    retrieved_doc_types = [m.get("type", "") for m in retrieved_metadatas]
    retrieved_item_ids = [str(m.get("item_id", "")) for m in retrieved_metadatas]

    binary_rel = [1 if doc_id in expected_set else 0 for doc_id in retrieved_doc_ids]
    hits = sum(binary_rel)
    first_hit_rank = next(
        (idx + 1 for idx, rel in enumerate(binary_rel) if rel),
        None,
    )

    ideal_hits = min(len(expected_set), top_k)
    ideal_rel = [1] * ideal_hits + [0] * (top_k - ideal_hits)

    item_hits = sum(1 for item_id in retrieved_item_ids if item_id in expected_item_ids)
    type_hits = sum(1 for doc_type in retrieved_doc_types if doc_type in expected_doc_types)

    return {
        "sample_id": sample.get("sample_id", ""),
        "question": sample.get("question", ""),
        "mode": sample.get("mode", "customer"),
        "category": sample.get("category", "uncategorised"),
        "expected_doc_ids": json.dumps(expected_doc_ids, ensure_ascii=False),
        "retrieved_doc_ids": json.dumps(retrieved_doc_ids, ensure_ascii=False),
        "retrieved_doc_types": json.dumps(retrieved_doc_types, ensure_ascii=False),
        "retrieved_item_ids": json.dumps(retrieved_item_ids, ensure_ascii=False),
        "hit_at_k": 1.0 if hits > 0 else 0.0,
        "recall_at_k": hits / len(expected_set) if expected_set else 0.0,
        "precision_at_k": hits / top_k if top_k else 0.0,
        "mrr_at_k": (1.0 / first_hit_rank) if first_hit_rank else 0.0,
        "ndcg_at_k": dcg(binary_rel) / dcg(ideal_rel) if ideal_hits else 0.0,
        "item_hit_at_k": 1.0 if item_hits > 0 else 0.0,
        "doc_type_hit_at_k": 1.0 if type_hits > 0 else 0.0,
        "top_1_doc_id": retrieved_doc_ids[0] if retrieved_doc_ids else "",
        "top_1_type": retrieved_doc_types[0] if retrieved_doc_types else "",
        "top_1_item_id": retrieved_item_ids[0] if retrieved_item_ids else "",
    }


def print_summary(df: pd.DataFrame, top_k: int) -> None:
    metrics = [
        "hit_at_k",
        "recall_at_k",
        "precision_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "item_hit_at_k",
        "doc_type_hit_at_k",
    ]

    print("\n" + "=" * 78)
    print(f"  EMBEDDING RETRIEVAL SUMMARY @ {top_k}")
    print("=" * 78)
    print(f"  Embedding model: {EMBEDDING_MODEL_LABEL}")
    print(f"  Samples scored : {len(df)}")
    print("  " + "-" * 74)
    for metric in metrics:
        print(f"  {metric:<24} {df[metric].mean():.4f}")

    print("\n  Per-category:")
    for category in sorted(df["category"].unique()):
        subset = df[df["category"] == category]
        print(
            f"  {category:<28} n={len(subset):<3} "
            f"hit={subset['hit_at_k'].mean():.3f} "
            f"recall={subset['recall_at_k'].mean():.3f} "
            f"mrr={subset['mrr_at_k'].mean():.3f}"
        )
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate embedding retrieval using expected_doc_ids."
    )
    parser.add_argument(
        "--test-set",
        default=str(SCRIPT_DIR / "embedding.json"),
        help="Path to embedding ground truth JSON.",
    )
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.top_k < 1:
        sys.exit("[Embedding Eval] --top-k must be >= 1.")
    if not Path(args.test_set).exists():
        sys.exit(f"[Embedding Eval] Test set not found: {args.test_set}")

    samples = load_test_set(args.test_set)
    rows = []
    start = time.monotonic()

    print(f"[Embedding Eval] Loaded {len(samples)} sample(s).")
    print(f"[Embedding Eval] top_k={args.top_k}")

    for idx, sample in enumerate(samples, 1):
        question = sample["question"]
        mode = sample.get("mode", "customer")
        scopes = "+".join(sorted(expected_scopes(sample)))
        print(f"  [{idx}/{len(samples)}] {mode:<8} scope={scopes:<14} | {question[:80]}...")
        q_start = time.monotonic()
        nodes, retrieval_scope = retrieve_nodes(sample, args.top_k)
        row = score_sample(sample, nodes, args.top_k)
        row["latency_seconds"] = round(time.monotonic() - q_start, 3)
        row["embedding_model"] = EMBEDDING_MODEL_LABEL
        row["retrieval_scope"] = retrieval_scope
        rows.append(row)

    df = pd.DataFrame(rows)
    print_summary(df, args.top_k)

    output_path = args.output or (
        PROJECT_DIR
        / f"embedding_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    json_path = str(output_path).replace(".csv", "_detailed.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    elapsed = time.monotonic() - start
    print(f"[Embedding Eval] Results saved -> {output_path}")
    print(f"[Embedding Eval] Detailed report -> {json_path}")
    print(f"[Embedding Eval] Done in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
