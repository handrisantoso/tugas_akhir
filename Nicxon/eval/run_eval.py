#!/usr/bin/env python3
"""
RAG evaluation runner.

Two main modes:

1. EVAL — runs a queries CSV against a Chroma DB folder, computes retrieval
   metrics per-query and in aggregate, and (optionally) calls a multimodal
   LLM judge for response-quality grading. Outputs CSV + JSON + TXT to
   `eval/results/<label>/`.

2. COMPARE — reads existing aggregate.json files from previous EVAL runs and
   prints a side-by-side table.

Usage:

  python eval\\run_eval.py --dataset eval\\sample_dataset.csv \\
      --db-path vector_db\\chroma_db_multimodal_2 --label google

  python eval\\run_eval.py --dataset my_eval.csv \\
      --db-path vector_db\\chroma_db_openrouter --label openrouter --judge

  python eval\\run_eval.py --compare openrouter google
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure local-folder imports resolve regardless of CWD.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Project root for resolving env file + DB paths.
PROJECT_ROOT = HERE.parent

from dataset import EvalQuery, load_dataset
from retrieval_metrics import aggregate_per_query
from runner import (
    EMBEDDING_MODEL_DEFAULT,
    metrics_for_query,
    run_query,
)


# ----------------------------------------------------------------------
# Env / API key
# ----------------------------------------------------------------------

ENV_CANDIDATES = [
    HERE / ".env",
    PROJECT_ROOT / "RAG" / ".env",
    PROJECT_ROOT / "embedding" / ".env",
    PROJECT_ROOT / ".env",
]


def _load_env() -> str:
    """Load GOOGLE_API_KEY from any nearby .env, else env. Returns key or empty string."""
    try:
        from dotenv import load_dotenv  # type: ignore
        for p in ENV_CANDIDATES:
            if p.exists():
                load_dotenv(p, override=False)
    except ImportError:
        for p in ENV_CANDIDATES:
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    return os.environ.get("GOOGLE_API_KEY", "")


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------

def _write_per_query(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    # Stable column order: dataset fields first, then metrics, then latencies.
    preferred_order = [
        "query_id", "query_text", "image_path", "expected_book_ids",
        "search_mode", "primary_book_ids", "primary_similarities",
        "first_hit_rank",
        "hit@1", "hit@5", "hit@10",
        "recall@1", "recall@5", "recall@10",
        "mrr", "ndcg@10",
        "embedding_latency_ms", "query_latency_ms", "latency_ms",
        "judge_relevance", "judge_faithfulness", "judge_helpfulness",
        "judge_mean", "judge_reason", "judge_error",
        "error",
    ]
    columns: List[str] = []
    for k in preferred_order:
        if any(k in row for row in rows):
            columns.append(k)
    # Append any extra keys we didn't anticipate.
    for row in rows:
        for k in row.keys():
            if k not in columns:
                columns.append(k)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_aggregate(metrics: Dict[str, Any],
                     extra: Dict[str, Any],
                     out_dir: Path) -> None:
    payload = {**metrics, **extra}
    (out_dir / "aggregate.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines: List[str] = []
    lines.append(f"Eval label: {extra.get('label')}")
    lines.append(f"DB path:    {extra.get('db_path')}")
    lines.append(f"Model:      {extra.get('embedding_model')}")
    lines.append(f"Dataset:    {extra.get('dataset_path')}")
    lines.append(f"Queries:    {metrics.get('n_queries', 0)}")
    lines.append("")
    lines.append("RETRIEVAL METRICS")
    for k in ("mean_recall@1", "mean_recall@5", "mean_recall@10",
              "mrr", "hit_rate@1", "hit_rate@5", "hit_rate@10",
              "ndcg@10", "mean_latency_ms"):
        if k in metrics:
            lines.append(f"  {k:<20} {metrics[k]:.4f}")
    if "judge" in payload:
        lines.append("")
        lines.append("RESPONSE QUALITY (LLM-as-judge)")
        for k, v in payload["judge"].items():
            if v is None:
                lines.append(f"  {k:<20} (no scored queries)")
            else:
                lines.append(f"  {k:<20} {v:.3f}")

    (out_dir / "aggregate.txt").write_text("\n".join(lines), encoding="utf-8")
    print()
    print("\n".join(lines))


# ----------------------------------------------------------------------
# EVAL command
# ----------------------------------------------------------------------

async def _maybe_judge(args, query: EvalQuery, primary_book_ids: List[str]) -> Optional[Dict[str, Any]]:
    """If --judge was passed, run the full RAG engine for this query, judge it,
    and return a dict of judge fields. Otherwise return None.

    The judge step is intentionally heavy: it calls the live engine to generate
    the response (so what gets graded matches what users see). Skip it for
    pure retrieval-only comparisons.
    """
    if not args.judge:
        return None

    # Lazily import the engine so the retrieval-only path doesn't pay for it.
    sys.path.insert(0, str(PROJECT_ROOT / "RAG"))
    from rag_engine import LibraryRAGEngine  # type: ignore

    # Read image bytes if any so the engine and the judge both see it.
    image_bytes: Optional[bytes] = None
    image_jpeg: Optional[bytes] = None
    if query.has_image and query.resolved_image_path:
        image_bytes = query.resolved_image_path.read_bytes()

    # Engine instance — single per process is fine for eval.
    if not hasattr(_maybe_judge, "_engine"):
        _maybe_judge._engine = LibraryRAGEngine()  # type: ignore[attr-defined]
    engine = _maybe_judge._engine  # type: ignore[attr-defined]

    response_data = await engine.process_query(
        query.query_text or "",
        image_bytes=image_bytes,
        mime_type="image/jpeg" if image_bytes else None,
    )
    generated = response_data.get("response", "")
    retrieved_books = response_data.get("search_results", [])

    # Try to also recover the resized JPEG the engine produced, for the judge.
    # The engine stores it in conversation memory.
    cm = getattr(engine, "conversation_manager", None)
    if cm is not None and getattr(cm, "last_image_jpeg", None):
        image_jpeg = cm.last_image_jpeg

    # Judge — uses the same google.genai client.
    from google import genai
    from llm_judge import call_judge

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    score = call_judge(
        client=client,
        model_id=args.judge_model,
        query_text=query.query_text,
        retrieved_books=retrieved_books,
        generated_response=generated,
        image_jpeg=image_jpeg,
    )
    return score.as_dict()


def _aggregate_judge(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Optional[float]]]:
    keys = ("judge_relevance", "judge_faithfulness", "judge_helpfulness", "judge_mean")
    if not any(any(k in r for k in keys) for r in rows):
        return None
    sums: Dict[str, float] = {k: 0.0 for k in keys}
    counts: Dict[str, int] = {k: 0 for k in keys}
    for r in rows:
        for k in keys:
            v = r.get(k)
            if v is None or v == "":
                continue
            try:
                sums[k] += float(v)
                counts[k] += 1
            except (TypeError, ValueError):
                continue
    return {
        "mean_" + k.replace("judge_", ""): (sums[k] / counts[k]) if counts[k] else None
        for k in keys
    }


async def cmd_eval(args) -> int:
    from runner import _is_gemini, QWEN_MODEL_NAME

    # Resolve model id shorthand
    model_aliases = {
        "qwen": QWEN_MODEL_NAME,
        "gemini": EMBEDDING_MODEL_DEFAULT,
    }
    model_id = model_aliases.get(args.embedding_model, args.embedding_model)
    use_gemini = _is_gemini(model_id)

    client = None
    if use_gemini:
        api_key = _load_env()
        if not api_key:
            print("ERROR: GOOGLE_API_KEY required for Gemini models.", file=sys.stderr)
            return 1
    else:
        # Local model — try to load API key anyway (non-fatal, judge may need it later)
        try:
            _load_env()
        except RuntimeError:
            pass  # fine for retrieval-only local runs

    db_path = Path(args.db_path).resolve()
    if not db_path.exists():
        print(f"ERROR: Chroma DB not found at {db_path}", file=sys.stderr)
        return 1

    dataset_path = Path(args.dataset).resolve()
    queries = load_dataset(dataset_path, project_root=PROJECT_ROOT)
    print(f"Loaded {len(queries):,} queries from {dataset_path}")
    print(f"Model:   {model_id}  ({'Gemini API' if use_gemini else 'local'})")

    # Lazy-import heavy deps after we know the dataset is loadable.
    try:
        import chromadb
    except ImportError as e:
        print(f"ERROR: missing dependency ({e}). pip install chromadb")
        return 1

    if use_gemini:
        try:
            from google import genai
        except ImportError as e:
            print(f"ERROR: missing dependency ({e}). pip install google-genai")
            return 1
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        client = genai.Client(api_key=api_key)

    chroma = chromadb.PersistentClient(path=str(db_path))
    text_col = chroma.get_collection(name=args.text_collection)
    image_col = chroma.get_collection(name=args.image_collection)
    print(f"Connected — text:{text_col.count():,} image:{image_col.count():,}")

    rows: List[Dict[str, Any]] = []
    n = len(queries)
    for i, q in enumerate(queries, start=1):
        try:
            result = run_query(
                client=client,
                model_id=model_id,
                text_collection=text_col,
                image_collection=image_col,
                query=q,
            )
            row = metrics_for_query(q, result)
        except Exception as e:
            row = {
                "query_id": q.query_id,
                "query_text": (q.query_text or "")[:200],
                "image_path": q.image_path or "",
                "expected_book_ids": "|".join(q.expected_book_ids),
                "error": f"runner: {e}",
            }
            print(f"  [{i}/{n}] {q.query_id}: ERROR {e}")
        else:
            judge_dict = await _maybe_judge(args, q, result.primary_book_ids)
            if judge_dict:
                row.update(judge_dict)
            mark = "OK " if not row.get("error") else "ERR"
            print(f"  [{i}/{n}] {q.query_id}: {mark} mode={row.get('search_mode')} "
                  f"hit@10={row.get('hit@10')} mrr={row.get('mrr', 0):.3f}")
        rows.append(row)

    # ---- Aggregate ----
    out_dir = HERE / "results" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_per_query(rows, out_dir / "per_query.csv")

    metrics = aggregate_per_query(rows)
    extra: Dict[str, Any] = {
        "label": args.label,
        "db_path": str(db_path),
        "embedding_model": model_id,
        "dataset_path": str(dataset_path),
        "judge_used": bool(args.judge),
        "judge_model": args.judge_model if args.judge else None,
    }
    judge_summary = _aggregate_judge(rows)
    if judge_summary is not None:
        extra["judge"] = judge_summary

    _write_aggregate(metrics, extra, out_dir)
    print(f"\nResults written to: {out_dir}")
    return 0


# ----------------------------------------------------------------------
# COMPARE command
# ----------------------------------------------------------------------

def cmd_compare(args) -> int:
    payloads: Dict[str, Dict[str, Any]] = {}
    for label in args.compare:
        path = HERE / "results" / label / "aggregate.json"
        if not path.exists():
            print(f"WARNING: no aggregate at {path} — skipping {label}")
            continue
        payloads[label] = json.loads(path.read_text(encoding="utf-8"))

    if not payloads:
        print("Nothing to compare.")
        return 1

    rows = [
        ("queries",        "n_queries"),
        ("Recall@1",       "mean_recall@1"),
        ("Recall@5",       "mean_recall@5"),
        ("Recall@10",      "mean_recall@10"),
        ("MRR",            "mrr"),
        ("HitRate@10",     "hit_rate@10"),
        ("NDCG@10",        "ndcg@10"),
        ("latency_ms",     "mean_latency_ms"),
    ]

    labels = list(payloads.keys())
    col_w = max(12, max(len(lbl) for lbl in labels) + 2)

    header = f"{'metric':<14}" + "".join(f"{lbl:>{col_w}}" for lbl in labels)
    print(header)
    print("-" * len(header))
    for label_text, key in rows:
        cells = []
        for lbl in labels:
            v = payloads[lbl].get(key)
            if v is None:
                cells.append(f"{'-':>{col_w}}")
            elif key == "n_queries":
                cells.append(f"{int(v):>{col_w}d}")
            else:
                cells.append(f"{v:>{col_w}.4f}")
        print(f"{label_text:<14}" + "".join(cells))

    # Judge summary if any payload has one.
    if any("judge" in p for p in payloads.values()):
        print()
        print("Judge (mean 1-5)")
        for label_text, key in [
            ("relevance",    "mean_relevance"),
            ("faithfulness", "mean_faithfulness"),
            ("helpfulness",  "mean_helpfulness"),
            ("overall",      "mean_mean"),
        ]:
            cells = []
            for lbl in labels:
                v = (payloads[lbl].get("judge") or {}).get(key)
                cells.append(
                    f"{'-':>{col_w}}" if v is None else f"{v:>{col_w}.3f}"
                )
            print(f"{label_text:<14}" + "".join(cells))

    return 0


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate the library RAG retrieval (and optionally responses)."
    )
    p.add_argument("--dataset", help="Path to queries CSV.")
    p.add_argument("--db-path",
                   help="Path to a ChromaDB folder.")
    p.add_argument("--label", default="default",
                   help="Output label; results land in eval/results/<label>/.")
    p.add_argument("--embedding-model", default=EMBEDDING_MODEL_DEFAULT,
                   help=f"Embedding model id (default: {EMBEDDING_MODEL_DEFAULT}).")
    p.add_argument("--text-collection", default="library_books_text")
    p.add_argument("--image-collection", default="library_books_image")
    p.add_argument("--judge", action="store_true",
                   help="Also run multimodal LLM-as-judge for response quality.")
    p.add_argument("--judge-model", default="gemini-2.5-flash",
                   help="Judge LLM model id (default: gemini-2.5-flash).")
    p.add_argument("--compare", nargs="+",
                   help="Compare existing run labels side-by-side, instead of running an eval.")
    args = p.parse_args()

    if args.compare:
        return cmd_compare(args)

    if not args.dataset or not args.db_path:
        p.error("--dataset and --db-path are required for an eval run "
                "(or use --compare to compare existing runs).")

    return asyncio.run(cmd_eval(args))


if __name__ == "__main__":
    sys.exit(main())
