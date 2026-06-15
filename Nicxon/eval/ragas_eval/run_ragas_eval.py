#!/usr/bin/env python3
"""
Optional RAGAS-based evaluation for the library RAG system.

This script is fully separate from the main `eval/run_eval.py` harness. It is
not imported anywhere else; running it requires the packages listed in
`eval/ragas_eval/requirements.txt` (RAGAS + datasets + langchain-google-genai),
which are NOT installed by default.

Usage:
    python eval\\ragas_eval\\run_ragas_eval.py --check
    python eval\\ragas_eval\\run_ragas_eval.py --dataset eval\\my_eval.csv --db-path vector_db\\chroma_db_multimodal_2 --label google_ragas
    python eval\\ragas_eval\\run_ragas_eval.py --dataset eval\\my_eval.csv --db-path ... --label ... --with-references

Notes
-----
- Reuses the main harness's `dataset.py` loader (so the same CSV drives both
  pipelines) and `runner.py` for the actual retrieval step.
- Skips image-only queries — RAGAS does not support multimodal input. HYBRID
  queries are evaluated using only their text portion. The script logs which
  queries were skipped at the start of each run.
- Calls the live RAG engine to generate the responses that RAGAS will rate, so
  what gets graded is exactly what users would see.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Local imports from the main eval folder.
HERE = Path(__file__).resolve().parent          # eval/ragas_eval/
EVAL_ROOT = HERE.parent                          # eval/
PROJECT_ROOT = EVAL_ROOT.parent                  # repo root
sys.path.insert(0, str(EVAL_ROOT))               # for dataset.py, runner.py

from dataset import EvalQuery, load_dataset      # noqa: E402

# ---------------------------------------------------------------------------
# Defaults — kept conservative on cost
# ---------------------------------------------------------------------------

JUDGE_MODEL_DEFAULT = "gemini-2.5-flash"
EMBEDDING_MODEL_DEFAULT = "text-embedding-004"   # RAGAS uses an embedder for answer_relevancy

# Same constants as RAG/config.py — only used by the retrieval step here.
GENERATION_EMBEDDING_MODEL = "gemini-embedding-2"
TEXT_COLLECTION = "library_books_text"
IMAGE_COLLECTION = "library_books_image"

ENV_CANDIDATES = [
    HERE / ".env",
    EVAL_ROOT / ".env",
    PROJECT_ROOT / "RAG" / ".env",
    PROJECT_ROOT / "embedding" / ".env",
    PROJECT_ROOT / ".env",
]


# ---------------------------------------------------------------------------
# Env loading
# ---------------------------------------------------------------------------

def _load_env() -> str:
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

    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Put it in eval/.env or RAG/.env."
        )
    return key


# ---------------------------------------------------------------------------
# Lazy imports for the optional packages
# ---------------------------------------------------------------------------

def _check_optional_imports() -> Dict[str, str]:
    """Return {package: version_or_error} for each required package."""
    statuses: Dict[str, str] = {}

    def _try(name: str, import_path: str) -> str:
        try:
            mod = __import__(import_path, fromlist=["__version__"])
            return getattr(mod, "__version__", "(version unknown)")
        except ImportError as e:
            return f"NOT INSTALLED ({e})"

    statuses["ragas"] = _try("ragas", "ragas")
    statuses["datasets"] = _try("datasets", "datasets")
    statuses["langchain_google_genai"] = _try(
        "langchain-google-genai", "langchain_google_genai"
    )
    return statuses


def _missing_packages(statuses: Dict[str, str]) -> List[str]:
    return [k for k, v in statuses.items() if v.startswith("NOT INSTALLED")]


# ---------------------------------------------------------------------------
# Building the RAGAS LLM + embedding wrappers
# ---------------------------------------------------------------------------

def _build_ragas_llms(judge_model: str, embedding_model: str, api_key: str):
    """Wrap Google GenAI for RAGAS via LangChain adapters.

    RAGAS's `evaluate()` accepts plain LangChain LLMs/Embeddings, so we wrap
    Gemini through `langchain-google-genai`. Returns a tuple
    `(judge_llm_wrapper, embeddings_wrapper)` where the wrappers are the
    RAGAS-side adapters that point at LangChain primitives.
    """
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    chat = ChatGoogleGenerativeAI(
        model=judge_model,
        google_api_key=api_key,
        temperature=0.0,  # determinism over creativity for grading
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model=f"models/{embedding_model}" if not embedding_model.startswith("models/") else embedding_model,
        google_api_key=api_key,
    )
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embeddings)


def _check_models(judge_model: str, embedding_model: str, api_key: str) -> None:
    """Verify both models actually respond. Used by --check."""
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    chat = ChatGoogleGenerativeAI(
        model=judge_model, google_api_key=api_key, temperature=0.0
    )
    msg = chat.invoke("Reply with the single word OK.")
    print(f"  judge {judge_model}: ok ({(msg.content or '').strip()[:40]!r})")

    emb = GoogleGenerativeAIEmbeddings(
        model=f"models/{embedding_model}" if not embedding_model.startswith("models/") else embedding_model,
        google_api_key=api_key,
    )
    vec = emb.embed_query("smoke test")
    print(f"  embedding {embedding_model}: ok (dim={len(vec)})")


# ---------------------------------------------------------------------------
# Live engine — generate responses for RAGAS to grade
# ---------------------------------------------------------------------------

async def _generate_responses(queries: List[EvalQuery]) -> List[Dict[str, Any]]:
    """Run each query through the live engine and capture the inputs RAGAS needs.

    Image-only queries (no text) are skipped with a warning. HYBRID queries
    keep their text and image goes through the engine, but only the text
    portion makes it into the RAGAS dataset (RAGAS doesn't see the image).
    """
    sys.path.insert(0, str(PROJECT_ROOT / "RAG"))
    from rag_engine import LibraryRAGEngine  # type: ignore

    engine = LibraryRAGEngine()
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for i, q in enumerate(queries, start=1):
        if not q.has_text:
            skipped.append(f"{q.query_id} (image-only — RAGAS needs text)")
            continue

        image_bytes = (
            q.resolved_image_path.read_bytes()
            if q.has_image and q.resolved_image_path
            else None
        )
        try:
            resp = await engine.process_query(
                q.query_text,
                image_bytes=image_bytes,
                mime_type="image/jpeg" if image_bytes else None,
            )
        except Exception as e:
            print(f"  [{i}/{len(queries)}] {q.query_id}: ENGINE ERROR {e}")
            continue

        # RAGAS wants `retrieved_contexts` as a list of strings — the document
        # text per retrieved book. Strip metadata; RAGAS only needs the prose.
        contexts: List[str] = []
        for sr in resp.get("search_results", []):
            doc = sr.get("document") or ""
            if doc:
                contexts.append(doc[:2000])  # bound prompt size

        rows.append({
            "query_id": q.query_id,
            "user_input": q.query_text,
            "response": resp.get("response", ""),
            "retrieved_contexts": contexts,
            "reference": "",  # filled in later if --with-references
        })
        print(f"  [{i}/{len(queries)}] {q.query_id}: response generated "
              f"(contexts={len(contexts)})")

    if skipped:
        print(f"\nSkipped {len(skipped)} queries (RAGAS does not support multimodal input):")
        for s in skipped:
            print(f"  - {s}")

    return rows


def _read_references_from_csv(csv_path: Path) -> Dict[str, str]:
    """Return {query_id: reference} for rows that have a non-empty reference cell."""
    import csv as _csv
    out: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = _csv.DictReader(f)
        if "reference" not in (reader.fieldnames or []):
            return out
        for row in reader:
            qid = (row.get("query_id") or "").strip()
            ref = (row.get("reference") or "").strip()
            if qid and ref:
                out[qid] = ref
    return out


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def _run_ragas(rows: List[Dict[str, Any]],
                judge_model: str,
                embedding_model: str,
                with_references: bool,
                api_key: str) -> Dict[str, Any]:
    """Call ragas.evaluate and return aggregate + per-query scores."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
    )

    metrics = [faithfulness, answer_relevancy]
    if with_references:
        from ragas.metrics import context_precision, context_recall
        metrics += [context_precision, context_recall]

    judge_wrapped, embed_wrapped = _build_ragas_llms(
        judge_model=judge_model,
        embedding_model=embedding_model,
        api_key=api_key,
    )

    # Build the HF dataset RAGAS expects
    ds_dict: Dict[str, List[Any]] = {
        "user_input": [r["user_input"] for r in rows],
        "response": [r["response"] for r in rows],
        "retrieved_contexts": [r["retrieved_contexts"] for r in rows],
    }
    if with_references:
        ds_dict["reference"] = [r.get("reference", "") for r in rows]

    ds = Dataset.from_dict(ds_dict)

    print(f"\nRunning RAGAS evaluate over {len(rows)} queries with "
          f"{len(metrics)} metrics — this may take several minutes...")
    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=judge_wrapped,
        embeddings=embed_wrapped,
    )

    # `result` is a RAGAS `EvaluationResult`; convert to per-query rows + means
    df = result.to_pandas()
    per_query: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        row_out = {
            "query_id": r["query_id"],
            "user_input": r["user_input"][:200],
            "n_contexts": len(r["retrieved_contexts"]),
        }
        for m in metrics:
            col = getattr(m, "name", str(m))
            if col in df.columns:
                row_out[col] = float(df.iloc[i][col]) if not _is_nan(df.iloc[i][col]) else None
        per_query.append(row_out)

    aggregate: Dict[str, Any] = {"n_queries": len(rows)}
    for m in metrics:
        col = getattr(m, "name", str(m))
        if col in df.columns:
            vals = [v for v in df[col].tolist() if not _is_nan(v)]
            aggregate[f"mean_{col}"] = (sum(vals) / len(vals)) if vals else None
            aggregate[f"n_scored_{col}"] = len(vals)

    return {"per_query": per_query, "aggregate": aggregate}


def _is_nan(v: Any) -> bool:
    try:
        return v != v  # standard NaN check, dependency-free
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_outputs(label: str,
                    per_query: List[Dict[str, Any]],
                    aggregate: Dict[str, Any],
                    extra: Dict[str, Any]) -> Path:
    out_dir = HERE / "results" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # per_query.csv
    import csv as _csv
    if per_query:
        cols: List[str] = []
        for row in per_query:
            for k in row:
                if k not in cols:
                    cols.append(k)
        with (out_dir / "per_query.csv").open("w", encoding="utf-8", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for row in per_query:
                writer.writerow(row)

    # aggregate.json
    payload = {**aggregate, **extra}
    (out_dir / "aggregate.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    # aggregate.txt
    lines = [
        f"RAGAS evaluation",
        f"label:     {label}",
        f"db_path:   {extra.get('db_path')}",
        f"judge:     {extra.get('judge_model')}",
        f"embedder:  {extra.get('embedding_model')}",
        f"queries:   {aggregate.get('n_queries', 0)}",
        f"with refs: {extra.get('with_references', False)}",
        "",
        "Mean scores:",
    ]
    for k, v in aggregate.items():
        if k.startswith("mean_") and isinstance(v, (int, float)):
            lines.append(f"  {k:<30} {v:.4f}")
        elif k.startswith("mean_"):
            lines.append(f"  {k:<30} (no scored queries)")
    (out_dir / "aggregate.txt").write_text("\n".join(lines), encoding="utf-8")
    print()
    print("\n".join(lines))
    return out_dir


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_check(args) -> int:
    print("Checking optional packages...")
    statuses = _check_optional_imports()
    for k, v in statuses.items():
        print(f"  {k}: {v}")
    missing = _missing_packages(statuses)
    if missing:
        print(f"\nInstall missing packages with:\n"
              f"    pip install -r eval\\ragas_eval\\requirements.txt")
        return 1

    print("\nLoading API key...")
    api_key = _load_env()
    print("  GOOGLE_API_KEY: ok")

    print(f"\nProbing models (judge={args.judge_model}, embed={args.embedding_model})...")
    try:
        _check_models(args.judge_model, args.embedding_model, api_key)
    except Exception as e:
        print(f"  model probe failed: {e}")
        return 1

    print("\nAll checks passed. Ready to run a real eval.")
    return 0


def cmd_eval(args) -> int:
    statuses = _check_optional_imports()
    missing = _missing_packages(statuses)
    if missing:
        print("ERROR: missing required packages:", ", ".join(missing))
        print("Install with:  pip install -r eval\\ragas_eval\\requirements.txt")
        return 1

    api_key = _load_env()

    dataset_path = Path(args.dataset).resolve()
    queries = load_dataset(dataset_path, project_root=PROJECT_ROOT)
    print(f"Loaded {len(queries):,} queries from {dataset_path}")

    if args.with_references:
        refs = _read_references_from_csv(dataset_path)
        if not refs:
            print("ERROR: --with-references was set but the dataset has no `reference` column "
                  "(or all references are empty). Add a `reference` column with ground-truth "
                  "answer strings.")
            return 1
        print(f"  Found {len(refs):,} reference answers")

    db_path = Path(args.db_path).resolve()
    if not db_path.exists():
        print(f"ERROR: Chroma DB not found at {db_path}")
        return 1

    # Generate responses via the live engine
    print("\nGenerating responses through the RAG engine...")
    rows = asyncio.run(_generate_responses(queries))
    if not rows:
        print("No responses generated; nothing to evaluate.")
        return 1

    if args.with_references:
        before = len(rows)
        rows = [r for r in rows if (r["query_id"] in refs)]
        for r in rows:
            r["reference"] = refs[r["query_id"]]
        dropped = before - len(rows)
        if dropped:
            print(f"  Dropped {dropped} queries that had no reference answer.")
        if not rows:
            print("No queries with references; cannot run reference-based metrics.")
            return 1

    # Run RAGAS
    try:
        result = _run_ragas(
            rows=rows,
            judge_model=args.judge_model,
            embedding_model=args.embedding_model,
            with_references=args.with_references,
            api_key=api_key,
        )
    except Exception as e:
        print(f"ERROR: RAGAS evaluate failed: {e}")
        return 1

    extra = {
        "label": args.label,
        "db_path": str(db_path),
        "judge_model": args.judge_model,
        "embedding_model": args.embedding_model,
        "with_references": args.with_references,
        "dataset_path": str(dataset_path),
    }
    out_dir = _write_outputs(
        label=args.label,
        per_query=result["per_query"],
        aggregate=result["aggregate"],
        extra=extra,
    )
    print(f"\nResults written to: {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Run RAGAS evaluation against the library RAG system."
    )
    p.add_argument("--check", action="store_true",
                   help="Only verify packages, API key, and model connectivity. No eval.")
    p.add_argument("--dataset",
                   help="Path to queries CSV (same schema as the main eval harness).")
    p.add_argument("--db-path",
                   help="Path to a ChromaDB folder.")
    p.add_argument("--label", default="ragas_default",
                   help="Output label; results land in eval/ragas_eval/results/<label>/.")
    p.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT,
                   help=f"Judge LLM model id (default: {JUDGE_MODEL_DEFAULT}).")
    p.add_argument("--embedding-model", default=EMBEDDING_MODEL_DEFAULT,
                   help=f"Embedding model used by RAGAS internals "
                        f"(default: {EMBEDDING_MODEL_DEFAULT}).")
    p.add_argument("--with-references", action="store_true",
                   help="Also run context_precision + context_recall. "
                        "Dataset must have a `reference` column with ground-truth answers.")
    args = p.parse_args()

    if args.check:
        return cmd_check(args)

    if not args.dataset or not args.db_path:
        p.error("--dataset and --db-path are required (or use --check).")

    return cmd_eval(args)


if __name__ == "__main__":
    sys.exit(main())
