#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
PROJECT_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(EVAL_ROOT))

from dataset import EvalQuery, load_dataset

JUDGE_PROVIDER_DEFAULT     = "groq"
JUDGE_MODEL_GROQ_DEFAULT   = "llama-3.3-70b-versatile"
JUDGE_MODEL_GOOGLE_DEFAULT = "gemini-3.1-flash-lite"
JUDGE_MODEL_OPENROUTER_DEFAULT = "meta-llama/llama-3.3-70b-instruct"
EMBEDDING_MODEL_DEFAULT    = "gemini-embedding-2"

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


def _load_env() -> Dict[str, str]:
    try:
        from dotenv import load_dotenv
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

    google_key = os.environ.get("GOOGLE_API_KEY")
    if not google_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Put it in eval/.env or RAG/.env."
        )
    return {
        "google": google_key,
        "groq": os.environ.get("GROQ_API_KEY") or "",
        "openrouter": os.environ.get("OPENROUTER_API_KEY") or "",
    }


def _check_optional_imports() -> Dict[str, str]:
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
    statuses["langchain_groq"] = _try("langchain-groq", "langchain_groq")
    statuses["langchain_openai"] = _try("langchain-openai", "langchain_openai")
    return statuses


def _missing_packages(statuses: Dict[str, str]) -> List[str]:
    return [k for k, v in statuses.items() if v.startswith("NOT INSTALLED")]


def _build_ragas_llms_groq(judge_model: str, embedding_model: str, keys: Dict[str, str]):
    from langchain_groq import ChatGroq
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    groq_key = keys.get("groq")
    if not groq_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Add it to your .env file.\n"
            "Get a free key at: https://console.groq.com/keys"
        )

    chat = ChatGroq(
        model=judge_model,
        api_key=groq_key,
        temperature=0.0,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model=f"models/{embedding_model}" if not embedding_model.startswith("models/") else embedding_model,
        google_api_key=keys["google"],
    )
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embeddings)


def _build_ragas_llms_google(judge_model: str, embedding_model: str, keys: Dict[str, str]):
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    chat = ChatGoogleGenerativeAI(
        model=judge_model,
        google_api_key=keys["google"],
        temperature=0.0,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model=f"models/{embedding_model}" if not embedding_model.startswith("models/") else embedding_model,
        google_api_key=keys["google"],
    )
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embeddings)


def _build_ragas_llms_openrouter(judge_model: str, embedding_model: str, keys: Dict[str, str]):
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    openrouter_key = keys.get("openrouter")
    if not openrouter_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found. Add it to your .env file.\n"
            "Get a key at: https://openrouter.ai/keys"
        )

    chat = ChatOpenAI(
        model=judge_model,
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=4096,
        default_headers={
            "HTTP-Referer": "https://github.com/library-rag",
            "X-Title": "Library RAG Eval",
        },
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model=f"models/{embedding_model}" if not embedding_model.startswith("models/") else embedding_model,
        google_api_key=keys["google"],
    )
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embeddings)


def _build_ragas_llms(judge_provider: str, judge_model: str, embedding_model: str, keys: Dict[str, str]):
    if judge_provider == "groq":
        return _build_ragas_llms_groq(judge_model, embedding_model, keys)
    elif judge_provider == "google":
        return _build_ragas_llms_google(judge_model, embedding_model, keys)
    elif judge_provider == "openrouter":
        return _build_ragas_llms_openrouter(judge_model, embedding_model, keys)
    else:
        raise ValueError(f"Unknown judge_provider={judge_provider!r}. Use 'groq', 'google', or 'openrouter'.")


def _check_models(judge_provider: str, judge_model: str, embedding_model: str, keys: Dict[str, str]) -> None:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    if judge_provider == "groq":
        from langchain_groq import ChatGroq
        groq_key = keys.get("groq")
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY not found. Add it to your .env file.")
        chat = ChatGroq(model=judge_model, api_key=groq_key, temperature=0.0)
        print(f"  judge provider: Groq (Meta/Llama — independent from Gemini RAG engine)")
    elif judge_provider == "openrouter":
        from langchain_openai import ChatOpenAI
        or_key = keys.get("openrouter")
        if not or_key:
            raise RuntimeError("OPENROUTER_API_KEY not found. Add it to your .env file.")
        chat = ChatOpenAI(
            model=judge_model,
            api_key=or_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
        )
        print(f"  judge provider: OpenRouter")
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        chat = ChatGoogleGenerativeAI(
            model=judge_model, google_api_key=keys["google"], temperature=0.0
        )
        print(f"  judge provider: Google")

    msg = chat.invoke("Reply with the single word OK.")
    print(f"  judge {judge_model}: ok ({(msg.content or '').strip()[:40]!r})")

    emb = GoogleGenerativeAIEmbeddings(
        model=f"models/{embedding_model}" if not embedding_model.startswith("models/") else embedding_model,
        google_api_key=keys["google"],
    )
    vec = emb.embed_query("smoke test")
    print(f"  embedding {embedding_model}: ok (dim={len(vec)})")


async def _generate_responses(queries: List[EvalQuery], delay: float = 0.0, cache_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    sys.path.insert(0, str(PROJECT_ROOT / "RAG"))
    from rag_engine import LibraryRAGEngine

    engine = LibraryRAGEngine()
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []

    if cache_path and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                loaded_rows = json.load(f)
            for r in loaded_rows:
                if "I apologize, but I encountered an issue" not in r.get("response", ""):
                    rows.append(r)
            print(f"Loaded {len(rows)} valid previously generated responses from cache (discarded {len(loaded_rows) - len(rows)} API errors).")
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}")

    cached_ids = {r["query_id"] for r in rows}

    for i, q in enumerate(queries, start=1):
        if not q.has_text:
            skipped.append(f"{q.query_id} (image-only — RAGAS needs text)")
            continue

        if q.query_id in cached_ids:
            print(f"  [{i}/{len(queries)}] {q.query_id}: SKIP (loaded from cache)")
            continue

        engine.clear_conversation()

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

        if delay > 0 and i < len(queries):
            await asyncio.sleep(delay)

        contexts: List[str] = []
        for sr in resp.get("search_results", []):
            doc = sr.get("document") or ""
            if doc:
                contexts.append(doc[:2000])

        row_data = {
            "query_id": q.query_id,
            "user_input": q.query_text,
            "response": resp.get("response", ""),
            "retrieved_contexts": contexts,
            "reference": "",
        }
        rows.append(row_data)

        if cache_path and "I apologize, but I encountered an issue" not in row_data["response"]:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)

        print(f"  [{i}/{len(queries)}] {q.query_id}: response generated "
              f"(contexts={len(contexts)})")

    if skipped:
        print(f"\nSkipped {len(skipped)} queries (RAGAS does not support multimodal input):")
        for s in skipped:
            print(f"  - {s}")

    return rows


def _read_references_from_csv(csv_path: Path) -> Dict[str, str]:
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


def _run_ragas(rows: List[Dict[str, Any]],
                judge_provider: str,
                judge_model: str,
                embedding_model: str,
                with_references: bool,
                keys: Dict[str, str],
                max_workers: int = 1) -> Dict[str, Any]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
    )

    metrics = [faithfulness, answer_relevancy]
    if with_references:
        from ragas.metrics import context_precision, context_recall
        metrics += [context_precision, context_recall]

    judge_wrapped, embed_wrapped = _build_ragas_llms(
        judge_provider=judge_provider,
        judge_model=judge_model,
        embedding_model=embedding_model,
        keys=keys,
    )

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
        run_config=RunConfig(max_workers=max_workers),
    )

    df = result.to_pandas()
    per_query: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        row_out = {
            "query_id": r["query_id"],
            "user_input": r["user_input"][:200],
            "response": r["response"],
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
        return v != v
    except Exception:
        return False


def _write_outputs(label: str,
                    per_query: List[Dict[str, Any]],
                    aggregate: Dict[str, Any],
                    extra: Dict[str, Any]) -> Path:
    out_dir = HERE / "results" / label
    out_dir.mkdir(parents=True, exist_ok=True)

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

    payload = {**aggregate, **extra}
    (out_dir / "aggregate.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        f"RAGAS evaluation",
        f"label:          {label}",
        f"db_path:        {extra.get('db_path')}",
        f"judge_provider: {extra.get('judge_provider')}",
        f"judge:          {extra.get('judge_model')}",
        f"embedder:       {extra.get('embedding_model')}",
        f"queries:        {aggregate.get('n_queries', 0)}",
        f"with refs:      {extra.get('with_references', False)}",
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


def cmd_check(args) -> int:
    print("Checking optional packages...")
    statuses = _check_optional_imports()
    for k, v in statuses.items():
        print(f"  {k}: {v}")
    missing = _missing_packages(statuses)
    if args.judge_provider == "google":
        missing = [m for m in missing if m not in ("langchain_groq", "langchain_openai")]
    elif args.judge_provider == "groq":
        missing = [m for m in missing if m != "langchain_openai"]
    elif args.judge_provider == "openrouter":
        missing = [m for m in missing if m != "langchain_groq"]
    if missing:
        print(f"\nInstall missing packages with:\n"
              f"    pip install -r eval\\ragas_eval\\requirements.txt")
        return 1

    print("\nLoading API keys...")
    keys = _load_env()
    print("  GOOGLE_API_KEY: ok")
    if keys.get("groq"):
        print("  GROQ_API_KEY:   ok")
    elif args.judge_provider == "groq":
        print("  GROQ_API_KEY:   MISSING — add to .env before running eval")
        print("  Get a free key at: https://console.groq.com/keys")
        return 1
    if keys.get("openrouter"):
        print("  OPENROUTER_API_KEY: ok")
    elif args.judge_provider == "openrouter":
        print("  OPENROUTER_API_KEY: MISSING — add to .env before running eval")
        print("  Get a key at: https://openrouter.ai/keys")
        return 1

    print(f"\nProbing models (provider={args.judge_provider}, judge={args.judge_model}, embed={args.embedding_model})...")
    try:
        _check_models(args.judge_provider, args.judge_model, args.embedding_model, keys)
    except Exception as e:
        print(f"  model probe failed: {e}")
        return 1

    print("\nAll checks passed. Ready to run a real eval.")
    return 0


def cmd_eval(args) -> int:
    statuses = _check_optional_imports()
    missing = _missing_packages(statuses)
    if args.judge_provider == "google":
        missing = [m for m in missing if m not in ("langchain_groq", "langchain_openai")]
    elif args.judge_provider == "groq":
        missing = [m for m in missing if m != "langchain_openai"]
    elif args.judge_provider == "openrouter":
        missing = [m for m in missing if m != "langchain_groq"]
    if missing:
        print("ERROR: missing required packages:", ", ".join(missing))
        print("Install with:  pip install -r eval\\ragas_eval\\requirements.txt")
        return 1

    keys = _load_env()
    if args.judge_provider == "groq" and not keys.get("groq"):
        print("ERROR: GROQ_API_KEY not found. Add it to your .env file.")
        print("  Get a free key at: https://console.groq.com/keys")
        return 1
    if args.judge_provider == "openrouter" and not keys.get("openrouter"):
        print("ERROR: OPENROUTER_API_KEY not found. Add it to your .env file.")
        print("  Get a key at: https://openrouter.ai/keys")
        return 1

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

    out_dir = Path(HERE / "results" / args.label)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "generation_cache.json"

    print("\nGenerating responses through the RAG engine...")
    if args.delay > 0:
        print(f"  (throttling: {args.delay}s delay between queries)")
    rows = asyncio.run(_generate_responses(queries, delay=args.delay, cache_path=cache_path))
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

    try:
        result = _run_ragas(
            rows=rows,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            embedding_model=args.embedding_model,
            with_references=args.with_references,
            keys=keys,
            max_workers=args.max_workers,
        )
    except Exception as e:
        print(f"ERROR: RAGAS evaluate failed: {e}")
        return 1

    extra = {
        "label": args.label,
        "db_path": str(db_path),
        "judge_provider": args.judge_provider,
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
    p.add_argument("--judge-provider", default=JUDGE_PROVIDER_DEFAULT,
                   choices=["groq", "google", "openrouter"],
                   help=f"Provider for the RAGAS judge LLM (default: {JUDGE_PROVIDER_DEFAULT}). "
                        f"'groq' uses Llama via Groq API (requires GROQ_API_KEY, free at console.groq.com). "
                        f"'openrouter' routes to any model via OpenRouter (requires OPENROUTER_API_KEY). "
                        f"'google' uses Gemini via Google API.")
    p.add_argument("--judge-model", default=None,
                   help="Judge LLM model id. Defaults to llama-3.3-70b-versatile for Groq, "
                        "meta-llama/llama-3.3-70b-instruct for OpenRouter, "
                        "and gemini-3.1-flash-lite for Google.")
    p.add_argument("--embedding-model", default=EMBEDDING_MODEL_DEFAULT,
                   help=f"Embedding model used by RAGAS internals "
                        f"(default: {EMBEDDING_MODEL_DEFAULT}; always Google).")
    p.add_argument("--with-references", action="store_true",
                   help="Also run context_precision + context_recall. "
                        "Dataset must have a `reference` column with ground-truth answers.")
    p.add_argument("--delay", type=float, default=12.0, metavar="SECONDS",
                   help="Seconds to wait between RAG engine queries during the generation "
                        "phase. Default: 12.0 (ensures we stay under Gemini's 15 RPM free tier limit).")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Number of concurrent workers RAGAS uses to grade the results. "
                        "Defaults to 1 to bypass strict free-tier concurrency limits "
                        "(like OpenRouter). Set higher if you have paid API access.")
    args = p.parse_args()

    if args.judge_model is None:
        if args.judge_provider == "groq":
            args.judge_model = JUDGE_MODEL_GROQ_DEFAULT
        elif args.judge_provider == "openrouter":
            args.judge_model = JUDGE_MODEL_OPENROUTER_DEFAULT
        else:
            args.judge_model = JUDGE_MODEL_GOOGLE_DEFAULT

    if args.check:
        return cmd_check(args)

    if not args.dataset or not args.db_path:
        p.error("--dataset and --db-path are required (or use --check).")

    return cmd_eval(args)


if __name__ == "__main__":
    sys.exit(main())
