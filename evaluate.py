"""
Metrics (always run, all samples have ground_truth):
  - faithfulness                             % of claims grounded in context
  - answer_relevancy                         how well the answer addresses the question
  - llm_context_precision_with_reference     top retrieved chunks actually useful?
  - context_recall                           retrieval coverage vs ground truth
  - answer_correctness                       factual correctness (semantic + claim F1)
"""
import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from openai import AsyncOpenAI
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    _Faithfulness,
    _AnswerRelevancy,
    _LLMContextPrecisionWithReference,
    _LLMContextRecall,
    _AnswerCorrectness,
)
from ragas.run_config import RunConfig

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"

if not OPENROUTER_API_KEY:
    sys.exit("[Eval] ERROR: OPENROUTER_API_KEY is not set in your .env file.")

DEFAULT_JUDGE_MODEL      = "google/gemini-3-flash-preview"
DEFAULT_JUDGE_MAX_TOKENS = 2000
DEFAULT_RUN_CONFIG = RunConfig(
    max_workers=2,
    max_retries=10,
    max_wait=60,
    timeout=240,
)

# ---------------------------------------------------------------------------
# RAGAS LLM
# ---------------------------------------------------------------------------
def _build_evaluator_llm(model: str, max_tokens: int) -> LangchainLLMWrapper:
    chat = ChatOpenAI(
        model=model,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE,
        max_tokens=max_tokens,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainLLMWrapper(chat)

# ---------------------------------------------------------------------------
# RAGAS Embeddings
# ---------------------------------------------------------------------------
def _build_relevancy_embeddings() -> LangchainOpenAIEmbeddings:
    return LangchainOpenAIEmbeddings(
        model="google/gemini-embedding-001",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE,
        model_kwargs={"encoding_format": "float"},
        check_embedding_ctx_length=False,
        max_retries=6,
        retry_min_seconds=4,
        retry_max_seconds=30,
    )


def _build_similarity_embeddings():
    from ragas.embeddings.openai_provider import (
        OpenAIEmbeddings as RagasOpenAIEmbeddings,
    )

    class _FloatEncodingEmbeddings(RagasOpenAIEmbeddings):
        """Thin subclass that forces encoding_format='float' on every call."""

        async def aembed_text(self, text, **kwargs):
            kwargs.setdefault("encoding_format", "float")
            return await super().aembed_text(text, **kwargs)

        async def aembed_texts(self, texts, **kwargs):
            kwargs.setdefault("encoding_format", "float")
            return await super().aembed_texts(texts, **kwargs)

        def embed_text(self, text, **kwargs):
            kwargs.setdefault("encoding_format", "float")
            return super().embed_text(text, **kwargs)

        def embed_texts(self, texts, **kwargs):
            kwargs.setdefault("encoding_format", "float")
            return super().embed_texts(texts, **kwargs)

    async_client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE,
    )
    return _FloatEncodingEmbeddings(
        client=async_client,
        model="google/gemini-embedding-001",
    )

# ---------------------------------------------------------------------------
# Test set loading
# ---------------------------------------------------------------------------
def load_test(
    path: str,
    mode_filter: str | None = None,
    category_filter: str | None = None,
) -> list[dict]:
    """Loads and optionally filters the JSON test set."""
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if mode_filter:
        samples = [s for s in samples if s.get("mode") == mode_filter]
    if category_filter:
        samples = [s for s in samples if s.get("category") == category_filter]

    if not samples:
        filters = []
        if mode_filter:
            filters.append(f"mode={mode_filter!r}")
        if category_filter:
            filters.append(f"category={category_filter!r}")
        sys.exit(f"[Eval] No samples found ({', '.join(filters) or 'no filter'}). Aborting.")

    cats  = {}
    modes = {}
    for s in samples:
        c = s.get("category", "uncategorised")
        m = s.get("mode", "customer")
        cats[c] = cats.get(c, 0) + 1
        modes[m] = modes.get(m, 0) + 1

    print(f"[Eval] Loaded {len(samples)} test sample(s):")
    for m, n in sorted(modes.items()):
        print(f"         mode={m}: {n}")
    for c, n in sorted(cats.items()):
        print(f"         category={c}: {n}")
    return samples

# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def run_pipeline(samples: list[dict]) -> list[dict]:
    print("\n[Eval] Initialising RAG pipeline (this may take a moment)...")
    from rag import retrieve_and_generate

    results: list[dict] = []
    total = len(samples)

    for i, sample in enumerate(samples, 1):
        if "question" not in sample:
            print(f"  [Eval] Sample {i} missing 'question' field. Skipping.")
            continue

        question     = sample["question"]
        mode         = sample.get("mode", "customer")
        ground_truth = sample.get("ground_truth", "")
        category     = sample.get("category", "uncategorised")
        has_gt       = bool(ground_truth and str(ground_truth).strip())

        print(f"  [{i}/{total}] mode={mode:8} cat={category:<20} | {question[:60]}...")

        try:
            output = retrieve_and_generate(question, mode)
        except Exception as e:
            print(f"         ❌ Pipeline error: {e}")
            continue

        results.append({
            "sample_id":    i - 1,
            "question":     question,
            "answer":       output["answer"],
            "contexts":     output["contexts"],
            "full_context": output["context"],
            "ground_truth": ground_truth if has_gt else "",
            "mode":         mode,
            "category":     category,
            "has_gt":       has_gt,
        })
        if i < total:
            time.sleep(1)

    print(f"\n[Eval] Pipeline complete: {len(results)}/{total} samples collected.\n")
    return results

# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------
def run_ragas(
    results: list[dict],
    evaluator_llm,
    relevancy_emb,
    similarity_emb,
    run_config: RunConfig,
) -> pd.DataFrame:
    ref_free_metrics = [
        _Faithfulness(llm=evaluator_llm),
        _AnswerRelevancy(llm=evaluator_llm, embeddings=relevancy_emb, strictness=1),
    ]
    ref_based_metrics = [
        _LLMContextPrecisionWithReference(llm=evaluator_llm),
        _LLMContextRecall(llm=evaluator_llm),
        _AnswerCorrectness(llm=evaluator_llm, embeddings=similarity_emb),
    ]
    print("[Eval] Running reference-free metrics (faithfulness, answer_relevancy)...")

    samples_all = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
        )
        for r in results
    ]
    ds_all     = EvaluationDataset(samples=samples_all)
    scores_all = evaluate(
        ds_all,
        metrics=ref_free_metrics,
        run_config=run_config,
        raise_exceptions=False,
    )
    df = scores_all.to_pandas()

    # Attach metadata columns
    df["sample_id"] = [r["sample_id"] for r in results]
    df["mode"]      = [r["mode"]      for r in results]
    df["category"]  = [r["category"]  for r in results]
    df["has_gt"]    = [r["has_gt"]    for r in results]
    df["question"]  = [r["question"]  for r in results]
    df["answer"]    = [r["answer"]    for r in results]
    df["ground_truth"] = [r["ground_truth"] for r in results]

    #Reference-based: samples with ground truth
    gt_results = [r for r in results if r["has_gt"]]
    if gt_results:
        print(f"[Eval] Running reference-based metrics on {len(gt_results)} "
              f"sample(s) with ground truth...")

        samples_gt = [
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["ground_truth"],
            )
            for r in gt_results
        ]
        ds_gt     = EvaluationDataset(samples=samples_gt)
        scores_gt = evaluate(
            ds_gt,
            metrics=ref_based_metrics,
            run_config=run_config,
            raise_exceptions=False,
        )
        df_gt = scores_gt.to_pandas()
        df_gt["sample_id"] = [r["sample_id"] for r in gt_results]

        new_cols = [c for c in df_gt.columns
                    if c not in df.columns and c != "sample_id"]
        if new_cols:
            df = df.merge(
                df_gt[["sample_id"] + new_cols],
                on="sample_id",
                how="left",
            )
    else:
        print("[Eval] No ground_truth provided — skipping reference-based metrics.")

    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
_META_COLS = frozenset({
    "question", "answer", "contexts", "ground_truth", "full_context",
    "user_input", "response", "retrieved_contexts", "reference",
    "reference_contexts",
    "mode", "has_gt", "sample_id", "category", "run",
})


def _metric_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of numeric metric columns (excludes metadata)."""
    return [
        c for c in df.columns
        if c not in _META_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def print_per_sample(df: pd.DataFrame) -> None:
    """Prints a scorecard for every individual sample — fast visual scan."""
    metrics = _metric_cols(df)
    if not metrics:
        return

    print("\n" + "=" * 100)
    print("  PER-SAMPLE SCORECARD")
    print("=" * 100)

    # Header
    metric_labels = [m[:12] for m in metrics]
    header = f"  {'#':>3}  {'mode':>8}  {'category':<20}  " + "  ".join(
        f"{ml:>12}" for ml in metric_labels
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for _, row in df.iterrows():
        sid  = int(row.get("sample_id", 0))
        mode = str(row.get("mode", ""))[:8]
        cat  = str(row.get("category", ""))[:20]
        vals = []
        for m in metrics:
            v = row.get(m)
            if pd.isna(v):
                vals.append(f"{'—':>12}")
            else:
                vals.append(f"{v:>12.4f}")
        print(f"  {sid:>3}  {mode:>8}  {cat:<20}  " + "  ".join(vals))

    print("=" * 100)


def print_summary(df: pd.DataFrame) -> None:
    metrics = _metric_cols(df)
    if not metrics:
        print("[Eval] No metric columns found to summarise.")
        return

    print("\n" + "=" * 80)
    print("  RAGAS EVALUATION SUMMARY")
    print("=" * 80)

    # ── Per-mode breakdown ──────────────────────────────────────────────────
    for mode in sorted(df["mode"].unique()):
        subset = df[df["mode"] == mode]
        print(f"\n  Mode: {mode.upper()}  (n={len(subset)})")
        print("  " + "-" * 65)
        for col in metrics:
            valid = subset[col].dropna()
            if valid.empty:
                print(f"    {col:<44}  (all NaN)")
                continue
            mean = valid.mean()
            bar  = "█" * int(mean * 20)
            pct_valid = 100 * len(valid) / len(subset)
            print(f"    {col:<44} {mean:.4f}  {bar}  ({pct_valid:.0f}% scored)")

    # Per-category breakdown
    categories = sorted(c for c in df["category"].unique() if c != "uncategorised")
    if categories:
        print(f"\n  Per-category breakdown (n per category):")
        print("  " + "-" * 65)

        # Show counts
        for c in categories:
            n = len(df[df["category"] == c])
            print(f"    {c:<20} n={n}")
        print()

        # Metric table
        cat_labels = [c[:16] for c in categories]
        header = f"    {'metric':<40} " + " ".join(f"{cl:>16}" for cl in cat_labels)
        print(header)
        print("    " + "─" * (len(header) - 4))

        for col in metrics:
            row = f"    {col:<40} "
            for c in categories:
                valid = df[df["category"] == c][col].dropna()
                if not valid.empty:
                    row += f" {valid.mean():>15.3f}"
                else:
                    row += f" {'—':>15}"
            print(row)

    # Overall averages 
    print(f"\n  Overall averages (n={len(df)}):")
    print("  " + "-" * 65)
    for col in metrics:
        valid = df[col].dropna()
        pct = 100 * len(valid) / len(df)
        if not valid.empty:
            bar = "█" * int(valid.mean() * 20)
            print(f"    {col:<44} {valid.mean():.4f}  {bar}  "
                  f"({pct:.0f}% scored)")

    print("=" * 80 + "\n")


def save_results(df: pd.DataFrame, output_path: str) -> None:
    drop_cols = ["contexts", "retrieved_contexts", "full_context",
                 "reference_contexts"]
    save_df = df.drop(
        columns=[c for c in drop_cols if c in df.columns],
        errors="ignore",
    )
    save_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[Eval] Results saved → {output_path}")


def save_detailed_json(
    results: list[dict],
    df: pd.DataFrame,
    output_path: str,
) -> None:
    metrics = _metric_cols(df)
    detailed = []

    for r in results:
        sid = r["sample_id"]
        row = df[df["sample_id"] == sid]
        scores = {}
        if not row.empty:
            for m in metrics:
                v = row.iloc[0].get(m)
                scores[m] = round(float(v), 4) if pd.notna(v) else None

        detailed.append({
            "sample_id":    sid,
            "mode":         r["mode"],
            "category":     r["category"],
            "question":     r["question"],
            "ground_truth": r["ground_truth"],
            "answer":       r["answer"],
            "scores":       scores,
            "num_contexts": len(r["contexts"]),
            "full_context": r["full_context"],
        })

    json_path = output_path.replace(".csv", "_detailed.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    print(f"[Eval] Detailed report → {json_path}")


def _print_variance_summary(all_runs: list[pd.DataFrame]) -> None:
    if len(all_runs) < 2:
        return

    print("\n" + "=" * 80)
    print(f"  VARIANCE SUMMARY (across {len(all_runs)} runs)")
    print("=" * 80)
    print("  LLM-as-judge scores fluctuate even at temperature=0.")
    print("  Differences smaller than the std below are noise.\n")

    metrics = _metric_cols(all_runs[0])

    print(f"  {'metric':<44} {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    print("  " + "-" * 80)
    for col in metrics:
        per_run_means = [
            df[col].dropna().mean()
            for df in all_runs
            if col in df.columns
        ]
        per_run_means = [m for m in per_run_means if not pd.isna(m)]
        if not per_run_means:
            continue
        s = pd.Series(per_run_means)
        print(f"  {col:<44} {s.mean():>8.4f}  {s.std():>8.4f}  "
              f"{s.min():>8.4f}  {s.max():>8.4f}")
    print("=" * 80 + "\n")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation for the Shopee RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py                              # full test suite, 1 run
  python evaluate.py --mode customer              # customer samples only
  python evaluate.py --category ranking           # ranking samples only
  python evaluate.py --repeats 3                  # 3 runs for variance
  python evaluate.py --category refusal_customer  # test refusal behaviour
        """,
    )
    parser.add_argument("--test-set",   default="test.json",
                        help="Path to test set JSON (default: test.json)")
    parser.add_argument("--mode",       default=None,
                        help="Filter: 'customer' or 'owner'")
    parser.add_argument("--category",   default=None,
                        help="Filter: e.g. 'ranking', 'product_lookup', 'refusal_customer'")
    parser.add_argument("--output",     default=None,
                        help="Output CSV path (default: eval_results_<timestamp>.csv)")
    parser.add_argument("--judge",      default=DEFAULT_JUDGE_MODEL,
                        help=f"Judge model (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS,
                        help="Max tokens for judge output (default: 2000)")
    parser.add_argument("--repeats",    type=int, default=1,
                        help="Run N times to estimate LLM-as-judge variance (default: 1)")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="RAGAS async concurrency (default: 2; lower if rate-limited)")
    args = parser.parse_args()

    if not Path(args.test_set).exists():
        sys.exit(f"[Eval] Test set not found: {args.test_set}")
    if args.repeats < 1:
        sys.exit("[Eval] --repeats must be >= 1.")
    if args.max_tokens < 500:
        sys.exit("[Eval] --max-tokens must be >= 500.")

    output_path = args.output or (
        f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    # Load test set
    samples = load_test(
        args.test_set,
        mode_filter=args.mode,
        category_filter=args.category,
    )

    # Run RAG pipeline
    pipeline_start = time.monotonic()
    results = run_pipeline(samples)
    pipeline_elapsed = time.monotonic() - pipeline_start

    if not results:
        sys.exit("[Eval] No results collected. Aborting.")

    print(f"[Eval] Pipeline: {pipeline_elapsed:.1f}s total, "
          f"{pipeline_elapsed / len(results):.1f}s/sample.\n")

    # Set up RAGAS evaluator
    print(f"[Eval] Setting up RAGAS evaluator "
          f"(judge={args.judge}, max_tokens={args.max_tokens}, "
          f"workers={args.max_workers})...")

    evaluator_llm  = _build_evaluator_llm(args.judge, args.max_tokens)
    relevancy_emb  = _build_relevancy_embeddings()
    similarity_emb = _build_similarity_embeddings()

    run_config = RunConfig(
        max_workers=args.max_workers,
        max_retries=DEFAULT_RUN_CONFIG.max_retries,
        max_wait=DEFAULT_RUN_CONFIG.max_wait,
        timeout=DEFAULT_RUN_CONFIG.timeout,
    )

    # Run RAGAS evaluation
    all_runs: list[pd.DataFrame] = []
    for run_idx in range(args.repeats):
        if args.repeats > 1:
            print(f"\n[Eval] ───── RAGAS run {run_idx + 1}/{args.repeats} ─────")
        run_start = time.monotonic()
        df = run_ragas(results, evaluator_llm, relevancy_emb, similarity_emb,
                       run_config)
        df["run"] = run_idx + 1
        all_runs.append(df)
        elapsed = time.monotonic() - run_start
        print(f"[Eval] Run {run_idx + 1} completed in {elapsed:.1f}s.")

    combined = pd.concat(all_runs, ignore_index=True)

    if args.repeats > 1:
        metric_cols = _metric_cols(combined)
        per_sample_mean = (
            combined.groupby("sample_id")[metric_cols].mean().reset_index()
        )
        meta = combined.drop_duplicates("sample_id")[
            ["sample_id", "mode", "category", "question", "answer", "ground_truth"]
        ]
        summary_df = per_sample_mean.merge(meta, on="sample_id", how="left")
        print_per_sample(summary_df)
        print_summary(summary_df)
        _print_variance_summary(all_runs)
    else:
        print_per_sample(combined)
        print_summary(combined)

    # Save outputs
    save_results(combined, output_path)
    save_detailed_json(results, combined if args.repeats == 1 else summary_df,
                       output_path)

    total_elapsed = time.monotonic() - pipeline_start
    print(f"\n[Eval] Done. Total wall time: {total_elapsed:.0f}s.\n")


if __name__ == "__main__":
    main()