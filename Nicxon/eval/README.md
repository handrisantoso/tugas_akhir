# RAG Evaluation Harness

Lightweight evaluation tooling for the library RAG system. Built to compare
**different embedding models** without touching application code, and to layer
in optional response-quality grading via an LLM-as-judge.

## What it measures

**Retrieval metrics** (modality-agnostic, computed against expected book_ids):
- `Recall@1`, `Recall@5`, `Recall@10`
- `MRR` — mean reciprocal rank of the first correct hit
- `HitRate@K` — fraction of queries with at least one expected book in the top-K
- `NDCG@10` — discounted gain (only meaningful when relevance is graded; falls
  back to a binary version when there's exactly one expected book per query)

**Optional response-quality metrics** (LLM-as-judge, multimodal):
- `relevance` — does the response address the query (and the uploaded image)?
- `faithfulness` — is the response grounded in the retrieved books?
- `helpfulness` — would a real user find this useful?

Each rated 1-5 by a judge model that sees the user's text, the user's image
(if any), the retrieved books, and the generated response.

## Files

| File | Purpose |
|---|---|
| `dataset.py`            | Load and validate the queries CSV. |
| `runner.py`             | Embed queries, run searches against a Chroma collection, capture top-K + latency. |
| `retrieval_metrics.py`  | Pure-Python implementations of Recall@K, MRR, NDCG, etc. |
| `llm_judge.py`          | Optional multimodal LLM-as-judge for response quality. |
| `run_eval.py`           | Single-entrypoint CLI that ties the above together. |
| `sample_dataset.csv`    | A 6-row example dataset showing the expected schema. |

## Dataset format

CSV with these columns:

| column | required | type | description |
|---|---|---|---|
| `query_id`            | yes  | str   | Unique id you choose; used in the output filename. |
| `query_text`          | yes  | str   | The user's text. May be empty for image-only queries. |
| `image_path`          | no   | str   | Absolute or project-relative path to a cover image, if any. |
| `expected_book_ids`   | yes  | str   | Pipe-separated list of correct book_ids, e.g. `isbn13_9781234567890|isbn13_9780987654321`. |
| `expected_search_mode`| no   | str   | One of `TEXT`, `IMAGE`, `HYBRID`. If omitted, the engine decides. |
| `notes`               | no   | str   | Free text for your own reference. |

A query is considered "correctly retrieved at K" if **any** of the expected
book_ids appears in the top-K results (`HitRate@K` and `Recall@K` collapse to
the same number when there's one expected id; with multiple expected ids,
Recall@K is `|hit ∩ expected| / |expected|`).

## Quick start

```cmd
:: Retrieval-only run against the current default DB
python eval\run_eval.py --dataset eval\sample_dataset.csv --db-path vector_db\chroma_db_multimodal_2

:: Compare two DBs by running twice and labelling outputs
python eval\run_eval.py --dataset my_eval.csv --db-path vector_db\chroma_db_openrouter --label openrouter
python eval\run_eval.py --dataset my_eval.csv --db-path vector_db\chroma_db_google    --label google

:: Add LLM-as-judge response quality on top of retrieval (slower, costs tokens)
python eval\run_eval.py --dataset my_eval.csv --db-path vector_db\chroma_db_google --label google --judge
```

Outputs land in `eval/results/<label>/`:

- `per_query.csv`     — one row per query with retrieved ids, similarities, latency, hit info, judge scores (if requested)
- `aggregate.json`    — single-file summary with all metrics
- `aggregate.txt`     — human-readable summary

## Comparing models

After running multiple labels, compare them at a glance:

```cmd
python eval\run_eval.py --compare openrouter google
```

Prints a side-by-side table reading from each label's `aggregate.json`.

## Implementation notes

- Embedding lives in `runner.py` and mirrors `RAG/rag_engine.py` exactly:
  same model id, same image preprocessing pipeline. So when the embedding model
  changes (e.g. you point at a `chroma_db` built from a different ingest run),
  swap the model id in `runner.py` to match.
- The harness queries Chroma directly — it deliberately does not call the full
  RAG engine, so it's fast and isolates the embedding+retrieval layer. If you
  enable `--judge`, *that's* where the engine's response generator gets called.
- No framework dependency. Plain numpy/pandas. The judge LLM uses the same
  google-genai client the rest of the project uses.

## Optional: RAGAS add-on

If you want academically-recognised methodology on top of the metrics above,
the optional `eval/ragas_eval/` folder ships a separate runner that calls the
official RAGAS metrics (`faithfulness`, `answer_relevancy`, and optionally
`context_precision` + `context_recall` when references are provided).

It's fully isolated — no imports from this folder reach into the RAGAS folder,
and RAGAS is not installed by default. See `eval/ragas_eval/README.md` for
setup. Reasonable to leave alone until you actually want to run it.
