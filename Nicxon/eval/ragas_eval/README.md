# RAGAS evaluation (optional)

Optional add-on to the main `eval/` harness that runs the official
[RAGAS](https://docs.ragas.io) metrics on top of the same dataset format. Use
this when you want academically-recognised methodology in your writeup. Skip
this if the retrieval metrics + custom multimodal judge in `eval/run_eval.py`
already cover what you need.

This folder is **fully self-contained**. The main `eval/run_eval.py` does NOT
import anything from here, so you can leave RAGAS uninstalled until you
actually want to run it.

The folder is named `ragas_eval/` (rather than `ragas/`) on purpose — Python
would otherwise resolve `import ragas` to this directory and shadow the real
PyPI package.

## Status

- **Reference-free metrics**: ready to run as soon as packages are installed.
  - `faithfulness`
  - `answer_relevancy`
- **Reference-based metrics**: also wired up, but require a `reference` column
  in your dataset CSV (a written ground-truth answer per query).
  - `context_precision`
  - `context_recall`
- **Multimodal**: NOT supported. RAGAS expects text query + text contexts +
  text response. Image-only queries are skipped with a warning. HYBRID queries
  are evaluated using only their text portion.

## Setup

You'll need three packages on top of what the project already uses:

```cmd
pip install -r eval\ragas_eval\requirements.txt
```

That installs:
- `ragas` — the metric library
- `datasets` — the HuggingFace dataset format RAGAS expects
- `langchain-google-genai` — adapter so RAGAS can call `gemini-3.1-flash-lite`,
  `gemini-2.5-flash`, etc., through its LangChain interface

Verify everything is wired up:

```cmd
python eval\ragas_eval\run_ragas_eval.py --check
```

This validates the packages, the API key, and that the configured judge LLM +
embedding model both respond. No eval runs, no dataset needed.

## Choosing a judge model

RAGAS uses one LLM under the hood for everything (claim extraction, NLI checks,
ranking judgments). The choice matters more here than in the custom multimodal
judge because RAGAS uses it for *every internal step*, not just one final call.

Reasonable picks for this project:

| Model | When to use |
|---|---|
| `gemini-2.5-flash` (default) | Default. Cheap, fast, good quality. |
| `gemini-2.5-pro` | Stronger reasoning, more expensive. Worth it if you need defensible numbers for a writeup. |
| `gemini-3.1-flash-lite` | DON'T — same model as the response generator, biases the eval. |

Pass via `--judge-model gemini-2.5-pro` if you want to override the default.
Same for embeddings: `--embedding-model text-embedding-004` (RAGAS uses the
embedding model for `answer_relevancy`'s question-generation similarity step).

## Running

```cmd
:: Reference-free run (just faithfulness + answer_relevancy)
python eval\ragas_eval\run_ragas_eval.py ^
    --dataset eval\my_eval.csv ^
    --db-path vector_db\chroma_db_multimodal_2 ^
    --label google_ragas

:: With references (adds context_precision + context_recall)
:: Requires a `reference` column in the CSV with a ground-truth answer string
python eval\ragas_eval\run_ragas_eval.py ^
    --dataset eval\my_eval.csv ^
    --db-path vector_db\chroma_db_multimodal_2 ^
    --label google_ragas_with_refs ^
    --with-references
```

Outputs land in `eval\ragas_eval\results\<label>\`:
- `per_query.csv`  — one row per query with each RAGAS metric
- `aggregate.json` — summary numbers
- `aggregate.txt`  — human-readable summary

These don't overlap with `eval\results\` so the two harnesses can coexist.

## Dataset format

Same CSV schema as the main harness (see `eval/README.md`), with one optional
extra column:

| column | required for RAGAS | notes |
|---|---|---|
| `query_id`             | yes  | reuse from main dataset |
| `query_text`           | yes  | image-only queries are skipped |
| `image_path`           | no   | ignored by RAGAS |
| `expected_book_ids`    | no   | ignored by RAGAS (used by the main harness only) |
| `expected_search_mode` | no   | ignored |
| `reference`            | yes if `--with-references` | ground-truth answer string |
| `notes`                | no   | ignored |

So the same CSV can drive both harnesses — the RAGAS runner just reads the
columns it needs and ignores the rest.

## Honest caveats

- The reference-based metrics need *narrative* references, not book ids. You're
  writing 50–100 short paragraphs. Plan for that.
- RAGAS scores are not directly comparable to the custom judge scores in the
  main harness. Different prompts, different parsing, different temperatures.
  Pick one and report consistently.
- RAGAS calls its judge LLM many times per query (claim decomposition, then
  one entailment check per claim, then ranking judgments). A 50-query run can
  easily produce 500+ judge calls. Watch your daily quota.
- Some metrics (notably `answer_relevancy`) call the embedding model in
  addition to the LLM. Keep that model fast.
