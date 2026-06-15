"""
Optional multimodal LLM-as-judge for end-to-end response quality.

This is the only piece of the eval harness that:
  - calls the full RAG engine (so the response is generated exactly as users see it)
  - can use a *different* LLM as the rater (the "judge" model)
  - sees the user's image when judging (the standard text-only RAGAS-style
    evaluators can't do this)

Scores returned per (query, response):
  relevance     1-5  — does the response address the user's intent (text + image)?
  faithfulness  1-5  — is the response grounded in the retrieved books?
  helpfulness   1-5  — would a real user find this useful?

The judge is asked for a JSON blob. We parse leniently: extract the first
{...} object from the response and pull whichever fields are present. Missing
fields default to None and are reported as such — no silent zeros.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a library book search assistant.
You will rate one response on three dimensions, each on a 1-5 scale (5 is best).

Dimensions:
- relevance: Does the response actually address the user's request? If the user attached
  an image, does the response acknowledge and use it appropriately? An image-only query
  asking "find similar" should be answered with visually or topically related books, not
  with a generic conversation.
- faithfulness: Are the response's claims supported by the retrieved books shown to you?
  Penalize hallucinated books, made-up authors, fabricated dates, or claims that contradict
  the snippets.
- helpfulness: Would a real library user find this response useful? Consider clarity,
  specificity, and whether it includes enough information to act on.

Respond ONLY with a single JSON object, no prose around it. Schema:
{"relevance": <1-5>, "faithfulness": <1-5>, "helpfulness": <1-5>, "reason": "<one sentence>"}
"""


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_score_json(text: str) -> Dict[str, Any]:
    """Pull the first {...} block from `text` and parse it as JSON.

    Returns a dict with whichever of the four expected keys we could parse.
    Missing fields are recorded as None so the caller can distinguish
    "judge said 1" from "judge didn't say".
    """
    parsed: Dict[str, Any] = {
        "relevance": None,
        "faithfulness": None,
        "helpfulness": None,
        "reason": None,
        "raw": text,
    }
    if not text:
        return parsed

    m = _JSON_RE.search(text)
    if not m:
        return parsed
    blob = m.group(0)
    try:
        data = json.loads(blob)
    except Exception:
        # Common LLM tic: trailing comma. Try a quick repair.
        try:
            data = json.loads(re.sub(r",\s*([}\]])", r"\1", blob))
        except Exception:
            return parsed

    for k in ("relevance", "faithfulness", "helpfulness", "reason"):
        if k in data:
            v = data[k]
            if k != "reason" and isinstance(v, (int, float)):
                v = max(1, min(5, int(round(v))))
            parsed[k] = v
    return parsed


# ----------------------------------------------------------------------
# Building the judge prompt
# ----------------------------------------------------------------------

def _format_books_for_judge(retrieved_books: List[Dict[str, Any]]) -> str:
    if not retrieved_books:
        return "(no books retrieved)"
    lines: List[str] = []
    for i, b in enumerate(retrieved_books, start=1):
        meta = b.get("metadata") or {}
        title = (meta.get("title") or "").strip() or "(no title)"
        bid = meta.get("book_id", "?")
        snippet = (b.get("document") or "")[:600].replace("\n", " ").strip()
        lines.append(f"{i}. book_id={bid} | title={title}\n   snippet: {snippet}")
    return "\n".join(lines)


def build_judge_prompt(query_text: str,
                       retrieved_books: List[Dict[str, Any]],
                       generated_response: str) -> str:
    return (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        f"=== USER TEXT ===\n"
        f"{query_text or '(empty)'}\n\n"
        f"=== RETRIEVED BOOKS (top-{len(retrieved_books)}) ===\n"
        f"{_format_books_for_judge(retrieved_books)}\n\n"
        f"=== GENERATED RESPONSE ===\n"
        f"{generated_response}\n\n"
        f"Rate now."
    )


# ----------------------------------------------------------------------
# Judge call
# ----------------------------------------------------------------------

@dataclass
class JudgeScore:
    relevance: Optional[int] = None
    faithfulness: Optional[int] = None
    helpfulness: Optional[int] = None
    reason: Optional[str] = None
    raw: Optional[str] = None
    error: Optional[str] = None

    @property
    def mean(self) -> Optional[float]:
        scores = [s for s in (self.relevance, self.faithfulness, self.helpfulness)
                  if isinstance(s, int)]
        return sum(scores) / len(scores) if scores else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "judge_relevance": self.relevance,
            "judge_faithfulness": self.faithfulness,
            "judge_helpfulness": self.helpfulness,
            "judge_mean": round(self.mean, 3) if self.mean is not None else None,
            "judge_reason": self.reason,
            "judge_error": self.error or "",
        }


def call_judge(client, model_id: str,
               query_text: str,
               retrieved_books: List[Dict[str, Any]],
               generated_response: str,
               image_jpeg: Optional[bytes] = None) -> JudgeScore:
    """Run the judge LLM and return a JudgeScore.

    `client` is a google.genai.Client. `model_id` is the judge model id
    (recommended: a stronger model than the response generator, e.g.
    `gemini-2.5-pro` or `gemini-2.5-flash`).
    """
    prompt = build_judge_prompt(query_text, retrieved_books, generated_response)

    contents: Any = prompt
    if image_jpeg:
        try:
            from google.genai import types
            image_part = types.Part.from_bytes(data=image_jpeg, mime_type="image/jpeg")
            contents = [image_part, prompt]
        except Exception as e:
            # Fall through to text-only — keep the score, lose the image input.
            return JudgeScore(error=f"could not attach image: {e}")

    try:
        resp = client.models.generate_content(model=model_id, contents=contents)
        text = (resp.text or "").strip()
    except Exception as e:
        return JudgeScore(error=f"judge call failed: {e}")

    parsed = _parse_score_json(text)
    return JudgeScore(
        relevance=parsed["relevance"],
        faithfulness=parsed["faithfulness"],
        helpfulness=parsed["helpfulness"],
        reason=parsed["reason"],
        raw=parsed["raw"],
    )
