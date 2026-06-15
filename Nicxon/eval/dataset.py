"""
Load and validate evaluation datasets.

Schema (CSV):
  query_id              required
  query_text            required (may be empty if image_path is set)
  image_path            optional
  expected_book_ids     required, pipe-separated
  expected_search_mode  optional: TEXT|IMAGE|HYBRID
  notes                 optional

The loader returns a list of EvalQuery records. Empty / "none" / "null" cells
are normalised to None for optional fields.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EvalQuery:
    query_id: str
    query_text: str
    expected_book_ids: List[str]
    image_path: Optional[str] = None
    expected_search_mode: Optional[str] = None
    notes: Optional[str] = None

    # Resolved by the loader so the runner doesn't need to think about CWD.
    resolved_image_path: Optional[Path] = field(default=None, repr=False)

    @property
    def has_image(self) -> bool:
        return self.image_path is not None and bool(self.image_path.strip())

    @property
    def has_text(self) -> bool:
        return bool((self.query_text or "").strip())


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------

_NONE_TOKENS = {"", "none", "null", "n/a", "na"}


def _norm(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    if v.lower() in _NONE_TOKENS:
        return None
    return v


def _resolve_image(raw_path: str, project_root: Path) -> Path:
    """Resolve an image path that may be absolute or project-relative."""
    p = Path(raw_path)
    if p.is_absolute():
        return p
    # Try as-given relative to CWD first, then relative to project root.
    cwd_path = (Path.cwd() / p).resolve()
    if cwd_path.exists():
        return cwd_path
    return (project_root / p).resolve()


def load_dataset(csv_path: str | Path,
                 project_root: Optional[Path] = None) -> List[EvalQuery]:
    """Read and validate an eval CSV. Returns a list of EvalQuery."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    project_root = project_root or csv_path.resolve().parent.parent

    queries: List[EvalQuery] = []
    seen_ids: set[str] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"query_id", "query_text", "expected_book_ids"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Dataset {csv_path} is missing required columns: {sorted(missing)}"
            )

        for line_no, row in enumerate(reader, start=2):
            qid = (row.get("query_id") or "").strip()
            if not qid:
                raise ValueError(f"{csv_path}:{line_no} — empty query_id")
            if qid in seen_ids:
                raise ValueError(f"{csv_path}:{line_no} — duplicate query_id '{qid}'")
            seen_ids.add(qid)

            text = (row.get("query_text") or "").rstrip("\n")
            image_path = _norm(row.get("image_path"))
            expected_raw = _norm(row.get("expected_book_ids"))
            expected: List[str] = []
            if expected_raw:
                expected = [b.strip() for b in expected_raw.split("|") if b.strip()]
            if not expected:
                raise ValueError(
                    f"{csv_path}:{line_no} — expected_book_ids is empty for query '{qid}'"
                )

            mode = _norm(row.get("expected_search_mode"))
            if mode is not None:
                mode_upper = mode.upper()
                if mode_upper not in {"TEXT", "IMAGE", "HYBRID"}:
                    raise ValueError(
                        f"{csv_path}:{line_no} — expected_search_mode must be TEXT|IMAGE|HYBRID, got {mode!r}"
                    )
                mode = mode_upper

            if not text and not image_path:
                raise ValueError(
                    f"{csv_path}:{line_no} — query '{qid}' has neither text nor image"
                )

            resolved = None
            if image_path:
                resolved = _resolve_image(image_path, project_root)
                if not resolved.exists():
                    raise FileNotFoundError(
                        f"{csv_path}:{line_no} — image not found for query '{qid}': {resolved}"
                    )

            queries.append(EvalQuery(
                query_id=qid,
                query_text=text,
                expected_book_ids=expected,
                image_path=image_path,
                expected_search_mode=mode,
                notes=_norm(row.get("notes")),
                resolved_image_path=resolved,
            ))

    return queries
