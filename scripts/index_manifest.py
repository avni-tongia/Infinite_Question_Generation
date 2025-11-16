#!/usr/bin/env python3
"""
index_manifest.py

Build a central manifest tying together:
- Chapters (from hcverma_with_examples.json)
- Examples (references from chapters)
- Problems (references from chapters)
- Equations (from equations.jsonl, assigned to chapters by page_span)

Output:
    scripts/data/manifest.json

Structure (top-level):
{
  "chapters": [
    {
      "chapter_id": "Ch01",
      "chapter_number": 1,
      "chapter_title": "Chapter 1: ...",
      "page_span": [start_page, end_page],
      "examples": ["Ch01_Ex01", "Ch01_Ex02", ...],
      "problems": ["Ch01_Q001", "Ch01_Q002", ...],
      "equations": ["eq_p0017_001", "eq_p0019_002", ...]
    },
    ...
  ],
  "pages": {
    "17": {
      "equations": ["eq_p0017_001", "eq_p0017_002"]
    },
    ...
  }
}
"""

import json
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------
# PATHS (relative to .../Infinite_Question_Generation/scripts)
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

STRUCT_JSON = ROOT_DIR / "data" / "hcverma_with_examples.json"
EQ_JSONL    = ROOT_DIR / "data" / "equations" / "equations.jsonl"

OUT_PATH    = ROOT_DIR / "data" / "manifest.json"


# ---------------------------------------------------------------------
# I/O HELPERS
# ---------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    path = Path(path)
    items: List[Dict] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------

def build_chapter_index(chapters: List[Dict]) -> List[Dict]:
    """
    Prepare a lightweight index for chapters with page spans.

    Each entry:
    {
      "chapter_number": int,
      "chapter_title": str,
      "chapter_id": str,
      "page_start": int,
      "page_end": int,
      "examples": [example_ids...],
      "problems": [problem_ids...]
    }
    """
    index: List[Dict] = []

    for ch in chapters:
        span = ch.get("page_span") or [None, None]
        start, end = span
        chap_num = ch.get("chapter_number")

        # Build chapter_id from chapter_number if available
        chap_id = f"Ch{int(chap_num):02d}" if chap_num is not None else None

        example_ids = [
            e["example_id"]
            for e in ch.get("examples", [])
            if isinstance(e, dict) and "example_id" in e
        ]

        problem_ids = [
            p["problem_id"]
            for p in ch.get("problems", [])
            if isinstance(p, dict) and "problem_id" in p
        ]

        index.append({
            "chapter_number": chap_num,
            "chapter_title": ch.get("chapter_title"),
            "chapter_id": chap_id,
            "page_start": start,
            "page_end": end,
            "examples": example_ids,
            "problems": problem_ids,
        })

    # Sort by page_start so everything is ordered in reading order
    index.sort(key=lambda c: (c["page_start"] if c["page_start"] is not None else 10**9))
    return index


def assign_equations_to_chapters(
    eqs: List[Dict],
    chapter_index: List[Dict]
) -> Dict[str, List[str]]:
    """
    Given equations with a 'page' field and a chapter index with page spans,
    return a mapping:
        chapter_id -> [equation_id, ...]
    """
    by_chapter: Dict[str, List[str]] = {ch["chapter_id"]: [] for ch in chapter_index if ch["chapter_id"]}

    for eq in eqs:
        eq_id = eq.get("equation_id")
        page = eq.get("page")
        if eq_id is None or page is None:
            continue

        # Find chapter whose span contains this page
        for ch in chapter_index:
            ch_id = ch.get("chapter_id")
            if ch_id is None:
                continue

            start = ch.get("page_start")
            end   = ch.get("page_end")
            if start is None or end is None:
                continue

            if start <= page <= end:
                by_chapter.setdefault(ch_id, []).append(eq_id)
                break

    return by_chapter


def build_page_index(eqs: List[Dict]) -> Dict[str, Dict[str, List[str]]]:
    """
    Build a simple page-wise index of equations.

    Returns:
    {
      "17": {
        "equations": ["eq_p0017_001", ...]
      },
      ...
    }
    """
    pages: Dict[str, Dict[str, List[str]]] = {}

    for eq in eqs:
        eq_id = eq.get("equation_id")
        page = eq.get("page")
        if eq_id is None or page is None:
            continue

        key = str(page)
        if key not in pages:
            pages[key] = {"equations": []}
        pages[key]["equations"].append(eq_id)

    return pages


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    # 1) Load enriched chapter structure (with examples + problems)
    if not STRUCT_JSON.exists():
        raise FileNotFoundError(f"Structured file not found: {STRUCT_JSON}")
    chapters = load_json(STRUCT_JSON)

    # 2) Load equations (already cleaned) – optional, may be empty
    eqs = load_jsonl(EQ_JSONL)

    # 3) Build chapter index & assign equations
    chapter_index = build_chapter_index(chapters)
    eq_by_chapter = assign_equations_to_chapters(eqs, chapter_index)
    page_index = build_page_index(eqs)

    # 4) Construct manifest
    manifest: Dict[str, Any] = {"chapters": [], "pages": page_index}

    for ch in chapter_index:
        ch_id = ch.get("chapter_id")
        manifest["chapters"].append({
            "chapter_id": ch_id,
            "chapter_number": ch.get("chapter_number"),
            "chapter_title": ch.get("chapter_title"),
            "page_span": [ch.get("page_start"), ch.get("page_end")],
            "examples": ch.get("examples", []),
            "problems": ch.get("problems", []),
            "equations": eq_by_chapter.get(ch_id, []),
        })

    # 5) Save
    save_json(manifest, OUT_PATH)
    print(f"[INFO] Built manifest with {len(manifest['chapters'])} chapters.")
    print(f"[INFO] Wrote manifest -> {OUT_PATH}")


if __name__ == "__main__":
    main()
