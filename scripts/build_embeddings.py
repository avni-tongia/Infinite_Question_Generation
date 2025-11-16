#!/usr/bin/env python3
"""
build_embeddings.py

Build vector embeddings for HC Verma examples.

Input:
    scripts/data/examples.jsonl
        Each line:
        {
            "example_id": "Ch01_Ex01",
            "chapter_number": 1,
            "chapter_title": "...",
            "page_span": [17, 19],
            "raw_example_label": "1.1",
            "question_text": "...",
            "solution_text": "...",
            ...
        }

Output:
    scripts/data/embeddings/examples_embeddings.jsonl

        Each line:
        {
            "example_id": "Ch01_Ex01",
            "chapter_number": 1,
            "chapter_title": "...",
            "text": "QUESTION TEXT USED FOR EMBEDDING",
            "embedding": [0.0123, -0.0456, ...]
        }

You can later load this file in your SFT dataset builder for similarity search,
clustering, or selecting neighbours for variant generation.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------
# PATHS (relative to .../Infinite_Question_Generation/scripts)
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

EXAMPLES_JSONL = ROOT_DIR / "data" / "examples.jsonl"
EMB_DIR        = ROOT_DIR / "data" / "embeddings"
EX_EMB_JSONL   = EMB_DIR / "examples_embeddings.jsonl"


# ---------------------------------------------------------------------
# I/O UTILITIES
# ---------------------------------------------------------------------

def load_examples(path: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def save_jsonl(items: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# EMBEDDING BACKEND
# ---------------------------------------------------------------------
# This is the only part you need to modify if you change providers.
# For now, we stub a simple "backend" to keep the script runnable
# without external dependencies. Replace `dummy_embedding_backend`
# with a real OpenAI / sentence-transformers backend when ready.
# ---------------------------------------------------------------------

EMBEDDING_DIM = 256  # change if your real model has a different dimension


def dummy_embedding_backend(texts: List[str]) -> List[List[float]]:
    """
    Dummy backend: deterministic pseudo-embeddings based on character codes.
    This is ONLY for debugging the pipeline structure. For real use,
    replace this with calls to your embedding model (e.g., OpenAI).
    """
    import math
    vecs: List[List[float]] = []
    for t in texts:
        # simple hash-based embedding of fixed dimension
        v = [0.0] * EMBEDDING_DIM
        for i, ch in enumerate(t.encode("utf-8")[: 4 * EMBEDDING_DIM]):
            idx = i % EMBEDDING_DIM
            v[idx] += (ch / 255.0)
        # L2-normalize
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Wrapper to call the chosen backend. Right now uses dummy backend
    so the script runs without external packages or API keys.

    To use a real model:
      - Replace dummy_embedding_backend(...) with your OpenAI or
        sentence-transformers call.
    """
    return dummy_embedding_backend(texts)


# ---------------------------------------------------------------------
# CORE PIPELINE
# ---------------------------------------------------------------------

def build_example_text(ex: Dict[str, Any]) -> str:
    """
    Decide what text to embed for an example.

    Options:
      - question only
      - question + first sentence of solution
      - full question + full solution

    For now we embed: [chapter title] + [question_text]
    to keep it specific to the question but aware of topic.
    """
    chapter_title = ex.get("chapter_title") or ""
    q = ex.get("question_text") or ""
    # Keep it simple and short-ish
    text = f"{chapter_title}\n\nQuestion: {q}".strip()
    return text


def build_embeddings_for_examples() -> None:
    if not EXAMPLES_JSONL.exists():
        raise FileNotFoundError(f"examples.jsonl not found at: {EXAMPLES_JSONL}")

    examples = load_examples(EXAMPLES_JSONL)
    print(f"[INFO] Loaded {len(examples)} examples from {EXAMPLES_JSONL}")

    # Prepare texts
    texts: List[str] = []
    meta: List[Dict[str, Any]] = []

    for ex in examples:
        text = build_example_text(ex)
        texts.append(text)
        meta.append({
            "example_id": ex.get("example_id"),
            "chapter_number": ex.get("chapter_number"),
            "chapter_title": ex.get("chapter_title"),
            "text": text,
        })

    # Batch to avoid memory issues (and to adapt later to real API rate limits)
    BATCH_SIZE = 64
    all_embeddings: List[List[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        print(f"[INFO] Embedding batch {start}–{start + len(batch) - 1} ...")
        emb_batch = get_embeddings(batch)
        all_embeddings.extend(emb_batch)

    assert len(all_embeddings) == len(meta), "Embedding count mismatch."

    # Merge meta + embeddings
    out_records: List[Dict[str, Any]] = []
    for m, emb in zip(meta, all_embeddings):
        rec = dict(m)
        rec["embedding"] = emb
        out_records.append(rec)

    save_jsonl(out_records, EX_EMB_JSONL)
    print(f"[INFO] Wrote {len(out_records)} example embeddings -> {EX_EMB_JSONL}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    build_embeddings_for_examples()


if __name__ == "__main__":
    main()
