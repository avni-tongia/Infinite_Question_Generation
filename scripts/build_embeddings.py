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
            "solution_text": "..."
        }

Output:
    scripts/data/embeddings/examples_embeddings.jsonl

        Each line:
        {
            "example_id": "Ch01_Ex01",
            "chapter_number": 1,
            "chapter_title": "...",
            "text": "TEXT USED FOR EMBEDDING",
            "embedding": [0.0123, -0.0456, ...]   # dummy but deterministic
        }

Right now this uses a dummy embedding backend so the pipeline works
without any external model or API. Later you can replace the backend
with a real embedding model (OpenAI / sentence-transformers) without
changing the rest of the code.
"""

import json
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
# DUMMY EMBEDDING BACKEND
# ---------------------------------------------------------------------
# This is ONLY to keep your pipeline running without external models.
# It produces deterministic vectors that have no real semantic meaning,
# but they are stable and shaped correctly. You can swap this out later.
# ---------------------------------------------------------------------

EMBEDDING_DIM = 256  # arbitrary dimension for dummy embeddings


def dummy_embedding_backend(texts: List[str]) -> List[List[float]]:
    """
    Dummy backend: create a pseudo-embedding from the UTF-8 bytes of the text.
    This is deterministic (same text -> same vector) but NOT semantically meaningful.
    """
    import math

    vecs: List[List[float]] = []

    for t in texts:
        v = [0.0] * EMBEDDING_DIM
        # Use up to 4 * EMBEDDING_DIM bytes to mix into the vector
        data = t.encode("utf-8")[: 4 * EMBEDDING_DIM]
        for i, b in enumerate(data):
            idx = i % EMBEDDING_DIM
            v[idx] += b / 255.0

        # L2-normalize so magnitudes are comparable
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        v = [x / norm for x in v]
        vecs.append(v)

    return vecs


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Wrapper used by the rest of the script.
    Right now this calls the dummy backend.

    Later, to use a real model:
      - Replace this function body with OpenAI / sentence-transformers calls.
    """
    return dummy_embedding_backend(texts)


# ---------------------------------------------------------------------
# CORE PIPELINE
# ---------------------------------------------------------------------

def build_example_text(ex: Dict[str, Any]) -> str:
    """
    Decide what text to embed for an example.

    For now:
      [chapter title]
      + "Question: " + question_text

    This keeps embeddings relatively short and focused on the question,
    but with a bit of chapter context.
    """
    chapter_title = ex.get("chapter_title") or ""
    q = ex.get("question_text") or ""
    text = f"{chapter_title}\n\nQuestion: {q}".strip()
    return text


def build_embeddings_for_examples() -> None:
    if not EXAMPLES_JSONL.exists():
        raise FileNotFoundError(f"examples.jsonl not found at: {EXAMPLES_JSONL}")

    examples = load_examples(EXAMPLES_JSONL)
    print(f"[INFO] Loaded {len(examples)} examples from {EXAMPLES_JSONL}")

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

    # Batch for future real backends / APIs (still useful even with dummy)
    BATCH_SIZE = 64
    all_embeddings: List[List[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        print(f"[INFO] Embedding batch {start}–{start + len(batch) - 1} ...")
        emb_batch = get_embeddings(batch)
        all_embeddings.extend(emb_batch)

    assert len(all_embeddings) == len(meta), "Embedding count mismatch."

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
