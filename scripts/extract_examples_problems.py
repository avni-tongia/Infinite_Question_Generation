# 1
"""
# extract_examples_problems.py
Parses chapter text (from structure_text.py) to extract:
- Solved examples: (question_text, solution_text, page_span)
- End-of-chapter problems: (question_text, page_span)

Outputs:
- data/examples.jsonl
- data/problems.jsonl
- data/hcverma_with_examples.json   (chapters with example/problem IDs backfilled)
"""
from pathlib import Path
import re, json
from typing import List, Dict, Tuple

# --- ROOTS ---
ROOT_DIR = Path(__file__).resolve().parent

# Candidate locations for the structured file
CANDIDATES = [
    ROOT_DIR / "data" / "hcverma_structured.json",       # scripts/data/...
    ROOT_DIR.parent / "data" / "hcverma_structured.json" # project/data/...
]

def resolve_structured_path() -> Path:
    tried = []
    for p in CANDIDATES:
        tried.append(str(p))
        if p.exists():
            print(f"[INFO] Using structured file at: {p}")
            return p
    # Not found: helpful diagnostics
    print("[ERROR] Could not find hcverma_structured.json. Tried:")
    for t in tried:
        print("   -", t)
    # Show directory listings to help debugging
    scripts_data = ROOT_DIR / "data"
    top_data = ROOT_DIR.parent / "data"
    print(f"[DEBUG] Listing {scripts_data}:", list(scripts_data.glob("*")) if scripts_data.exists() else "MISSING")
    print(f"[DEBUG] Listing {top_data}:", list(top_data.glob("*")) if top_data.exists() else "MISSING")
    raise FileNotFoundError("hcverma_structured.json not found in expected locations.")

IN_STRUCT = resolve_structured_path()

OUT_STRUCT = ROOT_DIR / "data" / "hcverma_with_examples.json"
OUT_EX = ROOT_DIR / "data" / "examples.jsonl"
OUT_PB = ROOT_DIR / "data" / "problems.jsonl"

# 29
# ---------- PATTERNS ----------
# We keep heading detection tolerant to OCR/edition quirks.
EXAMPLE_HEAD = re.compile(r"(?im)^\s*(example(?:\s*\d+)?|solved\s*example(?:\s*\d+)?)\b.*$")
SOLUTION_HEAD = re.compile(r"(?im)^\s*(solution|sol\.)\b.*$")
PROBLEM_HEAD = re.compile(r"(?im)^\s*(question\s*\d+|q\.?\s*\d+)\b.*$")

# Page marker inserted by preprocess_pdf.py and carried by structure_text.py
PAGE_MARK = re.compile(r"(?im)^---\s*page\s+(\d+)\s*---\s*$")

# 39
def load_structured(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# 44
def dump_jsonl(lines: List[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 50
def find_page_hits(text: str) -> List[int]:
    """Return a sorted list of page numbers that appear inside this text block."""
    return [int(m.group(1)) for m in PAGE_MARK.finditer(text)]

# 55
def estimate_span_from_text(ch_page_span: Tuple[int, int], block_text: str) -> Tuple[int, int]:
    """
    Estimate page span for an example/problem by scanning embedded '--- page N ---' markers.
    Fallback to the chapter span if markers aren't found (keeps alignment usable).
    """
    hits = find_page_hits(block_text)
    if hits:
        return (min(hits), max(hits))
    return (ch_page_span[0], ch_page_span[1])

# 65
def normalize_whitespace(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# 71
def split_into_paragraphs(text: str) -> List[str]:
    """Light splitter to help with boundary detection if needed later."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

# 76
def extract_examples_from_chapter(chapter: Dict) -> List[Dict]:
    """
    Heuristic parser:
    - Find an example heading.
    - Collect text until 'Solution' heading → that's question_text.
    - Collect from 'Solution' until next example/problem heading (or EOF) → solution_text.
    """
    content = chapter.get("content", "")
    ch_span = tuple(chapter.get("page_span", (0, 0)))
    lines = content.splitlines()

    examples = []
    i = 0
    ex_idx = 0

    while i < len(lines):
        line = lines[i]
        if EXAMPLE_HEAD.match(line):
            # Start of an example block
            ex_start = i
            # Find solution heading for this example (if any)
            sol_start = None
            j = i + 1
            while j < len(lines):
                if EXAMPLE_HEAD.match(lines[j]) or PROBLEM_HEAD.match(lines[j]):
                    break
                if sol_start is None and SOLUTION_HEAD.match(lines[j]):
                    sol_start = j
                j += 1
            ex_end = j  # exclusive

            block = "\n".join(lines[ex_start:ex_end])
            # Split question vs solution
            if sol_start is not None:
                q_block = "\n".join(lines[ex_start:sol_start])
                s_block = "\n".join(lines[sol_start:ex_end])
                # Strip the headings from blocks to keep cleaner text
                # Remove the first line (Example heading) and first 'Solution' line
                q_lines = q_block.splitlines()
                if q_lines:
                    q_lines = q_lines[1:]  # drop "Example ..." line
                s_lines = s_block.splitlines()
                if s_lines:
                    s_lines = s_lines[1:]  # drop "Solution ..." line

                question_text = normalize_whitespace("\n".join(q_lines))
                solution_text = normalize_whitespace("\n".join(s_lines))
            else:
                # No explicit "Solution" heading; treat whole block as question.
                question_text = normalize_whitespace(block)
                solution_text = ""

            # Estimate span
            block_span = estimate_span_from_text(ch_span, block)

            ex_idx += 1
            examples.append({
                "example_id": f"Ch{chapter['chapter_number']:02d}_Ex{ex_idx:02d}",
                "chapter_number": chapter["chapter_number"],
                "chapter_title": chapter.get("chapter_title", f"Chapter {chapter['chapter_number']}"),
                "page_span": list(block_span),
                "question_text": question_text,
                "solution_text": solution_text,
                "concept_tags": [],
                "difficulty": None,
                "equations_used": [],
                "function_signatures": []
            })

            i = ex_end
            continue

        i += 1

    return examples

# 140
def extract_problems_from_chapter(chapter: Dict) -> List[Dict]:
    """
    Heuristic parser for end-of-chapter problems:
    - Find a problem heading like "Question 5" or "Q. 5".
    - Capture text until the next problem/example heading or EOF.
    """
    content = chapter.get("content", "")
    ch_span = tuple(chapter.get("page_span", (0, 0)))
    lines = content.splitlines()

    problems = []
    i = 0
    pb_idx = 0

    while i < len(lines):
        line = lines[i]
        if PROBLEM_HEAD.match(line):
            pb_start = i
            j = i + 1
            while j < len(lines):
                if PROBLEM_HEAD.match(lines[j]) or EXAMPLE_HEAD.match(lines[j]):
                    break
                j += 1
            pb_end = j  # exclusive

            block = "\n".join(lines[pb_start:pb_end])
            # Strip heading line
            pb_lines = block.splitlines()
            if pb_lines:
                pb_lines = pb_lines[1:]
            question_text = normalize_whitespace("\n".join(pb_lines))
            block_span = estimate_span_from_text(ch_span, block)

            pb_idx += 1
            problems.append({
                "problem_id": f"Ch{chapter['chapter_number']:02d}_Q{pb_idx:02d}",
                "chapter_number": chapter["chapter_number"],
                "chapter_title": chapter.get("chapter_title", f"Chapter {chapter['chapter_number']}"),
                "page_span": list(block_span),
                "question_text": question_text,
                "concept_tags": [],
                "difficulty": None,
                "equations_candidates": []
            })

            i = pb_end
            continue

        i += 1

    return problems

# 196
def main():
    if not IN_STRUCT.exists():
        raise FileNotFoundError(f"Structured chapters not found at {IN_STRUCT}")

    chapters = load_structured(IN_STRUCT)
    all_examples: List[Dict] = []
    all_problems: List[Dict] = []

    # Extract chapter-wise
    for ch in chapters:
        exs = extract_examples_from_chapter(ch)
        pbs = extract_problems_from_chapter(ch)

        # Backfill chapter with IDs + page spans for manifest building
        ch["examples"] = [{"example_id": e["example_id"], "page_span": e["page_span"]} for e in exs]
        ch["problems"] = [{"problem_id": p["problem_id"], "page_span": p["page_span"]} for p in pbs]

        all_examples.extend(exs)
        all_problems.extend(pbs)

    # Write outputs
    dump_jsonl(all_examples, OUT_EX)
    dump_jsonl(all_problems, OUT_PB)
    with OUT_STRUCT.open("w", encoding="utf-8") as f:
        json.dump(chapters, f, indent=2, ensure_ascii=False)

    print(f"[OK] Examples:  {len(all_examples)}  -> {OUT_EX}")
    print(f"[OK] Problems:  {len(all_problems)} -> {OUT_PB}")
    print(f"[OK] Chapters+IDs backfilled -> {OUT_STRUCT}")

# 227
if __name__ == "__main__":
    main()
