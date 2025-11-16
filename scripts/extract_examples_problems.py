#!/usr/bin/env python3
"""
Extract solved examples and end-of-chapter problems from structured HC Verma text.

We handle two broad example types:

1) Inline examples:
    "2 Example 1.1  A block of mass m..."
    ...
    "... Solution : ..."

2) Worked Out Examples pages:
    "Worked Out Examples"
    "1. Find the dimensional formulae..."
    ...
    "... Solution : ..."

We only keep examples where we can find BOTH:
  - a non-trivial question part
  - a non-trivial solution part

End-of-chapter problems are parsed from exercise / question sections.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

STRUCTURED_INPUT  = ROOT_DIR / "data" / "hcverma_structured.json"
STRUCTURED_OUTPUT = ROOT_DIR / "data" / "hcverma_with_examples.json"
EXAMPLES_JSONL    = ROOT_DIR / "data" / "examples.jsonl"
PROBLEMS_JSONL    = ROOT_DIR / "data" / "problems.jsonl"


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# "Worked Out Examples" heading
WORKED_OUT_HEADING = re.compile(
    r"(?im)^\s*worked\s+out\s+examples\b.*$"
)

# Inline example heading, allowing an OCR page number in front:
# "Example 1.1", "2 Example 1.1"
# OLD (anchored at start with ^)
# EXAMPLE_HEAD = re.compile(
#     r"(?im)^\s*(?:\d+\s+)?example\s+(\d+(?:\.\d+)?)\b.*$"
# )

# NEW: match "Example 5.2" anywhere in the line
EXAMPLE_HEAD = re.compile(
    r"(?i)example\s+(\d+(?:\.\d+)?)\b"
)


# We treat any line containing the word "solution" as a solution line
SOLUTION_FINDER = re.compile(r"(?i)\bsolution\b")

# Exercise / question section headings
EXERCISE_SECTION_HEAD = re.compile(
    r"(?im)^\s*(?:exercise|exercises|questions|objective\s+questions|"
    r"short\s+answer\s+type\s+questions|miscellaneous\s+questions|"
    r"multiple\s+choice\s+questions)\b.*$"
)

# Individual problem / example numbers, e.g. "1. ...", "Q. 2)", "3)"
PROBLEM_ITEM_HEAD = re.compile(
    r"(?im)^\s*(?:Q\.?\s*)?(\d+)[\.\)]\s+"
)

# "Question 3" style (used in problem fallback)
PROBLEM_HEAD = re.compile(
    r"(?im)^\s*(?:question|q\.?|prob(?:lem)?)\s*[:\-\.\)]*\s*\d+\b.*$"
)

# Split mid-line occurrences of "N." that look like example numbers:
# e.g. "... r h 3. The SI and CGS units ..." → ["... r h", "3. The SI and CGS units ..."]
INLINE_SUBQ_SPLIT = re.compile(
    r"(?=(?<!\d)\d{1,2}\.\s)"
)


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Collapse whitespace runs into single spaces and trim."""
    return re.sub(r"\s+", " ", text.strip())


def estimate_span_from_text(ch_span: Tuple[int, int], block: str) -> Tuple[int, int]:
    """For now, just return the chapter's page span."""
    return ch_span


def load_chapters(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(items: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Example extraction: Worked Out Examples
# ---------------------------------------------------------------------------

def parse_worked_out_section(
    chapter: Dict,
    lines: List[str],
    worked_idx: int,
    ch_span: Tuple[int, int]
) -> List[Dict]:
    """
    Parse a 'Worked Out Examples' section starting at worked_idx.

    We:
      - take everything until the next exercise / questions section (or EOF),
      - normalise lines by splitting on mid-line "N." patterns,
      - for each numbered example:
          * if there's a 'Solution' line inside its segment → split Q / A
          * otherwise → keep it as a question-only example (solution_text = "")
    """

    # Section boundaries in the raw line space
    section_start = worked_idx + 1
    section_end = len(lines)
    for j in range(section_start, len(lines)):
        if EXERCISE_SECTION_HEAD.match(lines[j]):
            section_end = j
            break

    raw_body_lines = lines[section_start:section_end]

    # Normalise: split lines on inline "N." so 1., 2., 3., ... start new pseudo-lines
    norm_lines: List[str] = []
    for L in raw_body_lines:
        for seg in INLINE_SUBQ_SPLIT.split(L):
            seg = seg.strip()
            if seg:
                norm_lines.append(seg)

    examples: List[Dict] = []
    n = len(norm_lines)

    i = 0
    while i < n:
        line = norm_lines[i]
        m_item = PROBLEM_ITEM_HEAD.match(line)
        if not m_item:
            i += 1
            continue

        # Candidate example anchored at this number
        ex_start = i
        j = i + 1
        while j < n and not PROBLEM_ITEM_HEAD.match(norm_lines[j]):
            j += 1

        seg_lines = norm_lines[ex_start:j]

        # ------------------------------------------------------------------
        # 1) Try to find a 'Solution' line inside this numbered block
        # ------------------------------------------------------------------
        sol_pos = None
        for k, L in enumerate(seg_lines):
            if SOLUTION_FINDER.search(L):
                sol_pos = k
                break

        if sol_pos is not None:
            # Question lines are everything before the solution line
            q_lines = seg_lines[:sol_pos]

            # Solution lines: strip 'Solution' label even if it appears mid-line
            first_sol_line = seg_lines[sol_pos]
            m_sol = re.search(r"(?i)\bsolution\b[:\.\-]?\s*(.*)", first_sol_line)
            sol_lines: List[str] = []
            if m_sol:
                rest = m_sol.group(1).strip()
                if rest:
                    sol_lines.append(rest)
            else:
                sol_lines.append(first_sol_line.strip())
            sol_lines.extend(seg_lines[sol_pos + 1:])

            q_raw = "\n".join(q_lines).strip()
            s_raw = "\n".join(sol_lines).strip()

            # Filter out very tiny / noisy fragments
            if len(q_raw) >= 20 and len(s_raw) >= 15:
                question_text = normalize_whitespace(q_raw)
                solution_text = normalize_whitespace(s_raw)
                block = "\n".join(seg_lines)
                span = estimate_span_from_text(ch_span, block)

                examples.append({
                    "line_index": section_start + ex_start,
                    "raw_example_label": None,
                    "question_text": question_text,
                    "solution_text": solution_text,
                    "block": block,
                    "page_span": list(span),
                })

        else:
            # ------------------------------------------------------------------
            # 2) No 'Solution' line inside this numbered block
            #    → keep it as a question-only example (solution_text = "")
            # ------------------------------------------------------------------
            q_raw = "\n".join(seg_lines).strip()
            if len(q_raw) >= 20:
                question_text = normalize_whitespace(q_raw)
                block = "\n".join(seg_lines)
                span = estimate_span_from_text(ch_span, block)

                examples.append({
                    "line_index": section_start + ex_start,
                    "raw_example_label": None,
                    "question_text": question_text,
                    "solution_text": "",  # no explicit solution detected
                    "block": block,
                    "page_span": list(span),
                })

        i = j

    return examples



# ---------------------------------------------------------------------------
# Example extraction: Inline "Example 1.1" style
# ---------------------------------------------------------------------------

def parse_inline_examples(
    chapter: Dict,
    lines: List[str],
    ch_span: Tuple[int, int]
) -> List[Dict]:
    """
    Parse inline "Example 1.1" style solved examples.

    Pattern:
        "2 Example 1.1 ..."
        ...
        "... Solution : ...
        ..."

    Behaviour:
      - If a 'Solution' line is found inside the example block → split Q / A.
      - If no 'Solution' is found → keep it as a question-only example
        (solution_text = ""), so nothing silently disappears.
    """
    n = len(lines)
    raw_examples: List[Dict] = []
    i = 0

    while i < n:
        line = lines[i]
        m_ex = EXAMPLE_HEAD.search(line)
        if not m_ex:
            i += 1
            continue

        raw_label = m_ex.group(1)  # "1.1", "2.3", etc.
        ex_start = i
        sol_start = None

        j = i + 1
        while j < n:
            # Stop when we hit the next block start
            if EXAMPLE_HEAD.search(lines[j]) \
               or WORKED_OUT_HEADING.match(lines[j]) \
               or EXERCISE_SECTION_HEAD.match(lines[j]):
                break

            # First line containing "Solution" marks start of solution
            if sol_start is None and SOLUTION_FINDER.search(lines[j]):
                sol_start = j

            j += 1

        ex_end = j

        # ------------------------------------------------------------------
        # Case 1: we found a 'Solution' line in this example block
        # ------------------------------------------------------------------
        if sol_start is not None and sol_start < ex_end:
            q_block = "\n".join(lines[ex_start:sol_start])

            s_lines_full = lines[sol_start:ex_end]
            if not s_lines_full:
                i = ex_end
                continue

            first_sol_line = s_lines_full[0]
            m_sol = re.search(r"(?i)\bsolution\b[:\.\-]?\s*(.*)", first_sol_line)
            sol_lines: List[str] = []
            if m_sol:
                rest = m_sol.group(1).strip()
                if rest:
                    sol_lines.append(rest)
            else:
                sol_lines.append(first_sol_line.strip())
            sol_lines.extend(s_lines_full[1:])

            q_lines = q_block.splitlines()
            if q_lines:
                # Drop the "Example ..." heading line
                q_lines = q_lines[1:]

            question_raw = "\n".join(q_lines).strip()
            solution_raw = "\n".join(sol_lines).strip()

            # Require a non-trivial question; solution can be shorter
            if len(question_raw) < 20:
                i = ex_end
                continue

            question_text = normalize_whitespace(question_raw)
            solution_text = normalize_whitespace(solution_raw)
            block = "\n".join(lines[ex_start:ex_end])
            span = estimate_span_from_text(ch_span, block)

            raw_examples.append({
                "line_index": ex_start,
                "raw_example_label": raw_label,
                "question_text": question_text,
                "solution_text": solution_text,
                "block": block,
                "page_span": list(span),
            })

        # ------------------------------------------------------------------
        # Case 2: NO 'Solution' inside this Example block
        #         → keep question-only example
        # ------------------------------------------------------------------
        else:
            q_block = "\n".join(lines[ex_start:ex_end])
            q_lines = q_block.splitlines()
            if q_lines:
                # Drop "Example ..." heading line
                q_lines = q_lines[1:]
            question_raw = "\n".join(q_lines).strip()

            if len(question_raw) >= 20:
                question_text = normalize_whitespace(question_raw)
                block = "\n".join(lines[ex_start:ex_end])
                span = estimate_span_from_text(ch_span, block)

                raw_examples.append({
                    "line_index": ex_start,
                    "raw_example_label": raw_label,
                    "question_text": question_text,
                    "solution_text": "",  # no explicit solution in this block
                    "block": block,
                    "page_span": list(span),
                })

        i = ex_end

    return raw_examples



def extract_examples_from_chapter(chapter: Dict) -> List[Dict]:
    """Combine inline and worked-out examples for a chapter."""
    content = chapter.get("content", "")
    ch_span = tuple(chapter.get("page_span", (0, 0)))
    lines = content.splitlines()

    chapter_number = chapter.get("chapter_number", 0)
    chapter_title = chapter.get("chapter_title", f"Chapter {chapter_number}")

    raw_examples: List[Dict] = []

    # 1) Worked Out Examples sections
    for idx, line in enumerate(lines):
        if WORKED_OUT_HEADING.match(line):
            raw_examples.extend(parse_worked_out_section(chapter, lines, idx, ch_span))

    # 2) Inline "Example n.m" blocks
    raw_examples.extend(parse_inline_examples(chapter, lines, ch_span))

    # 3) Sort + dedupe by (line_index, question_text prefix)
    raw_examples.sort(key=lambda ex: ex["line_index"])
    examples: List[Dict] = []
    seen_keys = set()

    for ex_idx, ex in enumerate(raw_examples, start=1):
        key = (ex["line_index"], ex["question_text"][:50])
        if key in seen_keys:
            continue
        seen_keys.add(key)

        examples.append({
            "example_id": f"Ch{int(chapter_number):02d}_Ex{ex_idx:02d}",
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            "page_span": ex["page_span"],
            "raw_example_label": ex.get("raw_example_label"),
            "question_text": ex["question_text"],
            "solution_text": ex["solution_text"],
            "concept_tags": [],
            "difficulty": None,
            "equations_used": [],
            "function_signatures": [],
        })

    return examples


# ---------------------------------------------------------------------------
# Problem extraction (unchanged logic)
# ---------------------------------------------------------------------------

def extract_problems_from_chapter(chapter: Dict) -> List[Dict]:
    """
    Extract end-of-chapter problems from exercise / questions sections.
    """
    content = chapter.get("content", "")
    ch_span = tuple(chapter.get("page_span", (0, 0)))
    lines = content.splitlines()

    chapter_number = chapter.get("chapter_number", 0)
    chapter_title = chapter.get("chapter_title", f"Chapter {chapter_number}")

    problems: List[Dict] = []
    pb_idx = 0

    section_starts: List[int] = [
        idx for idx, line in enumerate(lines) if EXERCISE_SECTION_HEAD.match(line)
    ]

    # Fallback if we have no explicit EXERCISE headings
    if not section_starts:
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            if PROBLEM_HEAD.match(line):
                pb_start = i
                j = i + 1
                while j < n:
                    if PROBLEM_HEAD.match(lines[j]) \
                       or EXAMPLE_HEAD.match(lines[j]) \
                       or WORKED_OUT_HEADING.match(lines[j]):
                        break
                    j += 1
                pb_end = j

                block = "\n".join(lines[pb_start:pb_end])
                pb_lines = block.splitlines()
                if pb_lines:
                    pb_lines = pb_lines[1:]
                question_raw = "\n".join(pb_lines)
                question_text = normalize_whitespace(question_raw)

                if len(question_text) < 10:
                    i = pb_end
                    continue

                span = estimate_span_from_text(ch_span, block)
                pb_idx += 1
                problems.append({
                    "problem_id": f"Ch{int(chapter_number):02d}_Q{pb_idx:03d}",
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "page_span": list(span),
                    "raw_question_label": None,
                    "question_text": question_text,
                    "concept_tags": [],
                    "difficulty": None,
                    "equations_candidates": [],
                })

                i = pb_end
                continue

            i += 1

        return problems

    # Normal path: explicit exercise sections
    n = len(lines)
    for idx, start in enumerate(section_starts):
        section_start = start + 1
        section_end = section_starts[idx + 1] if idx + 1 < len(section_starts) else n

        i = section_start
        while i < section_end:
            line = lines[i]
            m_item = PROBLEM_ITEM_HEAD.match(line)
            is_problem_line = bool(m_item) or PROBLEM_HEAD.match(line)

            if is_problem_line:
                pb_start = i
                j = i + 1
                while j < section_end:
                    if PROBLEM_ITEM_HEAD.match(lines[j]) \
                       or PROBLEM_HEAD.match(lines[j]) \
                       or EXERCISE_SECTION_HEAD.match(lines[j]) \
                       or EXAMPLE_HEAD.match(lines[j]) \
                       or WORKED_OUT_HEADING.match(lines[j]):
                        break
                    j += 1
                pb_end = j

                block = "\n".join(lines[pb_start:pb_end])
                pb_lines = block.splitlines()
                if pb_lines:
                    pb_lines = pb_lines[1:]
                question_raw = "\n".join(pb_lines)
                question_text = normalize_whitespace(question_raw)

                if len(question_text) < 10:
                    i = pb_end
                    continue

                raw_label = None
                if m_item:
                    raw_label = m_item.group(1)

                span = estimate_span_from_text(ch_span, block)
                pb_idx += 1
                problems.append({
                    "problem_id": f"Ch{int(chapter_number):02d}_Q{pb_idx:03d}",
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "page_span": list(span),
                    "raw_question_label": raw_label,
                    "question_text": question_text,
                    "concept_tags": [],
                    "difficulty": None,
                    "equations_candidates": [],
                })

                i = pb_end
                continue

            i += 1

    return problems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    chapters = load_chapters(STRUCTURED_INPUT)

    # Ensure chapter_number is always present
    for idx, ch in enumerate(chapters, start=1):
        if "chapter_number" not in ch or ch["chapter_number"] is None:
            ch["chapter_number"] = idx

    all_examples: List[Dict] = []
    all_problems: List[Dict] = []

    for ch in chapters:
        exs = extract_examples_from_chapter(ch)
        pbs = extract_problems_from_chapter(ch)

        ch["examples"] = [{"example_id": ex["example_id"]} for ex in exs]
        ch["problems"] = [{"problem_id": pb["problem_id"]} for pb in pbs]

        all_examples.extend(exs)
        all_problems.extend(pbs)

        print(
            f"[INFO] Chapter {ch['chapter_number']:02d}: "
            f"{len(exs)} examples, {len(pbs)} problems"
        )

    save_json(chapters, STRUCTURED_OUTPUT)
    write_jsonl(all_examples, EXAMPLES_JSONL)
    write_jsonl(all_problems, PROBLEMS_JSONL)

    print(f"[OK] Chapters with annotations → {STRUCTURED_OUTPUT}")
    print(f"[OK] Total examples extracted: {len(all_examples)} → {EXAMPLES_JSONL}")
    print(f"[OK] Total problems extracted: {len(all_problems)} → {PROBLEMS_JSONL}")


if __name__ == "__main__":
    main()
