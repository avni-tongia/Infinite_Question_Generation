#!/usr/bin/env python3
"""
Extract solved examples and end-of-chapter problems from structured HC Verma text.

Assumptions (based on your description + book style):
- All examples are labelled on their own line as:
    "Example 1.1   A block of mass m..."
  where "1.1" means Chapter 1, Example 1.

- Worked-out examples:
    There is a heading line like:
        "Worked Out Examples"
    followed by fully solved, numbered examples like:
        "1. A block of mass m..."
        ...
        "Solution"
        ...

- We only keep examples where we can find BOTH:
    - a question part
    - a Solution block with non-trivial content

- End-of-chapter problems are under headings like:
    "EXERCISE", "OBJECTIVE QUESTIONS", "SHORT ANSWER TYPE QUESTIONS", etc.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from pathlib import Path

# Base directory = folder containing this script (i.e., .../Infinite_Question_Generation/scripts)
ROOT_DIR = Path(__file__).resolve().parent

# Input from the structuring script
STRUCTURED_INPUT = ROOT_DIR / "data" / "hcverma_structured.json"

# Outputs (same folder)
STRUCTURED_OUTPUT = ROOT_DIR / "data" / "hcverma_with_examples.json"
EXAMPLES_JSONL    = ROOT_DIR / "data" / "examples.jsonl"
PROBLEMS_JSONL    = ROOT_DIR / "data" / "problems.jsonl"


# ============================================================================
# REGEX PATTERNS
# ============================================================================

# Heading of the worked-out examples section
WORKED_OUT_HEADING = re.compile(
    r"(?im)^\s*worked\s+out\s+examples\b.*$"
)

# Inline examples, e.g.:
#   "Example 1.1 A block of mass m ..."
EXAMPLE_HEAD = re.compile(
    r"(?im)^\s*example\s+(\d+(?:\.\d+)?)\b.*$"   # captures "1.1", "2.3", etc.
)

# "Solution" line inside each example block
SOLUTION_HEAD = re.compile(
    r"(?im)^\s*solution\s*[:\.\-]?\s*$"
)

# End-of-chapter exercise / question section headings
EXERCISE_SECTION_HEAD = re.compile(
    r"(?im)^\s*(?:exercise|exercises|questions|objective\s+questions|"
    r"short\s+answer\s+type\s+questions|miscellaneous\s+questions|"
    r"multiple\s+choice\s+questions)\b.*$"
)

# Individual problem starters inside those sections:
#   "1. A block of mass...", "Q. 2) A particle...", "3) Find the..."
PROBLEM_ITEM_HEAD = re.compile(
    r"(?im)^\s*(?:Q\.?\s*)?(\d+)[\.\)]\s+"
)

# Alternate "Question 3" style
PROBLEM_HEAD = re.compile(
    r"(?im)^\s*(?:question|q\.?|prob(?:lem)?)\s*[:\-\.\)]*\s*\d+\b.*$"
)

# ============================================================================
# UTILS
# ============================================================================

def normalize_whitespace(text: str) -> str:
    """
    Collapse all whitespace runs into single spaces and trim.
    Suitable for training text, not for pretty printing.
    """
    return re.sub(r"\s+", " ", text.strip())


def estimate_span_from_text(ch_span: Tuple[int, int], block: str) -> Tuple[int, int]:
    """
    Placeholder: for now, just return the chapter's page span.
    If you want finer granularity later, you can implement
    proportional page allocation here.
    """
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


# ============================================================================
# EXAMPLE EXTRACTION
# ============================================================================

def parse_worked_out_section(
    chapter: Dict,
    lines: List[str],
    worked_idx: int,
    ch_span: Tuple[int, int]
) -> List[Dict]:
    """
    Parse the 'Worked Out Examples' section starting at worked_idx.

    We expect examples like:
        1. A block of mass m...
           ...
        Solution
           ...
        2. Another example...
           ...
        Solution
           ...
    """
    section_start = worked_idx + 1

    # Section ends at next exercise heading or end of chapter
    section_end = len(lines)
    for j in range(section_start, len(lines)):
        if EXERCISE_SECTION_HEAD.match(lines[j]):
            section_end = j
            break

    raw_examples: List[Dict] = []
    i = section_start

    while i < section_end:
        line = lines[i]
        m_item = PROBLEM_ITEM_HEAD.match(line)  # e.g., "1.  A block ..."
        if m_item:
            ex_start = i
            j = i + 1
            # Move until next numbered item or section end
            while j < section_end and not PROBLEM_ITEM_HEAD.match(lines[j]):
                j += 1
            ex_end = j

            seg_lines = lines[ex_start:ex_end]

            # Locate "Solution" inside this segment
            sol_pos = None
            for k, L in enumerate(seg_lines):
                if SOLUTION_HEAD.match(L):
                    sol_pos = k
                    break

            # Require explicit Solution with content after it
            if sol_pos is None:
                i = ex_end
                continue

            q_part = "\n".join(seg_lines[:sol_pos]).strip()

            # Keep the text on the "Solution..." line after the label
            solution_first = seg_lines[sol_pos]
            after_label = re.sub(r"(?im)^\s*solution\b[:\.\-]?\s*", "", solution_first).strip()

            solution_lines = []
            if after_label:
                solution_lines.append(after_label)
            solution_lines.extend(seg_lines[sol_pos + 1:])

            s_part = "\n".join(solution_lines).strip()


            # Filter out tiny fragments (usually noise)
            if len(q_part) < 20 or len(s_part) < 20:
                i = ex_end
                continue

            question_text = normalize_whitespace(q_part)
            solution_text = normalize_whitespace(s_part)
            block = "\n".join(seg_lines)
            span = estimate_span_from_text(ch_span, block)

            raw_examples.append({
                "line_index": ex_start,
                "raw_example_label": None,
                "question_text": question_text,
                "solution_text": solution_text,
                "block": block,
                "page_span": list(span),
            })

            i = ex_end
            continue

        i += 1

    return raw_examples


def parse_inline_examples(
    chapter: Dict,
    lines: List[str],
    ch_span: Tuple[int, int]
) -> List[Dict]:
    """
    Parse inline "Example 1.1" style solved examples from the whole chapter text.

    Pattern:
        Example 1.1  A block of mass m...
        ...
        Solution
        ...
        [next Example / Worked Out Examples / Exercise / EOF]
    """
    raw_examples: List[Dict] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        m_ex = EXAMPLE_HEAD.match(line)
        if m_ex:
            raw_label = m_ex.group(1)  # "1.1", "2.3", etc.
            ex_start = i

            sol_start = None
            j = i + 1
            while j < n:
                # Termination conditions for this example block
                if EXAMPLE_HEAD.match(lines[j]) \
                   or WORKED_OUT_HEADING.match(lines[j]) \
                   or EXERCISE_SECTION_HEAD.match(lines[j]):
                    break

                if SOLUTION_HEAD.match(lines[j]) and sol_start is None:
                    sol_start = j

                j += 1

            ex_end = j

            # Require explicit Solution with content
            if sol_start is None or sol_start >= ex_end - 1:
                i = ex_end
                continue

            q_block = "\n".join(lines[ex_start:sol_start])

# Build solution block while preserving text after "Solution"
            s_lines_full = lines[sol_start:ex_end]
            if not s_lines_full:
                i = ex_end
                continue

            first_sol_line = s_lines_full[0]
            after_label = re.sub(r"(?im)^\s*solution\b[:\.\-]?\s*", "", first_sol_line).strip()

            solution_lines = []
            if after_label:
                solution_lines.append(after_label)
            solution_lines.extend(s_lines_full[1:])

            q_lines = q_block.splitlines()
            if q_lines:
                # Drop the "Example ..." heading line
                q_lines = q_lines[1:]

            question_raw = "\n".join(q_lines).strip()
            solution_raw = "\n".join(solution_lines).strip()


            # Filter short fragments
            if len(question_raw) < 20 or len(solution_raw) < 20:
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

            i = ex_end
            continue

        i += 1

    return raw_examples


def extract_examples_from_chapter(chapter: Dict) -> List[Dict]:
    """
    Extract all solved examples from a chapter, combining:
    - Inline "Example 1.1" style
    - Worked Out Examples section

    Only examples that have a clear Question + Solution pair are kept.
    """
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

    # 2) Inline examples
    raw_examples.extend(parse_inline_examples(chapter, lines, ch_span))

    # 3) Sort by line_index and assign example_ids
    raw_examples.sort(key=lambda ex: ex["line_index"])

    examples: List[Dict] = []
    seen_keys = set()

    for ex_idx, ex in enumerate(raw_examples, start=1):
        # dedupe by (line_index, start-of-question)
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
            "function_signatures": []
        })

    return examples


# ============================================================================
# PROBLEM EXTRACTION
# ============================================================================

def extract_problems_from_chapter(chapter: Dict) -> List[Dict]:
    """
    Extract end-of-chapter problems.

    Strategy:
      1) Identify exercise / question sections via EXERCISE_SECTION_HEAD.
      2) Within those, individual problems start at PROBLEM_ITEM_HEAD ("1.", "Q. 2)", ...)
         or PROBLEM_HEAD ("Question 3").
      3) Each problem continues until the next problem anchor, the next exercise heading,
         or EOF.
    """
    content = chapter.get("content", "")
    ch_span = tuple(chapter.get("page_span", (0, 0)))
    lines = content.splitlines()

    chapter_number = chapter.get("chapter_number", 0)
    chapter_title = chapter.get("chapter_title", f"Chapter {chapter_number}")

    problems: List[Dict] = []
    pb_idx = 0

    # 1) Find all exercise / question section starts
    section_starts: List[int] = [
        idx for idx, line in enumerate(lines) if EXERCISE_SECTION_HEAD.match(line)
    ]

    # ----------------------------------------------------------------------
    # Fallback path: no explicit EXERCISE-like headings found
    # ----------------------------------------------------------------------
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
                    pb_lines = pb_lines[1:]  # drop "Question 1" line
                question_raw = "\n".join(pb_lines)
                question_text = normalize_whitespace(question_raw)

                # filter tiny fragments
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
                    "equations_candidates": []
                })

                i = pb_end
                continue

            i += 1

        return problems

    # ----------------------------------------------------------------------
    # Normal path: parse sections marked as EXERCISE / OBJECTIVE QUESTIONS / etc.
    # ----------------------------------------------------------------------
    n = len(lines)
    for idx, start in enumerate(section_starts):
        section_start = start + 1  # skip the heading line
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
                    pb_lines = pb_lines[1:]  # drop "1." / "Q. 2" / "Question 3" line
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
                    "equations_candidates": []
                })

                i = pb_end
                continue

            i += 1

    return problems


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    chapters = load_chapters(STRUCTURED_INPUT)

    # Assign chapter_number if missing
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

    # Save updated chapters
    save_json(chapters, STRUCTURED_OUTPUT)

    # Save flat corpora
    write_jsonl(all_examples, EXAMPLES_JSONL)
    write_jsonl(all_problems, PROBLEMS_JSONL)

    print(f"[OK] Chapters with annotations → {STRUCTURED_OUTPUT}")
    print(f"[OK] Total examples extracted: {len(all_examples)} → {EXAMPLES_JSONL}")
    print(f"[OK] Total problems extracted: {len(all_problems)} → {PROBLEMS_JSONL}")


if __name__ == "__main__":
    main()
