#!/usr/bin/env python3
"""
build_sft_datasets.py

This script builds all SFT datasets needed for the Infinite Question Generator.

Currently implemented:

1. solution_explanation.jsonl
   - Input: example question
   - Output: example solution text

2. variant_generation.jsonl
   - Input: a worked example question + instructions to create a new, similar question
   - Output: an end-of-chapter problem statement (treated as a "variant")

3. template_extraction.jsonl
   - Input: example question + solution + instructions
   - Output: a generalized "template" (for now, we reuse the solution as a proxy)

4. tool_tagging.jsonl
   - Input: example question + instructions
   - Output: a list of equation_ids that are relevant (coarse, based on chapter mapping)

NOTE:
For now, equation-to-example mapping is coarse (chapter-level). This still
allows the model to learn which equations often co-occur with which types of
questions, but we can refine it later with better linking.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

EXAMPLES_JSONL  = ROOT_DIR / "data" / "examples.jsonl"
PROBLEMS_JSONL  = ROOT_DIR / "data" / "problems.jsonl"
EQ_JSONL        = ROOT_DIR / "data" / "equations" / "equations.jsonl"

SFT_DIR = ROOT_DIR / "data" / "sft"
SFT_SOLUTION_EXPLANATION = SFT_DIR / "solution_explanation.jsonl"
SFT_VARIANT_GENERATION   = SFT_DIR / "variant_generation.jsonl"
SFT_TEMPLATE_EXTRACTION  = SFT_DIR / "template_extraction.jsonl"
SFT_TOOL_TAGGING         = SFT_DIR / "tool_tagging.jsonl"


# ---------------------------------------------------------------------
# I/O HELPERS
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_jsonl(items: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# DATASET 1: SOLUTION EXPLANATION
# ---------------------------------------------------------------------

def build_solution_explanation_dataset(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each example:

    INPUT:
        Instruction + question_text

    OUTPUT:
        solution_text
    """
    dataset: List[Dict[str, Any]] = []

    for ex in examples:
        question = ex.get("question_text") or ""
        solution = ex.get("solution_text") or ""

        if not question.strip() or not solution.strip():
            continue

        input_prompt = (
            "You are a physics tutor.\n"
            "Explain the solution to the following HC Verma example in clear, step-by-step form.\n\n"
            f"Question:\n{question}"
        )

        output_answer = solution.strip()

        rec = {
            "input": input_prompt,
            "output": output_answer,
            "metadata": {
                "example_id": ex.get("example_id"),
                "chapter_number": ex.get("chapter_number"),
                "chapter_title": ex.get("chapter_title"),
            },
        }
        dataset.append(rec)

    return dataset


# ---------------------------------------------------------------------
# DATASET 2: VARIANT GENERATION
# ---------------------------------------------------------------------

def build_variant_generation_dataset(
    examples: List[Dict[str, Any]],
    problems: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each end-of-chapter problem, pick a base worked example from the same chapter
    and train the model to generate a NEW question of similar type.

    We treat the problem question text as the "target variant".
    """
    dataset: List[Dict[str, Any]] = []

    # Group examples by chapter_number
    examples_by_ch: Dict[Any, List[Dict[str, Any]]] = {}
    for ex in examples:
        ch = ex.get("chapter_number")
        if ch is None:
            continue
        examples_by_ch.setdefault(ch, []).append(ex)

    # Round-robin cursor per chapter so we don't always pick the first example
    chapter_cursor: Dict[Any, int] = {ch: 0 for ch in examples_by_ch.keys()}

    for pb in problems:
        chapter_number = pb.get("chapter_number")
        if chapter_number is None:
            continue

        problem_q = pb.get("question_text") or pb.get("text") or ""
        if not problem_q.strip():
            continue

        ex_list = examples_by_ch.get(chapter_number)
        if not ex_list:
            # No example in this chapter → skip
            continue

        idx = chapter_cursor[chapter_number] % len(ex_list)
        chapter_cursor[chapter_number] += 1
        base_example = ex_list[idx]

        base_q = base_example.get("question_text") or ""
        if not base_q.strip():
            continue

        input_prompt = (
            "You are a physics question generator.\n"
            "Below is a worked example question from HC Verma.\n"
            "Using the same underlying physics concept and a similar level of difficulty,\n"
            "generate a NEW question with different numbers or a different real-life context.\n"
            "The new question should be self-contained and clearly stated.\n\n"
            f"Worked Example:\n{base_q}\n\n"
            "New question:"
        )

        output_question = problem_q.strip()

        rec = {
            "input": input_prompt,
            "output": output_question,
            "metadata": {
                "base_example_id": base_example.get("example_id"),
                "target_problem_id": pb.get("problem_id"),
                "chapter_number": chapter_number,
                "chapter_title": base_example.get("chapter_title"),
            },
        }
        dataset.append(rec)

    return dataset


# ---------------------------------------------------------------------
# DATASET 3: TEMPLATE EXTRACTION
# ---------------------------------------------------------------------

def build_template_extraction_dataset(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Train the model to extract a general physics TEMPLATE from a worked example.

    INPUT:
        - example question
        - example solution
        - instructions asking for a general template

    OUTPUT:
        - generalized template (for now we reuse solution_text as a proxy,
          and let the model learn abstraction via instructions)
    """

    dataset: List[Dict[str, Any]] = []

    for ex in examples:
        question = ex.get("question_text") or ""
        solution = ex.get("solution_text") or ""

        if not question.strip() or not solution.strip():
            continue

        input_prompt = (
            "Extract the underlying general physics TEMPLATE from the following worked HC Verma example.\n"
            "A template should:\n"
            "- Remove specific numbers\n"
            "- Use general variable names (m, v, θ, μ, etc.)\n"
            "- State the physics principles involved\n"
            "- Outline the sequence of equations\n"
            "- Describe the steps at a conceptual level\n\n"
            f"Example Question:\n{question}\n\n"
            f"Example Solution:\n{solution}\n\n"
            "Template:"
        )

        # For now we use the solution as the target; over SFT, the model learns
        # to respond in a more templated/abstract way because of the instructions.
        output_template = solution.strip()

        rec = {
            "input": input_prompt,
            "output": output_template,
            "metadata": {
                "example_id": ex.get("example_id"),
                "chapter_number": ex.get("chapter_number"),
                "chapter_title": ex.get("chapter_title"),
            },
        }
        dataset.append(rec)

    return dataset


# ---------------------------------------------------------------------
# DATASET 4: TOOL / EQUATION TAGGING
# ---------------------------------------------------------------------

def build_tool_tagging_dataset(
    examples: List[Dict[str, Any]],
    equations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Train the model to identify which equation_ids are relevant for a given example.

    We use the 'examples' field in equations.jsonl (as produced by extract_equations.py)
    which currently lists example_ids at chapter-level granularity.

    For each example, we collect all equations whose 'examples' list contains that
    example's ID, and use their equation_ids as the target.

    INPUT:
        - example question
        - instruction to list relevant equations

    OUTPUT:
        - equation_ids string (comma-separated) for that example
    """
    dataset: List[Dict[str, Any]] = []

    # Build mapping: example_id -> [equation_id, ...]
    eqs_by_example: Dict[str, List[str]] = {}
    for eq in equations:
        eq_id = eq.get("equation_id")
        if not eq_id:
            continue

        for ex_id in eq.get("examples", []):
            if not ex_id:
                continue
            eqs_by_example.setdefault(ex_id, []).append(eq_id)

    for ex in examples:
        ex_id = ex.get("example_id")
        if not ex_id:
            continue

        question = ex.get("question_text") or ""
        if not question.strip():
            continue

        eq_ids = eqs_by_example.get(ex_id, [])
        if not eq_ids:
            # No mapped equations for this example → skip for now
            continue

        # Make IDs unique and stable
        unique_ids = sorted(set(eq_ids))

        input_prompt = (
            "You are a physics solver.\n"
            "Given the following HC Verma example question, list the IDs of the physics equations\n"
            "that are most relevant to solve it. Use only the equation_ids provided by the tool,\n"
            "and do not invent new ones.\n\n"
            f"Question:\n{question}\n\n"
            "Relevant equation_ids (comma-separated):"
        )

        output_str = ", ".join(unique_ids)

        rec = {
            "input": input_prompt,
            "output": output_str,
            "metadata": {
                "example_id": ex_id,
                "chapter_number": ex.get("chapter_number"),
                "chapter_title": ex.get("chapter_title"),
            },
        }
        dataset.append(rec)

    return dataset


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    if not EXAMPLES_JSONL.exists():
        raise FileNotFoundError(f"Examples file not found: {EXAMPLES_JSONL}")
    examples = load_jsonl(EXAMPLES_JSONL)
    print(f"[INFO] Loaded {len(examples)} examples.")

    # Dataset 1: solution explanation
    solution_explanation_ds = build_solution_explanation_dataset(examples)
    print(f"[INFO] Built solution_explanation dataset with {len(solution_explanation_ds)} entries.")
    save_jsonl(solution_explanation_ds, SFT_SOLUTION_EXPLANATION)
    print(f"[INFO] Saved -> {SFT_SOLUTION_EXPLANATION}")

    # Dataset 2: variant generation
    if PROBLEMS_JSONL.exists():
        problems = load_jsonl(PROBLEMS_JSONL)
        print(f"[INFO] Loaded {len(problems)} problems.")

        variant_generation_ds = build_variant_generation_dataset(examples, problems)
        print(f"[INFO] Built variant_generation dataset with {len(variant_generation_ds)} entries.")
        save_jsonl(variant_generation_ds, SFT_VARIANT_GENERATION)
        print(f"[INFO] Saved -> {SFT_VARIANT_GENERATION}")
    else:
        print(f"[WARN] Problems file not found: {PROBLEMS_JSONL}")
        print("[WARN] Skipping variant_generation dataset.")

    # Dataset 3: template extraction
    template_extraction_ds = build_template_extraction_dataset(examples)
    print(f"[INFO] Built template_extraction dataset with {len(template_extraction_ds)} entries.")
    save_jsonl(template_extraction_ds, SFT_TEMPLATE_EXTRACTION)
    print(f"[INFO] Saved -> {SFT_TEMPLATE_EXTRACTION}")

    # Dataset 4: tool / equation tagging
    if EQ_JSONL.exists():
        equations = load_jsonl(EQ_JSONL)
        print(f"[INFO] Loaded {len(equations)} equations.")

        tool_tagging_ds = build_tool_tagging_dataset(examples, equations)
        print(f"[INFO] Built tool_tagging dataset with {len(tool_tagging_ds)} entries.")
        save_jsonl(tool_tagging_ds, SFT_TOOL_TAGGING)
        print(f"[INFO] Saved -> {SFT_TOOL_TAGGING}")
    else:
        print(f"[WARN] Equations file not found: {EQ_JSONL}")
        print("[WARN] Skipping tool_tagging dataset.")


if __name__ == "__main__":
    main()
