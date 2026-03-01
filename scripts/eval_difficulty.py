"""
Evaluate difficulty-control and preference metrics from a scored generation JSONL.

Metrics:
- Target Hit Rate (THR): predicted bucket == target_level
- Monotonicity Triplet Accuracy (MTA): mean(hard) > mean(medium) > mean(easy)
- Parsimony Ratio (Srikedaar's Preference): ratio of equations in text vs solver steps.
- Bloat Rate: % of questions with redundant/unused information.
"""

import argparse
import json
import re
from collections import defaultdict


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate difficulty and preference metrics.")
    ap.add_argument(
        "--in_path",
        default="runs/constrained_sft/gen_constrained_scored_v0.jsonl",
        help="Path to scored JSONL file",
    )
    ap.add_argument(
        "--scorer",
        choices=["v0", "v1"],
        default="v0",
        help="Which scorer suffix to read from the JSONL keys (v0 or v1)",
    )
    return ap.parse_args()


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def calculate_parsimony_score(text: str) -> float:
    """
    Srikedaar's Preference Metric:
    Calculates a structural complexity score based on equation density.
    """
    # Find all equation-like patterns (lines with '=' or LaTeX markers)
    eqs = re.findall(r'=', text)
    # We normalize by character count to get a 'density' metric
    return len(eqs) / (len(text) / 100) if len(text) > 0 else 0.0


def main():
    args = parse_args()
    in_path = args.in_path
    scorer = args.scorer

    score_key = f"difficulty_score_{scorer}"
    bucket_key = f"difficulty_bucket_{scorer}"

    print(f"[INFO] Evaluating metrics from: {in_path}")

    data = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_meta" in obj:
                continue
            data.append(obj)

    # --- THR (Target Hit Rate) ---
    total = len(data)
    hit = 0
    for r in data:
        if r.get(bucket_key) == r.get("target_level"):
            hit += 1
    thr = hit / total if total else 0.0

    # --- Srikedaar's Preference Metrics (Parsimony & Bloat) ---
    parsimony_scores = []
    bloated_count = 0
    
    for r in data:
        q_text = r.get("chosen_question", "")
        p_score = calculate_parsimony_score(q_text)
        parsimony_scores.append(p_score)
        
        # A question is considered 'bloated' if its length is excessive (>650 chars)
        # while its equation density is low, indicating 'fluff' text.
        if len(q_text) > 650 and p_score < 0.5:
            bloated_count += 1
            
    avg_parsimony = mean(parsimony_scores)
    bloat_rate = bloated_count / total if total else 0.0

    # --- Monotonicity (MTA) ---
    by_concept_level = defaultdict(list)
    by_level_scores = defaultdict(list)

    for r in data:
        concept = r.get("concept", "UNKNOWN")
        level = r.get("target_level", "UNKNOWN")
        s = r.get(score_key, None)
        if s is not None:
            by_concept_level[(concept, level)].append(float(s))
            by_level_scores[level].append(float(s))

    concepts = sorted(set(r.get("concept", "UNKNOWN") for r in data))
    mono_total = 0
    mono_ok = 0

    for c in concepts:
        e = by_concept_level.get((c, "easy"), [])
        m = by_concept_level.get((c, "medium"), [])
        h = by_concept_level.get((c, "hard"), [])
        if e and m and h:
            mono_total += 1
            if (mean(h) > mean(m)) and (mean(m) > mean(e)):
                mono_ok += 1

    mta = mono_ok / mono_total if mono_total else 0.0

    # --- Final Output ---
    print("\n=== Difficulty Control (Avni's Metrics) ===")
    print(f"Target Hit Rate (THR): {thr:.3f}")
    print(f"Monotonicity (MTA):    {mta:.3f}")

    print("\n=== Preference Constraints (Srikedaar's Metrics) ===")
    print(f"Average Parsimony Score: {avg_parsimony:.3f} (Ideal: 0.8 - 1.2)")
    print(f"Structural Bloat Rate:   {bloat_rate:.2%} (Lower is better)")
    
    print("\n--- Diagnostic Means ---")
    print(f"mean(score | easy):   {mean(by_level_scores.get('easy', [])):.3f}")
    print(f"mean(score | hard):   {mean(by_level_scores.get('hard', [])):.3f}")


if __name__ == "__main__":
    main()