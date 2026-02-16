"""
Evaluate difficulty-control metrics from a scored generation JSONL.

Metrics:
- Target Hit Rate (THR): predicted bucket == target_level
- Monotonicity Triplet Accuracy (MTA):
  For each concept: mean(hard) > mean(medium) > mean(easy)
- Mean score by target bucket (diagnostic)
- Separation: mean(hard) - mean(easy) (diagnostic)
"""

import argparse
import json
from collections import defaultdict


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate difficulty metrics from scored JSONL.")
    ap.add_argument(
        "--in_path",
        default="runs/baseline_sft/gen_sft_baseline_scored_v0.jsonl",
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


def main():
    args = parse_args()
    in_path = args.in_path
    scorer = args.scorer

    score_key = f"difficulty_score_{scorer}"
    bucket_key = f"difficulty_bucket_{scorer}"

    print(f"[INFO] Evaluating difficulty metrics from: {in_path}")
    print(f"[INFO] Using keys: {score_key}, {bucket_key}")

    data = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_meta" in obj:
                continue
            data.append(obj)

    # --- THR ---
    total = len(data)
    hit = 0
    for r in data:
        if r.get(bucket_key) == r.get("target_level"):
            hit += 1
    thr = hit / total if total else 0.0

    # --- group scores by (concept, target_level) ---
    by_concept_level = defaultdict(list)
    by_level_scores = defaultdict(list)

    for r in data:
        concept = r.get("concept", "UNKNOWN")
        level = r.get("target_level", "UNKNOWN")
        s = r.get(score_key, None)

        if s is None:
            continue
        by_concept_level[(concept, level)].append(float(s))
        by_level_scores[level].append(float(s))

    # --- MTA ---
    concepts = sorted(set(r.get("concept", "UNKNOWN") for r in data))
    mono_total = 0
    mono_ok = 0

    for c in concepts:
        e = by_concept_level.get((c, "easy"), [])
        m = by_concept_level.get((c, "medium"), [])
        h = by_concept_level.get((c, "hard"), [])
        if not (e and m and h):
            continue

        mean_e = mean(e)
        mean_m = mean(m)
        mean_h = mean(h)

        mono_total += 1
        if (mean_h > mean_m) and (mean_m > mean_e):
            mono_ok += 1

    mta = mono_ok / mono_total if mono_total else 0.0

    # --- Diagnostics ---
    mean_easy = mean(by_level_scores.get("easy", []))
    mean_med = mean(by_level_scores.get("medium", []))
    mean_hard = mean(by_level_scores.get("hard", []))
    separation = mean_hard - mean_easy

    print("\n=== Difficulty Eval ===")
    print(f"Records: {total}")
    print(f"Target Hit Rate (THR): {thr:.3f}")
    print(f"Monotonicity Triplet Accuracy (MTA): {mta:.3f}  (concepts evaluated: {mono_total})")

    print("\n--- Score diagnostics (means) ---")
    print(f"mean(score | easy):   {mean_easy:.3f}")
    print(f"mean(score | medium): {mean_med:.3f}")
    print(f"mean(score | hard):   {mean_hard:.3f}")
    print(f"separation (hard-easy): {separation:.3f}")


if __name__ == "__main__":
    main()