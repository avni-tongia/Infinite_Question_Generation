"""
Compute difficulty-control metrics from a scored generation JSONL.

Metrics (v0):
- Target Hit Rate (THR): predicted bucket == target_level
- Monotonicity Triplet Accuracy (MTA):
  For each concept, compare mean score across easy/medium/hard targets:
  mean(hard) > mean(medium) > mean(easy)
"""

import json
from collections import defaultdict

#raise SystemExit("EVAL FILE DEFINITELY STARTED")

IN_PATH = "runs/baseline_sft/gen_sft_baseline_scored.jsonl"


def main():
    data = []
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_meta" in obj:
                continue
            data.append(obj)

    # --- Target Hit Rate ---
    total = 0
    hit = 0
    for r in data:
        total += 1
        if r["difficulty_bucket_v0"] == r["target_level"]:
            hit += 1
    thr = hit / total if total else 0.0

    # --- Monotonicity (concept-level) ---
    by_concept_level = defaultdict(list)
    for r in data:
        by_concept_level[(r["concept"], r["target_level"])].append(r["difficulty_score_v0"])

    mono_total = 0
    mono_ok = 0
    concepts = sorted(set(r["concept"] for r in data))
    for c in concepts:
        e = by_concept_level.get((c, "easy"), [])
        m = by_concept_level.get((c, "medium"), [])
        h = by_concept_level.get((c, "hard"), [])
        if not (e and m and h):
            continue

        mean_e = sum(e) / len(e)
        mean_m = sum(m) / len(m)
        mean_h = sum(h) / len(h)

        mono_total += 1
        if (mean_h > mean_m) and (mean_m > mean_e):
            mono_ok += 1

    mta = mono_ok / mono_total if mono_total else 0.0

    print("=== Difficulty Eval (v0) ===")
    print(f"Records: {len(data)}")
    print(f"Target Hit Rate (THR): {thr:.3f}")
    print(f"Monotonicity Triplet Accuracy (MTA): {mta:.3f}  (concepts evaluated: {mono_total})")


if __name__ == "__main__":
    main()
