"""
Score difficulty for a JSONL generation file.

Reads:
- runs/.../gen_*.jsonl

Writes:
- same records, augmented with:
  difficulty_features, difficulty_score, predicted_bucket
"""

import json
import os
from typing import Optional

from difficulty.scorer import score_question


IN_PATH = "runs/baseline_sft/gen_sft_baseline.jsonl"
OUT_PATH = "runs/baseline_sft/gen_sft_baseline_scored.jsonl"

print(f"[INFO] Scoring: {IN_PATH} -> {OUT_PATH}")

def main():
    assert os.path.exists(IN_PATH), f"Input not found: {IN_PATH}"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(IN_PATH, "r", encoding="utf-8") as f_in, open(OUT_PATH, "w", encoding="utf-8") as f_out:
        for line in f_in:
            obj = json.loads(line)

            # Keep meta header untouched
            if "_meta" in obj:
                f_out.write(json.dumps(obj) + "\n")
                continue

            q = obj.get("question", "").strip()
            feats, s, b = score_question(q)

            obj["difficulty_features_v0"] = feats
            obj["difficulty_score_v0"] = s
            obj["difficulty_bucket_v0"] = b

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote scored file: {OUT_PATH}")


if __name__ == "__main__":
    main()
