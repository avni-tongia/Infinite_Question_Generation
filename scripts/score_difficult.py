"""
Score difficulty for a JSONL generation file.

Reads:
- --in_path runs/.../gen_*.jsonl

Writes:
- --out_path runs/.../gen_*_scored.jsonl

Augments each record with:
- difficulty_features_v0
- difficulty_score_v0
- difficulty_bucket_v0
"""

import argparse
import json
import os

from difficulty.scorer import score_question


def parse_args():
    ap = argparse.ArgumentParser(description="Score difficulty for a generation JSONL file.")
    ap.add_argument(
        "--in_path",
        default="runs/baseline_sft/gen_sft_baseline.jsonl",
        help="Input JSONL path (default: baseline SFT file)",
    )
    ap.add_argument(
        "--out_path",
        default="runs/baseline_sft/gen_sft_baseline_scored.jsonl",
        help="Output JSONL path (default: baseline scored file)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    in_path = args.in_path
    out_path = args.out_path

    print(f"[INFO] Scoring: {in_path} -> {out_path}")

    assert os.path.exists(in_path), f"Input not found: {in_path}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            obj = json.loads(line)

            # Keep meta header untouched
            if "_meta" in obj:
                f_out.write(json.dumps(obj) + "\n")
                continue

            # Support both baseline format ("question") and constrained format ("chosen_question")
            q = (obj.get("question") or obj.get("chosen_question") or "").strip()

            feats, s, b = score_question(q)

            obj["difficulty_features_v0"] = feats
            obj["difficulty_score_v0"] = float(s)
            obj["difficulty_bucket_v0"] = b

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote scored file: {out_path}")


if __name__ == "__main__":
    main()
