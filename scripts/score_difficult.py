"""
Score difficulty for a JSONL generation file.

Reads:
- --in_path runs/.../gen_*.jsonl

Writes:
- --out_path runs/.../gen_*_scored.jsonl

Adds keys:
- difficulty_features_<scorer>
- difficulty_score_<scorer>
- difficulty_bucket_<scorer>
"""

import argparse
import json
import os

from difficulty.scorer import score_question

def auto_out_path(in_path: str, scorer_name: str) -> str:
    """
    Convert:
      runs/.../gen_xxx.jsonl
    to:
      runs/.../gen_xxx_scored_<scorer>.jsonl
    """
    base, ext = os.path.splitext(in_path)
    return f"{base}_scored_{scorer_name}{ext or '.jsonl'}"

def parse_args():
    ap = argparse.ArgumentParser(description="Score difficulty for a generation JSONL file.")
    ap.add_argument(
        "--in_path",
        default="runs/baseline_sft/gen_sft_baseline.jsonl",
        help="Input JSONL path (default: baseline SFT file)",
    )
    ap.add_argument(
    "--out_path",
    default=None,
    help="Output JSONL path. If not provided, we auto-generate a scorer-specific path.",
    )
    ap.add_argument(
        "--scorer",
        choices=["v0", "v1"],
        default="v0",
        help="Which difficulty scorer to use (default: v0).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    in_path = args.in_path
    scorer_name = args.scorer  # <-- define FIRST

    out_path = args.out_path or auto_out_path(in_path, scorer_name)

    print(f"[INFO] Scoring ({scorer_name}): {in_path} -> {out_path}")

    assert os.path.exists(in_path), f"Input not found: {in_path}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    feat_key = f"difficulty_features_{scorer_name}"
    score_key = f"difficulty_score_{scorer_name}"
    bucket_key = f"difficulty_bucket_{scorer_name}"

    with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            obj = json.loads(line)

            if "_meta" in obj:
                f_out.write(json.dumps(obj) + "\n")
                continue

            q = (obj.get("question") or obj.get("chosen_question") or "").strip()

            feats, s, b = score_question(q, scorer=scorer_name)

            obj[feat_key] = feats
            obj[score_key] = float(s)
            obj[bucket_key] = b

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote scored file: {out_path}")


if __name__ == "__main__":
    main()
