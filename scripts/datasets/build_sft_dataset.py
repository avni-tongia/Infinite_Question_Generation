import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def make_prompt(bucket: str, style: str = "minimal") -> str:
    """
    style="minimal": purely difficulty conditioning
    style="exam": slightly more structured (often helps instruction models)
    """
    bucket = bucket.upper()
    if style == "exam":
        return (
            f"[DIFFICULTY={bucket}]\n"
            f"Task: Write ONE physics problem matching the requested difficulty.\n"
            f"Rules:\n"
            f"- Output ONLY the problem statement.\n"
            f"- Do NOT provide a solution.\n"
            f"- Include numbers/units where appropriate.\n"
        )
    # minimal default
    return (
        f"[DIFFICULTY={bucket}]\n"
        f"Generate a physics problem.\n"
        f"Return ONLY the problem statement (no solution, no hints)."
    )


def build_sft_example(ex: Dict[str, Any], prompt_style: str) -> Dict[str, Any]:
    bucket = ex["bucket_gt"].upper()
    prompt = make_prompt(bucket=bucket, style=prompt_style)
    completion = normalize_ws(ex["problem_text"])

    # Important: keep completion as the raw problem statement only.
    # If extraction included any leading "Q." markers, keep them (consistent with data).
    return {
        "prompt": prompt,
        "completion": completion,
        "meta": {
            "id": ex.get("id"),
            "bucket_gt": ex.get("bucket_gt"),
            "source_book": ex.get("source_book"),
            "chapter": ex.get("chapter"),
            "needs_figure": ex.get("needs_figure", False),
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_train", required=True)
    ap.add_argument("--in_val", required=True)
    ap.add_argument("--in_test", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompt_style", default="minimal", choices=["minimal", "exam"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def convert(inp: str, outname: str):
        inp_path = Path(inp)
        out_path = out_dir / outname

        rows_out = []
        n = 0
        for ex in read_jsonl(inp_path):
            # Minimal validation
            if "problem_text" not in ex or "bucket_gt" not in ex:
                raise ValueError(f"Missing required keys in example: {ex.keys()}")

            rows_out.append(build_sft_example(ex, args.prompt_style))
            n += 1

        write_jsonl(rows_out, out_path)
        print(f"[OK] wrote {out_path}  (n={n})")

    convert(args.in_train, "train_sft.jsonl")
    convert(args.in_val, "val_sft.jsonl")
    convert(args.in_test, "test_sft.jsonl")


if __name__ == "__main__":
    main()