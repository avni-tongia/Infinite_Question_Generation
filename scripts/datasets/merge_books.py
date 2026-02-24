import argparse
import json
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="List of input JSONL files")
    ap.add_argument("--out", required=True,
                    help="Output merged JSONL file")
    return ap.parse_args()


def main():
    args = parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    total_in = 0
    total_out = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for inp in args.inputs:
            inp_path = Path(inp)
            if not inp_path.exists():
                raise FileNotFoundError(f"{inp} not found")

            with inp_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    total_in += 1
                    ex = json.loads(line)

                    _id = ex.get("id")
                    if _id in seen_ids:
                        continue

                    seen_ids.add(_id)
                    fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    total_out += 1

    print(f"[OK] wrote {out_path}")
    print(f"[METRIC] input_rows={total_in}")
    print(f"[METRIC] output_rows={total_out}")
    print(f"[METRIC] duplicates_dropped={total_in - total_out}")


if __name__ == "__main__":
    main()