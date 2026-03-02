import argparse
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

with open(args.out, "w", encoding="utf-8") as fout:
    for file in args.inputs:
        with open(file, "r", encoding="utf-8") as fin:
            shutil.copyfileobj(fin, fout)

print("[OK] Pure concatenation complete.")