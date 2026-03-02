import json
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--in_file", required=True)
ap.add_argument("--out_file", required=True)
args = ap.parse_args()

with open(args.in_file, "r", encoding="utf-8") as fin, \
     open(args.out_file, "w", encoding="utf-8") as fout:

    for line in fin:
        row = json.loads(line)
        new_row = {
            "text": row["text"],
            "bucket_gt": row["bucket_prompt"]
        }
        fout.write(json.dumps(new_row) + "\n")

print("[OK] Converted file for scorer")