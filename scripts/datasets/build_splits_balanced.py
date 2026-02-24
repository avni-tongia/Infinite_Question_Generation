import argparse, json, random
from pathlib import Path
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--buckets", nargs="+", required=True)  # easy hard
    ap.add_argument("--split", nargs=3, type=float, default=[0.85,0.10,0.05])
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def bucket_counts(rows):
    d = defaultdict(int)
    for r in rows:
        d[r["bucket_gt"]] += 1
    return dict(d)

def main():
    args = parse_args()
    random.seed(args.seed)

    rows = [json.loads(l) for l in open(args.in_path, "r", encoding="utf-8")]
    by_bucket = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket_gt"]].append(r)

    counts = {b: len(by_bucket[b]) for b in args.buckets}
    min_count = min(counts.values())

    # Sample equal count per bucket
    sampled = {}
    for b in args.buckets:
        random.shuffle(by_bucket[b])
        sampled[b] = by_bucket[b][:min_count]

    # Split per bucket to preserve balance in each split
    split = args.split
    if abs(sum(split) - 1.0) > 1e-6:
        raise ValueError(f"--split must sum to 1.0, got {split}")

    n = min_count
    n_train = int(split[0] * n)
    n_val = int(split[1] * n)
    # rest is test
    n_test = n - n_train - n_val

    train, val, test = [], [], []
    for b in args.buckets:
        items = sampled[b]
        train += items[:n_train]
        val   += items[n_train:n_train+n_val]
        test  += items[n_train+n_val:n_train+n_val+n_test]

    # Shuffle within each split for training randomness
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)

    print(f"[OK] wrote splits to {out_dir}")
    print(f"[METRIC] bucket_counts_total={counts}")
    print(f"[METRIC] min_count_used={min_count}")
    print(f"[METRIC] per_bucket_n={{'train':{n_train}, 'val':{n_val}, 'test':{n_test}}}")
    print(f"[METRIC] train_counts={bucket_counts(train)}")
    print(f"[METRIC] val_counts={bucket_counts(val)}")
    print(f"[METRIC] test_counts={bucket_counts(test)}")

if __name__ == "__main__":
    main()