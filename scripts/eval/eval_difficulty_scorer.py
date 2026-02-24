import argparse, json, math
from pathlib import Path
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scorer", required=True)
    return ap.parse_args()

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def auc_roc(scores, labels):
    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = 0.0
    for i, (_, y) in enumerate(paired, start=1):
        if y == 1:
            rank_sum_pos += i
    U = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return U / (n_pos * n_neg)

def best_threshold_by_val(val_scores, val_labels):
    uniq = sorted(set(val_scores))
    if len(uniq) == 1:
        return uniq[0], 0.5

    candidates = []
    for i in range(len(uniq)-1):
        candidates.append((uniq[i] + uniq[i+1]) / 2.0)
    candidates = [uniq[0] - 1e-9] + candidates + [uniq[-1] + 1e-9]

    best_t, best_acc = None, -1
    for t in candidates:
        preds = [1 if s >= t else 0 for s in val_scores]
        acc = sum(int(p == y) for p, y in zip(preds, val_labels)) / len(val_labels)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, best_acc

def effect_size(scores_easy, scores_hard):
    if not scores_easy or not scores_hard:
        return float("nan")
    mu1 = sum(scores_hard)/len(scores_hard)
    mu0 = sum(scores_easy)/len(scores_easy)
    var1 = sum((x-mu1)**2 for x in scores_hard)/(len(scores_hard)-1)
    var0 = sum((x-mu0)**2 for x in scores_easy)/(len(scores_easy)-1)
    sp = math.sqrt(((len(scores_hard)-1)*var1 + (len(scores_easy)-1)*var0) /
                   (len(scores_hard)+len(scores_easy)-2))
    return (mu1 - mu0) / sp if sp > 0 else float("nan")

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mod_path, fn_name = args.scorer.split(":")
    mod = __import__(mod_path, fromlist=[fn_name])
    score_fn = getattr(mod, fn_name)

    val_rows = load_jsonl(args.val)
    test_rows = load_jsonl(args.test)

    def run(rows):
        scores, labels = [], []
        for r in rows:
            s = float(score_fn(r))
            scores.append(s)
            labels.append(1 if r["bucket_gt"] == "hard" else 0)
        return scores, labels

    val_scores, val_labels = run(val_rows)
    test_scores, test_labels = run(test_rows)

    val_auc = auc_roc(val_scores, val_labels)
    test_auc = auc_roc(test_scores, test_labels)

    t, val_best_acc = best_threshold_by_val(val_scores, val_labels)
    test_preds = [1 if s >= t else 0 for s in test_scores]
    test_acc = sum(int(p == y) for p,y in zip(test_preds, test_labels)) / len(test_labels)

    easy_scores = [s for s,y in zip(test_scores, test_labels) if y == 0]
    hard_scores = [s for s,y in zip(test_scores, test_labels) if y == 1]

    metrics = {
        "threshold": t,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "val_best_acc": val_best_acc,
        "test_acc": test_acc,
        "mean_easy": sum(easy_scores)/len(easy_scores),
        "mean_hard": sum(hard_scores)/len(hard_scores),
        "cohen_d": effect_size(easy_scores, hard_scores)
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()