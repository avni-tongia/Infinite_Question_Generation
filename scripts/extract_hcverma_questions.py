"""
extract_hcverma_questions.py

Extracts exercise problems from HC Verma structured.json
and outputs unified schema matching Irodov extractor.

Output:
data/processed/hcverma/problems.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from hashlib import sha1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structured_json", required=True,
                    help="data/processed/hcverma/structured.json")
    ap.add_argument("--out_jsonl", required=True,
                    help="data/processed/hcverma/problems.jsonl")
    ap.add_argument("--book_id", default="hcverma")
    ap.add_argument("--bucket_gt", default="easy")
    ap.add_argument("--min_len_chars", type=int, default=80)
    return ap.parse_args()


def needs_figure(text: str) -> bool:
    return re.search(r"(?i)\b(fig\.|figure)\b", text) is not None


def detect_questions_from_chapter_text(text: str):
    """
    Detect exercise problems numbered like:
      1. ...
      2. ...
    while avoiding:
      - numbered steps inside solutions
      - example numbering
      - short list fragments

    Heuristic: question start line must have enough payload after "N."
    """
    pattern = re.compile(r"(?m)^\s*(\d+)\.\s+(.*)$")
    matches = list(pattern.finditer(text))

    starts = []
    for m in matches:
        qnum = m.group(1).strip()
        first_line_rest = (m.group(2) or "").strip()

        # filters to reduce false positives
        if len(first_line_rest) < 20:
            continue
        if re.match(r"(?i)^(example|solution|proof)\b", first_line_rest):
            continue
        if re.match(r"(?i)^(hints?|answers?)\b", first_line_rest):
            continue

        starts.append(m)

    questions = []
    for i, m in enumerate(starts):
        start = m.start()
        end = starts[i+1].start() if i+1 < len(starts) else len(text)
        qtext = text[start:end].strip()
        qnum = m.group(1).strip()
        questions.append((qnum, qtext))

    return questions


def main():
    args = parse_args()

    structured = json.loads(
        Path(args.structured_json).read_text(encoding="utf-8", errors="ignore")
    )

    if not isinstance(structured, list):
        raise ValueError("structured.json expected to be a list of chapter dicts")

    rows = []

    for chapter in structured:
        chapter_num = chapter.get("chapter_number")
        chapter_title = chapter.get("chapter_title")
        content = chapter.get("content", "")

        if not content:
            continue

        questions = detect_questions_from_chapter_text(content)

        for qnum, qtext in questions:
            if len(qtext) < args.min_len_chars:
                continue

            text_hash = sha1(qtext.encode("utf-8", errors="ignore")).hexdigest()[:10]
            qid = f"{args.book_id}_ch{chapter_num}_q{qnum}_{text_hash}"

            rows.append({
                "id": qid,
                "source_book": args.book_id,
                "bucket_gt": args.bucket_gt,
                "part": None,
                "part_title": None,
                "chapter": chapter_num,
                "qnum": qnum,
                "page": None,
                "needs_figure": needs_figure(qtext),
                "problem_text": qtext
            })

    # Deduplicate by text hash
    seen = set()
    deduped = []

    for r in rows:
        h = sha1(r["problem_text"].encode("utf-8", errors="ignore")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        deduped.append(r)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Metrics ----
    lengths = [len(r["problem_text"]) for r in deduped]
    fig_cnt = sum(1 for r in deduped if r["needs_figure"])
    num_cnt = sum(1 for r in deduped if re.search(r"\d", r["problem_text"]) is not None)

    def pct(x, n):
        return 0.0 if n == 0 else 100.0 * x / n

    print(f"[OK] wrote {out_path}")
    print(f"[METRIC] extracted_questions_raw = {len(rows)}")
    print(f"[METRIC] deduped_questions      = {len(deduped)}")
    print(f"[METRIC] needs_figure_rate      = {pct(fig_cnt, len(deduped)):.2f}%")
    print(f"[METRIC] has_number_rate        = {pct(num_cnt, len(deduped)):.2f}%")

    if lengths:
        s = sorted(lengths)
        print(f"[METRIC] len_chars_p50         = {s[len(s)//2]}")
        print(f"[METRIC] len_chars_p90         = {s[int(0.9*len(s))-1]}")


if __name__ == "__main__":
    main()