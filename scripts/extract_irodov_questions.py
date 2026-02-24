import argparse, json, re
from pathlib import Path
from hashlib import sha1

WORDS = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
ROMAN = {"i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,"ix":9,"x":10}

def part_token_to_int(tok: str):
    if not tok: return None
    t = tok.strip().lower()
    if t in WORDS: return WORDS[t]
    if t in ROMAN: return ROMAN[t]
    try: return int(t)
    except: return None

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structured_json", required=True, help="data/processed/irodov/structured.json")
    ap.add_argument("--out_jsonl", required=True, help="data/processed/irodov/problems.jsonl")
    ap.add_argument("--book_id", default="irodov")
    ap.add_argument("--bucket_gt", default="hard")
    ap.add_argument("--question_start_regex", default=r"(?m)^\s*(\d+\.\d+)\s+")
    ap.add_argument("--min_len_chars", type=int, default=80)
    return ap.parse_args()

def needs_figure(text: str) -> bool:
    return re.search(r"(?i)\b(fig\.|figure)\b", text) is not None

def main():
    args = parse_args()
    structured = json.loads(Path(args.structured_json).read_text(encoding="utf-8", errors="ignore"))

    # Case 1: structure_text.py returns a list of chapter dicts (your case)
    if isinstance(structured, list):
        chapters = structured

    # Case 2: structure is wrapped inside dict
    elif isinstance(structured, dict):
        chapters = structured.get("chapters") or structured.get("sections") or structured.get("parts") or []

    else:
        chapters = []

    qpat = re.compile(args.question_start_regex)

    rows = []
    for ch in chapters:
        title = ch.get("chapter_title", "UNKNOWN")
        content = ch.get("content", "")
        if not content:
            continue

        # detect PART number from title, if present
        part_num = None
        mpart = re.search(r"(?i)\bpart\s+([a-z0-9ivx]+)\b", title)
        if mpart:
            part_num = part_token_to_int(mpart.group(1))

        matches = list(qpat.finditer(content))
        if len(matches) == 0:
            continue

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(content)
            qtext = content[start:end].strip()
            qnum = m.group(1).strip()

            if len(qtext) < args.min_len_chars:
                continue

            # chapter for concept graph: prefix of qnum (e.g., 3.14 -> 3)
            try:
                chapter_num = int(qnum.split(".")[0])
            except:
                chapter_num = None

            
            from hashlib import sha1
            text_hash = sha1(qtext.encode("utf-8", errors="ignore")).hexdigest()[:10]
            qid = f"{args.book_id}_{qnum.replace('.', '_')}_{text_hash}"
            
            rows.append({
                "id": qid,
                "source_book": args.book_id,
                "bucket_gt": args.bucket_gt,
                "part": part_num,
                "part_title": title,
                "chapter": chapter_num,
                "qnum": qnum,
                "page": None,
                "needs_figure": needs_figure(qtext),
                "problem_text": qtext
            })

    # Deduplicate by problem_text hash (OCR repeats headers sometimes)
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

    # ---- metrics ----
    lens = [len(r["problem_text"]) for r in deduped]
    fig_cnt = sum(1 for r in deduped if r["needs_figure"])
    num_cnt = sum(1 for r in deduped if re.search(r"\d", r["problem_text"]) is not None)
    parts = {r["part_title"] for r in deduped}

    def pct(x, n): return 0.0 if n == 0 else 100.0 * x / n
    print(f"[OK] wrote {out_path}")
    print(f"[METRIC] extracted_questions_raw = {len(rows)}")
    print(f"[METRIC] deduped_questions      = {len(deduped)}")
    print(f"[METRIC] parts_covered          = {len(parts)}")
    print(f"[METRIC] needs_figure_rate      = {pct(fig_cnt, len(deduped)):.2f}%")
    print(f"[METRIC] has_number_rate        = {pct(num_cnt, len(deduped)):.2f}%")
    if lens:
        s = sorted(lens)
        print(f"[METRIC] len_chars_p50         = {s[len(s)//2]}")
        print(f"[METRIC] len_chars_p90         = {s[int(0.9*len(s))-1]}")

if __name__ == "__main__":
    main()