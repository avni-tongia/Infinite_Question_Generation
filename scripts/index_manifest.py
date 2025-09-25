
"""
index_manifest.py
Builds a global manifest that links chapters to page spans and to multimodal objects.
Inputs:
- data/hcverma_structured.json
- data/hcverma_raw.txt (with --- Page N --- markers)
- data/equations/equations.jsonl
- data/figures/figures.jsonl
- data/tables/tables.jsonl
Output:
- data/manifest.json
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any

STRUCT_JSON = "data/hcverma_structured.json"
RAW_TEXT = "data/hcverma_raw.txt"
EQ_JSONL = "data/equations/equations.jsonl"
FIG_JSONL = "data/figures/figures.jsonl"
TBL_JSONL = "data/tables/tables.jsonl"
OUT_PATH = "data/manifest.json"

PAGE_MARKER = re.compile(r"^--- Page (\d+) ---\s*$")

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: str):
    items = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
    except FileNotFoundError:
        pass
    return items

def find_chapter_page_spans(chapters, raw_text_path: str):
    # Strategy: find the page number where each chapter title first occurs; span ends before next chapter start
    page_texts = []
    curr_page = None
    buf = []
    with open(raw_text_path, "r", encoding="utf-8") as f:
        for line in f:
            m = PAGE_MARKER.match(line.strip())
            if m:
                if curr_page is not None:
                    page_texts.append((curr_page, "".join(buf)))
                curr_page = int(m.group(1))
                buf = []
            else:
                if curr_page is not None:
                    buf.append(line)
    if curr_page is not None:
        page_texts.append((curr_page, "".join(buf)))

    # Build page index for search
    chapter_starts = []
    for ch in chapters:
        title = ch.get("chapter_title","").strip()
        start_page = None
        for pno, ptxt in page_texts:
            if title and title in ptxt:
                start_page = pno
                break
        # fallback heuristic: first page that contains "Chapter X"
        if start_page is None:
            # try simple "Chapter N" prefix search
            m = re.match(r"(Chapter\s+\d+)", title)
            if m:
                key = m.group(1)
                for pno, ptxt in page_texts:
                    if key in ptxt:
                        start_page = pno
                        break
        chapter_starts.append(start_page or -1)

    page_spans = []
    for i, st in enumerate(chapter_starts):
        if st == -1:
            page_spans.append([None, None])
            continue
        # end at the page before next start
        if i < len(chapter_starts) - 1 and chapter_starts[i+1] not in (-1, None):
            end_page = chapter_starts[i+1] - 1
        else:
            # until the last page
            end_page = page_texts[-1][0] if page_texts else None
        page_spans.append([st, end_page])
    return page_spans

def bucket_by_page(objs):
    by_page = {}
    for o in objs:
        p = o.get("page")
        if p is None: 
            continue
        by_page.setdefault(p, []).append(o)
    return by_page

def main():
    chapters = load_json(STRUCT_JSON)
    eqs = load_jsonl(EQ_JSONL)
    figs = load_jsonl(FIG_JSONL)
    tbls = load_jsonl(TBL_JSONL)
    spans = find_chapter_page_spans(chapters, RAW_TEXT)

    # Build manifest
    manifest = {"chapters": [], "pages": {}}
    for i, ch in enumerate(chapters):
        span = spans[i] if i < len(spans) else [None, None]
        manifest["chapters"].append({
            "chapter_title": ch.get("chapter_title"),
            "page_span": span,
            "objects": {
                "equations": [],
                "figures": [],
                "tables": []
            }
        })

    # Bucket by page, then attach to chapters whose span covers that page
    eq_by_page = bucket_by_page(eqs)
    fig_by_page = bucket_by_page(figs)
    tbl_by_page = bucket_by_page(tbls)

    # Fill pages section
    all_pages = set(eq_by_page.keys()) | set(fig_by_page.keys()) | set(tbl_by_page.keys())
    for p in sorted(all_pages):
        manifest["pages"][str(p)] = {
            "equations": [e["equation_id"] for e in eq_by_page.get(p,[]) if "equation_id" in e],
            "figures": [f["figure_id"] for f in fig_by_page.get(p,[]) if "figure_id" in f],
            "tables": [t["table_id"] for t in tbl_by_page.get(p,[]) if "table_id" in t]
        }

    # Attach IDs under chapters
    def add_to_chapter(obj_id, page, kind):
        for i, ch in enumerate(manifest["chapters"]):
            span = ch["page_span"]
            if span[0] is None or span[1] is None:
                continue
            if span[0] <= page <= span[1]:
                ch["objects"][kind].append(obj_id)
                break

    for e in eqs:
        p = e.get("page")
        if p is None or "equation_id" not in e:
            continue
        add_to_chapter(e["equation_id"], p, "equations")

    for f in figs:
        p = f.get("page")
        if p is None or "figure_id" not in f:
            continue
        add_to_chapter(f["figure_id"], p, "figures")

    for t in tbls:
        p = t.get("page")
        if p is None or "table_id" not in t:
            continue
        add_to_chapter(t["table_id"], p, "tables")

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        json.dump(manifest, out_f, indent=2)

    print(f"[SUCCESS] Wrote manifest -> {OUT_PATH}")

if __name__ == "__main__":
    main()
