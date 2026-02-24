from pathlib import Path
import re, json

import argparse
import yaml

ROMAN = {"i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,"ix":9,"x":10}
WORDS = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}

def part_token_to_int(tok: str):
    if tok is None:
        return None
    t = tok.strip().lower()
    if t in WORDS:
        return WORDS[t]
    if t in ROMAN:
        return ROMAN[t]
    return None

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_raw", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--book_id", required=True)
    ap.add_argument("--book_spec", required=True, help="configs/books/<book>.yaml")
    return ap.parse_args()

def load_raw_text(path: Path):
    """Read the raw text file and return as a string."""
    print(f"[DEBUG] Looking for raw text at: {Path(path).resolve()}")
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def parse_pages(raw_text):
    """
    Split raw_text into a list of {'page': int, 'text': str}.
    Relies on the hard marker '--- Page N ---' written by preprocess_pdf.py.
    Why: page-awareness lets us compute accurate chapter page spans for downstream linking.
    """
    pages = []
    # Split on the marker while keeping page numbers
    chunks = re.split(r"^-{3}\s*Page\s+(\d+)\s*-{3}\s*$", raw_text, flags=re.MULTILINE)
    # chunks looks like: ["", "1", "text...", "2", "text...", ...]
    for i in range(1, len(chunks), 2):
        page_num = int(chunks[i])
        page_text = chunks[i+1].strip()
        pages.append({"page": page_num, "text": page_text})
    return pages


def split_into_chapters(raw_text, chapter_heading_regex: str):
    """
    Walk pages, detect chapter headings, and build structured chapters with page spans.
    Heading pattern is intentionally tolerant (e.g., 'Chapter 1', 'CHAPTER 1 ...').
    Why: HC Verma formatting can vary between editions or OCR artifacts.
    """
    pages = parse_pages(raw_text)
    #chapter_heading_re = re.compile(r"(?im)^\s*chapter\s+(\d+)[^\n]*$", re.MULTILINE)
    # Only match clean chapter-heading lines like "CHAPTER 5" or "Chapter 12"
    # - ^ ... $  : the whole line must be just "chapter <number>"
    # - (?mi)    : multiline, case-insensitive (handles CHAPTER / Chapter)
    #chapter_heading_re = re.compile(r"(?mi)^\s*chapter\s+(\d+)\s*$")
    # chapter_heading_regex is read from the book_spec YAML
    chapter_heading_re = re.compile(chapter_heading_regex)



    chapters = []
    current = None

    for p in pages:
        text = p["text"]
        page_no = p["page"]

        # Look for a chapter heading anywhere on this page
        m = chapter_heading_re.search(text)
        if m:
            # If we were already collecting a chapter, close its span
            if current is not None:
                current["page_span"][1] = page_no - 1
                chapters.append(current)

            # Try to extract chapter number safely
            chap_num = None
            if m.lastindex is not None:
                # If there are capture groups
                if m.lastindex >= 2:
                    chap_num = m.group(2)
                elif m.lastindex == 1:
                    chap_num = m.group(1)

            try:
                chap_num = int(chap_num) if chap_num is not None else None
            except:
                chap_num = None
            # Start a new chapter record
            # Determine raw heading text (e.g., "PART ONE" or "CHAPTER 3")
            heading_text = m.group(0).strip()

            # Extract token safely (group 1 usually holds number or word)
            chap_token = None
            if m.lastindex is not None and m.lastindex >= 1:
                chap_token = m.group(1)

            chap_num = part_token_to_int(chap_token)

            current = {
                "chapter_number": chap_num,
                "chapter_title": heading_text,
                "page_span": [page_no, page_no],
                "content": text,
                "sections": [],
                "examples": [],
                "problems": []
            }
        else:
            # No chapter heading on this page — if we're inside a chapter, append content
            if current is not None:
                current["content"] += f"\n\n{('--- page ' + str(page_no) + ' ---')}\n\n" + text

    # Close the last open chapter at EOF
    if current is not None:
        current["page_span"][1] = pages[-1]["page"] if pages else 0
        chapters.append(current)

    # Deduplicate by chapter_title instead of chapter_number
    by_title = {}

    for ch in chapters:
        title = ch["chapter_title"]
        span = ch.get("page_span", [0, 0])
        width = span[1] - span[0]

        if title not in by_title:
            by_title[title] = ch
        else:
            prev = by_title[title]
            prev_span = prev.get("page_span", [0, 0])
            prev_width = prev_span[1] - prev_span[0]
            if width > prev_width:
                by_title[title] = ch

    deduped_chapters = list(by_title.values())

    # Sort by page start (preserves natural order for PART ONE, TWO, etc.)
    deduped_chapters.sort(key=lambda x: x["page_span"][0])

    return deduped_chapters


def save_outputs(chapters, out_json: Path, out_md: Path):
    """Save structured data as JSON and Markdown."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(chapters, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_md.open("w", encoding="utf-8") as f:
        for ch in chapters:
            title = ch.get("chapter_title", "Chapter")
            span = ch.get("page_span", [None, None])
            f.write(f"# {title}  (pages {span[0]}–{span[1]})\n\n{ch['content']}\n\n")


if __name__ == "__main__":
    args = parse_args()

    spec = yaml.safe_load(Path(args.book_spec).read_text(encoding="utf-8"))
    # For Irodov, you can keep this empty or set a permissive regex.
    chapter_heading_regex = spec.get("chapter_heading_regex", r"(?im)^\s*chapter\s+(\d+)[^\n]*$")

    print("[INFO] Loading raw text...")
    raw_text = load_raw_text(Path(args.in_raw))

    print("[INFO] Splitting into chapters...")
    chapters = split_into_chapters(raw_text, chapter_heading_regex)

    print(f"[INFO] Found {len(chapters)} chapters.")
    save_outputs(chapters, Path(args.out_json), Path(args.out_md))
    print(f"[INFO] Structured data saved to {args.out_json} and {args.out_md}")
