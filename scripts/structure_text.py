from pathlib import Path
import re, json

ROOT_DIR = Path(__file__).resolve().parent
INPUT_FILE = ROOT_DIR / "data" / "hcverma_raw.txt"   # <-- Path, not str
OUTPUT_JSON = ROOT_DIR / "data" / "hcverma_structured.json"
OUTPUT_MD   = ROOT_DIR / "data" / "hcverma_structured.md"


def load_raw_text(path):
    """Read the raw text file and return as a string."""
    print(f"[DEBUG] Looking for raw text at: {INPUT_FILE.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

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


def split_into_chapters(raw_text):
    """
    Walk pages, detect chapter headings, and build structured chapters with page spans.
    Heading pattern is intentionally tolerant (e.g., 'Chapter 1', 'CHAPTER 1 ...').
    Why: HC Verma formatting can vary between editions or OCR artifacts.
    """
    pages = parse_pages(raw_text)
    chapter_heading_re = re.compile(r"(?im)^\s*chapter\s+(\d+)[^\n]*$", re.MULTILINE)

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

            chap_num = int(m.group(1))
            # Start a new chapter record
            current = {
                "chapter_number": chap_num,
                "chapter_title": f"Chapter {chap_num}",
                "page_span": [page_no, page_no],  # will extend as we go
                "content": text,                   # accumulate raw text for now
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

    return chapters


def save_outputs(chapters):
    """Save structured data as JSON and Markdown."""
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chapters, f, indent=2)

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        for ch in chapters:
            title = ch.get("chapter_title", "Chapter")
            span = ch.get("page_span", [None, None])
            f.write(f"# {title}  (pages {span[0]}–{span[1]})\n\n{ch['content']}\n\n")


if __name__ == "__main__":
    print("[INFO] Loading raw text...")
    raw_text = load_raw_text(INPUT_FILE)

    print("[INFO] Splitting into chapters...")
    chapters = split_into_chapters(raw_text)

    print(f"[INFO] Found {len(chapters)} chapters.")
    save_outputs(chapters)
    print(f"[INFO] Structured data saved to {OUTPUT_JSON} and {OUTPUT_MD}")
