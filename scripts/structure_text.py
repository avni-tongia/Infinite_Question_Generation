import re
import json

# Input and output paths
INPUT_FILE = "data/hcverma_raw.txt"
OUTPUT_JSON = "data/hcverma_structured.json"
OUTPUT_MD = "data/hcverma_structured.md"

def load_raw_text(path):
    """Read the raw text file and return as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_into_chapters(raw_text):
    """
    Splits raw text into chapters based on the keyword 'Chapter'.
    Adjust the regex if chapter headings look different in HC Verma.
    """
    chapters = re.split(r"(?:^|\n)(Chapter\s+\d+.*)", raw_text)
    structured = []
    
    for i in range(1, len(chapters), 2):
        title = chapters[i].strip()
        content = chapters[i+1].strip()
        structured.append({
            "chapter_title": title,
            "content": content,
            "sections": []  # sections can be extracted later
        })
    return structured

def save_outputs(chapters):
    """Save structured data as JSON and Markdown."""
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chapters, f, indent=2)

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        for ch in chapters:
            f.write(f"# {ch['chapter_title']}\n\n{ch['content']}\n\n")

if __name__ == "__main__":
    print("[INFO] Loading raw text...")
    raw_text = load_raw_text(INPUT_FILE)

    print("[INFO] Splitting into chapters...")
    chapters = split_into_chapters(raw_text)

    print(f"[INFO] Found {len(chapters)} chapters.")
    save_outputs(chapters)
    print(f"[INFO] Structured data saved to {OUTPUT_JSON} and {OUTPUT_MD}")
