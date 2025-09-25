
"""
extract_figures.py
Extracts figures (raster images) from pages and pairs them with approximate captions from nearby text.
Outputs:
- data/figures/fig_*.png
- data/figures/figures.jsonl
"""
import json
from pathlib import Path
from typing import Dict, List
import fitz  # PyMuPDF

RAW_TEXT = "data/hcverma_raw.txt"
PDF_PATH = "data/HC_Verma.pdf"
PAGES_DIR = Path("data/pages")
FIG_DIR = Path("data/figures")
OUT_JSONL = FIG_DIR / "figures.jsonl"

def load_page_texts() -> Dict[int, str]:
    """Return mapping page_number -> text (from RAW_TEXT with markers)."""
    page_map = {}
    curr_page = None
    buf = []
    with open(RAW_TEXT, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("--- Page "):
                if curr_page is not None:
                    page_map[curr_page] = "".join(buf).strip()
                # new page
                try:
                    curr_page = int(line.split()[2])
                except Exception:
                    curr_page = None
                buf = []
            else:
                if curr_page is not None:
                    buf.append(line)
    if curr_page is not None:
        page_map[curr_page] = "".join(buf).strip()
    return page_map

def caption_guess(page_text: str) -> str:
    # Very light heuristic: look for lines containing 'Fig'/'Figure'
    for ln in page_text.splitlines():
        if "Figure" in ln or "Fig." in ln or "Fig " in ln:
            return ln.strip()[:200]
    # fallback: first non-empty sentence
    for ln in page_text.splitlines():
        if ln.strip():
            return ln.strip()[:200]
    return ""

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    page_texts = load_page_texts()
    doc = fitz.open(PDF_PATH)
    out = []

    try:
        fig_idx = 1
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            page_num = pno + 1
            # Iterate image list on this page
            for img in page.get_images(full=True):
                xref = img[0]
                # Extract image to a pixmap
                pix = fitz.Pixmap(doc, xref)
                # Save
                out_path = FIG_DIR / f"fig_{fig_idx:04d}.png"
                if pix.n < 5:  # GRAY or RGB
                    pix.save(out_path.as_posix())
                else:
                    # CMYK: convert to RGB first
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.save(out_path.as_posix())

                # Simple caption guess from page text
                cap = caption_guess(page_texts.get(page_num, ""))
                out.append({
                    "figure_id": f"fig_{fig_idx:04d}",
                    "page": page_num,
                    "image_path": out_path.as_posix(),
                    "bbox": None,  # PyMuPDF doesn't give bbox per image via get_images; would need xobject parsing
                    "caption_guess": cap,
                    "chapter_guess": None
                })
                fig_idx += 1
    finally:
        doc.close()

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    print(f"[INFO] Extracted {len(out)} figures -> {OUT_JSONL}")

if __name__ == "__main__":
    main()
