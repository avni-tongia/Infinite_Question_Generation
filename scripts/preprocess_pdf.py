
"""
preprocess_pdf.py
- Extracts text from PDF with pdfplumber.
- Falls back to OCR if a page has no selectable text.
- Renders and saves each page as a PNG for downstream figure/equation/table extraction.
- Writes a page-level log for provenance.
Outputs:
- data/hcverma_raw.txt
- data/page_log.jsonl
- data/pages/page_####.png
"""
import json
from pathlib import Path
import pdfplumber
from ocr_utils import render_page_to_image, ocr_image_to_text

PDF_PATH = "data/HC_Verma.pdf"
RAW_TEXT_OUT = "data/hcverma_raw.txt"
PAGE_LOG = "data/page_log.jsonl"
PAGES_DIR = "data/pages"

def main():
    Path(PAGES_DIR).mkdir(parents=True, exist_ok=True)
    collected = []

    with pdfplumber.open(PDF_PATH) as pdf, open(RAW_TEXT_OUT, "w", encoding="utf-8") as raw_f, open(PAGE_LOG, "w", encoding="utf-8") as log_f:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text_source = "pdf_text"
            # Always render page image for downstream steps
            img_path = render_page_to_image(PDF_PATH, page_number, dpi=300, out_dir=PAGES_DIR)

            if not text.strip():
                # OCR fallback
                ocr_text = ocr_image_to_text(img_path, lang="eng", psm=6)
                if ocr_text.strip():
                    text = ocr_text
                    text_source = "ocr"
                else:
                    text = ""
                    text_source = "empty"

            # Write page marker + text
            if text:
                raw_f.write(f"--- Page {page_number} ---\n{text}\n\n")

            # Log page metadata
            log = {
                "page": page_number,
                "text_source": text_source,
                "image_path": img_path,
                "char_len": len(text)
            }
            log_f.write(json.dumps(log) + "\n")

    print(f"[SUCCESS] Wrote raw text to {RAW_TEXT_OUT} and page log to {PAGE_LOG}")

if __name__ == "__main__":
    main()
