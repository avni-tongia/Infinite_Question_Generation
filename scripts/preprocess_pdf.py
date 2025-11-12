"""preprocess_pdf.py
- Extracts text from HC Verma PDF with pdfplumber.
- Falls back to OCR if a page has no selectable text.
- Renders and saves each page as a PNG for downstream figure/equation/table extraction.
- Writes a page-level log for provenance.

Outputs:
- data/hcverma_raw.txt         # master raw text with page markers
- data/page_log.jsonl          # one JSON per page: source, image path, lengths
- data/pages/page_####.png     # rendered page images
"""

import json
from pathlib import Path

import pdfplumber

from ocr_utils import render_page_to_image, ocr_image_to_text


# ---------- CONFIG ----------
# Reason:
# Keep paths + config here so all downstream scripts can rely on stable locations.
ROOT_DIR = Path(__file__).resolve().parent
PDF_PATH = ROOT_DIR.parent / "data" / "HC_Verma.pdf"
OUT_DIR = ROOT_DIR / "data"
PAGES_DIR = OUT_DIR / "pages"
RAW_TEXT_OUT = OUT_DIR / "hcverma_raw.txt"
PAGE_LOG = OUT_DIR / "page_log.jsonl"

PAGES_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_page_text(pdf_page) -> str:
    """
    Primary text extraction via pdfplumber.

    Why:
    - Selectable text from the PDF is cleaner than OCR.
    - We only fall back to OCR if this is empty.
    """
    text = (pdf_page.extract_text() or "").strip()
    return text


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    with pdfplumber.open(PDF_PATH) as pdf, \
            RAW_TEXT_OUT.open("w", encoding="utf-8") as raw_f, \
            PAGE_LOG.open("w", encoding="utf-8") as log_f:

        num_pages = len(pdf.pages)
        print(f"[INFO] Loaded {num_pages} pages from {PDF_PATH.name}")

        for idx, page in enumerate(pdf.pages, start=1):
            page_number = idx

            # 1) Render and save page image
            # Uses: render_page_to_image from ocr_utils.
            # Why:
            # - Downstream figure/table/equation detection will use these PNGs.
            # NEW (pass file path; capture the returned image path)
            img_path = Path(
                render_page_to_image(
                str(PDF_PATH),           # pass the PDF file path (string)
                page_number - 1,         # zero-based page index
                dpi=300,
                out_dir=str(PAGES_DIR)   # tell it where to save
                )
                    )


            # 2) Extract text, preferring PDF text
            text = extract_page_text(page)
            text_source = "pdf"

            if not text:
                # Fallback: OCR on the rendered page image
                # Uses: ocr_image_to_text from ocr_utils (Tesseract/other under the hood).
                ocr_text = ocr_image_to_text(img_path)
                ocr_text = (ocr_text or "").strip()
                if ocr_text:
                    text = ocr_text
                    text_source = "ocr"
                else:
                    # No text at all (blank/figure-only page).
                    # We still log and keep alignment via page marker.
                    text = ""
                    text_source = "none"

            # 3) Write page marker + text
            # Why:
            # - `--- Page N ---` is a hard contract used later to:
            #   - segment chapters,
            #   - attach equations/examples/problems to correct pages.
            raw_f.write(f"--- Page {page_number} ---\n{text}\n\n")

            # 4) Log page metadata
            # Why:
            # - Debugging (see which pages needed OCR),
            # - Provenance for SFT/data audits.
            log = {
                "page": page_number,
                "text_source": text_source,
                "image_path": str(img_path),
                "char_len": len(text),
            }
            log_f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Wrote raw text to {RAW_TEXT_OUT} and page log to {PAGE_LOG}")


if __name__ == "__main__":
    main()
