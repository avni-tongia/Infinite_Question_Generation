
import json
from pathlib import Path

import pdfplumber
from ocr_utils import render_page_to_image, ocr_image_to_text
import argparse

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_path", required=True, help="Path to input PDF")
    ap.add_argument("--out_dir", required=True, help="Output dir: data/processed/<book_id>")
    ap.add_argument("--book_id", required=True, help="Book identifier, e.g., hcverma, irodov")

    # OCR controls (Irodov often needs OCR)
    ap.add_argument("--enable_ocr", type=int, default=0, help="0/1: enable OCR fallback")
    ap.add_argument("--min_chars", type=int, default=50, help="Trigger OCR if extracted text shorter than this")
    ap.add_argument("--dpi", type=int, default=300, help="Render DPI for page images (OCR improves with higher DPI)")
    return ap.parse_args()


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
    args = parse_args()

    ROOT_DIR = Path(__file__).resolve().parent
    PDF_PATH = Path(args.pdf_path)
    OUT_DIR = Path(args.out_dir)
    PAGES_DIR = OUT_DIR / "pages"
    RAW_TEXT_OUT = OUT_DIR / "raw.txt"
    PAGE_LOG = OUT_DIR / "page_log.jsonl"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    with pdfplumber.open(PDF_PATH) as pdf, \
            RAW_TEXT_OUT.open("w", encoding="utf-8") as raw_f, \
            PAGE_LOG.open("w", encoding="utf-8") as log_f:
    

        for idx, page in enumerate(pdf.pages, start=1):
            page_number = idx

            # 1) Render and save page image
            # Uses: render_page_to_image from ocr_utils.
            # Why:
            # - Downstream figure/table/equation detection will use these PNGs.
            # NEW (pass file path; capture the returned image path)
            img_path = Path(
                render_page_to_image(
                    str(PDF_PATH),            # PDF file path
                    page_number - 1,          # zero-based page index
                    dpi=int(args.dpi),
                    out_dir=str(PAGES_DIR)    # where to save
                )
            )


            # 2) Extract text, preferring PDF text
            text = extract_page_text(page)
            text_source = "pdf"

            # Trigger OCR if:
            # - no extracted text, OR extracted text is too short (common for scanned PDFs)
            if (len(text.strip()) < int(args.min_chars)) and int(args.enable_ocr) == 1:
                ocr_text = ocr_image_to_text(img_path)
                ocr_text = (ocr_text or "").strip()
                if ocr_text:
                    text = ocr_text
                    text_source = "ocr"
                else:
                    text = ""
                    text_source = "none"
            elif not text:
                # No OCR allowed, but also no text
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
