
"""
ocr_utils.py
Helpers for rendering PDF pages to images and applying OCR (Tesseract).
- render_page_to_image: uses PyMuPDF to rasterize a page to PNG.
- ocr_image_to_text: uses pytesseract to OCR an image (English by default).
"""
from typing import Optional
import os
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

def render_page_to_image(pdf_path: str, page_number: int, dpi: int = 300, out_dir: str = "data/pages") -> str:
    """
    Render a 1-indexed page of a PDF to a PNG image.
    Returns the path to the saved image.
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number - 1)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir_p / f"page_{page_number:04d}.png"
        pix.save(out_path.as_posix())
        return out_path.as_posix()
    finally:
        doc.close()

def ocr_image_to_text(image_path: str, lang: str = "eng", psm: int = 6) -> str:
    """
    OCR an image to text using Tesseract.
    psm=6 assumes a uniform block of text. Adjust if needed.
    """
    img = Image.open(image_path)
    # Configure tesseract; --oem 1 uses LSTM-based engine, --psm X controls page segmentation
    config = f"--oem 1 --psm {psm}"
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    # Normalize newlines a bit
    return text.replace("\r\n", "\n").strip()
