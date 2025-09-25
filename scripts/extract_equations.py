
"""
extract_equations.py
Extracts equations from the raw text and (optionally) from page images.
- Text route: Heuristics to detect equation-like lines, converts to basic LaTeX.
- Image route: Crops candidate regions via PyMuPDF XObject scan or (simpler) full-page images with optional external OCR (Mathpix). Here we provide a stub hook for Mathpix.
Outputs:
- data/equations/equations.jsonl
- data/equations/eq_*.png (if image-cropped equations are produced)
Environment (optional for LaTeX from images):
- MATHPIX_APP_ID, MATHPIX_APP_KEY (if you later add Mathpix API calls)
"""
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF
from PIL import Image

RAW_TEXT = "data/hcverma_raw.txt"
PDF_PATH = "data/HC_Verma.pdf"
EQ_DIR = Path("data/equations")
OUT_JSONL = EQ_DIR / "equations.jsonl"

# --- Utilities ---
MATH_SYMBOL_MAP = {
    "×": r"\\times",
    "÷": r"\\div",
    "−": "-",   # unicode minus to ASCII minus
    "–": "-",
    "±": r"\\pm",
    "∓": r"\\mp",
    "≈": r"\\approx",
    "≃": r"\\simeq",
    "≅": r"\\cong",
    "≡": r"\\equiv",
    "∝": r"\\propto",
    "∞": r"\\infty",
    "α": r"\\alpha",
    "β": r"\\beta",
    "γ": r"\\gamma",
    "Δ": r"\\Delta",
    "δ": r"\\delta",
    "ε": r"\\epsilon",
    "θ": r"\\theta",
    "λ": r"\\lambda",
    "μ": r"\\mu",
    "π": r"\\pi",
    "σ": r"\\sigma",
    "φ": r"\\varphi",
    "ω": r"\\omega"
}

EQ_LINE_PAT = re.compile(
    r"""(?x)
    ^(?=.*[=+\-*/])        # must contain at least one math operator
    (?!.*Figure)           # avoid figure lines
    (?!.*Table)            # avoid table lines
    .{3,}$                 # some minimal length
    """
)

PAGE_MARKER = re.compile(r"^--- Page (\d+) ---\s*$")

def to_latex_simple(expr: str) -> str:
    """Very light conversion of a text equation to LaTeX-ish form."""
    # Replace known unicode symbols
    for k, v in MATH_SYMBOL_MAP.items():
        expr = expr.replace(k, v)
    # Handle caret-based superscripts like x^2 -> x^{2}
    expr = re.sub(r"\^(\w+)", r"^{\1}", expr)
    # Replace / with \over for simple a/b patterns (very naive)
    # We'll only do this for single-token over single-token to avoid havoc
    expr = re.sub(r"\b([A-Za-z0-9]+)\s*/\s*([A-Za-z0-9]+)\b", r"\\frac{\1}{\2}", expr)
    # Wrap in $...$
    return f"${expr.strip()}$"

def extract_text_equations() -> List[Dict]:
    eqs = []
    curr_page = None
    with open(RAW_TEXT, "r", encoding="utf-8") as f:
        for line in f:
            m = PAGE_MARKER.match(line.strip())
            if m:
                curr_page = int(m.group(1))
                continue
            # Heuristic: line looks like an equation
            if EQ_LINE_PAT.match(line.strip()):
                surface = line.strip()
                # Avoid false positives: long paragraphs with '=' only once and many words
                if len(surface.split()) > 20 and surface.count('=') == 1:
                    continue
                latex = to_latex_simple(surface)
                eqs.append({
                    "equation_id": f"eq_p{curr_page:04d}_{len(eqs)+1:03d}",
                    "page": curr_page,
                    "type": "text",
                    "surface": surface,
                    "latex": latex,
                    "image_path": None,
                    "chapter_guess": None
                })
    return eqs

def extract_image_equations_stub() -> List[Dict]:
    """
    Stub: we attempt to find small raster images that could be equations.
    For now, we conservatively return an empty list to avoid excessive false positives.
    You can later plug a detector or use Mathpix API per-crop.
    """
    return []

def write_jsonl(records: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    text_eqs = extract_text_equations()
    img_eqs = extract_image_equations_stub()
    all_eqs = text_eqs + img_eqs
    write_jsonl(all_eqs, OUT_JSONL)
    print(f"[INFO] Extracted {len(all_eqs)} equations -> {OUT_JSONL}")

if __name__ == "__main__":
    main()
