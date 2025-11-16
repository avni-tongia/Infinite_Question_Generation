"""
extract_equations.py
Extracts equations from the raw text and (optionally) from page images.
- Text route: Heuristics to detect equation-like lines, converts to basic LaTeX.
- Image route: Crops candidate regions via PyMuPDF XObject scan or (simpler) full-page images with optional external OCR (Mathpix). Here we provide a stub hook for Mathpix.
Outputs:
- scripts/data/equations/equations.jsonl
- scripts/data/equations/eq_*.png (if image-cropped equations are produced)
Environment (optional for LaTeX from images):
- MATHPIX_APP_ID, MATHPIX_APP_KEY (if you later add Mathpix API calls)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

# -----------------------------------------------------------------------------
# PATHS (relative to this script)
# -----------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

RAW_TEXT = ROOT_DIR / "data" / "hcverma_raw.txt"
PDF_PATH = ROOT_DIR / "data" / "HC_Verma.pdf"
EQ_DIR = ROOT_DIR / "data" / "equations"
OUT_JSONL = EQ_DIR / "equations.jsonl"

# Structured chapters + examples (output of structure_text + extract_examples_problems)
STRUCTURED_WITH_EX = ROOT_DIR / "data" / "hcverma_with_examples.json"

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
    ^(?=.*=)          # must contain an '=' sign
    .{3,}$            # some minimal length
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
    # Replace / with \frac for simple a/b patterns (very naive)
    expr = re.sub(r"\b([A-Za-z0-9]+)\s*/\s*([A-Za-z0-9]+)\b", r"\\frac{\1}{\2}", expr)
    # Wrap in $...$
    return f"${expr.strip()}$"


def extract_text_equations() -> List[Dict]:
    """
    Extract clean equation-like lines from hcverma_raw.txt.
    Keeps only standalone expressions like 'F = ma', 'v^2 = u^2 + 2as', etc.
    Drops:
      - TOC/publisher junk
      - sentences that just contain '='
      - very long lines
      - double-equals lines
      - lines with too many words
    """
    eqs: List[Dict] = []
    curr_page: Optional[int] = None

    with RAW_TEXT.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            # Page marker
            m = PAGE_MARKER.match(stripped)
            if m:
                curr_page = int(m.group(1))
                continue

            if not stripped:
                continue

            # Must contain exactly ONE '=' (no sentences like "x = y = z")
            if stripped.count("=") != 1:
                continue

            # Reject lines with '=' but too many words (sentences)
            if len(stripped.split()) > 6:
                continue

            # Skip front matter / TOC text
            junk_keywords = [
                "Ansari Road", "NEW DELHI", "Printed at",
                "hologram", "Chapters ", "Mechanics",
                "Thermodynamics", "Optics", "Modern physics",
                "publisher", "Acknowledgement", "Acknowledgment"
            ]
            if any(kw.lower() in stripped.lower() for kw in junk_keywords):
                continue

            # Reject lines starting with words — must look math-like
            if re.match(r"^[A-Za-z].*\s[A-Za-z]", stripped) and not re.match(r"^[A-Za-z]\s*=", stripped):
                continue

            # Reject lines ending with ',', '.', '–'
            if stripped.endswith((",", ".", "-", "—")):
                continue

            # At this point the line is probably a real equation
            surface = stripped
            latex = to_latex_simple(surface)

            eqs.append({
                "equation_id": f"eq_p{curr_page:04d}_{len(eqs)+1:03d}",
                "page": curr_page,
                "type": "text",
                "surface": surface,
                "latex": latex,
                "image_path": None,
                "chapter_guess": None,
                "examples": []
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
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# NEW: annotate equations with chapter + examples
# -----------------------------------------------------------------------------

def annotate_equations_with_chapters_and_examples(eqs: List[Dict]) -> None:
    """
    For each equation (with a 'page'), infer:
      - which chapter it belongs to → eq["chapter_guess"]
      - which examples are in that chapter → eq["examples"] = [example_id, ...]
    Uses scripts/data/hcverma_with_examples.json produced by earlier scripts.

    Note: this is at chapter granularity. All examples in the chapter are
    attached to each equation whose page falls in that chapter's page_span.
    """
    if not STRUCTURED_WITH_EX.exists():
        print(f"[WARN] {STRUCTURED_WITH_EX} not found. Skipping chapter/example linking.")
        return

    with STRUCTURED_WITH_EX.open("r", encoding="utf-8") as f:
        chapters = json.load(f)

    # Build a simple index: for each chapter, its page range and example_ids
    chapter_index = []
    for ch in chapters:
        span = ch.get("page_span") or [None, None]
        start, end = span
        if start is None or end is None:
            continue

        example_ids = []
        for ex_ref in ch.get("examples", []):
            # In hcverma_with_examples.json, examples are stored as {"example_id": "..."}
            ex_id = ex_ref.get("example_id")
            if ex_id:
                example_ids.append(ex_id)

        chapter_index.append({
            "chapter_number": ch.get("chapter_number"),
            "page_start": start,
            "page_end": end,
            "examples": example_ids,
        })

    # For each equation, locate its chapter + examples
    for eq in eqs:
        page = eq.get("page")
        if page is None:
            continue

        for ch in chapter_index:
            if ch["page_start"] <= page <= ch["page_end"]:
                eq["chapter_guess"] = ch["chapter_number"]
                # Attach example_ids at chapter level (coarse but useful)
                eq["examples"] = ch["examples"][:]
                break


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    text_eqs = extract_text_equations()
    img_eqs = extract_image_equations_stub()
    all_eqs = text_eqs + img_eqs

    # Annotate equations with chapter + examples
    annotate_equations_with_chapters_and_examples(all_eqs)

    write_jsonl(all_eqs, OUT_JSONL)
    print(f"[INFO] Extracted {len(all_eqs)} equations -> {OUT_JSONL}")


if __name__ == "__main__":
    main()
