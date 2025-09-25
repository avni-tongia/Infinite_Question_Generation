
"""
extract_tables.py
Extracts tables from the PDF using Camelot (lattice + stream).
Outputs:
- data/tables/table_####.csv
- data/tables/tables.jsonl
Requirements: ghostscript installed for Camelot; or switch to tabula-py if preferred.
"""
import json
from pathlib import Path
import camelot

PDF_PATH = "data/HC_Verma.pdf"
TBL_DIR = Path("data/tables")
OUT_JSONL = TBL_DIR / "tables.jsonl"

def main():
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    # Try lattice first (works with ruled tables); fall back to stream
    # You can also iterate over specific page ranges if needed.
    for flavor in ["lattice", "stream"]:
        try:
            tables = camelot.read_pdf(PDF_PATH, flavor=flavor, pages="all")
        except Exception as e:
            print(f"[WARN] Camelot {flavor} failed: {e}")
            continue

        for i, t in enumerate(tables):
            csv_path = TBL_DIR / f"table_{flavor}_{i:04d}.csv"
            t.to_csv(csv_path.as_posix())
            rec = {
                "table_id": f"tbl_{flavor}_{i:04d}",
                "page": t.parsing_report.get("page", None),
                "method": f"camelot-{flavor}",
                "csv_path": csv_path.as_posix(),
                "caption_guess": None,
                "chapter_guess": None
            }
            records.append(rec)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[INFO] Extracted {len(records)} tables -> {OUT_JSONL}")

if __name__ == "__main__":
    main()
