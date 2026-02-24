# File structure
data > raw > (place raw PDFs of books)

# Create a Virtual Environment
python -m venv venv
venv\Scripts\Activate

# Install Dependencies
pip install -r requirements.txt

# Run OCR Page extraction for HC Verma (Questions in the Bucket-Easy)
python scripts/preprocess_pdf.py --pdf_path data/raw/HC_Verma.pdf --out_dir data/processed/hcverma --book_id hcverma --enable_ocr 0 --min_chars 50 --dpi 300

# Run OCR Page extraction for Irodov (Questions in the Bucket-Hard)
python scripts/preprocess_pdf.py --pdf_path data/raw/Irodov.pdf --out_dir data/processed/irodov --book_id irodov --enable_ocr 1 --min_chars 50 --dpi 350

# Structuring Extracted Texts
python scripts/structure_text.py --in_raw data/processed/hcverma/raw.txt --out_json data/processed/hcverma/structured.json --out_md data/processed/hcverma/structured.md --book_id hcverma --book_spec configs/books/hcverma.yaml

python scripts/structure_text.py --in_raw data/processed/irodov/raw.txt --out_json data/processed/irodov/structured.json --out_md data/processed/irodov/structured.md --book_id irodov --book_spec configs/books/irodov.yaml

# Extracting problems for Irodov
python scripts/extract_irodov_questions.py `
--structured_json data/processed/irodov/structured.json `
--out_jsonl data/processed/irodov/problems.jsonl `
--book_id irodov `
--bucket_gt hard

# Extracting problms for HC Verma
python scripts/extract_hcverma_questions.py `
--structured_json data/processed/hcverma/structured.json `
--out_jsonl data/processed/hcverma/problems.jsonl `
--book_id hcverma `
--bucket_gt easy

# Merging the two into one singl dataset
python scripts/datasets/merge_books.py `
  --inputs data/processed/hcverma/problems.jsonl data/processed/irodov/problems.jsonl `
  --out data/merged/all_problems.jsonl

# Splitting dataset into test and train sets
python scripts/datasets/build_splits_balanced.py `
  --in data/merged/all_problems.jsonl `
  --out_dir data/merged/splits_easy_hard `
  --buckets easy hard `
  --split 0.85 0.10 0.05 `
  --seed 42