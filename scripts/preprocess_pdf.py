import pdfplumber

# Path to your PDF
pdf_path = "data/Biology2e-WEB.pdf"

# Path to save extracted raw text
output_path = "data/biology2e_raw.txt"

# Open PDF and extract text page by page
with pdfplumber.open(pdf_path) as pdf:
    all_text = ""
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            all_text += f"--- Page {i+1} ---\n{text}\n\n"

# Save the extracted text to a .txt file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(all_text)

print(f"✅ Raw text extracted to {output_path}")
