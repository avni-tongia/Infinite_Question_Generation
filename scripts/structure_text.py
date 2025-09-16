import re
import json

input_file = "data/biology2e_raw.txt"
output_json = "data/biology2e_structured.json"
output_md = "data/biology2e_structured.md"

with open(input_file, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split by chapters
chapters = re.split(r'CHAPTER\s+\d+', raw_text, flags=re.IGNORECASE)

structured_data = []

for i, chapter_text in enumerate(chapters[1:], 1):  # skip index 0, it's before first chapter
    # Optionally, split by sections (1.1, 1.2 etc.)
    sections = re.split(r'\n\d+\.\d+\s+', chapter_text)
    chapter_dict = {"chapter_number": i, "sections": []}
    
    for j, sec_text in enumerate(sections[1:], 1):
        chapter_dict["sections"].append({
            "section_number": f"{i}.{j}",
            "text": sec_text.strip(),
            "examples": [],  # will fill later
            "problems": []   # will fill later
        })
    
    structured_data.append(chapter_dict)

# Save JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(structured_data, f, indent=4)

# Save Markdown
with open(output_md, "w", encoding="utf-8") as f:
    for chapter in structured_data:
        f.write(f"# Chapter {chapter['chapter_number']}\n\n")
        for sec in chapter["sections"]:
            f.write(f"## Section {sec['section_number']}\n")
            f.write(sec['text'] + "\n\n")

print("✅ Structured JSON and Markdown created!")
