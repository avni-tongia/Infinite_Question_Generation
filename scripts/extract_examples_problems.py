import json
import re

input_json = "data/biology2e_structured.json"
output_json = "data/biology2e_with_examples.json"

# Load the structured JSON
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# Patterns to detect examples and problems
example_pattern = re.compile(r'(Example\s*\d*[:.-]?)', re.IGNORECASE)
problem_pattern = re.compile(r'(Exercise|Practice Problem|Question\s*\d*[:.-]?)', re.IGNORECASE)

# Process each section
for chapter in data:
    for section in chapter["sections"]:
        text = section["text"]

        # Extract examples
        examples = example_pattern.split(text)
        section["examples"] = []
        if len(examples) > 1:
            for i in range(1, len(examples), 2):
                section["examples"].append({
                    "title": examples[i].strip(),
                    "text": examples[i+1].strip() if i+1 < len(examples) else ""
                })

        # Extract problems
        problems = problem_pattern.split(text)
        section["problems"] = []
        if len(problems) > 1:
            for i in range(1, len(problems), 2):
                section["problems"].append({
                    "title": problems[i].strip(),
                    "text": problems[i+1].strip() if i+1 < len(problems) else ""
                })

# Save updated JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print(f"✅ Examples and problems extracted to {output_json}")
