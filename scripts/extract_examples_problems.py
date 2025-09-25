import json

INPUT_FILE = "data/hcverma_structured.json"
OUTPUT_FILE = "data/hcverma_with_examples.json"

def load_structured_data(path):
    """Load structured chapters/sections JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def enrich_with_examples(data):
    """
    Adds placeholder 'examples' and 'problems' fields to each chapter.
    Later, we’ll parse text more intelligently.
    """
    for chapter in data:
        # Add empty examples/problems lists
        chapter["examples"] = []
        chapter["problems"] = []
    return data

def save_json(data, path):
    """Writes updated structured data back to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    print("[INFO] Loading structured data...")
    structured_data = load_structured_data(INPUT_FILE)

    print("[INFO] Adding example/problem placeholders...")
    updated_data = enrich_with_examples(structured_data)

    save_json(updated_data, OUTPUT_FILE)
    print(f"[INFO] Data with examples/problems saved to {OUTPUT_FILE}")
