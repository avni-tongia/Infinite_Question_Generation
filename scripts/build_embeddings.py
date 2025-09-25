
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/hcverma_with_examples.json"
EMBEDDINGS_FILE = "data/hcverma_embeddings.npy"
METADATA_FILE = "data/hcverma_embeddings_with_metadata.json"

def load_data(path):
    """Load the JSON file with chapters and text."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_embeddings(chapters, model_name="all-MiniLM-L6-v2"):
    """Convert each chapter's text into embeddings."""
    model = SentenceTransformer(model_name)
    texts, metadata = [], []

    for idx, chapter in enumerate(chapters, 1):
        # Use the actual keys emitted by structure_text.py
        text = chapter.get("content", "").strip()
        title = chapter.get("chapter_title", f"Chapter {idx}")
        if not text:
            continue
        texts.append(text)
        metadata.append({
            "chapter_number": idx,
            "title": title
        })

    print(f"[INFO] Creating embeddings for {len(texts)} chapters...")
    vectors = model.encode(texts, show_progress_bar=True)
    return vectors, metadata

def save_outputs(vectors, metadata):
    """Save embeddings and metadata to disk."""
    np.save(EMBEDDINGS_FILE, vectors)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    print("[INFO] Loading data...")
    chapters = load_data(INPUT_FILE)

    print("[INFO] Building embeddings...")
    vectors, metadata = build_embeddings(chapters)

    print("[INFO] Saving embeddings...")
    save_outputs(vectors, metadata)

    print(f"[SUCCESS] Saved {len(vectors)} embeddings to {EMBEDDINGS_FILE}")
    print(f"[SUCCESS] Metadata saved to {METADATA_FILE}")
