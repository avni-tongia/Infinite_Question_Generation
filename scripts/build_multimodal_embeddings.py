
"""
build_multimodal_embeddings.py
Builds image embeddings for figures and image-type equations using a CLIP model.
Outputs:
- data/vectors/figures_clip.npy + figures_metadata.json
- data/vectors/equations_clip.npy + equations_metadata.json
"""
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

FIG_JSONL = "data/figures/figures.jsonl"
EQ_JSONL = "data/equations/equations.jsonl"
OUT_DIR = Path("data/vectors")

def load_jsonl(path: str):
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items

def encode_images(model, paths):
    vecs = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            vec = model.encode(img)
            vecs.append(vec)
        except Exception as e:
            print(f"[WARN] Failed to encode {p}: {e}")
    if len(vecs) == 0:
        return np.zeros((0, 512), dtype="float32")
    return np.vstack(vecs)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer("clip-ViT-B-32")

    # Figures
    figs = load_jsonl(FIG_JSONL)
    fig_paths = [f["image_path"] for f in figs if os.path.exists(f.get("image_path", ""))]
    fig_vecs = encode_images(model, fig_paths)
    np.save(OUT_DIR / "figures_clip.npy", fig_vecs)
    with open(OUT_DIR / "figures_metadata.json", "w", encoding="utf-8") as f:
        json.dump(figs, f, indent=2)
    print(f"[INFO] Encoded {len(fig_paths)} figures")

    # Equations (image-type only)
    eqs = load_jsonl(EQ_JSONL)
    eq_img = [e for e in eqs if e.get("type") == "image" and os.path.exists(e.get("image_path",""))]
    eq_paths = [e["image_path"] for e in eq_img]
    eq_vecs = encode_images(model, eq_paths)
    np.save(OUT_DIR / "equations_clip.npy", eq_vecs)
    with open(OUT_DIR / "equations_metadata.json", "w", encoding="utf-8") as f:
        json.dump(eq_img, f, indent=2)
    print(f"[INFO] Encoded {len(eq_paths)} equation images")

if __name__ == "__main__":
    main()
