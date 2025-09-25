# 📘 Multimodal HC Verma Pipeline

This repository processes the **HC Verma PDF** into a structured, multimodal dataset with **text, equations (LaTeX), figures, tables, and embeddings**.  
It produces a final `manifest.json` linking everything together for downstream **RAG, QA, and fine-tuning**.

---

## 🔧 Features

- **Text extraction** with OCR fallback (Tesseract) for scanned pages.  
- **Chapter structuring** into JSON + Markdown.  
- **Examples/Problems scaffolding** for future parsing.  
- **Equation extraction** (text → LaTeX, with image-equation stub).  
- **Figure extraction** (images + caption guesses).  
- **Table extraction** to CSV.  
- **Text embeddings** with SentenceTransformers.  
- **Image embeddings** (figures & equations) with CLIP.  
- **Cross-linking** into one coherent `manifest.json`.  
- **Single driver script (`run_pipeline.py`)** to run the entire pipeline end-to-end.  

---

## 🖥️ Setup

### 1. Create virtual environment
Windows (PowerShell):
```bash
python -m venv venv
venv\Scripts\activate
```

macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install system dependencies
Windows (Chocolatey):
```bash
choco install tesseract ghostscript
```

macOS (Homebrew):
```bash
brew install tesseract ghostscript
```

### 3. Install Python dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📂 Folder Structure

```
your-repo/
│
├── Scripts/                  # all Python scripts
│   ├── preprocess_pdf.py
│   ├── structure_text.py
│   ├── extract_examples_problems.py
│   ├── build_embeddings.py
│   ├── ocr_utils.py
│   ├── extract_equations.py
│   ├── extract_figures.py
│   ├── extract_tables.py
│   ├── build_multimodal_embeddings.py
│   └── index_manifest.py
│
├── data/                     # input + generated data
│   ├── HC_Verma.pdf          # input book (only manual file)
│   ├── hcverma_raw.txt
│   ├── page_log.jsonl
│   ├── pages/
│   ├── hcverma_structured.json
│   ├── hcverma_with_examples.json
│   ├── equations/
│   ├── figures/
│   ├── tables/
│   ├── vectors/
│   └── manifest.json
│
├── run_pipeline.py           # driver script (runs all steps in order)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 How to Run

From the project root:
```bash
python run_pipeline.py
```

This will execute all steps in order automatically and generate outputs in the `data/` folder.  

---

## ✅ Outputs

- **Raw text:** `data/hcverma_raw.txt`  
- **Chapters:** `data/hcverma_structured.json`, `hcverma_structured.md`  
- **Examples/Problems scaffold:** `data/hcverma_with_examples.json`  
- **Equations (LaTeX + stubs):** `data/equations/equations.jsonl` (+ PNGs if image detection added)  
- **Figures:** `data/figures/figures.jsonl` + PNGs  
- **Tables:** `data/tables/tables.jsonl` + CSVs  
- **Embeddings:**  
  - Text → `hcverma_embeddings.npy`, `hcverma_embeddings_with_metadata.json`  
  - Images → `vectors/figures_clip.npy`, `vectors/equations_clip.npy` (+ JSON metadata)  
- **Manifest:** `data/manifest.json`