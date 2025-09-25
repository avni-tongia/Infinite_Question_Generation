"""
run_pipeline.py
Runs the full multimodal pipeline in the correct order.
Make sure data/HC_Verma.pdf exists before running.
"""

import subprocess

SCRIPTS = [
    "Scripts/preprocess_pdf.py",
    "Scripts/structure_text.py",
    "Scripts/extract_examples_problems.py",
    "Scripts/build_embeddings.py",
    "Scripts/extract_equations.py",
    "Scripts/extract_figures.py",
    "Scripts/extract_tables.py",
    "Scripts/build_multimodal_embeddings.py",
    "Scripts/index_manifest.py"
]

def run_script(script):
    print(f"\n[RUNNING] {script}")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("[STDERR]", result.stderr)

def main():
    for script in SCRIPTS:
        run_script(script)
    print("\n✅ Pipeline complete! All outputs are in data/")

if __name__ == "__main__":
    main()
