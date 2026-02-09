"""
Generate a baseline dataset of questions using the SFT (LoRA) model.

Why:
- We need a fixed reference set to compare SFT vs RLFT later.
- This script produces JSONL with metadata for reproducible evaluation.

Output:
- runs/baseline_sft/gen_sft_baseline.jsonl
"""

import json
import os
import time
from datetime import datetime
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# -----------------------------
# CONFIG: set these two
# -----------------------------
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # change if different
LORA_ADAPTER_PATH = "llama32_1b_hcverma_sft"         # <-- replace with your adapter folder

OUT_PATH = "runs/baseline_sft/gen_sft_baseline.jsonl"

# Keep this small first; you can scale later
CONCEPTS: List[str] = [
    "Kinematics (1D motion)",
    "Newton's laws (block on rough surface)",
    "Work-energy theorem",
    "Momentum and impulse",
    "Circular motion basics",
]

TARGET_LEVELS = ["easy", "medium", "hard"]
SAMPLES_PER_PAIR = 5

GEN_KWARGS = dict(
    max_new_tokens=220,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
)

# Prompt format: simple + consistent
PROMPT_TEMPLATE = """### Task: variant_generation
### Instruction:
Generate a NEW physics question (HC Verma style) on the concept: {concept}
Target difficulty: {level}
Output ONLY the question statement (no solution).
### Response:
"""


def load_model():
    """Load base model + LoRA adapter."""
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    print("[INFO] Loading base model (may take time if downloading)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("[INFO] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
    model.eval()

    print("[INFO] Model ready.")
    return tokenizer, model


@torch.no_grad()
def generate_one(tokenizer, model, prompt: str) -> str:
    """Generate a single question text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, **GEN_KWARGS)
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    # Return only content after the response tag (keeps output clean)
    if "### Response:" in text:
        text = text.split("### Response:", 1)[-1].strip()
    return text.strip()


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    tokenizer, model = load_model()

    run_meta = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "base_model": BASE_MODEL,
        "lora_adapter_path": LORA_ADAPTER_PATH,
        "gen_kwargs": GEN_KWARGS,
        "concepts": CONCEPTS,
        "target_levels": TARGET_LEVELS,
        "samples_per_pair": SAMPLES_PER_PAIR,
    }

    # Write meta as first line so every run is self-describing
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": run_meta}) + "\n")

    idx = 0
    for concept in CONCEPTS:
        for level in TARGET_LEVELS:
            for _ in range(SAMPLES_PER_PAIR):
                prompt = PROMPT_TEMPLATE.format(concept=concept, level=level)
                q = generate_one(tokenizer, model, prompt)

                rec = {
                    "id": idx,
                    "concept": concept,
                    "target_level": level,
                    "prompt": prompt,
                    "question": q,
                    "gen_kwargs": GEN_KWARGS,
                    "time_utc": datetime.utcnow().isoformat(),
                }

                with open(OUT_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                idx += 1
                time.sleep(0.05)  # tiny delay to avoid bursty IO

    print(f"[OK] Saved {idx} records to: {OUT_PATH}")


if __name__ == "__main__":
    main()
