import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from difficulty.scorer import score_question


# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LORA_ADAPTER_PATH = "llama32_1b_hcverma_sft" 

OUT_PATH = "runs/constrained_sft/gen_constrained.jsonl"

CONCEPTS: List[str] = [
    "Kinematics (1D motion)",
    "Newton's laws (block on rough surface)",
    "Work-energy theorem",
    "Momentum and impulse",
    "Circular motion basics",
]

TARGET_LEVELS = ["easy", "medium", "hard"]
SAMPLES_PER_PAIR = 10
K_CANDIDATES = 12

GEN_KWARGS = dict(
    max_new_tokens=220,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
)

# Updated Template to include Srikedaar's Preference Constraint
PROMPT_TEMPLATE = """[DIFFICULTY={level}][PREFERENCE=PARSIMONIOUS]
### Task: variant_generation
### Instruction:
Generate a NEW physics question (HC Verma style) on the concept: {concept}
Target difficulty: {level}
Requirement: Ensure the question is parsimonious (no irrelevant info).
Output ONLY the question statement (no solution).
### Response:
"""

TARGET_SCORE_CENTER_V0 = {"easy": 0.35, "medium": 0.80, "hard": 1.30}
TARGET_SCORE_CENTER_V1 = {"easy": 0.70, "medium": 1.05, "hard": 1.40}


def parse_args():
    ap = argparse.ArgumentParser(description="Generate constrained Best-of-K dataset.")
    ap.add_argument("--scorer", choices=["v0", "v1"], default="v0")
    ap.add_argument("--out_path", default=OUT_PATH)
    return ap.parse_args()


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_one(tokenizer, model, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, **GEN_KWARGS)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "### Response:" in text:
        text = text.split("### Response:", 1)[-1].strip()
    return text.strip()


def pick_best(candidates: List[Dict[str, Any]], target_level: str, scorer_name: str) -> Dict[str, Any]:
    """
    Enhanced selection logic for Preference Constraints.
    1. Primary: Match target difficulty bucket.
    2. Secondary: Penalize 'bloat' (Parsimony).
    3. Tertiary: Reward conceptual interaction (Concept Coupling).
    """
    target_level = target_level.lower()
    
    # 1. Prefer correct bucket match
    matches = [c for c in candidates if c["bucket"] == target_level]
    pool = matches if matches else candidates

    def calculate_preference_score(cand):
        # Preference A: Parsimony Penalty
        # Penalize questions that are too long (potential reward gaming)
        text_len = len(cand["text"])
        parsimony_penalty = 0
        if text_len > 800:
            parsimony_penalty = -0.6  # Heavy penalty for extreme bloat
        elif text_len > 500:
            parsimony_penalty = -0.2

        # Preference B: Concept Coupling Bonus
        # If we are targeting 'hard', reward higher V1 conceptual scores
        coupling_bonus = 0
        if target_level == "hard" and scorer_name == "v1":
            coupling_bonus = 0.2 if cand["score"] > 1.2 else 0

        # Closeness to target center (Standard tie-break)
        center = TARGET_SCORE_CENTER_V1[target_level] if scorer_name == "v1" else TARGET_SCORE_CENTER_V0[target_level]
        closeness = 1.0 / (1.0 + abs(cand["score"] - center))

        return closeness + parsimony_penalty + coupling_bonus

    return max(pool, key=calculate_preference_score)


def main():
    args = parse_args()
    scorer_name = args.scorer
    out_path = args.out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tokenizer, model = load_model()

    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scorer": scorer_name,
        "selection_logic": "Parsimony-aware Best-of-K",
    }

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": run_meta}) + "\n")

    idx = 0
    for concept in CONCEPTS:
        for level in TARGET_LEVELS:
            for _ in range(SAMPLES_PER_PAIR):
                prompt = PROMPT_TEMPLATE.format(concept=concept, level=level.upper())

                candidates = []
                for k in range(K_CANDIDATES):
                    q = generate_one(tokenizer, model, prompt)
                    feats, s, b = score_question(q, scorer=scorer_name)

                    candidates.append({
                        "k": k, "text": q, "features": feats,
                        "score": float(s), "bucket": b,
                    })

                chosen = pick_best(candidates, level, scorer_name)

                rec = {
                    "id": idx,
                    "concept": concept,
                    "target_level": level,
                    "chosen_question": chosen["text"],
                    "chosen_bucket": chosen["bucket"],
                    "chosen_score": chosen["score"],
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                }

                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()

                idx += 1
                if idx % 5 == 0:
                    print(f"[PROGRESS] wrote {idx} constrained samples...")
                time.sleep(0.02)

    print(f"[OK] Saved {idx} records to: {out_path}")


if __name__ == "__main__":
    main()