"""
Generate a constraint-enforced dataset of questions using Best-of-K selection.

Why:
- Shows that constraints (difficulty control) improve outputs even BEFORE RLFT.
- Provides a strong baseline to compare against RLFT/DPO later.

Output:
- runs/constrained_sft/gen_constrained.jsonl

Selection rule:
- Generate K candidates for each (concept, target_level)
- Score each candidate using difficulty.scorer.score_question with selected scorer (v0 or v1)
- Prefer candidates whose bucket == target_level
- Break ties using closeness to a target score "center" (scorer-specific)
- If none match the bucket, pick the globally closest by score.
"""

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
# CONFIG: update if needed
# -----------------------------
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LORA_ADAPTER_PATH = "llama32_1b_hcverma_sft"  # folder containing adapter_config.json

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

# Best-of-K
K_CANDIDATES = 12

GEN_KWARGS = dict(
    max_new_tokens=220,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
)

PROMPT_TEMPLATE = """### Task: variant_generation
### Instruction:
Generate a NEW physics question (HC Verma style) on the concept: {concept}
Target difficulty: {level}
Output ONLY the question statement (no solution).
### Response:
"""

# Target score centers (tie-break) — tune later after histograms
TARGET_SCORE_CENTER_V0 = {
    "easy": 0.35,
    "medium": 0.80,
    "hard": 1.30,
}

TARGET_SCORE_CENTER_V1 = {
    "easy": 0.70,
    "medium": 1.05,
    "hard": 1.40,
}


def parse_args():
    ap = argparse.ArgumentParser(description="Generate constrained Best-of-K dataset.")
    ap.add_argument(
        "--scorer",
        choices=["v0", "v1"],
        default="v0",
        help="Difficulty scorer used for selection (v0 or v1).",
    )
    ap.add_argument(
        "--out_path",
        default=OUT_PATH,
        help=f"Output JSONL path (default: {OUT_PATH})",
    )
    return ap.parse_args()


def load_model():
    """Load base model + LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    # Avoid repeated pad_token warnings during generate
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_one(tokenizer, model, prompt: str) -> str:
    """Generate a single question text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, **GEN_KWARGS)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "### Response:" in text:
        text = text.split("### Response:", 1)[-1].strip()
    return text.strip()


def target_center(target_level: str, scorer_name: str) -> float:
    """Return the tie-break score center for a given target bucket and scorer."""
    if scorer_name == "v1":
        return TARGET_SCORE_CENTER_V1[target_level]
    return TARGET_SCORE_CENTER_V0[target_level]


def pick_best(cands: List[Dict[str, Any]], target_level: str, scorer_name: str) -> Dict[str, Any]:
    """
    Pick best candidate for the target level.
    cands entries contain: text, feats, score, bucket
    """
    center = target_center(target_level, scorer_name)

    # Prefer correct bucket first
    in_bucket = [c for c in cands if c["bucket"] == target_level]
    pool = in_bucket if in_bucket else cands

    # Choose closest to center score
    best = min(pool, key=lambda c: abs(c["score"] - center))
    return best


def main():
    args = parse_args()
    scorer_name = args.scorer
    out_path = args.out_path

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tokenizer, model = load_model()

    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "lora_adapter_path": LORA_ADAPTER_PATH,
        "gen_kwargs": GEN_KWARGS,
        "concepts": CONCEPTS,
        "target_levels": TARGET_LEVELS,
        "samples_per_pair": SAMPLES_PER_PAIR,
        "k_candidates": K_CANDIDATES,
        "scorer": scorer_name,
        "target_score_center_v0": TARGET_SCORE_CENTER_V0,
        "target_score_center_v1": TARGET_SCORE_CENTER_V1,
        "selection": "prefer bucket match; tie-break by closeness to target score center",
    }

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": run_meta}) + "\n")

    idx = 0
    for concept in CONCEPTS:
        for level in TARGET_LEVELS:
            for _ in range(SAMPLES_PER_PAIR):
                prompt = PROMPT_TEMPLATE.format(concept=concept, level=level)

                # Generate K candidates and score each
                candidates = []
                for k in range(K_CANDIDATES):
                    q = generate_one(tokenizer, model, prompt)

                    # IMPORTANT: score with the chosen scorer
                    feats, s, b = score_question(q, scorer=scorer_name)

                    candidates.append({
                        "k": k,
                        "text": q,
                        "features": feats,
                        "score": float(s),
                        "bucket": b,
                    })

                chosen = pick_best(candidates, level, scorer_name)

                rec = {
                    "id": idx,
                    "concept": concept,
                    "target_level": level,
                    "prompt": prompt,
                    "chosen_question": chosen["text"],
                    f"chosen_score_{scorer_name}": chosen["score"],
                    f"chosen_bucket_{scorer_name}": chosen["bucket"],
                    "candidates": [
                        {
                            "k": c["k"],
                            f"score_{scorer_name}": c["score"],
                            f"bucket_{scorer_name}": c["bucket"],
                        }
                        for c in candidates
                    ],
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