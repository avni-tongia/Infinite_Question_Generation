%%writefile train_sft.py
import os
from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# -----------------------------
# Paths & model configuration
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent
SFT_DIR = ROOT_DIR / "scripts" / "data" / "sft"

# Your 4 SFT JSONL files
SFT_FILES = [
    "solution_explanation.jsonl",
    "variant_generation.jsonl",
    "template_extraction.jsonl",
    "tool_tagging.jsonl",
]

# Base model to fine-tune
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Where to save LoRA adapter + merged model
OUTPUT_DIR = ROOT_DIR / "outputs" / "llama32_1b_hcverma_sft"


def load_sft_datasets():
    """Load and concatenate the 4 JSONL SFT datasets."""
    datasets = []
    for fname in SFT_FILES:
        path = SFT_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"SFT file missing: {path}")
        ds = load_dataset("json", data_files=str(path), split="train")
        datasets.append(ds)
    return concatenate_datasets(datasets)


def formatting_func(example: dict) -> str:
    """
    Turn each row into a single training text string.

    Your JSONL rows should look like:
      {
        "task": "...",
        "input": "...",
        "output": "..."
      }
    """
    task = example.get("task", "")
    inp = example.get("input", "")
    out = example.get("output", "")

    # Simple instruction-format prompt; you can tweak later if needed
    text = (
        f"### Task: {task}\n"
        f"### Instruction:\n{inp}\n\n"
        f"### Response:\n{out}"
    )
    return text


def supports_bf16() -> bool:
    return (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[INFO] Loading SFT datasets...")
    train_dataset = load_sft_datasets()
    print(f"[INFO] Total training samples: {len(train_dataset)}")

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------
    # 4-bit quantization config
    # -----------------------------
    print("[INFO] Preparing 4-bit quantization config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if supports_bf16() else torch.float16,
    )

    print("[INFO] Loading base model (this may take a bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # -----------------------------
    # LoRA configuration
    # -----------------------------
    # Target modules are standard Llama Q/K/V/O and MLP projections
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # -----------------------------
    # Training configuration
    # -----------------------------
    bf16_flag = supports_bf16()
    fp16_flag = torch.cuda.is_available() and not bf16_flag

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,                 # Start with 1 epoch; can increase later
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,      # Effective batch size = 2 * 4 = 8
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        max_seq_length=1024,
        packing=True,                       # Pack multiple samples into each sequence
        bf16=bf16_flag,
        fp16=fp16_flag,
        report_to=[],                       # Disable W&B etc.
    )

    print("[INFO] Starting SFT training with LoRA...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        formatting_func=formatting_func,
    )

    trainer.train()
    print("[INFO] Training complete. Saving model + tokenizer...")

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"[OK] Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
