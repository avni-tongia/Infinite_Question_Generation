import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Optional PEFT / QLoRA
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)

    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=2)

    ap.add_argument("--seed", type=int, default=42)

    # LoRA / QLoRA
    ap.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    ap.add_argument("--use_4bit", action="store_true", help="Enable 4-bit quantization (QLoRA)")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                    help="Comma-separated target module names")

    # Mask prompt loss (recommended ON)
    ap.add_argument("--mask_prompt_loss", action="store_true",
                    help="Compute loss only on completion tokens (recommended)")

    # Mixed precision
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")

    return ap.parse_args()


def load_jsonl_dataset(train_file: str, val_file: str):
    data_files = {"train": train_file, "validation": val_file}
    ds = load_dataset("json", data_files=data_files)
    return ds


def build_text(prompt: str, completion: str, tokenizer) -> str:
    """
    Simple format: prompt + two newlines + completion + EOS
    This is compatible across instruct models (Qwen/Mistral) without relying on chat templates.
    """
    prompt = prompt.strip()
    completion = completion.strip()
    eos = tokenizer.eos_token or ""
    return f"{prompt}\n\n{completion}{eos}"


def tokenize_and_mask(example: Dict, tokenizer, max_len: int, mask_prompt_loss: bool) -> Dict:
    prompt = example["prompt"]
    completion = example["completion"]

    full_text = build_text(prompt, completion, tokenizer)

    # Tokenize full text
    tok = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_attention_mask=True,
    )

    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]

    labels = input_ids.copy()

    if mask_prompt_loss:
        # Tokenize the prefix (prompt + "\n\n") so we can mask its tokens in labels
        prefix = f"{prompt.strip()}\n\n"
        prefix_tok = tokenizer(
            prefix,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        prefix_len = len(prefix_tok["input_ids"])

        # Mask labels for prefix tokens
        for i in range(min(prefix_len, len(labels))):
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class DataCollatorForCausalLM:
    tokenizer: any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pad input_ids/attention_mask/labels to max length in batch
        max_len = max(len(f["input_ids"]) for f in features)

        if self.pad_to_multiple_of is not None:
            if max_len % self.pad_to_multiple_of != 0:
                max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        pad_id = self.tokenizer.pad_token_id

        for f in features:
            ids = f["input_ids"]
            am = f["attention_mask"]
            lb = f["labels"]

            pad_len = max_len - len(ids)

            batch_input_ids.append(ids + [pad_id] * pad_len)
            batch_attention_mask.append(am + [0] * pad_len)
            batch_labels.append(lb + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def make_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    # Ensure pad token exists for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    model_kwargs = {"torch_dtype": torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)}

    if args.use_4bit:
        # QLoRA path
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError("BitsAndBytesConfig not available. Install bitsandbytes + recent transformers.") from e

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **{k: v for k, v in model_kwargs.items() if v is not None},
    )

    # LoRA / QLoRA
    if args.use_lora:
        if LoraConfig is None:
            raise RuntimeError("peft is not installed. pip install peft")

        if args.use_4bit:
            if prepare_model_for_kbit_training is None:
                raise RuntimeError("prepare_model_for_kbit_training unavailable. Update peft.")
            model = prepare_model_for_kbit_training(model)

        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    args = parse_args()
    set_seed(args.seed)

    ds = load_jsonl_dataset(args.train_file, args.val_file)

    model, tokenizer = make_model_and_tokenizer(args)

    # Tokenize dataset
    def _map_fn(ex):
        return tokenize_and_mask(ex, tokenizer, args.max_seq_len, args.mask_prompt_loss)

    ds_tok = ds.map(
        _map_fn,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing + masking prompt loss",
    )

    collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,

        bf16=args.bf16,
        fp16=args.fp16,

        report_to="none",

        # Helps stability
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[OK] saved model to {args.output_dir}")


if __name__ == "__main__":
    main()