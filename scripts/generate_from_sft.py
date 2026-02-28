import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--bucket", required=True, choices=["easy", "hard"])
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    return ap.parse_args()


def build_prompt(bucket: str):
    if bucket == "easy":
        return (
            "Generate an EASY physics question suitable for undergraduate level. "
            "It should require basic formula application and minimal multi-step reasoning. "
            "Provide the question followed by a short solution.\n\n"
        )
    else:
        return (
            "Generate a HARD physics question similar in difficulty to Irodov. "
            "It should require multi-step reasoning and deeper conceptual understanding. "
            "Provide the question followed by a detailed solution.\n\n"
        )


def main():
    args = parse_args()

    print("[INFO] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(args.bucket)

    print(f"[INFO] Generating {args.num_samples} samples for bucket={args.bucket}")

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.num_samples):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

            text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            record = {
                "bucket_prompt": args.bucket,
                "generation_id": f"{args.bucket}_{i}",
                "text": text,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0:
                print(f"[INFO] Generated {i+1}/{args.num_samples}")

    print(f"[OK] Saved generations to {args.out_file}")


if __name__ == "__main__":
    main()