"""
Full fine-tuning (no LoRA) variant of train_sft.py.

Used for the LoRA-vs-FullFT ablation. Bypasses Unsloth (which is LoRA-focused);
uses vanilla transformers + trl.SFTTrainer with 8-bit AdamW + gradient checkpointing.

Memory budget on 5090-32GB (1.5B + Full FT bf16 + 8-bit AdamW + bs=4 ga=8 seq=2048):
    Estimated ~20-22 GB peak. Tight on 32GB; if OOM, drop bs=2 ga=16.

Usage:
    python scripts/train_sft_full.py \\
        --output-dir outputs/sft_fullft_lr5e5 \\
        --lr 5e-5 \\
        --no-wandb

Output: saved as a regular HF model dir (no LoRA adapter), directly vllm-loadable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--dataset", default="data/sft.jsonl")
    ap.add_argument("--problem-field", default="problem")
    ap.add_argument("--solution-field", default="solution")
    ap.add_argument("--output-dir", default="outputs/sft_fullft")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=4,
                    help="per-device batch — Full FT is heavier than LoRA")
    ap.add_argument("--ga", type=int, default=8,
                    help="grad accumulation — gives effective bs=32 to match LoRA runs")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--run-name", default="sft-fullft")
    ap.add_argument("--wandb-project", default="code-rl-pipeline")
    args = ap.parse_args()

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import SFTTrainer, SFTConfig

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # Same trl 0.24 + transformers 5.5 stub patch as train_dpo / train_grpo
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model.config.use_cache = False  # required for grad checkpointing
    # Don't call model.gradient_checkpointing_enable() here — let SFTConfig
    # handle it below (trl wraps the model and a manual enable can conflict).

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Full FT mode: {n_params/1e6:.1f}M params, all trainable")

    print(f"Loading dataset: {args.dataset}")
    if args.dataset.endswith((".jsonl", ".json")) or os.path.isfile(args.dataset):
        ds = load_dataset("json", data_files=args.dataset, split="train")
        print(f"  loaded local jsonl: {len(ds)} samples")
    else:
        ds = load_dataset(args.dataset, split="train")
        print(f"  loaded HF: {len(ds)} samples, columns: {ds.column_names}")

    pf, sf = args.problem_field, args.solution_field
    if pf not in ds.column_names or sf not in ds.column_names:
        raise ValueError(f"Dataset columns {ds.column_names} don't contain {pf}/{sf}")

    def fmt(example):
        msgs = [
            {"role": "user", "content": example[pf]},
            {"role": "assistant", "content": example[sf]},
        ]
        return {"text": tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )}

    ds = ds.map(fmt, num_proc=4, remove_columns=ds.column_names)

    cfg = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none" if args.no_wandb else "wandb",
        run_name=args.run_name,
        max_length=args.max_seq_length,   # trl 0.24 renamed from max_seq_length
        dataset_text_field="text",
        seed=args.seed,
        packing=False,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=cfg,
    )

    print(f"Training Full FT for {args.epochs} epochs at lr={args.lr} ...")
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model -> {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
