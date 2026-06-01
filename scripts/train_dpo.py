"""
D7 — DPO with trl + LoRA on top of the SFT'd model.

Reads preference pairs from a jsonl file with one of these schemas per line:

    {"prompt": "...", "chosen": "...", "rejected": "..."}      # plain text
    {"prompt": [...], "chosen": [...], "rejected": [...]}      # chat-formatted

For coding pipeline we generate (prompt, chosen, rejected) by:
    - chosen   = sampled completion that PASSED unit tests
    - rejected = sampled completion that FAILED unit tests
See scripts/build_dpo_pairs.py (D6).

Usage:
    python scripts/train_dpo.py \\
        --sft-model outputs/sft_merged \\
        --data data/dpo_pairs.jsonl \\
        --output-dir outputs/dpo

Memory budget on 5090-32GB (1.5B + LoRA r=16 + bs=4 × ga=4 + max_length=2048):
    ~10-13 GB peak. Comfortable. Dominant transient is the chosen+rejected
    logits over vocab=151,936 (~5 GB). ref_model=None + peft_config uses the
    adapter-disable trick so we don't pay for a second model copy.
    No QLoRA needed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset


def load_pairs(path: str) -> Dataset:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("prompt", "chosen", "rejected"):
                if key not in row:
                    raise ValueError(f"missing key '{key}' in: {row}")
            rows.append(row)
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-model", default="outputs/sft_merged",
                    help="Path to SFT-merged model OR HF id")
    ap.add_argument("--data", default="data/dpo_pairs.jsonl")
    ap.add_argument("--output-dir", default="outputs/dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--ga", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-prompt-length", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--run-name", default="dpo-qwen25coder-1.5b")
    ap.add_argument("--wandb-project", default="code-rl-pipeline")
    args = ap.parse_args()

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Defer imports — heavy.
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import DPOTrainer, DPOConfig

    print(f"Loading tokenizer + model: {args.sft_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False  # required for gradient checkpointing
    # trl 0.24's DPOTrainer expects `model.warnings_issued` (dict). In transformers 5.5+
    # this attribute moved or was removed for non-instrumented models — add a stub.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    # NOTE: do NOT call model.gradient_checkpointing_enable() here.
    # When peft_config is passed below, DPOTrainer wraps the model with PEFT
    # internally; manual GC enable on the unwrapped base gets clobbered or
    # breaks the input-grad hook. We pass gradient_checkpointing=True via
    # DPOConfig instead (see below).

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    print(f"Loading data: {args.data}")
    ds = load_pairs(args.data)
    print(f"  {len(ds)} preference pairs")

    cfg = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none" if args.no_wandb else "wandb",
        run_name=args.run_name,
        seed=args.seed,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ref_model=None + peft_config: trl uses the base (pre-LoRA) model as reference,
    # which is exactly what we want for DPO on top of an SFT'd LoRA stack.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,  # trl 0.13: replaces deprecated tokenizer=
        peft_config=peft_config,
    )

    print(f"Training DPO for {args.epochs} epoch(s) ...")
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    print(f"DPO LoRA saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
