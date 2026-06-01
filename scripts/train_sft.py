"""
D3 — SFT with Unsloth + LoRA on Qwen2.5-Coder-1.5B-Instruct.

Default dataset: ise-uiuc/Magicoder-OSS-Instruct-75K (fields: 'problem', 'solution').
Compatible with any HF code-instruction dataset by overriding the field names.

Outputs:
    {output_dir}/             LoRA adapter
    {output_dir}_merged/      16-bit merged weights for vLLM inference

Usage:
    python scripts/train_sft.py \\
        --output-dir outputs/sft \\
        --n-samples 5000 \\
        --epochs 2

Memory budget on 5090-32GB (1.5B + LoRA r=32 + bs=8 × ga=4 + max_seq=2048):
    ~6-8 GB peak. Huge headroom. (LoRA + grad-checkpointing keeps activations
    and optimizer state tiny; the dominant cost is the 3 GB bf16 base weights.)
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
    ap.add_argument("--dataset", default="data/sft.jsonl",
                    help="Either HF dataset id (e.g. ise-uiuc/Magicoder-OSS-Instruct-75K) "
                         "or path to a local .jsonl file produced by prep_sft_data.py")
    ap.add_argument("--problem-field", default="problem")
    ap.add_argument("--solution-field", default="solution")
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--output-dir", default="outputs/sft")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--target-modules", default="all",
                    help="'all' (q,k,v,o,gate,up,down) | 'attn' (q,k,v,o) | 'mlp' (gate,up,down) | comma-list")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="QLoRA: load base model in nf4. Needed for 14B+ on 32GB GPU; "
                         "also works for 7B if memory pressure")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ga", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--run-name", default="sft-qwen25coder-1.5b")
    ap.add_argument("--wandb-project", default="code-rl-pipeline")
    ap.add_argument("--save-merged", action="store_true",
                    help="Also save 16-bit merged weights for vLLM serving")
    args = ap.parse_args()

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Import after argparse — Unsloth patches torch on import.
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    print(f"Loading model: {args.model}  (load_in_4bit={args.load_in_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=args.load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve --target-modules
    if args.target_modules == "all":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    elif args.target_modules == "attn":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif args.target_modules == "mlp":
        target_modules = ["gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = [s.strip() for s in args.target_modules.split(",") if s.strip()]
    print(f"  LoRA target_modules: {target_modules}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    print(f"Loading dataset: {args.dataset}")
    if args.dataset.endswith((".jsonl", ".json")) or os.path.isfile(args.dataset):
        # Local jsonl produced by prep_sft_data.py — already sampled + filtered
        ds = load_dataset("json", data_files=args.dataset, split="train")
        print(f"  loaded local jsonl: {len(ds)} samples (no further sampling)")
    else:
        # HF dataset id — load full + sample
        ds = load_dataset(args.dataset, split="train")
        if args.n_samples and args.n_samples < len(ds):
            ds = ds.shuffle(seed=args.seed).select(range(args.n_samples))
        print(f"  {len(ds)} samples · columns: {ds.column_names}")

    pf, sf = args.problem_field, args.solution_field
    if pf not in ds.column_names or sf not in ds.column_names:
        raise ValueError(
            f"Dataset columns {ds.column_names} don't contain "
            f"--problem-field='{pf}' / --solution-field='{sf}'. Override flags."
        )

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
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        seed=args.seed,
        packing=False,
        optim="adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # trl 0.13: replaces deprecated tokenizer=
        train_dataset=ds,
        args=cfg,
    )

    print(f"Training for {args.epochs} epochs ...")
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving LoRA adapter -> {args.output_dir}")
    trainer.save_model(args.output_dir)

    if args.save_merged:
        merged_dir = f"{args.output_dir}_merged"
        print(f"Saving merged 16-bit weights -> {merged_dir}")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"Merged model ready for vLLM serving at: {merged_dir}")


if __name__ == "__main__":
    main()
