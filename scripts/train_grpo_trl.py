"""
D8/D9 fallback — GRPO via trl.GRPOTrainer (used if verl Docker can't run).

Reward function: run sampled completion against problem-specific unit tests via
sandbox_executor. Reward = +1 if all tests pass, -1 otherwise.

5090 32GB notes:
    Estimated peak ~12-16 GB at default config (LoRA r=16, bs=4, num_gen=4) —
    fits comfortably. The risk is **VRAM creep over long runs** from PyTorch
    allocator fragmentation. Mitigations baked in:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set by activate hook)
      - GRPOConfig.torch_empty_cache_steps=10 — periodic empty_cache()
    If creep still hits past ~25 GB after a few hundred steps, drop
    --num-generations 4 → 2 (still valid GRPO).

Input data format — one jsonl line per RL prompt:
    {
        "prompt":      [{"role": "user", "content": "..."}, ...]   # chat format
        "tests":       "<test harness code as a string>",
        "entry_point": "<function name to invoke in check(...)>"   # optional
    }

If `entry_point` is empty/missing, the test harness is appended to the script
without an explicit `check(...)` call — useful for MBPP-style asserts that
already reference the function by name.

Usage:
    python scripts/train_grpo_trl.py \\
        --sft-model outputs/sft_merged \\
        --data data/rl_prompts.jsonl \\
        --max-steps 30

Why this exists:
    verl is the production target (ByteDance internal stack). If verl Docker
    blocks > 1 day per the M1/M4 fallback rule, swap to this. The README will
    document the rationale — it's still a real GRPO run with verifiable rewards.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch
from datasets import Dataset

# Allow `from sandbox_executor import ...` and `from eval_humaneval import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbox_executor import run_code  # noqa: E402
from eval_humaneval import extract_code  # noqa: E402


def load_rl_prompts(path: str) -> Dataset:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" not in row:
                raise ValueError(f"missing 'prompt' in: {row}")
            row.setdefault("tests", "")
            row.setdefault("entry_point", "")
            rows.append(row)
    return Dataset.from_list(rows)


def make_reward_fn():
    """
    trl.GRPOTrainer passes:
        prompts:      list[str | list[dict]]    (one per problem in the batch)
        completions:  list[str | list[dict]]    (one per generated sample)
        **kwargs:     other dataset columns broadcast to per-completion list

    For a batch of B problems × G generations, lengths are B*G. Per-problem
    columns ('tests', 'entry_point') come pre-broadcasted by trl.
    """
    def reward_fn(prompts, completions, tests=None, entry_point=None, **_) -> List[float]:
        rewards: List[float] = []
        for i, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else completion[0]["content"]
            code = extract_code(text)
            test_block = tests[i] if tests else ""
            entry = entry_point[i] if entry_point else ""
            script = code + "\n\n" + test_block
            if entry:
                script += f"\ncheck({entry})\n"
            else:
                script += "\n"
            res = run_code(script, timeout=10)
            rewards.append(1.0 if res["passed"] else -1.0)
        return rewards
    return reward_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-model", default="outputs/sft_merged",
                    help="Path to SFT-merged model OR HF id")
    ap.add_argument("--data", default="data/rl_prompts.jsonl")
    ap.add_argument("--output-dir", default="outputs/grpo_trl")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--ga", type=int, default=4)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--max-completion-length", type=int, default=768)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--run-name", default="grpo-trl-qwen25coder-1.5b")
    ap.add_argument("--wandb-project", default="code-rl-pipeline")
    args = ap.parse_args()

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Defer heavy imports
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import GRPOTrainer, GRPOConfig

    print(f"Loading tokenizer + model: {args.sft_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    # trl 0.24's GRPOTrainer expects `model.warnings_issued` (dict). In transformers 5.5+
    # this attribute moved/was removed — add a stub before wrapping.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    # Pass gradient_checkpointing via GRPOConfig below — manual enable here
    # conflicts with PEFT wrapping done inside GRPOTrainer.

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
    ds = load_rl_prompts(args.data)
    print(f"  {len(ds)} RL prompts")

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        beta=args.beta,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        report_to="none" if args.no_wandb else "wandb",
        run_name=args.run_name,
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # 5090-32GB long-run fragmentation mitigation: empty CUDA cache every
        # N steps. Cheap; required to avoid VRAM creep on GRPO with HF generate.
        torch_empty_cache_steps=10,
        # Use vLLM for rollout if available (much faster than HF generate)
        use_vllm=False,  # set True only if you've configured vllm in this env
    )

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=[make_reward_fn()],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print(f"Training GRPO (trl) for {args.max_steps} steps ...")
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    print(f"GRPO LoRA saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
