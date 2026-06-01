"""
D5 — generate N candidates per problem using the SFT-merged model + vLLM.

Default source: MBPP train split (374 problems with assert-style tests).
Use --source jsonl --input PATH to feed your own problems.

Output format (one line per problem):
    {
        "task_id":      str,
        "prompt_text":  str,            # the exact chat-rendered string fed to vLLM
        "user_message": str,            # raw user content (kept for re-templating later)
        "test_harness": str,            # executable test code (asserts or HumanEval-style check)
        "entry_point":  str,            # function name for HumanEval-style check; "" for MBPP
        "completions":  list[str]
    }

Usage:
    python scripts/generate_rollouts.py \\
        --model outputs/sft_merged \\
        --source mbpp_train \\
        --n 4 --temp 0.8 \\
        --out data/rollouts.jsonl

Memory: vLLM with --gpu-mem-util 0.85 on 5090-32GB; 1.5B fits easily (~27 GB static reservation, KV capacity ~785k tokens).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def load_mbpp_train():
    """MBPP full train split (374 problems)."""
    ds = load_dataset("mbpp", split="train", trust_remote_code=True)
    rows = []
    for i, ex in enumerate(ds):
        text = ex.get("text") or ex.get("prompt") or ""
        tests = ex.get("test_list") or []
        setup = ex.get("test_setup_code") or ""
        rows.append({
            "task_id": f"mbpp/train/{ex.get('task_id', i)}",
            "user_message": (
                f"{text}\n\n"
                f"Your function must satisfy this test:\n```\n{tests[0] if tests else ''}\n```\n\n"
                f"Write only the function definition in a single ```python code block."
            ),
            "test_harness": (setup + "\n" if setup else "") + "\n".join(tests),
            "entry_point": "",
        })
    return rows


def load_jsonl_source(path: str):
    """
    Custom jsonl. Each line should have:
        task_id, user_message (or text), test_harness, entry_point (optional)
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            rows.append({
                "task_id": ex["task_id"],
                "user_message": ex.get("user_message") or ex["text"],
                "test_harness": ex["test_harness"],
                "entry_point": ex.get("entry_point", ""),
            })
    return rows


SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Solve the problem with a single self-contained Python function. "
    "Respond with the function inside a ```python code block, no extra prose."
)


def make_prompt_text(tokenizer, user_message: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/sft_merged",
                    help="vLLM-loadable model path or HF id")
    ap.add_argument("--source", default="mbpp_train",
                    choices=["mbpp_train", "jsonl"])
    ap.add_argument("--input", default=None, help="path to source jsonl (for --source jsonl)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of problems (e.g. 400)")
    ap.add_argument("--n", type=int, default=4, help="candidates per problem")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--out", default="data/rollouts.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.source == "jsonl":
        if not args.input:
            raise SystemExit("--source jsonl requires --input PATH")
        problems = load_jsonl_source(args.input)
    else:
        problems = load_mbpp_train()

    if args.limit:
        problems = problems[: args.limit]
    print(f"Source: {args.source} · {len(problems)} problems · {args.n} candidates each "
          f"= {len(problems) * args.n} total rollouts")

    print(f"Loading model: {args.model}")
    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util, seed=args.seed)
    tokenizer = llm.get_tokenizer()

    prompt_texts: List[str] = [make_prompt_text(tokenizer, p["user_message"]) for p in problems]

    # NOTE: do NOT set seed= on SamplingParams — vLLM uses the same seed for
    # all `n` candidates of one request, which collapses diversity. Global
    # seed is set on LLM(seed=...) above for reproducibility of the run.
    sp = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n,
    )

    print(f"Generating ...")
    outputs = llm.generate(prompt_texts, sp)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for problem, prompt_text, output in zip(problems, prompt_texts, outputs):
            row = {
                "task_id": problem["task_id"],
                "prompt_text": prompt_text,
                "user_message": problem["user_message"],
                "test_harness": problem["test_harness"],
                "entry_point": problem["entry_point"],
                "completions": [c.text for c in output.outputs],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(problems)} problems × {args.n} candidates -> {args.out}")


if __name__ == "__main__":
    main()
