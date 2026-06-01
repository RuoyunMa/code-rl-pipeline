"""
Convert data/rollouts.jsonl (from generate_rollouts.py) into the schema that
train_grpo_trl.py expects:

    {
        "prompt":      [{"role": "user", "content": "..."}, ...],
        "tests":       "<test harness code>",
        "entry_point": "<function name>"   # empty for mbpp-style asserts
    }

The system message is included so trl.GRPOTrainer's tokenizer can apply the
chat template at training time.

Usage:
    python scripts/prep_rl_prompts.py \\
        --rollouts data/rollouts.jsonl \\
        --out data/rl_prompts.jsonl \\
        --train-n 300
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Solve the problem with a single self-contained Python function. "
    "Respond with the function inside a ```python code block, no extra prose."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="data/rollouts.jsonl")
    ap.add_argument("--out", default="data/rl_prompts.jsonl")
    ap.add_argument("--train-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rows = []
    with open(args.rollouts) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["user_message"]},
                ],
                "tests": r["test_harness"],
                "entry_point": r.get("entry_point", ""),
                "task_id": r["task_id"],
            })

    rng.shuffle(rows)
    if args.train_n and args.train_n < len(rows):
        rows = rows[: args.train_n]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} RL prompts -> {args.out}")


if __name__ == "__main__":
    main()
