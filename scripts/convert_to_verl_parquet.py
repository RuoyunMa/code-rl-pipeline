"""
D6 — convert RL prompts to verl Parquet format.

Each row has the schema verl expects for custom-reward-function GRPO:

    {
        "data_source":  str,
        "prompt":       list[{"role": str, "content": str}],   # chat messages
        "ability":      str,
        "reward_model": {
            "style":        "rule",
            "ground_truth": str,    # JSON-encoded {test_harness, entry_point}
        },
        "extra_info": {"task_id": str, "split": str, "index": int},
    }

The custom reward (`scripts/code_reward.py`) reads `ground_truth`, extracts
code from the model response, and runs the test harness in a sandbox.

Outputs (default):
    data/code_train.parquet            # 270 prompts (formal training)
    data/code_val.parquet              #  30 prompts (eval)
    data/code_train_dryrun.parquet     #   5 prompts (M4 dry run)
    data/code_val_dryrun.parquet       #   3 prompts (M4 dry run)

Usage:
    python scripts/convert_to_verl_parquet.py \\
        --source mbpp_train \\
        --train-n 270 --val-n 30 \\
        --out-dir data
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Solve the problem with a single self-contained Python function. "
    "Respond with the function inside a ```python code block, no extra prose."
)


def load_mbpp_train_problems():
    """374 MBPP train problems with assert tests."""
    ds = load_dataset("mbpp", split="train", trust_remote_code=True)
    out = []
    for i, ex in enumerate(ds):
        text = ex.get("text") or ex.get("prompt") or ""
        tests = ex.get("test_list") or []
        setup = ex.get("test_setup_code") or ""
        if not tests:
            continue
        user = (
            f"{text}\n\n"
            f"Your function must satisfy this test:\n```\n{tests[0]}\n```\n\n"
            f"Write only the function definition in a single ```python code block."
        )
        out.append({
            "task_id": f"mbpp/train/{ex.get('task_id', i)}",
            "user_message": user,
            "test_harness": (setup + "\n" if setup else "") + "\n".join(tests),
            "entry_point": "",
        })
    return out


def to_verl_row(problem: dict, split: str, index: int, data_source: str = "code_unit_test_mbpp") -> dict:
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem["user_message"]},
        ],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps({
                "test_harness": problem["test_harness"],
                "entry_point": problem["entry_point"],
            }),
        },
        "extra_info": {
            "task_id": problem["task_id"],
            "split": split,
            "index": index,
        },
    }


def write_parquet(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"  -> {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="mbpp_train", choices=["mbpp_train"])
    ap.add_argument("--train-n", type=int, default=270)
    ap.add_argument("--val-n", type=int, default=30)
    ap.add_argument("--dryrun-train-n", type=int, default=5)
    ap.add_argument("--dryrun-val-n", type=int, default=3)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)

    if args.source == "mbpp_train":
        problems = load_mbpp_train_problems()
    else:
        raise SystemExit(f"unknown source: {args.source}")

    rng.shuffle(problems)
    total_needed = args.train_n + args.val_n
    if len(problems) < total_needed:
        print(f"WARN: only {len(problems)} problems available, requested {total_needed}; "
              f"trimming train_n / val_n proportionally.")
        ratio = len(problems) / total_needed
        args.train_n = int(args.train_n * ratio)
        args.val_n = len(problems) - args.train_n

    train_problems = problems[: args.train_n]
    val_problems = problems[args.train_n : args.train_n + args.val_n]

    print(f"Splits: train={len(train_problems)} val={len(val_problems)}")

    train_rows = [to_verl_row(p, "train", i) for i, p in enumerate(train_problems)]
    val_rows = [to_verl_row(p, "val", i) for i, p in enumerate(val_problems)]
    write_parquet(train_rows, out_dir / "code_train.parquet")
    write_parquet(val_rows, out_dir / "code_val.parquet")

    # Dry-run subsets (re-use the first N of each split)
    dr_train = train_rows[: args.dryrun_train_n]
    dr_val = val_rows[: args.dryrun_val_n]
    write_parquet(dr_train, out_dir / "code_train_dryrun.parquet")
    write_parquet(dr_val, out_dir / "code_val_dryrun.parquet")

    # Sanity: re-load and validate one row
    df = pd.read_parquet(out_dir / "code_train.parquet")
    sample = df.iloc[0].to_dict()
    print("\nSample row schema:")
    for k, v in sample.items():
        if isinstance(v, str) and len(v) > 80:
            v = v[:80] + "..."
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
