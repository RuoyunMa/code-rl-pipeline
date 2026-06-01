"""
D2 PM — Prep SFT data: download Magicoder-OSS-Instruct-75K, sample, write to JSONL.

Output schema (one JSON object per line):
    {"problem": "...", "solution": "..."}

train_sft.py loads this JSONL via load_dataset("json", data_files=...) and trains.

Usage:
    python scripts/prep_sft_data.py --out data/sft.jsonl --n 5000

Filters applied:
    - Drop rows with empty problem or solution
    - Drop rows where (problem + solution) tokenize to > max_tokens
      (default 1800 to leave room for chat template overhead within 2048 ctx)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ise-uiuc/Magicoder-OSS-Instruct-75K")
    ap.add_argument("--problem-field", default="problem")
    ap.add_argument("--solution-field", default="solution")
    ap.add_argument("--n", type=int, default=5000, help="Number of samples to keep")
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=1800,
                    help="Drop samples whose problem+solution exceeds this token budget")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    help="Tokenizer used to enforce --max-tokens budget")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    print(f"  total rows: {len(ds)} · columns: {ds.column_names}")

    pf, sf = args.problem_field, args.solution_field
    if pf not in ds.column_names or sf not in ds.column_names:
        raise ValueError(
            f"Dataset columns {ds.column_names} don't contain "
            f"'{pf}' / '{sf}'. Override --problem-field / --solution-field."
        )

    # Load tokenizer for length filtering
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Shuffle deterministically and iterate, collecting until we hit n.
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    kept = []
    n_empty = 0
    n_long = 0
    for idx in indices:
        if len(kept) >= args.n:
            break
        row = ds[idx]
        problem = (row.get(pf) or "").strip()
        solution = (row.get(sf) or "").strip()
        if not problem or not solution:
            n_empty += 1
            continue
        # token-budget filter
        t = len(tok.encode(problem)) + len(tok.encode(solution))
        if t > args.max_tokens:
            n_long += 1
            continue
        kept.append({"problem": problem, "solution": solution})

    print(f"Kept {len(kept)} / target {args.n}  "
          f"(dropped: empty={n_empty}, too-long={n_long})")

    with out_path.open("w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote -> {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    # Quick sanity dump of first sample
    print("\n--- First sample preview ---")
    print(f"problem (first 200 chars):  {kept[0]['problem'][:200]}")
    print(f"solution (first 200 chars): {kept[0]['solution'][:200]}")


if __name__ == "__main__":
    main()
