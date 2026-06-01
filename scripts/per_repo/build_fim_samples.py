"""
Build FIM (fill-in-middle) training samples from a target git repo.

For each .py file in the repo (filtered for tests/docs/etc):
    - tokenize with Qwen tokenizer
    - drop too-short or too-long files
    - sample N random (prefix, middle, suffix) splits per file
    - format as Qwen2.5-Coder FIM: <|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>...

Output (one JSON object per line, ready for train_sft.py via --dataset):
    {"problem": "<fim prompt text>", "solution": "<gold middle text>"}

We use 'problem'/'solution' keys to be compatible with train_sft.py's expected
schema (it applies chat template). This means we're going FIM-as-instruction-tuning:
problem = "Complete this code fill-in-middle: <fim_prefix>...<fim_suffix>"
solution = "<middle text>"

Usage:
    python scripts/per_repo/build_fim_samples.py \\
        --repo-dir /tmp/flask \\
        --out data/per_repo/flask/train.jsonl \\
        --out-eval data/per_repo/flask/holdout.jsonl \\
        --n-per-file 8 --train-frac 0.9
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


EXCLUDE_DIR_PATTERNS = ["tests/", "/test_", "/tests/", "docs/", "examples/",
                         "__pycache__", ".eggs", "build/", "dist/", "site-packages/"]


def is_target_file(path: Path, repo_dir: Path) -> bool:
    if path.suffix != ".py":
        return False
    rel = str(path.relative_to(repo_dir))
    if any(p in rel for p in EXCLUDE_DIR_PATTERNS):
        return False
    if path.name.startswith("test_") or path.name.endswith("_test.py"):
        return False
    return True


def sample_fim(text: str, tokenizer, min_tokens: int = 100, max_tokens: int = 4000,
               middle_min: int = 20, middle_max: int = 200,
               n_samples: int = 8, rng: random.Random | None = None):
    """Yield (prefix, middle, suffix) tuples for FIM training."""
    if rng is None:
        rng = random.Random()
    tok_ids = tokenizer.encode(text, add_special_tokens=False)
    n_tok = len(tok_ids)
    if n_tok < min_tokens or n_tok > max_tokens:
        return
    out = []
    for _ in range(n_samples):
        mid_len = rng.randint(middle_min, min(middle_max, n_tok // 4))
        start = rng.randint(20, n_tok - mid_len - 20)
        prefix_ids = tok_ids[:start]
        middle_ids = tok_ids[start:start + mid_len]
        suffix_ids = tok_ids[start + mid_len:]
        prefix = tokenizer.decode(prefix_ids)
        middle = tokenizer.decode(middle_ids)
        suffix = tokenizer.decode(suffix_ids)
        yield prefix, middle, suffix


def make_problem(prefix: str, suffix: str) -> str:
    """Format as instruction so train_sft.py's chat template handles it cleanly."""
    return (
        "Fill in the missing code between the prefix and suffix. "
        "Respond with ONLY the middle code, no markdown or explanation.\n\n"
        "## Prefix:\n```python\n" + prefix + "\n```\n\n"
        "## Suffix:\n```python\n" + suffix + "\n```\n\n"
        "## Middle (complete this):"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True, help="path to cloned repo")
    ap.add_argument("--out", required=True, help="train jsonl out path")
    ap.add_argument("--out-eval", required=True, help="held-out jsonl out path")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--n-per-file", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=0.9,
                    help="fraction of FILES used for train (rest held out)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    repo_dir = Path(args.repo_dir)
    files = sorted(
        f for f in repo_dir.rglob("*.py") if is_target_file(f, repo_dir)
    )
    print(f"Found {len(files)} target .py files in {repo_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(files)
    n_train_files = int(len(files) * args.train_frac)
    train_files = files[:n_train_files]
    eval_files = files[n_train_files:]
    print(f"  train files: {len(train_files)}, eval files: {len(eval_files)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_eval).parent.mkdir(parents=True, exist_ok=True)

    def write_samples(file_list, out_path, label):
        n = 0
        with open(out_path, "w") as f:
            for fp in file_list:
                try:
                    text = fp.read_text(errors="replace")
                except Exception:
                    continue
                for prefix, middle, suffix in sample_fim(
                    text, tok, n_samples=args.n_per_file, rng=rng
                ):
                    sample = {
                        "problem": make_problem(prefix, suffix),
                        "solution": middle,
                        "_file": str(fp.relative_to(repo_dir)),
                    }
                    f.write(json.dumps(sample) + "\n")
                    n += 1
        print(f"  wrote {n} {label} samples -> {out_path}")
        return n

    write_samples(train_files, args.out, "train")
    write_samples(eval_files, args.out_eval, "eval")


if __name__ == "__main__":
    main()
