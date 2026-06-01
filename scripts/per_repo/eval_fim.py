"""
Per-repo FIM held-out evaluation.

For each held-out FIM sample, generate the missing middle with the model and
compute:
  - exact_match: bool, predicted middle == gold middle after strip
  - edit_similarity: token-level edit ratio (0..1, 1 = identical)
  - symbol_recall: fraction of identifiers from gold middle present in predicted

Usage:
    python scripts/per_repo/eval_fim.py \\
        --model outputs/per_repo/flask_lora_merged \\
        --data data/per_repo/flask/holdout.jsonl \\
        --out results/per_repo/flask_eval.jsonl \\
        --limit 200
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher


def extract_middle(text: str) -> str:
    """The prompt asks for raw middle code. Strip any markdown code blocks if present."""
    m = re.search(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def token_edit_ratio(a: str, b: str) -> float:
    """0..1, 1=identical, via SequenceMatcher on whitespace tokens."""
    ta = a.split()
    tb = b.split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return SequenceMatcher(None, ta, tb).ratio()


IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def symbols(s: str) -> Counter:
    return Counter(IDENT_RE.findall(s))


def symbol_recall(gold: str, pred: str) -> float:
    g = symbols(gold)
    p = symbols(pred)
    if not g:
        return 1.0
    total_gold = sum(g.values())
    overlap = sum(min(g[k], p[k]) for k in g)
    return overlap / total_gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = []
    with open(args.data) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Eval set: {len(rows)} held-out FIM samples")

    print(f"Loading model: {args.model}")
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=args.temp, max_tokens=args.max_tokens)

    prompts = []
    for r in rows:
        msgs = [
            {"role": "user", "content": r["problem"]},
        ]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    print("Generating...")
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_sec = time.time() - t0
    print(f"  generated in {gen_sec:.1f}s")

    n_exact = 0
    total_edit = 0.0
    total_sym = 0.0
    out_rows = []
    for r, o in zip(rows, outs):
        gold = r["solution"]
        pred_raw = o.outputs[0].text
        pred = extract_middle(pred_raw)
        exact = (pred.strip() == gold.strip())
        edit = token_edit_ratio(gold, pred)
        sym = symbol_recall(gold, pred)
        if exact:
            n_exact += 1
        total_edit += edit
        total_sym += sym
        out_rows.append({
            "file": r.get("_file", ""),
            "exact_match": exact,
            "edit_similarity": edit,
            "symbol_recall": sym,
            "gold": gold[:200],
            "pred": pred[:200],
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    summary = {
        "model": args.model,
        "n": len(out_rows),
        "exact_match": n_exact / len(out_rows) if out_rows else 0,
        "edit_similarity_mean": total_edit / len(out_rows) if out_rows else 0,
        "symbol_recall_mean": total_sym / len(out_rows) if out_rows else 0,
        "gen_sec": round(gen_sec, 1),
    }
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print("FIM eval summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
