"""
HumanEval pass@1 evaluation via vLLM.

Usage:
    python scripts/eval_humaneval.py \\
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --out results/humaneval_baseline.jsonl \\
        --run-name baseline-humaneval

Outputs:
    <out>                      raw per-problem rows (jsonl)
    <out>.summary.json         {pass@1, n_pass, total, ...}
    W&B run (unless --no-wandb): humaneval/pass@1, humaneval/n_correct
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Allow imports from this dir when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

# NOTE: vLLM is imported lazily inside main() so other scripts (e.g.
# train_grpo_trl.py) can import extract_code from this module without needing
# vllm in their env.


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the LAST python code block (models often wrap an example first, then the answer);
    fall back to the raw text stripped."""
    matches = CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def build_eval_script(problem: dict, completion: str) -> str:
    """
    Build a runnable script:
        [function definition] + [test harness] + check(entry_point)

    If the model returned a full `def {entry_point}` we use it as-is.
    Otherwise we treat the completion as the function body and
    prepend the original prompt (which contains the signature + docstring).
    """
    code = extract_code(completion)
    entry = problem["entry_point"]
    if f"def {entry}" in code:
        func_src = code
    else:
        func_src = problem["prompt"] + code
    return func_src + "\n\n" + problem["test"] + f"\ncheck({entry})\n"


def make_chat_prompt(tokenizer, problem: dict) -> str:
    system = (
        "You are an expert Python programmer. "
        "Complete the function given its signature and docstring. "
        "Respond with the full function definition inside a single ```python code block."
    )
    user = (
        "Complete this Python function. Reply with the full function in one ```python block.\n\n"
        f"```python\n{problem['prompt']}```"
    )
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=1, help="samples per problem (pass@k support)")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--exec-timeout", type=float, default=10.0)
    ap.add_argument("--out", default="results/humaneval_raw.jsonl")
    ap.add_argument("--wandb-project", default="code-rl-pipeline")
    ap.add_argument("--run-name", default="baseline-humaneval")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # Lazy imports (kept out of module top so other scripts can reuse extract_code)
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from sandbox_executor import run_code

    print("Loading HumanEval ...")
    ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    print(f"  {len(ds)} problems")

    print(f"Loading model: {args.model}")
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util)
    tokenizer = llm.get_tokenizer()

    sp = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n,
    )

    prompts = [make_chat_prompt(tokenizer, p) for p in ds]
    print(f"Generating {len(prompts)} prompts × {args.n} candidates ...")
    outputs = llm.generate(prompts, sp)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_pass = 0
    n_total = len(ds) * args.n

    with open(args.out, "w") as f:
        for problem, output in zip(ds, outputs):
            for cand in output.outputs:
                completion = cand.text
                script = build_eval_script(problem, completion)
                res = run_code(script, timeout=args.exec_timeout)
                passed = bool(res["passed"])
                if passed:
                    n_pass += 1
                row = {
                    "task_id": problem["task_id"],
                    "passed": passed,
                    "completion": completion,
                    "stderr": (res.get("stderr") or "")[:300],
                    "error": res.get("error"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pass_at_1 = n_pass / n_total
    print(f"\nHumanEval pass@1 = {pass_at_1:.4f}  ({n_pass}/{n_total})")

    summary = {
        "benchmark": "humaneval",
        "model": args.model,
        "pass@1": pass_at_1,
        "n_pass": n_pass,
        "total": n_total,
        "temperature": args.temp,
        "n_samples": args.n,
    }
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary -> {summary_path}")

    if not args.no_wandb:
        import wandb
        wandb.log({"humaneval/pass@1": pass_at_1, "humaneval/n_correct": n_pass})
        wandb.finish()


if __name__ == "__main__":
    main()
