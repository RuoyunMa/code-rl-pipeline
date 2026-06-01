"""
MBPP pass@1 evaluation via vLLM.

We use the `sanitized` split (cleaned 257 problems) by default — it's the
common reporting target. Pass `--split full` for the 500-problem test split.

Each problem gives:
    - text:       natural-language description
    - test_list:  3 assertion strings (the function signature is implicit there)
    - test_setup_code (rare): preamble run before the asserts

Strategy:
    Feed `text` + the first assertion (signature hint) to the model in a chat
    prompt. After generation, prepend test_setup_code, append all asserts,
    and run the script. Pass iff exit code == 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow imports from this dir when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from sandbox_executor import run_code  # noqa: E402
from eval_humaneval import extract_code  # noqa: E402  (reuse)


def _description(problem: dict) -> str:
    """MBPP full split uses 'text'; sanitized split uses 'prompt'. Handle both."""
    return problem.get("text") or problem.get("prompt") or ""


def make_chat_prompt(tokenizer, problem: dict) -> str:
    system = (
        "You are an expert Python programmer. "
        "Solve the problem with a single self-contained Python function. "
        "Respond with the function inside a ```python code block, no extra prose."
    )
    test_hint = problem["test_list"][0]
    user = (
        f"{_description(problem)}\n\n"
        f"Your function must satisfy this test:\n```\n{test_hint}\n```\n\n"
        f"Write only the function definition in a single ```python code block."
    )
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_eval_script(problem: dict, completion: str) -> str:
    code = extract_code(completion)
    setup = problem.get("test_setup_code") or ""
    asserts = "\n".join(problem["test_list"])
    return f"{setup}\n{code}\n{asserts}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--split", default="sanitized", choices=["full", "sanitized"])
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--exec-timeout", type=float, default=10.0)
    ap.add_argument("--out", default="results/mbpp_raw.jsonl")
    ap.add_argument("--wandb-project", default="code-rl-pipeline")
    ap.add_argument("--run-name", default="baseline-mbpp")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    print(f"Loading MBPP ({args.split}) ...")
    if args.split == "sanitized":
        ds = load_dataset("mbpp", "sanitized", split="test", trust_remote_code=True)
    else:
        ds = load_dataset("mbpp", split="test", trust_remote_code=True)
    print(f"  {len(ds)} problems · columns: {ds.column_names}")

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
                    "task_id": problem.get("task_id"),
                    "passed": passed,
                    "completion": completion,
                    "stderr": (res.get("stderr") or "")[:300],
                    "error": res.get("error"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pass_at_1 = n_pass / n_total
    print(f"\nMBPP pass@1 ({args.split}) = {pass_at_1:.4f}  ({n_pass}/{n_total})")

    summary = {
        "benchmark": "mbpp",
        "split": args.split,
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
        wandb.log({"mbpp/pass@1": pass_at_1, "mbpp/n_correct": n_pass})
        wandb.finish()


if __name__ == "__main__":
    main()
