"""
LiveCodeBench (v6) pass@1 evaluation via vLLM, LeetCode-style problems only.

Dataset: bzantium/livecodebench (1055 problems, May 2023 – Apr 2025).
We filter to platform=="leetcode" (444 problems) for clean functional-test eval.
AtCoder/Codeforces use stdin/stdout and need a different harness — skipped here.

Contamination control:
    Qwen2.5-Coder-1.5B was released 2024-09-18. Use --post-cutoff to keep only
    problems with contest_date > 2024-09-18 for true post-training-cutoff eval.

Output format (per problem):
    {"question_id", "passed", "n_tests", "n_passed", "difficulty", "contest_date"}

Usage:
    python scripts/eval_livecodebench.py \\
        --model outputs/dpo_v2_merged \\
        --out results/livecodebench_dpo_v2.jsonl \\
        --post-cutoff      # optional: filter to post-2024-09 problems

Summary written to <out>.summary.json with:
    pass@1, broken down by difficulty and by date-window.
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def extract_code(text: str) -> str:
    """Pull the LAST python code block; fallback to raw stripped text."""
    matches = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def decode_private_tests(blob: str):
    """LCB encodes private_test_cases as base64(zlib(pickle(json_string)))."""
    try:
        inner = pickle.loads(zlib.decompress(base64.b64decode(blob)))
        return json.loads(inner)
    except Exception:
        return []


def get_method_name(starter_code: str) -> str:
    """Extract the method name from a LeetCode Solution class starter."""
    m = re.search(r"def\s+(\w+)\s*\(\s*self", starter_code)
    return m.group(1) if m else "solve"


def make_chat_prompt(tokenizer, problem: dict) -> str:
    system = (
        "You are an expert Python programmer solving a competitive programming problem. "
        "Implement the Solution class method. "
        "Respond with the complete Solution class inside a single ```python code block, "
        "no extra prose."
    )
    user = (
        problem["question_content"]
        + "\n\n```python\n" + problem["starter_code"] + "```\n\n"
        + "Complete the class. Use type hints from typing as needed."
    )
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_eval_script(generated: str, method_name: str, test_cases: list, timeout_each: int = 5) -> str:
    """
    Build a single-process script that runs all test cases for this problem.
    Exit code 0 ⇒ all tests passed; non-zero ⇒ at least one failed.

    Uses json.loads for input args list, json.loads for expected output where parseable,
    else falls back to plain string comparison.
    """
    code = extract_code(generated)
    tc_json = json.dumps(test_cases)
    harness = f"""
from typing import List, Tuple, Optional, Dict, Set, Any
import json
import sys

{code}

_test_cases = json.loads({tc_json!r})
_method = {method_name!r}
_failed = 0
_failed_msgs = []
for i, tc in enumerate(_test_cases):
    try:
        # LCB convention: 'input' is newline-separated JSON-encoded args, one per line.
        # For single-arg methods, the whole input is ONE JSON value (no newlines).
        raw_in = tc["input"].strip()
        if "\\n" in raw_in:
            args = [json.loads(line) for line in raw_in.split("\\n") if line.strip()]
        else:
            args = [json.loads(raw_in)]

        expected_raw = tc["output"]
        try:
            expected = json.loads(expected_raw)
        except Exception:
            expected = expected_raw  # plain string

        sol = Solution()
        got = getattr(sol, _method)(*args)

        ok = (got == expected)
        if not ok:
            try:
                ok = (json.dumps(got, sort_keys=True) == json.dumps(expected, sort_keys=True))
            except Exception:
                pass
        if not ok:
            _failed += 1
            _failed_msgs.append("tc[{{}}] got={{!r}} expected={{!r}}".format(i, got, expected))
    except Exception as e:
        _failed += 1
        _failed_msgs.append("tc[{{}}] {{}}: {{}}".format(i, type(e).__name__, str(e)[:80]))

if _failed:
    print("FAILED {{}}/{{}}: ".format(_failed, len(_test_cases)) + " ; ".join(_failed_msgs[:5]))
    sys.exit(1)
else:
    print("PASSED {{}}/{{}}".format(len(_test_cases), len(_test_cases)))
    sys.exit(0)
"""
    return harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--exec-timeout", type=int, default=10)
    ap.add_argument("--out", default="results/livecodebench.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of problems (debug)")
    ap.add_argument("--post-cutoff", action="store_true",
                    help="filter to contest_date > 2024-09-18 (Qwen2.5-Coder release)")
    ap.add_argument("--platform", default="leetcode",
                    help="comma-sep list of platforms to keep (default: leetcode only)")
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from sandbox_executor import run_code

    print("Loading bzantium/livecodebench ...")
    ds = load_dataset("bzantium/livecodebench", split="test", trust_remote_code=False)
    print(f"  total: {len(ds)} problems")

    platforms = {s.strip() for s in args.platform.split(",")}
    filtered = [e for e in ds if e["platform"] in platforms]
    print(f"  after platform filter ({platforms}): {len(filtered)}")

    if args.post_cutoff:
        cutoff = "2024-09-18"
        filtered = [e for e in filtered if (e["contest_date"] or "")[:10] > cutoff]
        print(f"  after post-cutoff (>{cutoff}): {len(filtered)}")

    if args.limit:
        filtered = filtered[: args.limit]
        print(f"  --limit={args.limit}, using {len(filtered)}")

    print(f"Loading model: {args.model}")
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util)
    tokenizer = llm.get_tokenizer()

    sp = SamplingParams(temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens, n=args.n)

    prompts = [make_chat_prompt(tokenizer, p) for p in filtered]
    print(f"Generating {len(prompts)} prompts ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    gen_sec = time.time() - t0
    print(f"  generation done in {gen_sec:.1f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_pass = 0
    rows = []
    print("Grading ...")
    t0 = time.time()
    for problem, output in zip(filtered, outputs):
        method = get_method_name(problem["starter_code"])
        # Use both public + private test cases for maximum strictness
        try:
            pub = json.loads(problem["public_test_cases"])
        except Exception:
            pub = []
        priv = decode_private_tests(problem["private_test_cases"])
        all_tests = (pub or []) + (priv or [])
        if not all_tests:
            # No tests to grade — skip (treat as fail)
            rows.append({"question_id": problem["question_id"], "passed": False,
                         "n_tests": 0, "difficulty": problem["difficulty"],
                         "contest_date": str(problem["contest_date"]), "error": "no_tests"})
            continue

        for cand in output.outputs:
            script = build_eval_script(cand.text, method, all_tests)
            res = run_code(script, timeout=args.exec_timeout)
            passed = bool(res["passed"])
            if passed:
                n_pass += 1
            rows.append({
                "question_id": problem["question_id"],
                "passed": passed,
                "n_tests": len(all_tests),
                "difficulty": problem["difficulty"],
                "contest_date": str(problem["contest_date"]),
                "stderr_head": (res.get("stderr") or "")[:300],
            })
            break  # only score first candidate for pass@1; loop here supports n>1 later
    grade_sec = time.time() - t0
    print(f"  grading done in {grade_sec:.1f}s")

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    pass_at_1 = n_pass / len(rows) if rows else 0.0
    # Breakdowns
    by_diff = defaultdict(lambda: [0, 0])  # [n_pass, total]
    by_post_cutoff = [0, 0]  # [n_pass, total] for post-cutoff problems
    cutoff = "2024-09-18"
    for r in rows:
        by_diff[r["difficulty"]][1] += 1
        if r["passed"]:
            by_diff[r["difficulty"]][0] += 1
        if (r["contest_date"] or "")[:10] > cutoff:
            by_post_cutoff[1] += 1
            if r["passed"]:
                by_post_cutoff[0] += 1

    summary = {
        "benchmark": "livecodebench_v6",
        "model": args.model,
        "pass@1": pass_at_1,
        "n_pass": n_pass,
        "total": len(rows),
        "temperature": args.temp,
        "n_samples": args.n,
        "gen_sec": round(gen_sec, 1),
        "grade_sec": round(grade_sec, 1),
        "by_difficulty": {k: {"pass": v[0], "total": v[1],
                              "pass@1": (v[0] / v[1]) if v[1] else 0.0}
                          for k, v in by_diff.items()},
        "post_cutoff_subset": {
            "cutoff_date": cutoff,
            "pass": by_post_cutoff[0],
            "total": by_post_cutoff[1],
            "pass@1": (by_post_cutoff[0] / by_post_cutoff[1]) if by_post_cutoff[1] else 0.0,
        },
        "platforms": list(platforms),
    }
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nLiveCodeBench pass@1 = {pass_at_1:.4f}  ({n_pass}/{len(rows)})")
    print(f"  by difficulty: {json.dumps(summary['by_difficulty'], indent=2)}")
    print(f"  post-cutoff ({cutoff}+): {summary['post_cutoff_subset']}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
