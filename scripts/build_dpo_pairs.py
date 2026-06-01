"""
D6 — convert rollouts.jsonl into DPO preference pairs by running unit tests.

Pipeline:
    rollouts.jsonl (from generate_rollouts.py)
        → for each completion: extract code → run test harness in sandbox → pass/fail
        → group by problem; if pass>=1 and fail>=1, emit (chosen, rejected) pairs
        → write data/dpo_pairs.jsonl in trl-compatible {prompt, chosen, rejected} format

The reward is binary (passed unit tests vs not). Among passing completions we
prefer the SHORTEST one as `chosen` (cheaper to generate, less likely to be
gaming length). Among failing we pick the LONGEST as `rejected` (often
verbose buggy attempts).

Multiple pairs per problem can be emitted via --pairs-per-problem.

Usage:
    python scripts/build_dpo_pairs.py \\
        --rollouts data/rollouts.jsonl \\
        --out data/dpo_pairs.jsonl \\
        --pairs-per-problem 2 \\
        --workers 8

Also writes a sidecar `<out>.stats.json` with pass-rate and pair-yield stats.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sandbox_executor import run_code  # noqa: E402
from eval_humaneval import extract_code  # noqa: E402


def build_eval_script(completion: str, test_harness: str, entry_point: str) -> str:
    code = extract_code(completion)
    script = code + "\n\n" + (test_harness or "")
    if entry_point:
        script += f"\ncheck({entry_point})\n"
    else:
        script += "\n"
    return script


def grade_one(args: Tuple[int, int, str, str, str]) -> Tuple[int, int, bool]:
    """Worker: returns (problem_idx, completion_idx, passed)."""
    pi, ci, completion, test_harness, entry = args
    script = build_eval_script(completion, test_harness, entry)
    res = run_code(script, timeout=10)
    return pi, ci, bool(res["passed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="data/rollouts.jsonl")
    ap.add_argument("--out", default="data/dpo_pairs.jsonl")
    ap.add_argument("--pairs-per-problem", type=int, default=2,
                    help="cap number of (chosen, rejected) pairs emitted per problem")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel sandbox workers")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading rollouts: {args.rollouts}")
    problems = []
    with open(args.rollouts) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))
    print(f"  {len(problems)} problems")

    # Build the work list
    work = []
    for pi, prob in enumerate(problems):
        for ci, comp in enumerate(prob["completions"]):
            work.append((pi, ci, comp, prob["test_harness"], prob["entry_point"]))
    print(f"Grading {len(work)} completions across {args.workers} workers ...")

    # Parallel grade
    pass_matrix: List[List[bool]] = [[False] * len(p["completions"]) for p in problems]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(grade_one, w) for w in work]
        for fut in as_completed(futures):
            pi, ci, passed = fut.result()
            pass_matrix[pi][ci] = passed
            done += 1
            if done % 200 == 0 or done == len(work):
                print(f"  graded {done}/{len(work)}")

    # Build pairs
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    n_problems_with_pairs = 0
    n_passes = 0
    n_total = 0
    with open(args.out, "w") as f:
        for prob, passes in zip(problems, pass_matrix):
            n_total += len(passes)
            n_passes += sum(passes)
            comps = prob["completions"]
            passed_idx = [i for i, p in enumerate(passes) if p]
            failed_idx = [i for i, p in enumerate(passes) if not p]
            if not passed_idx or not failed_idx:
                continue

            # Prefer shortest pass / longest fail for first pair, then random pairs
            passed_sorted = sorted(passed_idx, key=lambda i: len(comps[i]))
            failed_sorted = sorted(failed_idx, key=lambda i: -len(comps[i]))

            pair_candidates = []
            pair_candidates.append((passed_sorted[0], failed_sorted[0]))
            # supplement with random pairs (without duplicates)
            extra_seen = {pair_candidates[0]}
            for _ in range(args.pairs_per_problem * 4):
                p = rng.choice(passed_idx)
                r = rng.choice(failed_idx)
                if (p, r) not in extra_seen:
                    extra_seen.add((p, r))
                    pair_candidates.append((p, r))
                if len(pair_candidates) >= args.pairs_per_problem:
                    break

            problem_pairs = 0
            for pi_, ri_ in pair_candidates[: args.pairs_per_problem]:
                row = {
                    "prompt": prob["prompt_text"],
                    "chosen": comps[pi_],
                    "rejected": comps[ri_],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_pairs += 1
                problem_pairs += 1
            if problem_pairs > 0:
                n_problems_with_pairs += 1

    pass_rate = n_passes / n_total if n_total else 0.0
    stats = {
        "n_problems": len(problems),
        "n_completions_total": n_total,
        "n_completions_passed": n_passes,
        "pass_rate": pass_rate,
        "n_problems_with_pairs": n_problems_with_pairs,
        "n_pairs_emitted": n_pairs,
        "pairs_per_problem_cap": args.pairs_per_problem,
    }
    Path(args.out + ".stats.json").write_text(json.dumps(stats, indent=2))
    print()
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {n_pairs} pairs -> {args.out}")


if __name__ == "__main__":
    main()
