# Benchmark plan — post-pivot 2026-05-19

After raising the contamination concern + asking to align with Cursor Composer 2's
benchmark stack (Terminal-Bench 2.0 / SWE-Bench Multilingual), here's the revised
benchmark strategy.

## Background

HumanEval (2021) and MBPP (2021) are saturated for modern code models:
- Qwen2.5-Coder-1.5B base scores 73% on HumanEval / 63% on MBPP — close to ceiling
- Contamination check (`results/contamination_check.md`) confirms our SFT data is clean,
  but Qwen's 5.5T-token pretraining has likely absorbed HE-distribution implicitly
- Single-problem noise at n=164 is ~3pp; our typical post-training gains are 0-3pp
  → benchmark resolution barely exceeds the noise floor

Cursor Composer 2 (2025) uses **Terminal-Bench 2.0, SWE-Bench Multilingual, CursorBench**
as headline metrics. For our 1.5B/7B fine-tuning project these are not all feasible.

## Benchmark choices

### Tier 1 — primary discriminators (every run)

#### LiveCodeBench v6 ★ new addition

- **Status:** primary metric going forward
- **Size:** 1055 problems (May 2023 – Apr 2025), updated quarterly
- **Why:** contamination-free by construction (release-date tagging); we can filter to
  problems released AFTER Qwen2.5-Coder-1.5B's 2024-09 cutoff
- **Problem type:** competitive-programming style (LeetCode/Codeforces/AtCoder),
  signature + input/output test cases
- **Expected score on 1.5B base:** ~15-25% pass@1 (Qwen team reports 19.6 on v5 for similar size)
- **Eval cost:** vLLM batched generation — ~3 min for full 1055 problems on 5090 with 1.5B
- **Integration:** new `scripts/eval_livecodebench.py` (reuse our sandbox_executor)
- **Reporting:** pass@1, plus break out by difficulty (easy/medium/hard) and by release date
  (pre-cutoff vs post-cutoff) to expose contamination-induced inflation

### Tier 2 — legacy reference (kept for continuity)

#### HumanEval (164) + MBPP-sanitized (257)

- **Status:** kept as sanity-check
- **Why kept:** all prior runs have these numbers; needed for continuity tables; well-understood
- **Caveat:** mark as "saturated / partially contaminated via Qwen pretraining" in writeups

### Tier 3 — feasible for 7B+, skip for 1.5B

#### SWE-Bench-Verified small subset (50-100 of 500 tasks)

- **Status:** added for 7B model only
- **Size:** sample 50-100 tasks from the 500-task Verified set
- **Why:** real GitHub issue resolution — much closer to actual IDE-assistant product workflows than synthetic HumanEval-style tasks
- **Expected score:** 1.5B → likely 0-1%, not useful. 7B Qwen2.5-Coder → likely 5-15%
- **Eval cost:** ~5-10 min per task (clone repo, apply patch, run pytest in container);
  50 tasks ≈ 4-8 hours
- **Integration:** use official `swebench` harness or `swe-bench` HF dataset
- **Risk:** needs docker + repo clones; we still don't have docker.io installed (sudo blocked).
  Workaround: install in user-mode (rootless docker) or use podman; or run subset on
  a Lambda Cloud box if needed

### Tier 4 — skip entirely

#### Terminal-Bench 2.0

- **Reason for skipping:** 89 tasks; agent-based (multi-turn LLM + terminal sandbox);
  1.5B / 7B without agent training will score near 0
- **What you'd need:** a strong code-completion-only 7B is not enough; would need
  agent-style continued training + harness like `mini-swe-agent`
- **When to revisit:** if we ever try a 14B+ model with explicit agent training

#### CursorBench

- **Reason for skipping:** Cursor's internal, not public

## Eval suite implementation

```
scripts/
    eval_humaneval.py         # existing, no change
    eval_mbpp.py              # existing, no change
    eval_livecodebench.py     # NEW — needs writing
    eval_swebench.py          # NEW — only for 7B runs
    eval_all.sh               # NEW — orchestrator: runs all available evals for a model

results/
    <model>_humaneval.summary.json
    <model>_mbpp.summary.json
    <model>_livecodebench.summary.json    # NEW: pass@1 + breakdown by date / difficulty
    <model>_swebench.summary.json          # NEW: only for 7B+
```

## Migration plan

1. **Build `eval_livecodebench.py` now** (highest priority, contamination-fix)
2. **Re-eval all existing checkpoints** (base, SFT, DPO_v2, DPO_v1, GRPO_30, GRPO_200, DPO_INT4)
   on LiveCodeBench → update `final_summary.md` table with LCB column
3. **For all new experiments** (7B SFT/DPO/GRPO, 14B QLoRA, per-repo), include LCB by default
4. **SWE-Bench-Verified subset** wait until docker is installed (D-next ask) OR run on Lambda

## Sources

- [LiveCodeBench](https://livecodebench.github.io/), [v6 release info](https://llm-stats.com/benchmarks/livecodebench-v6)
- [SWE-Bench](https://www.swebench.com/), [Multilingual](https://www.swebench.com/multilingual.html)
- [Terminal-Bench 2.0](https://www.tbench.ai/leaderboard/terminal-bench/2.0), [paper](https://arxiv.org/html/2601.11868v1)
- [Cursor Composer 2 report](https://cursor.com/resources/Composer2.pdf) — uses TB2 / SWE-bench-M / CursorBench
- [Qwen2.5-Coder decon method](https://arxiv.org/abs/2409.12186)
