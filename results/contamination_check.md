# Contamination check — SFT data vs HumanEval

**Date:** 2026-05-19
**Question raised by:** Ruoyun — is Qwen2.5-Coder-1.5B already trained on HumanEval / MBPP such that our SFT on Magicoder-similar data is just teaching the answers?

## TL;DR

**Direct contamination is negligible: 0.36% of our SFT data has ANY 10-gram overlap with HumanEval test set, and most of those are surface-level phrasing matches.** But the underlying concern is valid for a different reason — see "Real risks" below.

## Method

Replicated Qwen2.5-Coder paper's decontamination heuristic (10-gram string overlap, lower-cased, punctuation-stripped, whitespace-collapsed) and applied it in reverse: scan `data/sft.jsonl` (5000 Magicoder samples we trained on) against HumanEval test set (164 problems, ~14.5k unique 10-grams).

Script: inline python in journal, [see Bash output transcript](./D2_journal.md).

## Findings

| Metric | Value |
|---|---:|
| SFT samples scanned | 5000 |
| Samples with ≥1 HumanEval 10-gram overlap | 18 (0.36%) |
| Median overlap on the 18 hits | 1-3 grams (surface phrasing) |
| Max single-sample overlap | 9 grams |

Largest hit: `sft[269]` shares 9 ten-grams with `HumanEval/94`. Sample preview:
> "You are tasked with creating a program to calculate the sum of prime numbers up to a given limit..."

HumanEval/94 is a "sum the digits of skipped primes" problem — semantically related but not identical. Magicoder's authors removed direct matches; this is a near-paraphrase that slipped through the exact-match filter (they admit this in their paper).

## What this rules out

- We did NOT train the model on verbatim HumanEval answers
- Our SFT's +0.6pp HumanEval gain over base is NOT explained by Magicoder containing answers

## What this does NOT rule out (the real risks)

1. **Qwen2.5-Coder base pretraining (5.5T tokens, 70% code, released 2024-09) is itself near-saturated on HumanEval-style problems.** HumanEval (2021) and MBPP (2021) are 3 years older than Qwen's pretraining cutoff. Even with the Qwen team's decontamination, the base model has seen MANY structurally-similar Python signature+docstring → function-body completion examples during pretraining. Our 73.17% baseline is consistent with a model that has implicitly learned HumanEval's distribution.

2. **Magicoder's distribution overlaps HumanEval's style**, even after decontamination — both are Python function completions with descriptions. Training on Magicoder probably DOES move the model in a HumanEval-friendly direction, even if no specific problem matches.

3. **Our +0.6pp gain is within noise (~3pp single-sample noise at n=164).** Don't over-interpret a 1-problem difference.

## Implications for benchmark choice going forward

HumanEval and MBPP should be **legacy / sanity-check metrics**, not primary discriminators. Both are saturated for code models of Qwen's vintage. The primary metric should be **contamination-free by construction**:

- **LiveCodeBench v6** (1055 problems, May 2023 – Apr 2025) — collects new problems continuously; we can filter to post-Qwen-2.5-Coder-release (post Sep 2024) for true contamination-free eval
- (For 7B+) SWE-Bench-Verified small subset (50-100 tasks from the 500-task verified set, real GitHub issues)

These will be added in the next round of experiments. See `benchmark_plan.md` for the integration plan.

## Sources

- [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186) — 10-gram decon method
- [Magicoder paper](https://arxiv.org/abs/2312.02120) — decon of OSS-Instruct (9 exact matches removed)
- [LiveCodeBench](https://livecodebench.github.io/) — contamination-free by release-date tagging
