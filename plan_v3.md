# plan_v3 — overnight execution plan (post-pivot)

Replaces the original `14day_plan.md` after the 2026-05-19 pivot. The original is
preserved for history; the actual execution from here on follows this file.

## Triggers for the pivot (user's input)

1. **Contamination concern** — HumanEval / MBPP are old (2021); Qwen2.5-Coder-1.5B
   (2024-09) probably saw structurally-similar problems in 5.5T pretraining tokens
   even with the team's 10-gram decontamination. Our +0.6pp HE gain is hard to
   distinguish from base saturation noise.
   - Verified our SFT data is clean: 0.36% has any 10-gram overlap, max 9 grams
     on one sample (see `results/contamination_check.md`).
   - But conclusion holds: HE/MBPP can't be the primary discriminator going forward.

2. **Benchmark modernization** — Cursor Composer 2 reports Terminal-Bench 2.0,
   SWE-Bench Multilingual, CursorBench. Of these:
   - Terminal-Bench 2.0: out of reach for 1.5B/7B coder model (agent benchmark)
   - SWE-Bench Multilingual: too heavy for 1.5B; viable subset for 7B+
   - CursorBench: not public
   - **Addition: LiveCodeBench v6** (1055 problems, 2023-2025, contamination-free by date) — fills the same role as HE/MBPP but contamination-resistant.
   - Plan in `benchmark_plan.md`.

3. **More ambitious experiments** — GPU stays alive overnight; design 8-24h experiments. No more 25-minute timidity.

4. **7B SFT parameter sweep** — like the 1.5B sweep but at 7B scale. Memory analysis first.

5. **Per-repo training as staff MLE** — industry survey + design doc + execution.
   See `per_repo_training.md`.

## What's already running

- Block 0: C1 + C3 (Full FT 1.5B variants) — finishing the SFT ablation sweep.
  Started 2026-05-19 02:55Z, ETA ~13:30Z (C1 ~50 min, C3 ~60 min).
  Will close out `results/lora_sweep.md`.

## Memory budget for 7B Qwen2.5-Coder

Estimate based on observed 1.5B numbers (LoRA SFT 12 GB, Full FT 25.7 GB):

| 7B config | Est. peak VRAM | Fits 32GB? |
|---|---:|:--|
| LoRA r=32 all-7, bs=8 seq=2048 | ~24 GB | ✓ |
| LoRA r=64 all-7, bs=8 | ~25 GB | ✓ |
| LoRA r=128 all-7, bs=8 | ~26 GB | ✓ tight |
| LoRA r=32 all-7, bs=4 | ~22 GB | ✓ |
| QLoRA r=32 (nf4 base) | ~10-12 GB | ✓ very comfortable |
| QLoRA r=128 (nf4 base) | ~12-13 GB | ✓ comfortable |
| Full FT bf16 + 8bit-Adam | ~50 GB | ❌ |
| Full FT QLoRA-base + 8bit-Adam | ~30 GB | ⚠ borderline |

So the 7B sweep can include LoRA r=32/64/128 (full bf16 base) and QLoRA r=32/128.

## Execution order (priority queue)

| # | Block | Time | Outputs |
|---|---|---:|---|
| 1 | (finish) C1 + C3 Full FT 1.5B | 50 min more | `results/lora_sweep.md` |
| 2 | **Build `eval_livecodebench.py` and re-eval existing 1.5B checkpoints** (base, SFT v1, DPO v2, GRPO 200) on LiveCodeBench v6 | 1.5h | LCB scores in `results/*_livecodebench.summary.json` |
| 3 | **7B SFT sweep — top 4 configs**: LoRA r=32 all-7, LoRA r=128 all-7, QLoRA r=32, QLoRA r=128, eval each on HE/MBPP/LCB | 5-6h | `outputs/7B_*_merged/`, `results/7B_sweep.md` |
| 4 | **7B DPO** on best 7B-SFT (multi-temp rollouts, build 2000+ pairs, train LoRA r=16) + eval | 1.5h | `outputs/7B_dpo_merged/`, eval results |
| 5 | **7B GRPO 500 steps** on best 7B-DPO + eval | 2h | `outputs/7B_grpo_merged/`, eval results |
| 6 | **Per-repo Flask POC** (FIM data gen + LoRA train + held-out FIM eval + LCB regression check + vLLM hot-swap demo) | 3-4h | `outputs/per_repo/flask_r16/`, `results/per_repo_flask.md` |
| 7 | **1.5B real-scale GRPO** (2000 steps, num_gen=8, full MBPP-train+HE-train) | 4h | `outputs/1.5B_grpo_real_merged/`, eval results |
| 8 | (stretch) 14B QLoRA SFT | 6-8h | `outputs/14B_qlora_sft_merged/` |
| 9 | (stretch) pass@k re-eval with n=10 across all checkpoints | 2-3h | updated comparison tables |
| 10 | Final aggregation: update README + final_summary with full matrix | 30 min | `final_summary_v2.md` |

**Realistic budget (sequential)**: blocks 1-7 = ~18h. Plenty of headroom in 24h.

## Reporting milestones (user asked for "report after each block")

After each block: print 1-paragraph status + key numbers + next-up.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| 7B OOM on bs=8 | fall back to bs=4 ga=8; tested fit envelope above |
| QLoRA on cu13 / sm_120 setup issue | Unsloth supports nf4 on Blackwell as of 2026.5.4; preemptively pin nightly if static fails |
| LiveCodeBench dataset download slow / sandbox harness mismatch | Cache HF dataset; reuse our existing `sandbox_executor.py` for stdin/stdout test cases |
| Per-repo FIM eval metric ambiguity (no standard) | Use edit-similarity + exact-match + symbol-recall; report all three for transparency |
| Real-scale GRPO reward collapse over 2000 steps | warmup → lower lr, monitor KL; restart from checkpoint if needed |
| Total runtime overshoots 24h | This is acceptable — GPU stays alive; if morning hits and we're at block 5, that's still a huge advance over D2 |

## Per-block file outputs to expect

```
results/
    lora_sweep.md                          # block 1
    eval/                                  # block 2 onwards (LCB)
        base_livecodebench.summary.json
        sft_livecodebench.summary.json
        dpo_v2_livecodebench.summary.json
        ...
    7B_sweep.md                            # block 3
    7B_dpo.md                              # block 4
    7B_grpo.md                             # block 5
    per_repo_flask.md                      # block 6
    1.5B_real_scale_grpo.md                # block 7
    final_summary_v2.md                    # block 10
outputs/
    A4_r64_all_merged/, A5_r128_all_merged/, B1_r32_attn_merged/, B2_r32_mlp_merged/
    C1_fullft_lr5e5/, C3_fullft_lr1e5/
    7B_sft_r32_merged/, 7B_sft_r128_merged/, 7B_qlora_r32/, 7B_qlora_r128/
    7B_dpo_merged/, 7B_grpo_merged/
    per_repo/flask_r16/, per_repo/requests_r16/
    1.5B_grpo_real_2000s_merged/
```

## What I will NOT do without checking back

- Push to GitHub (needs gh auth)
- Install docker / nvidia-container-toolkit (needs sudo)
- Run SWE-Bench-Verified (depends on docker)
- Modify CLAUDE.md (project context — only update when user signs off)
- Anything that uses user's API keys (W&B, HF token)
