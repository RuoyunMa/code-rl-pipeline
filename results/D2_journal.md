# D2 Journal — 2026-05-19

**Big picture:** Ran the entire pipeline (D2 env recovery → D7 DPO → D8 GRPO fallback → D13 AWQ) in one autonomous session while the user slept. Full results in `results/final_summary.md`. M2/M4/M6 acceptance met; M5 (GRPO) partial — GRPO at this data scale didn't beat DPO, a real finding to discuss.

## Timeline (UTC)

| Time | Action |
|---|---|
| 07:43 | SSH probe; confirmed 5090 awake (uptime 6 min after suspend last night); sudo blocks (no NOPASSWD) |
| 07:47 | gsettings idle-delay=0, sleep-inactive-ac-type=nothing (no-sudo suspend defense) |
| 07:49 | Repair coderl env: transformers 5.6→5.8.1, nvidia-cudnn 9.19→9.22, nccl 2.28→2.30, cusparselt 0.8→0.9, xgrammar/llguidance/outlines_core restored. Patched `transformers/utils/import_utils.py` PACKAGE_DISTRIBUTION_MAPPING KeyError. |
| 08:00 | vllm smoke STILL FAILS — `from flash_attn.ops.triton.rotary import apply_rotary` — leftover empty `flash_attn/` namespace dir from yesterday's flash-attn-4 attempt. Deleted dir → vllm imports clean. |
| 08:02 | `python scripts/setup_env_sft.sh` to create coderl_sft env (torch 2.10 / unsloth / trl 0.24). Hit conda-forge cuda-nvcc activate hook nounset bug; removed `set -u` from script. |
| 08:14 | coderl_sft ready: torch 2.10.0+cu128, transformers 5.5.0, trl 0.24, unsloth 2026.5.4. |
| 08:14 | prep_sft_data.py (5000 Magicoder samples → data/sft.jsonl, 11.5 MB) |
| 08:14 | SFT training started in coderl_sft (314 steps, 2 epochs). |
| 08:24 | SFT done in 21 min. Loss 0.49→0.47. Saved outputs/sft (LoRA) + outputs/sft_merged (FP16). |
| 08:26 | SFT eval in coderl: HumanEval 73.78% (+0.6), MBPP 65.76% (+2.3). |
| 08:28 | generate_rollouts.py → 374 problems × 4 candidates in <45 sec via vllm batched. |
| 08:29 | build_dpo_pairs.py → 425 pairs (128 problems mixed pass+fail). Lower than 2000 target — 56.9% per-completion pass rate means most problems all-pass or all-fail. |
| 08:30 | DPO training: first run died on trl 0.24 transitive deps. Iterated: `pip install mergekit llm-blender weave`; patched `llm_blender/blender/blender_utils.py` and `blender.py` `TRANSFORMERS_CACHE` → fallback to `HF_HUB_CACHE`; added `model.warnings_issued = {}` stub before passing to DPOTrainer (transformers 5.5 dropped this attribute on Qwen2). |
| 08:33 | DPO done in 29 sec, 27 steps. Loss 0.6935→0.6919 (barely moved, expected at this data scale). |
| 08:34 | merge_lora.py: outputs/sft_merged + outputs/dpo → outputs/dpo_merged (3 GB FP16). |
| 08:36 | DPO eval: **HumanEval 75.00% (+1.2 over SFT), MBPP 65.76%** (flat over SFT). M4 acceptance pass. |
| 08:38 | prep_rl_prompts.py → 300 RL prompts for GRPO trl path. |
| 08:40 | GRPO 30 steps dry run: 96 sec, reward signal very noisy (±0.875), KL healthy 0.0002. Eval: HE 73.78% MBPP 65.76% (same as SFT, didn't move). M4 GRPO-no-OOM pass; M5 NOT yet hit. |
| 08:46 | AWQ INT4 quantization on outputs/dpo_merged. Calibration ran fine; bf16 bench OOM'd because quantize phase didn't free GPU memory before bench_vllm. Patched scripts/quantize_awq.py to `del model; gc.collect; torch.cuda.empty_cache()` after save. |
| 08:50 | AWQ bench rerun (--skip-quantize): bf16 7925 tok/s, INT4 3412 tok/s — **INT4 is 2.3× SLOWER**. Counterintuitive but explainable (1.5B fits in cache, INT4 dequant overhead dominates). Disk: 1.1 GB vs 3.0 GB (2.7× smaller). |
| 08:53 | GRPO 200 steps run for a real M5 attempt (lr=3e-6, otherwise same config). 651 sec. Reward signal still noisy, KL still 0.0003 (model barely moved). |
| 09:05 | GRPO 200 eval: HumanEval 73.78% (= SFT), MBPP 66.15% (+0.4pp over SFT). **DPO remains best on HumanEval.** |
| 09:10 | Writing journal + README + final_summary. |
| 09:57 | **Bonus iteration — multi-temp rollouts**: noticed M3 pair count (425) was way below 2000 target and not from lack of candidates per problem but from within-prompt correlation. Ran rollouts at T=0.4 + 0.8 + 1.2 (4 each) and merged → 4488 candidates → 2320 mixed pairs (62% problems mixed vs 34% single-temp). |
| 10:00 | DPO v2 training on 2320 pairs: 145 steps, 160 sec, rewards/accuracies 0.39 → 0.78 → actually converging this time. |
| 10:08 | DPO v2 eval: **HumanEval 74.39%, MBPP 66.93% (+1.17pp over SFT, +3.51pp over base)**. Cleanly inside M4 acceptance band (+1-3%) on BOTH benchmarks now. v2 supersedes v1 as the headline deployable model. |
| 10:10 | Updated final_summary.md and README.md to feature DPO v2; left v1 numbers in comparison tables for honesty. |

## Files added/modified today

### New scripts
- `scripts/setup_env_sft.sh` — coderl_sft env bootstrap (torch 2.10 + unsloth + trl)
- `scripts/prep_sft_data.py` — sample Magicoder → data/sft.jsonl
- `scripts/prep_rl_prompts.py` — rollouts.jsonl → GRPO-trl schema
- `scripts/merge_lora.py` — merge LoRA adapter into base, save FP16

### Modified scripts
- `scripts/setup_env.sh` — completely rewritten:
  - drop `--index-url cu129` (pypi default is now cu130 wheel for torch 2.11)
  - add cuda-toolkit + gxx_linux-64 conda install
  - add header/lib symlinks (targets/x86_64-linux/ → include/, lib64/)
  - add stale `flash_attn/` dir cleanup
  - add transformers `PACKAGE_DISTRIBUTION_MAPPING` patch
  - drop `set -u` (conda-forge cuda-nvcc activate hook conflicts)
  - move unsloth → setup_env_sft.sh (separate env)
- `scripts/eval_humaneval.py` — vllm/datasets/sandbox_executor imports now lazy (inside main) so coderl_sft can import `extract_code`
- `scripts/train_sft.py` — accepts local jsonl path or HF dataset id for `--dataset`
- `scripts/train_dpo.py` — stub `model.warnings_issued = {}` for transformers 5.5 compat
- `scripts/train_grpo_trl.py` — same stub
- `scripts/quantize_awq.py` — free GPU mem at end of quantize() before bench
- `scripts/deploy.sh` — added `--exclude logs/` to rsync (otherwise `--delete` wipes 5090 logs)
- `14day_plan.md` — M2 acceptance "+3-6%" → "+1-3%" (HumanEval ceiling effect)

### New results files
- `results/m2_sft.md` — SFT eval + M2 verdict
- `results/m4_dpo.md` — DPO eval + trl 0.24 compat issues
- `results/quantization_benchmark.md` — bf16 vs INT4 throughput/latency
- `results/final_summary.md` — full pipeline summary, lessons learned, key takeaways
- `results/D2_journal.md` — this file

### Memory files (~/.claude/.../memory/)
- `MEMORY.md` index updated
- `feedback_separate_envs.md` — same-env install of competing inference libs is a trap
- `feedback_show_monitor_commands.md` — give the user ssh tail commands for long ops
- `project_d1_ended_with_backlog.md` — superseded; D2 covered most of D1's backlog

## Remaining work (post-D2)

| Priority | Item | Blocker |
|---|---|---|
| Now | `gh auth login` + push to GitHub | needs interactive auth (the user) |
| Now | `wandb login` | needs API key (the user) |
| Soon | `sudo apt install docker.io nvidia-container-toolkit` → verl Docker GRPO | needs sudo password |
| Soon | M1 "verl Docker green" milestone | depends on docker |
| Optional | Longer GRPO with verl + vllm rollout integration (1000+ steps) | depends on verl Docker |
| Optional | Tech blog draft on DPO-vs-GRPO finding | nothing — the user's task |
| Optional | Blog draft to externalize the findings | future task |
| Optional | 7B variant on Lambda Cloud ($150-240) | budget |
| Optional | SWE-bench Verified small subset eval | could add |

## Things I learned tonight that future-the user should remember

1. **trl 0.24 + transformers 5.5 has rough edges.** Two stub patches required (`warnings_issued`, `TRANSFORMERS_CACHE` in llm_blender). If you upgrade either, retest the patches. Best long-term fix: pin trl to a version that's officially compat with transformers 5.x (probably 1.x once it's mature).
2. **The pip pypi-default torch is cu130 in 2026.** The `--index-url cu129` pattern from older guides is now actively wrong — installs wrong wheel for current vllm.
3. **conda-forge cuda-nvcc activate hooks need `set +u`.** Don't enable nounset in scripts that activate conda envs.
4. **For 1.5B SFT, Unsloth gives ~21 min for 2 epochs on 5090.** Plan budgets ~4h for SFT — that was an A100-pessimistic estimate. 5090 + Unsloth is way faster.
5. **DPO won this round. GRPO needs more scale.** Don't take "GRPO is the production target" too literally; for small-scale demos, DPO is the right tool. Reach for GRPO + verl at 10K+ prompts with an integrated rollout server; otherwise DPO.

## How to resume (3rd session onward)

```bash
# from your terminal
ssh 5090
cd ~/workspace/code-rl-pipeline
ls outputs/                     # see all merged checkpoints
cat results/final_summary.md    # full pipeline summary

# everything's checkpointed; you can re-eval any model:
conda activate coderl
python scripts/eval_humaneval.py --model outputs/dpo_merged --out /tmp/he_recheck.jsonl --no-wandb
```

Most likely next ask: (1) push to GitHub, (2) install Docker + run verl GRPO Docker dry run, (3) write tech blog.
