# code-rl-pipeline

[![lint](https://github.com/RuoyunMa/code-rl-pipeline/actions/workflows/lint.yml/badge.svg)](https://github.com/RuoyunMa/code-rl-pipeline/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A from-scratch **SFT → DPO → GRPO → AWQ** post-training stack for code LLMs, built on a **single RTX 5090 (32 GB)** to answer one concrete question:

> Once you control for benchmark contamination, **where do post-training gains on an already-strong open coder model actually come from** — supervised fine-tuning, reinforcement learning, or just scaling the base?

The short answer surprised me, and most of the value here is in the negative results. Reproducible end-to-end in ~9 GPU-hours.

---

## TL;DR — what the experiments showed

1. **Scaling dominates; generic SFT is a no-op at 7B.** On a contamination-resistant benchmark, going 1.5B → 7B base buys **+11.8pp**; *all* post-training on top adds **+2.9pp**. Fine-tuning Qwen2.5-Coder-7B on 5k Magicoder samples moved every metric within noise (and slightly *regressed* MBPP) — the instruction distribution is already absorbed by the base model's pretraining. The same SFT gives **+5pp** at 1.5B, where there's still headroom.
2. **RL is the only real post-training lift at 7B** — +1.5pp on top of SFT, the largest clean delta we get at that scale.
3. **DPO ≈ GRPO at 7B, but DPO > GRPO at 1.5B / small data.** Binary verifiable rewards are high-variance; pairwise preference is the better signal until you have thousands of prompts plus a fast rollout loop.
4. **Multi-temperature rollouts unlock DPO.** Sampling at T = 0.4 / 0.8 / 1.2 (vs a single temperature) broke within-prompt correlation and **5×'d the usable preference-pair yield (425 → 2320)**, taking the DPO trainer's `rewards/accuracies` from 0.39 to 0.78 — a real gradient where there was barely one.
5. **Per-repo LoRA is a real, under-explored win.** A tiny adapter trained on one codebase **doubled same-repo internal-symbol recall (18.8% → 40.5%)** with no regression on general benchmarks.
6. **QLoRA is free at 7B** — matches bf16 LoRA across every benchmark at 14 GB instead of 24 GB.
7. **AWQ INT4 is *slower* than bf16 at 1.5B on Blackwell** (2.3× fewer tok/s) — at this size the model is compute-/cache-bound, so dequant overhead dominates the bandwidth savings. Quantization here is a disk-size and scale-up tool, not a free speedup.

The methodological throughline: **measure the base before setting targets, and pick a benchmark that isn't already in the training data.**

## Headline numbers

LCB = LiveCodeBench v6 LeetCode subset (444 problems); **post-cutoff** = filtered to `contest_date > 2024-09-18` (post Qwen2.5-Coder release, ~136 unseen problems) — the contamination-resistant metric.

| Pipeline | HumanEval | MBPP | LCB v6 | LCB post-cutoff |
|---|---:|---:|---:|---:|
| Qwen2.5-Coder-1.5B base | 73.17% | 63.42% | 13.51% | 2.94% |
| + 1.5B SFT (LoRA r=128) | 74.39% | 69.65% | 19.14% | 5.15% |
| + 1.5B DPO v2 (2320 pairs) | 74.39% | 66.93% | 16.67% | 4.41% |
| **Qwen2.5-Coder-7B base** | 85.98% | 82.49% | 35.59% | 14.71% |
| + 7B SFT (LoRA r=32) | 85.37% | 82.10% | 36.94% | 16.18% |
| **+ 7B DPO / GRPO (best)** ⭐ | 85.37% | 82.10% | **38.29%** | **17.65%** |
| + FastAPI per-repo LoRA r=16 | 73.78% | 67.32% | 17.79% | — |
| ↳ same-repo symbol recall | — | — | — | **40.5%** vs 18.8% base |

**Decomposing the 7B LCB-post-cutoff number (14.71% → 17.65%):** scaling base 1.5B → 7B = **+11.77pp**, 7B SFT = +1.47pp, 7B RL on top = +1.47pp. The training contributes ~20% of the absolute gain; scaling the base contributes ~80%.

## The findings in detail

### 1. Pick a benchmark that isn't in the training data
HumanEval and MBPP are partially absorbed by Qwen's 5.5T-token pretraining, which compresses the deltas and flatters fine-tuning. LiveCodeBench v6 filtered to problems released *after* the model's cutoff shows the largest, cleanest separations (e.g. 1.5B base→SFT is +1.2pp on HumanEval but **+5.6pp** on LCB). For any modern coder model, LCB-post-cutoff is the metric that actually moves.

### 2. Scaling vs SFT vs RL — decomposed
Training Qwen2.5-Coder-7B on 5k Magicoder samples does essentially nothing on HumanEval/MBPP/LCB beyond noise, and slightly regresses MBPP (−0.4pp). The same dataset gives +5pp LCB at 1.5B. **The marginal value of a generic instruction dataset shrinks fast as the base improves** — to lift a strong 7B base you need much more / higher-quality / domain-specific data, or a different objective (FIM, agent traces). After SFT, the only reliable lift comes from RL on verifiable rewards.

### 3. DPO vs GRPO depends on scale and data volume
At 7B, DPO and GRPO converged to the **identical** end-state (38.29% LCB, 17.65% post-cutoff). At 1.5B with limited data, DPO (2320 pairs, 145 steps) clearly beat GRPO (300 prompts, 200 steps): binary pass/fail reward has `reward_std ≈ 0.9` throughout, so without `n ≫ 8` generations or a large prompt set the gradient direction is noisy. GRPO's edge should appear at much larger prompt counts with a fast (vLLM) rollout loop — the verl path is scaffolded but not run here.

### 4. Multi-temperature rollouts as a data lever
Naive single-temperature (T=0.8) rollouts yielded only 128/374 = 34% mixed-outcome problems (most problems are all-pass or all-fail, so they produce no preference pair). Mixing T = 0.4 / 0.8 / 1.2 raised that to 232/374 = 62% while per-completion pass rate stayed ~constant — the gain came from breaking within-prompt correlation, not from changing competence. Result: 2320 pairs vs 425, and a real preference signal. *Lesson: when verifiable-reward DPO is pair-starved, vary temperature before adding candidates.*

### 5. Per-repo LoRA
Industry assistants (Copilot, Cursor, Cody) lean on RAG + long context rather than per-project weights. A 37 MB LoRA trained on FIM samples from one codebase (465 FastAPI `.py` files) **doubled same-repo internal-symbol recall (18.8% → 40.5%)** while general benchmarks stayed flat — i.e. it baked the repo's API conventions into the weights without forgetting. With multi-adapter hot-swap, one base + N adapters could serve N codebases.

### 6. QLoRA at 7B loses nothing
nf4-base QLoRA matched bf16 LoRA on every benchmark at 7B, dropping train memory 24 GB → 14 GB. On a 32 GB card that's the difference between "barely fits" and "room to scale" — default to QLoRA.

### 7. AWQ INT4 is for scale, not speed (at 1.5B on Blackwell)
INT4 throughput was 3411 tok/s vs bf16 7924 (2.3× *slower*): at 1.5B the weights fit in cache, memory bandwidth isn't the bottleneck, and dequant kernel overhead dominates. The win is disk (1.1 GB vs 3.0 GB). AWQ pays off at 7B+ on memory-bound GPUs.

## Quick reproduce

Tested on Ubuntu 24.04, RTX 5090, driver ≥ 595, CUDA toolkit 13.0 (conda-forge).

```bash
# 1. envs (kept separate — vllm and unsloth/trl don't co-exist cleanly)
bash scripts/setup_env.sh        # coderl: vllm 0.21 + eval
bash scripts/setup_env_sft.sh    # coderl_sft: unsloth + trl on torch 2.10

# 2. baseline (measure BEFORE training)
conda activate coderl && bash scripts/run_baseline.sh

# 3. SFT
python scripts/prep_sft_data.py                       # data/sft.jsonl (5000 Magicoder samples)
conda activate coderl_sft
python scripts/train_sft.py --output-dir outputs/sft --save-merged --no-wandb

# 4. multi-temperature rollouts -> DPO
for T in 0.4 0.8 1.2; do
  python scripts/generate_rollouts.py --model outputs/sft_merged --source mbpp_train \
      --n 4 --temp $T --out data/rollouts_T${T/./}.jsonl
done
python scripts/build_dpo_pairs.py --rollouts data/rollouts_multi.jsonl --out data/dpo_pairs.jsonl
python scripts/train_dpo.py --sft-model outputs/sft_merged --data data/dpo_pairs.jsonl --output-dir outputs/dpo --no-wandb

# 5. GRPO (trl) + AWQ
python scripts/train_grpo_trl.py --sft-model outputs/sft_merged --data data/rl_prompts.jsonl --max-steps 200 --no-wandb
python scripts/quantize_awq.py --model outputs/dpo_merged --out outputs/dpo_awq_int4 --calib-n 128
```

Full step-by-step (with eval commands) is in [`results/final_summary_v2.md`](results/final_summary_v2.md).

## Repo layout

```
scripts/
  setup_env*.sh            # two isolated conda envs (vllm vs unsloth/trl)
  prep_sft_data.py         # Magicoder -> data/sft.jsonl
  train_sft.py             # Unsloth + LoRA / QLoRA
  generate_rollouts.py     # vLLM N-sampling (multi-temperature)
  build_dpo_pairs.py       # sandbox-graded rollouts -> (chosen, rejected)
  train_dpo.py             # trl DPO
  train_grpo_trl.py        # trl GRPO (verifiable reward); verl path scaffolded
  code_reward.py           # unit-test pass/fail reward
  eval_*.py                # HumanEval / MBPP / LiveCodeBench (sandboxed)
  quantize_awq.py          # AutoAWQ INT4 + throughput bench
  per_repo/                # FIM sample builder + per-repo eval (symbol recall)
results/                   # eval JSONs, per-milestone notes, final_summary_v2.md
data/, outputs/            # gitignored
```

## Stack

PyTorch 2.10–2.11 · CUDA 13 · vLLM 0.21 (inference/eval/rollouts) · HuggingFace transformers / peft / trl 0.24 · Unsloth (SFT speedup) · AutoAWQ 0.2.9 · bitsandbytes.

## Limitations / not yet done

- **GRPO is the trl implementation** (algorithm-validated); the verl + vLLM-rollout production path is scaffolded (`train_grpo_verl.sh`, `convert_to_verl_parquet.py`) but not run — it needs Docker and is where GRPO's large-data advantage would show.
- **Single-GPU scale.** Models are 1.5B / 7B; no ≥14B or multi-node run.
- Per-run metrics live in `results/*.json`; W&B logging is wired (`--no-wandb` used for the headline runs).
- LiveCodeBench is the contamination control, but no SWE-bench / agentic-task evaluation.
