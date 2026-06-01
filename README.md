# code-rl-pipeline

End-to-end post-training pipeline for a coding LLM on a **single RTX 5090 (32 GB Blackwell, sm_120, CUDA 13)**.
SFT → DPO → GRPO (trl fallback) → AWQ INT4 deployment, with a verifiable-reward grader (passes unit tests) wiring DPO/GRPO.

### Headline numbers (best from each scale)

| Pipeline | HE | MBPP | LCB v6 | LCB post-cutoff (contamination-free) |
|---|---:|---:|---:|---:|
| Qwen2.5-Coder-1.5B base | 73.17% | 63.42% | 13.51% | 2.94% |
| + 1.5B SFT (LoRA r=128) | 74.39% | 69.65% | 19.14% | 5.15% |
| + 1.5B DPO v2 (2320 pairs) | 74.39% | 66.93% | 16.67% | 4.41% |
| **Qwen2.5-Coder-7B base** | **85.98%** | **82.49%** | **35.59%** | **14.71%** |
| + 7B SFT (LoRA r=32) | 85.37% | 82.10% | 36.94% | 16.18% |
| **+ 7B DPO / GRPO (best)** ⭐ | 85.37% | 82.10% | **38.29%** | **17.65%** |
| **+ FastAPI per-repo LoRA r=16** (on 1.5B DPO v2) | 73.78% (-0.6) | 67.32% (+0.4) | 17.79% (+1.1) | — |
|     ↳ same-repo symbol_recall | — | — | — | **40.5%** vs **18.8%** base |

**Decomposing the 7B LCB-post-cutoff lift (14.71% → 17.65% = +2.94pp)**:
- 7B SFT on Magicoder: **+1.47pp** (within noise on most metrics — SFT is essentially absorbed by Qwen's 5.5T pretraining at this scale)
- 7B DPO/GRPO on top of SFT: **+1.47pp** (real RL signal)
- **Most of the absolute number (1.5B → 7B → trained) comes from scaling base (+11.77pp), not training (+2.94pp).**

Best generic-purpose model: **7B DPO** (`outputs/7B_dpo_merged`). Best per-repo demo: **FastAPI adapter** (`outputs/per_repo/fastapi_merged`, 37 MB adapter).

Single-GPU budget: **~9h GPU time on a single RTX 5090 32GB** produced all of this. **QLoRA at 7B matches bf16 LoRA** across all benchmarks while using 14 GB instead of 24 GB.

## Why this exists

Built as a 2-week proof I can run the full LLM post-training stack independently — to transition from recommendation-systems infra to LLM post-training, specifically for [Trae](https://www.trae.ai/) (ByteDance's AI coding agent). Not chasing SOTA; chasing **pipeline-correct + measurable positive delta + reproducible on commodity hardware**.

Full motivation, design decisions, and an honest experimental writeup in [`results/final_summary.md`](results/final_summary.md).

## Quick reproduce

Tested on Ubuntu 24.04, RTX 5090, driver ≥ 595, CUDA toolkit 13.0 (provided via conda-forge).

```bash
# 1. bootstrap the main inference/eval env (~15 min, downloads ~10GB of wheels)
bash scripts/setup_env.sh
conda activate coderl

# 2. bootstrap the SFT env (unsloth + torch 2.10 — kept separate from vllm env)
bash scripts/setup_env_sft.sh

# 3. baseline eval
bash scripts/run_baseline.sh                # 2 min total on 5090

# 4. SFT data + training
conda activate coderl
python scripts/prep_sft_data.py             # writes data/sft.jsonl (5000 samples)
conda activate coderl_sft
python scripts/train_sft.py --output-dir outputs/sft --save-merged --no-wandb  # ~21 min

# 5. eval SFT
conda activate coderl
python scripts/eval_humaneval.py --model outputs/sft_merged --out results/humaneval_sft.jsonl --no-wandb
python scripts/eval_mbpp.py    --model outputs/sft_merged --out results/mbpp_sft.jsonl --no-wandb

# 6. rollouts (multi-temperature for better DPO pair yield) + DPO
for T in 0.4 0.8 1.2; do
    python scripts/generate_rollouts.py --model outputs/sft_merged --source mbpp_train \
        --n 4 --temp $T --out data/rollouts_T${T/./}.jsonl
done
# merge into one (374 problems × 12 candidates each)
python -c "
import json
merged = {}
for p in ['data/rollouts_T04.jsonl', 'data/rollouts_T08.jsonl', 'data/rollouts_T12.jsonl']:
    for line in open(p):
        r = json.loads(line)
        m = merged.setdefault(r['task_id'], {**r, 'completions': []})
        m['completions'].extend(r['completions'])
open('data/rollouts_multi.jsonl', 'w').writelines(
    json.dumps(merged[k]) + '\n' for k in sorted(merged))
"
python scripts/build_dpo_pairs.py --rollouts data/rollouts_multi.jsonl --out data/dpo_pairs.jsonl --pairs-per-problem 10 --workers 24
conda activate coderl_sft
python scripts/train_dpo.py --sft-model outputs/sft_merged --data data/dpo_pairs.jsonl --output-dir outputs/dpo --no-wandb
python scripts/merge_lora.py --base outputs/sft_merged --adapter outputs/dpo --out outputs/dpo_merged

# 7. eval DPO
conda activate coderl
python scripts/eval_humaneval.py --model outputs/dpo_merged --out results/humaneval_dpo.jsonl --no-wandb
python scripts/eval_mbpp.py    --model outputs/dpo_merged --out results/mbpp_dpo.jsonl    --no-wandb

# 8. GRPO via trl fallback (verl Docker is the production target; trl is the algorithm-validated equivalent)
python scripts/prep_rl_prompts.py --rollouts data/rollouts.jsonl --train-n 300
conda activate coderl_sft
python scripts/train_grpo_trl.py --sft-model outputs/sft_merged --data data/rl_prompts.jsonl --max-steps 200 --no-wandb

# 9. AWQ INT4 quantization + throughput bench
conda activate coderl
python scripts/quantize_awq.py --model outputs/dpo_merged --out outputs/dpo_awq_int4 --calib-n 128
# writes results/quantization_benchmark.md
```

## Architecture decisions worth calling out

These came from real experiments, not assumptions. See [`results/final_summary.md`](results/final_summary.md) for the full writeup.

- **DPO > GRPO at small data scales.** With 425 preference pairs in 27 DPO steps, we got +1.2pp HumanEval. With 300 RL prompts in 200 GRPO steps (10× more compute), we got +0pp HumanEval. Binary verifiable rewards are high-variance; pairwise contrastive signal wins until you have ~thousands of prompts AND verl/vllm rollout integration to keep throughput up. Lesson: don't reach for GRPO before validating with DPO.
- **AWQ INT4 is SLOWER than bf16 at 1.5B on Blackwell.** Throughput dropped 2.3× (7925 → 3412 tok/s). At 1.5B the model fits comfortably in cache and INT4 dequant overhead dominates over the bandwidth savings. Quantization here is for **disk size** (1.1 GB vs 3.0 GB), not speed. AWQ's real win is at 7B+ on memory-bound GPUs.
- **Baseline measurement before threshold-setting.** Original M2 spec said "HumanEval +3-6% over base", assuming a ~62% base. Actual base measured 73% (Qwen2.5-Coder is stronger than the planning doc thought). Reset to "+1-3%" before training; landed inside the new band.
- **Separate conda envs per inference framework.** Mixing vllm + sglang in one env on day 1 cost ~2 hours of dep-conflict cleanup (flash-attn vs flash-attn-4 namespace, transformers version pins). The `coderl` env runs vllm + eval; the `coderl_sft` env runs unsloth + trl on torch 2.10. They never touch each other.

## Repo layout

```
14day_plan.md                  ← per-day execution checklist + risk log
CLAUDE.md                      ← project context for Claude Code sessions
requirements.txt               ← env reference (actual versions in setup_env.sh)
scripts/
  setup_env.sh                 # bootstrap coderl env (vllm 0.21, torch 2.11+cu130)
  setup_env_sft.sh             # bootstrap coderl_sft env (unsloth + torch 2.10)
  deploy.sh                    # rsync Mac → 5090
  prep_sft_data.py             # download Magicoder, sample 5000 → data/sft.jsonl
  train_sft.py                 # Unsloth + LoRA r=32
  eval_humaneval.py            # vLLM + sandbox unit-test executor
  eval_mbpp.py
  sandbox_executor.py
  generate_rollouts.py         # vLLM N-sampling for DPO/GRPO data
  build_dpo_pairs.py           # sandbox-grade rollouts → (chosen, rejected)
  prep_rl_prompts.py           # rollouts.jsonl → GRPO-trl schema
  train_dpo.py                 # trl DPO LoRA r=16
  train_grpo_trl.py            # trl GRPO LoRA (verl fallback)
  convert_to_verl_parquet.py   # verl Parquet schema (for the verl Docker path)
  train_grpo_verl.sh           # verl GRPO main path (requires Docker)
  code_reward.py               # custom reward fn for verl
  merge_lora.py                # merge LoRA adapter into base, save FP16
  quantize_awq.py              # AutoAWQ INT4 + vLLM bench
data/                          # SFT/RL data (gitignored)
outputs/                       # trained models (gitignored)
results/                       # eval + benchmark markdown + jsonl
```

## Known not-yet-done (post-D2)

- verl GRPO via Docker — requires `sudo apt install docker.io nvidia-container-toolkit`, blocked when run autonomously
- W&B logging — runs use `--no-wandb`; metrics are in per-run logs
- Tech blog draft on the DPO-vs-GRPO finding
- 7B variant on Lambda Cloud for $150-240 to validate at production scale

## Stack

- **PyTorch** 2.11 (vllm env) / 2.10 (sft env) · CUDA 13
- **vLLM** 0.21 (inference, eval, rollouts)
- **HuggingFace** transformers 5.5-5.8 · datasets 4.3-4.8 · peft 0.19 · accelerate
- **trl** 0.24 (DPO, GRPO) · **Unsloth** latest (SFT speedup, ~2× over HF Trainer)
- **AutoAWQ** 0.2.9 (INT4 quantization)
- **bitsandbytes** 0.49 (8-bit AdamW for training)
