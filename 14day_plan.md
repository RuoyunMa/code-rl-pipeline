# code-rl-pipeline — 14-Day Execution Plan

**Project:** End-to-end coding LLM post-training demo (SFT → DPO → GRPO + AWQ INT4)
**Owner:** @RuoyunMa
**Hardware:** Single **RTX 5090 32 GB** (Blackwell sm_120)  ·  hardware ready Wed 2026-05-13
**Start:** D1 = first weekend after install ⇒ **2026-05-16 (Sat)**  ·  **End:** **2026-05-31 (Sun)**
**Pace:** Weekdays 4h / Weekends 8h  ·  ~80 work hours  ·  ~50 GPU hours
**North star:** Pipeline runs end-to-end + measurable positive delta + GitHub-ready + writeup-ready. Not SOTA.

> **Hardware-change note (2026-05-11):** original plan assumed A100 40 GB. Audit on 5090 32 GB:
> only **GRPO via verl** needs real config tightening; SFT and DPO fit comfortably (peak 6-13 GB)
> because LoRA + grad-checkpointing keeps activations and optimizer state small. Full details
> in §"Memory budget" below. Schedule re-anchored to install date — alternative anchoring (start
> D1 = Wed 2026-05-13 with weekday hours) is also viable; pick whichever fits work calendar.

---

## Locked technical decisions

| Layer | Choice |
|---|---|
| Base model | Qwen/Qwen2.5-Coder-1.5B-Instruct |
| SFT | Unsloth + LoRA |
| DPO | trl.DPOTrainer + LoRA |
| RL | verl (GRPO)  ·  fallback: trl GRPOTrainer |
| Quantization | AutoAWQ INT4 + vLLM serving |
| Eval | HumanEval + MBPP (+ optional LiveCodeBench / SWE-bench Verified small subset) |
| Tracking | W&B + GitHub |

**Fallback rule:** If M1 (verl Docker) or M4 (verl GRPO dry run) blocks > 1 day → switch to trl GRPOTrainer. Document the rationale in README — it's a +1 signal, not a -1.

---

## Milestones (re-anchored, D1 = 2026-05-16)

| # | Milestone | Target Date | Acceptance |
|---|---|---|---|
| M1 | Infrastructure ready | D2 · 2026-05-17 | Baseline numbers logged · GitHub repo public · verl Docker runs official 1.5B GRPO example |
| M2 | SFT complete | D4 · 2026-05-19 | HumanEval +1–3% over base (revised from +3-6% on 2026-05-19: actual base = 73.17%, near 1.5B Qwen ceiling; see results/D1_journal.md) |
| M3 | RL data ready | D6 · 2026-05-21 | 2000 DPO pairs + 300 verl-format RL prompts |
| M4 | DPO done + GRPO dry run | D8 · 2026-05-24 | DPO +1–3% over SFT · GRPO 5 prompts × 3 steps no OOM |
| M5 | GRPO training complete | D12 · 2026-05-28 | ≥ DPO eval · healthy reward curve |
| M6 | Quantization + delivery | D14 · 2026-05-31 | AWQ INT4 deployed · README · tech blog draft · resume bullets ready |

---

## Day-by-day

> Tick `[x]` as you complete. `GPU` column is rough on-GPU time budget.

### Week 1 — Infra + SFT + Data prep

> **Pre-D1 (Wed 2026-05-13 → Fri 2026-05-15):** hardware install + smoke. Verify
> `nvidia-smi` shows RTX 5090 32 GB · `torch.cuda.get_device_capability()` returns
> `(12, 0)` · CUDA driver ≥ 555 series. Pull verl Docker (Blackwell tag) so D2 EVE
> isn't blocked on download. **Verify verl LoRA key namespace in the pulled image**
> (the script uses `actor_rollout_ref.actor.lora_rank`; key names occasionally
> change between verl releases — grep the image's verl source for `lora_rank` to
> confirm. If LoRA keys differ, full-FT will silently run instead → OOM at ~36 GB.)

#### D1 · ~~Sat 2026-05-16~~ → actual 2026-05-15 (started early)
- [x] **AM** — Run `bash scripts/setup_env.sh` — actually torch **2.11+cu130** (not cu129; pypi default in 2026 is already cu130 — `setup_env.sh` cu129 hardcode is now a bug, see D1 journal) · vllm 0.21 · Blackwell cap (12, 0) verified
- [x] **PM** — Baseline on HumanEval + MBPP — NO_WANDB=1 (W&B login slipped to D2)
- [x] Record baseline numbers in `results/baseline.md` → **HumanEval 73.17% · MBPP 63.42%** (HumanEval 比预期 ~62% 高 11pp, M2 SFT 阈值要重订)

#### D2 · Sun 2026-05-17  ← **M1 deadline**
- [ ] **AM-0** — **Wake 5090 + disable suspend** (`sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`)
- [ ] **AM-1** — Repair `coderl` env damage from sglang attempt (transformers 5.6→5.8.1, nvidia-cudnn-cu13 9.22, nccl 2.30.4, cusparselt 0.9.1, xgrammar 0.2, outlines_core 0.2.14, llguidance>=1.3) · verify `from vllm import LLM` still works
- [ ] **AM-2** — `wandb login` · create GitHub repo `code-rl-pipeline` public · push initial scaffold
- [ ] **PM** — Prep SFT data: download Magicoder-OSS-Instruct, sample ~5000 examples → `data/sft.jsonl`
- [ ] **EVE** — Install docker.io + nvidia-container-toolkit · pull verl Blackwell Docker · run official 1.5B GRPO example (1 step)
- [ ] **EVE/late** — sglang comparison in **separate `coderl_sglang` env** (NOT same env — confirmed conflict in D1)
- [ ] **M1 CHECK** — Baseline logged ✅ · repo public · verl Docker green on 5090

#### D3 · ~~Mon 2026-05-18~~ — collapsed into D2 (compute was way under-budgeted vs A100 plan)
- [x] SFT script + run completed on D2; outputs/sft + outputs/sft_merged saved.

#### D4 · ~~Tue 2026-05-19~~ — collapsed into D2  ← **M2 actual**
- [x] SFT eval: **HumanEval 73.78% (+0.6pp), MBPP 65.76% (+2.3pp)**
- [x] **M2 CHECK** — MBPP comfortably inside +1-3% band; HumanEval marginal (+0.6pp) due to 1.5B ceiling. **PASS** on MBPP, borderline on HE.

#### D5 · ~~Wed 2026-05-20~~ — collapsed into D2
- [x] Rollouts: 374 problems × 4 candidates via vLLM batched in <45 sec (not the planned 3h — 5090 + vllm 0.21 is ~50k input tok/s)

#### D6 · ~~Thu 2026-05-21~~ — collapsed into D2  ← **M3 actual**
- [x] **DPO pairs**: 425 emitted (target was 2000 — fell short because base SFT pass rate 56.9% makes most problems all-pass or all-fail, no preference signal). To reach 2000 need n=8 generations + temperature sweep.
- [x] **RL prompts**: `data/rl_prompts.jsonl` (300 prompts, JSON not Parquet — direct trl GRPO schema since verl Docker route is blocked on sudo)
- [x] **M3 CHECK** — pair count is short of target but functional; data validated end-to-end through DPO + GRPO trl training.

#### D-rest · Fri 2026-05-22 — Rest day / buffer

### Week 2 — DPO + GRPO + Delivery

#### D7 · ~~Sat 2026-05-23~~ — collapsed into D2  ← **M4 actual**
- [x] **DPO** with trl (LoRA r=16): 425 pairs, 27 steps, 29 sec. **HumanEval 75.00% (+1.22pp over SFT, +1.83pp over base) · MBPP 65.76% (flat)**. M4 acceptance PASS on HE.
- [x] **DECISION POINT** triggered: verl Docker route blocked (sudo needed for `docker.io` + `nvidia-container-toolkit`). Took **trl GRPOTrainer fallback** path. Rationale captured in README + final_summary.

#### D8 · ~~Sun 2026-05-24~~ — collapsed into D2
- [x] **GRPO dry run 30 steps**: 96 sec, no OOM, reward signal noisy (expected at this scale), KL healthy 0.0002.

#### D9-D11 — collapsed into D2 ← **M5 attempt**
- [x] **GRPO 200 steps formal-ish run** (verl Docker not available): 651 sec, lr=3e-6. Eval: HumanEval 73.78% (= SFT) · MBPP 66.15% (+0.39pp over SFT). Reward signal still noisy; model barely moved.
- [x] **M5 verdict — partial**: GRPO at 300 prompts + 200 steps + binary verifiable reward did NOT beat DPO at 425 pairs + 27 steps. Real finding for the demo narrative: DPO wins at small data scale; GRPO needs verl + vllm-rollout-integration + 1000+ steps to demonstrate its advantage. Not a project failure — an experimental result worth discussing.

#### D12 — N/A (folded into above)

#### D13 · ~~Sat 2026-05-30~~ — collapsed into D2
- [x] **AWQ INT4 quantization** on `outputs/dpo_merged` → `outputs/dpo_awq_int4` (1.1 GB vs 3.0 GB FP16, 2.7× smaller on disk)
- [x] **vLLM throughput bench** (`results/quantization_benchmark.md`): bf16 7925 tok/s vs INT4 3412 tok/s — **INT4 is 2.3× SLOWER**. Counterintuitive but explainable (1.5B fits in cache → INT4 dequant overhead dominates over bandwidth savings). Real finding for the demo narrative.
- [x] **INT4 accuracy**: HumanEval 73.78% (-1.22pp), MBPP 64.59% (-1.17pp). Quality hit is small but real.
- [ ] **PM** — Draft Chinese tech blog. Deferred (manual writing task).

#### D14 · Sun 2026-05-31  ← **M6 deadline · DELIVERY**
- [ ] **AM** — Compile W&B screenshots + comparison table + demo screenshots → `results/final_summary.md`
- [ ] **PM** — Polish README · publish GitHub repo · finalize resume bullets
- [ ] **M6 CHECK** — Repo public · README complete · blog draft saved · resume materials in `deliverables/`

---

## Risk log (live)

| Risk | Trigger | Mitigation |
|---|---|---|
| **Blackwell stack mismatch** | D1 AM env install | Pin torch≥2.11+cu129, bnb≥0.46 matching cu12x, vllm sm_120 wheel, latest unsloth from git; verify with `python -c "import torch; print(torch.cuda.get_device_capability())"` expects `(12, 0)` |
| **bitsandbytes 5090 wheel missing** | SFT/DPO launch fails on import | Try `pip install -U bitsandbytes`; fall back to `pip install bitsandbytes --pre`; last resort: rebuild from source against installed cu12x |
| **vLLM Blackwell wheel missing** | Baseline eval crashes on LLM() init | Pull vllm nightly: `pip install --pre vllm` or build from source against torch 2.11+cu129 |
| **verl Docker won't start on 5090** | D2 EVE | Pull the latest `verlai/verl` cu128/Blackwell tag from Docker Hub; if none, build from source with torch 2.11+cu129 base; last resort: skip verl, use trl GRPOTrainer fallback (document in README — +1 signal, not -1) |
| SFT no gain | Day 4 AM | Retune LoRA rank · check tokenizer special tokens · check train loss curve |
| Sandbox executor unsafe / slow | Day 5 EVE | Strip imports filter · `subprocess.run(timeout=5, ...)` · run on subset |
| **GRPO verl OOM on dry run** (32 GB tighter than 40 GB) | Day 8 | `rollout.gpu_memory_utilization` 0.30→0.25 · `rollout.n` 4→2 · `max_response_length` 640→512 · `train_batch_size` 16→8 |
| GRPO trl VRAM creep over long runs | Day 9–11 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set proactively) · periodic `torch.cuda.empty_cache()` · drop `num_generations` 4→2 |
| GRPO reward collapse | Day 10–11 | Increase `kl_loss_coef` (0.04 → 0.08) · reduce `lr` to 5e-7 · restart |
| AWQ accuracy drop > 2% | Day 13 | Try `w_bit=8` · or skip quantization, document attempt |

---

## Configs reference

### SFT (Unsloth)
```python
{
    "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "lora_r": 32, "lora_alpha": 64,
    "lr": 5e-5,
    "batch_size": 8, "gradient_accumulation": 4,
    "epochs": 2, "max_seq_length": 2048,
}
```

### DPO (trl)
```python
{
    "beta": 0.1,
    "lr": 5e-6,
    "batch_size": 4, "gradient_accumulation": 4,
    "max_length": 2048,
}
```

### GRPO (verl, 5090-32GB · single GPU · **LoRA r=16** instead of full FT)
```bash
# Tightened for 5090 32GB. Authoritative source: scripts/train_grpo_verl.sh
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/data/code_train.parquet \
  data.val_files=/data/code_val.parquet \
  data.train_batch_size=16 \                  # was 32
  data.max_prompt_length=512 \
  data.max_response_length=640 \              # was 768
  actor_rollout_ref.model.path=outputs/sft_merged \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \              # was 8
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \     # was 2
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.04 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.lora_rank=16 \                       # NEW — verl LoRA support
  actor_rollout_ref.actor.lora_alpha=32 \
  actor_rollout_ref.actor.target_modules=all-linear \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \       # NEW — keeps ref off GPU
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \      # was 0.4
  actor_rollout_ref.rollout.n=4 \
  custom_reward_function.path=/workspace/scripts/code_reward.py \
  custom_reward_function.name=compute_score \
  trainer.logger=['console','wandb'] \
  trainer.project_name=code_grpo_verl \
  trainer.total_epochs=1 \
  trainer.save_freq=10 \
  trainer.test_freq=5 \
  trainer.n_gpus_per_node=1 \
  ray_kwargs.ray_init.num_cpus=8
```

### Memory budget (5090 32GB · per workload · audited 2026-05-11)

| Workload | Peak est. | Fits ≤30 GB? | Notes |
|---|---:|---|---|
| SFT (Unsloth + LoRA r=32, bs=8, seq=2048) | **6-8 GB** | ✅ huge margin | No changes needed |
| DPO (trl + LoRA r=16, bs=4, max_len=2048) | **10-13 GB** | ✅ comfortable | No QLoRA needed; logits ~5 GB is dominant transient |
| GRPO verl — current full-FT config | 28-34 GB | ❌ borderline / OOM | DO NOT run as-is |
| GRPO verl — tightened + LoRA r=16 (above) | **25-28 GB** | ✅ ~3 GB headroom | See knobs above |
| GRPO trl fallback (LoRA r=16, bs=4, num_gen=4) | **12-16 GB** | ✅ | Risk = VRAM creep over long runs |
| vLLM eval / rollout (gpu_mem_util=0.85) | 27 GB (static) | ✅ ~5 GB headroom | KV capacity ~785k tok, well over demand |

**Dominant transients to remember:**
- Logits at vocab=151,936 are the killer: B × seq × ~0.3 MB/tok in bf16 (e.g., DPO chosen+rejected ~5 GB)
- KV cache is small (GQA 2 kv-heads): ~28 KB/token
- nf4-quant base saves only ~2 GB on a 1.5B model — not worth the kernel headache unless scaling to 7B+

### Plan B if OOM (on 5090, in order of escalation)

1. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (always — fights fragmentation)
2. **GRPO verl:** lower `rollout.gpu_memory_utilization` 0.30 → 0.25; `rollout.n` 4 → 2; `max_response_length` 640 → 512
3. **GRPO trl fallback:** `num_generations` 4 → 2; `max_completion_length` 768 → 512; add periodic `torch.cuda.empty_cache()` between rollout batches
4. **DPO** (extremely unlikely to OOM): chunk logits via trl's `loss_type="ipo"` or shrink `max_length` 2048 → 1536
5. **SFT** (won't OOM): N/A

### Recommended Docker image (verl on 5090)
Check Docker Hub for the **latest `verlai/verl` tag with cu128 / Blackwell support** as of install date. The verl team ships updated images quickly. If 2026-05-13 has no Blackwell-tagged image yet, fall back to:
```
verlai/verl:app-verl0.8-cu128-...   # (verify the exact suffix on Docker Hub)
```
If no compatible image exists, build verl from source against torch 2.11+cu129 inside a fresh CUDA 12.9 base image (~30 min). Document in README either way.

---

## Repo layout (target)

```
code-rl-pipeline/
├── README.md
├── data/
│   ├── sft.jsonl
│   ├── dpo_pairs.jsonl
│   ├── code_train.parquet
│   └── code_val.parquet
├── scripts/
│   ├── eval_humaneval.py
│   ├── eval_mbpp.py
│   ├── train_sft.py
│   ├── generate_rollouts.py
│   ├── build_dpo_pairs.py
│   ├── sandbox_executor.py
│   ├── train_dpo.py
│   ├── convert_to_verl_parquet.py
│   ├── train_grpo_verl.sh
│   └── quantize_awq.py
├── configs/
├── results/
│   ├── baseline.md
│   ├── sft_eval.md
│   ├── dpo_eval.md
│   ├── grpo_eval.md
│   ├── quantization_benchmark.md
│   └── final_summary.md
└── deliverables/
    ├── README.md
    └── blog_draft_zh.md
```

---

## Tech blog options (D13 PM)

1. **RL for Coding LLM: From a Recommendation Ranking Perspective** — turns rec background into asset
2. **LLM Post-Training Infra: FSDP / Megatron / verl Design Tradeoffs** — systems angle, plays to ML infra strength
3. **Why Coding Agent Reward Design Is Harder Than General RLHF** — domain-specific insight, hardest to fake

→ Decide on D7 PM so blog can be drafted in parallel.

---

## Open decisions (defer until M2 done)

- [ ] LinkedIn outreach: Week 1 (parallel) vs Week 3 (after demo done)?
- [ ] 7B multi-GPU upgrade on Lambda/RunPod (~$150–240) — yes/no?
- [ ] External applications (Anthropic / OpenAI / xAI) — start in parallel?

---

## Daily log (fill as you go)

### Pre-D1 · 2026-05-13 → 2026-05-15 — hardware install + smoke
- Status:
- Blockers:
- Notes:

### D1 · 2026-05-16
- Status:
- Blockers:
- Notes:

### D2 · 2026-05-17
- Status:
- Blockers:
- Notes:

### D3 · 2026-05-18
- Status:
- Blockers:
- Notes:

### D4 · 2026-05-19
- Status:
- Blockers:
- Notes:

### D5 · 2026-05-20
- Status:
- Blockers:
- Notes:

### D6 · 2026-05-21
- Status:
- Blockers:
- Notes:

### D7 · 2026-05-23
- Status:
- Blockers:
- Notes:

### D8 · 2026-05-24
- Status:
- Blockers:
- Notes:

### D9 · 2026-05-25
- Status:
- Blockers:
- Notes:

### D10 · 2026-05-26
- Status:
- Blockers:
- Notes:

### D11 · 2026-05-27
- Status:
- Blockers:
- Notes:

### D12 · 2026-05-28
- Status:
- Blockers:
- Notes:

### D13 · 2026-05-30
- Status:
- Blockers:
- Notes:

### D14 · 2026-05-31
- Status:
- Blockers:
- Notes:
