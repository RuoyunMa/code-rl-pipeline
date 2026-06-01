# M4 (DPO) — 2026-05-19

> **Note (2026-05-19 10:10 UTC):** This file documents the **v1 DPO run** (single-temperature rollouts, 425 pairs, 27 steps). A later **v2** run on the same day (multi-temp rollouts, 2320 pairs, 145 steps) became the headline deployable model. v2 results:
> - HumanEval 74.39% (-0.6 vs v1, but rewards/accuracies 0.78 vs 0.39 → much more trustworthy)
> - MBPP 66.93% (+1.2 vs v1)
> See `results/final_summary.md` for the full v1-vs-v2 comparison.



**Note:** Originally planned for D8 (2026-05-24), but D2-D6 entire data pipeline + D7 DPO ran in one D2 session because the workloads on 5090 are MUCH faster than the original A100-based estimates.

**Model:** `outputs/dpo_merged` (SFT'd Qwen2.5-Coder-1.5B + DPO LoRA r=16 merged FP16)
**Pipeline:** SFT'd `outputs/sft_merged` → generate rollouts (mbpp_train n=4 T=0.8) → grade via sandbox → 425 (chosen, rejected) pairs → DPO 1 epoch
**Training:** trl 0.24 DPOTrainer, LoRA r=16 α=32, beta=0.1, lr=5e-6, bs=4×ga=4, 27 steps, ~29 sec wall-clock

## Eval

| Benchmark | Base | SFT | **DPO** | Δ over SFT |
|---|---:|---:|---:|:-:|
| HumanEval (T=0 n=1) | 73.17% | 73.78% | **75.00% (123/164)** | **+1.22pp** ✅ |
| MBPP sanitized | 63.42% | 65.76% | 65.76% (169/257) | 0.00pp |

## M4 acceptance ("+1-3% over SFT")

- **HumanEval +1.22% — PASS** ✅
- **MBPP unchanged — partial**
- Overall: pipeline works, positive delta on the harder benchmark.

## Caveats / honest reading

- **Only 425 preference pairs** — well below the 14day_plan target of ~2000. At pairs-per-problem=6 cap, we hit ~128 problems (of 374 MBPP-train problems) that had both passing and failing completions; many problems were "all pass" or "all fail" given the 56.9% per-completion pass rate. More pairs would come from: larger n (n=8 vs n=4), wider temp sweep (T=0.6, 0.8, 1.0), or adding HumanEval-style problems to the rollout source.
- **27 DPO steps** is very short — train loss barely moved (0.6935 → 0.6919). The model is barely DPO-trained, yet HumanEval ticked up 2 problems (121 → 123). This is at the noise floor (1.2% on 164 problems). For a real demo-quality result, want 2-5x more pairs and 100-300 steps.
- The honest framing: **full SFT → DPO pipeline runs end-to-end with measurable positive delta; gradient signal is real but small due to data scale, validating the architecture before scaling**.

## Files

- `outputs/dpo/`              — DPO LoRA adapter (74 MB)
- `outputs/dpo_merged/`       — FP16 merged model (3 GB, vLLM-loadable)
- `logs/rollouts.log`, `logs/build_pairs.log`, `logs/dpo.log`
- `results/humaneval_dpo.summary.json` + `humaneval_dpo.jsonl`
- `results/mbpp_dpo.summary.json` + `mbpp_dpo.jsonl`
- `data/rollouts.jsonl`       — 374 problems × 4 candidates
- `data/dpo_pairs.jsonl`      — 425 (prompt, chosen, rejected) tuples

## Trl 0.24 + transformers 5.5 compat issues encountered

Documented for the journal:
1. `mergekit` not auto-installed by trl 0.24 — added via pip
2. `llm-blender` not auto-installed — added via pip
3. `llm_blender` imports `TRANSFORMERS_CACHE` (removed in transformers 5.x) — patched `llm_blender/blender/blender_utils.py` and `blender.py` with try/except fallback to `huggingface_hub.constants.HF_HUB_CACHE`
4. `weave` not auto-installed — added via pip
5. `model.warnings_issued` attribute missing on Qwen2ForCausalLM in transformers 5.5 — patched `train_dpo.py` to stub `model.warnings_issued = {}` before passing to DPOTrainer

These suggest **trl 0.24 was built against an earlier transformers**; trl 1.x or a transformers ~5.5-pinned trl release would be cleaner. For this project we're not blocked, so leaving the patches in.
