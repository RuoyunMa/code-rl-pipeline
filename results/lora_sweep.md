# SFT ablation sweep — 1.5B Qwen2.5-Coder on Magicoder 5000 samples

Effective batch = bs × ga = 32, 2 epochs (314 steps for LoRA variants), max_seq=2048. Eval at T=0 greedy, n=1.

| Variant | Trainable | % of total | Train time | HumanEval pass@1 | Δ vs base | MBPP pass@1 | Δ vs base | Notes |
|---|---:|---:|---:|---:|:--|---:|:--|---|
| [base] FP16 (no SFT) | 0K | 0.00% | 0s | 73.17% | +0.00pp | 63.42% | +0.00pp | Qwen2.5-Coder-1.5B-Instruct as-is, no training |
| [ref] A1: LoRA r=32 all-7 (original SFT) | 36.9M | 2.34% | 21m14s | 73.78% | +0.61pp | 65.76% | +2.33pp | the SFT we used for downstream DPO/GRPO |
| LoRA r=64 all-7 modules | 73.9M | 4.68% | 21m19s | 70.73% | -2.44pp | 66.15% | +2.72pp |  |
| LoRA r=128 all-7 modules | 147.7M | 9.35% | 22m15s | 74.39% | +1.22pp | 69.65% | +6.23pp |  |
| LoRA r=32 attn-only (qkvo) | 5.9M | 0.38% | 21m07s | 73.17% | +0.00pp | 65.37% | +1.95pp |  |
| LoRA r=32 MLP-only (gate/up/down) | 28.2M | 1.79% | 21m14s | 73.17% | +0.00pp | 66.54% | +3.11pp |  |
| Full FT lr=5e-5 8bitAdam | 1.58B | 100.00% | 11m31s | 73.17% | +0.00pp | 70.43% | +7.00pp |  |
| Full FT lr=1e-5 8bitAdam | 1.58B | 100.00% | 11m30s | 72.56% | -0.61pp | 64.20% | +0.78pp |  |
