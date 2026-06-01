# Quantization Benchmark — AWQ INT4 vs bf16

**Bench prompts:** 100  ·  **max_tokens:** 512

| Metric | bf16 baseline | AWQ INT4 | Δ |
|---|---:|---:|---:|
| Throughput (tok/s) | 7924.7 | 3411.8 | **0.43×** |
| Total elapsed (s) | 1.96 | 5.19 | — |
| Output tokens | 15570 | 17696 | — |
| Latency p50 (s) | 0.416 | 1.209 | **0.34×** |
| Latency mean (s) | 0.397 | 1.169 | — |
| Latency max (s) | 0.418 | 1.357 | — |

## Models

- bf16:  `outputs/dpo_merged`
- int4:  `outputs/dpo_awq_int4`  (quantization = `awq`)

## Accuracy

Run `eval_humaneval.py` separately on both models to compare pass@1.
The accuracy delta belongs in `results/final_summary.md`, not here.
