"""
D13 — AutoAWQ INT4 quantization + vLLM throughput / latency benchmark.

Pipeline:
    1. Load fp16/bf16 model (the best post-RL checkpoint, e.g. outputs/grpo_merged)
    2. Calibrate AWQ on a small set of code-instruction prompts (~128 samples)
    3. Save INT4 weights
    4. Benchmark vLLM throughput + latency: bf16 baseline vs INT4
    5. Print a comparison table; persist results/quantization_benchmark.md

Accuracy comparison is done separately by re-running:
    python scripts/eval_humaneval.py --model outputs/grpo_awq_int4 --run-name eval-int4

Usage:
    python scripts/quantize_awq.py \\
        --model outputs/grpo_merged \\
        --out outputs/grpo_awq_int4 \\
        --calib-n 128 \\
        --bench-prompts 100

Memory: AWQ calibration on 1.5B model ~ 8-12 GB. Benchmarks fit easily.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median


def quantize(args) -> None:
    """Run AutoAWQ INT4 calibration + save."""
    print(f"[quantize] loading model: {args.model}")
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    from datasets import load_dataset

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    model = AutoAWQForCausalLM.from_pretrained(
        args.model, safetensors=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"[quantize] loading calibration data: {args.calib_dataset} (n={args.calib_n})")
    ds = load_dataset(args.calib_dataset, split="train")
    ds = ds.shuffle(seed=42).select(range(min(args.calib_n, len(ds))))
    pf, sf = args.problem_field, args.solution_field
    if pf not in ds.column_names or sf not in ds.column_names:
        raise SystemExit(
            f"Calibration dataset columns {ds.column_names} don't contain "
            f"--problem-field='{pf}' / --solution-field='{sf}'."
        )

    def fmt(ex):
        msgs = [
            {"role": "user", "content": ex[pf]},
            {"role": "assistant", "content": ex[sf]},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    calib_data = [fmt(ex) for ex in ds]

    print("[quantize] running AWQ calibration ...")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"[quantize] saving INT4 weights -> {args.out}")
    model.save_quantized(args.out)
    tokenizer.save_pretrained(args.out)

    # Persist quant config for traceability
    Path(args.out, "awq_config.json").write_text(json.dumps(quant_config, indent=2))

    # Free GPU memory before downstream bench_vllm() — otherwise the bf16 LLM
    # constructor reports "Free memory < desired GPU memory utilization" because
    # the calibration model is still resident.
    del model
    del tokenizer
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def bench_vllm(model_path: str, n_prompts: int, max_tokens: int,
               quantization: str | None, label: str) -> dict:
    """Benchmark vLLM throughput + latency on a fixed prompt set.

    IMPORTANT: this function explicitly tears down vLLM at the end so that
    callers can safely run a second `bench_vllm(...)` in the same process
    without OOM (vLLM holds a large KV-cache pool until the LLM object is
    GC'd AND CUDA caches are emptied).
    """
    from vllm import LLM, SamplingParams
    from datasets import load_dataset

    print(f"\n[bench:{label}] loading model: {model_path}"
          + (f" (quantization={quantization})" if quantization else ""))
    # AWQ kernels are tuned for fp16; bf16 path uses bf16 weights+activations.
    dtype = "float16" if quantization == "awq" else "bfloat16"
    kwargs = dict(model=model_path, dtype=dtype, gpu_memory_utilization=0.85)
    if quantization:
        kwargs["quantization"] = quantization
    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()

    # Use HumanEval prompts as the benchmark set — realistic for our use case.
    ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    prompts_src = list(ds)[:n_prompts]

    SYS = (
        "You are an expert Python programmer. "
        "Complete the function. Respond with the full function in one ```python block."
    )

    def build(p):
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"```python\n{p['prompt']}```"},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompts = [build(p) for p in prompts_src]

    # Warmup
    print(f"[bench:{label}] warmup ...")
    llm.generate(prompts[:3], SamplingParams(temperature=0.0, max_tokens=64))

    # Throughput pass: greedy, max_tokens fixed
    print(f"[bench:{label}] throughput pass ({len(prompts)} prompts, max_tokens={max_tokens})")
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = out_tokens / elapsed

    # Per-prompt latency: 1-by-1 single-prompt (proxy for serving latency)
    # Use a smaller subset to keep this fast.
    print(f"[bench:{label}] per-prompt latency on first 10 prompts ...")
    latencies = []
    for p in prompts[:10]:
        t = time.time()
        llm.generate([p], SamplingParams(temperature=0.0, max_tokens=128))
        latencies.append(time.time() - t)

    stats = {
        "label": label,
        "model": model_path,
        "quantization": quantization or "none",
        "n_prompts": n_prompts,
        "max_tokens": max_tokens,
        "total_output_tokens": out_tokens,
        "elapsed_sec": round(elapsed, 2),
        "throughput_tok_per_sec": round(throughput, 1),
        "latency_p50_sec": round(median(latencies), 3),
        "latency_mean_sec": round(mean(latencies), 3),
        "latency_max_sec": round(max(latencies), 3),
    }

    # --- Tear down vLLM so a subsequent bench_vllm() can grab the GPU.
    # Without this, the second LLM() will OOM on 5090-32GB because vLLM holds
    # the KV-cache pool until both Python GC and CUDA caches are released.
    try:
        from vllm.distributed import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:  # noqa: BLE001
        pass
    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    return stats


def write_report(bf16_stats: dict, int4_stats: dict, out_path: Path) -> None:
    speedup = (
        int4_stats["throughput_tok_per_sec"] / bf16_stats["throughput_tok_per_sec"]
        if bf16_stats["throughput_tok_per_sec"] else 0.0
    )
    p50_speedup = (
        bf16_stats["latency_p50_sec"] / int4_stats["latency_p50_sec"]
        if int4_stats["latency_p50_sec"] else 0.0
    )
    md = f"""# Quantization Benchmark — AWQ INT4 vs bf16

**Bench prompts:** {bf16_stats['n_prompts']}  ·  **max_tokens:** {bf16_stats['max_tokens']}

| Metric | bf16 baseline | AWQ INT4 | Δ |
|---|---:|---:|---:|
| Throughput (tok/s) | {bf16_stats['throughput_tok_per_sec']} | {int4_stats['throughput_tok_per_sec']} | **{speedup:.2f}×** |
| Total elapsed (s) | {bf16_stats['elapsed_sec']} | {int4_stats['elapsed_sec']} | — |
| Output tokens | {bf16_stats['total_output_tokens']} | {int4_stats['total_output_tokens']} | — |
| Latency p50 (s) | {bf16_stats['latency_p50_sec']} | {int4_stats['latency_p50_sec']} | **{p50_speedup:.2f}×** |
| Latency mean (s) | {bf16_stats['latency_mean_sec']} | {int4_stats['latency_mean_sec']} | — |
| Latency max (s) | {bf16_stats['latency_max_sec']} | {int4_stats['latency_max_sec']} | — |

## Models

- bf16:  `{bf16_stats['model']}`
- int4:  `{int4_stats['model']}`  (quantization = `{int4_stats['quantization']}`)

## Accuracy

Run `eval_humaneval.py` separately on both models to compare pass@1.
The accuracy delta belongs in `results/final_summary.md`, not here.
"""
    out_path.write_text(md)
    print(f"\n[report] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/grpo_merged",
                    help="bf16 model to quantize (and benchmark as baseline)")
    ap.add_argument("--out", default="outputs/grpo_awq_int4",
                    help="path to save AWQ INT4 weights")
    ap.add_argument("--calib-dataset", default="ise-uiuc/Magicoder-OSS-Instruct-75K")
    ap.add_argument("--calib-n", type=int, default=128)
    ap.add_argument("--problem-field", default="problem")
    ap.add_argument("--solution-field", default="solution")
    ap.add_argument("--bench-prompts", type=int, default=100)
    ap.add_argument("--bench-max-tokens", type=int, default=512)
    ap.add_argument("--report", default="results/quantization_benchmark.md")
    ap.add_argument("--skip-quantize", action="store_true",
                    help="reuse existing INT4 weights at --out")
    ap.add_argument("--skip-bf16-bench", action="store_true")
    ap.add_argument("--skip-int4-bench", action="store_true")
    args = ap.parse_args()

    if not args.skip_quantize:
        quantize(args)

    bf16_stats = None
    int4_stats = None

    if not args.skip_bf16_bench:
        bf16_stats = bench_vllm(
            model_path=args.model,
            n_prompts=args.bench_prompts,
            max_tokens=args.bench_max_tokens,
            quantization=None,
            label="bf16",
        )
        print(json.dumps(bf16_stats, indent=2))

    if not args.skip_int4_bench:
        int4_stats = bench_vllm(
            model_path=args.out,
            n_prompts=args.bench_prompts,
            max_tokens=args.bench_max_tokens,
            quantization="awq",
            label="awq_int4",
        )
        print(json.dumps(int4_stats, indent=2))

    if bf16_stats and int4_stats:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_report(bf16_stats, int4_stats, report_path)


if __name__ == "__main__":
    main()
