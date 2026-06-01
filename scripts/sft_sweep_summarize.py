"""
Aggregate per-variant summary jsons in results/sweep/ into a markdown table.

Comparison rows include the baseline (D1) and the original SFT (M2) so the
sweep numbers can be read against the prior state.

Output: results/lora_sweep.md
"""

from __future__ import annotations

import json
from pathlib import Path


# Known prior runs to include as reference rows
PRIOR_RUNS = [
    {
        "variant": "[base] FP16 (no SFT)",
        "trainable_params": 0,
        "trainable_pct": 0.0,
        "train_sec": 0,
        "humaneval_pass1": 0.7317073170731707,
        "mbpp_pass1": 0.6342412451361867,
        "notes": "Qwen2.5-Coder-1.5B-Instruct as-is, no training",
    },
    {
        "variant": "[ref] A1: LoRA r=32 all-7 (original SFT)",
        "trainable_params": 36929536,
        "trainable_pct": 2.34,
        "train_sec": 1274,
        "humaneval_pass1": 0.7378048780487805,
        "mbpp_pass1": 0.6575875486381323,
        "notes": "the SFT we used for downstream DPO/GRPO",
    },
]

VARIANT_NOTES = {
    "A4_r64_all":      ("LoRA r=64 all-7 modules",     73.9e6,   4.68),
    "A5_r128_all":     ("LoRA r=128 all-7 modules",    147.7e6,  9.35),
    "B1_r32_attn":     ("LoRA r=32 attn-only (qkvo)",  5.9e6,    0.38),
    "B2_r32_mlp":      ("LoRA r=32 MLP-only (gate/up/down)", 28.2e6, 1.79),
    "C1_fullft_lr5e5": ("Full FT lr=5e-5 8bitAdam",    1580e6,   100.0),
    "C3_fullft_lr1e5": ("Full FT lr=1e-5 8bitAdam",    1580e6,   100.0),
}


def fmt_secs(s):
    if s < 60:
        return f"{s}s"
    return f"{s//60}m{s%60:02d}s"


def fmt_pct(p):
    return f"{p*100:.2f}%"


def fmt_params(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    return f"{n/1e3:.0f}K"


def main():
    sweep_dir = Path("results/sweep")
    rows = list(PRIOR_RUNS)

    for variant, (label, n_params, pct) in VARIANT_NOTES.items():
        f = sweep_dir / f"{variant}.summary.json"
        if not f.exists():
            rows.append({
                "variant": label,
                "trainable_params": n_params,
                "trainable_pct": pct,
                "train_sec": None,
                "humaneval_pass1": None,
                "mbpp_pass1": None,
                "notes": "(not run yet)",
            })
            continue
        d = json.loads(f.read_text())
        rows.append({
            "variant": label,
            "trainable_params": n_params,
            "trainable_pct": pct,
            "train_sec": d["train_sec"],
            "humaneval_pass1": d["humaneval_pass1"],
            "mbpp_pass1": d["mbpp_pass1"],
            "notes": "",
        })

    base_he = PRIOR_RUNS[0]["humaneval_pass1"]
    base_mb = PRIOR_RUNS[0]["mbpp_pass1"]

    md = ["# SFT ablation sweep — 1.5B Qwen2.5-Coder on Magicoder 5000 samples\n"]
    md.append("Effective batch = bs × ga = 32, 2 epochs (314 steps for LoRA variants), "
              "max_seq=2048. Eval at T=0 greedy, n=1.\n")
    md.append("| Variant | Trainable | % of total | Train time | HumanEval pass@1 | Δ vs base | MBPP pass@1 | Δ vs base | Notes |")
    md.append("|---|---:|---:|---:|---:|:--|---:|:--|---|")

    for r in rows:
        he = r["humaneval_pass1"]
        mb = r["mbpp_pass1"]
        if he is None:
            he_str = "—"
            he_d = "—"
            mb_str = "—"
            mb_d = "—"
            train_str = "—"
        else:
            he_str = fmt_pct(he)
            he_d = f"{(he - base_he)*100:+.2f}pp"
            mb_str = fmt_pct(mb)
            mb_d = f"{(mb - base_mb)*100:+.2f}pp"
            train_str = fmt_secs(r["train_sec"])

        md.append(
            f"| {r['variant']} "
            f"| {fmt_params(r['trainable_params'])} "
            f"| {r['trainable_pct']:.2f}% "
            f"| {train_str} "
            f"| {he_str} | {he_d} "
            f"| {mb_str} | {mb_d} "
            f"| {r['notes']} |"
        )

    out_path = Path("results/lora_sweep.md")
    out_path.write_text("\n".join(md) + "\n")
    print(f"Wrote {out_path}")
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
