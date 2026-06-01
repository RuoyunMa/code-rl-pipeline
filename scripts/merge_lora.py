"""
Merge a LoRA adapter into a base model and save FP16 merged weights for vLLM serving.

Usage:
    python scripts/merge_lora.py \\
        --base outputs/sft_merged \\
        --adapter outputs/dpo \\
        --out outputs/dpo_merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path or HF id of base model")
    ap.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    ap.add_argument("--out", required=True, help="Output dir for merged model")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading base: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    print(f"Loading adapter: {args.adapter}")
    peft_model = PeftModel.from_pretrained(base, args.adapter)
    print("Merging ...")
    merged = peft_model.merge_and_unload()
    print(f"Saving merged model -> {out}")
    merged.save_pretrained(out, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(out)
    print("Done.")


if __name__ == "__main__":
    main()
