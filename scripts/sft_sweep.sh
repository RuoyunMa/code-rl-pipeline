#!/usr/bin/env bash
# SFT ablation sweep: LoRA rank / target_modules / Full FT × lr
#
# Each variant:
#   1. trains in coderl_sft env (with unsloth for LoRA / vanilla transformers for full FT)
#   2. saves to outputs/<variant>_merged
#   3. evals HumanEval + MBPP in coderl env
#   4. writes results/<variant>.summary.json with combined stats
#
# Variants planned (see results/lora_sweep.md after run):
#   A4: LoRA r=64 all-modules    | unsloth + train_sft.py --lora-r 64 --lora-alpha 128
#   A5: LoRA r=128 all-modules   | unsloth + train_sft.py --lora-r 128 --lora-alpha 256
#   B1: LoRA r=32 attn-only      | unsloth + train_sft.py --target-modules attn
#   B2: LoRA r=32 mlp-only       | unsloth + train_sft.py --target-modules mlp
#   C1: Full FT lr=5e-5          | train_sft_full.py --lr 5e-5
#   C3: Full FT lr=1e-5          | train_sft_full.py --lr 1e-5
#
# Usage:
#   bash scripts/sft_sweep.sh
#
# Output:
#   logs/sweep/<variant>.{train,eval}.log
#   results/sweep/<variant>.summary.json
#   results/lora_sweep.md  (aggregated table — written by sft_sweep_summarize.py at end)

set -eo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/sweep results/sweep

source ~/miniconda3/etc/profile.d/conda.sh

run_variant() {
    local name=$1
    local env=$2          # coderl_sft for SFT, coderl for eval
    local cmd=$3
    local out_model=$4    # path to merged FP16 dir to eval

    echo
    echo "═══════════════════════════════════════════════════════════"
    echo "[SWEEP] $name  ($(date -u +%H:%M:%SZ))"
    echo "═══════════════════════════════════════════════════════════"

    # Skip if eval results already exist
    if [ -f "results/sweep/${name}.summary.json" ]; then
        echo "  skip: results/sweep/${name}.summary.json already present"
        return 0
    fi

    # --- train (skip if eval files already exist, e.g. crash mid-run) ---
    local t0=$(date +%s)
    local train_sec=0
    if [ -f "results/sweep/${name}_humaneval.summary.json" ] && \
       [ -f "results/sweep/${name}_mbpp.summary.json" ]; then
        echo "  train+eval already done, just need to recombine summary"
        # Try to read train_sec from prior log
        train_sec=$(grep -oE "train_runtime[^,]+" "logs/sweep/${name}.train.log" 2>/dev/null \
            | tail -1 | grep -oE "[0-9]+\.?[0-9]*" | head -1 | awk '{print int($1)}')
        [ -z "$train_sec" ] && train_sec=0
    else
        conda activate "$env"
        eval "$cmd" > "logs/sweep/${name}.train.log" 2>&1
        train_sec=$(($(date +%s) - t0))
        echo "  train done in ${train_sec}s"
        conda deactivate
    fi

    # --- merge LoRA if needed ---
    # Full FT scripts already write FP16 directly to out_model. LoRA scripts
    # need merge_lora.py to produce a vllm-loadable FP16.
    if [[ "$name" == A* || "$name" == B* ]]; then
        local lora_dir="outputs/${name}"
        if [ ! -d "$out_model" ]; then
            conda activate coderl_sft
            python scripts/merge_lora.py --base Qwen/Qwen2.5-Coder-1.5B-Instruct \
                --adapter "$lora_dir" --out "$out_model" >> "logs/sweep/${name}.train.log" 2>&1
            conda deactivate
        fi
    fi

    # --- eval (skip if both summary files already exist) ---
    local eval_sec=0
    if [ -f "results/sweep/${name}_humaneval.summary.json" ] && \
       [ -f "results/sweep/${name}_mbpp.summary.json" ]; then
        echo "  eval files already exist, skipping eval"
    else
        conda activate coderl
        t0=$(date +%s)
        python scripts/eval_humaneval.py --model "$out_model" \
            --out "results/sweep/${name}_humaneval.jsonl" --no-wandb \
            >> "logs/sweep/${name}.eval.log" 2>&1
        python scripts/eval_mbpp.py --model "$out_model" --split sanitized \
            --out "results/sweep/${name}_mbpp.jsonl" --no-wandb \
            >> "logs/sweep/${name}.eval.log" 2>&1
        eval_sec=$(($(date +%s) - t0))
        echo "  eval done in ${eval_sec}s"
        conda deactivate
    fi

    # --- combine summary (use absolute path to a known-good python) ---
    $HOME/miniconda3/envs/coderl/bin/python -c "
import json
he = json.load(open('results/sweep/${name}_humaneval.summary.json'))
mb = json.load(open('results/sweep/${name}_mbpp.summary.json'))
summary = {
    'variant': '${name}',
    'train_sec': ${train_sec},
    'eval_sec': ${eval_sec},
    'humaneval_pass1': he['pass@1'],
    'humaneval_n_pass': he['n_pass'],
    'mbpp_pass1': mb['pass@1'],
    'mbpp_n_pass': mb['n_pass'],
}
json.dump(summary, open('results/sweep/${name}.summary.json', 'w'), indent=2)
print(json.dumps(summary, indent=2))
"
}

# === A4: LoRA r=64 all-modules ===
run_variant "A4_r64_all" coderl_sft \
    "python scripts/train_sft.py --output-dir outputs/A4_r64_all \
        --lora-r 64 --lora-alpha 128 --target-modules all \
        --save-merged --no-wandb --run-name sweep-A4-r64" \
    "outputs/A4_r64_all_merged"

# === A5: LoRA r=128 all-modules ===
run_variant "A5_r128_all" coderl_sft \
    "python scripts/train_sft.py --output-dir outputs/A5_r128_all \
        --lora-r 128 --lora-alpha 256 --target-modules all \
        --save-merged --no-wandb --run-name sweep-A5-r128" \
    "outputs/A5_r128_all_merged"

# === B1: LoRA r=32 attn-only ===
run_variant "B1_r32_attn" coderl_sft \
    "python scripts/train_sft.py --output-dir outputs/B1_r32_attn \
        --lora-r 32 --lora-alpha 64 --target-modules attn \
        --save-merged --no-wandb --run-name sweep-B1-attn" \
    "outputs/B1_r32_attn_merged"

# === B2: LoRA r=32 mlp-only ===
run_variant "B2_r32_mlp" coderl_sft \
    "python scripts/train_sft.py --output-dir outputs/B2_r32_mlp \
        --lora-r 32 --lora-alpha 64 --target-modules mlp \
        --save-merged --no-wandb --run-name sweep-B2-mlp" \
    "outputs/B2_r32_mlp_merged"

# === C1: Full FT lr=5e-5 ===
run_variant "C1_fullft_lr5e5" coderl_sft \
    "python scripts/train_sft_full.py --output-dir outputs/C1_fullft_lr5e5 \
        --lr 5e-5 --no-wandb --run-name sweep-C1-fullft-lr5e5" \
    "outputs/C1_fullft_lr5e5"

# === C3: Full FT lr=1e-5 ===
run_variant "C3_fullft_lr1e5" coderl_sft \
    "python scripts/train_sft_full.py --output-dir outputs/C3_fullft_lr1e5 \
        --lr 1e-5 --no-wandb --run-name sweep-C3-fullft-lr1e5" \
    "outputs/C3_fullft_lr1e5"

echo
echo "═══════════════════════════════════════════════════════════"
echo "[SWEEP DONE] aggregating results -> results/lora_sweep.md"
echo "═══════════════════════════════════════════════════════════"
python scripts/sft_sweep_summarize.py
echo
echo "All done at $(date -u +%H:%M:%SZ)"
