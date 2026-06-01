#!/usr/bin/env bash
# 7B SFT sweep on Qwen2.5-Coder-7B-Instruct.
#
# 4 configs:
#   M1: LoRA r=32  all-7 modules, bf16 base
#   M2: LoRA r=128 all-7 modules, bf16 base
#   M3: QLoRA r=32  all-7 modules, nf4 base (load_in_4bit)
#   M4: QLoRA r=128 all-7 modules, nf4 base
#
# Memory estimates (5090 32 GB):
#   M1: ~24 GB    | M2: ~26 GB    | M3: ~11 GB    | M4: ~13 GB
#
# Time estimates (4× 1.5B, so 4× ~21 min = ~85 min per LoRA config):
#   M1: ~80-90 min | M2: ~90 min | M3: ~60-70 min | M4: ~70-80 min
#   Total ~5h
#
# Each variant:
#   1. trains in coderl_sft (Unsloth + 4bit option)
#   2. saves merged FP16 to outputs/7B_<variant>_merged via --save-merged
#   3. evals HumanEval + MBPP + LiveCodeBench in coderl env

set -eo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/7b_sweep results/7b_sweep
source ~/miniconda3/etc/profile.d/conda.sh

run_variant() {
    local name=$1
    local extra_train_args=$2

    echo
    echo "═══════════════════════════════════════════════════════════"
    echo "[7B SWEEP] $name  ($(date -u +%H:%M:%SZ))"
    echo "═══════════════════════════════════════════════════════════"

    if [ -f "results/7b_sweep/${name}.summary.json" ]; then
        echo "  skip: ${name}.summary.json present"
        return 0
    fi

    local out_lora="outputs/7B_${name}"
    local out_merged="outputs/7B_${name}_merged"

    # --- train ---
    if [ -d "$out_merged" ] && [ -f "$out_merged/model.safetensors" ]; then
        echo "  merged model already exists, skipping training"
        local train_sec=0
    else
        conda activate coderl_sft
        local t0=$(date +%s)
        # bs=4 ga=8 to control memory; effective bs=32 (same as 1.5B)
        python scripts/train_sft.py \
            --model Qwen/Qwen2.5-Coder-7B-Instruct \
            --output-dir "$out_lora" \
            --bs 4 --ga 8 \
            --epochs 2 \
            --target-modules all \
            $extra_train_args \
            --save-merged --no-wandb \
            --run-name "sweep-7B-${name}" \
            > "logs/7b_sweep/${name}.train.log" 2>&1
        local train_sec=$(($(date +%s) - t0))
        echo "  train done in ${train_sec}s"
        conda deactivate
    fi

    # --- eval (HumanEval + MBPP + LiveCodeBench) ---
    if [ ! -d "$out_merged" ]; then
        echo "  ERROR: $out_merged not found, can't eval"
        return 0
    fi

    conda activate coderl
    local t0=$(date +%s)
    python scripts/eval_humaneval.py --model "$out_merged" \
        --out "results/7b_sweep/${name}_humaneval.jsonl" --no-wandb \
        >> "logs/7b_sweep/${name}.eval.log" 2>&1 || echo "  HumanEval failed"
    python scripts/eval_mbpp.py --model "$out_merged" --split sanitized \
        --out "results/7b_sweep/${name}_mbpp.jsonl" --no-wandb \
        >> "logs/7b_sweep/${name}.eval.log" 2>&1 || echo "  MBPP failed"
    python scripts/eval_livecodebench.py --model "$out_merged" \
        --out "results/7b_sweep/${name}_lcb.jsonl" \
        >> "logs/7b_sweep/${name}.eval.log" 2>&1 || echo "  LCB failed"
    local eval_sec=$(($(date +%s) - t0))
    echo "  eval done in ${eval_sec}s"
    conda deactivate

    # --- combine summary ---
    $HOME/miniconda3/envs/coderl/bin/python -c "
import json, os

def maybe(p):
    return json.load(open(p)) if os.path.exists(p) else None

he = maybe('results/7b_sweep/${name}_humaneval.summary.json')
mb = maybe('results/7b_sweep/${name}_mbpp.summary.json')
lcb = maybe('results/7b_sweep/${name}_lcb.summary.json')

summary = {
    'variant': '${name}',
    'train_sec': ${train_sec},
    'eval_sec': ${eval_sec},
    'humaneval_pass1': he['pass@1'] if he else None,
    'humaneval_n_pass': he['n_pass'] if he else None,
    'mbpp_pass1': mb['pass@1'] if mb else None,
    'mbpp_n_pass': mb['n_pass'] if mb else None,
    'lcb_pass1': lcb['pass@1'] if lcb else None,
    'lcb_n_pass': lcb['n_pass'] if lcb else None,
    'lcb_post_cutoff_pass1': lcb['post_cutoff_subset']['pass@1'] if lcb else None,
}
json.dump(summary, open('results/7b_sweep/${name}.summary.json', 'w'), indent=2)
print(json.dumps(summary, indent=2))
"
}

# === M1: LoRA r=32 ===
run_variant "M1_lora_r32" "--lora-r 32 --lora-alpha 64"

# === M2: LoRA r=128 ===
run_variant "M2_lora_r128" "--lora-r 128 --lora-alpha 256"

# === M3: QLoRA r=32 ===
run_variant "M3_qlora_r32" "--lora-r 32 --lora-alpha 64 --load-in-4bit"

# === M4: QLoRA r=128 ===
run_variant "M4_qlora_r128" "--lora-r 128 --lora-alpha 256 --load-in-4bit"

echo
echo "═══════════════════════════════════════════════════════════"
echo "[7B SWEEP DONE] all 4 variants"
echo "═══════════════════════════════════════════════════════════"

# Final aggregation
echo
echo "[7B sweep results]"
for f in results/7b_sweep/*.summary.json; do
    name=$(basename "$f" .summary.json)
    if [[ "$name" == *_humaneval ]] || [[ "$name" == *_mbpp ]] || [[ "$name" == *_lcb ]]; then continue; fi
    $HOME/miniconda3/envs/coderl/bin/python -c "
import json, sys
d = json.load(open(sys.argv[1]))
he = '{:.2f}%'.format(d['humaneval_pass1']*100) if d.get('humaneval_pass1') is not None else 'n/a'
mb = '{:.2f}%'.format(d['mbpp_pass1']*100) if d.get('mbpp_pass1') is not None else 'n/a'
lcb = '{:.2f}%'.format(d['lcb_pass1']*100) if d.get('lcb_pass1') is not None else 'n/a'
t = d.get('train_sec', '?')
print('  {:24} HE={}  MBPP={}  LCB={}  train={}s'.format(d['variant'], he, mb, lcb, t))
" "$f"
done
echo
echo "Done at $(date -u +%H:%M:%SZ)"
