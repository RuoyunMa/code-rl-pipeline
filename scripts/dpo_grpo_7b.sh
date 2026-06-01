#!/usr/bin/env bash
# 7B DPO + GRPO end-to-end on top of best 7B SFT.
#
# Base: outputs/7B_M1_lora_r32_merged (chosen because best MBPP + LCB)
#
# Steps:
#   1. Generate rollouts (multi-temp, mbpp_train) with 7B SFT
#   2. Build DPO pairs (parallel sandbox grading)
#   3. Train DPO LoRA r=16 (trl in coderl_sft)
#   4. Merge -> outputs/7B_dpo_merged
#   5. Eval HE + MBPP + LCB
#   6. Prep RL prompts, train GRPO 200 steps
#   7. Merge -> outputs/7B_grpo_merged
#   8. Eval HE + MBPP + LCB

set -eo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/7b_dpo_grpo results/7b_dpo_grpo
source ~/miniconda3/etc/profile.d/conda.sh

BASE_MERGED="outputs/7B_M1_lora_r32_merged"
LOGD="logs/7b_dpo_grpo"
RESD="results/7b_dpo_grpo"

# ============================================
# STEP 1: Rollouts with 7B SFT (multi-temp)
# ============================================
echo "=== [7B] Step 1: Multi-temp rollouts ==="
if [ -f data/7B_rollouts_multi.jsonl ]; then
    echo "  skip: rollouts file present"
else
    conda activate coderl
    for T in 0.4 0.8 1.2; do
        OUT="data/7B_rollouts_T${T/./}.jsonl"
        echo "  generating T=$T -> $OUT"
        python scripts/generate_rollouts.py --model "$BASE_MERGED" \
            --source mbpp_train --n 4 --temp $T --out "$OUT" \
            > "$LOGD/rollouts_T${T/./}.log" 2>&1
    done
    # Merge
    $HOME/miniconda3/envs/coderl/bin/python -c "
import json
merged = {}
for p in ['data/7B_rollouts_T04.jsonl', 'data/7B_rollouts_T08.jsonl', 'data/7B_rollouts_T12.jsonl']:
    for line in open(p):
        r = json.loads(line)
        m = merged.setdefault(r['task_id'], {**r, 'completions': []})
        m['completions'].extend(r['completions'])
with open('data/7B_rollouts_multi.jsonl', 'w') as f:
    for k in sorted(merged):
        f.write(json.dumps(merged[k]) + '\n')
print('merged', len(merged), 'problems')
"
    conda deactivate
fi

# ============================================
# STEP 2: Build DPO pairs
# ============================================
echo "=== [7B] Step 2: Build DPO pairs ==="
if [ -f data/7B_dpo_pairs.jsonl ]; then
    echo "  skip: pairs file present"
else
    conda activate coderl
    python scripts/build_dpo_pairs.py --rollouts data/7B_rollouts_multi.jsonl \
        --out data/7B_dpo_pairs.jsonl --pairs-per-problem 10 --workers 24 \
        > "$LOGD/build_pairs.log" 2>&1
    conda deactivate
    tail -5 "$LOGD/build_pairs.log"
fi

# ============================================
# STEP 3: Train DPO
# ============================================
echo "=== [7B] Step 3: Train DPO ==="
if [ -d outputs/7B_dpo_merged ] && [ -f outputs/7B_dpo_merged/model.safetensors ]; then
    echo "  skip: 7B_dpo_merged present"
else
    conda activate coderl_sft
    T0=$(date +%s)
    python scripts/train_dpo.py --sft-model "$BASE_MERGED" \
        --data data/7B_dpo_pairs.jsonl --output-dir outputs/7B_dpo \
        --bs 2 --ga 8 \
        --no-wandb > "$LOGD/dpo_train.log" 2>&1
    echo "  DPO train: $(($(date +%s) - T0))s"
    python scripts/merge_lora.py --base "$BASE_MERGED" --adapter outputs/7B_dpo \
        --out outputs/7B_dpo_merged >> "$LOGD/dpo_train.log" 2>&1
    conda deactivate
fi

# ============================================
# STEP 4: Eval 7B DPO
# ============================================
echo "=== [7B] Step 4: Eval 7B DPO ==="
if [ -f "$RESD/dpo.summary.json" ]; then
    echo "  skip: eval done"
else
    conda activate coderl
    python scripts/eval_humaneval.py --model outputs/7B_dpo_merged \
        --out "$RESD/dpo_humaneval.jsonl" --no-wandb >> "$LOGD/dpo_eval.log" 2>&1
    python scripts/eval_mbpp.py --model outputs/7B_dpo_merged --split sanitized \
        --out "$RESD/dpo_mbpp.jsonl" --no-wandb >> "$LOGD/dpo_eval.log" 2>&1
    python scripts/eval_livecodebench.py --model outputs/7B_dpo_merged \
        --out "$RESD/dpo_lcb.jsonl" >> "$LOGD/dpo_eval.log" 2>&1
    $HOME/miniconda3/envs/coderl/bin/python -c "
import json, os
he = json.load(open('$RESD/dpo_humaneval.summary.json'))
mb = json.load(open('$RESD/dpo_mbpp.summary.json'))
lcb = json.load(open('$RESD/dpo_lcb.summary.json'))
s = {
    'variant': '7B_dpo',
    'humaneval_pass1': he['pass@1'], 'humaneval_n_pass': he['n_pass'],
    'mbpp_pass1': mb['pass@1'], 'mbpp_n_pass': mb['n_pass'],
    'lcb_pass1': lcb['pass@1'], 'lcb_n_pass': lcb['n_pass'],
    'lcb_post_cutoff_pass1': lcb['post_cutoff_subset']['pass@1'],
}
json.dump(s, open('$RESD/dpo.summary.json', 'w'), indent=2)
print(json.dumps(s, indent=2))
"
    conda deactivate
fi

# ============================================
# STEP 5: Prep RL prompts (reuse rollouts)
# ============================================
echo "=== [7B] Step 5: Prep RL prompts ==="
if [ ! -f data/7B_rl_prompts.jsonl ]; then
    conda activate coderl
    python scripts/prep_rl_prompts.py --rollouts data/7B_rollouts_multi.jsonl \
        --out data/7B_rl_prompts.jsonl --train-n 400 > "$LOGD/prep_rl.log" 2>&1
    conda deactivate
fi

# ============================================
# STEP 6: Train GRPO 7B
# ============================================
echo "=== [7B] Step 6: Train GRPO (200 steps) ==="
if [ -d outputs/7B_grpo_merged ] && [ -f outputs/7B_grpo_merged/model.safetensors ]; then
    echo "  skip: 7B_grpo_merged present"
else
    conda activate coderl_sft
    T0=$(date +%s)
    python scripts/train_grpo_trl.py --sft-model "$BASE_MERGED" \
        --data data/7B_rl_prompts.jsonl --output-dir outputs/7B_grpo \
        --max-steps 200 --lr 2e-6 --bs 2 --ga 4 --num-generations 4 \
        --no-wandb > "$LOGD/grpo_train.log" 2>&1
    echo "  GRPO train: $(($(date +%s) - T0))s"
    python scripts/merge_lora.py --base "$BASE_MERGED" --adapter outputs/7B_grpo \
        --out outputs/7B_grpo_merged >> "$LOGD/grpo_train.log" 2>&1
    conda deactivate
fi

# ============================================
# STEP 7: Eval 7B GRPO
# ============================================
echo "=== [7B] Step 7: Eval 7B GRPO ==="
if [ -f "$RESD/grpo.summary.json" ]; then
    echo "  skip: eval done"
else
    conda activate coderl
    python scripts/eval_humaneval.py --model outputs/7B_grpo_merged \
        --out "$RESD/grpo_humaneval.jsonl" --no-wandb >> "$LOGD/grpo_eval.log" 2>&1
    python scripts/eval_mbpp.py --model outputs/7B_grpo_merged --split sanitized \
        --out "$RESD/grpo_mbpp.jsonl" --no-wandb >> "$LOGD/grpo_eval.log" 2>&1
    python scripts/eval_livecodebench.py --model outputs/7B_grpo_merged \
        --out "$RESD/grpo_lcb.jsonl" >> "$LOGD/grpo_eval.log" 2>&1
    $HOME/miniconda3/envs/coderl/bin/python -c "
import json
he = json.load(open('$RESD/grpo_humaneval.summary.json'))
mb = json.load(open('$RESD/grpo_mbpp.summary.json'))
lcb = json.load(open('$RESD/grpo_lcb.summary.json'))
s = {
    'variant': '7B_grpo',
    'humaneval_pass1': he['pass@1'], 'humaneval_n_pass': he['n_pass'],
    'mbpp_pass1': mb['pass@1'], 'mbpp_n_pass': mb['n_pass'],
    'lcb_pass1': lcb['pass@1'], 'lcb_n_pass': lcb['n_pass'],
    'lcb_post_cutoff_pass1': lcb['post_cutoff_subset']['pass@1'],
}
json.dump(s, open('$RESD/grpo.summary.json', 'w'), indent=2)
print(json.dumps(s, indent=2))
"
    conda deactivate
fi

echo
echo "=== [7B] DPO+GRPO done ==="
echo "Done at $(date -u +%H:%M:%SZ)"
