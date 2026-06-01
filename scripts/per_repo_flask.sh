#!/usr/bin/env bash
# Per-repo Flask POC end-to-end:
#   1. clone pallets/flask
#   2. build FIM samples
#   3. train per-repo LoRA r=16 on dpo_v2_merged (1.5B)
#   4. merge -> outputs/per_repo/flask_merged
#   5. eval held-out FIM (exact-match, edit-similarity, symbol-recall)
#   6. eval HE+MBPP+LCB (regression check — must not drop more than 2pp)
#
# Total ETA: ~1h on 1.5B base (much faster than 7B)

set -eo pipefail
cd "$(dirname "$0")/.."

LOGD="logs/per_repo"
RESD="results/per_repo"
DATAD="data/per_repo/flask"
mkdir -p "$LOGD" "$RESD" "$DATAD"

source ~/miniconda3/etc/profile.d/conda.sh

# ============================================
# STEP 1: clone Flask
# ============================================
REPO_DIR=/tmp/flask
if [ -d "$REPO_DIR" ]; then
    echo "[1] Flask already cloned at $REPO_DIR"
else
    echo "[1] Cloning pallets/flask ..."
    git clone --depth 1 https://github.com/pallets/flask "$REPO_DIR" 2>&1 | tail -3
fi
echo "  files: $(find $REPO_DIR -name '*.py' | wc -l)"

# ============================================
# STEP 2: build FIM samples
# ============================================
if [ -f "$DATAD/train.jsonl" ] && [ -f "$DATAD/holdout.jsonl" ]; then
    echo "[2] FIM samples already built"
else
    echo "[2] Building FIM samples ..."
    conda activate coderl
    python scripts/per_repo/build_fim_samples.py \
        --repo-dir "$REPO_DIR" \
        --out "$DATAD/train.jsonl" \
        --out-eval "$DATAD/holdout.jsonl" \
        --n-per-file 8 --train-frac 0.9 \
        > "$LOGD/build_fim.log" 2>&1
    conda deactivate
    tail -5 "$LOGD/build_fim.log"
fi
echo "  train: $(wc -l < $DATAD/train.jsonl), holdout: $(wc -l < $DATAD/holdout.jsonl)"

# ============================================
# STEP 3: train per-repo LoRA on dpo_v2 (1.5B)
# ============================================
if [ -d outputs/per_repo/flask_merged ] && [ -f outputs/per_repo/flask_merged/model.safetensors ]; then
    echo "[3] Per-repo LoRA already trained"
else
    echo "[3] Training per-repo LoRA r=16 ..."
    conda activate coderl_sft
    T0=$(date +%s)
    python scripts/train_sft.py \
        --model outputs/dpo_v2_merged \
        --dataset "$DATAD/train.jsonl" \
        --output-dir outputs/per_repo/flask_lora \
        --lora-r 16 --lora-alpha 32 \
        --target-modules all \
        --bs 2 --ga 16 --epochs 2 \
        --max-seq-length 1536 \
        --lr 1e-5 \
        --save-merged --no-wandb \
        --run-name per-repo-flask \
        > "$LOGD/train.log" 2>&1
    echo "  train: $(($(date +%s) - T0))s"
    # The merged dir comes out as outputs/per_repo/flask_lora_merged via --save-merged
    if [ -d outputs/per_repo/flask_lora_merged ]; then
        mv outputs/per_repo/flask_lora_merged outputs/per_repo/flask_merged
    fi
    conda deactivate
fi

# ============================================
# STEP 4: eval held-out FIM
# ============================================
if [ -f "$RESD/flask_fim.summary.json" ]; then
    echo "[4] FIM eval already done"
else
    echo "[4] Eval held-out FIM (200 samples) ..."
    conda activate coderl
    # Eval the per-repo model
    python scripts/per_repo/eval_fim.py \
        --model outputs/per_repo/flask_merged \
        --data "$DATAD/holdout.jsonl" \
        --out "$RESD/flask_fim.jsonl" --limit 200 \
        > "$LOGD/eval_fim_adapter.log" 2>&1
    # Also eval the base (dpo_v2) for delta
    python scripts/per_repo/eval_fim.py \
        --model outputs/dpo_v2_merged \
        --data "$DATAD/holdout.jsonl" \
        --out "$RESD/flask_fim_base.jsonl" --limit 200 \
        > "$LOGD/eval_fim_base.log" 2>&1
    conda deactivate
fi

# ============================================
# STEP 5: HE+MBPP+LCB regression check
# ============================================
if [ -f "$RESD/flask_regression.json" ]; then
    echo "[5] Regression eval already done"
else
    echo "[5] Eval HE+MBPP+LCB for regression check ..."
    conda activate coderl
    python scripts/eval_humaneval.py --model outputs/per_repo/flask_merged \
        --out "$RESD/flask_humaneval.jsonl" --no-wandb \
        > "$LOGD/regress_he.log" 2>&1
    python scripts/eval_mbpp.py --model outputs/per_repo/flask_merged --split sanitized \
        --out "$RESD/flask_mbpp.jsonl" --no-wandb \
        > "$LOGD/regress_mbpp.log" 2>&1
    python scripts/eval_livecodebench.py --model outputs/per_repo/flask_merged \
        --out "$RESD/flask_lcb.jsonl" \
        > "$LOGD/regress_lcb.log" 2>&1
    $HOME/miniconda3/envs/coderl/bin/python -c "
import json
he = json.load(open('$RESD/flask_humaneval.summary.json'))
mb = json.load(open('$RESD/flask_mbpp.summary.json'))
lcb = json.load(open('$RESD/flask_lcb.summary.json'))
# base (dpo_v2) for delta
import os
base_he = 0.7439024390243902  # known from results/humaneval_dpo_v2.summary.json
base_mb = 0.669260700389105
base_lcb = 0.16666666666666666  # from results/lcb/dpo_v2.summary.json
out = {
    'model': 'outputs/per_repo/flask_merged',
    'humaneval': {'pass@1': he['pass@1'], 'vs_dpo_v2_base': (he['pass@1']-base_he)*100},
    'mbpp': {'pass@1': mb['pass@1'], 'vs_dpo_v2_base': (mb['pass@1']-base_mb)*100},
    'lcb': {'pass@1': lcb['pass@1'], 'vs_dpo_v2_base': (lcb['pass@1']-base_lcb)*100},
    'lcb_post_cutoff': lcb['post_cutoff_subset'],
}
json.dump(out, open('$RESD/flask_regression.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
"
    conda deactivate
fi

# ============================================
# Final report
# ============================================
echo
echo "=== Per-repo Flask POC results ==="
echo "[FIM eval -- adapter vs base]"
echo "  adapter:"; cat "$RESD/flask_fim.summary.json"
echo
echo "  base (dpo_v2):"; cat "$RESD/flask_fim_base.summary.json"
echo
echo "[General benchmark regression]"
cat "$RESD/flask_regression.json"
echo
echo "Done at $(date -u +%H:%M:%SZ)"
