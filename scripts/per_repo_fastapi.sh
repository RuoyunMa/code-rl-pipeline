#!/usr/bin/env bash
# Per-repo POC with FastAPI (much bigger than Flask: 519 .py files, 32K lines).
# Should produce ~8000 FIM samples, ~500 train steps — real signal.
#
# Same flow as per_repo_flask.sh but bigger repo and more samples/file.

set -eo pipefail
cd "$(dirname "$0")/.."

LOGD="logs/per_repo_fastapi"
RESD="results/per_repo_fastapi"
DATAD="data/per_repo/fastapi"
mkdir -p "$LOGD" "$RESD" "$DATAD"

source ~/miniconda3/etc/profile.d/conda.sh

REPO_DIR=/tmp/repos/fastapi

# Step 1: ensure clone
if [ ! -d "$REPO_DIR" ]; then
    echo "[1] Cloning tiangolo/fastapi ..."
    mkdir -p /tmp/repos
    git clone --depth 1 https://github.com/tiangolo/fastapi "$REPO_DIR" 2>&1 | tail -3
else
    echo "[1] Using existing $REPO_DIR"
fi

# Step 2: build FIM
if [ -f "$DATAD/train.jsonl" ] && [ -f "$DATAD/holdout.jsonl" ]; then
    echo "[2] FIM samples already built"
else
    echo "[2] Building FIM samples (n=16 per file, 519 files expected) ..."
    conda activate coderl
    python scripts/per_repo/build_fim_samples.py \
        --repo-dir "$REPO_DIR" \
        --out "$DATAD/train.jsonl" \
        --out-eval "$DATAD/holdout.jsonl" \
        --n-per-file 16 --train-frac 0.9 \
        > "$LOGD/build_fim.log" 2>&1
    conda deactivate
    tail -5 "$LOGD/build_fim.log"
fi
echo "  train: $(wc -l < $DATAD/train.jsonl)  holdout: $(wc -l < $DATAD/holdout.jsonl)"

# Step 3: train per-repo LoRA
if [ -d outputs/per_repo/fastapi_merged ] && [ -f outputs/per_repo/fastapi_merged/model.safetensors ]; then
    echo "[3] Already trained"
else
    echo "[3] Training per-repo LoRA r=16 on dpo_v2_merged ..."
    conda activate coderl_sft
    T0=$(date +%s)
    python scripts/train_sft.py \
        --model outputs/dpo_v2_merged \
        --dataset "$DATAD/train.jsonl" \
        --output-dir outputs/per_repo/fastapi_lora \
        --lora-r 16 --lora-alpha 32 \
        --target-modules all \
        --bs 2 --ga 16 --epochs 2 \
        --max-seq-length 1536 \
        --lr 1e-5 \
        --save-merged --no-wandb \
        --run-name per-repo-fastapi \
        > "$LOGD/train.log" 2>&1
    echo "  train: $(($(date +%s) - T0))s"
    if [ -d outputs/per_repo/fastapi_lora_merged ]; then
        mv outputs/per_repo/fastapi_lora_merged outputs/per_repo/fastapi_merged
    fi
    conda deactivate
fi

# Step 4: eval FIM (adapter + base)
if [ -f "$RESD/fastapi_fim.summary.json" ]; then
    echo "[4] FIM eval done"
else
    echo "[4] Eval FIM (limit 300) ..."
    conda activate coderl
    python scripts/per_repo/eval_fim.py \
        --model outputs/per_repo/fastapi_merged \
        --data "$DATAD/holdout.jsonl" \
        --out "$RESD/fastapi_fim.jsonl" --limit 300 \
        > "$LOGD/eval_fim_adapter.log" 2>&1
    python scripts/per_repo/eval_fim.py \
        --model outputs/dpo_v2_merged \
        --data "$DATAD/holdout.jsonl" \
        --out "$RESD/fastapi_fim_base.jsonl" --limit 300 \
        > "$LOGD/eval_fim_base.log" 2>&1
    conda deactivate
fi

# Step 5: regression eval
if [ -f "$RESD/fastapi_regression.json" ]; then
    echo "[5] regression eval done"
else
    echo "[5] HE+MBPP+LCB regression check ..."
    conda activate coderl
    python scripts/eval_humaneval.py --model outputs/per_repo/fastapi_merged \
        --out "$RESD/fastapi_humaneval.jsonl" --no-wandb > "$LOGD/regress_he.log" 2>&1
    python scripts/eval_mbpp.py --model outputs/per_repo/fastapi_merged --split sanitized \
        --out "$RESD/fastapi_mbpp.jsonl" --no-wandb > "$LOGD/regress_mbpp.log" 2>&1
    python scripts/eval_livecodebench.py --model outputs/per_repo/fastapi_merged \
        --out "$RESD/fastapi_lcb.jsonl" > "$LOGD/regress_lcb.log" 2>&1
    $HOME/miniconda3/envs/coderl/bin/python -c "
import json
he = json.load(open('$RESD/fastapi_humaneval.summary.json'))
mb = json.load(open('$RESD/fastapi_mbpp.summary.json'))
lcb = json.load(open('$RESD/fastapi_lcb.summary.json'))
base_he, base_mb, base_lcb = 0.7439024390243902, 0.669260700389105, 0.16666666666666666
out = {
    'model': 'outputs/per_repo/fastapi_merged',
    'humaneval': {'pass@1': he['pass@1'], 'vs_dpo_v2_base_pp': (he['pass@1']-base_he)*100},
    'mbpp': {'pass@1': mb['pass@1'], 'vs_dpo_v2_base_pp': (mb['pass@1']-base_mb)*100},
    'lcb': {'pass@1': lcb['pass@1'], 'vs_dpo_v2_base_pp': (lcb['pass@1']-base_lcb)*100},
    'lcb_post_cutoff_pass1': lcb['post_cutoff_subset']['pass@1'],
}
json.dump(out, open('$RESD/fastapi_regression.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
"
    conda deactivate
fi

# Report
echo
echo "=== Per-repo FastAPI POC results ==="
echo "[FIM eval]"
echo "  adapter:"; cat "$RESD/fastapi_fim.summary.json"; echo
echo "  base:";    cat "$RESD/fastapi_fim_base.summary.json"; echo
echo "[Regression]"
cat "$RESD/fastapi_regression.json"
echo
echo "Done at $(date -u +%H:%M:%SZ)"
