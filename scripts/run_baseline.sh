#!/usr/bin/env bash
# D1 PM — baseline evaluation on Qwen2.5-Coder-1.5B-Instruct
#
# Run from repo root:
#     bash scripts/run_baseline.sh
#
# Override model:
#     MODEL=Qwen/Qwen2.5-Coder-7B-Instruct bash scripts/run_baseline.sh
#
# Skip W&B (offline / no key set):
#     NO_WANDB=1 bash scripts/run_baseline.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
RESULTS_DIR="${RESULTS_DIR:-results}"
WANDB_FLAG=""
if [[ "${NO_WANDB:-0}" == "1" ]]; then
    WANDB_FLAG="--no-wandb"
fi

cd "$(dirname "$0")/.."
mkdir -p "$RESULTS_DIR"

log() { echo -e "\033[1;36m==> $*\033[0m"; }

log "Sandbox executor smoke test"
python scripts/sandbox_executor.py

log "HumanEval baseline ($MODEL)"
python scripts/eval_humaneval.py \
    --model "$MODEL" \
    --out "$RESULTS_DIR/humaneval_baseline.jsonl" \
    --run-name "baseline-humaneval" \
    $WANDB_FLAG

log "MBPP baseline (sanitized) ($MODEL)"
python scripts/eval_mbpp.py \
    --model "$MODEL" \
    --split sanitized \
    --out "$RESULTS_DIR/mbpp_baseline.jsonl" \
    --run-name "baseline-mbpp" \
    $WANDB_FLAG

log "Baseline summary"
echo "--- HumanEval ---"
cat "$RESULTS_DIR/humaneval_baseline.summary.json"
echo
echo "--- MBPP ---"
cat "$RESULTS_DIR/mbpp_baseline.summary.json"
echo

# Append to results/baseline.md
{
    echo "# Baseline — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "**Model:** \`$MODEL\`"
    echo
    echo "## HumanEval"
    echo '```json'
    cat "$RESULTS_DIR/humaneval_baseline.summary.json"
    echo
    echo '```'
    echo
    echo "## MBPP (sanitized)"
    echo '```json'
    cat "$RESULTS_DIR/mbpp_baseline.summary.json"
    echo
    echo '```'
} > "$RESULTS_DIR/baseline.md"
log "Wrote $RESULTS_DIR/baseline.md"
# Note: json.dumps doesn't write trailing newline; the explicit `echo` lines above add one
# before the closing fence, which keeps the markdown valid.
