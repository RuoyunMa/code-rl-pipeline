#!/usr/bin/env bash
# Sync code-rl-pipeline from Mac → 5090 via SSH (over Tailscale or LAN).
#
# Usage:
#     REMOTE=user@5090.your-tailnet.ts.net bash scripts/deploy.sh
#
# Or persist in your shell rc:
#     export REMOTE=user@5090.your-tailnet.ts.net
#     bash scripts/deploy.sh
#
# Re-run any time to push latest scripts. Excludes outputs/data/results so
# training artifacts on the 5090 stay put.

set -euo pipefail

REMOTE="${REMOTE:?ERROR: set REMOTE=user@host (e.g. REMOTE=ruoyun@5090.ts.net)}"
REMOTE_DIR="${REMOTE_DIR:-~/workspace/code-rl-pipeline}"
SRC_DIR="$(cd "$(dirname "$0")/.."; pwd)"

echo "==> Syncing $SRC_DIR/ -> $REMOTE:$REMOTE_DIR/"

# Make sure the remote dir exists
ssh "$REMOTE" "mkdir -p '$REMOTE_DIR'"

rsync -avh --delete \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '.git' \
    --exclude 'outputs/' \
    --exclude 'data/' \
    --exclude 'results/' \
    --exclude 'wandb/' \
    --exclude 'logs/' \
    "$SRC_DIR/" \
    "$REMOTE:$REMOTE_DIR/"

echo
echo "Done. Next:"
echo "    ssh $REMOTE"
echo "    cd $REMOTE_DIR"
echo "    bash scripts/setup_env.sh        # D1 AM — ~15-30 min"
