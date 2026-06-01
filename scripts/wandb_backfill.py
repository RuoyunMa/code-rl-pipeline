"""
Retroactively push our training curves to W&B from the saved trainer_state.json
files inside each checkpoint dir. Use this AFTER running `wandb login`.

Background — why backfill is even possible:

  HuggingFace `Trainer` (and trl SFT/DPO/GRPO trainers built on it) write a
  `trainer_state.json` next to every checkpoint. It contains a full `log_history`
  list: one entry per `logging_steps` step, with loss, lr, grad_norm, plus
  trainer-specific fields (rewards/accuracies, KL, entropy, etc.).

  So the curves still exist on disk — we just didn't ship them to W&B at
  train time because we were running `--no-wandb` (W&B wasn't logged in
  during the autonomous run).

Usage:

  1) Once: log in
       conda activate coderl
       wandb login         # paste API key

  2) Backfill everything (auto-detects all `outputs/*/trainer_state.json`):
       python scripts/wandb_backfill.py --root outputs --project code-rl-pipeline

  3) Or just one run:
       python scripts/wandb_backfill.py \\
           --run outputs/7B_dpo \\
           --project code-rl-pipeline \\
           --run-name 7B_dpo_v2

This is a write-only / no-GPU script — it just parses JSON and posts to W&B's
HTTP API. Safe to run on the Mac (no need to ssh to the 5090) if you rsync
trainer_state.json files over first; or run directly on the 5090.

Note: some columns (`rewards/accuracies`, `kl`, etc.) only exist for DPO/GRPO
runs. They'll be auto-skipped on plain SFT runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def find_runs(root: Path) -> Iterable[Path]:
    """Yield every directory under `root` containing a `trainer_state.json`."""
    for ts in root.rglob("trainer_state.json"):
        yield ts.parent


def push_run(run_dir: Path, project: str, run_name: str | None = None) -> None:
    import wandb

    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        print(f"  skip {run_dir} — no trainer_state.json")
        return

    state = json.loads(state_path.read_text())
    log_history = state.get("log_history") or []
    if not log_history:
        print(f"  skip {run_dir} — empty log_history")
        return

    # Try to extract config from training_args.json if present
    args_path = run_dir / "training_args.bin"  # transformers saves pickle
    cfg: dict = {}
    # Also pull adapter_config.json if it's a LoRA run, gives lora_r etc.
    adapter_cfg = run_dir / "adapter_config.json"
    if adapter_cfg.exists():
        try:
            cfg.update({"adapter": json.loads(adapter_cfg.read_text())})
        except Exception:
            pass

    name = run_name or run_dir.name
    wandb.init(
        project=project,
        name=name,
        config={**cfg, "backfilled_from": str(state_path), "run_dir": str(run_dir)},
        reinit=True,
    )

    n_steps = 0
    for entry in log_history:
        step = entry.get("step")
        # Each entry is a flat dict of metric → value. Strip step so we don't
        # double-log it under the metric name.
        metrics = {k: v for k, v in entry.items() if k != "step"}
        wandb.log(metrics, step=step)
        n_steps += 1

    print(f"  backfilled {n_steps} log steps from {run_dir} -> wandb run '{name}'")
    wandb.finish()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--root", type=Path, help="recurse this dir for any trainer_state.json")
    g.add_argument("--run", type=Path, help="single run dir containing trainer_state.json")
    ap.add_argument("--project", default="code-rl-pipeline")
    ap.add_argument("--run-name", default=None, help="override run name (single-run mode)")
    args = ap.parse_args()

    if args.run:
        push_run(args.run, args.project, args.run_name)
    else:
        runs = sorted(find_runs(args.root))
        print(f"Found {len(runs)} trainer_state.json under {args.root}")
        for r in runs:
            push_run(r, args.project, run_name=None)


if __name__ == "__main__":
    main()
