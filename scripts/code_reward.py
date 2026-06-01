"""
verl custom reward function for code RL.

Wired into verl GRPO via:
    custom_reward_function.path=/workspace/scripts/code_reward.py
    custom_reward_function.name=compute_score

Interface (verl 0.6+):
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

Where:
    data_source   — the data_source field from the parquet row
    solution_str  — model's generated response (string)
    ground_truth  — value from parquet row's reward_model.ground_truth
                    For us: JSON-encoded {test_harness, entry_point}
    extra_info    — dict from parquet row's extra_info (task_id, split, index)

Returns:
    1.0 if all unit tests pass within timeout, else 0.0.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Optional


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    matches = CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def _run_code(code: str, timeout: float = 10.0) -> bool:
    """Self-contained subprocess sandbox (mirrors sandbox_executor.py to avoid
    fragile relative-import issues when verl loads this file dynamically)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            timeout=timeout,
            capture_output=True,
            text=True,
            errors="replace",
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: Optional[dict] = None,
) -> float:
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except Exception:
        return 0.0

    test_harness = (gt or {}).get("test_harness", "") or ""
    entry = (gt or {}).get("entry_point", "") or ""

    code = _extract_code(solution_str)
    script = code + "\n\n" + test_harness
    if entry:
        script += f"\ncheck({entry})\n"
    else:
        script += "\n"

    return 1.0 if _run_code(script, timeout=10) else 0.0
