"""
Code execution sandbox.

D1 baseline version: subprocess + timeout + stdout/stderr capture.
D5 will harden this with:
  - resource limits (RLIMIT_CPU, RLIMIT_AS, RLIMIT_FSIZE)
  - process group kill (so timeout reaps spawned children)
  - import filter / network block

For now, this is "good enough" to evaluate model-generated HumanEval / MBPP
solutions on a trusted machine. DO NOT expose this to untrusted code outside.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Dict


def run_code(code: str, timeout: float = 10.0) -> Dict[str, object]:
    """
    Execute `code` as a Python script in a subprocess.

    Returns:
        dict with keys: passed (bool), returncode, stdout, stderr, error.
        `passed` == True iff the subprocess exited with code 0 within the timeout.
    """
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
            errors="replace",  # don't crash on non-UTF-8 child output
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": None,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "passed": False,
            "returncode": None,
            "stdout": (e.stdout or "") if isinstance(e.stdout, str) else "",
            "stderr": (e.stderr or "") if isinstance(e.stderr, str) else "",
            "error": "timeout",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "passed": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _smoke_tests() -> None:
    r = run_code("print('hello')\nassert 1 + 1 == 2\n")
    assert r["passed"], r
    r = run_code("assert 1 == 2\n")
    assert not r["passed"], r
    r = run_code("import time\ntime.sleep(20)\n", timeout=2)
    assert r["error"] == "timeout", r
    r = run_code("raise ValueError('x')\n")
    assert not r["passed"] and "ValueError" in (r["stderr"] or ""), r
    print("sandbox_executor smoke tests passed.")


if __name__ == "__main__":
    _smoke_tests()
