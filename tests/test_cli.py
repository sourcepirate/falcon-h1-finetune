"""The CLI must stay importable when the training stack is broken."""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_cli_import_does_not_pull_torch():
    result = _run("import cyberft.cli, sys; assert 'torch' not in sys.modules, 'cli imported torch'")
    assert result.returncode == 0, result.stderr
