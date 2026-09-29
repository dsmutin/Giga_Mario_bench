"""Run scripts written by ``score prepare``, preferably through Nextflow."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from giga_mario_bench.giga_mario import repo_root


def execute(
    exec_root: Path | None = None,
    *,
    use_nextflow: bool = True,
) -> int:
    """Execute every generated cell script.

    Nextflow is the default orchestrator. Each script checks which artifacts
    already exist. Without Nextflow, the same scripts run under this interpreter.
    """
    root = repo_root()
    scripts_root = Path(exec_root) if exec_root else root / "src" / "giga_mario_bench" / "exec"
    scripts = sorted(
        path
        for path in scripts_root.rglob("*.py")
        if path.name != "__init__.py"
    )
    if not scripts:
        raise FileNotFoundError(f"no exec scripts under {scripts_root}; run score prepare")
    workflow = root / "workflows" / "score.nf"
    if use_nextflow and shutil.which("nextflow") and workflow.is_file():
        command = [
            "nextflow",
            "run",
            str(workflow),
            "--exec_root",
            str(scripts_root),
            "--python",
            sys.executable,
        ]
        print("score exec via nextflow: " + " ".join(command))
        completed = subprocess.run(command, cwd=root, check=False)
        return int(completed.returncode)
    print("score exec via python")
    for script in scripts:
        completed = subprocess.run([sys.executable, str(script)], check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0
