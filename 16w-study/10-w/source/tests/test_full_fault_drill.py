from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_real_child_process_exit_and_all_faults_recover(tmp_path):
    script = Path(__file__).resolve().parents[1] / "run_demo.py"
    output = tmp_path / "artifacts"
    completed = subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "fault-results.json").read_text(encoding="utf-8"))
    assert report["all_faults_recovered"] is True
    assert report["faults"]["process_exit"]["child_exit_code"] == 73
    assert report["faults"]["process_exit"]["recovered_state"] == "SUCCEEDED"
    assert report["release_rollback_verified"] is True
    assert report["trace_versions_verified"] is True

