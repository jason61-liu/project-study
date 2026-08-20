from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WEEK15 = ROOT / "15-w"
WEEK16 = ROOT / "16-w"


def run(command: list[str], cwd: Path, pythonpath: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pythonpath)
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], WEEK15, WEEK15 / "src")
    run([sys.executable, "-m", "deep_research_agent.eval_gate", "--baseline", "evals/baseline.json"], WEEK15, WEEK15 / "src")
    combined_path = os.pathsep.join((str(WEEK16 / "src"), str(WEEK15 / "src")))
    env = dict(os.environ)
    env["PYTHONPATH"] = combined_path
    for module, output in (
        ("portfolio.strengthening", "results/strengthening.json"),
        ("portfolio.verify", "results/security-fault.json"),
        ("portfolio.feedback", "results/feedback-loop.json"),
        ("portfolio.release", "results/release-drill.json"),
    ):
        print("+", module)
        subprocess.run([sys.executable, "-m", module, "--output", output], cwd=WEEK16, env=env, check=True)
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=WEEK16,
        env=env,
        check=True,
    )
    from_path = WEEK16 / "src"
    sys.path.insert(0, str(from_path))
    from portfolio.controls import scan_public_artifact

    leaks = []
    for base in (WEEK16 / "docs", WEEK16 / "results"):
        for path in base.rglob("*"):
            if path.is_file() and scan_public_artifact(path.read_text(errors="ignore")):
                leaks.append(str(path.relative_to(WEEK16)))
    if leaks:
        raise SystemExit(f"public artifact scan failed: {leaks}")
    print("CI EVAL GATE PASSED; regression rollback and public artifact scan verified")


if __name__ == "__main__":
    main()

