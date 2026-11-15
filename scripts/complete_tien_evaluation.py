"""One-shot saved-prediction evaluation after completed KNN/DT train-CV runs.

No training or new inference. Existing outputs and AI_USAGE.md are not overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.model_artifacts import load_model_runs
from scripts.verify_models import execute


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knn-run", type=Path, required=True)
    parser.add_argument("--dt-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/04_evaluation_tien.ipynb")
    parser.add_argument("--preservation-snapshot", type=Path, required=True)
    parser.add_argument("--allowed-change", action="append", default=[])
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("Invalid run-id")
    runs, selection, _ = load_model_runs(args.knn_run, args.dt_run)
    if selection["status"] != "completed_train_cv":
        raise ValueError("Finish full train-CV first")
    for algorithm, model_run in runs.items():
        if not (ROOT / "logs" / algorithm / model_run.name / "completion.json").is_file():
            raise ValueError("Finish and verify both KNN/DT runs before evaluation")
    orchestration = ROOT / "logs/evaluation" / f"orchestration-{args.run_id}"
    orchestration.mkdir(parents=True, exist_ok=False)
    run = ROOT / "report/evaluation" / args.run_id
    tag = f"{args.run_id}-final"
    verify = [sys.executable, "-B", "scripts/verify_evaluation.py", "--run-dir", str(run),
              "--tag", tag, "--notebook", str(args.notebook),
              "--preservation-snapshot", str(args.preservation_snapshot)]
    for name in args.allowed_change:
        verify += ["--allowed-change", name]
    commands = [
        ("evaluation.log", [sys.executable, "-B", "scripts/evaluate_saved_models.py", "--dataset-dir", "data",
                            "--knn-run", str(runs["knn"]), "--dt-run", str(runs["dt"]), "--run-id", args.run_id]),
        ("notebook_create.log", [sys.executable, "-B", "scripts/create_evaluation_notebook.py", "--run-dir", str(run),
                                 "--output", str(args.notebook)]),
        ("verification.log", verify),
        ("report.log", [sys.executable, "-B", "scripts/write_evaluation_report.py", "--run-dir", str(run),
                         "--verification-dir", str(ROOT / "logs/evaluation" / f"verification-{tag}")]),
    ]
    for name, command in commands:
        if execute(command, orchestration / name)["exit_code"]:
            raise RuntimeError(f"Step failed; inspect {orchestration / name}")
    with (orchestration / "completion.json").open("x", encoding="utf-8") as stream:
        json.dump({"run_id": args.run_id, "contributor": "Đặng Quang Tiến", "member_number": 4,
                   "finished_utc": datetime.now(timezone.utc).isoformat(), "evaluation": "passed",
                   "notebook": "executed", "report": "written", "new_official_test_inference": False},
                  stream, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
