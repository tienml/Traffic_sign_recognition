"""Audit/export a new KNN/DT train-CV run and append results to their reports.

One-shot command: existing export/notebook/report markers are rejected. It never
trains, infers, removes claims, rewrites historical hashes, or edits AI_USAGE.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
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
    parser.add_argument("--verification-tag", required=True)
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/03_models.ipynb")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--preservation-snapshot", type=Path, required=True)
    parser.add_argument("--allowed-change", action="append", default=[])
    args = parser.parse_args(argv)
    runs, selection, _ = load_model_runs(args.knn_run, args.dt_run)
    if selection["status"] != "completed_train_cv":
        raise ValueError("Require full train-CV, not smoke")
    pair = ["--knn-run", str(runs["knn"]), "--dt-run", str(runs["dt"])]
    logs = ROOT / "logs/knn" / runs["knn"].name
    logs.mkdir(parents=True, exist_ok=True)
    markers = {a: f"### Kết quả CV `{runs[a].name}`" for a in runs}
    for algorithm, marker in markers.items():
        if marker in (ROOT / f"report/notes_{algorithm}.md").read_text(encoding="utf-8"):
            raise FileExistsError("Report section already exists")
    audit = [sys.executable, "-B", "scripts/check_model_artifacts.py", *pair]
    if args.source_root:
        audit += ["--source-root", str(args.source_root)]
    verify = [sys.executable, "-B", "scripts/verify_models.py", "--tag", args.verification_tag,
              "--notebook", str(args.notebook), "--preservation-snapshot", str(args.preservation_snapshot)]
    for name in args.allowed_change:
        verify += ["--allowed-change", name]
    commands = [("artifact_check.log", audit),
                ("export.log", [sys.executable, "-B", "scripts/summarize_models.py", *pair]),
                ("notebook_create.log", [sys.executable, "-B", "scripts/create_model_notebook.py", *pair,
                                         "--output", str(args.notebook)]),
                ("verification_console.log", verify)]
    for name, command in commands:
        if execute(command, logs / name)["exit_code"]:
            raise RuntimeError(f"Step failed; inspect {logs / name}")
    for item in selection["algorithms"]:
        algorithm = item["algorithm"]
        text = (f"\n{markers[algorithm]}\n\n"
                "Đặng Quang Tiến thực hiện mở rộng tìm tham số trên phần mô hình người3. "
                f"Baseline F1={item['baseline']['mean_f1_macro']:.6f}; "
                f"selected F1={item['selected']['mean_f1_macro']:.6f}. "
                f"Params: `{json.dumps(item['selected']['params'], sort_keys=True)}`. "
                "Chỉ train-CV; cùng-fold OOF có selection bias, chưa có điểm official test mới.\n")
        with (ROOT / f"report/notes_{algorithm}.md").open("a", encoding="utf-8") as stream:
            stream.write(text)
        target = ROOT / "logs" / algorithm / runs[algorithm].name
        target.mkdir(parents=True, exist_ok=True)
        with (target / "completion.json").open("x", encoding="utf-8") as stream:
            json.dump({"finished_utc": datetime.now(timezone.utc).isoformat(),
                       "algorithm": algorithm, "artifact_check": "passed", "export": "passed",
                       "notebook": "passed", "verification": "passed",
                       "official_test_evaluated": False}, stream, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
