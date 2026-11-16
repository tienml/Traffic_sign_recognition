"""Verify member4 evaluation: focused tests, original suite, notebook and artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_models import execute, preservation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/04_evaluation_tien.ipynb")
    parser.add_argument("--preservation-snapshot", type=Path,
                        default=ROOT / "data/model3_preservation_before.json")
    parser.add_argument("--relocations", type=Path)
    parser.add_argument("--allowed-change", action="append", default=[],
                        help="Explicit authorized refactor path; repeat for each changed file")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("Invalid verification tag")
    run = args.run_dir.resolve()
    if not run.is_relative_to(ROOT / "report/evaluation"):
        raise ValueError("Expected local evaluation run")
    logs = ROOT / "logs/evaluation" / f"verification-{args.tag}"
    logs.mkdir(parents=True, exist_ok=False)
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "contributor": "Đặng Quang Tiến",
              "member_number": 4, "steps": {}, "run_id": run.name}
    result["steps"]["focused"] = execute([sys.executable, "-B", "-m", "unittest",
        "src.tests.test_evaluation_saved", "src.tests.test_evaluation_runner", "-v"], logs / "focused_tests.log")
    result["steps"]["full_suite"] = execute([sys.executable, "-B", "-m", "unittest",
        "discover", "-s", "src/tests", "-v"], logs / "full_suite.log")
    text = (logs / "full_suite.log").read_text(encoding="utf-8")
    failures = re.findall(r"^FAIL: (\S+)", text, re.MULTILINE)
    errors = re.findall(r"^ERROR: (\S+)", text, re.MULTILINE)
    result["known_baseline_failure_only"] = (failures == ["test_pca_variance_curve_is_cumulative_and_capped"]
                                              and not errors)
    result["steps"]["pip_check"] = execute([sys.executable, "-B", "-m", "pip", "check"], logs / "pip_check.log")
    result["steps"]["pip_freeze"] = execute([sys.executable, "-B", "-m", "pip", "freeze"], logs / "pip_freeze.log")
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    notebook_path = args.notebook.resolve()
    if not notebook_path.is_relative_to(ROOT / "notebooks"):
        raise ValueError("Notebook must be under this repository's notebooks/")
    nb = nbformat.read(notebook_path, as_version=4)
    if nb.metadata.get("evaluation_run") != run.name:
        raise ValueError("Notebook run_id does not match the requested evaluation run")
    started = time.perf_counter()
    execution_error = None
    try:
        km = KernelManager(kernel_name="python3")
        km.kernel_spec.argv = [sys.executable, "-B", "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(nb, km=km, timeout=180, resources={"metadata": {"path": str(ROOT)}})
        client.execute()
    except Exception as error:
        execution_error = f"{type(error).__name__}: {error}"
    finally:
        executed_path = logs / notebook_path.name
        with executed_path.open("x", encoding="utf-8") as stream:
            nbformat.write(nb, stream)
        if "km" in locals() and km.has_kernel:
            km.shutdown_kernel(now=True)
    cell_errors = [o for c in nb.cells if c.cell_type == "code"
                   for o in c.get("outputs", []) if o.output_type == "error"]
    result["steps"]["notebook"] = {"exit_code": int(bool(execution_error or cell_errors)),
        "error": execution_error, "errors": len(cell_errors), "seconds": time.perf_counter()-started,
        "code_cells": sum(c.cell_type == "code" for c in nb.cells),
        "executed_copy": executed_path.relative_to(ROOT).as_posix(),
        "source_notebook_overwritten": False}
    with (logs / "notebook_execution.log").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result["steps"]["notebook"], ensure_ascii=False, indent=2)+"\n")
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                stream.write(f"\nCELL {i}, execution_count={cell.get('execution_count')}\n")
                for output in cell.get("outputs", []):
                    if output.output_type == "stream":
                        stream.write(output.get("text", ""))
                    elif output.output_type == "error":
                        stream.write("\n".join(output.get("traceback", []))+"\n")
    result["preservation"] = preservation(args.preservation_snapshot.resolve(),
        logs / "preservation.json", allowed_changes=args.allowed_change, relocations=args.relocations)
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["new_work_passed"] = (result["steps"]["focused"]["exit_code"] == 0
        and result["steps"]["notebook"]["exit_code"] == 0
        and result["steps"]["pip_check"]["exit_code"] == 0
        and result["preservation"]["protected_unchanged"]
        and (result["steps"]["full_suite"]["exit_code"] == 0 or result["known_baseline_failure_only"]))
    with (logs / "verification.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["new_work_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
