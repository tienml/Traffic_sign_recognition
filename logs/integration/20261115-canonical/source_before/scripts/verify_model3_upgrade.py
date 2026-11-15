"""Record actual test/environment/notebook execution and preservation evidence.

Only new logs under logs/model3_upgrade/verification-<tag> are written. The
baseline suite is not silently fixed if a historical person-2 test fails.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_TESTS = ["src.tests.test_preprocessing", "src.tests.test_features",
                  "src.tests.test_models", "src.tests.test_knn_summary",
                  "src.tests.test_decision_tree"]


def execute(command, path):
    path = Path(path).resolve()
    started = time.perf_counter()
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"}
    with path.open("x", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, check=False)
    return {"exit_code": process.returncode, "seconds": time.perf_counter() - started,
            "log": str(path.relative_to(ROOT)).replace(chr(92), "/")}


def preservation(snapshot, output):
    before = json.loads(snapshot.read_text(encoding="utf-8"))
    changed = []
    for name, expected in before.items():
        path = ROOT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            changed.append({"path": name, "before_sha256": expected, "after_sha256": actual})
    permitted = {"report/notes_knn.md", "report/notes_dt.md", "AI_USAGE.md"}
    unexpected = [item for item in changed if item["path"] not in permitted]
    # Documentation is append-only from HEAD, except a user-edited document whose
    # current state was explicitly captured after the harness change notice.
    # Never restore that document from HEAD just to make this audit green.
    user_baseline_path = ROOT / "data/model3_user_document_baseline_20261115.json"
    user_baseline = (json.loads(user_baseline_path.read_text(encoding="utf-8"))
                     if user_baseline_path.is_file() else {})
    append_only = {}
    for item in changed:
        path = item["path"]
        if path in permitted:
            if user_baseline.get("path") == path and user_baseline.get("user_edit_notice") is True:
                old = user_baseline["text_prefix"].encode("utf-8")
            else:
                old = subprocess.check_output(["git", "show", "HEAD:" + path], cwd=ROOT)
            # Account for checkout's CRLF conversion, without changing old text.
            current = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
            append_only[path] = current.startswith(old.replace(b"\r\n", b"\n"))
    result = {"tracked_before": len(before), "changed": changed,
              "unexpected_changes": unexpected, "documentation_append_only": append_only,
              "user_document_baseline": (user_baseline.get("path") or None),
              "protected_unchanged": not unexpected and all(append_only.values())}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--notebook", type=Path)
    args = parser.parse_args(argv)
    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.tag):
        parser.error("Invalid tag")
    logs = ROOT / "logs/model3_upgrade" / f"verification-{args.tag}"
    logs.mkdir(parents=True, exist_ok=False)
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
              "baseline": args.baseline, "steps": {}}
    for name, command in (("pip_check", [sys.executable, "-B", "-m", "pip", "check"]),
                          ("pip_freeze", [sys.executable, "-B", "-m", "pip", "freeze"])):
        result["steps"][name] = execute(command, logs / f"{name}.log")
    if args.baseline:
        test_command = [sys.executable, "-B", "-m", "unittest", *ORIGINAL_TESTS, "-v"]
    else:
        result["steps"]["focused"] = execute([sys.executable, "-B", "-m", "unittest",
            "src.tests.test_model_upgrade", "src.tests.test_model_upgrade_runner", "-v"], logs / "focused_tests.log")
        test_command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "src/tests", "-v"]
    result["steps"]["full_suite"] = execute(test_command, logs / "full_suite.log")
    if args.notebook:
        # This new notebook is artifact-driven and defaults to no training/inference.
        import nbformat
        from nbclient import NotebookClient
        path = args.notebook.resolve()
        if not path.is_relative_to(ROOT / "notebooks"):
            raise ValueError("Notebook must be under this repository's notebooks/")
        notebook = nbformat.read(path, as_version=4)
        started = time.perf_counter()
        notebook_error = None
        try:
            client = NotebookClient(notebook, timeout=180, resources={"metadata": {"path": str(ROOT)}},
                                    kernel_name="python3")
            # Explicit kernel command ensures the prepared venv, not global Python.
            from jupyter_client import KernelManager
            km = KernelManager(kernel_name="python3", transport="ipc") if os.name != "nt" else KernelManager(kernel_name="python3")
            km.kernel_spec.argv = [sys.executable, "-B", "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            client.km = km
            client.execute()
        except Exception as error:
            notebook_error = f"{type(error).__name__}: {error}"
        finally:
            nbformat.write(notebook, path)
            if "km" in locals() and km.has_kernel:
                km.shutdown_kernel(now=True)
        errors = [o for cell in notebook.cells if cell.cell_type == "code"
                  for o in cell.get("outputs", []) if o.output_type == "error"]
        result["steps"]["notebook"] = {"path": str(path.relative_to(ROOT)).replace(chr(92), "/"),
            "seconds": time.perf_counter() - started, "error": notebook_error,
            "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
            "error_outputs": len(errors), "exit_code": int(bool(notebook_error or errors))}
        with (logs / "notebook_execution.log").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result["steps"]["notebook"], ensure_ascii=False, indent=2) + "\n")
            for number, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                stream.write(f"\nCELL {number}, execution_count={cell.get('execution_count')}\n")
                for output in cell.get("outputs", []):
                    if output.output_type == "stream":
                        stream.write(output.get("text", ""))
                    elif output.output_type == "error":
                        stream.write("\n".join(output.get("traceback", [])) + "\n")
    snapshot = ROOT / "data/model3_preservation_before.json"
    if snapshot.is_file():
        result["preservation"] = preservation(snapshot, logs / "preservation.json")
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["all_steps_passed"] = all(s["exit_code"] == 0 for s in result["steps"].values())
    with (logs / "verification.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_steps_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
