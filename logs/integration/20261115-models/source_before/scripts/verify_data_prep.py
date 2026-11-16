"""Kiểm tra phần dữ liệu, chạy notebook và lưu bằng chứng tái lập cục bộ."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
NOTEBOOKS = ("01_eda.ipynb", "02_preprocessing_ablation.ipynb")


def run_command(arguments: list[str], filename: str) -> dict:
    """Chạy bằng interpreter hiện tại, lưu stdout/stderr và giữ mã lỗi thật."""
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        check=False,
    )
    output = result.stdout + result.stderr
    # Chỉ lưu đường dẫn tương đối trong bằng chứng đưa vào Git.
    for root_spelling in (str(ROOT), str(ROOT).replace("\\", "/")):
        output = output.replace(root_spelling, "<PROJECT>")
    (LOGS / filename).write_text(output, encoding="utf-8")
    print(f"{filename}: exit={result.returncode}", flush=True)
    if result.returncode:
        print(output)
        raise RuntimeError(f"Lệnh thất bại, xem logs/{filename}")
    return {
        "log": filename,
        "exit_code": result.returncode,
        "seconds": time.perf_counter() - started,
    }


def execute_notebook(name: str) -> dict:
    """Chạy kernel mới đúng môi trường, giữ output kể cả khi ô code lỗi."""
    path = ROOT / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager()
    # Chỉ rõ executable để không vô tình chạy kernel hệ thống khác venv.
    manager._kernel_spec = KernelSpec(
        argv=[
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        display_name="Python 3",
        language="python",
        name="python3",
    )
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=900,
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT / "notebooks")}},
    )
    started = time.perf_counter()
    print(f"Đang chạy {name} từ đầu...", flush=True)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
        manager.cleanup_resources()
        lines = []
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            lines.append(
                f"cell={cell.id}; execution_count={cell.execution_count}"
            )
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    lines.append(output.text)
                elif output.output_type == "error":
                    lines.append(f"ERROR: {output.ename}: {output.evalue}")
        text = "\n".join(lines).replace(str(ROOT), "<PROJECT>")
        (LOGS / f"{path.stem}.log").write_text(text, encoding="utf-8")
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code_cells] == list(
        range(1, len(code_cells) + 1)
    )
    assert not any(
        output.output_type == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    seconds = time.perf_counter() - started
    print(
        f"{name}: {len(code_cells)} ô code, 0 lỗi, {seconds:.2f}s", flush=True
    )
    return {
        "notebook": name,
        "code_cells": len(code_cells),
        "errors": 0,
        "seconds": seconds,
    }


def smoke_main() -> dict:
    """Chạy main.py thật, dừng khi tới features, không chạy GridSearchCV."""
    started = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, "-u", "main.py"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    def stop() -> None:
        """Chỉ dừng cây tiến trình vừa tạo, không chạm Python khác."""
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                process.terminate()

    timer = threading.Timer(180, stop)
    timer.start()
    reached_features = False
    lines = []
    try:
        for line in process.stdout:
            lines.append(line)
            if "[Người 2]" in line:
                reached_features = True
                stop()
                break
        process.wait(timeout=30)
    finally:
        timer.cancel()
        stop()
        process.stdout.close()
        (LOGS / "main_smoke.log").write_text(
            "".join(lines).replace(str(ROOT), "<PROJECT>")
            + f"\nreached_features={reached_features}; exit={process.returncode}\n"
            + "Chủ động dừng trước huấn luyện; CHƯA chạy GridSearchCV/đánh giá đầy đủ.\n",
            encoding="utf-8",
        )
    if not reached_features:
        raise RuntimeError(
            "main.py chưa tới build_features; xem main_smoke.log"
        )
    print(
        "main.py: đã tới build_features, chủ động dừng trước GridSearchCV.",
        flush=True,
    )
    return {
        "reached_features": True,
        "intentionally_stopped": True,
        "process_exit_code": process.returncode,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    """Chạy các bước tuần tự để tránh cạnh tranh RAM và sai lệch thời gian."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", action="append", choices=NOTEBOOKS)
    parser.add_argument(
        "--skip-cli",
        action="store_true",
        help="Chỉ chạy notebook khi đang phát triển, không phải kiểm tra đầy đủ.",
    )
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "LOKY_MAX_CPU_COUNT": "2",
            "MPLBACKEND": "module://matplotlib_inline.backend_inline",
        }
    )
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "cli_skipped": args.skip_cli,
        "status": "running",
        "steps": {},
    }
    filename = "notebook_run.json" if args.skip_cli else "verification.json"
    try:
        if not args.skip_cli:
            for label, arguments, logfile in (
                ("freeze", ["-m", "pip", "freeze"], "pip_freeze.txt"),
                ("pip_check", ["-m", "pip", "check"], "pip_check.log"),
                (
                    "tests",
                    ["-m", "unittest", "discover", "-s", "src/tests", "-v"],
                    "unittest.log",
                ),
                (
                    "preprocessing",
                    ["-m", "src.preprocessing"],
                    "preprocessing_cli.log",
                ),
            ):
                report["steps"][label] = run_command(arguments, logfile)
        for name in args.notebook or NOTEBOOKS:
            report["steps"][name] = execute_notebook(name)
        if not args.skip_cli:
            report["steps"]["main_smoke"] = smoke_main()
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (LOGS / filename).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
