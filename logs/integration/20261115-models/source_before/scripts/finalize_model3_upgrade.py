"""Finish the new full run: audit, export, notebook, logs and append-only results.

No training, inference or official-test access is performed by this command.
An already-exported run is rejected; it never overwrites historical outputs.
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

import pandas as pd
from scripts.verify_model3_upgrade import execute, preservation


MARKER = "## Kết quả full-run nâng cấp đã thực thi"


def append(path, text, marker):
    current = path.read_text(encoding="utf-8")
    if marker in current:
        raise FileExistsError(f"Results section already exists: {path.name}")
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write("\n" + text.rstrip() + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--verification-tag", required=True)
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    if not run.is_relative_to(ROOT / "report/model3_upgrade"):
        raise ValueError("Run must be under report/model3_upgrade")
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    if selection["status"] != "completed_train_cv" or selection["official_test_evaluated"]:
        raise ValueError("Require completed full train-CV only")
    run_id = selection["run_id"]
    logs = ROOT / "logs/model3_upgrade" / run_id
    commands = [
        ("artifact_check.log", [sys.executable, "-B", "scripts/check_model3_artifacts.py", "--run-dir", str(run)]),
        ("export.log", [sys.executable, "-B", "scripts/summarize_model3_upgrade.py", "--run-dir", str(run)]),
        ("notebook_create.log", [sys.executable, "-B", "scripts/create_model3_notebook.py", "--run-dir", str(run)]),
    ]
    for name, command in commands:
        result = execute(command, logs / name)
        print(name, result, flush=True)
        if result["exit_code"] != 0:
            raise RuntimeError(f"Finalization step failed: {name}; see real log")
    verify = execute([sys.executable, "-B", "scripts/verify_model3_upgrade.py", "--tag",
                      args.verification_tag, "--notebook", "notebooks/03_model3_upgrade.ipynb"],
                     logs / "verification_console.log")
    verification_path = ROOT / "logs/model3_upgrade" / f"verification-{args.verification_tag}/verification.json"
    checked = json.loads(verification_path.read_text(encoding="utf-8"))
    if (checked["steps"]["focused"]["exit_code"] != 0
            or checked["steps"]["notebook"]["exit_code"] != 0
            or checked["steps"]["pip_check"]["exit_code"] != 0
            or not checked["preservation"]["protected_unchanged"]):
        raise RuntimeError("Focused tests, notebook, environment or preservation failed; inspect verification logs")
    full_log = ROOT / checked["steps"]["full_suite"]["log"]
    full_text = full_log.read_text(encoding="utf-8")
    failure_names = re.findall(r"^FAIL: (\S+)", full_text, re.MULTILINE)
    error_names = re.findall(r"^ERROR: (\S+)", full_text, re.MULTILINE)
    if checked["steps"]["full_suite"]["exit_code"] != 0 and (
            failure_names != ["test_pca_variance_curve_is_cumulative_and_capped"] or error_names):
        raise RuntimeError("New full-suite regression; do not label unknown failures as preexisting PCA")
    report_path = ROOT / "report/notes_model3_upgrade.md"
    table = ROOT / f"report/tables/model3_{run_id}_summary.csv"
    summary = pd.read_csv(table)
    deltas = pd.read_csv(ROOT / f"report/tables/model3_{run_id}_fold_deltas.csv")
    rows = [MARKER, "", "**Phần nâng cấp: Đặng Quang Tiến (người 4), có trợ lý AI hỗ trợ.** "
            "Phần KNN/DT gốc của Nguyễn Hoàng (người 3) giữ nguyên.", "",
            f"Run `{run_id}` đã hoàn tất trên **39.209 ảnh train / 1.307 track / 3 frozen folds**. "
            "Không dùng official test; không suy kết quả full từ smoke. Baseline và selected được tính trong cùng lượt.", "",
            "| Mô hình | Baseline CV macro-F1 ± SD | Selected CV macro-F1 ± SD | Chênh lệch (đpt) | Cấu hình selected |",
            "|---|---:|---:|---:|---|"]
    timing_notes = []
    for item in selection["algorithms"]:
        base, best = item["baseline"], item["selected"]
        rows.append(f"| {item['algorithm'].upper()} | {base['mean_f1_macro']:.6f} ± {base['std_f1_macro']:.6f} "
                    f"| {best['mean_f1_macro']:.6f} ± {best['std_f1_macro']:.6f} "
                    f"| {100*item['delta_cv_f1_macro']:+.4f} | `{json.dumps(best['params'], sort_keys=True)}` |")
        relevant = deltas.loc[deltas.algorithm.eq(item["algorithm"])]
        timing_notes.extend(["", f"- {item['algorithm'].upper()} delta macro-F1 từng fold: "
                     + ", ".join(f"{v:+.6f}" for v in relevant.delta_f1_macro)
                     + ". Đây không phải CI hoặc kiểm định ý nghĩa.",
                     f"- {item['algorithm'].upper()} CV wall toàn grid **{item['cv_wall_seconds']:.3f}s**, "
                     f"refit riêng **{item['refit_seconds']:.3f}s**, bundle **{item['model_bytes']/2**20:.3f}MiB**. "
                     "Không so tốc độ trực tiếp với môi trường lịch sử."])
    rows.extend(timing_notes)
    rows.extend(["", f"[Bảng tổng hợp](tables/model3_{run_id}_summary.csv), "
                  f"[toàn bộ candidates](tables/model3_{run_id}_candidates.csv), "
                  f"[điểm từng fold](tables/model3_{run_id}_fold_scores.csv), "
                  f"[delta từng fold](tables/model3_{run_id}_fold_deltas.csv).", "",
                  f"![Baseline và selected, SD không phải CI](figures/model3_{run_id}_cv_comparison.png)", "",
                  f"![KNN theo k](figures/model3_{run_id}_knn_k_curve.png)", "",
                  f"![Lỗi OOF theo kích thước](figures/model3_{run_id}_oof_size_errors.png)", "",
                  "Các hình có bản dark cùng basename `_dark.png`; bảng số đầy đủ nằm trong tables. "
                  "OOF class/group/error tables và quality thresholds có cùng tiền tố. "
                  "Brightness/contrast không được dùng để thay đổi winner sau search.", "",
                  f"Notebook thực thi: [03_model3_upgrade.ipynb](../notebooks/03_model3_upgrade.ipynb). "
                  f"Notebook execution errors={checked['steps']['notebook']['error_outputs']}; "
                  "Run All chỉ đọc và xác minh artifacts, không retrain/inference.", "",
                  f"Logs: [full console](../logs/model3_upgrade/{run_id}/console.log), "
                  f"[artifact audit](../logs/model3_upgrade/{run_id}/artifact_check.log), "
                  f"[export](../logs/model3_upgrade/{run_id}/export.log), "
                  f"[verification](../logs/model3_upgrade/verification-{args.verification_tag}/verification.json). "
                  "Focused tests và notebook pass. Full suite còn lỗi PCA người2 có sẵn; không tuyên bố toàn suite pass.", "",
                  "Kết luận chỉ thuộc phạm vi grid, biểu diễn HOG cố định và một seed/ba folds; "
                  "selected-CV có selection bias. Chưa chạy official test cho model mới và không thay điểm test cũ. "
                  "Không xóa ảnh khó, không cân bằng lại test, không coi mô tả OOF là chứng minh nhân quả."])
    append(report_path, "\n".join(rows), MARKER)
    for algorithm, name in (("knn", "notes_knn.md"), ("dt", "notes_dt.md")):
        item = next(i for i in selection["algorithms"] if i["algorithm"] == algorithm)
        text = (f"### Kết quả bổ sung `{run_id}`\n\n"
                "**Đặng Quang Tiến (người 4)** phụ trách phần nâng cấp, có trợ lý AI hỗ trợ. "
                f"Cùng HOG/folds, baseline CV macro-F1 **{item['baseline']['mean_f1_macro']:.6f}** → "
                f"selected **{item['selected']['mean_f1_macro']:.6f}** "
                f"(delta **{100*item['delta_cv_f1_macro']:+.4f} đpt**). "
                f"Selected `{json.dumps(item['selected']['params'], sort_keys=True)}`. "
                "Đây là kết quả train-CV, chưa là kết quả official test hoặc nested evaluation. "
                "Bảng/hình, timing, OOF và logs xem [báo cáo nâng cấp](notes_model3_upgrade.md#kết-quả-full-run-nâng-cấp-đã-thực-thi). "
                "Các kết quả lịch sử ở trên không bị thay thế.")
        append(ROOT / "report" / name, text, f"### Kết quả bổ sung `{run_id}`")
    snapshot = ROOT / "data/model3_preservation_before.json"
    preserved = preservation(snapshot, logs / "final_preservation.json")
    if not preserved["protected_unchanged"]:
        raise RuntimeError("Unexpected changes to protected files")
    with (logs / "completion.json").open("x", encoding="utf-8") as stream:
        json.dump({"finished_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                   "upgrade_contributor": "Đặng Quang Tiến", "member_number": 4,
                   "original_model_contributor": "Nguyễn Hoàng", "ai_assisted": True,
                   "artifact_check": "passed", "export": "passed", "notebook": "passed",
                   "focused_tests": "passed", "full_suite_exit": checked["steps"]["full_suite"]["exit_code"],
                   "known_baseline_pca_failure_preserved": True, "protected_unchanged": True,
                   "official_test_evaluated": False}, stream, indent=2)
    print(summary.to_string(index=False))
    print("Finalized actual results; historical artifacts and unrelated source preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
