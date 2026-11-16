"""Create notebook03 from saved KNN/DT runs; never train or infer automatically."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.model_artifacts import load_model_runs


def create(knn_run, destination, *, dt_run):
    runs, selection, _ = load_model_runs(knn_run, dt_run)
    destination = Path(destination).resolve()
    for algorithm, path in runs.items():
        if not path.is_relative_to(ROOT / "report" / algorithm):
            raise ValueError("Expected runs inside report/knn and report/dt")
    if not destination.is_relative_to(ROOT / "notebooks") or destination.exists():
        raise FileExistsError("Notebook output must be new and inside notebooks/")
    if selection["status"] != "completed_train_cv":
        raise ValueError("Notebook requires completed full train-CV runs")
    run_paths = {a: p.relative_to(ROOT).as_posix() for a, p in runs.items()}
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        md("# Huấn luyện và tối ưu KNN / Decision Tree\n\n"
           "**Phần mô hình người3:** Nguyễn Hoàng triển khai KNN/DT gốc; "
           "Đặng Quang Tiến (người4) thực hiện phần mở rộng k và regularization DT. "
           "Có AI hỗ trợ, không thay trách nhiệm tự xác minh của thành viên.\n\n"
           "Notebook đọc kết quả CV/OOF đã lưu trong **report/knn** và **report/dt**, "
           "không train, predict, nạp model hoặc chạy official test khi Run All. "
           "Điểm test lịch sử không phải điểm test của cấu hình CV mới."),
        md("## 1. Dữ liệu và giao thức\n\n"
           "Giữ nguyên RGB → ROI → resize32 AREA → /255 → HOG324 của người1/2. "
           "StandardScaler fit fold-train; 39.209 ảnh, 1.307 track, frozen3fold seed42. "
           "Các kết quả/claims lịch sử không bị thay thế.\n\n"
           "[Báo cáo KNN](../report/notes_knn.md) · [Báo cáo DT](../report/notes_dt.md) · "
           "[EDA](../report/notes_eda_preprocessing.md)"),
        code("from pathlib import Path\nimport json, sys\nimport pandas as pd\nimport numpy as np\n"
             "from IPython.display import display, Image\n"
             "ROOT=Path.cwd().resolve()\nif not (ROOT/'src').is_dir(): ROOT=ROOT.parent\n"
             "if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))\n"
             "from scripts.model_artifacts import load_model_runs, resolve_evidence\n"
             f"RUNS={{a:ROOT/p for a,p in {run_paths!r}.items()}}\n"
             "RUNS,selection,config=load_model_runs(RUNS['knn'],RUNS['dt'])\n"
             "assert selection['status']=='completed_train_cv'\n"
             "assert selection['official_test_read'] is False\n"
             "assert selection['official_test_evaluated'] is False\n"
             "PREFIX='models_'+RUNS['knn'].name\nTABLES=ROOT/'report/tables'\nFIGURES=ROOT/'report/figures'\n"
             "print('Saved train-CV runs:',RUNS)\nprint('No automatic training or model inference.')"),
        md("## 2. Frozen folds và grid\n\n"
           "KNN28 cấu hình: k={3,5,7,11,15,21,31}, uniform/distance, Euclidean/Manhattan. "
           "DT11 cấu hình: giữ entropy/depth20/split5; thử leaf/class_weight/pruning. "
           "Baseline đứng đầu, chọn mean macro-F1, hòa lấy cấu hình đầu."),
        code("manifest=pd.read_csv(RUNS['knn']/'fold_manifest.csv')\n"
             "assert len(manifest)==39209 and manifest.Group.nunique()==1307\n"
             "assert manifest.groupby('Group').ValidationFold.nunique().eq(1).all()\n"
             "display(manifest.groupby('ValidationFold').agg(images=('CSVRow','size'),tracks=('Group','nunique')))\n"
             "display(pd.DataFrame([{'algorithm':a,'candidate_id':c['candidate_id'],**c['params']}\n"
             " for a,key in [('knn','knn_candidates'),('dt','dt_candidates')] for c in config[key]]))"),
        md("## 3. Baseline và cấu hình được chọn\n\n"
           "So sánh cùng môi trường/folds. Mean fold F1 khác pooled OOF F1. "
           "SD3fold không phải CI. Selected OOF dùng chính folds chọn tham số nên có selection bias, "
           "không phải nested/unbiased estimate."),
        code("summary=pd.read_csv(TABLES/f'{PREFIX}_summary.csv')\n"
             "candidates=pd.read_csv(TABLES/f'{PREFIX}_candidates.csv')\n"
             "display(summary[['algorithm','role','params','cv_f1_macro_mean','cv_f1_macro_std',"
             "'pooled_oof_f1_macro','train_images','tracks']])\n"
             "display(pd.read_csv(TABLES/f'{PREFIX}_fold_deltas.csv'))\n"
             "display(Image(filename=str(FIGURES/f'{PREFIX}_cv_comparison.png')))"),
        md("## 4. Đường cong k và regularization\n\n"
           "KNN tối ưu tái dùng neighbor queries theo metric/fold, batch có giới hạn; "
           "hòa ở biên k dùng sklearn fallback. DT giữ grid có mục tiêu, không exhaustive interactions."),
        code("display(Image(filename=str(FIGURES/f'{PREFIX}_knn_k_curve.png')))\n"
             "for algorithm in ['knn','dt']:\n"
             " print(algorithm.upper())\n"
             " display(candidates.loc[candidates.algorithm.eq(algorithm)].sort_values('mean_f1_macro',ascending=False,kind='stable').head(8))"),
        md("## 5. H2 — lớp và số track\n\n"
           "Support/track/F1 chỉ mô tả. Lớp ít ảnh không tất yếu yếu; frame cùng track "
           "không độc lập. Không chọn mô hình bằng bảng lỗi official test."),
        code("classes=pd.read_csv(TABLES/f'{PREFIX}_oof_classes.csv')\n"
             "for algorithm in ['knn','dt']:\n"
             " part=classes.loc[classes.algorithm.eq(algorithm)&classes.role.eq('selected')]\n"
             " print(algorithm.upper(),'weakest classes')\n display(part.sort_values('f1').head(10))"),
        md("## 6. H3 — kích thước, chất lượng và cặp nhầm\n\n"
           "Ngưỡng kích thước32 từ EDA. Brightness/contrast đo trên ROI đã resize32; "
           "quartile fit fold-train, khác brightness toàn ảnh ở EDA. Chưa kiểm soát "
           "class/track nên không suy nhân quả, chưa đo blur."),
        code("groups=pd.read_csv(TABLES/f'{PREFIX}_oof_groups.csv')\n"
             "pairs=pd.read_csv(TABLES/f'{PREFIX}_oof_error_pairs.csv')\n"
             "display(groups)\ndisplay(pd.read_csv(TABLES/f'{PREFIX}_quality_thresholds.csv').head(6))\n"
             "display(Image(filename=str(FIGURES/f'{PREFIX}_oof_size_errors.png')))\n"
             "for algorithm in ['knn','dt']:\n"
             " display(pairs.loc[pairs.algorithm.eq(algorithm)&pairs.role.eq('selected')].head(12))"),
        md("## 7. Timing và provenance\n\n"
           "Candidate KNN timings không gồm shared scaler/query; không cộng chúng như end-to-end. "
           "CV wall time và refit được ghi riêng. Không so tốc độ giữa môi trường khác nhau. "
           "Hashes gốc giữ nguyên; bản đồ chuyển đường dẫn chỉ tìm đúng bytes đã ghi nhận."),
        code("display(pd.DataFrame([{'algorithm':item['algorithm'],'candidates':item['candidate_count'],"
             "'cv_seconds':item['cv_wall_seconds'],'refit_seconds':item['refit_seconds'],"
             "'model_MiB':item['model_bytes']/2**20} for item in selection['algorithms']]))\n"
             "provenance=json.loads((ROOT/'logs/knn'/RUNS['knn'].name/'export_provenance.json').read_text(encoding='utf-8'))\n"
             "for relative,expected in provenance['sources_sha256'].items():\n"
             " resolve_evidence(relative,expected,project_root=ROOT)\n"
             "print('Recorded-source hashes verified; no models deserialized.')"),
        md("## 8. Tái lập có chủ ý\n\n"
           "Run-id phải mới; các lệnh train không tự chạy khi Run All.\n\n"
           "```powershell\n"
           "& data/.venv-model3/Scripts/python.exe scripts/train_models.py preflight --dataset-dir data --run-id hog-preflight-new\n"
           "& data/.venv-model3/Scripts/python.exe scripts/train_models.py tune --dataset-dir data --run-id hog-full-new\n"
           "& data/.venv-model3/Scripts/python.exe scripts/summarize_models.py --knn-run report/knn/hog-full-new --dt-run report/dt/hog-full-new\n"
           "```\n\n"
           "Pins môi trường: scripts/requirements_models.txt. Chưa official-test score cho models mới; "
           "test đã được EDA thăm dò. Chưa repeated/nested CV, chưa CI/significance. "
           "Bảng/ảnh nằm report/tables và report/figures, có bản dark."),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (KNN/DT models)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "model_runs": run_paths,
        "original_model_contributor": "Nguyễn Hoàng", "tuning_contributor": "Đặng Quang Tiến",
        "ai_assisted": True, "automatic_training": False, "automatic_inference": False})
    with destination.open("x", encoding="utf-8") as stream:
        nbformat.write(notebook, stream)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knn-run", type=Path, required=True)
    parser.add_argument("--dt-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/03_models.ipynb")
    args = parser.parse_args(argv)
    print(create(args.knn_run, args.output, dt_run=args.dt_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
