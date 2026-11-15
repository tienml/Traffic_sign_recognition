"""Create an artifact-driven, executable person-3 notebook (no automatic training).

The explicit --run-dir selects a completed local run. Notebook generation never
reads official test or loads a model. Run training from the separate documented CLI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def create(run_dir, destination):
    run_dir = Path(run_dir).resolve()
    destination = Path(destination).resolve()
    if not run_dir.is_relative_to(ROOT / "report/model3_upgrade"):
        raise ValueError("Expected a model3_upgrade run under this repository")
    if not destination.is_relative_to(ROOT / "notebooks") or destination.exists():
        raise FileExistsError("Notebook path must be new and under notebooks/")
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    if selection.get("status") != "completed_train_cv":
        raise ValueError("Notebook report requires a completed full train-CV run")
    relative = run_dir.relative_to(ROOT).as_posix()
    run_id = selection["run_id"]
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [
        md("# Đặng Quang Tiến (Người 4) — Nâng cấp K-NN và Decision Tree\n\n"
           "**Người phụ trách phần nâng cấp: Đặng Quang Tiến (người 4)**, có trợ lý AI hỗ trợ triển khai/thực thi. "
           "Phần KNN/DT gốc và kết quả lịch sử của **Nguyễn Hoàng (người 3)** được giữ nguyên. "
           "Tên kỹ thuật `model3_upgrade` chỉ phần mô hình được nâng cấp, không phải người thực hiện.\n\n"
           "Notebook chỉ trình bày **phần nâng cấp do Đặng Quang Tiến phụ trách**. Giữ nguyên mã nguồn, "
           "tham số HOG và kết quả của người 1/2; không sửa các full-run/claim cũ.\n\n"
           "**Giới hạn:** đây là kết quả lựa chọn bằng CV trên train, không phải điểm official test. "
           "OOF của cấu hình được chọn trên chính các fold này vẫn có selection bias; không gọi là nested CV."),
        md("## 1. Động cơ từ EDA\n\n"
           "- H1: RGB tốt hơn grayscale mới có bằng chứng thăm dò cho KNN pixel; notebook này không thay màu/đặc trưng.\n"
           "- H2: train mất cân bằng khoảng 10,71 lần; số frame không đồng nghĩa số track độc lập. "
           "Thử class weighting cho cây là một giả thuyết, không mặc định lớp hiếm đều kém.\n"
           "- H3: ảnh nhỏ và nhiều lớp tốc độ chỉ khác chữ số. Kiểm tra lỗi OOF theo kích thước, "
           "độ sáng/tương phản, lớp và số track; không dùng lỗi official test để chọn cấu hình.\n"
           "- KNN cũ chọn k=7 ở biên grid {3,5,7}; mở rộng đến 31. DT thử leaf size, class weight và pruning.\n\n"
           "Tham khảo: [EDA](../report/notes_eda_preprocessing.md), "
           "[KNN gốc](../report/notes_knn.md), [DT gốc](../report/notes_dt.md)."),
        code("from pathlib import Path\nimport json, hashlib, sys\nimport pandas as pd\nimport numpy as np\nfrom IPython.display import display, Image, Markdown\n\n"
             "ROOT = Path.cwd().resolve()\nif not (ROOT / 'src').is_dir():\n    ROOT = ROOT.parent\n"
             "assert (ROOT / 'src').is_dir(), 'Mở notebook từ root repo hoặc notebooks/'\n"
             f"RUN = ROOT / {relative!r}\nPREFIX = {'model3_' + run_id!r}\n"
             "TABLES = ROOT / 'report/tables'\nFIGURES = ROOT / 'report/figures'\n"
             "selection = json.loads((RUN / 'selection.json').read_text(encoding='utf-8'))\n"
             "config = json.loads((RUN / 'config.json').read_text(encoding='utf-8'))\n"
             "assert selection['status'] == 'completed_train_cv'\n"
             "assert selection['official_test_read'] is False\n"
             "assert selection['official_test_evaluated'] is False\n"
             "print('Run:', selection['run_id'])\nprint('Stored environment:', config['environment'])\n"
             "print('Notebook does not train, predict, deserialize models or read official test.')"),
        md("## 2. Giao thức và phạm vi thay đổi\n\n"
           "RGB → ROI → 32×32 AREA → /255 → HOG 324 chiều **giữ nguyên người 2**. "
           "StandardScaler chỉ fit fold-train. Dùng nguyên manifest 3 folds theo (ClassId, TrackId) "
           "của full KNN trước đó, seed 42.\n\n"
           "KNN: 28 cấu hình (7 giá trị k × 2 weights × 2 metrics). DT: 11 cấu hình "
           "có baseline và thay đổi leaf size/class weight/pruning có kiểm soát. "
           "Chọn mean macro-F1, hòa lấy cấu hình đầu; baseline đứng đầu. "
           "Chỉ HOG được nâng cấp, không chạy lại Pixel/PCA hay đổi thuật toán ngoài phạm vi."),
        code("manifest = pd.read_csv(RUN / 'fold_manifest.csv')\n"
             "assert len(manifest) == 39209 and manifest.Group.nunique() == 1307\n"
             "assert manifest.groupby('Group').ValidationFold.nunique().eq(1).all()\n"
             "assert not manifest.Path.duplicated().any()\n"
             "assert set(manifest.ValidationFold) == {0, 1, 2}\n"
             "display(manifest.groupby('ValidationFold').agg(images=('CSVRow','size'), tracks=('Group','nunique')))\n"
             "display(pd.DataFrame([{'algorithm': a, 'candidate_id': c['candidate_id'], **c['params']}\n"
             " for a, key in [('knn','knn_candidates'),('dt','dt_candidates')] for c in config[key]]))"),
        md("## 3. Baseline chạy lại và cấu hình được chọn\n\n"
           "So sánh chính là baseline và candidate **trong cùng lượt chạy, cùng môi trường và cùng folds**. "
           "Không so smoke với full. Mean macro-F1 theo fold khác với macro-F1 gộp OOF; SD ba folds không phải CI. "
           "Nếu selected trùng baseline hoặc chênh lệch nhỏ/không nhất quán, không tuyên bố cải thiện chắc chắn."),
        code("summary = pd.read_csv(TABLES / f'{PREFIX}_summary.csv')\n"
             "candidates = pd.read_csv(TABLES / f'{PREFIX}_candidates.csv')\n"
             "fold_deltas = pd.read_csv(TABLES / f'{PREFIX}_fold_deltas.csv')\n"
             "display(summary[['algorithm','role','params','cv_f1_macro_mean','cv_f1_macro_std',\n"
             " 'cv_accuracy_mean','pooled_oof_f1_macro','train_images','tracks']])\n"
             "display(fold_deltas)\n"
             "for item in selection['algorithms']:\n"
             "    print(item['algorithm'], 'delta mean CV F1:', item['delta_cv_f1_macro'])\n"
             "    print('selected:', item['selected']['params'])\n"
             "display(Image(filename=str(FIGURES / f'{PREFIX}_cv_comparison.png')))"),
        md("## 4. Đường cong k và regularization DT\n\n"
           "Đường cong chỉ dùng validation folds trên train. Không điều chỉnh grid theo official test. "
           "DT giữ entropy/depth20/split5 của baseline để cô lập leaf size, class weight và một nhánh ccp_alpha; "
           "đây không phải exhaustive search tất cả tương tác."),
        code("display(Image(filename=str(FIGURES / f'{PREFIX}_knn_k_curve.png')))\n"
             "for algorithm in ['knn','dt']:\n"
             "    print(algorithm.upper(), 'top candidates; same-fold selection scores')\n"
             "    display(candidates.loc[candidates.algorithm.eq(algorithm)].sort_values(\n"
             "        'mean_f1_macro', ascending=False, kind='stable').head(8))"),
        md("## 5. Đối chiếu H2 — support, track, recall/F1 từng lớp\n\n"
           "OOF đảm bảo ảnh của validation track không nằm trong fold-train. Tuy nhiên cấu hình selected "
           "đã được chọn bằng các fold này. Bảng chỉ mô tả quan hệ, không chứng minh ít mẫu là nguyên nhân. "
           "Mỗi lớp có số track ít hơn nhiều so với số frame; không coi mọi frame là quan sát độc lập."),
        code("classes = pd.read_csv(TABLES / f'{PREFIX}_oof_classes.csv')\n"
             "for algorithm in ['knn','dt']:\n"
             "    part = classes.loc[classes.algorithm.eq(algorithm) & classes.role.eq('selected')]\n"
             "    print(algorithm.upper(), 'weakest classes')\n"
             "    display(part.sort_values('f1').head(10))\n"
             "    print('Descriptive Spearman counts vs F1:',\n"
             "          part[['train_tracks','f1']].corr(method='spearman').iloc[0,1])"),
        md("## 6. Đối chiếu H3 — kích thước, chất lượng và nhầm tốc độ\n\n"
           "Nhóm kích thước <32 được định nghĩa từ EDA. Brightness/contrast ở đây đo trên **ROI đã resize32**, "
           "không phải brightness toàn ảnh của mẫu EDA cũ. Ngưỡng quartile được tính riêng trên fold-train, "
           "sau đó áp vào validation. Không học thêm đặc trưng hoặc chọn model bằng các nhóm này. "
           "Tỷ lệ lỗi theo nhóm chưa kiểm soát class/track nên không suy nhân quả; chưa đo blur."),
        code("groups = pd.read_csv(TABLES / f'{PREFIX}_oof_groups.csv')\n"
             "pairs = pd.read_csv(TABLES / f'{PREFIX}_oof_error_pairs.csv')\n"
             "thresholds = pd.read_csv(TABLES / f'{PREFIX}_quality_thresholds.csv')\n"
             "display(groups)\ndisplay(thresholds.head(6))\n"
             "display(Image(filename=str(FIGURES / f'{PREFIX}_oof_size_errors.png')))\n"
             "for algorithm in ['knn','dt']:\n"
             "    display(pairs.loc[pairs.algorithm.eq(algorithm) & pairs.role.eq('selected')].head(12))"),
        md("## 7. Chi phí và khả năng tái lập\n\n"
           "KNN tìm hàng xóm một lần cho mỗi metric/fold rồi tái sử dụng cho nhiều k/weights. "
           "Trường hợp hòa ở biên k dùng predict sklearn trực tiếp để giữ semantics. "
           "Không tạo distance matrix toàn dataset. **Không cộng thời gian shared query lặp trong các dòng candidate.** "
           "cv_wall_seconds đo toàn thuật toán; refit tách riêng. Chạy tuần tự, numeric threads=2. "
           "Không phải benchmark so tốc độ với máy chạy lịch sử, không phải latency realtime."),
        code("feature_audit = json.loads((RUN / 'feature_audit.json').read_text(encoding='utf-8'))\n"
             "display(pd.DataFrame([{'algorithm': item['algorithm'],\n"
             " 'candidates': item['candidate_count'], 'cv_wall_seconds': item['cv_wall_seconds'],\n"
             " 'refit_seconds': item['refit_seconds'], 'model_MiB': item['model_bytes']/2**20}\n"
             " for item in selection['algorithms']]))\n"
             "print('Train-only extraction seconds:', feature_audit['extraction_seconds'])\n"
             "print('HOG dimensions:', feature_audit['dimensions'])\n"
             "provenance_path = ROOT / 'logs/model3_upgrade' / selection['run_id'] / 'export_provenance.json'\n"
             "provenance = json.loads(provenance_path.read_text(encoding='utf-8'))\n"
             "for relative, expected in provenance['sources_sha256'].items():\n"
             "    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative\n"
             "print('Saved-source SHA256 checks passed; no models deserialized.')"),
        md("## 8. Chạy lại có chủ ý — không tự train khi Run All\n\n"
           "Chạy các lệnh sau từ terminal repo root; run-id phải mới, không ghi đè run hiện có:\n\n"
           "```powershell\n"
           "& data/.venv-model3/Scripts/python.exe scripts/train_model3_upgrade.py preflight --dataset-dir data --run-id upgrade-preflight-new\n"
           "& data/.venv-model3/Scripts/python.exe scripts/train_model3_upgrade.py smoke --dataset-dir data --run-id upgrade-smoke-new\n"
           "& data/.venv-model3/Scripts/python.exe scripts/train_model3_upgrade.py tune --dataset-dir data --run-id upgrade-full-new\n"
           "& data/.venv-model3/Scripts/python.exe scripts/summarize_model3_upgrade.py --run-dir report/model3_upgrade/upgrade-full-new\n"
           "```\n\n"
           "Môi trường tái lập riêng: `py -3.12 -m venv data/.venv-model3-repro`, sau đó "
           "`data/.venv-model3-repro/Scripts/python.exe -m pip install -r scripts/requirements_model3_upgrade.txt`. "
           "Không thay requirements chung của nhóm.\n\n"
           "Notebook mặc định chỉ đọc artifacts vì full training tốn tài nguyên. Không có official-test mode "
           "trong CLI nâng cấp. Giữ nguyên kết quả test/claim trước đó; chưa có điểm test của model mới. "
           "Chưa nested/repeated CV, chưa CI; cần kiểm chứng riêng trước khi khẳng định nâng cấp tổng quát hóa tốt hơn. "
           "Các biểu đồ có bản dark trong figures và bảng số tương đương trong tables."),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (model3 upgrade)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
        "model3_upgrade": {"run_id": run_id, "automatic_training": False,
                           "contributor": "Đặng Quang Tiến", "member_number": 4,
                           "original_model_contributor": "Nguyễn Hoàng",
                           "ai_assisted": True, "official_test_read": False}})
    with destination.open("x", encoding="utf-8") as stream:
        nbformat.write(notebook, stream)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "notebooks/03_model3_upgrade.ipynb")
    args = parser.parse_args(argv)
    print(create(args.run_dir, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
