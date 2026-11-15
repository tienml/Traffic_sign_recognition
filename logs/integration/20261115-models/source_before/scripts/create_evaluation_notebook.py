"""Build member4's saved-artifact evaluation notebook with no model execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def create(run_dir, output):
    run, output = Path(run_dir).resolve(), Path(output).resolve()
    if not run.is_relative_to(ROOT / "report/evaluation") or not output.is_relative_to(ROOT / "notebooks"):
        raise ValueError("Expected local evaluation run and notebooks output")
    if output.exists():
        raise FileExistsError("Notebook already exists")
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    run_id = provenance["run_id"]
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        md("# Đặng Quang Tiến (Người 4) — Đánh giá và phân tích lỗi\n\n"
           "**Đóng góp:** đánh giá tích hợp, confusion matrix, per-class metrics, so sánh lỗi, "
           "ví dụ ảnh và đối chiếu EDA. Có trợ lý AI hỗ trợ, không coi thực thi tự động là xác minh cá nhân. "
           "Giữ nguyên EDA/đặc trưng/mô hình và mọi full-run/test claims của các thành viên.\n\n"
           "**Hai tập bằng chứng tách biệt:** official test của sáu pipeline lịch sử; OOF trên train "
           "của phần nâng cấp. Không fit, predict hoặc nạp joblib khi Run All; không có điểm official test của model nâng cấp."),
        md("## 1. Dữ liệu, provenance và phạm vi đánh giá\n\n"
           "Train39.209/Test12.630,43 lớp. Đối chiếu CSVRow/Path/ClassId trực tiếp với CSV local; "
           "xác minh selection/claim hashes và tính lại các metrics từ predictions lịch sử. "
           "Test đã được EDA thăm dò, không gọi là holdout hoàn toàn mù. "
           "Nguồn và sản phẩm đều có SHA256; không sửa ảnh/split/nhãn."),
        code("from pathlib import Path\nimport json, hashlib\nimport pandas as pd\nimport numpy as np\n"
             "from IPython.display import display, Image, Markdown\n"
             "ROOT=Path.cwd().resolve()\nif not (ROOT/'src').is_dir(): ROOT=ROOT.parent\n"
             f"RUN=ROOT/{run.relative_to(ROOT).as_posix()!r}\nPREFIX={'evaluation_'+run_id!r}\n"
             "TABLES=ROOT/'report/tables'\nFIGURES=ROOT/'report/figures'\n"
             "provenance=json.loads((RUN/'provenance.json').read_text(encoding='utf-8'))\n"
             "assert provenance['fit'] is False and provenance['predict'] is False\n"
             "assert provenance['joblib_deserialized'] is False\n"
             "assert provenance['new_official_test_inference'] is False\n"
             "print('Contributor:',provenance['contributor'],'member',provenance['member_number'])\n"
             "for relative, expected in provenance['sources_sha256'].items():\n"
             "    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected, relative\n"
             "for relative, expected in provenance['outputs_sha256'].items():\n"
             "    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected, relative\n"
             "print('All input/output SHA256 checks passed.')\n"
             "print('Implementation fingerprints are recorded separately; later source edits do not rewrite evidence.')"),
        md("## 2. Metrics của sáu pipeline lịch sử\n\n"
           "Accuracy phản ánh tỷ lệ ảnh đúng; Precision/Recall/F1 macro cho mỗi lớp trọng số bằng nhau. "
           "Tập chính thức có đủ43 lớp nên balanced accuracy bằng macro recall. "
           "Winner biểu diễn được chọn bằng CV trước test, không chọn lại mô hình bằng bảng này. "
           "Không diễn giải cột CV và test là cùng một phân phối hay hiệu quả có ý nghĩa thống kê."),
        code("summary=pd.read_csv(TABLES/f'{PREFIX}_official_summary.csv')\n"
             "display(summary[['algorithm','method','accuracy','precision_macro','recall_macro','f1_macro','balanced_accuracy',\n"
             " 'representation_is_cv_winner']])\n"
             "display(Image(filename=str(FIGURES/f'{PREFIX}_official_comparison.png')))"),
        md("## 3. Confusion matrices đầy đủ và chuẩn hóa theo nhãn thật\n\n"
           "Hàng=true, cột=predicted; chuẩn hóa mỗi hàng để không bị lớp đông chi phối. "
           "Mỗi model có cả count và row-normalized CSV43×43 dạng long. ClassId↔tên được lưu riêng. "
           "Không suy nguyên nhân ảnh mờ/tối từ ô nhầm lẫn."),
        code("mapping=pd.read_csv(TABLES/f'{PREFIX}_class_mapping.csv')\n"
             "display(mapping)\n"
             "for model in ['knn_hog','dt_hog']:\n"
             "    display(Image(filename=str(FIGURES/f'{PREFIX}_cm_{model}.png')))\n"
             "print('Các hình còn lại: knn_pixel, knn_hog_pca, dt_pixel, dt_hog_pca; xem report/figures.')\n"
             "confusions=pd.read_csv(TABLES/f'{PREFIX}_official_confusions.csv')\n"
             "assert len(confusions)==6*43*43\n"
             "assert confusions.groupby(['algorithm','method'])['count'].sum().eq(12630).all()"),
        md("## 4. H2 — lớp ít mẫu và hiệu năng từng lớp\n\n"
           "Xem support cùng Precision/Recall/F1, không mặc định mọi lớp hiếm đều kém. "
           "Các track tạo nhiều frame gần nhau; số frame không phải số quan sát độc lập. "
           "H2 của EDA được đối chiếu mô tả ở đây, không chứng minh quan hệ nhân quả."),
        code("classes=pd.read_csv(TABLES/f'{PREFIX}_official_per_class.csv')\n"
             "for algorithm in ['knn','dt']:\n"
             "    part=classes.loc[classes.algorithm.eq(algorithm)&classes.method.eq('hog')]\n"
             "    print(algorithm.upper(),'weakest classes')\n"
             "    display(part.sort_values('f1').head(10))\n"
             "    display(part.loc[part.class_id.isin([0,19,37,1])])\n"
             "display(Image(filename=str(FIGURES/f'{PREFIX}_per_class_hog.png')))"),
        md("## 5. H3 — kích thước và nhóm biển tốc độ\n\n"
           "Ngưỡng min(W,H)<32 lấy từ EDA, không tìm ngưỡng tối đa hóa chênh lệch lỗi. "
           "Bảng phân nhóm không điều chỉnh class/track và không chứng minh ảnh nhỏ gây lỗi. "
           "Nhóm tốc độ gồm ClassId0,1,2,3,4,5,7,8. Không đổi cấu hình từ phân tích test này."),
        code("groups=pd.read_csv(TABLES/f'{PREFIX}_official_error_groups.csv')\n"
             "display(groups.loc[groups.method.eq('hog')&groups.dimension.isin(['raw_min_dimension','roi_min_dimension','sign_family'])])\n"
             "display(Image(filename=str(FIGURES/f'{PREFIX}_official_error_groups.png')))\n"
             "pairs=pd.read_csv(TABLES/f'{PREFIX}_official_error_pairs.csv')\n"
             "for algorithm in ['knn','dt']:\n"
             "    print(algorithm.upper(),'top confusion pairs')\n"
             "    display(pairs.loc[pairs.algorithm.eq(algorithm)&pairs.method.eq('hog')].head(12))"),
        md("## 6. Ví dụ ảnh dự đoán sai\n\n"
           "Chọn4 cặp nhầm thường gặp,2 ảnh mỗi cặp theo thứ tự SHA256(seed42|Path). Đây là mẫu minh họa xác định, "
           "không phải mẫu ngẫu nhiên đại diện. Hiển thị ảnh gốc và ROI đúng CSV để người học kiểm tra; "
           "caption chỉ ghi true/prediction, không gán nguyên nhân chủ quan. Các hình không thay cho kiểm định định lượng."),
        code("examples=pd.read_csv(TABLES/f'{PREFIX}_error_examples.csv')\n"
             "display(examples)\n"
             "for algorithm in ['knn','dt']:\n"
             "    display(Image(filename=str(FIGURES/f'{PREFIX}_error_gallery_{algorithm}_hog.png')))"),
        md("## 7. So sánh ghép cặp KNN–DT\n\n"
           "Các mô hình được đối chiếu trên cùng ảnh: cả hai đúng, chỉ KNN đúng, chỉ DT đúng, cả hai sai. "
           "Đây là đếm mô tả, không p-value/CI; không giả định các frame độc lập. "
           "Không coi chênh lệch trên một benchmark là ưu thế phổ quát."),
        code("paired=pd.read_csv(TABLES/f'{PREFIX}_official_paired.csv')\ndisplay(paired)"),
        md("## 8. Audit trùng byte giữa official train/test\n\n"
           "SHA256 bytes ảnh kiểm tra trùng chính xác, không phát hiện mọi near-duplicate. "
           "Giữ nguyên official split; không loại ảnh khỏi metrics để tạo một benchmark khác. "
           "Số cặp trùng không tự định lượng ảnh hưởng đến hiệu năng."),
        code("overlap=pd.read_csv(TABLES/f'{PREFIX}_exact_split_overlap.csv')\n"
             "display(overlap)\n"
             "assert len(overlap)==provenance['exact_duplicate_pairs']\n"
             "print('Exact byte pairs:',len(overlap),'— split unchanged')"),
        md("## 9. Kết quả nâng cấp: chỉ OOF train, không trộn với official test\n\n"
           "Model mới do Đặng Quang Tiến nâng cấp phần KNN/DT của Nguyễn Hoàng. "
           "Bảng OOF có selection bias vì cấu hình được chọn trên chính folds đó; không phải nested/unbiased evaluation. "
           "Không so OOF F1 trực tiếp với official-test F1 như một mức cải thiện. "
           "Xem notebook03 để biết grid, delta từng fold và train–validation gap."),
        code("oof=pd.read_csv(TABLES/f'{PREFIX}_upgrade_oof_summary.csv')\n"
             "oof_pairs=pd.read_csv(TABLES/f'{PREFIX}_upgrade_oof_paired.csv')\n"
             "display(oof)\ndisplay(oof_pairs)\n"
             "print('No new official-test score for upgraded models.')"),
        md("## 10. Giới hạn và tái lập\n\n"
           "- Bài toán phân loại biển đã định vị, không detection hay kiểm chứng an toàn thực địa.\n"
           "- Test đã được EDA xem; số cặp trùng byte được xác minh ở mục8, near-duplicate chưa được loại trừ.\n"
           "- Một seed/ba fold; chưa nested/repeated CV hoặc CI theo track.\n"
           "- Không gán blur/ánh sáng là nguyên nhân; không thay đổi preprocessing/features/models của thành viên khác.\n"
           "- Script mới `scripts/evaluate_saved_models.py` chỉ đọc predictions, tạo output namespace mới; run-id phải mới.\n"
           "- Run All notebook chỉ đọc artifacts. Logs/test/env trong `logs/evaluation/`; báo cáo trong `report/notes_evaluation_tien.md`.\n\n"
           "```powershell\n"
           "& data/.venv-model3/Scripts/python.exe scripts/evaluate_saved_models.py --dataset-dir data "
           "--upgrade-run report/model3_upgrade/upgrade-full-20261115 --run-id tien-evaluation-new\n"
           "```\n\n"
           "Thành viên tự đọc, xác minh và giải thích nội dung trước khi nộp."),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (Tiến evaluation)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "contributor": "Đặng Quang Tiến",
        "member_number": 4, "ai_assisted": True, "evaluation_run": run_id,
        "automatic_training": False, "automatic_inference": False})
    with output.open("x", encoding="utf-8") as stream:
        nbformat.write(notebook, stream)
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=ROOT/'notebooks/04_evaluation_tien.ipynb')
    args=parser.parse_args(argv)
    print(create(args.run_dir,args.output))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
