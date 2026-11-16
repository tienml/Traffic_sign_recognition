"""Write a new member4 report from verified evaluation tables (no model calls)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def write_report(run_dir, verification_dir, output):
    run, verification_dir, output = Path(run_dir).resolve(), Path(verification_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Evaluation report already exists; do not overwrite completed work")
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    verification = json.loads((verification_dir / "verification.json").read_text(encoding="utf-8"))
    if not verification["new_work_passed"]:
        raise ValueError("Evaluation verification did not pass")
    prefix = f"evaluation_{run.name}"
    def table(name):
        return pd.read_csv(ROOT / f"report/tables/{prefix}_{name}.csv")
    summary = table("official_summary")
    classes = table("official_per_class")
    groups = table("official_error_groups")
    pairs = table("official_error_pairs")
    comparisons = table("official_paired")
    oof = table("upgrade_oof_summary")
    overlap = table("exact_split_overlap")
    test_text = (verification_dir / "full_suite.log").read_text(encoding="utf-8")
    total_match = re.search(r"Ran (\d+) tests", test_text)
    total = int(total_match[1]) if total_match else None
    focused_text = (verification_dir / "focused_tests.log").read_text(encoding="utf-8")
    focus_match = re.search(r"Ran (\d+) tests", focused_text)
    focused = int(focus_match[1]) if focus_match else None
    lines = ["# Đặng Quang Tiến (Người 4) — đánh giá và phân tích lỗi", "",
        "**Người phụ trách: Đặng Quang Tiến.** Trợ lý AI hỗ trợ code/thực thi/tổng hợp; thành viên cần tự xác minh và giải thích trước khi nộp. "
        "Phần KNN/DT gốc của Nguyễn Hoàng, EDA/tiền xử lý của người1 và đặc trưng của người2 được giữ nguyên.", "",
        "## 1. Phạm vi hoàn thành và nguồn bằng chứng", "",
        "Đánh giá từ sáu bộ predictions official-test lịch sử (KNN/DT × Pixel/HOG/HOG+PCA), "
        "đối chiếu đúng12.630 CSVRow/Path/ClassId với Test.csv hiện tại, xác minh frozen selection/claim và metrics. "
        "Không nạp joblib, fit, predict, chạy official test mới hoặc thay siêu tham số. "
        "Thêm đối chiếu OOF của phần nâng cấp trên39.209 ảnh train, tách hẳn khỏi official test.", "",
        "Giữ nguyên source đã làm của các thành viên, kể cả hàm evaluate_and_analyze cũ; bổ sung module/runner mới để không thay hành vi pipeline lịch sử. "
        "Mọi output mới có namespace evaluation_<run-id>_, không ghi đè bảng/hình cũ.", "",
        "## 2. Kết quả official test lịch sử", "",
        "Metrics thang0–1; macro trung bình đều43 lớp, zero_division=0. Balanced accuracy bằng macro recall do đủ43 lớp. "
        "Winner biểu diễn HOG của mỗi thuật toán được chọn bằng CV, không chọn lại theo test.", "",
        "| Thuật toán | Biểu diễn | Accuracy | Precision macro | Recall macro | F1 macro |",
        "|---|---|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.algorithm.upper()} | {row.method} | {row.accuracy:.6f} | {row.precision_macro:.6f} "
                     f"| {row.recall_macro:.6f} | {row.f1_macro:.6f} |")
    lines.extend(["", f"[Bảng số đầy đủ](tables/{prefix}_official_summary.csv).", "",
        f"![So sánh official-test lịch sử](figures/{prefix}_official_comparison.png)", "",
        "KNN HOG cao hơn DT HOG trong protocol hiện có; không khẳng định ưu thế phổ quát hoặc significance. "
        "CV và official test khác phân phối; không lấy chênh lệch test–CV để khẳng định leakage hoặc generalization tốt hơn.", "",
        "## 3. Confusion matrices và kết quả từng lớp", "",
        "Mỗi trong sáu model có hình count và row-normalized43×43; hàng nhãn thật, cột dự đoán. "
        "Count dùng chung thang0–750, normalized0–1 để đọc nhất quán. Bảng long chứa6×43×43 ô, có mapping43 tên lớp.", "",
        f"![KNN HOG confusion](figures/{prefix}_cm_knn_hog.png)", "",
        f"![DT HOG confusion](figures/{prefix}_cm_dt_hog.png)", "",
        f"[Toàn bộ ô confusion](tables/{prefix}_official_confusions.csv), "
        f"[mapping tên lớp](tables/{prefix}_class_mapping.csv), [metrics từng lớp](tables/{prefix}_official_per_class.csv).", "",
        f"![Per-class F1 của hai winner CV](figures/{prefix}_per_class_hog.png)", "",
        "### H2 của EDA: số mẫu không quyết định hoàn toàn chất lượng", "",
        "| ClassId | Train images | KNN HOG recall | DT HOG recall |",
        "|---|---:|---:|---:|"])
    for c in (0,19,37,1,30):
        a=classes.loc[classes.algorithm.eq("knn")&classes.method.eq("hog")&classes.class_id.eq(c)].iloc[0]
        b=classes.loc[classes.algorithm.eq("dt")&classes.method.eq("hog")&classes.class_id.eq(c)].iloc[0]
        lines.append(f"| {c} | {int(a.train_images)} | {a.recall:.4f} | {b.recall:.4f} |")
    lines.extend(["", "Các lớp0/19/37 cùng ít mẫu nhưng recall khác nhau; lớp1 đông mẫu vẫn có nhiều lỗi. "
        "Mất cân bằng đã được EDA xác nhận, nhưng không đủ để coi đó là nguyên nhân duy nhất. "
        "Không tự oversample hoặc sửa class weights từ quan sát official test này.", "",
        "## 4. H3 của EDA: kích thước và nhóm tốc độ", "",
        "| Nhóm ảnh gốc | n | Lỗi KNN HOG | Lỗi DT HOG |",
        "|---|---:|---:|---:|"])
    for group in ("lt32","ge32"):
        a=groups.loc[groups.algorithm.eq("knn")&groups.method.eq("hog")&groups.dimension.eq("raw_min_dimension")&groups.group.eq(group)].iloc[0]
        b=groups.loc[groups.algorithm.eq("dt")&groups.method.eq("hog")&groups.dimension.eq("raw_min_dimension")&groups.group.eq(group)].iloc[0]
        lines.append(f"| min(W,H) {'<32' if group=='lt32' else '≥32'} | {int(a.n)} | {a.error_rate:.2%} | {b.error_rate:.2%} |")
    lines.extend(["", "Đây là đối chiếu hậu nghiệm theo ngưỡng EDA, không kiểm soát class/track; "
        "không chứng minh kích thước là nguyên nhân. Chưa đo blur, không gán nhãn thiếu sáng/lỗi crop chỉ từ class hoặc ảnh minh họa.", "",
        f"![Các nhóm lỗi mô tả](figures/{prefix}_official_error_groups.png)", "",
        "Cặp nhầm có hướng thật→dự đoán của HOG:", ""])
    for algorithm in ("knn","dt"):
        part=pairs.loc[pairs.algorithm.eq(algorithm)&pairs.method.eq("hog")].head(5)
        lines.append(f"- {algorithm.upper()}: "+", ".join(f"{int(r.true_class)}→{int(r.predicted_class)}: {int(r.count)}" for r in part.itertuples())+".")
    lines.extend(["", "Nhiều cặp thuộc nhóm biển tốc độ cùng màu/khung, khác chữ số. Điều này xác nhận mẫu lỗi, "
        "chưa chứng minh HOG hoặc resize là nguyên nhân; không đổi đặc trưng người2 từ báo cáo này.", "",
        "## 5. Ví dụ ảnh sai có khả năng tái lập", "",
        "Mỗi thuật toán chọn4 cặp nhầm thường gặp,2 ảnh/cặp theo thứ tự SHA256(seed42|Path). "
        "Đây là mẫu minh họa xác định, không phải mẫu ngẫu nhiên đại diện. Hình hiển thị ảnh gốc + ROI theo CSV, true/pred class; "
        "không tự ghi nguyên nhân sai khi chưa có phép đo đối chứng.", "",
        f"![KNN HOG ảnh sai](figures/{prefix}_error_gallery_knn_hog.png)", "",
        f"![DT HOG ảnh sai](figures/{prefix}_error_gallery_dt_hog.png)", "",
        f"[Manifest ví dụ](tables/{prefix}_error_examples.csv).", "",
        "## 6. So sánh cùng ảnh — không giả định IID", "",
        "| Biểu diễn | Cả hai đúng | Chỉ KNN đúng | Chỉ DT đúng | Cả hai sai |",
        "|---|---:|---:|---:|---:|"])
    for r in comparisons.itertuples():
        lines.append(f"| {r.method} | {r.both_correct} | {r.first_only_correct} | {r.second_only_correct} | {r.both_wrong} |")
    lines.extend(["", "Chỉ là đếm ghép cặp mô tả; chưa dùng McNemar/CI do phụ thuộc frame/track và thiết kế thử nghiệm chưa đáp ứng giả định độc lập. "
        "Không gộp accuracy chênh lệch với F1 chênh lệch như cùng một chỉ số.", "",
        "## 7. Audit trùng nội dung official split", "",
        f"Phát hiện **{len(overlap)} cặp ảnh train–test trùng byte SHA256**. Train hashes lấy từ full upgrade audit; test hashes tính trực tiếp. "
        "Giữ nguyên split và toàn bộ12.630 ảnh trong metrics. Không loại bỏ để làm đẹp điểm hoặc thay benchmark. "
        "Chưa định lượng tác động; hash byte không loại trừ near-duplicates.", "",
        f"[Danh sách cặp](tables/{prefix}_exact_split_overlap.csv).", "",
        "## 8. Kết quả nâng cấp của Tiến — tách riêng OOF train", "",
        "| Thuật toán | Vai trò | Mean fold F1 | Pooled OOF F1 | Pooled OOF Accuracy |",
        "|---|---|---:|---:|---:|"])
    for r in oof.itertuples():
        lines.append(f"| {r.algorithm.upper()} | {r.role} | {r.mean_fold_f1_macro:.6f} | {r.f1_macro:.6f} | {r.accuracy:.6f} |")
    lines.extend(["", "Mỗi dự đoán OOF đến từ model không fit validation track, nhưng cấu hình được chọn trên chính các folds này nên vẫn selection bias; "
        "không phải nested/unbiased estimate. **Không có official-test score của model mới**. "
        "Không lấy chênh lệch OOF–test làm mức tăng chất lượng. [Báo cáo KNN](notes_knn.md), "
        "[báo cáo DT](notes_dt.md) và [notebook03](../notebooks/03_models.ipynb) giữ nguồn/delta/timing chi tiết.", "",
        "## 9. Kiểm thử, logs và notebook thực thi", "",
        f"Focused evaluation **{focused}/{focused} pass**; full suite **{total-1 if verification['known_baseline_failure_only'] else total}/{total} pass**. "
        + ("Còn đúng lỗi PCA float32 có sẵn tại src/tests/test_features.py:149. " if verification['known_baseline_failure_only'] else "Không có kiểm thử thất bại trong môi trường hiện tại. ")
        + "Không sửa code/tolerance người2 để làm đẹp báo cáo. "
        f"Notebook04 chạy **{verification['steps']['notebook']['code_cells']} code cells, "
        f"{verification['steps']['notebook']['errors']} error outputs**. Run All chỉ đọc artifacts, không fit/predict.", "",
        f"[Notebook04](../notebooks/04_evaluation_tien.ipynb), "
        f"[log đánh giá](../logs/evaluation/{run.name}/console.log), "
        f"[verification](../{verification_dir.relative_to(ROOT).as_posix()}/verification.json), "
        f"[provenance](evaluation/{run.name}/provenance.json).", "",
        "## 10. Giới hạn và phạm vi hoàn thành", "",
        "Phần người4 hoàn thành việc đánh giá/phân tích lỗi và bàn giao notebook/bằng chứng. "
        "Không thay phần đã làm của người khác hoặc viết lại main.py theo protocol mới. "
        "Bài toán chỉ phân loại biển đã định vị, không detection/realtime/an toàn triển khai. "
        "Test đã xem ở EDA, có trùng byte, một seed/ba folds và chưa uncertainty theo track. "
        "Nguồn giấy phép dữ liệu local vẫn cần người học xác minh trước khi tái phân phối.", "",
        "### Tái lập không huấn luyện lại", "", "```powershell",
        "& data/.venv-model3/Scripts/python.exe scripts/evaluate_saved_models.py --dataset-dir data --knn-run report/knn/hog-full-20261115 --dt-run report/dt/hog-full-20261115 --run-id tien-evaluation-new",
        "& data/.venv-model3/Scripts/python.exe scripts/create_evaluation_notebook.py --run-dir report/evaluation/tien-evaluation-new --output notebooks/04_evaluation_tien_new.ipynb",
        "```", "", "Mọi run-id/output mới phải chưa tồn tại; không overwrite artifacts cũ. Không commit/push trong lượt này."])
    with output.open("x",encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--verification-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'report/notes_evaluation_tien.md')
    args=parser.parse_args(argv)
    print(write_report(args.run_dir,args.verification_dir,args.output))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
