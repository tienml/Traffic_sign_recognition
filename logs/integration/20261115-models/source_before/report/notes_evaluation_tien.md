# Đặng Quang Tiến (Người 4) — đánh giá và phân tích lỗi

**Người phụ trách: Đặng Quang Tiến.** Trợ lý AI hỗ trợ code/thực thi/tổng hợp; thành viên cần tự xác minh và giải thích trước khi nộp. Phần KNN/DT gốc của Nguyễn Hoàng, EDA/tiền xử lý của người1 và đặc trưng của người2 được giữ nguyên.

## 1. Phạm vi hoàn thành và nguồn bằng chứng

Đánh giá từ sáu bộ predictions official-test lịch sử (KNN/DT × Pixel/HOG/HOG+PCA), đối chiếu đúng12.630 CSVRow/Path/ClassId với Test.csv hiện tại, xác minh frozen selection/claim và metrics. Không nạp joblib, fit, predict, chạy official test mới hoặc thay siêu tham số. Thêm đối chiếu OOF của phần nâng cấp trên39.209 ảnh train, tách hẳn khỏi official test.

Sau yêu cầu tổ chức lại ngày **2026-11-15**, logic đánh giá đã tích hợp trực tiếp vào
[src/evaluate.py](../src/evaluate.py), thay cho module `src/evaluation_saved.py` bổ sung
ban đầu. Runner chỉ phụ trách I/O/provenance và xuất sản phẩm; không có bản metrics core
thứ hai. Hàm `evaluate_and_analyze` vẫn tương thích pipeline cũ; luồng saved evaluation
không gọi inference adapter đó. Source người1/2, dữ liệu, kết quả và claims lịch sử
không đổi. Mọi output mới có namespace evaluation_<run-id>_, không ghi đè bảng/hình cũ.
Implementation hashes trong provenance vẫn là của lượt chạy ban đầu, không thay bằng
hash source sau refactor; bằng chứng tích hợp lưu riêng tại
[logs/integration/20261115-canonical](../logs/integration/20261115-canonical/).

## 2. Kết quả official test lịch sử

Metrics thang0–1; macro trung bình đều43 lớp, zero_division=0. Balanced accuracy bằng macro recall do đủ43 lớp. Winner biểu diễn HOG của mỗi thuật toán được chọn bằng CV, không chọn lại theo test.

| Thuật toán | Biểu diễn | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---|---:|---:|---:|---:|
| KNN | pixel | 0.557245 | 0.513669 | 0.490561 | 0.493948 |
| KNN | hog | 0.818369 | 0.813760 | 0.777906 | 0.789750 |
| KNN | hog_pca | 0.796675 | 0.792526 | 0.747460 | 0.763063 |
| DT | pixel | 0.635313 | 0.578621 | 0.558620 | 0.563179 |
| DT | hog | 0.656770 | 0.614162 | 0.585053 | 0.594987 |
| DT | hog_pca | 0.591607 | 0.560685 | 0.522457 | 0.536760 |

[Bảng số đầy đủ](tables/evaluation_tien-evaluation-20261115_official_summary.csv).

![So sánh official-test lịch sử](figures/evaluation_tien-evaluation-20261115_official_comparison.png)

KNN HOG cao hơn DT HOG trong protocol hiện có; không khẳng định ưu thế phổ quát hoặc significance. CV và official test khác phân phối; không lấy chênh lệch test–CV để khẳng định leakage hoặc generalization tốt hơn.

## 3. Confusion matrices và kết quả từng lớp

Mỗi trong sáu model có hình count và row-normalized43×43; hàng nhãn thật, cột dự đoán. Count dùng chung thang0–750, normalized0–1 để đọc nhất quán. Bảng long chứa6×43×43 ô, có mapping43 tên lớp.

![KNN HOG confusion](figures/evaluation_tien-evaluation-20261115_cm_knn_hog.png)

![DT HOG confusion](figures/evaluation_tien-evaluation-20261115_cm_dt_hog.png)

[Toàn bộ ô confusion](tables/evaluation_tien-evaluation-20261115_official_confusions.csv), [mapping tên lớp](tables/evaluation_tien-evaluation-20261115_class_mapping.csv), [metrics từng lớp](tables/evaluation_tien-evaluation-20261115_official_per_class.csv).

![Per-class F1 của hai winner CV](figures/evaluation_tien-evaluation-20261115_per_class_hog.png)

### H2 của EDA: số mẫu không quyết định hoàn toàn chất lượng

| ClassId | Train images | KNN HOG recall | DT HOG recall |
|---|---:|---:|---:|
| 0 | 210 | 0.3667 | 0.2333 |
| 19 | 210 | 0.8167 | 0.5333 |
| 37 | 210 | 0.8667 | 0.5333 |
| 1 | 2220 | 0.6181 | 0.4847 |
| 30 | 450 | 0.2467 | 0.1867 |

Các lớp0/19/37 cùng ít mẫu nhưng recall khác nhau; lớp1 đông mẫu vẫn có nhiều lỗi. Mất cân bằng đã được EDA xác nhận, nhưng không đủ để coi đó là nguyên nhân duy nhất. Không tự oversample hoặc sửa class weights từ quan sát official test này.

## 4. H3 của EDA: kích thước và nhóm tốc độ

| Nhóm ảnh gốc | n | Lỗi KNN HOG | Lỗi DT HOG |
|---|---:|---:|---:|
| min(W,H) <32 | 2292 | 26.61% | 45.99% |
| min(W,H) ≥32 | 10338 | 16.29% | 31.74% |

Đây là đối chiếu hậu nghiệm theo ngưỡng EDA, không kiểm soát class/track; không chứng minh kích thước là nguyên nhân. Chưa đo blur, không gán nhãn thiếu sáng/lỗi crop chỉ từ class hoặc ảnh minh họa.

![Các nhóm lỗi mô tả](figures/evaluation_tien-evaluation-20261115_official_error_groups.png)

Cặp nhầm có hướng thật→dự đoán của HOG:

- KNN: 1→5: 119, 1→2: 93, 2→5: 81, 7→8: 65, 5→1: 54.
- DT: 5→1: 131, 1→5: 118, 2→1: 103, 2→5: 89, 5→2: 89.

Nhiều cặp thuộc nhóm biển tốc độ cùng màu/khung, khác chữ số. Điều này xác nhận mẫu lỗi, chưa chứng minh HOG hoặc resize là nguyên nhân; không đổi đặc trưng người2 từ báo cáo này.

## 5. Ví dụ ảnh sai có khả năng tái lập

Mỗi thuật toán chọn4 cặp nhầm thường gặp,2 ảnh/cặp theo thứ tự SHA256(seed42|Path). Đây là mẫu minh họa xác định, không phải mẫu ngẫu nhiên đại diện. Hình hiển thị ảnh gốc + ROI theo CSV, true/pred class; không tự ghi nguyên nhân sai khi chưa có phép đo đối chứng.

![KNN HOG ảnh sai](figures/evaluation_tien-evaluation-20261115_error_gallery_knn_hog.png)

![DT HOG ảnh sai](figures/evaluation_tien-evaluation-20261115_error_gallery_dt_hog.png)

[Manifest ví dụ](tables/evaluation_tien-evaluation-20261115_error_examples.csv).

## 6. So sánh cùng ảnh — không giả định IID

| Biểu diễn | Cả hai đúng | Chỉ KNN đúng | Chỉ DT đúng | Cả hai sai |
|---|---:|---:|---:|---:|
| pixel | 5392 | 1646 | 2632 | 2960 |
| hog | 7696 | 2640 | 599 | 1695 |
| hog_pca | 6969 | 3093 | 503 | 2065 |

Chỉ là đếm ghép cặp mô tả; chưa dùng McNemar/CI do phụ thuộc frame/track và thiết kế thử nghiệm chưa đáp ứng giả định độc lập. Không gộp accuracy chênh lệch với F1 chênh lệch như cùng một chỉ số.

## 7. Audit trùng nội dung official split

Phát hiện **8 cặp ảnh train–test trùng byte SHA256**. Train hashes lấy từ full upgrade audit; test hashes tính trực tiếp. Giữ nguyên split và toàn bộ12.630 ảnh trong metrics. Không loại bỏ để làm đẹp điểm hoặc thay benchmark. Chưa định lượng tác động; hash byte không loại trừ near-duplicates.

[Danh sách cặp](tables/evaluation_tien-evaluation-20261115_exact_split_overlap.csv).

## 8. Kết quả nâng cấp của Tiến — tách riêng OOF train

| Thuật toán | Vai trò | Mean fold F1 | Pooled OOF F1 | Pooled OOF Accuracy |
|---|---|---:|---:|---:|
| KNN | baseline | 0.809718 | 0.812176 | 0.822311 |
| KNN | selected | 0.812096 | 0.813967 | 0.827030 |
| DT | baseline | 0.614898 | 0.616536 | 0.658777 |
| DT | selected | 0.621998 | 0.623811 | 0.666939 |

Mỗi dự đoán OOF đến từ model không fit validation track, nhưng cấu hình được chọn trên chính các folds này nên vẫn selection bias; không phải nested/unbiased estimate. **Không có official-test score của model mới**. Không lấy chênh lệch OOF–test làm mức tăng chất lượng. [Báo cáo nâng cấp](notes_model3_upgrade.md) và [notebook03](../notebooks/03_model3_upgrade.ipynb) giữ nguồn/delta/timing chi tiết.

## 9. Kiểm thử, logs và notebook thực thi

Trong lượt thực nghiệm ban đầu: focused evaluation **16/16 pass**; full suite **87/88 pass**.
Sau tích hợp canonical ngày2026-11-15: evaluation **18/18**, model **17/17** pass;
toàn suite **92/93**, vẫn đúng lỗi PCA float32 có sẵn tại src/tests/test_features.py:149.
Không sửa code/tolerance người2 để làm đẹp báo cáo. Notebook03/04 đã chạy lại **7/9 code
cells, 0 error outputs**; bản executed nằm thư mục verification mới, không ghi đè bản gốc.
API src/evaluate.py tính lại sáu official-test CSV và bốn OOF CSV khớp bảng đã lưu;
không nạp model hoặc inference lại. Xem [verification canonical](../logs/evaluation/verification-canonical-20261115/verification.json)
và [đối chiếu số liệu](../logs/integration/20261115-canonical/canonical_evidence_checks.json).
Run All chỉ đọc artifacts, không fit/predict.

[Notebook04](../notebooks/04_evaluation_tien.ipynb), [log đánh giá](../logs/evaluation/tien-evaluation-20261115/console.log), [verification](../logs/evaluation/verification-tien-evaluation-20261115-final/verification.json), [provenance](evaluation/tien-evaluation-20261115/provenance.json).

## 10. Giới hạn và phạm vi hoàn thành

Phần người4 hoàn thành việc đánh giá/phân tích lỗi và bàn giao notebook/bằng chứng.
Source model/evaluation đã tích hợp đúng file chuẩn theo yêu cầu ngày2026-11-15;
không đổi phần người1/2, artifacts/claim lịch sử hoặc viết lại main.py theo protocol mới. Bài toán chỉ phân loại biển đã định vị, không detection/realtime/an toàn triển khai. Test đã xem ở EDA, có trùng byte, một seed/ba folds và chưa uncertainty theo track. Nguồn giấy phép dữ liệu local vẫn cần người học xác minh trước khi tái phân phối.

### Tái lập không huấn luyện lại

```powershell
& data/.venv-model3/Scripts/python.exe scripts/evaluate_saved_models.py --dataset-dir data --upgrade-run report/model3_upgrade/upgrade-full-20261115 --run-id tien-evaluation-new
& data/.venv-model3/Scripts/python.exe scripts/create_evaluation_notebook.py --run-dir report/evaluation/tien-evaluation-new --output notebooks/04_evaluation_tien_new.ipynb
```

Mọi run-id/output mới phải chưa tồn tại; không overwrite artifacts cũ. Không commit/push trong lượt này.
