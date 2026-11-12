# Ghi chú: biểu diễn đặc trưng (Thành viên 2)

Tài liệu này tóm tắt phần `src/features.py` để đưa vào báo cáo mục 2 (thiết kế giải
pháp) và mục 5 (thảo luận, so sánh). Toàn bộ số liệu lấy từ
`report/tables/feature_ablation.csv` và `report/figures/feature_ablation.png`.

## 1. Lựa chọn kỹ thuật

Ảnh đầu vào đã qua bước tiền xử lý của nhóm: crop ROI, resize 32×32 `INTER_AREA`,
RGB, `float32` trong `[0, 1]`. Ba cách biểu diễn được so sánh:

| Phương pháp | Mô tả | Số chiều |
| :--- | :--- | ---: |
| `pixel` | Baseline đúng như đề gợi ý: duỗi vector pixel sau resize/normalize | 3072 |
| `hog` | HOG 9 hướng, ô 8×8, block 2×2, `L2-Hys`, rồi StandardScaler | 324 |
| `hog_pca` | HOG như trên, chuẩn hóa, PCA giữ 95% phương sai | 107 |

Lý do chọn:
- **HOG** mô tả hướng cạnh thay vì độ sáng từng điểm ảnh. Biển báo phân biệt chủ yếu
  bằng hình dạng và cấu trúc ký hiệu, nên hướng cạnh là tín hiệu ổn định hơn pixel thô
  trước thay đổi độ sáng và nền đường.
- **Chuẩn hóa trước PCA** để các chiều HOG (có độ lớn khác nhau giữa các block) cùng
  đóng góp vào phương sai; thiếu bước này PCA chủ yếu giữ các chiều có phương sai lớn
  do thang đo chứ không do lượng thông tin.
- **PCA giữ 95% phương sai** là mức mặc định hợp lý: đủ để bỏ các chiều gần như hằng
  số và nhiễu, nhưng vẫn giữ cấu trúc đủ cho 43 lớp. Không chọn số thành phần cố định
  trước vì 324 chiều HOG không cho biết trước bao nhiêu chiều là đủ.

## 2. Cách chống rò rỉ dữ liệu

- `FeatureExtractor.fit_transform` chỉ dùng cho **train**; `transform` cho test.
  Scaler (`StandardScaler`) và `PCA` được `fit` đúng một lần trên train, sau đó
  chỉ `transform` lên test — kể cả khi được gọi nhiều lần.
- Kiểm thử `test_scaler_and_pca_are_fitted_on_train_only` xác nhận tham số
  (mean, components) không đổi sau khi transform test.
- Kiểm thử `test_test_rows_do_not_depend_on_batch_composition` xác nhận đặc trưng
  của một ảnh test không phụ thuộc vào việc nó được transform cùng batch lớn hay lẻ.
- Baseline `pixel` không chuẩn hóa, đúng như mô tả "sau resize/normalize" của đề:
  ảnh đã được chia 255 ở bước tiền xử lý, không fit thêm tham số nào trên test.

## 3. Giao thức thí nghiệm

- Dùng **đúng mẫu ablation của bước tiền xử lý**: 8.000 train (phân tầng theo
  `ClassId`, seed 42) và 3.000 test (lấy ngẫu nhiên, seed 42), đọc lại từ
  `report/tables/ablation_sample_manifest.csv`. Nhờ vậy mọi so sánh trong nhóm — tiền
  xử lý, đặc trưng, mô hình — đều trên cùng một tập mẫu.
- Đầu dò: K-NN `k=3`, `metric=euclidean`, `algorithm=brute`, `n_jobs=1`,
  `threadpool_limits(2)`. Đây **không phải mô hình cuối**; chỉ dùng để giữ cố định
  phần mô hình khi so sánh cách biểu diễn.
- Kết quả có thể tái tạo bằng `python scripts/feature_ablation.py`; log ở
  `logs/features_ablation.log`.

## 4. Kết quả

| Phương pháp | Số chiều | Trích đặc trưng (s) | KNN fit+predict (s) | Accuracy | Macro-F1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `pixel` (baseline) | 3072 | ~0,0 | 3,64 | 0,4767 | 0,4076 |
| `hog` | 324 | 3,14 | 0,17 | 0,7373 | 0,6920 |
| `hog_pca` | 107 | 3,23 | 0,09 | 0,7427 | 0,7002 |

## 5. Thảo luận

**HOG hơn baseline rất rõ: Accuracy +0,26 và Macro-F1 +0,28.** Khoảng cách lớn như
vậy phù hợp với giả thuyết đã nêu: pixel thô phụ thuộc mạnh vào vị trí và độ sáng
tuyệt đối của từng điểm, trong khi ảnh GTSRB đã cắt ROI nhưng vẫn còn thay đổi về ánh
sáng, độ tương phản và nền. HOG đếm phân bố hướng cạnh theo ô nên bền hơn với các
thay đổi đó. Macro-F1 thấp hơn Accuracy ở mọi phương án, cho thấy các lớp ít mẫu
(nhóm biển phụ, biển cảnh báo) khó hơn — cần xem tiếp ở phần đánh giá.

**PCA giảm 3 lần số chiều (324 → 107) mà không mất hiệu năng, thậm chí nhích lên
+0,005 Accuracy và +0,008 Macro-F1.** Chênh lệch này nhỏ hơn biến động do mẫu, nên
kết luận đúng là *không mất mát đáng kể*, chứ không phải "PCA làm mô hình tốt hơn".
Về tốc độ, K-NN brute force chạy nhanh gần gấp đôi (0,17 s → 0,09 s) vì chi phí tính
khoảng cách tỉ lệ với số chiều; với toàn bộ 39.209 ảnh train, khác biệt này sẽ lớn hơn
nhiều. Chi phí đánh đổi: trích đặc trưng tốn thêm thời gian PCA (3,14 s → 3,23 s),
nhưng đây là chi phí một lần, tách khỏi chi phí dự đoán.

**Chi phí trích HOG (~0,3 ms/ảnh) là điểm cần cân nhắc.** Với 39.209 ảnh train, HOG
mất khoảng 13–15 giây (đo được khi chạy toàn bộ dữ liệu). Với K-NN, chi phí dự đoán
chiếm ưu thế hơn nhiều, nên HOG + PCA là lựa chọn hợp lý cho biểu diễn cuối.

## 6. Hạn chế và việc nên làm tiếp

- Số 107 thành phần PCA đã được xác nhận ổn định trên cả mẫu 8.000 ảnh lẫn toàn bộ
  39.209 ảnh train (phương sai giữ được: 0,9503 so với 0,9508 trên mẫu nhỏ).
  Có thể ghi thẳng con số này vào báo cáo.
- Kết quả trên là **test con**, không phải điểm cuối. Theo khuyến nghị của nhóm, các
  lựa chọn nên được xác nhận bằng cross-validation theo `(ClassId, TrackId)` trên train
  để giảm rò rỉ giữa các frame liên tiếp trong cùng một track.
- Chưa thử HOG trên ảnh xám. Ảnh GTSRB nhiều màu (viền đỏ, nền trắng/xanh) nên màu có
  thể hữu ích; cần thí nghiệm riêng trước khi kết luận.
- Tham số HOG hiện tại (9 hướng, ô 8×8) là cấu hình phổ biến, chưa được tune. Với ảnh
  32×32, ô 8×8 chỉ chia được 4×4 ô — thử ô 4×4 (nhiều đặc trưng hơn) là hướng mở rộng
  hợp lý, nhưng phải đo cả độ chính xác và tốc độ.
- Transfer learning / mô hình pre-trained là phần mở rộng, không thay thế phần so sánh
  bằng thuật toán đã học; chưa thực hiện.
