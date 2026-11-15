# Đặng Quang Tiến — phạm vi hoàn thiện đánh giá

## Người phụ trách

**Đặng Quang Tiến (người 4)** phụ trách phần nâng cấp mô hình và phần đánh giá/phân tích lỗi. Nguyễn Hoàng (người 3) giữ nguyên đóng góp KNN/Decision Tree gốc. EDA/tiền xử lý của người1 và đặc trưng của người2 không thay đổi. Trợ lý AI hỗ trợ triển khai/thực thi, không thay xác minh cá nhân của người học.

## Phạm vi đã cố định trước khi tổng hợp

- Chỉ học máy truyền thống, không mạng nơ-ron.
- Đợi full nâng cấp và verification hoàn tất rồi mới chạy đánh giá chính thức.
- Bản triển khai đầu bổ sung module riêng để bảo toàn nguồn cũ. Theo yêu cầu tổ chức lại ngày2026-11-15, logic hiện đã chuyển trực tiếp vào `src/evaluate.py`, nâng cấp model vào `src/models.py`; hai module bổ sung đã bỏ. Giữ tương thích hàm `evaluate_and_analyze` và API model cũ; không đổi source người1/2, main.py, dữ liệu, kết quả/notebook/log/claim lịch sử. Bản nguồn trước chuyển và kiểm tra mới lưu riêng trong `logs/integration/20261115-canonical/`.
- Đọc sáu predictions lịch sử KNN/DT × Pixel/HOG/HOG+PCA, xác minh CSVRow/Path/ClassId với Test.csv hiện tại và selection/claim/metrics; không nạp model, fit, predict hoặc official-test inference mới.
- Lưu Accuracy và Precision/Recall/F1 macro đủ43lớp, balanced accuracy; confusion count/row-normalized43×43 cho cả6 pipelines, bảng từng lớp và tên lớp, cặp nhầm có hướng, nhóm lỗi theo kích thước/loại biển, đếm đúng/sai ghép cặp KNN–DT.
- Chọn ví dụ ảnh lỗi của HOG winner từ4 cặp nhầm thường gặp,2 ảnh/cặp theo SHA256(seed42|Path); hiển thị ảnh gốc/ROI và nhãn thật/dự đoán. Không suy blur/tối/lỗi crop là nguyên nhân chỉ từ caption.
- Audit trùng byte train/test bằng SHA256, không loại ảnh, không thay split/nhãn hoặc tính lại một benchmark đã lọc để làm đẹp điểm.
- Bảng OOF nâng cấp để riêng, không trộn với official test. Chọn trên cùng folds có selection bias, không gọi nested/unbiased evaluation, không suy điểm test của model mới.

## Sản phẩm và vị trí

- Source canonical: `src/evaluate.py`; runners/figures/notebook/verification dưới `scripts/`, chỉ I/O/visualization/orchestration, không nhân đôi core metrics.
- Tests mới synthetic: `src/tests/test_evaluation_saved.py`, `src/tests/test_evaluation_runner.py`.
- Runs/provenance: `report/evaluation/<run-id>/`; đầu vào mô hình từ `report/knn/hog-full-20261115` và `report/dt/hog-full-20261115`, không còn folder nâng cấp riêng.
- Log console, môi trường, preservation/test/notebook: `logs/evaluation/`.
- Bảng: `report/tables/evaluation_<run-id>_*`.
- Hình: `report/figures/evaluation_<run-id>_*`, có light/dark cho numeric figures.
- Notebook: `notebooks/04_evaluation_tien.ipynb`, chạy đọc artifacts để tránh retrain ngoài ý muốn.
- Báo cáo kết quả đã tạo từ số liệu thực tế tại [notes_evaluation_tien.md](notes_evaluation_tien.md); tài liệu phạm vi này mô tả protocol, không thay bảng kết quả.

## Trạng thái hiện tại

**Đã hoàn tất và xác minh.** Full upgrade KNN/DT, export và notebook03 đã qua audit. Đánh giá từ sáu bộ official-test predictions và OOF nâng cấp đã xuất **12 bảng/20 hình**, notebook04 thực thi9 code cells không lỗi. Focused evaluation **16/16 pass**, toàn suite **87/88 pass**, chỉ còn lỗi float32 PCA người2 có sẵn. Xem [completion](../logs/evaluation/orchestration-tien-evaluation-20261115/completion.json), [verification](../logs/evaluation/verification-tien-evaluation-20261115-final/verification.json) và [báo cáo kết quả](notes_evaluation_tien.md). Không có official-test inference mới; việc tích hợp source model/evaluation ngày2026-11-15 được ghi riêng, không thay artifacts thực nghiệm lịch sử hoặc phần người1/2.

**Kiểm tra sau tích hợp canonical 2026-11-15:** model17/17 và evaluation18/18 pass;
toàn suite92/93, chỉ lỗi PCA cũ. Notebook03/04 chạy7/9 cells,0 lỗi, không overwrite
notebook gốc; API canonical đối chiếu sáu CSV official-test và bốn CSV OOF khớp
bảng đã lưu. Log/refactor audit mới tách khỏi log thực nghiệm ban đầu tại
[logs/integration/20261115-canonical](../logs/integration/20261115-canonical/).

**Sau gộp folder vào phần KNN/DT:** model20/20, evaluation18/18 pass; suite95/96 chỉ lỗi PCA cũ.
Notebook03_models/04 chạy7/9 cells không lỗi. Config/provenance/OOF/model bytes được giữ nguyên;
resolver kiểm tra relocation theo đúng hash lịch sử. [Bằng chứng](../logs/integration/20261115-models/).

## Giới hạn cần giữ

Test từng được EDA thăm dò, không hoàn toàn mù. Exact-duplicate không loại trừ near-duplicate. Phân tích nhóm không kiểm soát class/track confounding nên không suy nhân quả hoặc significance. Không phát hiện biển trong toàn cảnh hoặc cam kết realtime/an toàn triển khai. Giữ nguyên lỗi float32 PCA phần người2 nếu vẫn tồn tại; không sửa tolerance/code phần đó để làm đẹp full-suite. Không commit/push hoặc xuất bản nội dung/dữ liệu lên dịch vụ ngoài.
