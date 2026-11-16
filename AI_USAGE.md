# Nhật ký sử dụng AI

AI hỗ trợ viết code, phân tích, soạn tài liệu và thực thi kiểm tra. Kết quả trợ lý chạy
**không phải bằng chứng thành viên đã tự thực hiện hoặc hiểu mọi dòng**. Mỗi thành viên
cần đọc, xác minh, giải thích và chịu trách nhiệm về phần đóng góp.

## 1. Phạm vi hỗ trợ

| Thành viên | Công cụ | Nội dung có AI hỗ trợ |
|---|---|---|
| Cả nhóm | Gemini | Khởi tạo cấu trúc và khung I/O: main.py, config.py |
| Vũ Khánh Hùng | Claude Code | EDA, loader/cache/logging, tiền xử lý, ablation, tests, notebook01/02 và báo cáo EDA |
| Tạ Công Sơn | Claude Code | Pixel/HOG/PCA, FeatureExtractor, chống rò rỉ, tests và feature ablation |
| Nguyễn Hoàng | Claude Code | KNN/DT theo track, CV/tuning, khóa selection/test claim, lưu models, tests và báo cáo mô hình |
| Đặng Quang Tiến | Claude Code | Mở rộng tuning KNN/DT trong phần người3; evaluate.py, confusion/per-class/paired/group errors, audit trùng ảnh, notebook03/04; gộp cấu trúc và tài liệu |

Mô hình gốc của Nguyễn Hoàng; phần mở rộng tìm tham số do **Đặng Quang Tiến (người4)** thực hiện.
Tất cả nội dung mô hình hiện gộp trong models.py, báo cáo/logs/runs **knn/dt**, không tách folder nâng cấp.

## 2. Bằng chứng và giới hạn

- **EDA/đặc trưng:** [báo cáo EDA](report/notes_eda_preprocessing.md), notebook01/02 và logs.
  Các ablation dùng test con là thăm dò, không phải điểm cuối; không coi giải thích không lật ảnh là
  thí nghiệm mới. HOG/PCA người2 được giữ nguyên trong lượt tối ưu mô hình.
- **KNN/DT:** [notes_knn.md](report/notes_knn.md), [notes_dt.md](report/notes_dt.md),
  [notebook03](notebooks/03_models.ipynb), [logs KNN](logs/knn/), [logs DT](logs/dt/).
  HOG test lịch sử: KNN Accuracy0,818369/F1 0,789750; DT0,656770/0,594987.
  Tuning HOG mở rộng: CV F1 KNN0,809718→0,812096; DT0,614898→0,621998,
  **chưa có official-test score của models mới**. Full DT gốc chạy144 CV fits+3refit.
- **Đánh giá:** [notes_evaluation_tien.md](report/notes_evaluation_tien.md),
  [notebook04](notebooks/04_evaluation_tien.ipynb), [logs](logs/evaluation/).
  Đối chiếu sáu predictions lịch sử và bốn OOF CSV, không load model/fit/predict mới.
  Đánh giá đã xuất12 bảng/20 hình; train-CV xuất8 bảng/6 hình.
- **Kiểm chứng sau gộp:** model20/20 và evaluation18/18 pass; toàn suite95/96,
  chỉ lỗi sai số PCA float32 có sẵn ở src/tests/test_features.py:149. Notebook03/04 chạy7/9 cells,0 lỗi. Không sửa code/tolerance người2
  hoặc mô tả toàn suite là pass. Chi tiết lỗi cũ/smoke/regressions và giới hạn logs nằm trong báo cáo;
  có console lịch sử legacy encoding hoặc thiếu console do quyền sao chép, không giả lập/chạy lại để lấp.
- **Tổ chức lại ngày2026-11-15:** logic model/evaluation đặt trực tiếp trong src/models.py/src/evaluate.py;
  artifacts/log/model bundles chuyển vào knn/dt, nội dung tuning gộp vào hai báo cáo mô hình.
  Hashes/config/claim/kết quả gốc giữ nguyên, dùng bản đồ relocation và source archive để kiểm chứng;
  không ngụy tạo việc source mới đã tạo ra kết quả cũ. Notebook03 đổi trình bày/đường dẫn,
  notebook04 cập nhật resolver; bản trước chuyển được lưu trong lịch sử kiểm chứng.
  [Bằng chứng gộp và verification mới](logs/integration/20261115-models/).

## 3. Trách nhiệm và sử dụng dữ liệu

- Test đã từng được EDA thăm dò; selected OOF dùng cùng folds tuning nên có selection bias,
  không phải nested/unbiased estimate. Một seed/ba folds, SD không phải CI/significance.
- Nhóm lỗi và ảnh minh họa chỉ mô tả, không chứng minh nguyên nhân mờ/tối/crop.
  Trùng byte train–test không loại trừ near-duplicate; giữ nguyên official split.
- Nguồn GTSRB/PNG-CSV được dẫn trong [README](README.md); giấy phép bản local cần xác minh
  trước tái phân phối. Không đưa ảnh/cache/models vào Git hoặc gửi dữ liệu dự án ra dịch vụ ngoài.
- Việc tích hợp/gộp không retrain, retune hoặc official-test inference mới; không xóa claim,
  không commit/push tự động. Các thành viên tự xác minh trước khi nộp.
