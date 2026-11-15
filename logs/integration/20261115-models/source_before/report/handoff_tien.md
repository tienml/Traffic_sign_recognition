# Đặng Quang Tiến — hướng dẫn bàn giao phần bổ sung

## Phân công được ghi nhận

- Nguyễn Hoàng (người3): phần KNN/Decision Tree gốc và kết quả lịch sử.
- Đặng Quang Tiến (người4): nâng cấp tìm tham số KNN/regularization DT và phần đánh giá, confusion matrix, phân tích lỗi, logs/report/notebooks bổ sung.
- Người1/2: EDA/tiền xử lý và trích xuất đặc trưng được giữ nguyên. Mã nguồn, tests, notebook và artifacts lịch sử của các phần này không sửa.
- AI hỗ trợ triển khai/thực thi; không thay trách nhiệm tự kiểm tra và giải thích của thành viên.

## Đọc theo thứ tự

1. [Báo cáo nâng cấp mô hình](notes_model3_upgrade.md): giao thức cố định, grid, baseline cùng môi trường, kết quả CV/OOF, timing và giới hạn.
2. [Notebook03](../notebooks/03_model3_upgrade.ipynb): output từ full-run upgrade, không train lại khi Run All.
3. [Báo cáo đánh giá của Tiến](notes_evaluation_tien.md): sáu pipeline official-test lịch sử, confusion/per-class/paired/group diagnostics, ảnh lỗi và audit split.
4. [Notebook04](../notebooks/04_evaluation_tien.ipynb): output đánh giá saved artifacts, không inference lại.
5. [Nhật ký AI](../AI_USAGE.md): giữ các khai báo của thành viên khác, nối thêm phạm vi của Tiến.

**Các bước trên đã hoàn tất và có bằng chứng:** notebook03 có7 code cells và notebook04 có9 code cells, đều đã thực thi không lỗi. Phần nâng cấp có8 bảng/6 hình full-run; phần đánh giá có12 bảng/20 hình. Tests mới nâng cấp14/14 và evaluation16/16 pass; toàn suite87/88, giữ nguyên một lỗi PCA người2. Trạng thái chi tiết và kiểm tra bảo toàn phần cũ nằm trong completion/verification logs dưới đây.

## Bằng chứng thực thi

- Full train-CV: `report/model3_upgrade/upgrade-full-20261115/selection.json` và `logs/model3_upgrade/upgrade-full-20261115/console.log`.
- Audit/export/notebook upgrade: `logs/model3_upgrade/upgrade-full-20261115/completion.json`; kiểm thử chi tiết ở `logs/model3_upgrade/verification-upgrade-full-20261115-final/`.
- Evaluation: `report/evaluation/tien-evaluation-20261115/provenance.json` và `logs/evaluation/tien-evaluation-20261115/`.
- Verification evaluation: `logs/evaluation/verification-tien-evaluation-20261115-final/`.
- Trạng thái toàn bàn giao: `logs/evaluation/orchestration-tien-evaluation-20261115/completion.json`.
- Các bảng và hình mới có tiền tố `model3_upgrade-full-20261115_*` hoặc `evaluation_tien-evaluation-20261115_*`. Không đổi tên/ghi đè các bảng `knn_*`, `dt_*`, EDA hoặc feature ablation cũ.

## Cấu trúc canonical sau tích hợp 2026-11-15

- [src/models.py](../src/models.py): KNN/DT gốc cùng nâng cấp do Tiến phụ trách;
  `expanded_knn_candidates`, `regularized_dt_candidates`, `evaluate_knn_cv`,
  `evaluate_dt_cv`, `select_best`, `refit_model`. Giữ API/grid cũ, không đổi phần người1/2.
- [src/evaluate.py](../src/evaluate.py): validation saved predictions, metrics/confusion,
  per-class/paired/group error analysis và chọn ví dụ. Giữ adapter `evaluate_and_analyze`
  tương thích main.py; đánh giá audited không gọi adapter đó.
- Đã bỏ hai module bổ sung `src/model_upgrade.py`, `src/evaluation_saved.py` sau khi
  cập nhật callers/tests. Không thêm shim hoặc sao chép logic trong scripts.
- Scripts còn lại có vai trò CLI train/export/figures/notebook/verification riêng;
  các orchestration one-shot không phải lệnh chạy lại cho run đã hoàn thành.
- Source trước chuyển nằm [logs/integration/20261115-canonical/source_before/](../logs/integration/20261115-canonical/source_before/),
  chỉ để kiểm tra hash lịch sử, không phải code runtime. Log/check tích hợp ở thư mục cha.
  Không sửa config/provenance/claim để che source drift sau refactor.

## Các lệnh entry point

Dùng Python của `data/.venv-model3/Scripts/python.exe`. Môi trường mới có thể tạo riêng bằng Python3.12 và `scripts/requirements_model3_upgrade.txt`, không thay requirements chung.

- Train-CV có chủ ý: `scripts/train_model3_upgrade.py preflight|smoke|tune --dataset-dir data --run-id ten-moi`.
- Chỉ xuất artifacts upgrade: `scripts/summarize_model3_upgrade.py --run-dir report/model3_upgrade/<run-id>`; thêm `--export-id ten-moi` nếu đã export.
- Đánh giá không model: `scripts/evaluate_saved_models.py --dataset-dir data --upgrade-run report/model3_upgrade/upgrade-full-20261115 --run-id ten-moi`.
- Notebook generators: `scripts/create_model3_notebook.py` và `scripts/create_evaluation_notebook.py`, yêu cầu output mới.

CLI train upgrade không có official-test mode. Không xóa claims, chạy lại official test để lấy log hoặc ghi đè run-id cũ. Đánh giá lịch sử chỉ tính lại metrics từ predictions đã có, không phải chạy lại model.

## Kiểm chứng sau tích hợp 2026-11-15

- Tests model17/17 và evaluation18/18 pass; toàn suite92/93, đúng lỗi PCA float32 cũ.
- Notebook03 chạy7 code cells, notebook04 chạy9 code cells, đều0 error outputs;
  executed copies lưu trong tag verification mới, không sửa hai notebook gốc.
- API canonical đã tính lại sáu CSV official-test và bốn CSV OOF, metrics khớp bảng
  đã lưu; per-class/groups/confusions/paired/gallery official-test cũng được so khớp.
  Không fit model trên dataset, không nạp joblib hoặc official-test inference mới.
- Check model hashes/folds/OOF/candidates và source lịch sử từ archive đều pass;
  current-source drift được công khai, không đổi hash trong config/provenance cũ.
- Bằng chứng: [model verification](../logs/model3_upgrade/verification-canonical-20261115/verification.json),
  [evaluation verification](../logs/evaluation/verification-canonical-20261115/verification.json),
  [saved-evidence parity](../logs/integration/20261115-canonical/canonical_evidence_checks.json).
- Các bảng/hình, runs/claims và logs lịch sử được giữ nguyên; nguồn người1/2,
  main/config/requirements không đổi. Audit cuối cùng nằm trong thư mục log tích hợp.

## Điều cần giải thích khi trình bày

- Mean macro-F1 theo fold khác pooled OOF macro-F1; SD ba folds không phải khoảng tin cậy.
- Cấu hình mới được chọn trên cùng folds OOF nên vẫn selection bias; chưa nested/repeated CV hoặc CI.
- Không có điểm official test của model nâng cấp; không dùng điểm test lịch sử để chọn lại winner.
- EDA H2/H3 là giả thuyết: bảng support/kích thước và cặp nhầm chỉ mô tả, chưa chứng minh nguyên nhân lỗi.
- Ảnh lỗi được chọn tái lập bằng SHA256(seed42|Path) từ các cặp nhầm phổ biến, không đại diện ngẫu nhiên cho toàn test.
- Split chính thức giữ nguyên; hash byte phát hiện trùng chính xác nhưng không loại trừ near-duplicate.
- Test đã từng được EDA xem, không gọi là holdout hoàn toàn mù.
- Giữ nguyên lỗi PCA float32 nếu còn xuất hiện; không tuyên bố tất cả tests pass hoặc sửa người2 để làm đẹp.
- `main.py` là pipeline legacy: CV cũ không group track, scaler/PCA được tính trước CV. Không dùng đường chạy đó để tái tạo điểm audited hoặc inference official test lần nữa. Model/evaluation hiện đã tích hợp vào đúng src/models.py/src/evaluate.py; các CLI gọi trực tiếp các API canonical. Hàm legacy giữ tương thích, không đồng nghĩa giao thức main đã được sửa.
- Audit source của run cũ dùng `scripts/check_model3_artifacts.py --run-dir report/model3_upgrade/upgrade-full-20261115 --source-root logs/integration/20261115-canonical/source_before`. Checker vẫn strict nếu không truyền archive. Verification mới lưu executed notebook vào thư mục tag mới, không ghi đè notebook gốc.

Không commit/push hoặc xuất bản dữ liệu/nội dung dự án lên dịch vụ ngoài trong lượt thực hiện này.
