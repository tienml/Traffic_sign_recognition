# Bàn giao mô hình và đánh giá

## Nội dung và đóng góp

- Mô hình người3: Nguyễn Hoàng thực hiện KNN/DT gốc; Đặng Quang Tiến mở rộng tuning,
  nay gộp trong [notes_knn.md](notes_knn.md), [notes_dt.md](notes_dt.md) và
  [notebook03](../notebooks/03_models.ipynb). Logic tại src/models.py;
  helper DT gốc tại src/decision_tree.py, APIs cũ giữ tương thích.
- Đánh giá người4: Đặng Quang Tiến, [notes_evaluation_tien.md](notes_evaluation_tien.md),
  [notebook04](../notebooks/04_evaluation_tien.ipynb), src/evaluate.py.
- Không đổi source/tests/preprocessing/features người1/2, dữ liệu, benchmark gốc hoặc test claims.
  AI hỗ trợ không thay trách nhiệm cá nhân; xem [AI_USAGE](../AI_USAGE.md).

## Thư mục kết quả

| Phần | Runs | Logs | Models local |
|---|---|---|---|
| KNN | report/knn/hog-full-20261115 | logs/knn/hog-full-20261115 | data/models/knn/hog-full-20261115/hog.joblib |
| DT | report/dt/hog-full-20261115 | logs/dt/hog-full-20261115 | data/models/dt/hog-full-20261115/hog.joblib |
| Đánh giá | report/evaluation/tien-evaluation-20261115 | logs/evaluation/tien-evaluation-20261115 | Không nạp model |

Config/selection/audit shared của run cũ giữ nguyên bytes trong cả KNN/DT; consumers
chỉ lấy đúng thuật toán, không nhân đôi kết quả. Run-id bên trong metadata là tên lúc
thực nghiệm, không phải tên thư mục hiện tại. Bảng/hình chung có tiền tố
`models_hog-full-20261115_*` tại tables/figures; evaluation giữ nguyên tiền tố cũ.

[Bản đồ relocation](../logs/integration/20261115-models/relocations.json) lưu old→new paths
và source archives. File cũ được resolve theo exact SHA256; không sửa hash trong
config/provenance để che source/path drift. Đây là bằng chứng lịch sử, không phải
một bản code runtime thứ hai. Notebook03 trước chuyển lưu ở
[history](../logs/knn/hog-full-20261115/history/03_model3_upgrade.ipynb).

## Các lệnh chính

Dùng Python có đủ dependencies; môi trường đã chuẩn bị là data/.venv-model3.
Pins: [requirements_models.txt](../scripts/requirements_models.txt).

```powershell
# Chạy train-CV mới có chủ ý, không có official-test mode:
python scripts/train_models.py preflight --dataset-dir data --run-id hog-preflight-new
python scripts/train_models.py tune --dataset-dir data --run-id hog-full-new
# Export saved run; export-id mới nếu đã có sản phẩm:
python scripts/summarize_models.py --knn-run report/knn/hog-full-new --dt-run report/dt/hog-full-new
# Audit completed run, dùng đúng bản nguồn lúc thực nghiệm:
python scripts/check_model_artifacts.py --knn-run report/knn/hog-full-20261115 --dt-run report/dt/hog-full-20261115 --source-root logs/integration/20261115-canonical/source_before
# Đánh giá chỉ từ saved predictions:
python scripts/evaluate_saved_models.py --dataset-dir data --knn-run report/knn/hog-full-20261115 --dt-run report/dt/hog-full-20261115 --run-id evaluation-new
python -B -m unittest discover -s src/tests -v
```

Run-id/output mới phải chưa tồn tại; không overwrite/xóa test claims. Không chạy
one-shot finalize/complete vào run đã hoàn thành. Verifiers nhận preservation snapshot
và danh sách authorized changes; khi audit relocation cần truyền `--relocations`.
main.py là legacy, chưa grouped CV và fit scaler/PCA trước CV; không tái tạo điểm audited bằng main.

## Kết quả và giới hạn cần trình bày

- CV mean F1: KNN0,809718→0,812096 (k15/distance/Manhattan),
  DT0,614898→0,621998 (alpha0,0005). Không thay official-test scores cũ.
- Mean fold khác pooled OOF; selected OOF có selection bias, SD3fold không phải CI.
- H2/H3 và gallery chỉ mô tả, chưa nhân quả/significance. Gallery hash-ranked,
  không phải random representative sample. Exact8 cặp trùng byte không loại trừ near-duplicates.
- Test đã từng được EDA xem. Chưa có official-test score của models mới.
- Sau gộp:38/38 focused pass (model20/20, evaluation18/18), suite95/96 với đúng lỗi PCA float32 có sẵn.
  Notebook03/04 thực thi7/9cells,0 lỗi. Logs cũ giữ nguyên; logs sau gộp tại
  [logs/integration/20261115-models](../logs/integration/20261115-models/).
- Không train/inference mới khi gộp; models/results chuyển byte-preserving. Không commit/push.
