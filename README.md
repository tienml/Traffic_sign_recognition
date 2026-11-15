# Nhận dạng biển báo giao thông — FIT4013

Bài tập lớn Nhập môn Học máy, đề tài6. Phân loại **43 lớp GTSRB** bằng
Pixel/HOG/PCA kết hợp **KNN và Decision Tree**, không dùng học sâu.
Bài toán nhận dạng biển đã định vị, không phát hiện biển trong toàn cảnh.

## 1. Thành viên và phân công

| Thành viên | Nhánh | Phụ trách |
|---|---|---|
| Vũ Khánh Hùng | `feat/hung` | EDA, tiền xử lý — [preprocessing.py](src/preprocessing.py) |
| Tạ Công Sơn | `son` | Pixel/HOG/PCA, baseline — [features.py](src/features.py) |
| Nguyễn Hoàng | `hoang` | Huấn luyện KNN/DT — [models.py](src/models.py), [decision_tree.py](src/decision_tree.py) |
| Đặng Quang Tiến | `tien` | Đánh giá/phân tích lỗi — [evaluate.py](src/evaluate.py); mở rộng tuning KNN/DT trong phần mô hình người3 |

Toàn bộ phần mô hình được tổ chức chung dưới **knn/dt**, không tách folder nâng cấp.
Chi tiết hỗ trợ AI và trách nhiệm cá nhân: [AI_USAGE.md](AI_USAGE.md).

## 2. Dữ liệu và cài đặt

- Nguồn: [GTSRB/IJCNN2011](https://benchmark.ini.rub.de/gtsrb_dataset.html),
  [bản PNG/CSV tham khảo](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
  Giấy phép bản local cần xác minh trước tái phân phối.
- Giữ official split **39.209 train / 12.630 test**,43 lớp; không chia lại test.
- CLI mô hình nhận cả `data/Train.csv` và `data/archive/Train.csv`, kèm Test.csv,
  Meta.csv, Train/, Test/, Meta/. Nếu có cả hai layout, truyền `--dataset-dir` rõ ràng.
  Loader tiền xử lý/notebook EDA gốc dùng layout `data/archive/`.
- Không đưa ảnh, cache, venv hoặc model bundles vào Git.

Từ root repo, dùng Python≥3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"
```

Môi trường thực nghiệm HOG/đánh giá đã chuẩn bị tại `data/.venv-model3` (Python3.12).
Pins để tạo môi trường riêng: [requirements_models.txt](scripts/requirements_models.txt);
không cài chồng hoặc đổi requirements chung để che khác biệt môi trường.

## 3. Pipeline và kết quả

**RGB → crop ROI → resize32×32 AREA → /255 → đặc trưng → mô hình.**
Ảnh float32, không lật/augmentation mặc định. Scaler/PCA fit fold-train khi CV;
fit toàn train khi refit, test chỉ transform. Ba frozen folds theo `(ClassId,TrackId)`,
seed42; chọn mean macro-F1 trên train.

| Đặc trưng | Số chiều | Xử lý |
|---|---:|---|
| Pixel | 3072 | Duỗi pixel [0,1], không scaler thêm |
| HOG | 324 | 9 hướng, cell8×8, block2×2, L2-Hys + StandardScaler |
| HOG+PCA | 107 ở full fit cũ | HOG + StandardScaler + PCA95%; số chiều phụ thuộc fold/fit |

### Official-test lịch sử

| Model | Pixel Accuracy / F1 macro | HOG Accuracy / F1 macro | HOG+PCA Accuracy / F1 macro |
|---|---:|---:|---:|
| KNN | 0,557245 / 0,493948 | **0,818369 / 0,789750** | 0,796675 / 0,763063 |
| DT | 0,635313 / 0,563179 | **0,656770 / 0,594987** | 0,591607 / 0,536760 |

### Tuning HOG mở rộng trong cùng phần mô hình

| Model | Baseline CV F1 | Selected CV F1 | Cấu hình selected |
|---|---:|---:|---|
| KNN | 0,809718 | 0,812096 | k15 / distance / Manhattan |
| DT | 0,614898 | 0,621998 | entropy / depth20 / split5 / ccp_alpha0,0005 |

Kết quả tuning chỉ là **train-CV**, chưa có official-test score cho models mới.
OOF selected có selection bias; SD3fold không phải CI. Test đã từng được EDA thăm dò,
không phải holdout hoàn toàn mù. Audit phát hiện8 cặp ảnh trùng byte train–test;
không loại ảnh hoặc đổi benchmark. Phân tích nhóm/ảnh lỗi không chứng minh nhân quả.

## 4. Cách chạy

### Đọc kết quả đã có — không train lại

Mở notebooks trong VS Code và chọn đúng kernel:
- [01_eda.ipynb](notebooks/01_eda.ipynb): dữ liệu và giả thuyết EDA.
- [02_preprocessing_ablation.ipynb](notebooks/02_preprocessing_ablation.ipynb): ablation tiền xử lý.
- [03_models.ipynb](notebooks/03_models.ipynb): kết quả KNN/DT, grid/CV/OOF và timing.
- [04_evaluation_tien.ipynb](notebooks/04_evaluation_tien.ipynb): metrics/confusions/phân tích lỗi.

Notebook03/04 chỉ đọc saved artifacts khi Run All. Notebook01/02 có thăm dò dữ liệu/test
và chi phí thực thi riêng, không xem đó là điểm benchmark cuối.

### Chạy thí nghiệm mới có chủ ý

```powershell
# HOG KNN/DT, dùng lại frozen train folds; run-id phải mới ở cả hai thuật toán:
python scripts/train_models.py preflight --dataset-dir data --run-id hog-preflight-new
python scripts/train_models.py tune --dataset-dir data --run-id hog-full-new
# Xuất bảng/hình từ kết quả đã lưu; thêm --export-id mới nếu đã export:
python scripts/summarize_models.py --knn-run report/knn/hog-full-new --dt-run report/dt/hog-full-new
# Đánh giá bằng saved predictions, không load/fit/predict model:
python scripts/evaluate_saved_models.py --dataset-dir data --knn-run report/knn/hog-full-20261115 --dt-run report/dt/hog-full-20261115 --run-id evaluation-new
# Kiểm thử:
python -B -m unittest discover -s src/tests -v
```

CLI gốc [train_knn.py](scripts/train_knn.py)/[train_dt.py](scripts/train_dt.py) và exporters
vẫn tương thích; protocol chi tiết trong báo cáo. **Không chạy lại official test cho
selection đã có claim hoặc xóa claim để chạy lại.** CLI `train_models.py` không có mode test.

[main.py](main.py) là pipeline khung legacy: CV chưa group track, scaler/PCA được tính
trước CV. Không dùng `python main.py` để tái tạo các điểm đã audit hoặc inference test lần nữa.

## 5. Cấu trúc và tài liệu

| Vị trí | Nội dung |
|---|---|
| [src/](src/) | Tiền xử lý, đặc trưng, models, evaluate và tests |
| [scripts/](scripts/) | CLI train/export/verify, hình và notebook generators |
| [report/notes_knn.md](report/notes_knn.md), [notes_dt.md](report/notes_dt.md) | Toàn bộ phần mô hình: benchmark gốc và tuning mở rộng |
| [report/knn/](report/knn/), [report/dt/](report/dt/) | Run configs, CV, OOF, selections và audit dữ liệu |
| [report/notes_evaluation_tien.md](report/notes_evaluation_tien.md), [report/evaluation/](report/evaluation/) | Báo cáo đánh giá và provenance |
| [report/tables/](report/tables/), [report/figures/](report/figures/) | CSV/PNG; bảng so sánh chung có tiền tố models_hog-* |
| [logs/knn/](logs/knn/), [logs/dt/](logs/dt/), [logs/evaluation/](logs/evaluation/) | Console, môi trường và kiểm thử thực chạy |
| [logs/integration/](logs/integration/) | Snapshot nguồn và bản đồ chuyển đường dẫn, không phải code runtime |
| `data/models/knn/`, `data/models/dt/` | Joblib local, đã ignore; chỉ nạp nguồn tin cậy |

EDA: [báo cáo tiền xử lý](report/notes_eda_preprocessing.md).
Bàn giao/lệnh audit: [handoff_tien.md](report/handoff_tien.md).
Kiểm tra sau gộp: **38/38 focused tests**, toàn suite **95/96**, chỉ lỗi PCA float32
có sẵn tại [src/tests/test_features.py:149](src/tests/test_features.py#L149), không sửa
source/tolerance người2. Notebook03/04 chạy7/9 code cells,0 lỗi; số liệu/artifacts cũ giữ nguyên.
[Log kiểm chứng và bản đồ chuyển đường dẫn](logs/integration/20261115-models/).
