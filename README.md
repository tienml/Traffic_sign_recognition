# BTL Nhập môn Học máy (FIT4013) - Đề tài 6: Nhận dạng biển báo giao thông

## 1. Thành viên & Phân công
| Thành viên | Nhánh GitHub | Phụ trách |
| :--- | :--- | :--- |
| Vũ Khánh Hùng | `feat/hung` | EDA & Tiền xử lý ảnh (`src/preprocessing.py`) |
| Tạ Công Sơn | `son` | Trích xuất đặc trưng Pixel, HOG, PCA & Baseline (`src/features.py`) |
| Nguyễn Hoàng | `hoang` | Huấn luyện & Tối ưu K-NN, Decision Tree (`src/models.py`) |
| Thành viên 4 | `feat/4-evaluation` | Đánh giá, Confusion Matrix & Phân tích lỗi (`src/evaluate.py`) |

## 2. Hướng dẫn cài đặt và chạy chương trình
1. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Tải GTSRB và giải nén để có `data/archive/Train.csv`, `Test.csv`,
   `Meta.csv` và các thư mục `Train/`, `Test/`, `Meta/` bên trong `archive/`.
   Không đưa ảnh hoặc cache vào Git.
3. Chạy toàn bộ pipeline:
   ```bash
   python main.py
   ```

## 3. Tiền xử lý & EDA (Thành viên 1)

### Dữ liệu và tiền xử lý

- **Nguồn:** [GTSRB — IJCNN 2011](https://benchmark.ini.rub.de/gtsrb_dataset.html);
  tải bản PNG/CSV tham khảo tại [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
  Đặt dữ liệu như mục 2. **Giấy phép bản đang dùng cần xác minh trên trang tải.**
- Giữ nguyên **39.209 train / 12.630 test, 43 lớp**, không chia lại ngẫu nhiên.
- Quy trình: **RGB → crop ROI → resize 32×32 (`INTER_AREA`) → chia 255**.
  Đầu ra ảnh `float32`, `(N,32,32,3)`, `[0,1]`; nhãn nguyên 1D và 43 tên lớp.
- Không fit scaler/PCA; augmentation mặc định tắt, **không lật ảnh**.
  Cache nằm trong [data/cache/](data/cache/); thay nội dung ảnh cần tạo lại cache.

### Cách chạy

Dùng **Python ≥3.12**, khuyến nghị **3.13.2** đã kiểm thử. Từ gốc dự án, chạy PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"
python -m src.preprocessing
python -m unittest discover -s src/tests -v
```

Mở hai notebook dưới đây trong VS Code, chọn kernel của `venv`, rồi **Restart & Run All**:
- [01_eda.ipynb](notebooks/01_eda.ipynb): nguồn dữ liệu và 8 mục EDA.
- [02_preprocessing_ablation.ipynb](notebooks/02_preprocessing_ablation.ipynb): so sánh
  ROI, kích thước, màu và nội suy bằng KNN trên cùng 8.000 train / 3.000 test.

Để chạy kiểm tra tự động, hai notebook và lưu log:

```bash
python scripts/verify_data_prep.py
```

Script chỉ kiểm tra `main.py` tới đầu `build_features` rồi dừng, **không chạy GridSearchCV đầy đủ**.

### Kết quả và lưu ý

- Hai notebook đã lưu output; hình tại [report/figures/](report/figures/), bảng tại
  [report/tables/](report/tables/), log và môi trường tại [logs/](logs/).
- [Báo cáo EDA đầy đủ](report/notes_eda_preprocessing.md): 3 giả thuyết, hình minh họa,
  các trường hợp đặc biệt và lý do không lật biển trái/phải. Khai báo AI: [AI_USAGE.md](AI_USAGE.md).
- EDA/ablation là **thăm dò test con, không phải điểm cuối**. Giữ cấu hình hiện tại;
  xác nhận lựa chọn bằng CV theo `(ClassId, TrackId)` trên train để giảm rò rỉ.
- Chỉ phân loại biển đã định vị, chưa phát hiện biển trong toàn cảnh. Không commit ảnh gốc/cache.

## 4. Biểu diễn đặc trưng (`src/features.py`)

Ba cách biểu diễn ảnh 32×32×3 đã chuẩn hóa `[0, 1]`:

| Phương pháp | Cách làm | Số chiều (32×32) |
| :--- | :--- | ---: |
| `pixel` | Baseline: duỗi vector pixel thô | 3072 |
| `hog` | HOG 9 hướng, ô 8×8, block 2×2, `L2-Hys`, chuẩn hóa StandardScaler | 324 |
| `hog_pca` | HOG như trên rồi PCA giữ 95% phương sai | 107 (xác nhận trên cả 39.209 ảnh train, 95,03% phương sai) |

- Scaler và PCA **chỉ fit trên tập train**, tập test chỉ `transform`; PCA dùng
  `random_state=42`. `FeatureExtractor.fit/transform` tách riêng để không thể fit nhầm.
- Baseline `pixel` giữ nguyên giá trị `[0, 1]`, không chuẩn hóa thêm.
- Chạy thử mô-đun và kiểm thử:

```bash
python -m src.features
python -m unittest src.tests.test_features -v
```

- Chạy thí nghiệm so sánh trên mẫu ablation 8.000/3.000 (cần cache tiền xử lý):

```bash
python scripts/feature_ablation.py
```

Kết quả ghi ở [report/tables/feature_ablation.csv](report/tables/feature_ablation.csv),
[biểu đồ](report/figures/feature_ablation.png) và log `logs/features_ablation.log`.
K-NN (k=3, euclidean) ở đây chỉ là **đầu dò** để đo tác động của biểu diễn,
không phải mô hình cuối của nhóm.

## 5. K-NN theo track — Nguyễn Hoàng (Thành viên 3)

- Entry point riêng: [scripts/train_knn.py](scripts/train_knn.py), không thay `main.py` hoặc pipeline người 1/2. Dữ liệu local hiện nằm trực tiếp `data/Train.csv`, `data/Test.csv`, `data/Meta.csv`; CLI nhận cả layout này và `data/archive/`. Hướng dẫn `main.py` ở trên vẫn dùng archive. Khi cả hai layout tồn tại phải truyền `--dataset-dir` rõ ràng.
- Full train: 39.209 ảnh, 43 lớp, 1.307 track; cùng grouped CV 3 folds seed 42 cho Pixel/HOG/HOG+PCA, scaler/PCA chỉ fit fold-train. Mỗi biểu diễn 12 candidates; winner theo macro-F1 CV được khóa trước test, không chọn lại bằng test.
- Winner **HOG + StandardScaler + KNN(k=7, distance, Manhattan)**: CV macro-F1 **0,809718 ± 0,016981**; official test 12.630 ảnh: Accuracy **0,818369**, macro-F1 **0,789750**. Test Pixel: Accuracy 0,557245 / F1 0,493948; HOG+PCA: 0,796675 / 0,763063. Test đã từng được EDA thăm dò, không mô tả là hoàn toàn chưa xem.
- [Báo cáo đầy đủ, hạn chế và trường hợp đặc biệt](report/notes_knn.md), [bảng kết quả](report/tables/knn_summary.csv), [36 candidates](report/tables/knn_candidates.csv), [hình CV/test](report/figures/knn_cv_test.png), [logs và provenance](logs/knn/).

Từ root repo, tái xuất bảng/hình **chỉ từ artifact đã lưu**, không chạy lại train/test:

```powershell
& ./.venv/Scripts/python.exe scripts/summarize_knn.py
$env:PYTHONIOENCODING = "utf-8"
& ./.venv/Scripts/python.exe -m unittest src.tests.test_models src.tests.test_knn_summary -v
```

Các chế độ `preflight`, `smoke`, `tune`, `test` và điều kiện an toàn được ghi trong báo cáo. **Không chạy lại official test cho selection đã có claim**, không xóa claim để lấy log. Helper xuất `knn_*` vào `report/tables`, hình vào `report/figures`; run gốc giữ ở `report/knn`; loader log của lượt CLI mới vào `logs/knn/<run-id>/`. Không ghi đè bảng/hình của thành viên khác.

Kiểm thử do trợ lý chạy: 12/12 focused pass; full suite 49/50 pass, còn lỗi sai số float32 cumulative PCA tại `src/tests/test_features.py:149` (giữ nguyên code/test người 2). Chi tiết và log trong báo cáo. Việc sao chép console train/test lịch sử từ thư mục tạm bị hệ thống quyền chặn; không giả lập console hoặc chạy lại mô hình. Phạm vi đã hoàn thành là KNN, không tuyên bố đã tối ưu Decision Tree hoặc hoàn thành phân tích lỗi của người 4. Khai báo hỗ trợ AI tại [AI_USAGE.md](AI_USAGE.md).

## 6. Decision Tree — Nguyễn Hoàng (Thành viên 3), đã hoàn tất full run

- Bổ sung riêng [src/decision_tree.py](src/decision_tree.py), [scripts/train_dt.py](scripts/train_dt.py), [scripts/summarize_dt.py](scripts/summarize_dt.py) và [8 tests DT](src/tests/test_decision_tree.py); không thay pipeline hoặc kết quả KNN/người khác.
- **Đã chạy đủ144 CV fits +3 refit** trên39.209 ảnh/1.307 tracks,16 candidates mỗi Pixel/HOG/HOG+PCA, seed42, chính xác frozen3fold manifest KNN. Scaler/PCA fold-local, chọn macro-F1 CV và deterministic first tie; không sampling hoặc retune theo test.
- Winner **HOG + StandardScaler + DT(entropy, depth20, min_samples_split5)**: CV macro-F1 **0,614898 ±0,011907**; official test12.630 ảnh: Accuracy **0,656770**, macro-F1 **0,594987**. Test Pixel Accuracy0,635313/F1 0,563179; HOG+PCA0,591607/0,536760. KNN HOG F1 0,789750 cao hơn DT HOG trong protocol này; không khẳng định ý nghĩa thống kê. Test từng được EDA thăm dò.
- Kiểm thử **20/20 focused pass**, full **57/58** (cùng lỗi float32 người2 giữ nguyên). Đã sửa test gate kiểm tra HOG/image contract drift trước claim/Test.csv, regression cả HOG cùng dimensions và ROI drift. Smoke đầu lỗi JSON int64 đã sửa, giữ run lỗi; không chạy lại official test để lấy logs.
- [Báo cáo DT: kết quả, so sánh, trường hợp lỗi và timing](report/notes_dt.md); [full selection](report/dt/full-dt-01/selection.json), [test metrics](report/dt/test-dt-01/test_metrics.json), [bảng full](report/tables/dt_full-dt-01_summary.csv), [hình](report/figures/dt_full-dt-01_cv_test.png), [logs](logs/dt/). Exporter đã kiểm tra12.630 predictions/method, alignment và tính lại metrics; không fit/predict hoặc ghi đè. Hình sinh được nhưng viewer trả rỗng, chưa xác nhận trực quan layout.

```powershell
& ./.venv/Scripts/python.exe -m unittest src.tests.test_decision_tree src.tests.test_models src.tests.test_knn_summary -v
# Tái xuất full artifacts đã lưu; export-id phải mới, không inference lại:
& ./.venv/Scripts/python.exe scripts/summarize_dt.py --tune-dir report/dt/full-dt-01 --test-dir report/dt/test-dt-01 --export-id full-dt-review-02
```

Full tune/test đã hoàn tất; **không chạy lại official test cho selection đã có claim hoặc xóa claim**. Tổng search+refit13246,755s (~3,68h), tuần tự1 job/2 numeric threads; timing chưa gồm I/O/extraction và chưa đo peak RAM. Khai báo AI ở cuối [AI_USAGE.md](AI_USAGE.md).
