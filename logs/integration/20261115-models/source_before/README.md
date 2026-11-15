# BTL Nhập môn Học máy (FIT4013) - Đề tài 6: Nhận dạng biển báo giao thông

## 1. Thành viên & Phân công
| Thành viên | Nhánh GitHub | Phụ trách |
| :--- | :--- | :--- |
| Vũ Khánh Hùng | `feat/hung` | EDA & Tiền xử lý ảnh (`src/preprocessing.py`) |
| Tạ Công Sơn | `son` | Trích xuất đặc trưng Pixel, HOG, PCA & Baseline (`src/features.py`) |
| Nguyễn Hoàng | `hoang` | Huấn luyện & Tối ưu K-NN, Decision Tree (`src/models.py`) |
| Đặng Quang Tiến | `tien` | Đánh giá, Confusion Matrix & Phân tích lỗi (`src/evaluate.py`); nâng cấp KNN/DT trong `src/models.py` |

## 2. Hướng dẫn cài đặt và chạy chương trình
1. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Tải GTSRB và giải nén để có `data/archive/Train.csv`, `Test.csv`,
   `Meta.csv` và các thư mục `Train/`, `Test/`, `Meta/` bên trong `archive/`.
   Không đưa ảnh hoặc cache vào Git.
3. Pipeline khung ban đầu (legacy):
   ```bash
   python main.py
   ```
   `main.py` giữ tương thích API cũ, **không phải luồng tái tạo các kết quả đã audit**:
   CV cũ chưa group theo track và nhận đặc trưng scaler/PCA tính sẵn trước CV.
   Không chạy để đánh giá lại official test hoặc thay kết quả đã khóa.
   Dùng các CLI theo track ở mục 5–7 và đánh giá saved predictions ở mục 8.

## 3. Tiền xử lý & EDA (Thành viên 1)

### Dữ liệu và tiền xử lý

- **Nguồn:** [GTSRB — IJCNN 2011](https://benchmark.ini.rub.de/gtsrb_dataset.html);
  tải bản PNG/CSV tham khảo tại [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
  Đặt dữ liệu như mục 2. **Giấy phép bản đang dùng cần xác minh trên trang tải.**
- Giữ nguyên **39.209 train / 12.630 test, 43 lớp**, không chia lại ngẫu nhiên.
- Quy trình: **RGB → crop ROI → resize 32×32 (`INTER_AREA`) → chia 255**.
  Đầu ra ảnh `float32`, `(N,32,32,3)`, `[0,1]`; nhãn nguyên 1D và 43 tên lớp.
- Không fit scaler/PCA; augmentation mặc định tắt, **không lật ảnh**.
  Cache được tạo tại `data/cache/` khi bật cache; thay nội dung ảnh cần tạo lại cache.

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

## 7. Nâng cấp KNN/DT — Đặng Quang Tiến (Thành viên 4)

Logic đã được tích hợp trực tiếp vào [src/models.py](src/models.py), không còn mô-đun
`src/model_upgrade.py` song song:

- `expanded_knn_candidates`, `evaluate_knn_cv`: grid 28 cấu hình, k đến31; tái sử dụng
  neighbor queries theo metric/fold, batch có giới hạn, xử lý ties khớp sklearn.
- `regularized_dt_candidates`, `evaluate_dt_cv`: 11 cấu hình leaf size/class weight/pruning.
- `select_best`, `refit_model`: chọn mean macro-F1 deterministic, refit chỉ train.
- API KNN/DT gốc vẫn tương thích. `knn_candidates` cũ giữ nguyên parameter grid;
  `src/decision_tree.py` giữ helper/protocol DT lịch sử, không tạo circular import.
- Giữ nguyên HOG324, preprocessing và features người1/2; scaler fit fold-train trên
  đúng frozen 3fold manifest (39.209 ảnh/1.307 tracks). Chỉ học máy truyền thống.

| HOG model | Baseline CV macro-F1 | Selected CV macro-F1 | Selected |
|---|---:|---:|---|
| KNN | 0,809718 | 0,812096 | k15 / distance / Manhattan |
| DT | 0,614898 | 0,621998 | entropy / depth20 / split5 / ccp_alpha0,0005 |

Đây là **CV trên train**, chưa có official-test score cho model nâng cấp. OOF selected
có selection bias, không phải nested/unbiased estimate; SD ba folds không phải CI.
[Báo cáo nâng cấp](report/notes_model3_upgrade.md), [notebook03](notebooks/03_model3_upgrade.ipynb),
[bảng](report/tables/model3_upgrade-full-20261115_summary.csv), [logs](logs/model3_upgrade/).

```powershell
# Chỉ khi chủ ý chạy thí nghiệm mới; run-id phải mới, không ghi đè lịch sử:
& data/.venv-model3/Scripts/python.exe scripts/train_model3_upgrade.py preflight --dataset-dir data --run-id upgrade-preflight-new
& data/.venv-model3/Scripts/python.exe scripts/train_model3_upgrade.py tune --dataset-dir data --run-id upgrade-full-new
```

CLI chỉ có `preflight|smoke|tune`, không có `test`. Môi trường riêng dùng Python3.12;
reproducibility pins ở [scripts/requirements_model3_upgrade.txt](scripts/requirements_model3_upgrade.txt),
không thay requirements chung. Mỗi run mới hash mã nguồn canonical đang dùng.

## 8. Đánh giá & phân tích lỗi — Đặng Quang Tiến (Thành viên 4)

Logic nằm trực tiếp trong [src/evaluate.py](src/evaluate.py):
`validate_predictions`, `summarize_predictions`, `paired_comparison`,
`group_error_summary`, `select_error_examples`. Không còn mô-đun
`src/evaluation_saved.py` riêng. Hàm `evaluate_and_analyze` giữ tương thích main.py;
luồng đánh giá audited không gọi adapter inference đó.

- Kiểm tra CSVRow/Path/ClassId với CSV thật, official-test claims và OOF manifest.
- Accuracy, macro Precision/Recall/F1 đủ43 lớp, balanced accuracy; confusion count và
  row-normalized, per-class, cặp nhầm, nhóm lỗi theo EDA và so sánh đúng/sai ghép cặp.
- Đã đánh giá lại **sáu bộ predictions lịch sử**, không load joblib/fit/predict mới.
  KNN HOG test Accuracy0,818369/F1 0,789750; DT HOG0,656770/0,594987.
- OOF nâng cấp tách khỏi official test. Ảnh lỗi chọn xác định bằng hash, không đại diện
  ngẫu nhiên hay bằng chứng nhân quả. Audit phát hiện8 cặp ảnh trùng byte train–test;
  giữ nguyên official split, không loại ảnh để làm đẹp điểm.
- [Báo cáo đánh giá](report/notes_evaluation_tien.md), [notebook04](notebooks/04_evaluation_tien.ipynb),
  [12 bảng/20 hình](report/evaluation/tien-evaluation-20261115/provenance.json), [logs](logs/evaluation/).

```powershell
# Chỉ đọc predictions và artifacts đã lưu; run-id mới để không overwrite:
& data/.venv-model3/Scripts/python.exe scripts/evaluate_saved_models.py --dataset-dir data --upgrade-run report/model3_upgrade/upgrade-full-20261115 --run-id evaluation-new
# Tests synthetic của phần tích hợp, không train/inference trên dataset:
& data/.venv-model3/Scripts/python.exe -B -m unittest src.tests.test_model_upgrade src.tests.test_model_upgrade_runner src.tests.test_evaluation_saved src.tests.test_evaluation_runner -v
# Audit run cũ bằng đúng bản nguồn trước refactor, báo drift hiện tại riêng:
& data/.venv-model3/Scripts/python.exe scripts/check_model3_artifacts.py --run-dir report/model3_upgrade/upgrade-full-20261115 --source-root logs/integration/20261115-canonical/source_before
```

Không truyền `--source-root` thì checker yêu cầu source hiện tại khớp hash lúc train;
không sửa hash trong config cũ để vượt kiểm tra. Bản source snapshot là bằng chứng
lịch sử, **không phải mã ứng dụng hiện tại**.

### Cấu trúc và quy tắc lưu kết quả

| Vị trí | Vai trò |
|---|---|
| `src/preprocessing.py`, `src/features.py` | Tiền xử lý/đặc trưng người1/2, giữ nguyên |
| `src/models.py`, `src/decision_tree.py` | Logic model canonical và helper DT lịch sử |
| `src/evaluate.py` | Metrics, validation, confusion và error analysis canonical |
| `scripts/` | CLI train/export, xuất hình, tạo notebook, kiểm tra; không chứa bản model core thứ hai |
| `notebooks/03_model3_upgrade.ipynb`, `04_evaluation_tien.ipynb` | Notebook kết quả đã thực thi; Run All chỉ đọc artifacts |
| `report/model3_upgrade/`, `report/evaluation/` | Run artifacts/provenance; giữ nguyên hash và kết quả lịch sử |
| `report/tables/`, `report/figures/` | CSV/PNG theo run-id, không overwrite output thành viên khác |
| `logs/model3_upgrade/`, `logs/evaluation/`, `logs/integration/` | Log thực nghiệm, verification và bằng chứng refactor |
| `data/models/` | Model bundles local tin cậy, không đưa vào Git |

Việc tích hợp ngày **2026-11-15** chỉ chuyển logic/imports, không chạy lại full training
hoặc official test. Log mới và bản executed notebook được lưu riêng tại
[logs/integration/20261115-canonical/](logs/integration/20261115-canonical/), không ghi đè
notebook/log/kết quả cũ. Giữ nguyên lỗi sai số PCA float32 người2 nếu vẫn xuất hiện;
không tuyên bố toàn suite pass. **Verification sau tích hợp:** model tests17/17,
evaluation18/18 pass; toàn suite92/93, chỉ lỗi PCA cũ. Notebook03/04 chạy7/9 code cells,
đều0 lỗi; bản gốc không bị ghi đè. Đối chiếu bằng API canonical cho sáu CSV official-test
và bốn CSV OOF khớp bảng đã lưu; không có inference mới. Chi tiết bàn giao ở
[report/handoff_tien.md](report/handoff_tien.md).
