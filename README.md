# BTL Nhập môn Học máy (FIT4013) - Đề tài 6: Nhận dạng biển báo giao thông

## 1. Thành viên & Phân công
| Thành viên | Nhánh GitHub | Phụ trách |
| :--- | :--- | :--- |
| Vũ Khánh Hùng | `feat/hung` | EDA & Tiền xử lý ảnh (`src/preprocessing.py`) |
| Tạ Công Sơn | `son` | Trích xuất đặc trưng Pixel, HOG, PCA & Baseline (`src/features.py`) |
| Thành viên 3 | `feat/3-modeling` | Huấn luyện & Tối ưu K-NN, Decision Tree (`src/models.py`) |
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
