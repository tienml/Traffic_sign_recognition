# Ghi chú EDA và tiền xử lý GTSRB

Phạm vi: Thành viên 1. Tài liệu dùng để biên soạn báo cáo và bàn giao dữ liệu cho
các thành viên đặc trưng/mô hình/đánh giá; không thay thế chương LaTeX hoàn chỉnh
hoặc kết quả huấn luyện cuối cùng. Số liệu lấy từ hai notebook đã thực thi, không
phải số liệu minh họa.

## 1. Bài toán, nguồn và giả định

- Phân loại một ảnh biển đã định vị thành một trong **43 ClassId (0–42)**.
  Không giải bài toán phát hiện biển trong toàn cảnh đường phố.
- GTSRB (German Traffic Sign Recognition Benchmark), Institut für Neuroinformatik,
  Ruhr-Universität Bochum, benchmark tại IJCNN 2011. Tham khảo Stallkamp và cộng sự,
  *The German Traffic Sign Recognition Benchmark: A multi-class classification
  competition*, [DOI](https://doi.org/10.1109/IJCNN.2011.6033395).
- [Trang gốc](https://benchmark.ini.rub.de/gtsrb_dataset.html);
  [bản PNG/CSV trên Kaggle để tham khảo tải](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
  Repo yêu cầu bố cục `data/archive/Train.csv`, `Test.csv`, `Meta.csv`, các thư mục
  `Train/`, `Test/`, `Meta/`. Bản gốc PPM không đọc trực tiếp bằng pipeline này.
  Không có hồ sơ để xác nhận nguồn tải của bản local.
- **Giấy phép: cần xác minh.** Bản local không có README/LICENSE. Lần kiểm tra
  2026-11-11 chưa truy cập được trang gốc và nội dung Kaggle không đủ xác nhận
  điều khoản. Thành viên cần kiểm tra đúng trang tải trước khi sử dụng/tái phân
  phối. Không gán giấy phép dựa trên việc dữ liệu được công khai.
- Giả định mỗi ảnh có một biển, ROI/nhãn đúng. Chỉ biển Đức, chưa chứng minh tổng
  quát sang biển Việt Nam. Đây là nghiên cứu phân loại, không phải hệ thống an
  toàn đủ điều kiện triển khai trên xe tự lái.

## 2. Phân tích dữ liệu (EDA)

Nguồn chi tiết: [01_eda.ipynb](../notebooks/01_eda.ipynb),
[eda_summary.json](eda_summary.json), [bảng EDA](tables/) và [hình](figures/).

### 2.1. Quy mô, tính toàn vẹn và mất cân bằng

- **39.209 train, 12.630 test, 43 lớp**, cùng 43 ảnh Meta chuẩn.
- Train/Test không có NaN, dòng trùng, Path trùng hay Path thiếu. Meta có 1 NaN
  ở `SignId` của ClassId 30; không ảnh hưởng ClassId/ColorId/ShapeId, không tự
  suy đoán hay loại bỏ lớp.
- Train có **1.307 track**, trung vị 30 ảnh/track: 1.306 track đủ 30 ảnh và
  `00033_00019` có 29 ảnh. Không tự thêm ảnh để đổi tổng thành 39.210.
- Train: **210–2.250 ảnh/lớp**, tỉ lệ max/min **10,71 lần**. Lớp ít nhất:
  0, 19, 37; lớp nhiều nhất: 2. Test: 60–750 ảnh/lớp, tỉ lệ **12,5 lần**.
- Accuracy chịu ảnh hưởng của lớp phổ biến. Báo cáo thêm Macro-F1 (mỗi lớp có
  trọng số ngang nhau), rồi kiểm tra recall/confusion từng lớp ở bước đánh giá.

### 2.2. Kích thước, màu và chất lượng

- Trung vị **43×43** cho cả train/test. Train: Width 25–243, Height 25–225;
  test: Width 25–266, Height 25–232 pixel.
- Cả hai chiều dưới 32: **4.766 train / 1.632 test**. Ít nhất một chiều dưới 32:
  **6.505 train / 2.292 test**. Hai định nghĩa không được đánh đồng.
- Mẫu 500 ảnh: 500 ảnh có 3 kênh uint8; đây là kiểm tra mẫu, không phải kết luận
  đã kiểm định mọi định dạng gốc.
- Mẫu 2.000 train: độ sáng grayscale trung bình **0,3182**, trung vị **0,2906**,
  phạm vi **0,0341–0,9539**. Tương phản trung vị **0,1594**, nhỏ nhất **0,0099**.
- ROI/toàn ảnh trung vị **60,34% train / 60,06% test**. Chia 255 chỉ đổi thang,
  không sửa mờ, phơi sáng hay tương phản.
- Các lớp giới hạn tốc độ có hình dáng/viền giống nhau và chữ số nhỏ. Mean image
  là gợi ý các lớp khó phân biệt, không phải bằng chứng confusion matrix.

### 2.3. Thăm dò màu trong EDA cũ

KNN k=3 trên 5.000 train / 2.000 test **đều lấy mẫu phân tầng**, seed 42:

| Biểu diễn | Số chiều | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| RGB | 3.072 | 42,95% | 36,91% |
| Grayscale | 1.024 | 40,35% | 34,38% |

RGB hơn 2,60 đpt Accuracy và 2,53 đpt Macro-F1. Kết quả ủng hộ giữ thông tin màu
trong thăm dò này, không chứng minh RGB luôn tốt hơn cho mọi mô hình.

## 3. Tiền xử lý và cách chia dữ liệu

Chuỗi cố định: **đọc RGB → crop ROI đóng có kẹp biên → INTER_AREA 32×32 →
float32/255**.

- NumPy slice loại cận phải nên crop dùng `y1:y2+1, x1:x2+1`; ROI không hợp lệ
  thì giữ toàn ảnh. Ảnh thiếu/hỏng làm dừng, không bỏ âm thầm để lệch nhãn.
- Đầu ra `load_and_preprocess(data_dir)` giữ đúng thứ tự:
  `X_train_img, X_test_img, y_train, y_test, class_names`.
- Ảnh RGB float32: train `(39209,32,32,3)`, test `(12630,32,32,3)`, trong [0,1].
  Nhãn int64 một chiều; 43 tên lớp, index bằng ClassId.
- Giữ toàn bộ hai CSV train/test chính thức; không chia lại ngẫu nhiên, không
  tạo validation. Thành viên mô hình cần CV theo **(ClassId, TrackId)** để không
  đưa các khung hình gần giống của cùng biển vào cả train/validation fold.
- Không học mean/std, StandardScaler hay PCA ở phần tiền xử lý. Những bước học
  thống kê do phần đặc trưng/mô hình fit đúng trên train, hoặc trên train fold
  khi cross-validation; không sửa các mô-đun đó trong công việc này.
- Augmentation mặc định tắt; minh họa chỉ xoay nhẹ ±10°, dịch ≤5%, sáng/tương
  phản ≤10%. **Cấm lật ngang/dọc** vì có thể đổi nghĩa lớp 33↔34, 38↔39.
- `load_image` có tùy chọn keyword-only `image_size`/`interpolation` để chạy
  ablation; bỏ tùy chọn thì hành vi cũ được giữ. Không sửa config chung theo
  kết quả thăm dò, không đổi chữ ký `load_and_preprocess`.
- Cache pipeline kiểm tra cấu hình/CSV/thư mục nguồn/phiên bản và hợp đồng mảng;
  không băm từng ảnh. Thay bytes ảnh mà giữ CSV cần tạo lại cache.

## 4. Minh chứng tác động: ablation tiền xử lý

Nguồn: [02_preprocessing_ablation.ipynb](../notebooks/02_preprocessing_ablation.ipynb),
[CSV](tables/ablation_preprocessing.csv), [biểu đồ](figures/eda_ablation.png),
[summary](ablation_summary.json), [manifest](tables/ablation_sample_manifest.csv),
[dự đoán](tables/ablation_predictions.csv).

### 4.1. Thiết lập

- 8.000 train phân tầng theo lớp; 3.000 test ngẫu nhiên không hoàn lại, seed 42,
  **không phân tầng/cân bằng test**. Cùng danh sách/thứ tự ảnh cho mọi phương án.
- KNN k=3, uniform, Euclidean, brute, n_jobs=1; phép toán số giới hạn 2 luồng.
- Mốc: ROI=True, 32×32, RGB, AREA, chia 255. Mỗi phương án chỉ đổi một yếu tố
  so với mốc. Không scaler/PCA/augmentation, không thử tương tác nhiều yếu tố.
- Báo cáo Accuracy, Macro-F1 trên đủ 43 lớp, số chiều và fit+predict. Thời gian
  chuẩn bị/nén/nạp cache được đo riêng, không cộng vào thời gian mô hình.

### 4.2. Kết quả

| Phương án | Số chiều | Accuracy | Macro-F1 | ΔAccuracy so mốc (đpt) | ΔMacro-F1 so mốc (đpt) |
|---|---:|---:|---:|---:|---:|
| **ROI, 32×32, RGB, AREA (mốc)** | 3.072 | **47,67%** | **40,76%** | 0 | 0 |
| Không ROI, 32×32, RGB, AREA | 3.072 | 34,03% | 29,49% | −13,63 | −11,27 |
| ROI, 24×24, RGB, AREA | 1.728 | 48,00% | 40,84% | +0,33 | +0,08 |
| ROI, 48×48, RGB, AREA | 6.912 | 47,47% | 40,55% | −0,20 | −0,22 |
| ROI, 32×32, grayscale, AREA | 1.024 | 45,47% | 39,55% | −2,20 | −1,21 |
| ROI, 32×32, RGB, LINEAR | 3.072 | 47,37% | 40,60% | −0,30 | −0,17 |

Đọc thời gian fit+predict chính xác của lượt chạy mới nhất tại CSV/summary và
output notebook. Một lượt đo, không phải benchmark kiểm soát; không khẳng định
chênh lệch thời gian chỉ do tiền xử lý hay sẽ giữ nguyên trên máy khác.

### 4.3. Giải thích và quyết định bàn giao

1. **ROI có tác động lớn nhất trong các phép so sánh này:** tăng 13,63 đpt
   Accuracy / 11,27 đpt Macro-F1. Loại nền và căn vùng biển phù hợp với KNN pixel
   nhạy vị trí. Không suy rộng thành cam kết cho ảnh chưa được định vị.
2. **24×24 cao hơn mốc rất ít**, đồng thời giảm 43,75% số chiều. Làm mượt khi
   thu nhỏ có thể bớt nhiễu, nhưng mất nét chữ số. Không có kiểm định/multiple seed
   nên không kết luận +0,08 đpt Macro-F1 là ưu thế chắc chắn.
3. **48×48 không cải thiện trong mẫu này**, dù tăng số chiều 2,25 lần; phóng to
   không tái tạo chi tiết đã mất trong ảnh gốc, còn tăng chi phí khoảng cách.
4. **RGB tốt hơn grayscale trên mẫu này**, phù hợp thông tin màu nhóm biển.
   Hai biểu diễn khác cả số chiều/hình học khoảng cách, không chỉ màu.
5. **AREA nhỉnh hơn LINEAR rất ít**; AREA có cơ sở khi thu nhỏ nhưng số liệu
   không chứng minh tối ưu cho mọi ảnh, đặc biệt những ROI phải phóng to.

**Cấu hình tích hợp giữ nguyên: ROI=True, 32×32, RGB, INTER_AREA, float32/255.**
Đây là cấu hình đã thống nhất cho nhóm và bảo toàn hợp đồng HOG/RGB, không phải
cấu hình được chứng minh tối ưu toàn cục. **Đề xuất kiểm chứng 24×24 bằng CV theo
track trên train**, rồi nhóm mới quyết định có đổi biểu diễn/hợp đồng hay không.
Không tự ý sửa `config.py` theo điểm test.

### 4.4. Trung thực và hạn chế đánh giá

- Cả EDA và ablation đã xem các test con. Đây là thăm dò theo yêu cầu bài tập,
  **không phải đánh giá cuối cùng trên holdout hoàn toàn mù**.
- Không so trực tiếp điểm EDA 5.000/2.000 với ablation 8.000/3.000 để khẳng định
  một kỹ thuật tốt hơn: khác cỡ mẫu và cách lấy test.
- Không dùng test để tiếp tục tuning. CV theo track trên train giúp xác nhận
  lựa chọn nhưng không làm cho test đã xem trở thành chưa từng được xem.
- Một seed, một k cố định, một lượt timing, không khoảng tin cậy/kiểm định;
  ảnh trong cùng track không phải quan sát độc lập hoàn toàn.
- Chỉ kiểm tra Path, chưa hash toàn bộ ảnh để phát hiện trùng nội dung. Resize
  vuông có thể méo tỉ lệ, không xử lý mờ/ánh sáng/biển bị che khuất.

## 5. Tái lập, log và minh bạch AI

- Python thực nghiệm **3.13.2, Windows 11**; requirements ghim các phiên bản đã
  cài. Bộ pin hiện tại cần Python ≥3.12 do SciPy. Môi trường thực nghiệm kế thừa
  một số package hệ thống; chưa tuyên bố đã kiểm thử cài mới sạch trên mọi OS.
- [Script kiểm tra](../scripts/verify_data_prep.py) chạy tuần tự: pip freeze/check,
  unit tests, CLI tiền xử lý, hai notebook, smoke-test main. Kernel dùng đúng
  interpreter gọi script. Có thể chạy từ gốc bằng:

  ```powershell
  .\venv\Scripts\python.exe scripts/verify_data_prep.py
  ```

- [logs/preprocessing.log](../logs/preprocessing.log) ghi timestamp, seed, config,
  cache hit/miss/disabled, counts, X/y metadata, min/max, lỗi/bỏ qua và thời gian.
  Khi cache hit, `errors=0` là không có lỗi trong lượt nạp cache, **không** là
  khẳng định đã giải mã/kiểm tra lại mọi ảnh gốc.
- Tests dùng ảnh tổng hợp và thư mục tạm, không yêu cầu dữ liệu thật, không ghi
  nhiễu vào log dữ liệu thật. Lỗi log chỉ cảnh báo, lỗi ảnh vẫn fail-fast.
- Cache toàn bộ và cache ablation nằm trong `data/`, bị Git ignore. Các hình
  tổng hợp báo cáo và log được bàn giao; không stage ảnh gốc, cache hay venv.
- Smoke-test chỉ xác nhận `main.py` nạp dữ liệu và tới đầu `build_features` rồi
  **chủ động dừng**, chưa chạy xong feature extraction/GridSearchCV/đánh giá đầy đủ.
- Prompt nguyên văn: [prompt chính](../docs/prompts/01_eda_preprocessing_prompt.md),
  [prompt bổ sung](../docs/prompts/02_bo_sung_rubric_prompt.md).
  Xem [AI_USAGE.md](../AI_USAGE.md) để biết mức hỗ trợ AI. Phần người học tự kiểm
  tra/sửa **chưa khai báo**, không được coi log do AI chạy là đã tự kiểm chứng.

### 5.1. Bằng chứng kiểm tra ngày 2026-11-11

- **24/24 unit tests đạt**, gồm 14 kiểm thử cũ và 10 kiểm thử mới cho tùy chọn
  resize/default equality/logging. Lượt tích hợp cuối: 1,892 giây cho tests.
- CLI tiền xử lý nạp cache toàn bộ dữ liệu, đúng hợp đồng; thời gian loader
  4,403 giây, `errors=0 skipped=0` trong lượt nạp cache.
- EDA **9/9 ô code, 0 lỗi**, 93,43 giây. Ablation **6/6 ô code, 0 lỗi**,
  37,22 giây khi đủ cache; lần đầu chưa có cache 394,23 giây (khoảng 6,57 phút).
  Thời gian phụ thuộc I/O, CPU, kernel startup và cache của hệ điều hành.
- Fit+predict của lượt ablation cuối: mốc **2,587s**, không ROI **2,795s**,
  24×24 **1,456s**, 48×48 **5,618s**, grayscale **0,831s**, LINEAR **2,607s**.
  Các chỉ số Accuracy/Macro-F1 giống lượt chưa có cache; timing thay đổi giữa lượt.
- Kiểm tra độc lập: toàn bộ **11.000 ảnh mốc ablation khớp từng pixel với cache
  pipeline trước bổ sung**; nhãn/thứ tự mẫu khớp CSV; tính lại metric từ file dự
  đoán khớp bảng. Không chỉ kiểm tra shape của vài ảnh.
- **12 PNG, 150 dpi; 20 CSV** (16 EDA + 4 ablation). Hai prompt khớp nguyên văn,
  các pin package khớp môi trường, `pip check` không báo dependency hỏng.
- Main tới `build_features` trong 14,16 giây rồi bị chủ động dừng; exit=1 của
  tiến trình con trên Windows là do dừng bằng taskkill, không phải một lượt
  huấn luyện hoàn tất. Script kiểm tra tổng kết `status=passed` cho đúng phạm vi.
- Jupyter có cảnh báo transport TCP không mã hóa trong lượt chạy local; không
  có lỗi ô code. Không mở kernel ra mạng hoặc dùng chung máy/mạng không tin cậy
  mà chưa cấu hình bảo vệ phù hợp.
- Bằng chứng máy đọc được: [logs/verification.json](../logs/verification.json).
  Không stage/commit/push; các mô-đun `features.py`, `models.py`, `evaluate.py`
  và `main.py` không thay đổi.

## 6. Việc thuộc trách nhiệm người nhận/người học

1. Xác minh giấy phép và nguồn của đúng bản dữ liệu đã tải.
2. Tự đọc code/notebook, chạy lại các bước cần thiết, điền phần tự kiểm tra/sửa
   trong AI_USAGE và bảo đảm giải thích được thuật toán/giới hạn.
3. Thành viên mô hình xác nhận lựa chọn bằng CV theo track trên train; thành viên
   đánh giá kiểm tra các nhóm tốc độ dễ nhầm bằng dự đoán/confusion matrix thật.
4. Chuyển ghi chú/số liệu/hình sang báo cáo LaTeX nếu cần; không gọi các baseline
   thăm dò này là điểm cuối của đề tài.
5. Chưa có commit/push tự động. Gợi ý chia commit (không kèm ảnh/cache dữ liệu):
   - `feat(data-prep): add resize ablation options and reproducible logging`
   - `feat(data-prep): add executed preprocessing ablation and evidence`
   - `docs: document data provenance, prompts and reproducible environment`
