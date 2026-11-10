# BTL Nhập môn Học máy (FIT4013) - Đề tài 6: Nhận dạng biển báo giao thông

## 1. Thành viên & Phân công
| Thành viên | Nhánh GitHub | Phụ trách |
| :--- | :--- | :--- |
| Thành viên 1 | `feat/1-data-prep` | EDA & Tiền xử lý ảnh (`src/preprocessing.py`) |
| Thành viên 2 | `feat/2-features-baseline` | Trích xuất đặc trưng Pixel, HOG, PCA & Baseline (`src/features.py`) |
| Thành viên 3 | `feat/3-modeling` | Huấn luyện & Tối ưu K-NN, Decision Tree (`src/models.py`) |
| Thành viên 4 | `feat/4-evaluation` | Đánh giá, Confusion Matrix & Phân tích lỗi (`src/evaluate.py`) |

## 2. Hướng dẫn cài đặt và chạy chương trình
1. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Tải bộ dữ liệu và giải nén vào thư mục `data/`.
3. Chạy toàn bộ pipeline:
   ```bash
   python main.py
   ```
