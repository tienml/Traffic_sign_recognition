# Nhật ký sử dụng công cụ AI (AI Usage Log)

Nhóm khai báo việc sử dụng AI theo từng thành viên ở bảng dưới. AI có thể hỗ trợ viết code, phân tích và chạy kiểm thử; kết quả chạy tự động không đồng nghĩa thành viên đã tự kiểm tra hoặc giải thích được mọi dòng. Mỗi thành viên cần tự đọc, xác minh và chịu trách nhiệm về phần bàn giao.

| Thành viên | Công cụ AI sử dụng | Mục đích sử dụng | Phần nội dung/code áp dụng |
| :--- | :--- | :--- | :--- |
| Cả nhóm | Gemini | Khởi tạo cấu trúc thư mục chuẩn và khung hàm I/O | `main.py`, `src/config.py` |
| Vũ Khánh Hùng | Claude Code | Hỗ trợ viết code đọc ảnh và cache, logging, kiểm thử; vẽ biểu đồ EDA; soạn/chạy ablation;| `src/preprocessing.py`, `src/tests/test_preprocessing.py`, `notebooks/01_eda.ipynb`, `notebooks/02_preprocessing_ablation.ipynb`|
| Thành viên 2 (Sơn) | Claude Code | Hỗ trợ viết `FeatureExtractor` (pixel/HOG/PCA), kiểm thử chống rò rỉ, script ablation và biểu đồ | `src/features.py`, `src/tests/test_features.py`, `scripts/feature_ablation.py`, `report/figures/feature_ablation.png` |
| Người 3 | ... | ... | `src/models.py` |
| Người 4 | ... | ... | `src/evaluate.py` |

### Vũ Khánh Hùng — kiểm chứng và trách nhiệm cá nhân

- Trợ lý AI đã hỗ trợ thực thi kiểm thử/notebook trên dữ liệu local; bằng chứng nằm trong output notebook và `logs/`. Không coi đó là bằng chứng người học đã tự thực hiện các bước.
- **Nguồn tham khảo:** GTSRB/IJCNN 2011 và các trang tải được dẫn ở README, mục 0 notebook EDA. Giấy phép bản dữ liệu local còn cần xác minh; không tự khẳng định quyền tái phân phối.
- **Giới hạn:** KNN trong EDA/ablation là thăm dò test con, không phải kết quả cuối; giữ nguyên mô-đun đặc trưng, mô hình và đánh giá 
- **Bổ sung ngày 2026-11-12:** hệ thống hóa 3 giả thuyết từ kết quả đã có, diễn giải các hình và trường hợp đặc biệt, làm rõ việc giữ nhãn/hướng biển trái–phải và không lật ảnh. Không chạy thí nghiệm mới để chứng minh lợi ích của việc không lật; không coi phần diễn giải này là kiểm chứng cá nhân của thành viên.
