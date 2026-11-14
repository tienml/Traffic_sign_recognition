# Nhật ký sử dụng công cụ AI (AI Usage Log)

Nhóm khai báo việc sử dụng AI theo từng thành viên ở bảng dưới. AI có thể hỗ trợ viết code, phân tích và chạy kiểm thử; kết quả chạy tự động không đồng nghĩa thành viên đã tự kiểm tra hoặc giải thích được mọi dòng. Mỗi thành viên cần tự đọc, xác minh và chịu trách nhiệm về phần bàn giao.

| Thành viên | Công cụ AI sử dụng | Mục đích sử dụng | Phần nội dung/code áp dụng |
| :--- | :--- | :--- | :--- |
| Cả nhóm | Gemini | Khởi tạo cấu trúc thư mục chuẩn và khung hàm I/O | `main.py`, `src/config.py` |
| Vũ Khánh Hùng | Claude Code | Hỗ trợ viết code đọc ảnh và cache, logging, kiểm thử; vẽ biểu đồ EDA; soạn/chạy ablation;| `src/preprocessing.py`, `src/tests/test_preprocessing.py`, `notebooks/01_eda.ipynb`, `notebooks/02_preprocessing_ablation.ipynb`|
| Thành viên 2 (Sơn) | Claude Code | Hỗ trợ viết `FeatureExtractor` (pixel/HOG/PCA), kiểm thử chống rò rỉ, script ablation và biểu đồ | `src/features.py`, `src/tests/test_features.py`, `scripts/feature_ablation.py`, `report/figures/feature_ablation.png` |
| Nguyễn Hoàng (Người 3) | Claude Code | Hỗ trợ triển khai KNN theo track, CV/tuning và đóng băng cấu hình; tổng hợp artifact đã lưu, bảng/hình, logging, kiểm thử và tài liệu | `src/models.py`, `scripts/train_knn.py`, `scripts/summarize_knn.py`, `src/tests/test_models.py`, `src/tests/test_knn_summary.py`, `report/notes_knn.md`, các output `knn_*`, mục 5 README |
| Người 4 | ... | ... | `src/evaluate.py` |

### Vũ Khánh Hùng — kiểm chứng và trách nhiệm cá nhân

- Trợ lý AI đã hỗ trợ thực thi kiểm thử/notebook trên dữ liệu local; bằng chứng nằm trong output notebook và `logs/`. Không coi đó là bằng chứng người học đã tự thực hiện các bước.
- **Nguồn tham khảo:** GTSRB/IJCNN 2011 và các trang tải được dẫn ở README, mục 0 notebook EDA. Giấy phép bản dữ liệu local còn cần xác minh; không tự khẳng định quyền tái phân phối.
- **Giới hạn:** KNN trong EDA/ablation là thăm dò test con, không phải kết quả cuối; giữ nguyên mô-đun đặc trưng, mô hình và đánh giá 
- **Bổ sung ngày 2026-11-12:** hệ thống hóa 3 giả thuyết từ kết quả đã có, diễn giải các hình và trường hợp đặc biệt, làm rõ việc giữ nhãn/hướng biển trái–phải và không lật ảnh. Không chạy thí nghiệm mới để chứng minh lợi ích của việc không lật; không coi phần diễn giải này là kiểm chứng cá nhân của thành viên.

### Nguyễn Hoàng — phần KNN và trách nhiệm xác minh

- AI hỗ trợ code KNN/CV theo `(ClassId, TrackId)`, kiểm thử pipeline và đóng băng selection; báo cáo dựa trên full run `report/knn/full-knn-01` và test run `report/knn/20261113T194939Z-test-adf82d0d`. Lượt cập nhật tài liệu này không retrain/tune/inference hoặc chọn lại theo official test.
- AI tạo helper `scripts/summarize_knn.py` đọc artifacts, xác minh selection SHA256/claim, so khớp cấu hình và tính lại metrics từ predictions đã lưu; xuất CSV `report/tables/knn_*`, hình `report/figures/knn_cv_test.png`, provenance/log tại `logs/knn/`. Đây không phải điểm số do mô hình ngôn ngữ ước lượng. Kết quả HOG winner: CV macro-F1 0,809718 ± 0,016981, test Accuracy 0,818369 / macro-F1 0,789750.
- AI chạy 12 focused tests: pass; full suite 50 tests: 49 pass, 1 fail float32 cumulative variance tại `src/tests/test_features.py:149`. Nhật ký trước đó là 9/9 modeling và 46/47 toàn suite. Giữ nguyên phần nguồn/test người 1/2; không tuyên bố toàn bộ kiểm thử đạt. Log thực thi mới lưu ở `logs/knn/knn_focused_tests.log` và `knn_full_suite.log`.
- Giữ loader log lịch sử bằng bản sao từ run, không xóa nguồn. Sao chép console full training/testing từ thư mục tạm bị hệ thống quyền từ chối; không tạo console giả hoặc chạy lại official test để lấy log. Các console lịch sử có legacy encoding/escape; không tuyên bố đã khôi phục Unicode.
- Giới hạn được AI ghi rõ trong `report/notes_knn.md`: test từng được EDA thăm dò, một seed/ba fold, grid 12 candidates, hash metadata không chứng minh độc lập nội dung ảnh, chi phí lưu vector/tra cứu KNN, PCA dimensions theo từng fit, lớp recall thấp và cặp nhầm chỉ suy ra từ labels/predictions. Không suy đoán mờ/ánh sáng khi chưa xem ảnh, không thay phân tích lỗi người 4, không tuyên bố đã làm Decision Tree.
- Nguyễn Hoàng cần tự đọc, chạy/xác minh và giải thích phần bàn giao trước khi nộp. Những kiểm tra do trợ lý thực thi **không phải bằng chứng người học đã tự thực hiện hoặc hiểu mọi dòng**. Không sửa các mục người 1/2 trong README và nhật ký AI này; không commit/push trong lượt cập nhật.

### Nguyễn Hoàng — bổ sung Decision Tree (2026-11-14)

- Claude Code hỗ trợ viết `src/decision_tree.py`, `scripts/train_dt.py`, `scripts/summarize_dt.py`, `src/tests/test_decision_tree.py`, `report/notes_dt.md` và mục6 README. Giữ nguyên KNN và phần người1/2/4; không commit/push/xuất bản bên ngoài.
- AI chạy preflight/smoke rồi **full144 CV fits +3 refit** trên39.209 ảnh/1.307 tracks,16 candidates mỗi biểu diễn, cùng frozen manifest KNN, không sampling. `full-dt-01` đóng băng winner HOG(entropy/depth20/split5), CV macro-F1 0,614898 ±0,011907 trước test. `test-dt-01` đánh giá một lần mỗi biểu diễn trên12.630 ảnh: HOG Accuracy0,656770/F1 0,594987; Pixel0,635313/0,563179; HOG+PCA0,591607/0,536760. Không retune/chọn lại theo test hoặc chạy lại KNN.
- Smoke đầu bị lỗi ghi JSON do numpy.int64 tree statistics; AI sửa cast int, thêm test orchestration synthetic rồi chạy run v2 mới. Giữ run lỗi; không tạo log giả, không ghi đè hoặc chạy official test để xử lý lỗi.
- AI sửa reviewer defect: test gate so metadata.config.hog_params/image_contract với extraction hiện tại trước claim/Test.csv; regression HOG cùng dimensions nhưng block_norm drift và ROI contract drift. **20/20 focused pass**, full suite **57/58**, còn cùng lỗi float32 PCA người2 tại `src/tests/test_features.py:149`, không sửa code/tolerance người2. Logs `logs/dt/focused_tests_contract.log`, `full_suite_contract.log`; run JSON/CSV `report/dt/`, bundles ignored `data/models/dt/`.
- AI xuất đủ5 bảng full `report/tables/dt_full-dt-01_*` và hình từ saved artifacts, xác minh count12.630/method, alignment và recomputed metrics, không inference lại. Console full tune/test, export log và provenance nằm `logs/dt/`; console ngoài tune có legacy encoding/escape, console nội UTF-8 giữ CV. Viewer ảnh trả rỗng, chưa xác nhận trực quan. Tổng search+refit đo thật13246,755s (~3,68h), không gồm load/extract; chưa đo peak RAM. Báo cáo ghi timing boundaries, lớp recall thấp/cặp nhầm quan sát và so sánh KNN không retune: DT HOG F1 thấp hơn KNN HOG0,194763 trên test này, không suy ra significance.
- Test đã từng được EDA thăm dò; hash CSV không xác minh toàn bộ ảnh; một seed/3fold/grid hẹp và cây đơn có giới hạn. Nguyễn Hoàng cần tự đọc, chạy/xác minh, giải thích và chịu trách nhiệm; kết quả trợ lý chạy không phải xác nhận người học tự thực hiện.
