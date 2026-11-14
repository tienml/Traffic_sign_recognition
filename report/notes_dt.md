# Nguyễn Hoàng (Người 3) — Decision Tree: kết quả đầy đủ

## Trạng thái và bằng chứng

Đã hoàn tất full run [`full-dt-01`](dt/full-dt-01/selection.json): **39.209 ảnh, 43 lớp, 1.307 track; 144 CV fits + 3 refit**, không lấy mẫu. Selection được đóng băng trước official test. Run [`test-dt-01`](dt/test-dt-01/test_metrics.json) đánh giá **một lần mỗi biểu diễn trên 12.630 ảnh**, không fit/tune/chọn lại theo test. Winner **HOG** được xác định chỉ bằng mean macro-F1 CV. Test từng được EDA thăm dò, không gọi hoàn toàn chưa xem.

Không chạy lại/sửa KNN, main/config/requirements hoặc phần người 1/2/4. README/AI_USAGE chỉ cập nhật phần DT người 3. Không commit/push hoặc xuất bản bên ngoài; bundles lớn ở `data/models/dt/full-dt-01/` đã ignore.

## Giao thức và chống rò rỉ

- Dữ liệu local `data/`; CLI hỗ trợ cả `data/archive/` nhưng phải chỉ định layout rõ ràng. RGB → crop ROI → resize32×32 → /255; ảnh float32.
- Đọc lại chính xác [`fold_manifest.csv`](dt/full-dt-01/fold_manifest.csv) từ KNN `full-knn-01`: 3 grouped folds theo `(ClassId, TrackId)`, seed42, không sinh split khác. Kiểm tra row/path/label/group/assignment, không giao track, đủ43 lớp và mỗi dòng validation đúng một lần.
- Manifest SHA256 `c5453353dbe7fdf77013f978e61a20b5c55a95f299eacac70a33676b0c43a1fb`; Train.csv SHA256 `4e00ce71e4bdd9fc424d93f81ff82056c6b5cb9bbc48269741e628a134c4712b`.
- Pixel3072 chiều không scaler; HOG324 chiều + StandardScaler; HOG+PCA thêm PCA95% và float32. Scaler/PCA chỉ fit fold-train trong sklearn Pipeline, refit cuối dùng toàn train. PCA refit107 chiều; số chiều từng fold trong CV CSV. Scaler giữ hợp đồng biểu diễn chung với KNN, không khẳng định cây cần scaler.
- Grid16 candidates/biểu diễn: criterion={gini,entropy}, max_depth={None,10,20,30}, min_samples_split={2,5}. DecisionTreeClassifier random_state42, các tham số khác mặc định. Chọn macro-F1 mean, tie lấy candidate đầu ParameterGrid rồi pixel/hog/hog_pca. Không chọn theo thời gian hoặc test. Metrics trên43 lớp, zero_division0; SD ba folds không phải CI. Cột `test_*` trong CV CSV là fold-validation, không phải official test.
- Test gate xác minh full frozen selection, dataset/train/fold hashes, size, winner, model hashes và metadata params. Đã sửa reviewer defect: so cả **metadata.config.hog_params và image_contract với cấu hình extraction hiện tại**, chặn drift trước khi tạo exclusive `official_test_claim.json` hoặc đọc Test.csv. Regression bao gồm HOG block_norm đổi nhưng cùng số chiều và ROI contract drift; đều không chạm loader/claim.
- Claim giữ nguyên nếu lỗi, không xóa để chạy lại. Prediction batch128, kiểm tra nhãn/số lượng; runner gắn CSVRow/Path/nhãn từ Test.csv. Exporter chỉ đọc artifacts, xác minh selection/hash/claim,16 candidates từng phương pháp, candidate/winner, đủ12.630 predictions mỗi phương pháp, CSVRow tuần tự/Path duy nhất/nhãn hợp lệ và alignment giữa ba file predictions, rồi tính lại bốn metrics với tolerance1e-12. Exporter không đọc Test.csv hoặc xác minh lại hash của Test.csv; bước rà soát độc lập sau chạy đã đối chiếu cả ba file predictions với Test.csv thực tế và xác nhận đúng thứ tự/Path/ClassId. Không fit/predict lại.
- Run/model/export tạo độc quyền, không ghi đè. Config lưu versions, seed, source hashes và extraction contract. Joblib chỉ load local tin cậy: hash không làm pickle không tin cậy an toàn. Hash CSV không xác minh nội dung từng ảnh.

## Kết quả full

| Biểu diễn | criterion / max_depth / min_samples_split | CV macro-F1 ± SD | CV Accuracy | Test Accuracy | Test macro Precision | Test macro Recall | Test macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pixel | entropy / None /5 |0,503989 ±0,013122|0,591444|0,635313|0,578621|0,558620|0,563179|
| **HOG (winner CV)** | entropy /20 /5 |**0,614898 ±0,011907**|0,658769|**0,656770**|0,614162|0,585053|**0,594987**|
| HOG+PCA | gini /20 /5 |0,550699 ±0,006664|0,602670|0,591607|0,560685|0,522457|0,536760|

Cây full-refit Pixel: depth17,5.267 nodes,2.634 leaves; HOG: depth20,6.523 nodes,3.262 leaves; HOG+PCA: depth20,5.193 nodes,2.597 leaves. Depth=None là không giới hạn trước, không có nghĩa depth vô hạn.

Bảng đầy đủ: [summary](tables/dt_full-dt-01_summary.csv), [48 candidates](tables/dt_full-dt-01_candidates.csv), [timings](tables/dt_full-dt-01_timings.csv), [class summary](tables/dt_full-dt-01_class_summary.csv), [error pairs](tables/dt_full-dt-01_error_pairs.csv). [Hình CV/test](figures/dt_full-dt-01_cv_test.png) dùng monochrome + marker/legend và CSV tương đương; viewer trả nội dung rỗng nên chưa xác nhận trực quan layout.

## Trường hợp đặc biệt quan sát từ predictions

- HOG có recall thấp nhất ở lớp21:12/90=0,133333; lớp30:28/150=0,186667; lớp27:12/60=0,200000; lớp0:14/60=0,233333. Macro metrics làm lộ hiệu năng yếu ở lớp ít mẫu mà Accuracy tổng che khuất.
- Cặp nhầm HOG nhiều nhất theo chiều thật→dự đoán:5→1 (131),1→5 (118),2→1 (103),2→5 (89),5→2 (89),1→2 (86). Chỉ suy luận từ labels/predictions; chưa chứng minh nguyên nhân mờ/thiếu sáng hoặc đặc điểm ảnh. Không thay confusion matrix/phân tích lỗi của người4.
- Pixel test F1 cao hơn CV (0,563179 so0,503989) không chứng minh tổng quát hóa tốt hơn; tập test và grouped validation có phân bố khác nhau. Không đổi protocol vì thấy test.
- PCA làm giảm dimensions324→107 và thời gian nhưng F1 thấp hơn HOG ở CV và test. Đây là quan sát của run này, phù hợp khả năng PCA xoay trục bất lợi cho split axis-aligned, không phải chứng minh nguyên nhân hoặc mọi PCA đều kém.

## So sánh trung thực với KNN đã lưu

Cùng39.209 train/12.630 test, preprocessing và frozen grouped folds; không rerun KNN. KNN grid12 candidates/biểu diễn khác DT16, nên không phải ngân sách search bằng nhau hoặc kết luận thống kê.

| Biểu diễn | KNN test Accuracy / macro-F1 | DT test Accuracy / macro-F1 |
| --- | ---: | ---: |
|Pixel|0,557245 /0,493948|0,635313 /0,563179|
|HOG|0,818369 /0,789750|0,656770 /0,594987|
|HOG+PCA|0,796675 /0,763063|0,591607 /0,536760|

Hai winner CV đều HOG. DT HOG kém KNN HOG **0,194763 macro-F1 (~19,48 điểm phần trăm)** trên test này; Pixel DT cao hơn Pixel KNN không đảo kết luận winner. KNN HOG CV0,809718 ±0,016981 so DT0,614898 ±0,011907; không có nested CV, paired significance hoặc CI để khẳng định độ chắc chắn rộng hơn. Tham chiếu [báo cáo KNN](notes_knn.md).

## Timing và tài nguyên đo thật

| Biểu diễn | Search+refit (s) | Refit nằm trong cột trước (s) | Test predict (s) |
| --- | ---: | ---: | ---: |
|Pixel|8664,743|327,141|0,107251|
|HOG|3385,860|177,693|0,130244|
|HOG+PCA|1196,152|20,279|0,156553|

Tổng search+refit13246,755s (~3,68h), chưa tính train load74,296s và HOG extraction33,123s dùng chung **một lần**, không cộng hai lần từ hai dòng bảng. Pixel flatten gần như view (~0,000056s), không phải toàn chi phí ảnh. Test load26,990s; HOG extraction11,393s chỉ tính ở lần HOG đầu; HOG+PCA dùng lại nên feature_seconds gần0 không có nghĩa PCA miễn phí.

Search+refit bao gồm mọi CV fit/scoring và refit đã chọn, không gồm load/extract stateless hoặc lưu bundle. Predict là batch inference trên raw features đã có, bao gồm scaler/PCA transform trong pipeline, không gồm feature extraction, metrics hoặc CSV I/O; không phải latency end-to-end hay benchmark lặp nhiều lần. KNN HOG predict đã lưu104,088s so DT0,130s nhưng hai run khác thời điểm, không đo kiểm soát cùng tải; KNN refit gần0 do lưu vectors, không tương đương học split của cây.

Chạy tuần tự n_jobs1, numeric threads2, batch128. Một mảng ảnh/pixel train float32 ~459,48MiB, HOG~48,46MiB; fold indexing/cast/workspaces có thể copy. Chưa đo peak RAM, model resident size hoặc latency phân vị; không gọi459MiB là tổng RAM. Không giảm grid/sampling để rút ngắn.

## Kiểm thử, lịch sử và logs

- Sau sửa extraction gate: **20/20 focused pass** (8 DT +12 KNN), [focused log](../logs/dt/focused_tests_contract.log). Full suite **57/58 pass**, cùng lỗi có sẵn cumulative PCA float32 tại `src/tests/test_features.py:149`, sai khác5,9604645e-08; không sửa code/test người2. [Full suite log](../logs/dt/full_suite_contract.log).
- Preflight39.209 ảnh/1.307 tracks ở `dt-preflight-20261114`; smoke3.870 ảnh/129 nguyên tracks,9fits+3refit ở `dt-smoke-20261114-v2`, giữ assignment gốc. Smoke F1 Pixel0,142087/HOG0,311016/HOG+PCA0,298274 không phải kết quả full.
- Smoke đầu lỗi JSON numpy.int64 tree statistics; đã cast int, thêm synthetic orchestration regression và chạy idv2; giữ run lỗi, không ghi đè. Full run hiện thành công, selection frozen đủ3 methods.
- [Console full tune](../logs/dt/full-dt-01_full_console.log), [console UTF-8 của tune](../logs/dt/full-dt-01/console.log), [test console](../logs/dt/test-dt-01_full_console.log), loader `preprocessing.log` từng run. Console ngoài của tune có legacy encoding/escape, không tuyên bố toàn bộ Unicode đã phục hồi; console nội lưu CV đầy đủ.
- [Export provenance](../logs/dt/dt_full-dt-01_export_provenance.json), [export log](../logs/dt/full-dt-01_export.log). Export thành công nghĩa count/alignment/metrics validation đã qua; không tạo điểm số giả.

## Lệnh đã thực thi (từ root repo)

```powershell
& ./.venv/Scripts/python.exe -m unittest src.tests.test_decision_tree src.tests.test_models src.tests.test_knn_summary
& ./.venv/Scripts/python.exe scripts/train_dt.py tune --dataset-dir data --run-id full-dt-01 --batch-size 128
& ./.venv/Scripts/python.exe scripts/train_dt.py test --dataset-dir data --run-id test-dt-01 --selection report/dt/full-dt-01/selection.json --batch-size 128
& ./.venv/Scripts/python.exe scripts/summarize_dt.py --tune-dir report/dt/full-dt-01 --test-dir report/dt/test-dt-01 --export-id full-dt-01
```

**Không chạy lại tune/test các id trên, không xóa claim.** Nếu chỉ tái xuất từ artifacts, dùng export-id mới (ví dụ `full-dt-review-02`), không inference lại.

Giới hạn: cây đơn dễ overfit/không ổn định, một seed/3fold/grid hẹp, không pruning/class weights hay nested evaluation, test đã thăm dò trước đó và hash metadata không xác minh mọi ảnh. Đây chỉ là phân loại biển đã định vị, không phát hiện biển trong toàn cảnh. Thành viên cần tự đọc, kiểm chứng, giải thích và chịu trách nhiệm; tests do AI chạy không chứng minh người học tự thực hiện.
