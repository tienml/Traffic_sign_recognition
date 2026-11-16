# Đặng Quang Tiến (Người 4) — nâng cấp K-NN và Decision Tree

**Phân công đóng góp:** Nguyễn Hoàng (người 3) thực hiện phần K-NN/Decision Tree gốc. **Đặng Quang Tiến (người 4)** phụ trách phần nâng cấp được mô tả trong báo cáo này, có trợ lý AI hỗ trợ triển khai và thực thi theo khai báo trong `AI_USAGE.md`. Tên kỹ thuật `model3_upgrade` chỉ phần mô hình được nâng cấp, không phải số thứ tự của người thực hiện.

## Phạm vi được thực hiện

Phần bổ sung do **Đặng Quang Tiến (người 4)** phụ trách chỉ nâng cấp việc tìm siêu tham số, hiệu quả tính toán và chẩn đoán CV của phần mô hình do **người 3** xây dựng. Không thay đổi source/test người 1/2, `main.py`, cấu hình chung, requirements hoặc các kết quả KNN/DT/official-test claim đã có. Không dùng học sâu, không thêm SVM/ensemble, không đổi HOG hoặc PCA của người 2. Báo cáo KNN/DT giữ nguyên kết luận cũ.

**Cấu trúc hiện tại sau tích hợp 2026-11-15:** logic nâng cấp nằm trực tiếp trong
[src/models.py](../src/models.py), không còn module `src/model_upgrade.py` hoạt động.
Đánh giá người4 nằm trong [src/evaluate.py](../src/evaluate.py). Các API mô hình gốc
vẫn tương thích; grid mới có tên `expanded_knn_candidates` và `regularized_dt_candidates`
để không ghi đè contract grid cũ. Việc chuyển mã không retrain hoặc đổi số liệu dưới đây.
Bản nguồn trước chuyển và audit riêng ở [logs/integration/20261115-canonical](../logs/integration/20261115-canonical/);
config/hash của run lịch sử không bị sửa. Khi audit source run này, thêm
`--source-root logs/integration/20261115-canonical/source_before` vào lệnh checker;
không có tùy chọn đó thì checker giữ chế độ strict current-source.

## Cơ sở từ dữ liệu và EDA

- EDA H2 đo mất cân bằng train khoảng 10,71 lần và 1.307 track; một lớp ít frame không tất yếu có F1 thấp. Việc thử `class_weight` của DT là giả thuyết cần đo, không phải kết luận mặc định.
- EDA H3 nêu ảnh nhỏ/chất lượng thấp và khó phân biệt chữ số. Lượt nâng cấp lưu dự đoán ngoài fold (OOF), support/số track/F1 từng lớp, các cặp nhầm và tỷ lệ lỗi theo nhóm kích thước/chất lượng. Không dùng official-test predictions để chọn cấu hình.
- KNN cũ có winner k=7 ở biên grid {3,5,7}; mở rộng k là hướng có động cơ từ CV. Giữ cùng HOG và cùng folds để tách thay đổi mô hình khỏi thay đổi đặc trưng.
- H1 RGB–gray của EDA chỉ là bằng chứng cho KNN pixel; không suy rộng để đổi màu hoặc thêm HSV trong lượt này.

Tham khảo: [EDA và giả thuyết](notes_eda_preprocessing.md#25-ba-giả-thuyết-nghiên-cứu-từ-eda), [KNN lịch sử](notes_knn.md), [DT lịch sử](notes_dt.md).

## Giao thức cố định trước khi chạy full

1. Chỉ đọc `Train.csv` và ảnh train; không mở `Test.csv`, ảnh test, predictions official-test hoặc xóa/tạo lại official-test claim. CLI mới chỉ có `preflight`, `smoke`, `tune`, không có `test`.
2. Dùng nguyên full frozen manifest KNN 3 folds theo `(ClassId,TrackId)`, seed42; kiểm tra nhãn/Path/thứ tự/nhóm, đủ43 lớp ở mỗi phía, mỗi ảnh validation đúng một lần. Smoke chọn nguyên một track mỗi lớp mỗi fold, không đổi assignment; không lấy điểm smoke làm kết quả full.
3. Ảnh qua hàm public người1/2 không sửa đổi: RGB → ROI cận đóng → 32×32 AREA → /255 → **HOG324**. StandardScaler chỉ fit fold-train. Không chạy lại Pixel/HOG+PCA vì nằm ngoài thay đổi được chọn.
4. KNN 28 candidates: k={3,5,7,11,15,21,31}, weights={uniform,distance}, metric={euclidean,manhattan}. Baseline k7/distance/manhattan đứng đầu, bảo đảm không chọn cấu hình khác khi hòa điểm.
5. DT 11 candidates: giữ criterion=entropy, max_depth20, min_samples_split5 của baseline; min_samples_leaf={1,2,5,10} × class_weight={None,balanced} tại ccp_alpha0 (8); thêm ba alpha={0,0001;0,0005;0,001} với leaf1/weightNone. Đây là search có mục tiêu, chưa xét tất cả tương tác hoặc mọi criterion/depth.
6. Chọn mean macro-F1 ba folds, hòa lấy cấu hình đầu; không chọn bằng timing, nhóm chất lượng hoặc test. Baseline và selected được so trong cùng lượt chạy/cùng môi trường. Refitted model dùng toàn train, nhưng chưa được official-test evaluation.

## Hiệu quả tính toán và tính trung thực

- KNN dùng một lượt tìm max-k neighbors cho mỗi metric/fold rồi tái sử dụng cho nhiều k/weights; query theo batch, không materialize toàn bộ distance matrix. Xử lý khoảng cách0 và hòa nhãn theo sklearn; hòa ở biên k thì fallback sang predict sklearn cho đúng các hàng đó. Có tests đối chiếu toàn grid với sklearn.
- `transforms[].neighbor_searches` giữ shared neighbor timing một lần/metric/fold; `cv_seconds_sum` của candidate **không gồm shared scaler/query**, không được cộng làm thời gian end-to-end. `cv_wall_seconds` đo toàn thuật toán, refit riêng; DT fit/predict timing bao gồm dự đoán train để chẩn đoán train–validation.
- HOG được đọc theo batch để không giữ toàn bộ mảng RGB ~459MiB. Lưu SHA256 từng ảnh train và raw-HOG audit, không chỉ fingerprint CSV. Không sửa cache hay ảnh đầu vào. Model joblib mới nằm `data/models/model3_upgrade/` đã ignore; chỉ load nguồn tin cậy.
- Môi trường riêng `data/.venv-model3` dùng Python3.12 và system-site packages sẵn có, cài riêng OpenCV/scikit-image; không đổi requirements chung. Versions thực tế nằm trong config và logs, không giả định trùng môi trường lịch sử.

## Cách đọc chẩn đoán

- OOF là dự đoán từ model không được fit bằng track của hàng đó. Tuy nhiên lựa chọn winner sử dụng **chính các fold OOF**, nên điểm selected vẫn có selection bias; không phải nested/unbiased generalization estimate.
- Mean macro-F1 theo fold không bằng macro-F1 gộp OOF; lưu và ghi nhãn riêng. SD ba folds không phải CI, không tuyên bố significance từ vài phần nghìn chênh lệch.
- Nhóm ảnh nhỏ dùng ngưỡng có từ EDA `min(W,H)<32`; ROI-size lưu riêng. Brightness/contrast mới đo trên ROI đã resize32, khác mẫu EDA cũ đo độ sáng toàn ảnh; không gọi đó là cùng phép đo. Quartile được fit trên từng fold-train rồi áp vào validation.
- Bảng lớp/nhóm/cặp nhầm chỉ mô tả, chưa kiểm soát class/track confounding; không chứng minh ảnh nhỏ/tối hay thiếu mẫu là nguyên nhân. Chưa đo blur, chưa repeated/nested CV hoặc CI.

## Trạng thái kiểm thử đã xác minh

- Trước thay đổi: suite cũ **57/58 pass**, lỗi float32 cumulative-PCA ở `src/tests/test_features.py:149`, chênh5,9604645e-08; giữ nguyên phần người2.
- Sau triển khai ban đầu: **13/13 tests mới pass**, full suite **70/71 pass**, cùng lỗi PCA có sẵn, không lỗi hồi quy mới. Log tại [verification-implementation-20261115](../logs/model3_upgrade/verification-implementation-20261115/verification.json).
- Trạng thái cuối thực nghiệm: preflight, smoke3870 ảnh và full39209 ảnh đều hoàn tất; full-result/audit ở mục bên dưới. Focused upgrade cuối **14/14 pass**; khi đã bổ sung tests evaluation, toàn suite **87/88 pass**, chỉ còn cùng lỗi PCA người2. Notebook03 đã chạy7 code cells,0 error outputs. Không dùng điểm smoke thay kết quả full.
- **Sau tích hợp canonical ngày2026-11-15:** model tests **17/17**, evaluation **18/18**;
  toàn suite **92/93**, đúng lỗi PCA cũ. Notebook03/04 chạy7/9 cells,0 lỗi, lưu copy
  dưới log mới; artifacts/notebook gốc không overwrite. Kiểm tra hash nguồn lịch sử
  qua source archive và model/OOF/candidates đều pass. Xem
  [verification canonical](../logs/model3_upgrade/verification-canonical-20261115/verification.json).
  Không train/retune lại full dataset hoặc chạy inference official test.

## Tái tạo có chủ ý

```powershell
# Môi trường mới hoàn toàn (không ghi đè venv đang có):
py -3.12 -m venv data/.venv-model3-repro
& data/.venv-model3-repro/Scripts/python.exe -m pip install -r scripts/requirements_model3_upgrade.txt
# Các lệnh dưới dùng môi trường đã chuẩn bị ở lượt thực nghiệm này:
& data/.venv-model3/Scripts/python.exe scripts/train_model3_upgrade.py preflight --dataset-dir data --run-id upgrade-preflight-new
& data/.venv-model3/Scripts/python.exe scripts/train_model3_upgrade.py smoke --dataset-dir data --run-id upgrade-smoke-new
& data/.venv-model3/Scripts/python.exe scripts/train_model3_upgrade.py tune --dataset-dir data --run-id upgrade-full-new
& data/.venv-model3/Scripts/python.exe scripts/summarize_model3_upgrade.py --run-dir report/model3_upgrade/upgrade-full-new
& data/.venv-model3/Scripts/python.exe scripts/check_model3_artifacts.py --run-dir report/model3_upgrade/upgrade-full-new
```

Run-id phải mới; mọi output namespaced được tạo độc quyền, không overwrite. Nếu cần xuất lại bảng/hình từ run đã có, dùng `--export-id ten-moi` để tạo tên mới, không chạy lại model. Notebook `03_model3_upgrade.ipynb` sẽ đọc artifacts full-run mặc định, không tự train khi Run All. Không commit/push trong lượt nâng cấp này.

### Lưu ý artifacts smoke

Smoke chỉ chạy baseline một cấu hình/thuật toán, nên selected=baseline là điều được thiết kế, không phải bằng chứng grid mở rộng không cải thiện. Smoke3870 ảnh/129tracks đã hoàn tất, không đọc official test. Bản figure ban đầu có panel metric không được thử nhưng chưa ghi nhãn đủ rõ; giữ nguyên lịch sử và đã tái xuất namespace `layout-v2` có tiêu đề **SMOKE ONLY** và thông báo không được đánh giá. Không dùng các hình smoke làm hình kết quả full.

## Kết quả full-run nâng cấp đã thực thi

**Phần nâng cấp: Đặng Quang Tiến (người 4), có trợ lý AI hỗ trợ.** Phần KNN/DT gốc của Nguyễn Hoàng (người 3) giữ nguyên.

Run `upgrade-full-20261115` đã hoàn tất trên **39.209 ảnh train / 1.307 track / 3 frozen folds**. Không dùng official test; không suy kết quả full từ smoke. Baseline và selected được tính trong cùng lượt.

| Mô hình | Baseline CV macro-F1 ± SD | Selected CV macro-F1 ± SD | Chênh lệch (đpt) | Cấu hình selected |
|---|---:|---:|---:|---|
| KNN | 0.809718 ± 0.016981 | 0.812096 ± 0.014242 | +0.2378 | `{"metric": "manhattan", "n_neighbors": 15, "weights": "distance"}` |
| DT | 0.614898 ± 0.011907 | 0.621998 ± 0.013418 | +0.7100 | `{"ccp_alpha": 0.0005, "class_weight": null, "criterion": "entropy", "max_depth": 20, "min_samples_leaf": 1, "min_samples_split": 5}` |

- KNN delta macro-F1 từng fold: -0.001500, +0.004002, +0.004633. Đây không phải CI hoặc kiểm định ý nghĩa.
- KNN CV wall toàn grid **280.835s**, refit riêng **0.242s**, bundle **42.777MiB**. Không so tốc độ trực tiếp với môi trường lịch sử.

- DT delta macro-F1 từng fold: +0.005655, +0.010867, +0.004779. Đây không phải CI hoặc kiểm định ý nghĩa.
- DT CV wall toàn grid **3410.951s**, refit riêng **149.248s**, bundle **0.113MiB**. Không so tốc độ trực tiếp với môi trường lịch sử.

[Bảng tổng hợp](tables/model3_upgrade-full-20261115_summary.csv), [toàn bộ candidates](tables/model3_upgrade-full-20261115_candidates.csv), [điểm từng fold](tables/model3_upgrade-full-20261115_fold_scores.csv), [delta từng fold](tables/model3_upgrade-full-20261115_fold_deltas.csv).

![Baseline và selected, SD không phải CI](figures/model3_upgrade-full-20261115_cv_comparison.png)

![KNN theo k](figures/model3_upgrade-full-20261115_knn_k_curve.png)

![Lỗi OOF theo kích thước](figures/model3_upgrade-full-20261115_oof_size_errors.png)

Các hình có bản dark cùng basename `_dark.png`; bảng số đầy đủ nằm trong tables. OOF class/group/error tables và quality thresholds có cùng tiền tố. Brightness/contrast không được dùng để thay đổi winner sau search.

Notebook thực thi: [03_model3_upgrade.ipynb](../notebooks/03_model3_upgrade.ipynb). Notebook execution errors=0; Run All chỉ đọc và xác minh artifacts, không retrain/inference.

Logs: [full console](../logs/model3_upgrade/upgrade-full-20261115/console.log), [artifact audit](../logs/model3_upgrade/upgrade-full-20261115/artifact_check.log), [export](../logs/model3_upgrade/upgrade-full-20261115/export.log), [verification](../logs/model3_upgrade/verification-upgrade-full-20261115-final/verification.json). Focused tests và notebook pass. Full suite còn lỗi PCA người2 có sẵn; không tuyên bố toàn suite pass.

Kết luận chỉ thuộc phạm vi grid, biểu diễn HOG cố định và một seed/ba folds; selected-CV có selection bias. Chưa chạy official test cho model mới và không thay điểm test cũ. Không xóa ảnh khó, không cân bằng lại test, không coi mô tả OOF là chứng minh nhân quả.
