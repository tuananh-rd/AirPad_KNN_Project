# AirPad: xác minh nguồn và baseline

Cập nhật: 2026-09-30. Đây là báo cáo của coordinator, không phải báo cáo brain.

## Kiểm kê nguồn

- Đã tìm đệ quy tên file và nội dung Markdown/JSON trong workspace AirPad_KNN_Project và thư mục cha `a/outputs`, mở rộng kiểm kê tài liệu tới `a`. Không tìm thấy báo cáo brain trong các thư mục này. Các tài liệu hiện có là README.md, AirPad_Progress.md, AirPad_STGCN_Migration.md và AirPad_Project_Spec.md; không có bằng chứng tác giả brain để coi chúng là báo cáo brain.
- Tìm bổ sung trong kho artifact Antigravity ngoài workspace: chỉ một báo cáo Markdown có nhắc AirPad, `C:/Users/tuana/.gemini/antigravity/brain/fb3dd143-7dad-4ef4-ae8a-2d94c7c21d4e/phase1_report.md`. Đã đọc toàn bộ. Mtime UTC: 2026-09-28T16:03:35.685469+00:00. Nội dung: Phase 1 KNN, 83 test, các rủi ro FSM/motion. Không xác nhận hướng ST-GCN hoặc baseline SVM. Đây là báo cáo AirPad duy nhất tìm thấy ở kho này, chưa thể xác nhận là báo cáo brain mới nhất mà người dùng nhắc tới.
- Agent Session History của Orca đang tắt; không có khả năng tìm hội thoại qua `orca search` trong phiên này.

## So sánh

| Nguồn | Nội dung | Đánh giá |
|---|---|---|
| Artifact brain Phase 1 ngoài workspace | KNN, 83 test, Phase 2 chưa hoàn thiện | Cũ hơn code hiện tại; không đủ làm căn cứ đổi kiến trúc |
| AirPad_Progress.md | Review 2A, 122 pass/1 fail; quyết định ST-GCN; giao G0 | Không phản ánh 207 test hiện tại và SVM hiện hữu |
| AirPad_STGCN_Migration.md | Causal ST-GCN 24 frame và các ticket G0–G5 | Chưa được báo cáo brain tìm thấy xác nhận; không kích hoạt |
| README.md | StandardScaler + RBF-SVM, grouped training, threshold rejection | Khớp đường chạy thực tế; tuyên bố không leakage cần sửa theo bằng chứng dưới đây |

Ưu tiên báo cáo brain mới nhất khi tìm được. Không dùng mtime README/code thay thế xác nhận từ brain.

## Baseline thực tế đã xác minh

- `src/ml/svm.py`: StandardScaler, SVC(kernel="rbf"), class_weight="balanced"; fit scaler trên train fold.
- `tools/train.py`: tuning C/gamma, bundle algorithm `rbf_svm`; loader khôi phục scaler và fit lại SVM từ X_train/y_train.
- `app.py:create_pipeline` gọi loader này; runtime canonicalize 63 đặc trưng và gọi predict_with_rejection. Geometry/FSM tiếp tục xử lý hành động.
- `config/default.yaml`: rbf_svm, C=10, gamma=scale, decision_threshold=0, margin_threshold=0.15. Bundle đã train dùng C=10, gamma=0.01; phân biệt config mặc định và tham số model thực tế.
- `models/train_summary.json`: 61.066 mẫu, 9 lớp, group_key=user_id, 9.175 ID, macro-F1=0.967276. Tính từ confusion matrix lưu sẵn: accuracy=0.990142, balanced accuracy=0.968835. Đây là số liệu lịch sử, chưa tái train hay xác minh unseen-user độc lập.
- `PINCH_MIDDLE`: precision=0.828859, F1=0.883721; `PINCH_CLOSED`: recall=0.904332, F1=0.934701; L_SHAPE F1=0.939024. Ưu tiên phân tích các lớp này.
- Kiểm chứng độc lập: `.venv/Scripts/python.exe -m pytest -q`: 207 passed in 51.23s. Test xanh không chứng minh độ chính xác thực tế.

## Sai lệch đánh giá đã xác minh

1. `scratch/prepare_combined_dataset.py` chia dữ liệu custom thành năm chunk theo lớp rồi đổi user_id thành `tuana_chunk0..4`, session_id cũng bị đổi. Kiểm tra trực tiếp NPZ xác nhận combined chứa các ID này; bản custom gốc chứa một user `tuana`, một session `sess_20260929_132034_7a13ed`, tổng 1.066 mẫu (554 PINCH_CLOSED, 261 PINCH_MIDDLE, 251 L_SHAPE). Chia theo các ID mới đưa cùng người/session thật vào train và validation. Không thể suy diễn unseen-user cho ba lớp này.
2. `tools/train.py` chọn hyperparameters và báo OOF trên cùng tập grouped folds. Không có outer holdout/nested CV độc lập để đánh giá lựa chọn tham số.
3. `extract_groups` tự sinh `unknown_i` khi thiếu group; cần fail hoặc báo thiếu provenance, không coi mỗi frame là người độc lập.
4. Metrics train dùng `predict`, runtime dùng rejection. `compute_metrics` bỏ qua cặp không nằm trong labels; nếu tái dùng với UNKNOWN mà label list chỉ có lớp train thì có thể loại false negatives. Cần đo rejection đầy đủ.

## Trạng thái điều phối

Ticket G0-PACKAGE đã được gửi trước chỉ dẫn bổ sung của người dùng. Coordinator đã gửi yêu cầu dừng chỉnh sửa và ngắt lượt worker. Worker trả `worker_done` cho đúng task/dispatch lúc 2026-09-30T15:54:48Z, outcome failed theo yêu cầu dừng, xác nhận chỉ đọc và không sửa file nào. Coordinator chấp nhận trạng thái dừng; không chấp nhận đây là hoàn thành package. Ticket accuracy mới chỉ là bản nháp, chưa dispatch. Không thay đổi ST-GCN/model/dataset trong review này.

Nguồn còn thiếu: vị trí/nội dung báo cáo brain mới nhất để xác nhận mục tiêu kiến trúc và đối chiếu ưu tiên.
