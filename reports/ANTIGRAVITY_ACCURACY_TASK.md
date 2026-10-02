# ACC-01 — Đánh giá RBF-SVM đáng tin cậy trước cải thiện độ chính xác

Trạng thái: DRAFT, CHƯA DISPATCH. Chờ xác định báo cáo brain mới nhất; không coi kế hoạch ST-GCN là quyết định hiện hành.

## Target và quyền chỉnh sửa

Baseline: StandardScaler + RBF-SVM trên 63 đặc trưng, chín static poses; geometry/FSM là tầng quyết định hành động. Worker Antigravity chịu trách nhiệm `tools/train.py`, `src/ml/evaluation.py`, công cụ audit/evaluate mới, test tương ứng và tài liệu đánh giá. Coordinator chạy kiểm chứng độc lập và review. Không chỉnh package, FSM, motion, controller, production config, model đã có hoặc dataset gốc.

## Kết quả cần tạo

1. Audit provenance của ba NPZ hiện có; đối chiếu features/labels/metadata custom gốc và combined. Khôi phục group thật trong bản metadata riêng sau khi xác minh đối chiếu; không tự đoán người bằng cách xóa hậu tố ID. Ghi nguồn dữ liệu, hash, số mẫu/lớp/người/session và các cảnh báo.
2. Tạo split manifest cố định theo người thật; nếu dùng session split thì nêu rõ là đánh giá người đã biết, không unseen-user. Mọi frame cùng người thật chỉ nằm ở một split. Group thiếu/rỗng phải fail rõ ràng. Không cắt session thành người giả để đủ folds.
3. Tách train/validation/test hoặc outer grouped folds + inner grouped tuning. Scaler, chọn C/gamma và chọn thresholds chỉ học trên train/validation của outer fold. Test chỉ dùng một lần sau khi đóng lựa chọn. Ghi fingerprint split và tham số để tái lập.
4. Xuất baseline và candidate RBF-SVM trên cùng split; tìm kiếm C/gamma/class_weight nhỏ, có giới hạn. Candidate output nằm ở đường dẫn mới. Không ghi đè production bundle. Không huấn luyện ST-GCN.
5. Xuất macro-F1 trên chín lớp cố định, balanced accuracy, accuracy, per-class precision/recall/F1/support, confusion matrix, coverage và rejection rate, cả trước và sau rejection. UNKNOWN được giữ như dự đoán sai/rejection trong mẫu số; không loại các mẫu này để tăng điểm. Báo tỷ lệ dự đoán pinch sai trên negative poses. Chỉ gọi là false command activation khi thực sự đo event qua replay FSM dry-run.
6. Phân tích PINCH_CLOSED ↔ PINCH_MIDDLE và L_SHAPE. Thiếu dữ liệu người/session độc lập cho ba lớp custom phải báo unavailable và đề xuất số lượng/đa dạng dữ liệu cần thu; không bật webcam hoặc tự thu trong ticket. Có thể đo sáu lớp HaGRID và báo ba lớp chưa đủ bằng chứng, không quảng bá thành kết quả chín lớp.

## Tiêu chí nghiệm thu cố định trước chạy

- Tất cả 207 test cũ đạt; test mới chứng minh người/session thật không lọt giữa splits, missing groups bị từ chối, scaler/tuning không đọc test và UNKNOWN làm tăng false negatives đúng cách.
- Artifact chứa hash dataset/split, seed, tham số, fold support và dự đoán từng mẫu để coordinator tính lại metrics. Cùng input/seed cho cùng split và dự đoán.
- Baseline cũ 0.967276 chỉ được gắn nhãn lịch sử có leakage, không làm ngưỡng thắng cho split sạch mới.
- Chỉ tuyên bố candidate cải thiện khi macro-F1 tăng trên cùng outer test/holdout, báo delta và bất định theo nhóm, recall/precision từng lớp được đối chiếu, tỷ lệ false pinch activation không tăng tại coverage tương đương. Không tune lại sau khi xem test. Kết quả không tăng vẫn là bằng chứng hợp lệ; không promote model và mở ticket dữ liệu/sửa tiếp.
- Với các lớp chỉ có một người/session, phải ghi chưa đủ dữ liệu để chứng minh unseen-user; đó là blocker của accuracy claim, không được bỏ qua hoặc chia nhóm giả.
- Safety invariant giữ nguyên; không dùng webcam, OS input thật, cài dependencies hoặc thay model trả phí. Worker báo file đã đổi, kết quả/chưa chạy và blocker; coordinator chạy test.

## Review và bước tiếp theo

Coordinator kiểm tra split/provenance trước khi cho phép train lớn; tính lại metrics từ prediction records và so sánh baseline/candidate. Nếu leakage, metrics bỏ UNKNOWN hoặc thiếu reproducibility: yêu cầu worker sửa trong ticket tiếp theo. Nếu dữ liệu thiếu: giữ baseline, lập ticket thu dữ liệu có chủ ý; không chuyển kiến trúc khi chưa được báo cáo brain xác nhận.
