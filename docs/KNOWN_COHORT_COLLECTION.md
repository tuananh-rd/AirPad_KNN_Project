# Hướng Dẫn Thu Thập Dữ Liệu Cử Chỉ AirPad Cho Known-Cohort & Personal Use

Tài liệu này hướng dẫn quy trình thu thập dữ liệu cử chỉ bàn tay từ webcam cho dự án **AirPad**, áp dụng cho đối tượng triển khai mục tiêu: **tuana** (chế độ sử dụng cá nhân) hoặc **nhóm bạn bè thân thiết** (known-cohort).

---

## 1. Nguyên Tắc & Tiêu Chuẩn Độc Lập Dữ Liệu (Session Independence)

Để đảm bảo mô hình nhận diện cử chỉ hoạt động tin cậy và có thể kiểm chứng khách quan mà không bị rò rỉ dữ liệu (data leakage), quy trình thu thập tuân thủ các nguyên tắc:

1. **Tối thiểu 3 phiên thu độc lập cho MỖI người tham gia**:
   - Mỗi người (ví dụ: `tuana`, `friend_01`, ...) phải thực hiện tối thiểu 3 phiên thu riêng biệt (`session_id`).
   - 3 phiên này được phân bổ độc lập cho 3 vai trò:
     - **Tập Huấn Luyện (Train)**: Dùng để khớp mô hình nhận diện.
     - **Tập Thẩm Định (Validation)**: Dùng để tinh chỉnh siêu tham số và ngưỡng tin cậy.
     - **Tập Kiểm Thử Độc Lập (Test)**: Đánh giá độ chính xác trên các phiên độc lập chưa từng thấy lúc train.
2. **Không chia sẻ phiên qua các tập (Disjoint Sessions)**:
   - Mỗi phiên định danh bằng cặp `(user_id, session_id)` chỉ được gán duy nhất vào một tập (Train, Validation HOẶC Test).
   - Tuyệt đối không cắt nhỏ 1 phiên dài thành nhiều phần để gán vào cả Train và Test.
3. **Phủ đầy đủ 9 cử chỉ tĩnh trong từng vai trò**:
   - Mỗi vai trò (Train, Validation, Test) đều phải có đầy đủ mẫu ổn định của toàn bộ 9 tư thế tĩnh.
4. **Tách biệt khung hình chuyển tiếp (`is_transition`)**:
   - Khi tay bắt đầu di chuyển, đổi dáng hoặc hạ xuống, các khung hình đó là khung hình chuyển tiếp. Chúng được gắn cờ `is_transition = True` và được loại trừ khỏi tập huấn luyện/kiểm thử tư thế tĩnh.

---

## 2. Quy Định Định Danh (User ID & Session ID)

- **Quy ước đặt mã người dùng (`user_id`)**:
  - Người dùng chính: `tuana`
  - Các bạn tham gia: đặt mã ẩn danh nhất quán, ví dụ: `friend_01`, `friend_02`...
  - Không sử dụng tên thật đầy đủ hoặc thông tin cá nhân nhạy cảm trong `user_id`.
- **Quy ước đặt mã phiên (`session_id`)**:
  - Công cụ thu thập tự động sinh mã định dạng `sess_YYYYMMDD_HHMMSS_<hash>` nếu để trống.
  - Hoặc có thể chỉ định rõ ràng khi chạy lệnh:
    - Ví dụ cho `tuana`: `sess_tuana_s1_bright`, `sess_tuana_s2_dim`, `sess_tuana_s3_mixed`.
    - Ví dụ cho `friend_01`: `sess_f01_s1_desk`, `sess_f01_s2_couch`, `sess_f01_s3_evening`.

---

## 3. Danh Mục 9 Tư Thế Tĩnh Mục Tiêu & Phím Tắt

Hệ thống AirPad hỗ trợ 9 tư thế tĩnh chuẩn với các phím số từ `[1]` đến `[9]`:

| Phím | Tên Cử Chỉ (`label`) | Mô Tả Động Tác |
| :---: | :--- | :--- |
| `[1]` | `PINCH_CLOSED` | Đầu ngón cái và đầu ngón trỏ chạm vào nhau tạo thành vòng khép kín. |
| `[2]` | `PINCH_MIDDLE` | Đầu ngón cái và đầu ngón giữa chạm vào nhau tạo thành vòng khép kín. |
| `[3]` | `L_SHAPE` | Ngón cái và ngón trỏ duỗi thẳng vuông góc (hình chữ L), 3 ngón còn lại gập vào lòng bàn tay. |
| `[4]` | `OTHER` | Bàn tay ở trạng thái nghỉ, tư thế tự nhiên ngoài từ điển cử chỉ (dùng để giảm báo giả). |
| `[5]` | `POINT_ONE` | Chỉ duỗi duy nhất 1 ngón trỏ chỉ thẳng, ngón cái và các ngón khác gập lại. |
| `[6]` | `OPEN_PALM` | Lòng bàn tay mở hoàn toàn, cả 5 ngón duỗi thẳng tự nhiên. |
| `[7]` | `FIST` | Nắm đấm, cả 5 ngón tay gập chặt vào lòng bàn tay. |
| `[8]` | `TWO_FINGERS_PARALLEL`| Ngón trỏ và ngón giữa cùng duỗi thẳng song song, các ngón khác gập. |
| `[9]` | `THREE_FINGERS` | Ba ngón (trỏ, giữa, áp út) duỗi thẳng, ngón cái giữ ngón út. |

---

## 4. Biến Thể Thu Thập Để Tăng Độ Bền Vững (Robustness)

Giữa các phiên thu độc lập của cùng một người, nên thay đổi có chủ đích các yếu tố sau:

1. **Điều kiện ánh sáng (`lighting`)**:
   - `bright`: Ánh sáng ban ngày tự nhiên qua cửa sổ hoặc đèn bàn chiếu sáng rõ.
   - `dim`: Ánh sáng phòng buổi tối, ánh sáng gián tiếp hoặc đèn ngủ.
   - `mixed`: Ánh sáng hỗn hợp, ánh sáng chiếu từ một bên hoặc ngược sáng nhẹ từ màn hình.
2. **Khoảng cách từ tay đến webcam (`distance_cm`)**:
   - Gần: ~30 - 35 cm.
   - Vừa: ~45 - 50 cm (khoảng cách ngồi gõ bàn phím chuẩn).
   - Xa: ~60 - 70 cm.
3. **Góc nghiêng và xoay cổ tay**:
   - Giữ tư thế chuẩn thẳng góc với camera, nghiêng nhẹ sang trái/phải 15-20 độ, hoặc hướng chếch lên/xuống.
4. **Tay thuận và tay không thuận (`handedness`)**:
   - Thu thập cả hai tay chỉ khi ứng dụng thực tế của bạn dự kiến cần tương tác bằng cả hai tay. Nếu chỉ dùng một tay (ví dụ tay phải điều khiển chuột ảo), chỉ cần tập trung thu thập cho tay đó.
   - *Lưu ý*: Số lượng mẫu mục tiêu là hướng dẫn lập kế hoạch thu thập, không phải là cam kết đảm bảo tuyệt đối về độ chính xác mô hình.

---

## 5. Hướng Dẫn Thao Tác Bằng Công Cụ `tools.collect_data`

### 5.1. Khởi chạy công cụ bằng lệnh module Python chuẩn

Chạy lệnh trong terminal thông qua cú pháp module `-m tools.collect_data` để đảm bảo nạp đúng môi trường:

```powershell
# Xem trợ giúp CLI (không bật webcam)
.\.venv\Scripts\python.exe -m tools.collect_data --help

# Thu phiên 1 cho tuana (ánh sáng sáng, khoảng cách 45cm)
.\.venv\Scripts\python.exe -m tools.collect_data --output data/cohort/tuana_sess1.npz --user-id tuana --session-id sess_tuana_s1 --distance 45 --lighting bright

# Thu phiên 2 cho tuana (ánh sáng phòng tối)
.\.venv\Scripts\python.exe -m tools.collect_data --output data/cohort/tuana_sess2.npz --user-id tuana --session-id sess_tuana_s2 --distance 40 --lighting dim

# Thu phiên 3 cho tuana (khoảng cách xa hơn, ánh sáng hỗn hợp)
.\.venv\Scripts\python.exe -m tools.collect_data --output data/cohort/tuana_sess3.npz --user-id tuana --session-id sess_tuana_s3 --distance 60 --lighting mixed
```

*Cơ chế an toàn (Fail-closed)*: Nếu đường dẫn `--output` trỏ tới file đã tồn tại nhưng bị hỏng hoặc lỗi định dạng, công cụ sẽ lập tức báo lỗi và dừng (exit code 1) trước khi mở camera hay ghi dữ liệu, bảo vệ file hiện có không bị ghi đè.

### 5.2. Các phím điều khiển trong giao diện thu thập

- `[1]` .. `[9]`: Chọn cử chỉ mục tiêu cần thu.
- `[SPACE]` (Phím cách): Chụp và lưu **1 khung hình lẻ** khi bàn tay đã vào vị trí chuẩn.
- `[R]`: Bật/tắt chế độ **ghi liên tục** (Continuous Recording).
  - *Khuyến nghị*: Định vị bàn tay chuẩn -> Nhấn `[R]` để ghi chuỗi ngắn (30-50 frames trong 2-3 giây) kèm lắc nhẹ góc tay -> Nhấn `[R]` để dừng.
- `[T]`: Bật/tắt cờ **chuyển tiếp (`is_transition`)**.
  - Khi cần chuyển đổi giữa hai tư thế hoặc hạ tay xuống, bật `[T]`. HUD hiển thị `TRANSITION: YES`.
- `[S]`: Lưu ngay dataset ra đĩa mà không cần thoát.
- `[ESC]` hoặc `[Q]`: Lưu toàn bộ dữ liệu phiên thu và đóng ứng dụng.

---

## 6. Bảo Vệ Dữ Liệu & Lưu Trữ Cục Bộ

- **Định dạng dữ liệu**: Công cụ AirPad trích xuất toạ độ 21 điểm mốc 3D của MediaPipe (`(x, y, z)` chuẩn hóa, tổng cộng 63 số thực float32) và siêu dữ liệu phiên thu, lưu vào file `.npz` nén trên đĩa cục bộ.
- **Không lưu hình ảnh khuôn mặt**: Không có module nào lưu trữ video webcam hay hình ảnh khuôn mặt người dùng vào file dataset.

---

## 7. Kiểm Tra Độ Sẵn Sàng (Readiness Audit) Bằng `tools.cohort_sessions`

Trình thông dịch Python native trên Windows không tự động bung ký tự đại diện wildcard (`*`). Khi truyền nhiều file trong PowerShell, hãy liệt kê rõ ràng hoặc dùng mảng PowerShell:

### 7.1. Cú pháp cơ bản và kiểm tra trợ giúp CLI

```powershell
# Xem trợ giúp CLI
.\.venv\Scripts\python.exe -m tools.cohort_sessions --help
```

### 7.2. Kiểm tra bộ dữ liệu hiện có (data/custom_airpad.npz)

Bộ dữ liệu hiện có chỉ có 1 phiên duy nhất (`tuana::sess_20260929_132034_7a13ed`) và chỉ có 3/9 cử chỉ tĩnh. Khi kiểm tra với kế hoạch gán phiên này cho `train`, công cụ báo `NOT_READY`:

```powershell
# File kế hoạch reports/acc02/plan_existing_custom.json:
# {"tuana::sess_20260929_132034_7a13ed": "train"}

.\.venv\Scripts\python.exe -m tools.cohort_sessions --data data/custom_airpad.npz --roster tuana --mode personal --plan reports/acc02/plan_existing_custom.json --output reports/acc02/readiness_existing_custom.json
```

Kết quả: Báo cáo `NOT_READY` nêu rõ lý do: thiếu 2 phiên độc lập (chỉ có 1/3 phiên tối thiểu), thiếu 6 cử chỉ tĩnh trong tập train, và các tập validation/test hoàn toàn không có phiên nào.

### 7.3. Kiểm tra bộ dữ liệu mới đạt chuẩn READY

Khi đã thu đủ tối thiểu 3 phiên độc lập cho mỗi người tham gia, hãy chuẩn bị file kế hoạch phân bổ phiên theo cặp `(user_id, session_id)` (ví dụ `plan_personal.json`):

```json
{
  "tuana::sess_tuana_s1": "train",
  "tuana::sess_tuana_s2": "validation",
  "tuana::sess_tuana_s3": "test"
}
```

Chạy kiểm tra độ sẵn sàng với danh sách file rõ ràng:

```powershell
# Chế độ cá nhân (personal mode)
.\.venv\Scripts\python.exe -m tools.cohort_sessions --data data/cohort/tuana_sess1.npz data/cohort/tuana_sess2.npz data/cohort/tuana_sess3.npz --roster tuana --mode personal --plan plan_personal.json --output reports/cohort_readiness_tuana.json --manifest-out reports/manifest_tuana.json
```

Hoặc dùng mảng PowerShell an toàn cho nhiều file:

```powershell
$dataFiles = (Get-Item data/cohort/*.npz | Select-Object -ExpandProperty FullName)
.\.venv\Scripts\python.exe -m tools.cohort_sessions --data $dataFiles --roster tuana,friend_01 --mode cohort --plan plan_cohort.json --output reports/cohort_readiness_group.json --manifest-out reports/manifest_cohort.json
```
