# Sổ Tay Hướng Dẫn Sử Dụng Cử Chỉ AirPad (Vietnamese Function Guide)

Tài liệu này cung cấp hướng dẫn tra cứu chi tiết và quy tắc vận hành cho hệ thống điều khiển chuột ảo **AirPad** bằng cử chỉ bàn tay qua webcam trên Windows.

> [!IMPORTANT]
> **Phân định cấu hình sử dụng**:
> - **Mục 1 đến Mục 8** mô tả cấu hình điều khiển chuột ảo di sản (**Legacy Air-Mouse Profile**, `--profile legacy`), trong đó camera đảm nhận toàn bộ việc di chuyển con trỏ, nhấp và kéo chuột.
> - Người dùng cấu hình trợ năng kết hợp chuột vật lý (**Hybrid Assist Profile**, `--profile hybrid`), vui lòng chuyển trực tiếp đến **[Mục 9: Cấu Hình Trợ Năng Kết Hợp Chuột Vật Lý (Hybrid Assist Profile: HYB-01)](#9-cấu-hình-trợ-năng-kết-hợp-chuột-vật-lý-hybrid-assist-profile-hyb-01)**.

---

## 1. Hướng dẫn khởi chạy (Execution Commands)

> [!IMPORTANT]
> Mô hình sản xuất mặc định (`models/airpad_svm.json`) là mô hình ban đầu với độ chính xác cơ bản và **chưa** được thăng cấp (unpromoted). Để đạt độ nhạy và nhận diện tối ưu, luôn chỉ định tệp mô hình cá nhân hóa đã được kiểm định tại `reports/acc05/candidate_personal_svm.json` và sử dụng trình thông dịch Python trong môi trường ảo `.venv`:

- **Chế độ kiểm thử an toàn (Dry-Run / Không can thiệp chuột thật)**:
  ```powershell
  .venv\Scripts\python.exe app.py --camera --model reports/acc05/candidate_personal_svm.json
  ```
- **Chế độ điều khiển hệ điều hành thực tế (Real OS Control)**:
  ```powershell
  .venv\Scripts\python.exe app.py --camera --real-control --model reports/acc05/candidate_personal_svm.json
  ```

---

## 2. Hệ Tọa Độ & Không Gian Camera

- **Xem trước gương tự nhiên (Mirrored Preview)**: Khung hình webcam được lật ngang qua `cv2.flip(frame, 1)`.
- **Ánh xạ con trỏ**: Tọa độ đầu ngón trỏ (Landmark 8) trên khung hình xem trước được ánh xạ trực tiếp sang tọa độ màn hình:
  - Đưa tay sang **phải** $\rightarrow$ Tọa độ $x$ khung hình tăng $\rightarrow$ Con trỏ chuột di chuyển sang **phải**.
  - Đưa tay sang **trái** $\rightarrow$ Tọa độ $x$ khung hình giảm $\rightarrow$ Con trỏ chuột di chuyển sang **trái**.
  - Đưa tay lên **trên** $\rightarrow$ Tọa độ $y$ khung hình giảm $\rightarrow$ Con trỏ chuột di chuyển lên **trên**.
  - Đưa tay xuống **dưới** $\rightarrow$ Tọa độ $y$ khung hình tăng $\rightarrow$ Con trỏ chuột di chuyển xuống **dưới**.
- **Vùng hoạt động hiệu dụng (Active ROI)**: Mặc định là hình chữ nhật viền cam từ 10% đến 90% kích thước khung hình (`roi_x_min=0.1`, `roi_x_max=0.9`, `roi_y_min=0.1`, `roi_y_max=0.9`). Khi đầu ngón tay chạm mép ROI, con trỏ chạm cạnh màn hình tương ứng.

---

## 3. Quy Tắc Vũ Trang Trạng Thái (Arming Requirement)

> [!NOTE]
> Máy trạng thái FSM áp dụng cơ chế **vũ trang bắt buộc (`POINT_ONE` arming)**:
> 1. Để bắt đầu điều khiển từ trạng thái `PAUSED`, người dùng phải giữ tư thế `POINT_ONE` ổn định liên tục $\ge 100\text{ms}$ (`point_one_stable_ms`), đồng thời vượt qua bộ lọc đa số 5 khung hình (`MajorityFilter`) và cổng ổn định 2 khung hình (`StabilityGate`).
> 2. FSM chuyển sang `MOVING` và tích lũy trạng thái tái nhận diện (`_reacquire_count >= 2`).
> 3. Chỉ khi FSM đang ở `MOVING` và đã vũ trang thành công (`is_reacquired == True`), các cử chỉ như chụm ngón click (`CLICK_PENDING`), click phải (`RIGHT_CLICK_PENDING`), cuộn (`SCROLLING`), hay thu phóng (`ZOOM_DWELL`) mới được phép kích hoạt.

---

## 4. Bảng Cử Chỉ & Hành Động Chi Tiết (Gesture Function Table)

| Cử chỉ (`HandPose`) | Hình dạng bàn tay | Điều kiện kích hoạt & Ngưỡng thời gian | Hành động điều khiển (`Action`) | Trạng thái FSM |
| :--- | :--- | :--- | :--- | :--- |
| **`POINT_ONE`** | Ngón trỏ duỗi thẳng, các ngón khác gập | Đã vũ trang (`is_reacquired == True`) và duy trì `POINT_ONE` | **Di chuyển con trỏ chuột** theo đầu ngón trỏ (Landmark 8) | Duy trì `MOVING` |
| **`PINCH_CLOSED`** *(Nhấp ngắn)* | Đầu ngón cái chạm đầu ngón trỏ rồi mở ra | Đã vũ trang từ `MOVING`; thời gian giữ chạm $80\text{ms} \le \Delta t \le 300\text{ms}$ rồi nhả dứt khoát về `POINT_ONE` | **Chuột trái (`LEFT_CLICK`)** (giãn cách chống dội $50\text{ms}$) | `MOVING` $\rightarrow$ `CLICK_PENDING` $\rightarrow$ `MOVING` |
| **`PINCH_CLOSED`** *(Kéo thả)* | Đầu ngón cái chạm đầu ngón trỏ và giữ nguyên | Đã vũ trang từ `MOVING`; thời gian giữ chạm $\ge 400\text{ms}$ (`drag_threshold_ms`) | **Nhấn giữ chuột trái (`LEFT_DOWN`)**; di chuyển ngón tay để kéo đối tượng; nhả về `POINT_ONE` phát **thả chuột (`LEFT_UP`)** | `CLICK_PENDING` $\rightarrow$ `DRAGGING` $\rightarrow$ `MOVING` |
| **`PINCH_MIDDLE`** | Đầu ngón giữa chạm đầu ngón cái rồi mở ra | Đã vũ trang từ `MOVING`; thời gian giữ chạm $80\text{ms} \le \Delta t \le 300\text{ms}$ rồi nhả dứt khoát về `POINT_ONE` | **Chuột phải (`RIGHT_CLICK`)** (giãn cách chống dội $50\text{ms}$) | `MOVING` $\rightarrow$ `RIGHT_CLICK_PENDING` $\rightarrow$ `MOVING` |
| **`TWO_FINGERS_PARALLEL`** | Ngón trỏ và ngón giữa duỗi thẳng song song | Đã vũ trang từ `MOVING`; di chuyển cả bàn tay theo phương thẳng đứng | **Cuộn trang theo cơ chế hiện tại**: <br>• Di chuyển tay **xuống dưới** $\rightarrow$ Cuộn trang **lên trên** (`wheel up`) <br>• Di chuyển tay **lên trên** $\rightarrow$ Cuộn trang **xuống dưới** (`wheel down`) | `MOVING` $\rightarrow$ `SCROLLING` $\rightarrow$ `MOVING` |
| **`L_SHAPE`** | Ngón cái và ngón trỏ mở rộng vuông góc chữ L | Đã vũ trang từ `MOVING`; giữ yên tư thế $\ge 250\text{ms}$ (`dwell_zoom_ms`) để khóa trạng thái, sau đó co/giãn khẩu độ hai ngón | **Thu phóng ứng dụng (`Ctrl + Con lăn`)**: Phụ thuộc vào ứng dụng đang kích hoạt (trình duyệt, tài liệu văn bản); co hai ngón = thu nhỏ, xòe rộng hai ngón = phóng to. Không phải kính lúp toàn hệ thống (Magnifier). | `MOVING` $\rightarrow$ `ZOOM_DWELL` $\rightarrow$ `ZOOM_LATCHED` |
| **`THREE_FINGERS`** | Ba ngón tay (trỏ, giữa, áp út) duỗi thẳng | Đang ở `MOVING`, kết hợp gạt cổ tay ngang dứt khoát | **Điều hướng tiến / lùi**: <br>• Gạt sang trái: `Alt + Mũi tên trái` (Back) <br>• Gạt sang phải: `Alt + Mũi tên phải` (Forward) | Kích hoạt `SWIPE_BACK` hoặc `SWIPE_FORWARD` trong `MOVING` |
| **`OPEN_PALM`** | Toàn bộ 5 ngón tay mở rộng hoàn toàn | Cung cấp bởi bộ lọc thời gian (`MajorityFilter` và `StabilityGate`) | **Ngắt an toàn**: FSM lập tức gọi `_force_cleanup()` để nhả các phím/nút chuột đang giữ và chuyển về `PAUSED`. *Lưu ý: Hành động kích hoạt sau khi tư thế vượt qua bộ lọc thời gian, không tức thì tại thời điểm cơ học xòe tay.* | Ép FSM về `PAUSED` |
| **`FIST`** | Nắm chặt toàn bộ các ngón tay | Thực hiện khi đang ở trạng thái `MOVING` | **Tạm khóa con trỏ**: Đặt lại bộ đếm tái nhận diện (`_reacquire_count = 0`), tạm dừng di chuyển con trỏ; **không** ép FSM thoát về `PAUSED` như `OPEN_PALM`. | Duy trì `MOVING` (chặn xuất sự kiện di chuyển) |

---

## 5. Trạng Thái Không Xác Định & Mất Dấu

1. **`UNKNOWN` / `OTHER` (Nhận diện không chắc chắn)**:
   - Các tư thế này cũng phải đi qua bộ lọc thời gian.
   - Khi ở trạng thái `MOVING`, xuất hiện `UNKNOWN` hoặc `OTHER` sẽ tạm dừng di chuyển con trỏ và tăng bộ đếm không xác định (`_unknown_count`).
   - Nếu duy trì liên tục $\ge 5$ khung hình (`unknown_timeout_frames = 5`) khi đang trong `MOVING`, FSM sẽ kích hoạt `_force_cleanup()` và chuyển về `PAUSED`. Quy tắc timeout 5 khung hình này áp dụng trong `MOVING`, không đảm bảo chuyển `PAUSED` trong mọi trạng thái FSM khác.
2. **`NO_HAND` (Mất dấu bàn tay)**:
   - Khi MediaPipe không tìm thấy bàn tay (`landmarks is None`), FSM lập tức gọi `_force_cleanup()` giải phóng các nút bấm/phím giữ và chuyển về `PAUSED`.

---

## 6. Phím Tắt Khẩn Cấp & Giới Hạn An Toàn (Failsafe & Boundaries)

1. **Phím tắt dừng khẩn cấp (`Ctrl + Alt + Q`) & Hợp đồng nghiệm thu an toàn**:
   - Được đăng ký thông qua hook bàn phím mức thấp trên Windows.
   - **Cơ chế từ chối an toàn (Fail-Closed Admission)**: Khi khởi chạy với cờ `--real-control`, hệ thống bắt buộc kiểm tra và đăng ký thành công hook phím tắt khẩn cấp. Nếu thư viện `keyboard` bị thiếu hoặc đăng ký hook thất bại, chương trình lập tức từ chối khởi động và hoàn tác mọi tài nguyên một cách an toàn.
   - **Đồng bộ hóa an toàn qua `SafetyGate`**:
     - Lệnh dừng khẩn cấp chốt trạng thái `is_stopped = True` ngay tức thì.
     - Toàn bộ hành vi điều khiển tiến tới (`mouse_move`, `mouse_click`, `mouse_down`, `scroll`, `zoom`, `swipe`, `key_down`) bị từ chối triệt để ngay sau khi dừng.
     - Hành vi giải phóng (`mouse_up`, `key_up`, `cleanup`) được phép thực thi và đồng bộ hóa tuyến tính: các tác vụ đang thực thi dở dang hoàn tất ghi nhận trạng thái trước khi chuỗi giải phóng chạy.
     - **Bảo lưu và thử lại khi lỗi giải phóng**: Nếu một nút chuột hoặc phím bấm gặp lỗi khi nhả trong lúc dọn dẹp, hệ thống không bỏ qua mà tiếp tục lưu giữ trong danh sách phím đang giữ để thử lại ở các lần dọn dẹp tiếp theo.
   - **Khóa dừng vĩnh viễn (`Latched Stop`)**: Trạng thái dừng khẩn cấp này **không thể phục hồi thông qua bất kỳ cử chỉ bàn tay nào**. Người dùng bắt buộc phải đóng ứng dụng bằng phím `Q` hoặc `Esc` và khởi động lại chương trình để tiếp tục sử dụng.
   - *Giới hạn kỹ thuật*: Việc đăng ký phụ thuộc vào quyền hook bàn phím của hệ điều hành. Nếu luồng camera hoặc tiến trình chính bị treo sâu ở tầng driver hệ thống, phím tắt không thể bảo đảm an toàn phần cứng tuyệt đối.
2. **Góc dừng an toàn (`Failsafe Corner`)**:
   - Kiểm tra dựa trên **tọa độ pipeline đã ánh xạ** (`mapped_pos`), không phải tọa độ thực tế của con trỏ chuột vật lý.
   - Mặc định trong lớp `FailsafeMonitor` là phạm vi 10 pixel (`corner_size = 10`), và được khởi tạo trong `app.py` là 15 pixel (`corner_size = 15`).
   - Nếu tọa độ ánh xạ nằm trong vùng góc màn hình này, hệ thống sẽ tự động kích hoạt dừng khẩn cấp qua `SafetyGate`.
3. **Phím đóng giao diện (`Q` hoặc `Esc`)**:
   - Nhấn `Q` hoặc `Esc` khi cửa sổ xem trước OpenCV "AirPad Virtual Pad" đang có tiêu điểm (focus) để đóng camera và kết thúc chương trình.

---

## 7. Lưu Ý Hiển Thị Trên Màn Hình Xem Trước (HUD Disclaimer)

- Dòng chữ `Action: ...` trên HUD chỉ hiển thị giá trị được gán cho `PipelineResult.action_fired`.
- Các hành vi tương tác chuột (click trái, click phải, nhấn chuột kéo thả) do FSM phát trực tiếp tới bộ điều khiển hệ thống (`PlatformController`). Do đó, HUD có thể không nhấp nháy chữ "LEFT_CLICK" ở từng khung hình tương ứng.
- Người dùng nên theo dõi trạng thái **`FSM: ...`** trên HUD (`MOVING`, `CLICK_PENDING`, `DRAGGING`, `SCROLLING`, `ZOOM_LATCHED`) để xác nhận hệ thống đã nhận diện chính xác cử chỉ.

---

## 8. Hướng Dẫn Kiểm Thử An Toàn Cho Người Mới Bắt Đầu

1. **Bước 1: Chạy thử ở chế độ Dry-Run**:
   - Sử dụng lệnh kiểm thử mặc định (không có `--real-control`). Quan sát màn hình xem trước để làm quen với nhịp điệu của các cử chỉ, độ trễ vũ trang `POINT_ONE`, và các ngưỡng thời gian chụm ngón.
2. **Bước 2: Chạy thực tế trên cửa sổ an toàn**:
   - Khi kích hoạt `--real-control`, hãy mở một cửa sổ văn bản trống (ví dụ: **Notepad**) hoặc một tab trình duyệt trống.
   - Thử nghiệm đặt con trỏ bằng `POINT_ONE`, nhấp ngắn để đặt con trỏ soạn thảo, chụm giữ $\ge 400\text{ms}$ để bôi đen văn bản, và mở rộng bàn tay (`OPEN_PALM`) để tạm dừng an toàn trước khi áp dụng vào các tác vụ công việc quan trọng.

---

## 9. Cấu Hình Trợ Năng Kết Hợp Chuột Vật Lý (Hybrid Assist Profile: HYB-01)

> [!TIP]
> Cấu hình `--profile hybrid` phân định rõ vai trò điều khiển: chuột vật lý trên bàn giữ quyền kiểm soát độc quyền đối với vị trí con trỏ (`MOVE`), nhấp chuột trái/phải (`CLICK`), và kéo thả (`DRAGGING`). Webcam chỉ cung cấp trợ năng thứ cấp cho 2 thao tác: cuộn trang và chuyển tab ứng dụng.

### 9.1. Lệnh khởi chạy
- **Chế độ kiểm thử an toàn (Dry-Run)**:
  ```powershell
  .venv\Scripts\python.exe app.py --camera --profile hybrid --model reports/acc05/candidate_personal_svm.json
  ```
- **Chế độ điều khiển thực tế (Real OS Control)**:
  ```powershell
  .venv\Scripts\python.exe app.py --camera --profile hybrid --real-control --model reports/acc05/candidate_personal_svm.json
  ```

### 9.2. Cơ chế Bật / Tắt an toàn (`SPACE` Toggle)
- Khi khởi động, trợ năng cử chỉ luôn ở trạng thái **`OFF`** để ngăn ngừa can thiệp ngoài ý muốn.
- Nhấn phím **`SPACE` (phím cách)** trên bàn phím khi cửa sổ xem trước OpenCV ("AirPad Virtual Pad") đang có tiêu điểm để chuyển đổi giữa `OFF` và `READY`.
- **Hành vi khi `OFF`**: Trạng thái `OFF` nghiêm cấm phát sinh mọi hành vi điều khiển tiến tới mới (`mouse_move`, `mouse_click`, `scroll`, `switch_tab`, `key_down`), nhưng các thao tác giải phóng phím do chính controller sở hữu và quy trình dọn dẹp an toàn (`key_up`, `cleanup`) vẫn được phép thực thi để đảm bảo an toàn phần cứng.
- **Khi dừng khẩn cấp (`Ctrl + Alt + Q`)**: Trạng thái bị chốt chặn vĩnh viễn (`EMERGENCY_STOPPED`), phím `SPACE` không thể mở lại.

### 9.3. Bảng cử chỉ trợ năng Hybrid
| Cử chỉ | Tư thế bàn tay | Cơ chế hoạt động & Hướng | Phím / Thao tác hệ thống |
| :--- | :--- | :--- | :--- |
| **Cuộn dọc** | `TWO_FINGERS_PARALLEL` (2 ngón trỏ & giữa song song) | **Trực tiếp, không cần vũ trang `POINT_ONE`**. <br>• Đưa tay **LÊN** trong khung hình gương $\rightarrow$ Cuộn **LÊN** (`wheel up`) <br>• Đưa tay **XUỐNG** trong khung hình gương $\rightarrow$ Cuộn **XUỐNG** (`wheel down`) | `mouse.scroll(0, steps)` |
| **Chuyển Tab** | `THREE_FINGERS` (3 ngón trỏ, giữa, áp út duỗi thẳng) | **Kích hoạt 1 lần (One-shot) + Hồi chiêu 400ms**. <br>• Gạt cổ tay sang **TRÁI** trong khung hình gương $\rightarrow$ Tab trước <br>• Gạt cổ tay sang **PHẢI** trong khung hình gương $\rightarrow$ Tab sau | • Tab trước: `Ctrl + Shift + Tab` <br>• Tab sau: `Ctrl + Tab` |

### 9.4. Hướng dẫn vận hành & Quy tắc an toàn chi tiết
1. **Tiêu điểm ứng dụng tiền cảnh (Foreground App Focus)**:
   - Tổ hợp phím tắt chuyển tab (`Ctrl + Tab` / `Ctrl + Shift + Tab`) tác động trực tiếp vào **ứng dụng tương thích đang ở tiền cảnh (FOREGROUND)** (ví dụ: trình duyệt Chrome, Edge hoặc trình biên soạn VS Code).
   - Vì phím `SPACE` chỉ chuyển đổi trạng thái khi cửa sổ xem trước OpenCV đang có tiêu điểm, **người dùng cần sử dụng chuột vật lý để nhấp chọn ứng dụng mục tiêu** sau khi nhấn `SPACE` chuyển sang `READY`.
2. **Quy tắc rời tư thế 3 ngón (Pose Exit Requirement)**:
   - Sau khi phát lệnh chuyển tab, hệ thống yêu cầu người dùng phải **thực sự rời khỏi tư thế 3 ngón (`THREE_FINGERS`)** (hạ tay hoặc chuyển sang tư thế khác) và vượt qua thời gian hồi chiêu 400ms trước khi thực hiện lần gạt tiếp theo.
   - Hệ thống không tái vũ trang khi người dùng thu tay về vùng trung tính trong khi vẫn giữ 3 ngón (chống giật tab ngoài ý muốn).
   - Cấu hình Hybrid **không sử dụng lịch sử trình duyệt (`Alt + Mũi tên`)**; cử chỉ gạt 3 ngón chỉ dùng riêng cho chuyển tab.
3. **Cảnh báo phím bổ trợ vật lý dùng chung (Shared OS Modifiers)**:
   - **Tránh nhấn giữ các phím bổ trợ vật lý (`Ctrl`, `Shift`, `Alt`)** trên bàn phím trong lúc thực hiện cử chỉ cuộn hoặc chuyển tab. Trạng thái phím bổ trợ của hệ điều hành được chia sẻ toàn hệ thống; việc giữ phím vật lý kết hợp cử chỉ có thể vô tình kích hoạt các tổ hợp phím tắt ngoài dự kiến của hệ điều hành.
4. **Độc quyền chuột vật lý**:
   - Tuyệt đối không phát sinh sự kiện `MOVE`, `CLICK`, `mouseDown`, `mouseUp`, `zoom` hay lịch sử trình duyệt (`Alt+Mũi tên`). Không bao giờ tự ý can thiệp hay nhả các nút chuột vật lý mà người dùng đang thao tác trên bàn.
5. **Đánh giá tích lũy cuộn & Dập tắt rung giật (Synthetic Tests)**:
   - Trong kiểm thử tổng hợp (synthetic benchmark), bộ tích lũy sub-deadband ghi nhận đầy đủ chuyển động chậm $100 \times 0.002 = 0.200 \rightarrow 23$ bước cuộn (khắc phục lỗi nuốt bước vi cuộn của engine cũ).
   - Kiểm thử rung giật tĩnh giả lập khẳng định dập tắt dao động trong biên độ $\pm 0.0015$ (phát sinh 0 sự kiện cuộn).
   - Kiểm thử tần số giả lập đạt độ nhạy đồng nhất trên các mốc 15, 30 và 60 FPS. Các kết quả này được đo lường trên chuỗi vector toán học tổng hợp, chưa phản ánh đầy đủ mọi điều kiện camera và ánh sáng thực tế.
6. **Khả năng sử dụng vật lý chưa kiểm nghiệm (Physical Usability Pending)**:
   - Mức độ thoải mái công thái học và khả năng sử dụng vật lý thực tế với người dùng cuối trong môi trường sử dụng camera hàng ngày hiện **chưa được kiểm nghiệm thực tế** (pending physical user testing).
7. **Lưu ý dừng khẩn cấp**:
   - Do con trỏ chuột không bị ánh xạ bởi camera trong chế độ hybrid, góc dừng an toàn (`failsafe corner`) **không khả dụng**. Hãy luôn sử dụng tổ hợp phím **`Ctrl + Alt + Q`** để dừng khẩn cấp.
