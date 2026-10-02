# Báo Cáo Kỹ Thuật HYB-01 (Hybrid Assist Profile: Physical Mouse + Two-Finger Scroll + Three-Finger Tabs)

**Dự án**: AirPad Virtual Mouse Pad (Windows-first)  
**Nhiệm vụ**: HYB-01 — Hybrid Assist Profile (Đồng bộ Bằng chứng Thực nghiệm, Guide, Benchmark v3/v4 & Báo cáo Nghiệm thu Hoàn tất)  
**Thời gian cập nhật**: 2026-10-02  
**Môi trường thực thi**: Python 3.14.7 (.venv), Windows, Camera và OS control được cô lập an toàn bằng Mock (Zero Real OS / Camera Invocation in automated tests)  
**Trạng thái kiểm thử**: ĐÃ ĐƯỢC XÁC NHẬN CHÍNH THỨC — Bộ hồi quy toàn diện độc lập do Brain thực thi (Session 79772, chunk 248f35) đạt **408/408 PASSED in 183.57s** (toàn bộ 68 đường dẫn kiểm định không thay đổi trước/sau chạy); tự kiểm tra đạt **5/5 PASSED (chunk 87e298)**. Coordinator kiểm định độc lập cả hai benchmark (chunk af7e02, exit 0) đạt chuẩn 100% không can thiệp OS thật (`coordinator_shadow_gemini_final_20261002.json` và `coordinator_synthetic_gemini_final_20261002.json`). Bộ kiểm thử tập trung nội bộ của Worker đạt 149/149 PASSED (52 hybrid, 29 motion, 26 safety, 8 windows, 34 fsm/fsm2). Toàn bộ mã nguồn và dữ liệu đóng băng an toàn.

---

## 1. Thiết Kế Kỹ Thuật & Phân Định Quyền Kiểm Soát Hybrid Assist

Cấu hình **Hybrid Assist** (`--profile hybrid`) phân định trách nhiệm điều khiển trên hệ thống nhằm tối ưu hóa sự kết hợp giữa chuột vật lý và thị giác máy tính:
1. **Chuột vật lý giữ độc quyền quyền kiểm soát con trỏ**:
   - Di chuyển con trỏ chuột (`MOVE`).
   - Nhấp chuột trái (`LEFT_CLICK`), nhấp chuột phải (`RIGHT_CLICK`).
   - Kéo thả đối tượng (`DRAGGING`: `LEFT_DOWN` / `LEFT_UP`).
   - Webcam không phát sinh bất kỳ sự kiện di chuyển, nhấp, kéo hoặc thao tác lịch sử/thu phóng nào.
2. **Webcam cung cấp trợ năng thứ cấp qua cử chỉ whitelisted**:
   - **Cuộn trang dọc 2 ngón (`TWO_FINGERS_PARALLEL`)**: Cuộn trang dọc trực tiếp không cần vũ trang qua `POINT_ONE`. Đưa tay LÊN trong khung hình gương $\rightarrow$ cuộn LÊN (`wheel up`); đưa tay XUỐNG $\rightarrow$ cuộn XUỐNG (`wheel down`). Hỗ trợ tích lũy chuyển động chậm sub-deadband ($100 \times 0.002 = 0.200 \rightarrow 23$ bước cuộn). Bộ lọc dập tắt rung giật tĩnh ($\pm 0.0015 \rightarrow 0$) và chuyển động giật ngược subthreshold khi thả lỏng tay. Độ nhạy đồng nhất trên các tần số giả lập 15, 30 và 60 FPS.
   - **Chuyển tab 3 ngón (`THREE_FINGERS`)**: Gạt sang TRÁI trong khung hình gương $\rightarrow$ `Ctrl + Shift + Tab` (Tab trước); gạt sang PHẢI $\rightarrow$ `Ctrl + Tab` (Tab sau). Kích hoạt One-shot (1 lần duy nhất khi duy trì tư thế) và áp dụng thời gian hồi chiêu toàn cục 400ms (`cooldown_ms = 400.0`).
3. **Bảo toàn chế độ di sản (Legacy Backward Compatibility)**:
   - Tham số CLI `--profile {legacy,hybrid}` có giá trị mặc định là `legacy`.
   - Cấu hình `legacy` bảo lưu 100% máy trạng thái FSM, quy tắc vũ trang `POINT_ONE`, và kết quả replay đông kết ACC-07 (156 sự kiện controller: 151 MOVE, 5 SCROLL).

---

## 2. Hợp Đồng An Toàn, Kiểm Soát Vòng Đời & Đồng Bộ Tuyến Tính

### 2.1. Quản lý trạng thái Assist & Khóa dừng khẩn cấp
- Trong cấu hình `hybrid`, hệ thống luôn khởi động ở trạng thái **`OFF`** để ngăn ngừa can thiệp ngoài ý muốn.
- **Phím `SPACE`** trên bàn phím (khi cửa sổ xem trước camera có tiêu điểm) chuyển đổi trạng thái giữa `OFF` và `READY`.
- **Hành vi trạng thái `OFF`**: Khi ở trạng thái `OFF`, hệ thống cấm tuyệt đối mọi hành vi điều khiển tiến tới mới (`mouse_move`, `mouse_click`, `scroll`, `switch_tab`, `key_down`), nhưng các thao tác giải phóng phím do chính controller sở hữu và quy trình dọn dẹp an toàn (`key_up`, `cleanup`) vẫn được phép thực thi để đảm bảo an toàn phần cứng.
- **Dừng khẩn cấp (`Ctrl + Alt + Q`)**:
  - Đăng ký qua hook bàn phím mức thấp với cơ chế Fail-closed admission.
  - Khi được kích hoạt, trạng thái dừng khẩn cấp bị **chốt chặn vĩnh viễn (`latched`)**. Phím `SPACE` không thể giải phóng trạng thái này.
  - **Cảnh báo góc dừng (`Failsafe Corner`)**: Do cấu hình hybrid không điều khiển con trỏ chuột, tính năng dừng khẩn cấp bằng góc màn hình không khả dụng. CLI banner và status overlay trên HUD hiển thị cảnh báo rõ ràng: `"Failsafe Corner is UNAVAILABLE in hybrid profile (pointer is not mapped). Use Ctrl+Alt+Q or SPACE to toggle."`

### 2.2. Đồng bộ hóa tuyến tính vòng đời & Khắc phục lỗi đua luồng (P1 Repro 2b10b5)
- **Vấn đề đã khắc phục**: Trong điều kiện đa luồng, khi assist đang ở `READY`, một luồng khung hình nhận diện `nohand` (bàn tay rời khỏi khung hình) nếu bị hoãn ngay trước khi gán trạng thái, trong khi luồng chính gọi `toggle_assist()` về `OFF`, việc luồng `nohand` tiếp tục ghi đè trạng thái `READY` sẽ khiến `OFF` bị kích hoạt lại ngoài ý muốn (silent auto-reactivation).
- **Giải pháp xử lý**:
  1. Toàn bộ các thao tác ghi nhận vòng đời Hybrid (xử lý `nohand`, chuyển đổi `OFF`, reset mismatch, cập nhật `scroll_engine`/`swipe_engine`, chuyển trạng thái `SCROLL`/`TAB`/`READY`) đều được đồng bộ hóa tuyến tính bên trong khóa `SafetyGate._lock` (`RLock`).
  2. Bất biến trạng thái: Khi `assist_state` đang là **`OFF`**, **tuyệt đối không có bất kỳ khung hình nào** (kể cả `nohand`, mismatch hay cử chỉ khác) được phép tự động chuyển trạng thái về `READY`. Chỉ có lời gọi tường minh từ `toggle_assist()` mới có quyền kích hoạt lại `READY`.
  3. Khi đang thực thi giao dịch khung hình, trạng thái `assist_state` và cờ dừng `is_emergency_stopped` được kiểm tra lại ngay sau các callback trung gian (`rate_limiter.allow()`) trước khi dispatch tới `PlatformController`.
  4. Nếu `toggle_assist()` chuyển sang `OFF` trong lúc một khung hình đang chạy, khung hình đó lập tức bị hủy bỏ và không phát sự kiện.

### 2.3. Hợp đồng Modifier Tracking & Phục hồi lỗi bàn phím (Khắc phục lỗi kiểm thử 62acbb)
- **Vấn đề đã khắc phục**: Nếu lệnh `key_down` bị cổng an toàn từ chối (ví dụ do gate đã dừng khẩn cấp), việc gọi vô điều kiện `key_up` trong khối `finally:` sẽ gửi tín hiệu nhả phím cho một phím vật lý chưa từng được controller nhấn xuống (unowned physical modifier).
- **Giải pháp xử lý**:
  1. Danh sách modifier dự kiến (`['ctrl']` hoặc `['ctrl', 'shift']`) được đưa vào `attempted_mods` trước khi gọi `key_down`. Phím `tab` được ghi nhận vào `_held_keys` trước khi gọi `pyautogui.press('tab')`.
  2. Khối `finally:` kiểm tra điều kiện sở hữu: `if mod in self._held_keys:` trước khi gọi `self.key_up(mod)`. Nếu `key_down` bị cổng từ chối hoặc chưa được thực thi, controller sẽ không gọi `keyUp` cho modifier đó.
  3. Duyệt ngược danh sách `attempted_mods` (`reversed(attempted_mods)`) để lập tức gọi `key_up` cho từng modifier đã sở hữu (kể cả khi `keyDown` bị lỗi ở phím thứ hai như `shift`).
  4. Nếu thao tác `key_up` thất bại hoặc `press` bị lỗi, phím chưa được nhả được giữ lại trong `_held_keys` để phương thức `cleanup()` tiếp tục thử lại (idempotent retry).
  5. Fixture kiểm thử `reset_os_mocks` được chuẩn hóa để xóa toàn bộ `side_effect` trên các thuộc tính mock (`keyDown`, `keyUp`, `press`, `mouseUp`, `mouseDown`, `moveTo`, `scroll`, `hscroll`, `click`) trước và sau mỗi bài test, bảo lưu giá trị trả về của `size() = (1920, 1080)`.

### 2.4. Hợp đồng Tái vũ trang Chuyển Tab & Quỹ đạo Cửa sổ trượt (Swipe Rearm & Sliding Window Contract)
- **Cơ chế Cửa sổ trượt quỹ đạo gần nhất (Sliding Window Recent-Motion Trajectory)**:
  - **Khắc phục lỗi pha loãng vận tốc (Velocity Dilution Defect, chunk 201fad)**: Trong phiên bản trước, vận tốc được tính từ mẫu đầu tiên khi phát hiện tư thế `THREE_FINGERS` ($t_0$). Nếu người dùng giữ nguyên 3 ngón tay ổn định (ví dụ $x = 0.50$ trong $1.000\text{ms}$ đến $3.000\text{ms}$) rồi mới thực hiện gạt dứt khoát ($x = 0.62$ tại $3.190\text{ms}$, $\Delta x = 0.12$ trong $190\text{ms}$), vận tốc trung bình tính từ gốc ban đầu bị pha loãng ($\Delta x / \Delta t = 0.12 / 2190 \approx 0.000055 < 0.0003$), khiến cử chỉ bị nuốt hoàn toàn (`returns None`).
  - **Giải pháp**: Trong cấu hình Hybrid, `SwipeEngine` lưu trữ lịch sử chuyển động trong cửa sổ trượt $500\text{ms}$ (`window_ms = 500.0`). Khi cập nhật mỗi khung hình, engine quét qua các mẫu trong cửa sổ để tìm điểm gốc gần nhất thỏa mãn đồng thời:
    $$\Delta x \ge 0.08 \quad \text{và} \quad \frac{\Delta x}{\Delta t} \ge 0.0003 \text{ đơn vị/ms (tức } 0.3 \text{ đơn vị/s)}$$
  - Cơ chế này cho phép người dùng giữ tĩnh tư thế 3 ngón tùy ý (thử nghiệm thành công với $2\text{s}$ và $10\text{s}$) rồi gạt nhanh dứt khoát đều kích hoạt chính xác sự kiện (`TAB_NEXT` hoặc `TAB_PREV`).
  - Đồng thời, các chuyển động rung lắc tĩnh ($\pm 0.0015$) và trôi chậm (slow drift) trong suốt $10\text{s}$ đều được dập tắt hoàn toàn ($0$ sự kiện).
- **Hợp đồng Tái vũ trang (Rearm Contract)**:
  - **Cấu hình Hybrid**: Sau khi phát sự kiện chuyển tab (NEXT hoặc PREV), `SwipeEngine` chỉ được phép tái vũ trang sau khi:
    1. Người dùng thực sự **rời khỏi tư thế 3 ngón** (`is_three_fingers == False` hoặc mất dấu bàn tay, ghi nhận `_exit_observed_since_fire = True`).
    2. Thời gian trôi qua vượt quá thời gian hồi chiêu $400\text{ms}$ (`cooldown_ms`).
    - Hệ thống không tái vũ trang khi người dùng đưa tay trở lại vùng trung tính trong khi vẫn đang giữ nguyên 3 ngón tay (ngăn ngừa triệt để lỗi vô tình chuyển ngược tab khi hồi tay).
    - Khung hình đầu tiên sau khi tái vũ trang thành công sẽ tự động nạp mẫu hiện tại vào `_history` làm điểm mốc mới, không bị loại bỏ.
- **Cấu hình Legacy (Bảo toàn tuyệt đối)**:
  - Khắc phục lỗi hồi quy rò rỉ rơi xuống (fall-through regression): Trong khối kiểm tra `fired/rearm`, cấu hình Legacy bắt buộc trả về vô điều kiện `return None` ngay cả khi đã tái vũ trang thành công (khớp 100% mã nguồn gốc).
  - Chuỗi kiểm tra chuẩn mực: $x = 0.50@1000\text{ms}$, $0.61@1200\text{ms} \rightarrow \text{FORWARD}$, $0.50@1700\text{ms} \rightarrow \text{None}$ (tái vũ trang), $0.39@1730\text{ms} \rightarrow \text{None}$ (thiết lập gốc mới, không bị phát BACK giả lập). Khẳng định tính tương đương byte-for-byte và hành vi với engine gốc được nạp từ bản sao lưu `reports/hyb01/before/src/motion/swipe.py`.

### 2.5. Hợp đồng Không Bỏ Qua Bộ Phân Lớp (Fail-Closed Classifier Requirement)
- Loại bỏ hoàn toàn cơ chế bypass `self.classifier is None`.
- Cả cử chỉ cuộn trang (`is_scroll_eligible`) và chuyển tab (`is_tab_eligible`) đều bắt buộc:
  `self.classifier is not None and filtered_pose == target_pose and raw_pose == target_pose`
- Khi bộ phân lớp vắng mặt (`classifier is None`), hệ thống fail-closed và phát sinh 0 hành động.

---

## 3. Bằng Chứng Thực Nghiệm Benchmark Headless Shadow Replay (v3)

- **Tệp thực thi**: `reports/hyb01/run_hybrid_benchmark.py` (SHA-256: `24CF9FD9141C68AC81A6DB7A8F8B6C6706AB339CEE157AA8B501D5E854907973`)
- **Tệp kết quả xác thực v3**: `reports/hyb01/hybrid_shadow_benchmark_results_v3.json` (SHA-256: `A19E68691C0822CD7D43642BEB1577BE7A26CFEDC2B44DCD01D155F08539A555`, tạo độc quyền bằng cờ `open(path, 'x')`)
- **Minh bạch lịch sử tệp (Artifact Provenance Disclosure)**: Trong giai đoạn phát triển dự thảo ban đầu (draft development), các tệp ban đầu `hybrid_shadow_benchmark_results.json` và `hybrid_benchmark_results.json` đã bị ghi đè, sau đó được bảo lưu nguyên trạng; toàn bộ tập dữ liệu và mã băm đầu vào đóng băng (frozen inputs) hoàn toàn không bị ảnh hưởng. Các tệp v1 và v2 được lưu giữ nguyên trạng làm mốc lịch sử.
- **Kiểm định mã băm mật mã đầu vào (Input Cryptographic Pins)**:
  - Shadow dataset: `reports/acc06/trial_shadow_obs_20261001_162213_fb6d49.jsonl` (SHA-256: `2E2753DEF2D95BF9A33CC88C55EABA6F19560B7CD5DF2815CC037BAA3ABC8CD2`)
  - SVM Model bundle: `reports/acc05/candidate_personal_svm.json` (SHA-256: `66091D52D9DD6848FD8E35FC2641A31C4C2FFC9008329A4D2E9F85F8DD1246BE`)
  - ACC-07 Replay reference: `reports/acc07/dryrun_replay_events_v2.json` (SHA-256: `6BF9ABD3E8CA96B8BDCBD928E97A11A5B997E9C99072158A95A1AA97B03C3646`)
- **Cấu trúc tập dữ liệu Shadow**:
  - Tổng số dòng: **3.321** (gồm 1 bản ghi header và 3.320 bản ghi khung hình `record_type == "frame"`).
  - Khung hình phát hiện bàn tay (`hand_detected == True`): **1.875**
  - Khung hình vắng bàn tay (`hand_detected == False`): **1.445**
- **Minh bạch về thông lượng tính toán (Offline Headless FPS)**:
  > [!NOTE]
  > Tốc độ khung hình (headless FPS) đo lường thông lượng tính toán của CPU trong môi trường xử lý ngoại tuyến không đồng bộ (headless throughput). Thông số này **không đại diện cho độ trễ camera trực tiếp** (live camera capture latency) hay độ trễ phát sự kiện OS thời gian thực.
- **Kiểm định đối chứng phủ định (Negative Controls Verification)**:
  - Injected Zoom (Ctrl+Wheel): Bị phát hiện chính xác là `forbidden_zooms` (`injected_zoom_detected: true`).
  - Injected History (Alt+Left): Bị phát hiện chính xác là `forbidden_history` (`injected_history_detected: true`).
  - Injected MOVE: Bị phát hiện chính xác là `forbidden_moves` (`injected_move_detected: true`).
  - Injected Alt+Tab (window switch): Bị phát hiện chính xác là `forbidden_other = 3` (`injected_alt_tab_detected: true`).
  - Injected Horizontal SCROLL (dx!=0): Bị phát hiện chính xác là `forbidden_other = 1` (`injected_horizontal_scroll_detected: true`).
  - Injected Alt-alone (menu focus): Bị phát hiện chính xác là `forbidden_other = 2` (`injected_alt_alone_detected: true`).

| Chỉ số / Đặc tính | Cấu hình Legacy (Air-Mouse) | Cấu hình Hybrid (Assist OFF) | Cấu hình Hybrid (Assist ACTIVE) | Hợp đồng nghiệm thu |
| :--- | :--- | :--- | :--- | :--- |
| **Tổng số khung hình replayed** | 3.320 (lọc bỏ header) | 3.320 (lọc bỏ header) | 3.320 (lọc bỏ header) | Khớp chính xác 3.320 frame records |
| **Thông lượng xử lý ngoại tuyến** | 2.957s (1.122,9 headless fps) | 2.714s (1.223,1 headless fps) | 2.986s (1.111,8 headless fps) | Thông lượng CPU thuần túy |
| **Sự kiện MOVE** | 151 | **0** (tính động từ trace) | **0** (tính động từ trace) | Chuột vật lý giữ độc quyền con trỏ |
| **Sự kiện CLICK (Trái/Phải)** | 0 | **0** (tính động từ trace) | **0** (tính động từ trace) | Chuột vật lý giữ độc quyền click |
| **Sự kiện DOWN / UP (Kéo)** | 0 | **0** (tính động từ trace) | **0** (tính động từ trace) | Chuột vật lý giữ độc quyền kéo thả |
| **Sự kiện ZOOM (kể cả Ctrl+wheel)**| 0 | **0** (tính động từ trace) | **0** (tính động từ trace) | Không phát sinh thao tác thu phóng |
| **Sự kiện HISTORY (Alt+Arrows/Swipe)**| 0 | **0** (tính động từ trace) | **0** (tính động từ trace) | Không phát sinh thao tác lịch sử |
| **Sự kiện OTHER (Alt-alone, Alt+Tab, hscroll, unknown)**| 0 | **0** (tính động từ trace) | **0** (tính động từ trace) | Toàn bộ kênh không hợp lệ bị chặn |
| **Sự kiện SCROLL (Cuộn)** | 5 | 0 | 14 | Cuộn dọc 2 ngón whitelisted |
| **Sự kiện TAB (Chuyển tab)** | 0 | 0 | 0 (tập shadow không có gạt 3 ngón) | Sẵn sàng cho 3 ngón |
| **Tổng sự kiện Controller** | **156** | **0** | **14** | Khớp 100% baseline ACC-07 ở Legacy |
| **Khớp số lượng sự kiện (`counts_match`)** | **ĐẠT (156 / 156)** | N/A (Chế độ mới) | N/A (Chế độ mới) | Khớp 151 MOVE, 5 SCROLL |
| **Khớp từng sự kiện (`per_event_match`)** | **ĐẠT (156 / 156)** | N/A (Chế độ mới) | N/A (Chế độ mới) | Khớp 100% payload từng sự kiện |
| **Nút chuột giữ sau Cleanup** | `[]` (rỗng) | `[]` (rỗng) | `[]` (rỗng) | An toàn phần cứng tuyệt đối |
| **Phím bấm giữ sau Cleanup** | `[]` (rỗng) | `[]` (rỗng) | `[]` (rỗng) | An toàn phần cứng tuyệt đối |

---

## 4. Bằng Chứng Thực Nghiệm Dynamic Synthetic Intent Benchmark (v4)

- **Tệp thực thi**: `reports/hyb01/run_synthetic_intent_benchmark.py` (SHA-256: `4D6657AF2AF54FC4CFD19B0CCE0EE204DA3B03E04751BB5016054D87AC16E91D`)
- **Tệp kết quả xác thực v4**: `reports/hyb01/hybrid_synthetic_benchmark_results_v4.json` (SHA-256: `2969FE3B09FC1409874C7FA84E02DDB3FD045A4964D5C4821784FD1D7ED7A9CC`, tạo độc quyền bằng cờ `open(path, 'x')`)
- **Bảo lưu artifact v1, v2 và v3**: Các tệp `hybrid_synthetic_benchmark_results.json`, `hybrid_synthetic_benchmark_results_v2.json` và `hybrid_synthetic_benchmark_results_v3.json` được bảo lưu nguyên trạng.
- **Cải tiến và kiểm định trong v4**:
  1. **Theo dõi trạng thái phím bổ trợ theo trình tự thời gian (Ordered-State Auditor)**:
     - Duyệt chuỗi sự kiện theo thứ tự xuất hiện, lưu vết tập hợp phím đang giữ (`tracked_held_keys`).
     - Phát hiện chính xác Ctrl + Scroll $\rightarrow$ `forbidden_zooms`.
     - Phát hiện Alt + Left/Right $\rightarrow$ `forbidden_history`.
     - Phát hiện Alt + Tab $\rightarrow$ `forbidden_other` (window switch).
     - Phát hiện cuộn ngang ($dx \ne 0$) $\rightarrow$ `forbidden_other`.
     - Phát hiện nhấn/nhả phím Alt đơn lẻ (Alt-alone menu focus) $\rightarrow$ `forbidden_other` (vừa gắn cờ vi phạm, vừa tiếp tục theo dõi để nhận diện tổ hợp).
     - Nghiêm cấm mọi sự kiện `LEFT_UP` (mouseUp) kể cả khi đánh dấu cleanup (vì chế độ Hybrid không bao giờ tự nhấn nút chuột).
  2. **Kiểm định đối chứng phủ định toàn diện (6/6 Negative Controls Passed)**:
     - `injected_zoom_detected`: **True** (Ctrl+Wheel)
     - `injected_history_detected`: **True** (Alt+Left)
     - `injected_move_detected`: **True** (MOVE)
     - `injected_alt_tab_detected`: **True** (Alt+Tab, `forbidden_other = 3`)
     - `injected_horizontal_scroll_detected`: **True** (Horizontal scroll dx!=0, `forbidden_other = 1`)
     - `injected_alt_alone_detected`: **True** (Alt-alone menu focus, `forbidden_other = 2`)
  3. **Kịch bản 1: Cuộn chậm liên tục (100 frames, $dy = 0.002$, tổng 0.200)**:
     - Engine gốc (`reports/hyb01/before/src/motion/scroll.py`, mã băm xác thực `EEE878AD...`): **0 bước cuộn** (xác nhận lỗi nuốt bước vi cuộn của thuật toán di sản do xóa tích lũy sub-deadband).
     - Hybrid Standalone Engine & Pipeline: **23 bước cuộn** (tích lũy thành công, hướng cuộn đúng, 0 move).
     - Công bố minh bạch: Thử nghiệm với cấu hình bộ lọc đơn vị `Majority(1)`/`Stability(1)` để kiểm chứng toán học thuần túy của bộ tích lũy; không đại diện cho độ rung của tay người thật dưới bộ lọc sản xuất.
  4. **Kịch bản 2: Dập tắt rung giật tĩnh (100 frames, $\pm 0.0015$)**:
     - Số sự kiện phát sinh: **0**, Số bước cuộn tuyệt đối (`absolute_scroll_units`): **0**.
     - Khẳng định dập tắt dao động tại nguồn, không dựa vào bù trừ đại số.
  5. **Kịch bản 3: Chuyển tab 3 ngón, hồi vị trung tính vượt mốc 400ms & kiểm soát văng quá đà**:
     - $t = 150\text{ms}$: Gạt trái $\rightarrow$ phát `TAB_PREV` (Ctrl+Shift+Tab).
     - $t = 600\text{ms}$ (vượt mốc 400ms cooldown: $600 - 150 = 450\text{ms}$) và $t = 630\text{ms}$: Hồi tay về vùng trung tính khi **vẫn đang giữ 3 ngón** $\rightarrow 0$ hành động (`pre_exit_actions == []`).
     - $t = 750\text{ms}$: Văng quá đà (overshoot) sang phải khi **vẫn đang giữ 3 ngón** $\rightarrow 0$ hành động (`cooldown_crossed_while_still_three_rearm_suppressed == True`).
     - $t = 800\text{ms}$: Thực sự rời tư thế sang `OTHER` $\rightarrow 0$ hành động.
     - $t = 850\text{ms}$: Tái nhập lại tư thế `THREE_FINGERS` $\rightarrow 0$ hành động.
     - $t = 1000\text{ms}$: Gạt phải $\rightarrow$ phát `TAB_NEXT` (Ctrl+Tab) thành công một lần duy nhất.
  6. **Kịch bản 4: Chuyển đổi Assist ON $\rightarrow$ OFF dưới bộ lọc mặc định**:
     - Bắt đầu ở `READY`, chuyển sang `OFF` dưới bộ lọc thời gian mặc định (`Majority 5`, `Stability 2`), xử lý chuỗi chuyển đổi raw/filtered giữa `TWO_FINGERS_PARALLEL`, `OTHER` và `THREE_FINGERS`.
     - Tổng sự kiện phát sinh: **0**.
  7. **Kịch bản 5: Giữ tĩnh 3 ngón ấm (2s) rồi gạt dứt khoát**:
     - 50 khung hình giữ tĩnh tại $x = 0.50$ trong $2.000\text{ms}$, sau đó gạt sang $x = 0.62$ trong $160\text{ms}$.
     - Kết quả: `actions_fired: {'TAB_NEXT': 1}`, `warm_hold_swipe_success: true`.
     - Khắc phục triệt để khiếm khuyết pha loãng vận tốc của engine cũ.
  8. **Kiểm toán an toàn (Safety Audit)**:
     - `all_held_buttons_clean`: **True** (tất cả các kịch bản)
     - `all_held_keys_clean`: **True** (tất cả các kịch bản)
     - `all_forbidden_counts_zero`: **True** (0 MOVE, 0 CLICK, 0 DRAG, 0 ZOOM, 0 HISTORY, 0 OTHER trên toàn bộ 5 kịch bản)

| Kịch bản kiểm định ý định (Dynamic Intent) | Engine Gốc (Saved-Original Legacy) | Cấu hình Hybrid (Mới) | Kết quả & Bất biến nghiệm thu v4 |
| :--- | :--- | :--- | :--- | :--- |
| **1. Cuộn chậm liên tục (100 frames, $dy = 0.002$, tổng 0.200)** | **0 bước cuộn** (LỖI NUỐT BƯỚC ĐƯỢC XÁC NHẬN: $0.002 < 0.005$ deadzone bị xóa từng frame) | **23 bước cuộn** (tích lũy mượt mà thành công, 0 move) | Khắc phục triệt để lỗi nuốt bước vi cuộn |
| **2. Chống rung giật tĩnh (100 frames, rung $\pm 0.0015$)** | 0 bước cuộn, 0 move | **0 sự kiện controller**, **0 bước cuộn tuyệt đối** (`abs_units == 0`) | Triệt tiêu rung tĩnh hoàn toàn (không phụ thuộc bù trừ đại số) |
| **3. Gạt 3 ngón & Kiểm soát Cooldown / Hồi vị (Neutral Return)** | Cần vũ trang FSM MOVING | `TAB_PREV`: 1, Hồi vị trung tính vượt mốc 400ms (600ms, 630ms) và overshoot (750ms) khi vẫn giữ 3 ngón: **0**, Sau khi rời tư thế sang `OTHER` (800ms) và gạt phải lại (1000ms): `TAB_NEXT`: 1 | `pre_exit_actions == []`, `cooldown_crossed_while_still_three_rearm_suppressed == True`; One-shot chuẩn mực |
| **4. Chuyển đổi Assist ON $\rightarrow$ OFF với bộ lọc mặc định** | N/A | **0 sự kiện** (45 khung hình gồm raw TWO chuyển raw THREE qua OTHER đều bị dập tắt) | Trạng thái OFF dập tắt 100% can thiệp dưới bộ lọc mặc định |
| **5. Giữ tĩnh 3 ngón ấm (2s) rồi gạt dứt khoát** | Nuốt cử chỉ do velocity bị pha loãng tính từ t=0 ban đầu (`returns None`) | **`TAB_NEXT`: 1** (phát hiện chính xác qua sliding window 500ms, 0 sự kiện thừa) | Khắc phục triệt để lỗi nuốt cử chỉ khi giữ tĩnh trước khi gạt |
| **6. Đếm hành động cấm động từ Trace (Dynamic Forbidden Counts)** | N/A | **0 MOVE, 0 CLICK, 0 DRAG, 0 ZOOM, 0 HISTORY, 0 OTHER, 0 phím/nút giữ sau cleanup** | Toàn bộ 5 kịch bản âm tính hoàn toàn (`all_forbidden_counts_zero: true`) |

---

## 5. Giới Hạn Phạm Vi, Hướng Dẫn Vận Hành & Minh Bạch Nghiệm Thu

1. **Cập nhật tài liệu hướng dẫn ([`docs/AIRPAD_FUNCTION_GUIDE_VI.md`](file:///C:/Users/tuana/Documents/Codex/2026-09-28/a/outputs/AirPad_KNN_Project/docs/AIRPAD_FUNCTION_GUIDE_VI.md))**:
   - Gắn nhãn rõ ràng: **Mục 1 đến Mục 8 là cấu hình Legacy**, điều hướng người dùng cấu hình Hybrid trực tiếp đến **Mục 9**.
   - Loại bỏ các từ ngữ cam kết quá đà ("giải quyết triệt để mỏi tay", "triệt tiêu hoàn toàn", "đảm bảo"); gắn nhãn điều kiện thử nghiệm trên dữ liệu tổng hợp toán học.
   - Bổ sung hướng dẫn tiêu điểm tiền cảnh: Phím tắt chuyển tab (`Ctrl + Tab` / `Ctrl + Shift + Tab`) nhắm vào ứng dụng FOREGROUND tương thích; người dùng cần dùng chuột vật lý để nhấp chọn ứng dụng mục tiêu sau khi bật `SPACE` trong cửa sổ OpenCV preview.
   - Yêu cầu người dùng rời tư thế 3 ngón trước khi thực hiện lần gạt tiếp theo; cấu hình Hybrid không sử dụng lịch sử trình duyệt (`Alt + Mũi tên`).
   - Cảnh báo tránh nhấn giữ các phím bổ trợ vật lý (`Ctrl`, `Shift`, `Alt`) trên bàn phím trong lúc thực hiện cử chỉ do trạng thái phím bổ trợ được chia sẻ toàn hệ điều hành.
   - Thừa nhận công khai: Trải nghiệm công thái học và khả năng sử dụng vật lý thực tế trên người dùng hiện chưa được kiểm nghiệm lâm sàng.
   - Nêu rõ hành vi trạng thái `OFF`: Nghiêm cấm hành vi điều khiển tiến tới mới nhưng cho phép dọn dẹp và giải phóng phím sở hữu.
2. **Thu hẹp phạm vi tuyên bố vào bằng chứng thực nghiệm**:
   - Tất cả các kết luận về hiệu năng và hành vi được giới hạn chặt chẽ trong phạm vi 3.320 khung hình shadow replay ngoại tuyến và các kịch bản synthetic benchmark.
   - Thử nghiệm tương tác vật lý trực tiếp với webcam và người dùng cuối vẫn đang ở trạng thái **chờ thực hiện (pending physical testing)**.

---

## 6. Phân Định Quyền Sở Hữu Kiểm Thử & Trạng Thái Hồi Quy Toàn Diện

1. **Bộ hồi quy toàn diện độc lập do Brain / Coordinator thực thi chính thức**:
   - Lệnh kiểm thử đầy đủ: `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider tests reports/acc04/test_repair_validation.py reports/acc05/test_acc05_benchmark.py reports/acc07/test_acc07_dryrun.py`
   - **Phiên kiểm thử toàn diện 408 tests hoàn tất (Session 79772, chunk 248f35)**: Đạt **408/408 PASSED in 183.57s (Exit code 0)**; toàn bộ 68 đường dẫn kiểm định mã nguồn và dữ liệu không đổi trước/sau khi chạy (`68 paths unchanged before/after full`).
   - **Tự kiểm tra độc lập của Brain (chunk 87e298)**: **5/5 PASSED**.
   - **Kiểm định độc lập cả hai Benchmark có Guard từ Coordinator (chunk af7e02, exit 0)**:
     - Tạo độc quyền 2 artifact kiểm định: `coordinator_shadow_gemini_final_20261002.json` và `coordinator_synthetic_gemini_final_20261002.json`.
     - Chuỗi sự kiện có thứ tự (ordered traces), các kênh không hợp lệ được bơm (injected unknown channels, Alt-alone menu focus, cleanupUp) đều bị phát hiện chính xác, 0 cuộc gọi OS thật (`0 real OS`).
   - **Bảo lưu mốc lịch sử các phiên trước**:
     - Phiên 401 tests (Session 96323, chunk c791f3): Đạt 401/401 PASSED trong 395.87s trên phiên bản trước sửa đổi trượt.
     - Phiên 408 tests dở dang (Session 49845): Khởi chạy trên mã nguồn hoàn tất nhưng bị mất kết quả tiến trình sau khi môi trường runtime khởi động lại (được bảo lưu dưới dạng mốc lịch sử).
     - Replay độc lập của Coordinator (`coordinator_legacy_replay_20261002_final.json`, chunk 8212f0): 3.320 frames, 156 events = 151 MOVE + 5 SCROLL. Khớp chính xác 100% header, stages, frames và action_events với ACC-07; Positive OS guards bảo vệ hệ thống, outer callcount = 0, camera trực tiếp được giữ tạm dừng an toàn.
2. **Các bộ kiểm thử cục bộ do Worker thực thi và xác thực trên mã nguồn cuối cùng**:
   - `tests/test_hybrid.py`: **52/52 passed** (bao gồm 45 bài test ban đầu + 5 bài test trượt/hold + 2 bài test hồi quy bảo toàn Legacy SwipeEngine so sánh trực tiếp với engine gốc).
   - Bộ kiểm thử tập trung (`test_hybrid.py`, `test_motion.py`, `test_safety_contract.py`, `test_windows.py`, `test_fsm.py`, `test_fsm_phase2.py`): **149/149 passed in 1.81s**.
   - Bộ kiểm thử tập trung độc lập của Brain trước đó (chunk 1d7c78): **107 passed in 2.03s** (52 Hybrid + 29 motion + 26 safety).
   - Tự kiểm tra ứng dụng (`app.py --self-check`): **5/5 PASSED (chunk 0714ac)**.

---

## 7. Bảng Đối Chiếu Tệp Tin, Mã Băm SHA-256 (Tạo Từ Get-FileHash) & Diffs

Toàn bộ các mã băm SHA-256 dưới đây được kết xuất trực tiếp bằng lệnh PowerShell `Get-FileHash -Algorithm SHA256`. Mã băm trước nhiệm vụ đối chiếu chính xác với `reports/hyb01/before_manifest.txt` và các bản sao lưu trong `reports/hyb01/before/`. Toàn bộ 9 tệp diff đã được đối chiếu và cập nhật chuẩn xác theo lệnh `git diff --no-index`.

| Tệp tin | Trạng thái | SHA-256 Trước HYB-01 (`reports/hyb01/before_manifest.txt`) | SHA-256 Hiện tại (Get-FileHash) | Tệp Diff |
| :--- | :--- | :--- | :--- | :--- |
| `app.py` | Sửa đổi (Đông kết) | `B5C65CB0D8B631B356DAB9F9BC635AE86CACD9A4F8E6FC168B03F69E4083F6EB` | `5CD7DF319236C4AA83DEC7CAB810A89D3B6E163BC8DF4B28625EFE7D3398446E` | `reports/hyb01/diff_app.diff` |
| `src/platform/interface.py` | Sửa đổi (Đông kết) | `3DA1D921EED97412D7106669EB3CC725F860F68ECE3D12554C70182D5BB120FD` | `DEF10B11450A4635AEAAEE3DD02177909A70E494EFF012D85562703464EE6A03` | `reports/hyb01/diff_interface.diff` |
| `src/platform/windows.py` | Sửa đổi (Đông kết) | `107A418BC224639CFCA836EE711DB198522922981FA8FD82AA3109E43438C3E4` | `0BFBCECF28984092546F721A282C374C91B8C7D7D405F66839C1CB74FE8F5696` | `reports/hyb01/diff_windows.diff` |
| `src/platform/dry_run.py` | Sửa đổi (Đông kết) | `365F970CA60266738A08EDC086E1188231992359FA007258BD7485170746BA01` | `61D0113C0457AA005DDEBB408F2796F582297FF71612A6C58C3D04560E0D9930` | `reports/hyb01/diff_dry_run.diff` |
| `src/motion/scroll.py` | Sửa đổi (Đông kết) | `EEE878AD3DD049E975EC2C1475854C1BFDE31873E14F7C175A22B6351DD9C5EC` | `B4EEB90661CE3FA9BF074283916E66643B0D5FDEFCD5CAC1BAC693565CC81D70` | `reports/hyb01/diff_scroll.diff` |
| `src/motion/swipe.py` | Sửa đổi (Đông kết) | `D9E06C9EAF6E2B7459D9808B583A6610B92DA82C11A6E2C2A4AADB3A5617BF2E` | `1B6457B7470E7241229860F4F11E12AEBD985723D198F75626923F7392EA9A87` | `reports/hyb01/diff_swipe.diff` |
| `src/runtime/pipeline.py` | Sửa đổi (Đông kết) | `6A3A1F9D2BD33D36B46D289650A599BCEB356626DA35525B17C14B521811402C` | `267C0869685E405E93D83B8F4C799D605B71FF6E37956A5E9C8BD6129769F7B2` | `reports/hyb01/diff_pipeline.diff` |
| `docs/AIRPAD_FUNCTION_GUIDE_VI.md`| Sửa đổi (Đông kết) | `2FCD4BA792B6BB1742988F78D1AF91D0C0931C703ABAE0ACE8B27DC791A35AB6` | `2D894ADA30C5E89F1D6DE3F3565D327205638288DFFC9F2763B1AC1A445DB15B` | `reports/hyb01/diff_guide.diff` |
| `tests/test_hybrid.py` | Mới (Đông kết) | (Mới) | `53A242B208B2C61E5A76640FEA281F4FE6B90AEE54D0B08775988BA0CC7185E4` | (Tệp kiểm thử mới, 52 tests) |
| `tests/test_app.py` | **Nguyên vẹn** | `659A9AF1E3B075982CFC4D9BFDD4178C5DC1C019BD4834E4C673A3C9AF1339EA` | `659A9AF1E3B075982CFC4D9BFDD4178C5DC1C019BD4834E4C673A3C9AF1339EA` | `reports/hyb01/diff_test_app.diff` (2 bytes, không có khác biệt văn bản / no textual diff) |
| `src/safety/failsafe.py` | **Đông kết** | `B5C4259A97E945BE03CC14647AC6FDDDFBEF27243BBB5C8F5256BF6EB68DE312` | `B5C4259A97E945BE03CC14647AC6FDDDFBEF27243BBB5C8F5256BF6EB68DE312` | (Không chỉnh sửa) |
| `config/default.yaml` | **Đông kết** | `9E5FBCCD5BEAFFBEFE00602E6EFA94F4889D4C8873CCBD25306D2159130D2E31` | `9E5FBCCD5BEAFFBEFE00602E6EFA94F4889D4C8873CCBD25306D2159130D2E31` | (Không chỉnh sửa) |
| `models/airpad_svm.json` | **Đông kết** | `4CC3C55F5E4B571C5462DE31AFD6D5C4DB64C803466D306CDEFAB34B31D7B977` | `4CC3C55F5E4B571C5462DE31AFD6D5C4DB64C803466D306CDEFAB34B31D7B977` | (Không chỉnh sửa) |
| `reports/acc05/candidate_personal_svm.json` | **Đông kết** | `66091D52D9DD6848FD8E35FC2641A31C4C2FFC9008329A4D2E9F85F8DD1246BE` | `66091D52D9DD6848FD8E35FC2641A31C4C2FFC9008329A4D2E9F85F8DD1246BE` | (Không chỉnh sửa) |

### 7.1. Bảng mã băm các tệp kịch bản và artifact Benchmark nghiệm thu mới
| Tệp kịch bản / Artifact | SHA-256 (Get-FileHash) | Mục đích / Trạng thái |
| :--- | :--- | :--- |
| `reports/hyb01/run_hybrid_benchmark.py` | `24CF9FD9141C68AC81A6DB7A8F8B6C6706AB339CEE157AA8B501D5E854907973` | Kịch bản Shadow Replay Benchmark v3 (ordered auditor) |
| `reports/hyb01/hybrid_shadow_benchmark_results_v3.json` | `A19E68691C0822CD7D43642BEB1577BE7A26CFEDC2B44DCD01D155F08539A555` | Kết quả Shadow Replay Benchmark v3 (độc quyền) |
| `reports/hyb01/run_synthetic_intent_benchmark.py` | `4D6657AF2AF54FC4CFD19B0CCE0EE204DA3B03E04751BB5016054D87AC16E91D` | Kịch bản Dynamic Synthetic Intent Benchmark v4 |
| `reports/hyb01/hybrid_synthetic_benchmark_results_v4.json` | `2969FE3B09FC1409874C7FA84E02DDB3FD045A4964D5C4821784FD1D7ED7A9CC` | Kết quả Synthetic Intent Benchmark v4 (độc quyền) |
