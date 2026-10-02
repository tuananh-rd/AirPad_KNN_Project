# Báo Cáo Kỹ Thuật ACC-08 (Worker Report)

**Mã công việc**: ACC-08  
**Trạng thái**: Hoàn tất (`ACC08_DIRECT_DONE`)  
**Người thực hiện**: Antigravity Code Worker  
**Ngày thực hiện**: 2026-10-01  
**Môi trường**: Windows, Python venv (`.venv`)  

---

## 1. Tóm Tắt Nhiệm Vụ & Phân Tích Nguyên Nhân Gốc (Root Cause Analysis)

### 1.1. Vấn đề người dùng phản hồi
Người dùng ghi nhận hiện tượng con trỏ chuột bị đảo ngược phương ngang khi sử dụng camera thực tế (di chuyển tay sang phải nhưng con trỏ chuột chạy sang trái). Đồng thời, người dùng yêu cầu cung cấp tài liệu hướng dẫn cử chỉ chi tiết bằng tiếng Việt (`docs/AIRPAD_FUNCTION_GUIDE_VI.md`).

### 1.2. Phân tích nguyên nhân kỹ thuật (Double-Mirroring Bug)
1. Trong `app.run_camera()`:
   - Khung hình webcam được lật ngang qua `flipped_bgr = cv2.flip(frame, 1)` để hiển thị cửa sổ xem trước tự nhiên như nhìn vào gương (mirrored preview).
   - MediaPipe Hand Landmark detector phát hiện các điểm mốc trực tiếp trên khung hình đã lật này.
   - Khi người dùng đưa tay sang bên phải của mình (hướng nhìn đối diện webcam), hình ảnh bàn tay xuất hiện ở phía bên phải của khung hình xem trước, khiến tọa độ chuẩn hóa $x$ của ngón trỏ (Landmark 8) **tăng** (ví dụ từ $0.35$ lên $0.65$).
2. Tuy nhiên, `CursorMapper` theo cấu hình mặc định trong `config/default.yaml` (`motion.mirror_x: true`) lại thực hiện phép lật ngang thêm một lần nữa:
   $$\text{cur\_x} = 1.0 - x$$
3. Do phép lật bị áp dụng **hai lần** (OpenCV lật khung hình + CursorMapper đảo ngược tọa độ $x$), khi tay di chuyển sang phải ($x$ tăng), $\text{cur\_x}$ lại giảm, khiến con trỏ màn hình dịch chuyển sang trái (ngược hướng).

### 1.3. Phương án xử lý chuẩn tắc
- Trong hàm `app.run_camera()`: Tạo bản sao sâu `cam_config = copy.deepcopy(config)` và ghi đè `cam_config.setdefault("motion", {})["mirror_x"] = False`.
- Truyền `cam_config` vào `create_pipeline(cam_config, ...)` bên trong phiên camera.
- Giữ nguyên cấu hình mặc định toàn cục (`config/default.yaml`), logic của `CursorMapper`, cơ chế lật hình xem trước `cv2.flip(frame, 1)`, việc trích xuất đặc trưng chuẩn hóa và hướng di chuyển trục $y$.
- Cấu hình chung cho replay (`run_replay`) và các luồng gọi `create_pipeline` trực tiếp không bị biến đổi.

---

## 2. Danh Mục Tệp & Kiểm Định Bất Biến (File Inventory & Hashes)

### 2.1. Tệp sửa đổi và bổ sung trong phạm vi ACC-08
| Tệp | Mô tả | Mã băm SHA-256 |
| :--- | :--- | :--- |
| `app.py` | Sửa `run_camera` dùng `cam_config` với `mirror_x=False` | `B4852C0483A0D89AABBE973ECDDFE3AC8AD4295BEE222DDDF3D2A6880CE55DC6` |
| `tests/test_app.py` | Thêm kiểm thử tái hiện lỗi, kiểm thử 4 pha delta, và chống đột biến config | `BBF787F46E28900B535BAD1F772BDCC7E3D61C3110BD82BA9CBFFC7105FD3A14` |
| `docs/AIRPAD_FUNCTION_GUIDE_VI.md` | Tài liệu hướng dẫn sử dụng cử chỉ tiếng Việt chuẩn FSM | `A61E0AF0A40A096F77668D21FF3E3CDAF4D784C15018522979508746925E2BE1` |
| `reports/acc08/app.py.initial` | Bản sao snapshot `app.py` trước khi sửa | `687AD66A16907CC580709A53608BB292AC522D2A603F4CB56A4860EA3FD24B34` |
| `reports/acc08/test_app.py.initial` | Bản sao snapshot `tests/test_app.py` trước khi sửa | `12C56EBB5D7A5A7C028CFC4A51E0F348DF761E1A3377D08CD8B34ABF593E91FE` |
| `reports/acc08/app.py.diff` | Unified diff chuẩn không index cho `app.py` | `4EA22480E6B6B1420AC470C96D9852CFEA2B9B3470BFFAAFBAC69E7DFFF1C583` |
| `reports/acc08/test_app.py.diff` | Unified diff chuẩn không index cho `tests/test_app.py` | `4C56DBBEF44877FFAFEE325BE17107692EDD250250711FC38758EC3A767B8F7B` |

### 2.2. Kiểm định tính bất biến của các thành phần cốt lõi
| Thành phần bất biến | Trạng thái | Mã băm SHA-256 |
| :--- | :--- | :--- |
| `config/default.yaml` | Giữ nguyên tuyệt đối | `9E5FBCCD5BEAFFBEFE00602E6EFA94F4889D4C8873CCBD25306D2159130D2E31` |
| `src/motion/cursor.py` | Giữ nguyên tuyệt đối | `650EFE8B7028262126009E5E57ABBE410B87A7564ACB38D05E5F1F517B75F981` |
| `src/gesture/fsm.py` | Giữ nguyên tuyệt đối | `E525FCD126F10BFD5F2C9AF676B0F7DEEF1AA4FCED26BEDEADAEEF3223B2D3B8` |
| `src/runtime/pipeline.py` | Giữ nguyên tuyệt đối | `D1B5E858AE2BDE2F0DD08EBFBE3738B7DD6C80E4E8A2179DD1081ED6503209A4` |
| `reports/acc05/candidate_personal_svm.json` | Đóng băng, không thăng cấp | `66091D52D9DD6848FD8E35FC2641A31C4C2FFC9008329A4D2E9F85F8DD1246BE` |
| `models/airpad_svm.json` | Giữ nguyên mô hình gốc | `4CC3C55F5E4B571C5462DE31AFD6D5C4DB64C803466D306CDEFAB34B31D7B977` |

---

## 3. Bằng Chứng Thực Nghiệm & Kiểm Thử Hướng Di Chuyển (Directional Verification)

### 3.1. Tái hiện lỗi trên bản sao gốc (`app.py.initial`) và kiểm chứng bản sửa đổi
Kiểm thử `test_camera_cursor_direction_regression_reproduced_on_saved_initial_app` sử dụng `SourceFileLoader` nạp độc lập `reports/acc08/app.py.initial` và đối chiếu song song với `app.py`:
- **Trên mã gốc (`app.py.initial`)**:
  - `pipeline.cursor_mapper.mirror_x` có giá trị `True`.
  - Khi tay di chuyển sang phải trên preview ($x: 0.35 \rightarrow 0.65$), tọa độ con trỏ màn hình giảm từ $1319$ xuống $698$ ($\Delta x < 0$ $\rightarrow$ con trỏ chạy sang trái, tái hiện chính xác lỗi ngược hướng).
- **Trên mã đã sửa (`app.py`)**:
  - `pipeline.cursor_mapper.mirror_x` có giá trị `False`.
  - Khi tay di chuyển sang phải trên preview ($x: 0.35 \rightarrow 0.65$), tọa độ con trỏ màn hình tăng từ $600$ lên $1319$ ($\Delta x > 0$ $\rightarrow$ con trỏ chạy sang phải, hướng di chuyển chính xác).

### 3.2. Kiểm thử Live Camera End-to-End với 4 pha delta riêng biệt
Kiểm thử `test_run_camera_end_to_end_cursor_direction_x_and_y` thực thi hàm `run_camera(allow_real_control=False)` thông qua `FakeVideoCapture`, `FakeDetector`, `DryRunController`, và các bộ lọc thực của hệ thống (`MajorityFilter`, `StabilityGate`, `GestureFSM`, `CursorMapper`):
- **Pha ổn định (Settling Phase)**: 8 khung hình giữ tại $(0.50, 0.50)$ giúp FSM vượt qua điều kiện dwell $\ge 100\text{ms}$ và cổng ổn định để chuyển an toàn sang trạng thái `MOVING`.
- **Pha di chuyển SANG PHẢI (RIGHT Phase)**: 5 bước dịch chuyển $x$ từ $0.50$ đến $0.65$.
  - Mọi sự kiện `MOVE` đều có $\Delta x > 0$ (`dx = 21, 36, 47, 54, 60; dy = 0`).
- **Pha di chuyển SANG TRÁI (LEFT Phase)**: 5 bước dịch chuyển $x$ từ $0.65$ lùi về $0.40$.
  - Mọi sự kiện `MOVE` đều có $\Delta x < 0$ (`dx = -31, -57, -76, -89; dy = 0`).
- **Pha di chuyển XUỐNG DƯỚI (DOWN Phase)**: 5 bước dịch chuyển $y$ từ $0.50$ đến $0.65$.
  - Mọi sự kiện `MOVE` đều có $\Delta y > 0$ (`dy = 11, 21, 26, 31, 33`).
- **Pha di chuyển LÊN TRÊN (UP Phase)**: 5 bước dịch chuyển $y$ từ $0.65$ lùi về $0.40$.
  - Mọi sự kiện `MOVE` đều có $\Delta y < 0$ (`dy = -18, -32, -42, -51`).

*Ghi chú kỹ thuật*: `pipe.rate_limiter.allow` được mock trả về `True` trong phạm vi kiểm thử cô lập này như một thiết bị đo đạc thử nghiệm (test instrumentation) nhằm đảm bảo mỗi khung hình giả lập đều phát sinh sự kiện đo đạc mà không phụ thuộc vào đồng hồ trôi thời gian thực (`wall-clock sleep`). Toàn bộ logic lọc thời gian, máy trạng thái FSM, và biến đổi tọa độ của pipeline sản xuất được giữ nguyên vẹn 100%.

### 3.3. Kiểm thử bất biến đối tượng cấu hình (Non-Mutation Contract)
Kiểm thử `test_run_camera_does_not_mutate_config_and_preserves_generic_pipeline`:
- Patch `app.load_config` trả về chính xác cá thể đối tượng từ điển `initial_cfg`.
- Chạy `run_camera(allow_real_control=False)`.
- So sánh toàn diện `initial_cfg == pristine_cfg` (bản deepcopy đối chứng) và khẳng định `initial_cfg["motion"]["mirror_x"] == True`.
- Khẳng định pipeline của camera có `mirror_x=False`, trong khi pipeline khởi tạo thông thường với `mirror_x=True` và `mirror_x=False` phản ánh hai chiều ánh xạ ngược nhau chính xác.

---

## 4. Kết Quả Thực Thi Bộ Kiểm Thử (Test Suite Execution Outcomes)

1. **Bộ kiểm thử đơn vị `tests/test_app.py`**:
   - Lệnh: `.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_app.py`
   - Kết quả: **21/21 passed** (235.84s, exit code 0).
2. **Kiểm thử tự chẩn đoán CLI (`app.py --self-check`)**:
   - Lệnh: `.venv\Scripts\python.exe app.py --self-check`
   - Kết quả: **5/5 checks passed** (exit code 0).
3. **Bộ kiểm thử hồi quy toàn diện**:
   - Lệnh: `.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests reports/acc04/test_repair_validation.py reports/acc05/test_acc05_benchmark.py reports/acc07/test_acc07_dryrun.py`
   - Kết quả: **330/330 passed** (137.67s, exit code 0).

---

## 5. Tài Liệu Hướng Dẫn Cử Chỉ Tiếng Việt (`docs/AIRPAD_FUNCTION_GUIDE_VI.md`)

Tài liệu đã được biên soạn và cập nhật chuẩn xác theo đúng cấu trúc mã nguồn thực tế:
- **Cơ chế cuộn trang (`TWO_FINGERS_PARALLEL`)**: Phản ánh chính xác hành vi mã nguồn (`dy = cy - prev`): di chuyển tay xuống dưới cuộn trang lên (`wheel up`), di chuyển tay lên trên cuộn trang xuống (`wheel down`). Không sửa đổi mã cuộn.
- **Yêu cầu vũ trang (`POINT_ONE` arming)**: Làm rõ điều kiện tiên quyết cần duy trì `POINT_ONE` $\ge 100\text{ms}$ và tích lũy 2 frame ổn định trong `MOVING` trước khi hệ thống cho phép kích hoạt click, chuột phải, cuộn hoặc thu phóng.
- **Ngưỡng kéo chuột**: Ghi nhận chính xác điều kiện $\ge 400\text{ms}$ (`elapsed >= self.drag_threshold_ms`).
- **Tư thế ngắt an toàn (`OPEN_PALM`) & không xác định (`UNKNOWN`/`OTHER`)**: Nêu rõ hành động chỉ được kích hoạt sau khi tư thế đi qua bộ lọc thời gian đa số; timeout 5 khung hình áp dụng trong trạng thái `MOVING`.
- **Góc dừng an toàn (`Failsafe Corner`)**: Nêu rõ cơ chế kiểm tra trên tọa độ pipeline đã ánh xạ (`mapped_pos`), với kích thước vùng góc cấu hình là 15 pixel (`corner_size = 15`).
- **Phím tắt dừng khẩn cấp (`Ctrl + Alt + Q`)**: Làm rõ nguyên lý hook bàn phím mức hệ thống; nhấn phím sẽ khóa dừng vĩnh viễn (`latched stop`) và không thể phục hồi bằng bất kỳ cử chỉ bàn tay nào; người dùng phải đóng qua `Q`/`Esc` và khởi động lại ứng dụng. Nêu rõ giới hạn an toàn kỹ thuật (nếu luồng chính hoặc camera bị treo sâu thì không bảo đảm ngắt phần cứng tuyệt đối).
- **Phạm vi thu phóng (`L_SHAPE`)**: Gửi phím `Ctrl + Con lăn chuột` tới ứng dụng mục tiêu (trình duyệt, tài liệu), không phải kính lúp toàn màn hình của hệ điều hành.
- **Lệnh thực thi**: Nêu rõ việc chạy qua `.venv\Scripts\python.exe` kèm cờ `--model reports/acc05/candidate_personal_svm.json`, lưu ý mô hình sản xuất mặc định là mô hình yếu hơn và chưa được thăng cấp.

---

## 6. Giới Hạn Kỹ Thuật & Khuyến Nghị Vận Hành

1. Không mở thiết bị camera vật lý và không kích hoạt điều khiển chuột hệ điều hành thực trong suốt quá trình phát triển/kiểm thử.
2. Mô hình ứng viên cá nhân hóa tiếp tục được lưu giữ an toàn tại `reports/acc05/candidate_personal_svm.json` và không bị thăng cấp ghi đè vào `models/airpad_svm.json`.
3. Người dùng nên luôn chạy thử ở chế độ mặc định (dry-run) trước khi bật cờ `--real-control`, và nên thực hiện các thao tác kiểm thử ban đầu trên một cửa sổ ứng dụng an toàn (ví dụ: Notepad).

---
**Xác nhận hoàn thành**: ACC-08 đã được kiểm định và đóng băng hoàn tất.
`ACC08_DIRECT_DONE`
