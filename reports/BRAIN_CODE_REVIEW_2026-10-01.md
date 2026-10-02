# Brain — review bổ sung sau ACC-08 (2026-10-01)

Kết luận: giữ bản sửa hướng camera ACC-08; không có bằng chứng cần hoàn nguyên. Tuy nhiên, runtime chưa đủ tin cậy để tiếp tục điều khiển chuột thật: đã tái hiện lỗi đồng bộ dừng khẩn cấp và mất theo dõi thao tác nhả thất bại. Giữ phiên thử thủ công dừng. Không sửa code, không giao task cho Worker, không mở camera/điều khiển OS.

## Phạm vi và bằng chứng

Đọc báo cáo ACC-07/08, review trước, benchmark ACC-05 và shadow ACC-06; đối chiếu no-index diff với `reports/acc08/app.py.initial`. Đọc app, pipeline, FSM, bộ lọc, adapter, các engine motion, Windows controller, failsafe, canonicalization, SVM và khôi phục bundle. Workspace không có Git; không tạo Git hoặc hoàn nguyên file.

Bản sửa ACC-08 chỉ deepcopy cấu hình camera và đặt mirror_x=False, phù hợp detector nhận khung đã lật. Hash app/tests/guide và hai bundle vẫn khớp review đã chấp nhận. Model/nguồn training không đổi; accuracy trước đó 91,58% so với baseline 25,08% trên 1663 frame held-out, shadow 89,94% so với 27,96% trên 905 frame target. Không có benchmark nhận diện mới; các số này không chứng minh độ chính xác thao tác động hoặc độ an toàn OS và không đại diện cho bạn bè/ngày khác.

Kiểm thử độc lập hiện tại:

```powershell
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider tests/test_pipeline.py tests/test_windows.py tests/test_fsm.py tests/test_fsm_phase2.py tests/test_motion.py
# 78 passed in 0.25s, exit 0
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider tests/test_svm.py tests/test_evaluation.py tests/test_canonicalize.py tests/test_shadow_trial.py reports/acc05/test_acc05_benchmark.py
# 70 passed in 9.92s, exit 0
```

Tổng 148 test hiện hữu đạt. Không chạy lại toàn bộ 330 test; lần trước 330/330 đã đạt, code được xác minh hash không đổi. Các phép tái hiện dưới đây chạy script trong bộ nhớ, DryRun hoặc PyAutoGUI mock, không ghi/sửa test nguồn và không phát OS input. Test đang xanh không bao phủ các lỗi này.

## Phát hiện theo ưu tiên

### 1. P1 — dừng khẩn cấp đua với phát lệnh, có thể để nút trái bị giữ

Vị trí: `src/runtime/pipeline.py:83,92,190`, `src/gesture/fsm.py:225`, `src/platform/windows.py:45`.

Pipeline kiểm tra stop đầu frame nhưng không đồng bộ giao dịch phát lệnh với callback hotkey. WindowsController và FSM cập nhật sổ nút đang giữ sau mouseDown. Callback có thể cleanup trước khi sổ được cập nhật, rồi frame tiếp tục gửi lệnh/cập nhật trạng thái.

Hai interleaving xác định đã tái hiện:

- Đặt callback stop vào thời điểm `rate_limiter.allow()` trả về sau kiểm tra corner: `stopped=True, FSM=PAUSED, action=MOVE`, event MOVE `(960,540)` vẫn ghi sau stop. Dùng limiter thật, classifier POINT_ONE giả để cô lập dispatch.
- Dùng PyAutoGUI mock cho WindowsController, gọi stop trong `mouseDown` khi FSM bắt đầu kéo: kết quả `stopped=True, FSM=PAUSED, fsm_mouse_down=True, controller_held=['left']`. Cleanup đã chạy trước khi thao tác giữ được ghi nhận. Các frame tiếp theo bị stop chặn, không tự nhả nút này.

Đây là mô phỏng thứ tự chạy hợp lệ giữa hai luồng, chưa phải đo tần suất lỗi trên máy thật. Stop xảy ra trong mapper trước corner-check bị chặn đúng; cửa sổ lỗi nằm muộn hơn tại dispatch/held-input bookkeeping. Chỉ thêm một lần kiểm tra cờ trước lệnh không đủ để loại bỏ race.

### 2. P1 — cleanup xóa nút/phím khỏi sổ dù nhả thất bại

Vị trí: `src/platform/windows.py:70`.

`finally` discard held-state bất kể `mouseUp/keyUp` thành công. Tái hiện giữ left, mock mouseUp ném lỗi, cleanup hai lần: `tracked_after_failure=[], retry_mouseUp_calls=0`. Hệ thống không còn biết cần nhả lại, trong khi trạng thái OS chưa được xác nhận.

Ctrl/Alt trong zoom/swipe dùng trực tiếp PyAutoGUI, không đi qua `_held_keys`; lỗi keyUp cũng không được cleanup theo sổ hiện tại. Đây là đường rủi ro cùng nhóm release, không phải bằng chứng OS thực tế đã kẹt phím trong phiên trước.

### 3. P1 — real-control vẫn khởi tạo nếu hook dừng không đăng ký

Vị trí: `src/safety/failsafe.py:84`, `src/runtime/pipeline.py:67`, `app.py:115`.

register chỉ in lỗi và không bắt caller fail-closed. Tái hiện mock đăng ký không thành công, WindowsController/PyAutoGUI đều mock: pipeline trả về `real_controller=WindowsController, hotkey_registered=False, emergency_stopped=False`. Native PyAutoGUI FAILSAFE đã bị tắt; corner monitor chỉ kiểm tra tọa độ ánh xạ trong nhánh move. Thử trước đã đăng ký hook thành công, nhưng đó không bảo đảm lần khởi động sau.

### 4. P2 — mapper cố định 1920×1080, corner monitor dùng kích thước thật

Vị trí: `app.py:67,110`, `src/motion/cursor.py:27`.

create_pipeline không truyền screen_size hoặc gọi set_screen_size. Mock màn hình 1366×768: actual `(1366,768)`, mapper `(1920,1080)`, monitor `(1366,768)`. Kết quả mapping và corner-check không cùng hệ tọa độ, vùng biên/cảm giác điều khiển có thể sai. Ảnh hưởng phụ thuộc độ phân giải/DPI thật, chưa kết luận màn hình người dùng đang có kích thước này.

### 5. P2 — không reset trạng thái zoom/swipe khi đổi mode có bàn tay

Vị trí: `src/runtime/pipeline.py:110,212,224`.

Reset các engine chỉ ở NO_HAND; pipeline không báo exit pose cho SwipeEngine và không reset baseline zoom khi rời mode.

- Chuỗi filtered THREE_FINGERS gạt phải → POINT_ONE ổn định → THREE_FINGERS gạt phải ở vị trí khác: lần đầu SWIPE_FORWARD, sau exit `is_armed=False`, lần hai không có swipe. Engine không nhận nhánh `is_three_fingers=False`.
- POINT_ONE → L_SHAPE khẩu độ .12 → POINT_ONE vẫn .12 để thoát sạch → đổi khẩu độ .32 trong POINT_ONE → vào lại L_SHAPE giữ yên .32: baseline sau exit vẫn .12, phát ZOOM ngay khi vào lại dù chưa co/giãn trong mode mới. Dùng default filter/FSM và limiter instrumented cho event đo đạc; không suy ra lỗi classifier.

### 6. P2 — đường điều khiển thật có pause 100ms mặc định mỗi lệnh

Vị trí: `src/platform/windows.py:42`, thư viện cài đặt `.venv/Lib/site-packages/pyautogui/__init__.py:562,631`.

PyAutoGUI PAUSE=.1; controller gọi move/click/scroll không tắt pause theo từng lệnh. Vòng camera đồng bộ chờ lệnh OS, nên frame có MOVE phải cộng khoảng 100ms trước chi phí detector/model khác. Có thể làm mượt/nhạy kém hơn dry-run và thay đổi độ trễ bộ lọc, click timing. Đây là suy luận từ code thư viện cục bộ, chưa có benchmark latency thật; không tự hạ pause hoặc đổi kiến trúc.

## Đúng một ưu tiên tiếp theo — chỉ đề xuất, chưa giao

Task đề xuất: làm luồng dừng khẩn cấp/phát lệnh OS và theo dõi giữ/nhả thành một hợp đồng an toàn có đồng bộ, bao gồm fail-closed nếu hook không đăng ký. Giải quyết nhóm P1 trước việc điều khiển thật tiếp theo; không gộp sửa screen-size, zoom/swipe hay latency vào task này.

Phạm vi dự kiến: `src/runtime/pipeline.py`, `src/gesture/fsm.py`, `src/platform/windows.py`, `src/safety/failsafe.py`, phần khởi tạo trong `app.py` nếu cần, các test tương ứng và báo cáo. Chỉ Antigravity được triển khai. Giữ nguyên classifier/bundle/data, RGB/mirror, cử chỉ, ngưỡng và kiến trúc hiện có.

Tiêu chí: test interleaving bằng Event/Barrier/hook xác định (không sleep hên xui) cho stop trong MOVE và drag-start; không phát lệnh mới sau stop hoàn tất, không còn held-state chưa xử lý, release-failure giữ thông tin để retry và báo thất bại thay vì success giả, Ctrl/Alt đi qua tracking, hook-failure ngăn real-control. Toàn bộ test chỉ mock/DryRun, full 330 test nền cộng test mới và self-check đạt; benchmark/model hashes không đổi. Không mở webcam/OS input. Không giao task vì yêu cầu hiện tại là review.

## Trạng thái dừng và usage/credit

Process enumeration không có AirPad camera/pytest còn chạy, chỉ helper Antigravity. Worker đã ACC08_DIRECT_DONE và idle; không gửi thêm prompt. Không có worker cần reclaim trong Run hiện tại.

Orca account telemetry lần cuối status=ok: session đã dùng63%/300 phút, weekly77%; chưa chạm hạn mức. Không dùng API key, không mua credit, không đổi model/account, không tiêu lượt reset hoặc mở agent mới. Công cụ không cung cấp khóa billing/fallback; telemetry là snapshot và không thể đảm bảo ranh giới trừ credit theo thời gian thực. Sẽ dừng khi nhận tín hiệu hết hạn mức; không hứa có thể khóa chi tiêu chỉ bằng prompt.

OpenAI Docs dùng để kiểm tra giới hạn: https://learn.chatgpt.com/docs/pricing — available credits cho phép tiếp tục sau included limits; `/status`/usage dashboard cho xem hạn mức. Không thay đổi thiết lập thanh toán.
