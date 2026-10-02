# AirPad — đề xuất hướng chuột thật + cử chỉ bổ trợ

Ngày: 2026-10-02. Trạng thái: ĐỀ XUẤT, chưa triển khai/giao Worker. Chỉ Brain phân tích, chạy chẩn đoán DryRun và viết báo cáo; không sửa code, config, model hay dữ liệu. Yêu cầu hiện tại là đề xuất cải tiến, không phải lệnh triển khai.

## Phản hồi thực tế và kết luận

User cho biết POINT_ONE dùng được, scroll và click trái/phải khó dùng, zoom và ba ngón chưa dùng được; đổi sang click làm con trỏ rời mục tiêu. User muốn phần mềm dùng kết hợp cử chỉ tay và chuột, cho phép đề xuất bỏ click bằng cử chỉ. Flicker2/3 trước đó đã được user yêu cầu không sửa; giữ quyết định đó, không coi là bằng chứng mọi hành động đều an toàn.

Khuyến nghị: ưu tiên chế độ Hybrid/Gesture Assist. Chuột thật giữ quyền định vị, click trái/phải và kéo-thả; tay phụ chỉ bổ trợ cuộn, rồi zoom sau khi cuộn đạt. POINT_ONE vẫn có thể tồn tại trong AirMouse riêng nhưng không tự giành con trỏ trong Hybrid. Tắt click/drag cử chỉ trong Hybrid thay vì xóa implementation ngay. Trì hoãn ba-ngón navigation; không cố hoàn thiện cả9 cử chỉ cùng lúc. Giữ classifier/model và SafetyGate hiện hữu; đây là đổi phân công tương tác, không cần thay kiến trúc nhận diện.

## Bằng chứng đọc code và kiểm thử độc lập

Đọc FSM, pipeline, temporal, cursor/scroll/zoom/swipe engines, WindowsController, config và test chuỗi động. Các test hiện hữu kiểm tra sự kiện/nhãn đúng trong chuỗi lý tưởng, chưa đo hoàn thành thao tác trên mục tiêu thực tế hoặc phối hợp với chuột vật lý.

Focused regression command:

```text
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider tests/test_fsm.py tests/test_fsm_phase2.py tests/test_pipeline.py tests/test_motion.py tests/test_windows.py tests/test_safety_contract.py
```

Kết quả104 passed5.99s, exit0, chunk30e48a. Không chạy lại full356 trong vòng đề xuất này; full356 trước đó thuộc ACC09 acceptance, không phải chứng nhận usability.

Hai chẩn đoán trong bộ nhớ, pyautogui/keyboard giả trước imports, controllerDryRun, config camera chỉ deepcopy/mirror_x=False và RateLimiter dùng thời gian frame giả qua patch time.time (không chỉnh ngưỡng hoặc sleep). Classifier trả nhãn điều khiển để cô lập interaction, không phải đánh giá SVM. Chunkf2fcf2, exit0:

1. **Cursor drift khi bắt đầu pinch:** sau10 framePOINT_ONE ổn định, ngón trỏ landmark8 đổi x.50→.65 ngay khi rawPINCH_CLOSED. Với filter/FSM/motion mặc định, ba frame đầu vẫn filteredPOINT_ONE/MOVING và phátMOVE: điểm ban đầu(960,540)→(1046,540)→(1094,540)→(1133,540), rồi mới CLICK_PENDING. Raw nhãn đã đúng từ frame đầu, nên chỉ tăng accuracy classifier không loại được cửa sổ này. Đây là cơ chế tái hiện phù hợp phản hồi, không chứng minh chính xác tọa độ/chuỗi ở phiên camera của user.
2. **Scroll chậm bị bỏ trước khi tích lũy:** ScrollEngine mặc địnhdead_zone.005 nhận100 frame(dy=.002, dx0, dt33.3ms): tổngdịch chuyển chuẩn hóa.2 nhưng0 event vàaccum0. Đối chứng20 framedy.01 phát23 wheel units. Bộ dead-zone trên delta từngframe có thể làm chuyển động chậm không cuộn, dù tổngdịch chuyển đủ lớn. Chưa đo delta thật của user; không quy toàn bộ lỗi scroll cho cơ chế này hoặc tự chọnngưỡng mới.

Các vấn đề đã có bằng chứng trong review2026-10-01 vẫn nằm ngoài ACC09: zoom giữ baseline qua mode exit, swipe không nhận exit/rearm từ pipeline, mapper mặc định1920×1080. PyAutoGUI cài tại máy cóPAUSE=.1 mỗi API call; controller gọi đồng bộ không override. Đây là yếu tố độ trễ suy ra từ source, không phải latency benchmark thật. Không tự ý giảmPAUSE/bỏ fail-safe hoặc tạo workerthread.

Zoom hiện yêu cầu POINT_ONE arming, L_SHAPE ổn định và dwell250ms rồi đổi khẩu độ trong khi vẫn nhậnL_SHAPE; co ngón có thể làm mất tư thế (giả thuyết chưa có trace). Ba-ngón chỉ gạt ngang mới sinh Back/Forward, không phải giữ ba ngón là đủ. Chưa có trace thực tế để kết luận nguyên nhân từng chức năng; không nói user làm sai.

## Phân công tương tác đề xuất

| Thao tác | Hybrid mặc định | Mức ưu tiên |
|---|---|---|
|Định vị con trỏ, left/right click, drag|Chuột thật; cấm OS MOVE/CLICK/DOWN từ cử chỉ|Nền tảng đầu tiên|
|Cuộn dọc|Hai ngón của tay phụ, chỉ trong phiên gesture được bật rõ ràng|Chức năng bổ trợ đầu tiên|
|Zoom|Lệnh zoom bổ trợ, sau khi mode entry/exit và app hỗ trợ được kiểm chứng|Sau scroll|
|Ba-ngón Back/Forward|Tắt/chưa đưa vào bảnHybrid đầu|Trì hoãn|
|POINT_ONE AirMouse|Chế độ riêng bật chủ động; không chạy đồng thời quyền trỏHybrid|Tùy chọn, không xóa|

Hybrid không được bắt user POINT_ONE trước mỗi lần cuộn vì điều đó vừa giành con trỏ vừa làm workflow lặp thao tác. Thay bằng phiên gesture bật/tắt rõ ràng (phím tắt/nút giao diện, binding cần chốt khi triển khai); mặc định tắt, HUD rõ OFF/READY/SCROLL. Vào mode cần tư thế ổn định, frame đầu chỉ chụp baseline; rời mode/hand loss/unknown/dừng phải dừng output, reset baseline; không dùng mode latch để tiếp tục cuộn khi đổi tư thế. Chuột vật lý không bị can thiệp, đặc biệt trong lúc click/kéo. Mất chắc chắn chỉ dừng gesture, không khóa chuột thật.

Tay phụ có thể là tay trái, trong khi dữ liệu hiện chỉ của một user và điều kiện cũ; cần kiểm tra thực tế với tay định dùng, không mặc nhiên coi chuẩn hóa handedness đảm bảo chất lượng bằng nhau. Không yêu cầu thu thêm toàn bộ9 cử chỉ hoặc training lại trước khi có lỗi ghi nhận.

Nếu vẫn muốn click không chạm chuột về sau, giữ nó là tùy chọn AirMouse: chốt vị trí mục tiêu trước khi pinch, chặn cập nhậtcursor trong chuyển trạng thái, táchclickkhỏidrag và đo thời gian/offset thực tế. Có chi phí debounce/đóng băng/khôi phục, nên không là ưu tiên cho mục tiêuHybrid hiện tại.

## Đúng một task tiếp theo đề xuất — Hybrid v1: scroll-only assist

Chưa giao task vì user đang yêu cầu phương án; chờ chốt hướng và cách bật phiên, không tự triển khai kiến trúc/profile.

Mục tiêu: một profile tùy chọn riêng dùng chung detector/model/safety, với chuột vật lý sở hữucursor/click/drag và gesture chỉcuộndọc có bật/tắt rõ ràng. Không gộp sửaAirMouseclick,zoom,swipe,screen mapper,latencybackend hoặc retraining.

Phạm vi dự kiến cho duy nhất Antigravity: app.py (chọnprofile/toggle/HUD), src/runtime/pipeline.py và src/gesture/fsm.py (arming/policy theo profile), src/motion/scroll.py (tích lũy chuyển động chậm và lifecycle chỉ saureproduction/test), schema/config mới nếu cần để thêmprofileexplicit mà không đổi legacydefault, focusedtests/app/pipeline/FSM/motion/safety vàdocs functionguide/task reports. Whitelist action phải kiểm tra trướcsideeffectFSM lẫnpipeline; chỉ chặn cuốipipeline không đủ vìFSM phátclick/down trực tiếp. Không vô hiệu hóa release/cleanup, hook admission hay SafetyGate. Model/dataset/train/evaluation/vision/mirrorRGB và frozenbaseline artifacts ngoài scope.

Kiểm thử/tiêu chí dự kiến:

- DryRun vàOSmock proves tất cả9pose + transition +UNKNOWN/NO_HAND không phátMOVE/CLICK/DOWN/swipe/zoom trongHybrid; chỉ cho phépverticalscroll khiactivation/tưthếmodeđủ.
- Scroll không cầnPOINT_ONE arming, OFF luônzerooutput, entry/exit/reentry khôngdelta/modecũ; handloss/uncertain/emergencystop giữP1contract.
- Chậm, nhanh, đứngyên, rung vàtốcđộframe khác nhau được đo qua timestamp replay; không chỉ giảmdead_zone để đạt1case. Chưa tự đặtthreshold tối ưu.
- LegacyAirMouse tests vàfrozen3320-frame replay không đổi khikhôngchọnHybrid; full356 baseline+newtests vàselfcheck5/5 phải đạt, pins bảo toàn.
- Trước realcontrol mới, Brain review no-indexdiff/candidateprofile/safety tests; sau đóuser kiểm thử ngắn trêncửa sổtrống, vừađịnhvị/clickchuộtthật vừa cuộn tayphụ. Báo sốlần taskhoànthành, lệnhngoàiýmuốn vàđộtrễ, không thay bằngaccuracy từngframe. Quy trình tayphụ được khai báo rõ.

Benchmarkfalsemode/output đi trước dispatch và code onlyWorker. Không giao hai code task; không bậtHybrid/thu dữ liệu/đổiinput profile chỉ từ bảnđềxuất này.

## Tham khảo thiết kế

Microsoft Learn về nhiều loại input yêu cầu cân nhắc loại nào được hỗ trợ theo thời điểm/tình huống và loại nào ưu tiên: https://learn.microsoft.com/en-us/windows/apps/develop/input/multiple-input-design-guidelines . Đây là hướng dẫn nền cho phân quyềninput; khuyến nghịHybrid trên là suy luận riêng từ code/bằng chứngAirPad, không khẳng định tài liệu đã chứng minh webcamgesture tốt hơn conlănchuột.
