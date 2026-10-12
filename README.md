# Robot Panda gắp & đặt 3 vật vào Frame_B (robosuite + MuJoCo)

Robot **Franka Panda** gắp lần lượt **Obj_A, Obj_C, Obj_D** từ `Pickup_Zone_A` và đặt vào `Frame_B`.
Mỗi vật phải nằm **vuông góc (90°)** với trục chính của khung và trong **slot riêng, không chồng nhau**.

- Không huấn luyện mô hình nào: robot dùng bộ điều khiển OSC_POSE mặc định của robosuite và một **máy trạng thái** (state machine) xác định.
- Vị trí và góc 3 vật được **ngẫu nhiên theo seed**, nên chạy lại cùng seed cho đúng cùng bố cục.

## Cấu trúc thư mục

| File | Chức năng |
|---|---|
| `custom_env4.py` | Môi trường (3 vật, Frame_B, slot) và `PickPlaceController` (máy trạng thái) |
| `train_state4.py` | Chạy các episode **không cần cửa sổ** (tùy chọn `--render`), in kết quả từng vật |
| `evaluate4.py` | Chạy các episode **có cửa sổ** để quan sát robot |
| `benchmark4.py` | Chạy nhiều episode, tính các chỉ số và lưu `.csv` / `.json` |

## Cài đặt

Yêu cầu: requirements.txt.

```bash
pip install -r requirements.txt
```

## Cách chạy

### 1. Train

```bash
python train_state4.py --episodes 5 --seed 42 --render --sleep 0.02
```


### 2. Evaluate

```bash
python evaluate4.py --episodes 3 --sleep 0.02
```


### 3. Benchmark

```bash
python benchmark2.py --episodes 200 --seed 1000
```

Mỗi episode in một dòng. Cuối cùng in bảng tổng hợp và lưu `<out>.csv`, `<out>.json`.

**Tiêu chí thành công của một episode:** sau khi robot xong và cảnh đã ổn định, cả 3 vật đều thỏa:
sai số vị trí xy < 2,5 cm, sai số độ cao < 2 cm, sai số góc so với 90° < 15°, và không còn bị kẹp trong tay.

**Các chỉ số in ra:** tỉ lệ thành công (kèm khoảng tin cậy 95%), tỉ lệ từng vật, sai số vị trí và góc, thời gian mỗi vật và mỗi episode, và bảng các lỗi (vật nào, trạng thái nào, vì sao).

**Cột trong file CSV:** `seed, success, steps, sim_time_s, fail_object, fail_state, fail_reason`, rồi với mỗi vật: `<Obj>_ok, <Obj>_slot, <Obj>_xy_cm, <Obj>_yaw_deg`.

> Thời gian mô phỏng = số bước / 20 Hz, không phải thời gian chạy thật của máy.

## Cách robot hoạt động

Với mỗi vật (theo thứ tự A, C, D), controller đi qua các trạng thái:

1. `ALIGN_PICK`: di chuyển lên trên vật và xoay tay kẹp cho khớp góc của vật.
2. `DESCEND_PICK`: hạ xuống vật.
3. `CLOSE_GRIPPER`: đóng ngón kẹp.
4. `LIFT`: nâng lên.
5. `APPROACH_PLACE`: mang tới slot và xoay vật cho vuông góc với trục chính của Frame_B.
6. `DESCEND_PLACE`: hạ vật vào slot.
7. `OPEN_GRIPPER`: nhả vật.
8. `RETRACT`: rút lên, đo sai số, rồi chuyển sang vật tiếp theo.

Kết thúc ở `DONE` khi cả 3 vật đạt, hoặc `FAILED` nếu có lỗi (controller in lý do ra màn hình).

Một số điểm thiết kế:
- Khi đặt, robot điều khiển theo **vị trí và góc của chính vật**, không theo đầu kẹp, nên vật nằm đúng giữa slot.
- Tay kẹp luôn được giữ thẳng đứng (điều chỉnh roll/pitch), nếu không vật bị nghiêng sau khi xoay 90°.
- `choose_grasp_yaw` chọn góc kẹp tương đương (vật chữ nhật giống nhau sau 180°) trong ±100° gần góc cổ tay hiện tại nhất, để tránh xoay cổ tay gần 180° và chạm giới hạn khớp.
- Hàm `get_next_available_slot(frame_id, orientation)` trả về slot trống đầu tiên cùng góc 90° so với trục khung.

## Tùy chỉnh

Các hằng số nằm ở đầu `custom_env4.py`:

```python
RANDOMIZE_OBJECTS = True            # False = 3 vật xếp hàng cố định
RANDOM_X_RANGE = (-0.15, 0.15)      # vùng x đặt vật (m)
RANDOM_Y_RANGE = (-0.30, -0.08)     # vùng y đặt vật (m)
RANDOM_YAW_RANGE = np.deg2rad(90.0) # ±90° = mọi hướng
MIN_OBJ_DISTANCE = 0.11             # khoảng cách tối thiểu giữa các vật (m)
OBJ_HALF = [0.035, 0.020, 0.020]    # nửa kích thước vật (dài, rộng, cao)
FRAME_CENTER = [0.0, 0.20]          # tâm Frame_B
SLOT_SPACING = 0.09                 # khoảng cách giữa các slot (m)
PLACE_XY_TOL = 0.025                # ngưỡng sai số vị trí (m)
PLACE_YAW_TOL = np.deg2rad(15.0)    # ngưỡng sai số góc
```

Lưu ý khi chỉnh:
- Mở rộng vùng đặt vật quá xa có thể vượt tầm với của robot hoặc rơi vào Frame_B (y khoảng 0,145–0,255), khi đó tỉ lệ thành công giảm.
- Giảm `MIN_OBJ_DISTANCE` dưới khoảng 0,10 thì ngón kẹp có thể chạm vật bên cạnh.

## Kết quả benchmark (200 episode, seed 1000–1199, vị trí ngẫu nhiên)

| Chỉ số | Trước khi sửa góc kẹp | Sau khi sửa |
|---|---|---|
| Episode thành công | 199/200 | 200/200 |
| Sai số vị trí trung bình / lớn nhất | 0,68 / 1,70 cm | 0,68 / 1,70 cm |
| Sai số góc trung bình / lớn nhất | 0,23° / 1,09° | 0,23° / 1,09° |
| Thời gian mô phỏng mỗi episode | ~20 s | ~19,9 s |

