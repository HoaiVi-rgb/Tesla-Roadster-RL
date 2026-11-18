# ĐẶC TẢ KỸ THUẬT VẬT LÝ 2D (2D PHYSICS SPECIFICATION)
## DỰ ÁN: XE ĐÁP ĐẤT ROADSTER 4 VÒI PHUN (2D FLYING CAR SIMULATOR)
### TÁC GIẢ: SENIOR 2D PHYSICS ENGINEER

---

## 1. TỔNG QUAN KIẾN TRÚC VẬT LÝ (EXECUTIVE SUMMARY)

Tài liệu này xác lập mô hình toán học và kỹ thuật vật lý 2D chuẩn mực phục vụ bài toán điều khiển tiếp đất hạ cánh an toàn cho xe bay Roadster trang bị 4 vòi phun khí nén cố định.

Mô hình vật lý tuân thủ nguyên lý chuyển động vật rắn Newton-Euler 2D, giải quyết hoàn toàn thông qua phương trình vi phân và tích phân số **Symplectic Euler** tại tần số **120 Hz** ($\Delta t_{phys} = \frac{1}{120}\text{ s} \approx 0.008333\text{ s}$).

```
========================================================================================
                          CAM KẾT LIÊM CHÍNH VẬT LÝ (ZERO-CHEATING)
========================================================================================
[X] KHÔNG hard-code quỹ đạo rơi (No hard-coded trajectory).
[X] KHÔNG hard-code điều kiện đáp (No scripted landing).
[X] KHÔNG tự động ép góc nghiêng theta về 0 (No artificial angle clamping).
[X] KHÔNG tự động giảm vận tốc vx, vy khi gần đất (No artificial velocity damping).
[X] KHÔNG có logic "nếu gần mặt đất thì tự động cân bằng" (No ground auto-assist).
[V] TOÀN BỘ CHUYỂN ĐỘNG do tương tác giữa Trọng lực và Lực đẩy vòi phun tạo ra!
========================================================================================
```

---

## 2. HỆ QUY CHIẾU, HÌNH HỌC & THÔNG SỐ QUÁN TÍNH (GEOMETRY & INERTIA)

### 2.1. Hệ quy chiếu (Coordinate Frames)
1. **Hệ trục thế giới (World Frame $\mathcal{F}_W$)**:
   - Gốc tọa độ $(0, 0)$ đặt tại tâm mặt đất bãi đáp.
   - Trục $+X_W$ hướng sang phải theo phương ngang.
   - Trục $+Y_W$ hướng thẳng đứng lên trời.
2. **Hệ trục thân xe (Body Frame $\mathcal{F}_B$)**:
   - Gốc tọa độ $(0, 0)$ đặt trùng với Trọng tâm xe (Center of Mass - COM).
   - Trục $+X_B$ hướng dọc thân về đầu xe (sang phải khi xe thăng bằng).
   - Trục $+Y_B$ hướng vuông góc lên nóc xe.
   - Góc định hướng $\theta$ (rad): Góc nghiêng giữa trục $+X_B$ và $+X_W$.
     - $\theta = 0$: Xe nằm ngang hướng sang phải.
     - $\theta > 0$: Quay ngược chiều kim đồng hồ (CCW - Ngẩng đầu xe / Pitch Up).
     - $\theta < 0$: Quay cùng chiều kim đồng hồ (CW - Chúi đầu xe / Pitch Down).
   - Vận tốc góc $\omega = \dot{\theta}$ ($\text{rad/s}$): Chiều dương CCW.

### 2.2. Ma trận quay (Rotation Matrix)
Phép biến đổi một vector lực hoặc khoảng cách từ hệ trục thân xe $\mathcal{F}_B$ sang hệ trục thế giới $\mathcal{F}_W$:

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

$$\vec{v}_W = R(\theta) \vec{v}_B = \begin{bmatrix} v_{Bx} \cos\theta - v_{By} \sin\theta \\ v_{Bx} \sin\theta + v_{By} \cos\theta \end{bmatrix}$$

### 2.3. Thông số hình học và khối lượng
- **Khối lượng xe ($m$)**: $1,200.0\text{ kg}$.
- **Chiều dài thân ($L$)**: $4.4\text{ m}$.
- **Chiều cao thân ($H$)**: $0.9\text{ m}$.
- **Độ cao nghỉ của trọng tâm ($h_{rest}$)**: $0.72\text{ m}$ (khi xe đậu cân bằng trên mặt đất, mặt phẳng đáy lốp tiếp xúc $y=0$, trọng tâm nằm tại $y = 0.72\text{ m}$).
- **Mômen quán tính quanh trục Z đi qua trọng tâm ($I_{zz}$)**:
  Xem thân xe như một khối hộp chữ nhật 2D phân bố khối lượng đồng nhất:

$$I_{zz} = \frac{1}{12} m \left( L^2 + H^2 \right) = \frac{1}{12} \times 1,200 \times \left( 4.4^2 + 0.9^2 \right) = 100 \times (19.36 + 0.81) = 2,017.0\text{ kg}\cdot\text{m}^2$$

---

## 3. PHƯƠNG TRÌNH ĐỘNG LỰC HỌC NEWTON-EULER 2D (EQUATIONS OF MOTION)

Vật rắn có 3 bậc tự do phẳng: chuyển động tịnh tiến $(x, y)$ và chuyển động quay quanh trục $z$ ($\theta$).

### 3.1. Động lực học tịnh tiến (Translational Dynamics)
Tổng ngoại lực tác dụng lên xe trong hệ quy chiếu thế giới:

$$\vec{F}_{total}^W = \vec{F}_{gravity}^W + \vec{F}_{thrust}^W$$

Trong đó:
- Trọng lực đặt tại trọng tâm:
  $$\vec{F}_{gravity}^W = \begin{bmatrix} 0 \\ -m g \end{bmatrix} = \begin{bmatrix} 0 \\ -1,200 \times 9.81 \end{bmatrix} = \begin{bmatrix} 0 \\ -11,772.0 \end{bmatrix}\text{ N}$$
- Lực đẩy từ 4 vòi phun trong hệ thế giới:
  $$\vec{F}_{thrust}^W = R(\theta) \vec{F}_{thrust}^B$$

Gia tốc tuyến tính thế giới:

$$\vec{a}_W = \begin{bmatrix} a_x \\ a_y \end{bmatrix} = \frac{\vec{F}_{total}^W}{m} = \begin{bmatrix} F_{x, total}^W / m \\ F_{y, total}^W / m \end{bmatrix}$$

### 3.2. Động lực học quay (Rotational Dynamics)
Mômen ngoại lực tổng hợp $\tau_{net}$ tác dụng lên xe quay quanh trục $z$ đi qua trọng tâm:

$$\tau_{net} = \sum_{i=1}^4 u_i \left( \vec{r}_i^B \times \vec{F}_i^B \right) = \sum_{i=1}^4 u_i \left( r_{ix} F_{iy} - r_{iy} F_{ix} \right)$$

Gia tốc góc:

$$\alpha = \frac{\tau_{net}}{I_{zz}}$$

*(Lưu ý: Do trọng lực $\vec{F}_{gravity}$ tác dụng tại chính trọng tâm, cánh tay đòn $\vec{r}_{gravity} = \vec{0}$, mômen trọng lực quanh COM triệt tiêu bằng 0).*

---

## 4. ĐẶC TẢ HỆ THỐNG VÒI PHUN (4 COLD GAS THRUSTERS)

Xe trang bị 4 miệng phun khí nén cố định trên thân xe. Mỗi vòi được kích hoạt ở chế độ đóng/mở nhị phân $u_i \in \{0, 1\}$.

```
                    ^ +Y_B (Mái xe)
                    |
     ===========================================
     |                                         |
V3   |                    COM                  |   V4
[<]  |                  (0, 0)                 |  [>]
6000N|                                         | 6000N
     ===========================================
          |                               |
          |                               |
         [V1]                            [V2]
        9000N                           9000N
       Nâng đuôi                       Nâng đầu
     (r1 = [-1.8, -0.45])            (r2 = [+1.8, -0.45])
                    ---------------------------> +X_B (Đầu xe)
```

### 4.1. Bảng thông số kỹ thuật từng vòi phun
| Vòi | Tên chức năng | Tọa độ đặt $\vec{r}^B$ (m) | Hướng lực $\vec{d}^B$ | Lực danh định $|\vec{F}|$ (N) | Cánh tay đòn & Mômen $\tau$ (N$\cdot$m) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V1** | Nâng đuôi (Rear Lift) | $[-1.8, -0.45]$ | $[0.0, +1.0]$ | $9,000.0$ | $(-1.8) \times 9000 - (-0.45) \times 0 = \mathbf{-16,200.0}$ (Chúi đầu / CW) |
| **V2** | Nâng đầu (Front Lift) | $[+1.8, -0.45]$ | $[0.0, +1.0]$ | $9,000.0$ | $(+1.8) \times 9000 - (-0.45) \times 0 = \mathbf{+16,200.0}$ (Ngẩng đầu / CCW) |
| **V3** | Đẩy phải (Push Right) | $[-2.2, 0.0]$ | $[+1.0, 0.0]$ | $6,000.0$ | $(-2.2) \times 0 - (0.0) \times 6000 = \mathbf{0.0}$ |
| **V4** | Đẩy trái (Push Left) | $[+2.2, 0.0]$ | $[-1.0, 0.0]$ | $6,000.0$ | $(+2.2) \times 0 - (0.0) \times (-6000) = \mathbf{0.0}$ |

### 4.2. Bảng chân lý 16 hành động rời rạc (Action Decoding Table)
Hành động $a \in [0, 15]$ giải mã nhị phân: $u_1 = a \& 1$, $u_2 = (a \gg 1) \& 1$, $u_3 = (a \gg 2) \& 1$, $u_4 = (a \gg 3) \& 1$.

| Action $a$ | Nhị phân $(V_4 V_3 V_2 V_1)$ | Lực thân $\vec{F}^B$ (N) | Mômen $\tau$ (N$\cdot$m) | Ý nghĩa động học vật lý |
| :--- | :--- | :--- | :--- | :--- |
| **0** | `0 0 0 0` | $[0, 0]$ | $0$ | Rơi tự do thuần túy |
| **1** | `0 0 0 1` | $[0, +9000]$ | $-16,200$ | Nâng đuôi, sinh gia tốc góc quay CW $\alpha = -8.03\text{ rad/s}^2$ |
| **2** | `0 0 1 0` | $[0, +9000]$ | $+16,200$ | Nâng đầu, sinh gia tốc góc quay CCW $\alpha = +8.03\text{ rad/s}^2$ |
| **3** | `0 0 1 1` | $[0, +18000]$ | $0$ | **Nâng thẳng đứng cân bằng**: $18000 > 11772\text{ N} \implies a_y = +5.19\text{ m/s}^2$ |
| **4** | `0 1 0 0` | $[+6000, 0]$ | $0$ | Đẩy xe sang phải: $a_x = +5.0\text{ m/s}^2$ |
| **5** | `0 1 0 1` | $[+6000, +9000]$ | $-16,200$ | Đẩy phải + nâng đuôi |
| **6** | `0 1 1 0` | $[+6000, +9000]$ | $+16,200$ | Đẩy phải + nâng đầu |
| **7** | `0 1 1 1` | $[+6000, +18000]$ | $0$ | Đẩy phải + nâng thẳng đứng cân bằng |
| **8** | `1 0 0 0` | $[-6000, 0]$ | $0$ | Đẩy xe sang trái: $a_x = -5.0\text{ m/s}^2$ |
| **9** | `1 0 0 1` | $[-6000, +9000]$ | $-16,200$ | Đẩy trái + nâng đuôi |
| **10** | `1 0 1 0` | $[-6000, +9000]$ | $+16,200$ | Đẩy trái + nâng đầu |
| **11** | `1 0 1 1` | $[-6000, +18000]$ | $0$ | Đẩy trái + nâng thẳng đứng cân bằng |
| **12** | `1 1 0 0` | $[0, 0]$ | $0$ | Cả hai vòi ngang cùng mở $\implies$ lực triệt tiêu |
| **13** | `1 1 0 1` | $[0, +9000]$ | $-16,200$ | Hai vòi ngang triệt tiêu + nâng đuôi |
| **14** | `1 1 1 0` | $[0, +9000]$ | $+16,200$ | Hai vòi ngang triệt tiêu + nâng đầu |
| **15** | `1 1 1 1` | $[0, +18000]$ | $0$ | Toàn bộ 4 vòi cùng mở $\implies$ nâng thẳng cực đại |

---

## 5. MÔ HÌNH TIÊU THỤ KHÍ NÉN (FUEL DYNAMICS)

Bình khí nén khởi tạo ở mức $100.0\%$. Khi vòi phun mở, lượng khí tiêu hao tỷ lệ với độ lớn lực đẩy:
- Vòi nâng ($9,000\text{ N}$): Tiêu thụ $1.0 \% / \text{s}$.
- Vòi ngang ($6,000\text{ N}$): Tiêu thụ $\frac{6000}{9000} \times 1.0 = 0.667 \% / \text{s}$.

Công thức tiêu thụ trong một bước thời gian $\Delta t$:

$$\Delta\text{fuel} = \left( \sum_{i=1}^4 u_i \times \text{burn\_rate}_i \right) \Delta t$$

$$\text{fuel}_{t+\Delta t} = \max(0.0, \text{fuel}_t - \Delta\text{fuel})$$

> [!IMPORTANT]
> **Cơ chế ngắt lực khi cạn khí (Fuel Depletion Cutoff)**:
> Khi $\text{fuel} \le 0.0$, toàn bộ các vòi phun bị vô hiệu hóa ($F_{thrust} = 0, \tau_{thrust} = 0$), xe rơi hoàn toàn tự do theo định luật hấp dẫn. Không có hiện tượng "tiêu cực số" hay "chạy tiếp khi hết xăng".

---

## 6. PHÁT HIỆN VA CHẠM & TIÊU CHUẨN TIẾP ĐẤT (COLLISION & LANDING)

### 6.1. Đa giác tiếp xúc & Vị trí bánh xe
Xe có 6 điểm tiếp xúc cục bộ trong hệ trục thân xe:
1. Bánh sau (Rear wheel): $\vec{r}_{W1} = [-1.5, -0.72]\text{ m}$
2. Bánh trước (Front wheel): $\vec{r}_{W2} = [+1.5, -0.72]\text{ m}$
3. Góc dưới-sau (Bottom-rear bumper): $[-2.2, -0.45]\text{ m}$
4. Góc dưới-trước (Bottom-front bumper): $[+2.2, -0.45]\text{ m}$
5. Góc trên-trước (Top-front roof): $[+2.2, +0.45]\text{ m}$
6. Góc trên-sau (Top-rear roof): $[-2.2, +0.45]\text{ m}$

Mặt đất là đường phẳng vô hạn tại $y = 0$.
Khoảng cách nhỏ nhất từ các điểm tiếp xúc tới mặt đất:

$$y_{lowest} = \min_{i} \left( y_{COM} + \vec{r}_i^B \cdot [\sin\theta, \cos\theta]^T \right)$$

Va chạm xảy ra khi và chỉ khi:

$$y_{lowest} \le 0.0\text{ m}$$

### 6.2. Tiêu chuẩn tiếp đất an toàn (Safe Landing Ground Truth)
Khi có va chạm xảy ra ($y_{lowest} \le 0$), hệ thống vật lý thẩm định **4 điều kiện ngặt nghèo**:
1. Vận tốc tiếp đất phương đứng: $|v_y| \le 1.0\text{ m/s}$
2. Vận tốc trượt ngang: $|v_x| \le 0.5\text{ m/s}$
3. Độ nghiêng thân xe: $|\theta_{norm}| \le 5^\circ$ ($0.0872665\text{ rad}$)
4. Vận tốc quay: $|\omega| \le 0.2\text{ rad/s}$

- **Đáp an toàn (Safe Landing)**: Thỏa mãn đồng thời cả 4 điều kiện.
- **Va chạm phá hủy (Crash)**: Vi phạm ít nhất 1 trong 4 điều kiện.

### 6.3. Tiêu chuẩn vùng đáp mục tiêu (Target Zone)
Vùng đáp xanh định vị tại trung tâm: $x \in [-2.0, +2.0]\text{ m}$.
Tiếp đất được ghi nhận nằm trong vùng mục tiêu khi cả 2 bánh xe nằm trọn trong khoảng này:

$$-2.0\text{ m} \le x_{W1}^{world} \le +2.0\text{ m} \quad \text{và} \quad -2.0\text{ m} \le x_{W2}^{world} \le +2.0\text{ m}$$

---

## 7. PHƯƠNG PHÁP TÍCH PHÂN SỐ 120 HZ (NUMERICAL INTEGRATOR)

Hệ thống sử dụng **Tích phân Symplectic Euler (Semi-Implicit Euler)**:

$$\begin{aligned}
\vec{v}_{t+\Delta t} &= \vec{v}_t + \left( \frac{\vec{F}_{total}^W}{m} \right) \Delta t \\
\vec{p}_{t+\Delta t} &= \vec{p}_t + \vec{v}_{t+\Delta t} \Delta t \\
\omega_{t+\Delta t} &= \omega_t + \left( \frac{\tau_{net}}{I_{zz}} \right) \Delta t \\
\theta_{t+\Delta t} &= \theta_t + \omega_{t+\Delta t} \Delta t
\end{aligned}$$

### Ưu điểm vượt trội so với Explicit Euler thông thường:
1. **Bảo toàn diện tích pha (Symplectic Property)**: Trong các hệ cơ học bảo toàn như con lắc hay chuyển động rơi tự do không ma sát, Symplectic Euler không làm tăng giả tạo năng lượng cơ học theo thời gian.
2. **Độ ổn định cao ở tần số 120 Hz**: Bước thời gian $\Delta t = \frac{1}{120}\text{ s} \approx 0.00833\text{ s}$ đủ nhỏ để triệt tiêu sai số cắt cụt bậc cao, đảm bảo đường cong chuyển động liên tục trơn tru.

---

## 8. BÁO CÁO KẾT QUẢ KIỂM THỬ (UNIT TEST SUITE REPORT)

Bộ kiểm thử đơn vị tự động đã được lập trình đầy đủ trong `tests/` và chạy thông qua `pytest`:

```bash
$ python -m pytest tests/
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0
rootdir: F:\Programming Study\Reforcement Learning\Đồ án Học tăng cường
collected 23 items

tests\test_collision.py .......                                          [ 30%]
tests\test_fuel.py ..                                                    [ 39%]
tests\test_gravity.py ..                                                 [ 47%]
tests\test_rotation.py ..                                                [ 56%]
tests\test_thrust.py ...                                                 [ 69%]
tests\test_timestep.py ...                                               [ 82%]
tests\test_torque.py ....                                                [100%]

============================= 23 passed in 0.17s ==============================
```

### Chi tiết các ca kiểm thử:
1. **Gravity Tests (`test_gravity.py`)**:
   - `test_free_fall_acceleration`: Xác minh gia tốc không van chính xác $a_y = -9.81\text{ m/s}^2$, $a_x = 0$, $\alpha = 0$.
   - `test_free_fall_kinematics`: Xác minh $v_y(t) = -gt$ và $y(t) \approx y_0 - \frac{1}{2}gt^2$ sau 1.0 giây rơi.
2. **Thrust Tests (`test_thrust.py`)**:
   - `test_vertical_lift_acceleration`: Xác minh lực nâng $V_1+V_2$ ($18,000\text{ N}$) trừ trọng lực sinh gia tốc thẳng đứng chính xác $a_y = +5.19\text{ m/s}^2$.
   - `test_lateral_thrust_acceleration`: Xác minh $V_3$ sinh $a_x = +5.0\text{ m/s}^2$, $V_4$ sinh $a_x = -5.0\text{ m/s}^2$, và $V_3+V_4$ triệt tiêu.
   - `test_rotated_thrust_vector`: Xác minh khi xe nghiêng $90^\circ$, lực nâng vòi $V_1+V_2$ đổi hướng vuông góc sang phương ngang thế giới theo đúng ma trận quay $R(\theta)$.
3. **Torque Tests (`test_torque.py`)**:
   - `test_v1_torque_and_angular_acceleration`: Xác minh vòi $V_1$ sinh mômen $\tau = -16,200\text{ N}\cdot\text{m}$ và gia tốc góc $\alpha = -8.0317\text{ rad/s}^2$.
   - `test_v2_torque_and_angular_acceleration`: Xác minh vòi $V_2$ sinh mômen $\tau = +16,200\text{ N}\cdot\text{m}$ và gia tốc góc $\alpha = +8.0317\text{ rad/s}^2$.
   - `test_balanced_v1_v2_zero_torque`: Xác minh $V_1+V_2$ triệt tiêu hoàn toàn mômen ($\tau = 0.0$).
   - `test_lateral_thrusters_zero_torque`: Xác minh $V_3$ và $V_4$ đi qua trục tâm nên mômen $\tau = 0.0$.
4. **Rotation Tests (`test_rotation.py`)**:
   - `test_continuous_rotation_kinematics`: Tích phân góc quay $\theta$ khớp chính xác lý thuyết động học quay.
   - `test_flip_detection`: Tích phân góc quay tích lũy $\int \omega dt$ nhận diện chính xác số vòng lộn $360^\circ$ hoàn chỉnh.
5. **Collision Tests (`test_collision.py`)**:
   - `test_safe_landing_nominal`: Nhận diện thành công hạ cánh êm ái thỏa mãn cả 4 điều kiện ngặt.
   - `test_crash_high_vertical_velocity`: Nhận diện va chạm thất bại khi $|v_y| > 1.0\text{ m/s}$.
   - `test_crash_high_horizontal_velocity`: Nhận diện va chạm thất bại khi $|v_x| > 0.5\text{ m/s}$.
   - `test_crash_high_tilt_angle`: Nhận diện va chạm thất bại khi $|\theta| > 5^\circ$.
   - `test_crash_high_spin_rate`: Nhận diện va chạm thất bại khi $|\omega| > 0.2\text{ rad/s}$.
   - `test_target_zone_boundaries`: Nhận diện chính xác xe đáp ngoài vùng mục tiêu $[-2, 2]\text{ m}$.
   - `test_out_of_bounds_detection`: Phát hiện xe bay ra ngoài không gian mô phỏng $[-20, 20]\text{ m} \times [0, 25]\text{ m}$.
6. **Fuel Tests (`test_fuel.py`)**:
   - `test_fuel_consumption_rate`: Kiểm tra tốc độ hao tổn khí chính xác theo đơn vị $\%/\text{s}$.
   - `test_fuel_exhaustion_thruster_cutoff`: Xác minh khi khí nén bằng $0$, vòi phun hoàn toàn mất lực, xe rơi tự do.
7. **Timestep Tests (`test_timestep.py`)**:
   - `test_120hz_substepping_count`: Xác minh 1 step của Agent ($0.1\text{ s}$) thực thi chính xác 12 sub-steps $120\text{ Hz}$.
   - `test_early_collision_substep_termination`: Xác minh dừng sub-step ngay khi có va chạm xảy ra, không tính lố.
   - `test_20s_timeout_limit`: Xác minh giới hạn episode 20 giây ($200$ bước).

---

## 9. KẾT LUẬN & CHUYỂN GIAO SẢN PHẨM

Hệ thống vật lý 2D trong package `physics/` đã đạt độ hoàn thiện cao nhất (Production-ready), độc lập tuyệt đối với Agent, không chứa bất kỳ logic gian lận hay heuristic nào, sẵn sàng làm nền tảng vững chắc để phát triển Environment Gym, Heuristic Teacher, và thuật toán Deep Q-Learning ở các bước tiếp theo.
