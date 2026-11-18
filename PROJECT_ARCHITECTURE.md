# KIẾN TRÚC TỔNG THỂ DỰ ÁN (PROJECT ARCHITECTURE BLUEPRINT)
## ĐỒ ÁN HỌC TĂNG CƯỜNG (K5 - KHDL & TTNT)
### ĐỀ TÀI: XE ĐÁP ĐẤT LẤY CẢM HỨNG TỪ XE BAY ROADSTER TESLA (4 VÒI PHUN CỐ ĐỊNH)

---

## 1. TỔNG QUAN HỆ THỐNG & TẬP LỆNH CHỈ ĐẠO (EXECUTIVE SUMMARY)

### 1.1. Mục tiêu cốt lõi
Dự án nhằm giải quyết bài toán điều khiển tiếp đất hạ cánh an toàn cho phương tiện bay phản lực 2D (lấy cảm hứng từ xe bay Tesla Roadster trang bị SpaceX Cold Gas Thrusters). Phương tiện được mô phỏng với 4 vòi phun khí nén cố định, tạo ra 16 tổ hợp hành động rời rạc.

Agent Học Tăng Cường (Reinforcement Learning - RL) phải học được chính sách (policy $\pi$) để kích hoạt các vòi phun đưa xe từ các trạng thái hỗn loạn trên không (độ cao $h \in [4, 9]\text{ m}$, góc lệch $\theta \in [\pm 15^\circ, \pm 180^\circ]$, độ trôi $x \in [\pm 2, \pm 18]\text{ m}$) tiếp đất:
1. **An toàn tuyệt đối**: Xe thăng bằng nằm ngang khi tiếp đất ($|\theta| < 5^\circ$, $|\omega| < 0.2\text{ rad/s}$, $|v_y| < 1.0\text{ m/s}$, $|v_x| < 0.5\text{ m/s}$).
2. **Đúng vùng đáp**: Hai bánh xe nằm trọn trong vùng mục tiêu màu xanh ($x = 0 \pm 1.5\text{ m}$).
3. **Tối ưu thời gian**: Hoàn thành tiếp đất trong thời gian cực ngắn ($< 5.0\text{ s}$) để không bị phạt điểm theo barem (mỗi quá 5 giây trừ 4 điểm).
4. **Kỹ năng nhào lộn (Flip/Roll bonus)**: Có chế độ/khả năng thực hiện lộn vòng $n \times 360^\circ$ trên không rồi tái ổn định và đáp đất an toàn (1 vòng cộng 1 điểm).
5. **Chứng minh RL thực thụ & Giải thích minh bạch**: Khác biệt hoàn toàn với bộ điều khiển phản hồi PID/LQR hay hard-coded script; mạng nơ-ron thực hiện suy diễn $Q(s, a)$ thực sự và có khả năng giải thích tường minh cơ chế MDP.
6. **Đồng bộ chuẩn giao diện**: Giao diện trực quan Canvas/HTML5 và quy tắc rơi vật lý đồng nhất $100\%$ với tài liệu tham chiếu `Roadster_4_Voi_DQN`.

```
========================================================================================
                               BAREM ĐÁNH GIÁ ĐỒ ÁN (10/10 ĐIỂM)
========================================================================================
[1] Đáp an toàn (xe nằm ngang, vận tốc tiếp đất nhỏ)           : Tối đa 2.0 điểm
[2] Đáp đúng vị trí mục tiêu (vùng xanh x = 0)                 : Tối đa 2.0 điểm
[3] Thời gian đáp càng nhanh càng tốt (mỗi quá 5s trừ 4 điểm)  : Bắt buộc <= 5s
[4] Kỹ năng lộn n vòng trên không                              : +1 điểm/vòng (Bonus)
[5] Đảm bảo bản chất Học Tăng Cường & Giải thích khoa học      : Tối đa 2.0 điểm
[6] Share box chat lịch sử phát triển                          : Tối đa 1.0 điểm
[7] Giao diện & quy tắc tương đồng file mẫu Roadster_4_Voi_DQN : Chuẩn hóa bắt buộc
========================================================================================
```

---

### 1.2. Tuyên ngôn Kiến trúc (Architectural Manifesto)

$$\textbf{CORRECTNESS} > \textbf{REPRODUCIBILITY} > \textbf{EXPLAINABILITY} > \textbf{PERFORMANCE} > \textbf{UI}$$

- **Correctness (Tính đúng đắn toán học & vật lý)**: Mô hình tuân thủ định luật vật lý Newton-Euler 2D, phương trình vi phân chuyển động vật rắn, phương trình Bellman Optimality, và bảo toàn năng lượng/động lượng.
- **Reproducibility (Tính tái lập)**: Mọi kết quả huấn luyện và kiểm thử đều được kiểm soát bằng Random Seed cố định (`Seed 42` train, `Seed 10000` validation, `Seed 20000` test). Tuyệt đối không có yếu tố ngẫu nhiên ngoài tầm kiểm soát.
- **Explainability (Khả năng giải thích minh bạch)**: Mọi khái niệm $s, a, r, \pi, Q(s, a)$, TD error, Replay Buffer, Target Network, Reward Shaping đều được hiển thị trực quan và có công thức đối chiếu, sinh viên tự tin trả lời vấn đáp trước hội đồng giảng viên.
- **Academic Integrity (Liêm chính học thuật)**:
  - **KHÔNG** hard-code hành động thành công.
  - **KHÔNG** dùng heuristic controller chạy ngầm rồi giả vờ gọi là DQN.
  - **KHÔNG** đổi sang policy giả lập khi kiểm tra test case khó.

---

## 2. CƠ SỞ TOÁN HỌC & MÔ HÌNH HÓA REINFORCEMENT LEARNING (MDP FORMULATION)

Hệ thống được mô hình hóa dưới dạng một tiến trình quyết định Markov (Markov Decision Process - MDP):

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

```
                              +--------------------+
                              |                    |
            +-----------------|     Môi trường     |<----------------+
            |                 |  (Physics Engine)  |                 |
            | s_{t+1}, r_t    |                    |                 |
            | done, info      +--------------------+                 |
            |                                                        | Hành động a_t
            v                                                        | (16 tổ hợp van)
     +-------------+                                                 |
     | Trạng thái  |                                                 |
     | 9 chiều     |                                                 |
     +-------------+                                                 |
            |                                                        |
            v                                                        |
    +---------------+         Q-Network: 9 -> 64 -> 64 -> 16         |
    | Chính sách pi | ===============================================+
    | (DQN / DDQN)  |   a_t = argmax_a Q(s_t, a; theta)   (Inference)
    +---------------+   a_t = epsilon-greedy              (Training)
```

---

### 2.1. Không gian trạng thái (State Space $\mathcal{S} \subset \mathbb{R}^9$)

Vector trạng thái quan sát gồm 9 biến liên tục, được chuẩn hóa theo thang tham chiếu để giữ gradient ổn định:

$$s = \left[ \frac{x}{10},\; \frac{h}{15},\; \frac{v_x}{10},\; \frac{v_y}{10},\; \sin(\theta),\; \cos(\theta),\; \frac{\omega}{5},\; \frac{\text{fuel}}{100},\; \frac{t}{20} \right] \in \mathbb{R}^9$$

| Thành phần | Đại lượng vật lý | Đơn vị | Hệ số chuẩn hóa | Ý nghĩa kỹ thuật |
| :--- | :--- | :--- | :--- | :--- |
| $s_0 = x / 10$ | Vị trí ngang trọng tâm | $\text{m}$ | $/10$ | Đo độ lệch so với tâm bãi đáp ($x=0$). Miền bay $[-20, 20]$. |
| $s_1 = h / 15$ | Độ cao trọng tâm | $\text{m}$ | $/15$ | Độ cao xe cách mặt đất ($y=0$). Mặt đất tiếp xúc khi bánh chạm $h \approx 0.72\text{ m}$. |
| $s_2 = v_x / 10$ | Vận tốc ngang thế giới | $\text{m/s}$ | $/10$ | Cần triệt tiêu về $|v_x| < 0.5\text{ m/s}$ khi chạm đất. |
| $s_3 = v_y / 10$ | Vận tốc đứng thế giới | $\text{m/s}$ | $/10$ | Cần hãm tốc từ rơi tự do về $|v_y| < 1.0\text{ m/s}$. |
| $s_4 = \sin(\theta)$ | Sin góc nghiêng thân | $\text{rad}$ | Không | Tránh điểm gián đoạn chu kỳ góc $\pm \pi$; liên tục trơn. |
| $s_5 = \cos(\theta)$ | Cos góc nghiêng thân | $\text{rad}$ | Không | Khi xe thăng bằng: $\theta = 0 \implies \sin\theta = 0, \cos\theta = 1$. |
| $s_6 = \omega / 5$ | Vận tốc góc | $\text{rad/s}$ | $/5$ | Tốc độ quay thân xe, cần triệt tiêu $|\omega| < 0.2\text{ rad/s}$. |
| $s_7 = \text{fuel} / 100$ | Khí nén còn lại | $\%$ | $/100$ | Bắt đầu ở $1.0$ ($100\%$), giảm khi mở van khí. |
| $s_8 = t / 20$ | Thời gian bay đã trôi qua | $\text{s}$ | $/20$ | Giúp mạng nhận thức cận kề timeout $20\text{s}$ và áp lực thời gian $< 5\text{s}$. |

---

### 2.2. Không gian hành động (Action Space $\mathcal{A}$)

Phương tiện có 4 vòi phun độc lập hoạt động ở chế độ nhị phân (Bật / Tắt: $u_i \in \{0, 1\}$).
Không gian hành động gồm $2^4 = 16$ hành động rời rạc: $\mathcal{A} = \{0, 1, \dots, 15\}$.

Giải mã hành động thành trạng thái van thông qua phép toán bitwise:

$$u_{V1} = a \ \& \ 1 \quad (\text{Bit 0: V1 nâng đuôi})$$
$$u_{V2} = (a \gg 1) \ \& \ 1 \quad (\text{Bit 1: V2 nâng đầu})$$
$$u_{V3} = (a \gg 2) \ \& \ 1 \quad (\text{Bit 2: V3 đẩy phải})$$
$$u_{V4} = (a \gg 3) \ \& \ 1 \quad (\text{Bit 3: V4 đẩy trái})$$

```
+--------+------+------+------+------+--------------------------+------------------------------+
| Action |  V1  |  V2  |  V3  |  V4  | Lực tổng hợp             | Mômen tổng hợp               |
| a      | Đuôi |  Đầu | Phải | Trái | F_net (body frame)       | Torque tau_net (z-axis)      |
+--------+------+------+------+------+--------------------------+------------------------------+
| 0      | 0    | 0    | 0    | 0    | [0, 0]                   | 0                            |
| 1      | 1    | 0    | 0    | 0    | [0, +9000 N]             | tau_V1 = x1 * F1 = -16200 Nm |
| 2      | 0    | 1    | 0    | 0    | [0, +9000 N]             | tau_V2 = x2 * F2 = +16200 Nm |
| 3      | 1    | 1    | 0    | 0    | [0, +18000 N] (Nâng thẳng)| tau = 0 (Cân bằng mômen)     |
| 4      | 0    | 0    | 1    | 0    | [+6000 N, 0]             | 0                            |
| 5      | 1    | 0    | 1    | 0    | [+6000 N, +9000 N]       | -16200 Nm                    |
| ...    | ...  | ...  | ...  | ...  | ...                      | ...                          |
| 15     | 1    | 1    | 1    | 1    | [0, +18000 N]            | 0                            |
+--------+------+------+------+------+--------------------------+------------------------------+
```

---

### 2.3. Động lực học chuyển động (Physics & Transition Dynamics $\mathcal{P}$)

Mô hình vật lý tính toán tại tần số $120\text{ Hz}$ ($\Delta t_{phys} = \frac{1}{120}\text{ s} \approx 0.00833\text{ s}$).
Mỗi bước ra quyết định của Agent kéo dài $\Delta t_{agent} = 0.1\text{ s}$ ($10\text{ Hz}$).
Do đó, mỗi transition $(s_t, a_t, r_t, s_{t+1})$ thực thi chính xác **12 chu kỳ tích phân vật lý con (sub-steps)**.

#### Thông số hình học và cơ học của xe:
- Khối lượng xe: $m = 1,200\text{ kg}$.
- Kích thước thân: Chiều dài $L = 4.4\text{ m}$, Chiều cao $H = 0.9\text{ m}$.
- Trọng tâm (COM): Đặt tại gốc tọa độ cục bộ $(0, 0)$. Chiều $+x$ hướng về đầu xe (sang phải), $+y$ hướng lên nóc xe.
- Gia tốc trọng trường: $\vec{g} = [0, -9.81]^T\text{ m/s}^2 \implies P = mg = 11,772\text{ N}$.
- Mômen quán tính vật rắn quanh trục $z$ đi qua trọng tâm:

$$I_{zz} = \frac{1}{12} m (L^2 + H^2) = \frac{1}{12} \times 1,200 \times (4.4^2 + 0.9^2) = 100 \times (19.36 + 0.81) = 2,017.0\text{ kg}\cdot\text{m}^2$$

#### Bố trí và lực của các vòi phun (Local Body Coordinates):
- **V1 (Nâng đuôi)**: Đặt tại $\vec{r}_1 = [-1.8, -0.45]^T\text{ m}$. Lực hướng $+y_{body}$: $\vec{F}_1^{body} = [0, 9000]^T\text{ N}$.
  Mômen quay: $\tau_1 = x_1 F_{1y} - y_1 F_{1x} = (-1.8) \times 9000 - 0 = -16,200\text{ N}\cdot\text{m}$ (Làm chúc đầu/chúi mũi xuống).
- **V2 (Nâng đầu)**: Đặt tại $\vec{r}_2 = [+1.8, -0.45]^T\text{ m}$. Lực hướng $+y_{body}$: $\vec{F}_2^{body} = [0, 9000]^T\text{ N}$.
  Mômen quay: $\tau_2 = x_2 F_{2y} - y_2 F_{2x} = (+1.8) \times 9000 - 0 = +16,200\text{ N}\cdot\text{m}$ (Làm ngẩng đầu xe lên).
- **V3 (Đẩy phải)**: Đặt tại $\vec{r}_3 = [-2.2, 0.0]^T\text{ m}$. Lực hướng $+x_{body}$: $\vec{F}_3^{body} = [6000, 0]^T\text{ N}$. Mômen quay: $\tau_3 = 0\text{ N}\cdot\text{m}$.
- **V4 (Đẩy trái)**: Đặt tại $\vec{r}_4 = [+2.2, 0.0]^T\text{ m}$. Lực hướng $-x_{body}$: $\vec{F}_4^{body} = [-6000, 0]^T\text{ N}$. Mômen quay: $\tau_4 = 0\text{ N}\cdot\text{m}$.

#### Biến đổi tọa độ & Tích phân chuyển động:
Ma trận quay từ hệ trục thân xe (Body) sang hệ trục thế giới (World):

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

Lực tổng hợp và mômen tổng hợp trong hệ trục thế giới:

$$\vec{F}_{net}^{world} = R(\theta) \sum_{i=1}^4 u_i \vec{F}_i^{body} + \begin{bmatrix} 0 \\ -mg \end{bmatrix}$$

$$\tau_{net} = \sum_{i=1}^4 u_i \tau_i$$

Tích phân vi phân số (Semi-implicit Euler Integrator):

$$\vec{a}_{world} = \frac{\vec{F}_{net}^{world}}{m}, \quad \alpha = \frac{\tau_{net}}{I_{zz}}$$
$$\vec{v}_{t+\delta t} = \vec{v}_t + \vec{a}_{world} \delta t, \quad \vec{p}_{t+\delta t} = \vec{p}_t + \vec{v}_{t+\delta t} \delta t$$
$$\omega_{t+\delta t} = \omega_t + \alpha \delta t, \quad \theta_{t+\delta t} = \theta_t + \omega_{t+\delta t} \delta t$$

#### Va chạm và tiếp đất (Ground Contact & Terminal Check):
Xe được xác định tiếp xúc mặt đất ($y=0$) thông qua 4 điểm góc (bánh xe / cản trước sau):

$$\vec{p}_{corner\_i}^{world} = \vec{p}_{COM} + R(\theta) \vec{r}_{corner\_i}^{body}$$

Khi $\min_i (y_{corner\_i}) \le 0$, va chạm mặt đất xảy ra:
- **Tiếp đất an toàn (Safe Landing)** nếu và chỉ nếu đồng thời thỏa mãn:
  1. $|v_y| < 1.0\text{ m/s}$
  2. $|v_x| < 0.5\text{ m/s}$
  3. $|\theta| < 5^\circ$ ($0.08726\text{ rad}$)
  4. $|\omega| < 0.2\text{ rad/s}$
- **Va chạm thất bại (Crash)**: Bất kỳ góc nào chạm đất mà vi phạm ít nhất 1 trong 4 điều kiện trên.
- **Vượt biên (Out of Bounds)**: $|x| > 20\text{ m}$ hoặc $h > 25\text{ m}$.
- **Hết giờ (Timeout)**: $t \ge 20.0\text{ s}$ (hoặc 200 bước quyết định).

---

### 2.4. Hàm phần thưởng (Reward Architecture)

Để đảm bảo giá trị phần thưởng vừa phản ánh đúng mục tiêu đào tạo, vừa hỗ trợ gradient ổn định trong mạng học sâu, hệ thống phân tách tường minh thành **Raw Reward** và **Shaped Learning Reward**.

#### A. Phần thưởng gốc ($r_{base}$):
$$r_{base} = r_{terminal} + r_{zone} - w_{crash} - w_{boundary} - w_{timeout} - r_{contact\_penalties} - r_{step\_costs}$$

Trong đó:
1. $r_{terminal} = +200.0$ nếu đạt Safe Landing.
2. $r_{zone} = +30.0$ nếu cả hai bánh xe tiếp đất nằm trọn trong vùng đáp xanh ($|x_{wheels}| \le 1.5\text{ m}$).
3. $w_{crash} = -100.0$ nếu chạm đất không an toàn (Quy tắc mẫu: luôn bằng $-\frac{1}{2}$ thưởng an toàn).
4. $w_{boundary} = -250.0$ nếu bay vượt biên không gian mô phỏng ($|x| > 20\text{ m}$ hoặc $h > 25\text{ m}$).
5. $w_{timeout} = -500.0$ nếu hết thời gian $20\text{ s}$ mà chưa đáp.
6. Hình phạt động năng va chạm (chỉ tính tại thời điểm tiếp đất):
   $$r_{impact} = -w_{v}(v_x^2 + v_y^2), \quad \text{với } w_v = 2.0$$
7. Hình phạt góc nghiêng va chạm (chỉ tính tại thời điểm tiếp đất):
   $$r_{angle} = -w_{\theta}(1 - \cos\theta_{contact}), \quad \text{với } w_{\theta} = 10.0$$
8. Chi phí tiêu hao khí nén:
   $$r_{gas} = -w_{gas} \times \Delta\text{fuel}, \quad \text{với } w_{gas} = 0.2 / \% \text{ khí}$$
9. Chi phí thời gian bay:
   $$r_{time} = -w_{time} \times \Delta t, \quad \text{với } w_{time} = 1.0 / \text{s} \implies -0.1 \text{ mỗi step } 0.1\text{s}$$

#### B. Định lý Potential-Based Reward Shaping (PBRS - Ng, Harada & Russell 1999):
Để giải quyết vấn đề reward cực kỳ thưa thớt (sparse reward) khi xe ở trên không mà không làm thay đổi chính sách tối ưu gốc $\pi^*$, ta sử dụng hàm thế năng:

$$F(s, s') = \gamma \Phi(s') - \Phi(s)$$

Hàm thế năng $\Phi(s)$ đo khoảng cách Euclidean tới bãi đáp và độ lớn vận tốc:

$$\Phi(s) = -10 \sqrt{x^2 + \max(0, h - 0.72)^2} - 4 \sqrt{v_x^2 + v_y^2}$$

Đặc tính bắt buộc của PBRS:
- Tại mọi trạng thái kết thúc (Terminal state: safe landing, crash, oob, timeout), ta quy ước: $\Phi(s_{terminal}) = 0$.
- Không đưa góc nghiêng $\theta$ vào $\Phi(s)$ để tránh bẫy local minima và để cho phép hành động xoay lộn vòng tự do trên cao.
- **Chứng minh lý thuyết bảo toàn chính sách tối ưu**:

$$Q_{\mathcal{M}'}^*(s, a) = Q_{\mathcal{M}}^*(s, a) - \Phi(s) \implies \arg\max_a Q_{\mathcal{M}'}^*(s, a) = \arg\max_a Q_{\mathcal{M}}^*(s, a)$$

#### C. Phần thưởng huấn luyện ($r_{learn}$):
$$r_{learn} = 0.01 \times (r_{base} + F(s, s'))$$

Hệ số $0.01$ đưa reward về khoảng $[-5.0, +2.5]$, giúp Gradient Huber/MSE không bị bùng nổ, hoàn toàn không sử dụng Sign Clipping (vì sign clipping sẽ triệt tiêu sự chênh lệch giữa va chạm $-100$ và hết giờ $-500$).

---

### 2.5. Thuật toán RL: Double DQN kết hợp Huber Loss

```
    Online Q-Network (theta)                           Target Q-Network (theta^-)
+-------------------------------+                  +-------------------------------+
| Input s' (9)                  |                  | Input s' (9)                  |
| -> FC(64) -> ReLU             |                  | -> FC(64) -> ReLU             |
| -> FC(64) -> ReLU             |                  | -> FC(64) -> ReLU             |
| -> Q(s', a; theta) (16 values)|                  | -> Q(s', a; theta^-) (16 vals)|
+-------------------------------+                  +-------------------------------+
                |                                                  |
                v                                                  v
     a* = argmax_a Q(s', a; theta)                    Q(s', a*; theta^-)
                |                                                  |
                +-----------------------> y_t = r + gamma * Q -----+
                                                    |
                                                    v
                                    Loss(Q(s, a; theta), y_t)
```

1. **Giảm thiểu Overestimation Bias**:
   DQN cổ điển dùng $\max_a Q(s', a; \theta^-)$ dẫn đến ước lượng phóng đại Q-value. Double DQN tách biệt việc chọn hành động và lượng giá hành động:

$$a^* = \arg\max_{a} Q(s', a; \theta_{online})$$
$$y_t = r_t + \gamma (1 - done_t) Q(s', a^*; \theta_{target})$$

2. **Hàm mất mát Huber Loss**:

$$\mathcal{L}_{TD}(\theta) = \frac{1}{|B|} \sum_{(s, a, r, s', d) \in B} \text{Huber}(y_t - Q(s, a; \theta)), \quad \text{với } \delta = y_t - Q(s, a; \theta)$$

$$\text{Huber}(\delta) = \begin{cases} \frac{1}{2} \delta^2 & \text{nếu } |\delta| \le 1.0 \\ |\delta| - \frac{1}{2} & \text{ngược lại} \end{cases}$$

---

### 2.6. Demonstration Learning (Học có hướng dẫn - Phân định rõ với DQfD gốc)

> [!IMPORTANT]
> **Yêu cầu liêm chính khoa học:** Bản thử nghiệm trước đây cho thấy DQN học từ đầu ($\epsilon$-greedy ngẫu nhiên) đạt tỷ lệ $0/100$ ca test vì xác suất ngẫu nhiên hạ cánh êm dịu là cực kỳ hiếm ($< 10^{-6}$).
> Giải pháp chuẩn mực là tích hợp **Demonstration Learning**. Tuy nhiên, để bảo đảm tính trung thực tuyệt đối:
> - **KHÔNG** gọi hệ thống là "DQfD hoàn chỉnh" nếu chưa dùng Prioritized Replay Buffer và n-step TD returns.
> - Gọi chính xác là: **"Guided Double DQN with Supervised Large Margin Loss"** (lấy cảm hứng từ Hester et al. 2017).

#### Cơ chế Supervised Large Margin Classification Loss:
Khi lấy mẫu transition từ tập chuyên gia mẫu (demonstration dataset), mạng bị phạt nếu dự đoán Q-value của các hành động khác lớn hơn hành động của chuyên gia $a_E$:

$$\mathcal{L}_E(\theta) = \max_{a \in \mathcal{A}} \left[ Q(s, a; \theta) + \ell(a_E, a) \right] - Q(s, a_E; \theta)$$

Trong đó hàm margin $\ell(a_E, a)$ quy định:

$$\ell(a_E, a) = \begin{cases} 0 & \text{nếu } a = a_E \\ m & \text{nếu } a \ne a_E \quad (m = 0.2) \end{cases}$$

Hàm mất mát tổng hợp giai đoạn huấn luyện hỗn hợp:

$$\mathcal{L}_{total}(\theta) = \mathcal{L}_{TD}(\theta) + \lambda_E \mathcal{L}_E(\theta) + \lambda_2 \|\theta\|_2^2$$

Với: $\lambda_E = 5.0$, $margin = 0.2$, $\lambda_2 = 10^{-5}$ (L2 regularization).
Tỷ lệ hòa trộn mẫu (Mixed Batch): Mỗi mini-batch 64 phần tử gồm $25\%$ mẫu từ tập demonstration và $75\%$ mẫu từ online replay buffer.

---

## 3. CẤU TRÚC THƯ MỤC CHUẨN HÓA (DIRECTORY STRUCTURE)

Dự án được phân tách module hóa tuyệt đối, đáp ứng tiêu chuẩn phần mềm chuyên nghiệp:

```
Đồ án Học tăng cường/
├── PROJECT_ARCHITECTURE.md              # [BẢNG KIẾN TRÚC GỐC - KHÔNG ĐƯỢC PHÁ VỠ]
├── README.md                            # Hướng dẫn chạy và giải thích đồ án
├── requirements.txt                     # Thư viện phụ thuộc Python
├── k5-htc-doan.pdf                      # Đề bài gốc của giảng viên
├── Roadster_4_Voi_DQN.docx              # Đặc tả tham chiếu từ file mẫu
│
├── configs/                             # Cấu hình độc lập (YAML/JSON)
│   ├── env_config.yaml                  # Kích thước xe, vòi phun, tần số 120Hz/10Hz
│   ├── reward_config.yaml               # Hệ số thưởng/phạt, shaping, scaling
│   ├── dqn_config.yaml                  # Siêu tham số DQN/DDQN, learning rate, Huber
│   └── curriculum_config.yaml           # Các mức độ huấn luyện Level 1 -> Level 3
│
├── core/                                # [MODULE CỐT LÕI - KHÔNG SỬA TRÁI PHÉP]
│   ├── __init__.py
│   ├── types.py                         # Dataclass, Type hints (State, Action, StepResult)
│   ├── physics_engine.py                # Động lực học vật rắn 2D, 120Hz integrator
│   ├── roadster_env.py                  # Gym-like Environment wrapper
│   └── reward_engine.py                 # Hàm tính toán Raw Reward & Potential Shaping
│
├── agents/                              # [CÁC MÔ HÌNH AGENT & BỘ ĐIỀU KHIỂN]
│   ├── __init__.py
│   ├── networks.py                      # Kiến trúc PyTorch 9 -> 64 -> 64 -> 16
│   ├── dqn_agent.py                     # Double DQN Agent, epsilon-greedy, target sync
│   ├── replay_buffer.py                 # Uniform Replay Buffer & Demonstration Buffer
│   └── teacher_controller.py            # Heuristic PD/State-Machine Teacher tạo demo
│
├── training/                            # [PIPELINE HUẤN LUYỆN & CURRICULUM]
│   ├── __init__.py
│   ├── losses.py                        # TD loss, Supervised Margin loss, Combined loss
│   ├── curriculum_manager.py            # Quản lý chuyển cấp độ L1 -> L2 -> L3
│   ├── demonstration_generator.py       # Thu thập 500 tập mẫu an toàn từ Teacher
│   └── train_pipeline.py                # Script huấn luyện chính (Học có hướng dẫn -> RL)
│
├── evaluation/                          # [KIỂM ĐỊNH ĐỘC LẬP & TÍNH ĐIỂM BAREM]
│   ├── __init__.py
│   ├── benchmark_runner.py              # Đánh giá 100 episodes cố định (Seed 20000)
│   ├── metrics_collector.py             # Tính tỷ lệ an toàn, thời gian đáp, nhiên liệu
│   └── score_calculator.py              # Chấm điểm tự động theo tiêu chí 10 điểm của đề
│
├── web/                                 # [GIAO DIỆN MÔ PHỎNG & DEMO TRÌNH DUYỆT]
│   ├── index.html                       # Giao diện Sandbox tương đồng Roadster_4_Voi_DQN
│   ├── css/
│   │   └── style.css                    # Giao diện tối hiện đại, radar, thông số
│   ├── js/
│   │   ├── physics.js                   # Physics engine ported sang JS (đồng nhất Python)
│   │   ├── neural_net.js                # Bộ suy diễn mạng nơ-ron thuần JS (tính Q-value)
│   │   ├── visualizer.js                # Canvas render xe, vệt khí, mũi tên lực, sơ đồ nơ-ron
│   │   └── app.js                       # Controller điều khiển tương tác người dùng
│   └── weights/
│       └── trained_weights.json         # File JSON chứa trọng số mạng đã huấn luyện
│
├── scripts/                             # [CÔNG CỤ TIỆN ÍCH]
│   ├── export_to_web.py                 # Chuyển đổi PyTorch checkpoint sang trained_weights.json
│   ├── run_all_tests.py                 # Chạy toàn bộ test suites
│   └── generate_defense_slides.py       # Tạo tài liệu/slide chuẩn bị vấn đáp
│
└── tests/                               # [BỘ KIỂM THỬ TỰ ĐỘNG - UNIT & INTEGRATION]
    ├── test_physics_conservation.py     # Kiểm tra bảo toàn động lượng & rơi tự do
    ├── test_reward_integrity.py         # Kiểm tra tính chất PBRS telescoping sum
    ├── test_action_decoding.py          # Kiểm tra 16 tổ hợp bitwise van
    ├── test_agent_shapes.py             # Kiểm tra kích thước tensor mạng 9->64->64->16
    └── test_reproducibility.py          # Kiểm tra Seed cố định cho cùng kết quả
```

---

## 4. BẢNG PHÂN QUYỀN TRÁCH NHIỆM & RANH GIỚI BẢO VỆ MÃ NGUỒN (CODE GOVERNANCE)

Để tránh tình trạng các agent hoặc lập trình viên can thiệp làm sai lệch bản chất RL, bảng phân quyền sau đây là **QUY TẮC BẤT DI BẤT DỊCH**:

```
+---------------------------------------------------------------------------------------+
| NHÓM TỆP NGHIÊM CẤM THAY ĐỔI (FROZEN FILES - DO NOT MODIFY)                           |
+---------------------------------------------------------------------------------------+
| 1. core/physics_engine.py   : Luật vật lý thế giới, khối lượng 1200kg, 120Hz, vị trí  |
|                               vòi phun và mômen không được sửa đổi để "dễ đáp hơn".  |
| 2. evaluation/benchmark.py  : 100 ca test cố định (Seed 20000) dùng để chấm điểm;    |
|                               tuyệt đối không hạ thấp tiêu chuẩn để tăng tỷ lệ đỗ.     |
| 3. evaluation/score_calc.py : Barem điểm theo quy định k5-htc-doan.pdf.               |
| 4. PROJECT_ARCHITECTURE.md  : Tài liệu định hướng kiến trúc gốc.                     |
+---------------------------------------------------------------------------------------+

+---------------------------------------------------------------------------------------+
| NHÓM TỆP ĐƯỢC PHÉP HIỆU CHỈNH & TỐI ƯU (MUTABLE / TUNABLE FILES)                      |
+---------------------------------------------------------------------------------------+
| 1. configs/*.yaml           : Siêu tham số lr, batch size, target update, weights.    |
| 2. training/*.py            : Chiến lược lấy mẫu, tốc độ suy giảm epsilon, số epochs. |
| 3. agents/teacher_*.py      : Cải tiến logic giáo viên để tạo tập demo mượt hơn.      |
| 4. web/*                    : Cải thiện hiệu ứng thị giác Canvas, biểu đồ, nút bấm.   |
+---------------------------------------------------------------------------------------+

+---------------------------------------------------------------------------------------+
| CÁC BIẾN GLOBAL & DESIGN PATTERNS CẦN TRÁNH TUYỆT ĐỐI                                  |
+---------------------------------------------------------------------------------------+
| [X] KHÔNG dùng Global State chia sẻ giữa Physics và Agent.                            |
| [X] KHÔNG hardcode mảng hành động mẫu (fixed trajectory replay) trong hàm act().     |
| [X] KHÔNG cho phép Agent đọc biến 'ground_y' hoặc 'target_x' ẩn ngoài vector state.   |
| [X] KHÔNG âm thầm chuyển từ DQN sang Heuristic Controller khi xe chuẩn bị chạm đất.  |
| [X] KHÔNG dùng Sign Clipping (r = sign(r)) làm mất độ phân giải điểm số.             |
+---------------------------------------------------------------------------------------+
```

---

## 5. ĐẶC TẢ CHI TIẾT CÁC PIPELINE HỆ THỐNG

### 5.1. Physics Pipeline (Động học & Va chạm)
```
[Agent Action a_t (0-15)] 
        |
        v
[Bitwise Decode: u_1, u_2, u_3, u_4]
        |
        v
+----------------- FOR step = 1 to 12 (120 Hz) -------------------+
|  1. Tính lực vòi phun: F_thrusters = R(theta) * sum(u_i * F_i)  |
|  2. Tính tổng lực: F_net = F_thrusters + [0, -mg]^T             |
|  3. Tính mômen tổng: tau_net = sum(u_i * tau_i)                 |
|  4. Gia tốc tuyến tính: a = F_net / m                           |
|  5. Gia tốc góc: alpha = tau_net / I_zz                         |
|  6. Semi-implicit Euler cập nhật: v, p, omega, theta            |
|  7. Tiêu hao khí: fuel = max(0, fuel - dFuel)                   |
|  8. Kiểm tra va chạm 4 góc xe với mặt đất y = 0                 |
|     -> Nếu va chạm: DỪNG SUBSTEPS NGAY LẬP TỨC                  |
+-----------------------------------------------------------------+
        |
        v
[Tổng hợp trạng thái mới s_{t+1}, kiểm tra terminal, tính reward]
```

---

### 5.2. Training Pipeline (Guided DQfD + Curriculum RL)
```
GIAI ĐOẠN 1: THU THẬP TRUYỀN DẠY (TEACHER DEMONSTRATION GENERATION)
  [Teacher Controller] -> Chạy 500 episodes an toàn (Seed 42)
                       -> Lọc chỉ lưu lượt đáp an toàn (15,000 - 20,000 transitions)
                       -> Lưu vào Demonstration Buffer D_demo

GIAI ĐOẠN 2: HUẤN LUYỆN CÓ HƯỚNG DẪN (PRE-TRAINING PHASE)
  Lặp 75,000 updates:
  - Lấy Batch 64 từ D_demo
  - Tính L_total = L_TD + 5.0 * L_margin + 1e-5 * L_reg
  - Cập nhật weights Q_online
  - Sao chép sang Q_target mỗi 500 updates
  - Đánh giá trên Validation Set (20 states cố định, Seed 10000) mỗi 5,000 updates

GIAI ĐOẠN 3: HỌC TĂNG CƯỜNG TỰ DO KẾT HỢP CURRICULUM (FINE-TUNING RL)
  Khởi tạo Online Buffer D_online (50,000 dung lượng)
  Hòa trộn Batch: 25% từ D_demo + 75% từ D_online
  Curriculum:
    - Level 1: h in [1.2, 3]m, |theta| <= 3 deg, |x| <= 0.2m (300 episodes)
    - Level 2: h in [2.0, 5]m, |theta| <= 8 deg, |x| <= 0.5m (500 episodes)
    - Level 3: h in [4.0, 9]m, |theta| <= 15 deg, |x| <= 2.0m (Full domain)
    - Level Flip: h in [7.0, 10]m, |theta| in [90, 180] deg (Chế độ nhào lộn)
```

---

### 5.3. Evaluation & Benchmark Pipeline (Độc lập 100 Episodes)
```
Khởi tạo môi trường với Seed = 20000 (Độc lập hoàn toàn với Seed 42 và 10000)
Tải trọng số mạng đã lưu (weights)
Tắt hoàn toàn thăm dò: epsilon = 0.0 (Chỉ dùng greedy action)

FOR episode = 1 to 100:
   Reset xe ở vị trí ngẫu nhiên quy định: h in [4, 9]m, x in [-2, 2]m, theta in [-15, +15] deg
   WHILE not done:
      s_norm = normalize(s)
      q_values = Q_network(s_norm)
      a = argmax(q_values)
      s, r, done, info = env.step(a)
   Ghi nhận:
      - Trạng thái kết thúc: Safe Landing / Crash / Timeout / Out of Bounds
      - Thời gian đáp t_land
      - Vận tốc tiếp đất |v_y|, |v_x|
      - Góc tiếp đất |theta|
      - Vị trí tiếp đất x_land (khoảng cách tới x=0)
      - Số vòng lộn n_flips (nếu có)

TỔNG HỢP BÁO CÁO:
   - Tỷ lệ đáp an toàn (%)
   - Tỷ lệ đáp đúng vùng mục tiêu (%)
   - Thời gian đáp trung bình (s) -> Kiểm tra điều kiện <= 5s
   - Điểm số dự kiến theo barem K5 (Thang điểm 10)
```

---

### 5.4. Frontend & Web Deployment Pipeline
```
[PyTorch Checkpoint (.pth)]
        |
        v
[export_to_web.py] -> Trích xuất W1, b1, W2, b2, W3, b3 thành JSON
        |
        v
[trained_weights.json]
        |
        v
[Web Browser: index.html]
  ├── physics.js       : Tính toán vật lý 120Hz đồng bộ chính xác với Python
  ├── neural_net.js    : Phép nhân ma trận W*x + b và hàm kích hoạt ReLU
  ├── visualizer.js    : Vẽ Canvas 2D:
  │                      - Thân xe Tesla Roadster, bánh xe, góc quay
  │                      - Vệt khói vòi phun V1, V2, V3, V4
  │                      - Mũi tên vector lực & mômen
  │                      - Sơ đồ mạng nơ-ron (64 nodes) hiển thị trực quan kích hoạt
  │                      - Biểu đồ thời gian thực: Q-values 16 actions, Reward, Loss
  └── app.js           : Các chế độ: Thủ công / Tự đáp (Teacher) / DQN / Rơi tự do
```

---

## 6. GIAO DIỆN LẬP TRÌNH CHUẨN GIỮA CÁC MODULE (INTERFACE SPECIFICATIONS)

Mọi module phải tuân thủ nghiêm ngặt các hợp đồng dữ liệu sau (sử dụng Python `dataclass` và `typing`):

### 6.1. Core Types (`core/types.py`)
```python
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

@dataclass(frozen=True)
class VehicleState:
    x: float          # Vị trí ngang (m)
    h: float          # Độ cao trọng tâm cách mặt đất (m)
    vx: float         # Vận tốc ngang (m/s)
    vy: float         # Vận tốc đứng (m/s)
    theta: float      # Góc nghiêng thân xe (rad, 0 = nằm ngang, dương = ngẩng đầu)
    omega: float      # Vận tốc góc (rad/s)
    fuel: float       # Phần trăm khí nén còn lại [0, 100]
    time: float       # Thời gian đã trôi qua trong episode (s)

@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray          # Vector quan sát chuẩn hóa 9 chiều
    reward: float            # Phần thưởng của bước này (r_learn)
    raw_reward: float        # Phần thưởng gốc không scaling (r_base)
    done: bool               # Cờ báo hiệu kết thúc episode
    truncated: bool          # Cờ báo hiệu hết giờ (timeout)
    info: dict               # Thông tin chi tiết (loại kết thúc, telemetry)
```

### 6.2. Abstract Environment Contract
```python
class BaseRoadsterEnv:
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Khởi tạo lại môi trường, trả về (obs, info)"""
        raise NotImplementedError

    def step(self, action: int) -> StepResult:
        """Thực thi action (0-15) qua 12 substeps vật lý 120Hz"""
        raise NotImplementedError

    def get_raw_state(self) -> VehicleState:
        """Lấy trạng thái vật lý thực chưa chuẩn hóa"""
        raise NotImplementedError
```

### 6.3. Abstract Agent Contract
```python
class BaseAgent:
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Chọn hành động rời rạc (0-15) dựa trên epsilon-greedy hoặc greedy"""
        raise NotImplementedError

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Trả về mảng 16 giá trị Q-value cho 16 hành động"""
        raise NotImplementedError

    def update(self, batch_data: dict) -> dict:
        """Thực hiện một bước cập nhật trọng số nơ-ron, trả về các loss logs"""
        raise NotImplementedError
```

---

## 7. CHIẾN LƯỢC ĐẠT ĐIỂM TỐI ĐA (10/10) VÀ GIẢI BÀI TOÁN FLIP/ROLL

### 7.1. Bắt buộc: Tối ưu thời gian tiếp đất $< 5.0\text{ s}$
Đề bài quy định: **"Thời gian đáp càng nhanh càng tốt (mỗi quá 5 giây trừ 4 điểm)"**.
Đây là tiêu chí sống còn: Nếu xe bay tà tà lượn mất $6\text{ giây}$, sinh viên sẽ bị trừ tới 8 điểm!

**Giải pháp kiến trúc**:
1. **Phạt thời gian động**: Trong hàm reward gốc, tăng trọng số thời gian $w_{time}$ khi thời gian vượt quá $3.5\text{ s}$.
2. **Chiến lược hãm tốc chủ động (Aggressive Descent & Flare)**:
   - Thay vì rơi rề rà, Agent được dạy kỹ thuật "Suicide Burn / Skyhook Landing" tương tự tên lửa SpaceX Falcon 9:
   - Giai đoạn 1 ($t \in [0, 2]\text{s}$): Giữ nguyên tốc độ rơi hoặc mở vòi phụ trợ đẩy nhanh về phía bãi đáp.
   - Giai đoạn 2 ($t \in [2, 4]\text{s}$): Kích hoạt toàn lực cặp vòi nâng $V_1 + V_2$ ($18,000\text{ N}$) để hãm vận tốc đứng từ $-10\text{ m/s}$ về $-0.8\text{ m/s}$ ngay trước khi chạm đất.
   - Thời gian đáp trung bình được định chuẩn: **$3.2\text{ s} - 4.5\text{ s}$** $\implies$ Đạt điểm tuyệt đối mục thời gian!

### 7.2. Điểm cộng: Kỹ năng lộn vòng ($n$ vòng nhào lộn $+n$ điểm thưởng)
Đề bài quy định: **"Nếu được thì lộn n vòng (1 vòng 1 điểm - không giới hạn điểm cộng)"**.

**Giải pháp kiến trúc**:
1. **Theo dõi góc quay liên tục (Accumulated Rotation Angle)**:
   Không dùng góc modulo $\theta \in [-\pi, \pi]$ để tính vòng lộn, mà theo dõi tích phân góc quay liên tục $\theta_{accumulated} = \int \omega dt$.
   Số vòng lộn hoàn chỉnh:
   $$N_{flips} = \left\lfloor \frac{|\theta_{accumulated}| + \epsilon_{margin}}{2\pi} \right\rfloor$$
2. **Thiết kế Reward nhào lộn (Acrobatic Mode)**:
   - Khi xe ở độ cao lớn ($h > 6\text{ m}$), cấp thêm điểm cộng tức thời $+50$ cho mỗi chu kỳ hoàn thành $2\pi$ góc quay.
   - **Điều kiện ràng buộc**: Chỉ được nhận điểm cộng lộn vòng nếu **kết thúc lượt bay là tiếp đất an toàn**! Nếu lộn 5 vòng mà cắm đầu nát xe thì vẫn nhận kết quả Crash thất bại (0 điểm an toàn).
   - Tách riêng 2 model checkpoints:
     - `model_speed_landing.pth`: Ưu tiên đáp cực nhanh ($< 4\text{s}$) và chuẩn xác.
     - `model_flip_trick.pth`: Thực hiện đúng 1 vòng lộn ($360^\circ$) trên cao rồi cân bằng đáp êm dưới $5\text{s}$.

---

## 8. HƯỚNG DẪN GIẢI THÍCH TRƯỚC GIẢNG VIÊN (EXPLAINABILITY & VIVA DEFENSE GUIDE)

Để sinh viên tự tin giành điểm tối đa phần vấn đáp (2 điểm phần "Đảm bảo là học tăng cường và giải thích được"), dưới đây là 8 câu hỏi cốt lõi mà giảng viên thường chất vấn cùng câu trả lời chuẩn mực kiến trúc:

### Câu 1: Tại sao bài toán này là Reinforcement Learning mà không phải Heuristic/PID Controller?
- **Trả lời**: Hệ thống điều khiển dựa trên việc tối đa hóa hàm giá trị kỳ vọng tích lũy $R_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$. Tại mỗi bước, mạng nơ-ron nhận vector trạng thái 9 chiều $s_t$ và xuất ra 16 giá trị $Q(s_t, a)$, sau đó chọn $a^* = \arg\max_a Q(s, a)$. Bộ điều khiển Heuristic chỉ đóng vai trò giáo viên cung cấp dữ liệu khởi đầu (imitation) để giải quyết vấn đề thăm dò mù (sparse reward exploration), toàn bộ quá trình ra quyết định khi demo và test là do mạng Q-network tự tính toán.

### Câu 2: Tại sao phải sử dụng Double DQN thay vì DQN cổ điển?
- **Trả lời**: Trong DQN gốc, toán tử max: $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$ sử dụng cùng một mạng target để vừa chọn hành động vừa ước lượng giá trị, dẫn đến hiện tượng thiên lệch phóng đại (Maximization Bias / Overestimation). Double DQN tách việc này ra: dùng mạng online $\theta$ chọn hành động tốt nhất $a^* = \arg\max Q(s', a; \theta)$, rồi dùng mạng target $\theta^-$ để lượng giá $Q(s', a^*; \theta^-)$. Điều này giúp giá trị Q-value tiệm cận giá trị thực của trạng thái, tránh việc mạng nơ-ron quá tự tin vào các trạng thái ảo.

### Câu 3: Reward Shaping có làm sai lệch nghiệm tối ưu của bài toán không?
- **Trả lời**: Không, vì hàm shaping của chúng ta tuân thủ nghiêm ngặt định lý của Andrew Ng (1999) về **Potential-Based Reward Shaping (PBRS)**: $F(s, s') = \gamma \Phi(s') - \Phi(s)$. Về mặt toán học, tổng phần thưởng tích lũy theo chuỗi sẽ tạo thành một chuỗi lồng nhau (telescoping sum):

$$\sum_{t=0}^T \gamma^t F(s_t, s_{t+1}) = \sum_{t=0}^T \gamma^t (\gamma \Phi(s_{t+1}) - \Phi(s_t)) = \gamma^{T+1} \Phi(s_{T+1}) - \Phi(s_0)$$

Vì $\Phi(s_{terminal}) = 0$, tổng phần thưởng chỉ phụ thuộc vào trạng thái xuất phát $s_0$, do đó thứ tự ưu tiên giữa các chính sách $\pi$ không đổi, nghiệm tối ưu $\pi^*$ được bảo toàn $100\%$.

### Câu 4: Tại sao trong State lại dùng $\sin(\theta)$ và $\cos(\theta)$ thay vì chỉ dùng một góc $\theta$?
- **Trả lời**: Góc quay $\theta$ có tính tuần hoàn qua $2\pi$ (hoặc $\pm 180^\circ$). Nếu dùng trực tiếp $\theta$, tại điểm ranh giới $+180^\circ$ và $-180^\circ$, giá trị nhảy vọt đột ngột từ $+\pi$ sang $-\pi$ dù thân xe thực tế chỉ lệch nhau một vi phân vô cùng nhỏ. Mạng nơ-ron là hàm xấp xỉ liên tục, cú nhảy gián đoạn này sẽ phá vỡ gradient. Biểu diễn qua cặp $[\sin\theta, \cos\theta]$ tạo ra một đường cong trơn tru, liên tục trên vòng tròn đơn vị $S^1$.

### Câu 5: Tại sao vật lý chạy ở 120Hz nhưng Agent lại ra quyết định ở 10Hz?
- **Trả lời**: Trong thực tế cơ điện tử, các van phun khí nén cần thời gian trễ mở van và phản ứng cơ học (khoảng vài chục miligiây), việc bật tắt van ở tần số quá cao (120Hz) là phi thực tế và gây rung giật hỗn loạn (chattering). Hơn nữa, ở góc độ RL, ra quyết định ở 10Hz ($\Delta t = 0.1\text{s}$) giúp một episode 20 giây chỉ kéo dài 200 bước, giảm chiều dài đường chân trời quy hoạch (planning horizon), giúp tín hiệu reward chiết khấu $\gamma^t$ lan truyền ngược hiệu quả hơn rất nhiều so với 2400 bước nếu quyết định ở 120Hz.

### Câu 6: Làm sao giảng viên biết được trên Web HTML đang chạy mạng nơ-ron thật chứ không phải mã if-else?
- **Trả lời**: Trong giao diện Web, có một bảng điều khiển mở rộng hiển thị trực tiếp 16 giá trị Q-value thời gian thực của 16 vòi phun. Giảng viên có thể click vào từng node mạng nơ-ron để kiểm tra vector trọng số $W, b$ được tải trực tiếp từ tệp `trained_weights.json`. Khi người dùng dùng chuột kéo lệch xe, các giá trị Q-value lập tức nhảy số theo mạng nơ-ron tương ứng.

---

## 9. CHIẾN LƯỢC KIỂM THỬ TỰ ĐỘNG & TIÊU CHÍ NGHIỆM THU (TEST STRATEGY & ACCEPTANCE CRITERIA)

### 9.1. Bộ kiểm thử tự động (Automated Test Suites)
Mỗi thành phần phải vượt qua các bài kiểm thử đơn vị độc lập trước khi tích hợp:
1. `tests/test_physics_conservation.py`:
   - Kiểm tra rơi tự do không van: $v_y(t) = -gt$, $y(t) = y_0 - \frac{1}{2}gt^2$. Sai số $< 10^{-4}$.
   - Kiểm tra lực nâng cực đại ($V_1 + V_2 = 18,000\text{ N}$): Gia tốc nâng lý thuyết $a_y = (18000 - 1200 \times 9.81) / 1200 = +5.19\text{ m/s}^2$.
   - Kiểm tra mômen cân bằng khi mở cả $V_1$ và $V_2$: Gia tốc góc $\alpha \approx 0\text{ rad/s}^2$.
2. `tests/test_reward_integrity.py`:
   - Xác minh tính chất Telescoping Sum của PBRS: Tổng $F$ trên một quỹ đạo bất kỳ từ $s_0$ đến $s_{terminal}$ phải bằng $-\Phi(s_0)$.
3. `tests/test_action_decoding.py`:
   - Xác minh đúng đắn 16 tổ hợp bitwise từ số nguyên $0 \dots 15$.
4. `tests/test_agent_shapes.py`:
   - Đầu vào shape $(B, 9) \to$ Lớp 1 $(B, 64) \to$ Lớp 2 $(B, 64) \to$ Đầu ra $(B, 16)$.

### 9.2. Tiêu chí nghiệm thu (Acceptance Criteria)
Một bản build được xem là hoàn tất khi và chỉ khi:
- [x] Chạy `evaluation/benchmark_runner.py` trên 100 episodes cố định (Seed 20000):
  - **Tỷ lệ đáp an toàn**: $\ge 80\%$ (vượt mốc 73% lịch sử của tài liệu tham chiếu).
  - **Tỷ lệ đúng vùng xanh ($x=0$)**: $\ge 70\%$.
  - **Thời gian tiếp đất trung bình**: $\le 4.5\text{ s}$ (đáp ứng tiêu chuẩn $< 5\text{s}$).
  - **Số ca hết giờ (Timeout)**: $0 / 100$.
  - **Số ca bay vượt biên**: $0 / 100$.
- [x] File `web/index.html` mở được trực tiếp trên mọi trình duyệt hiện đại (Chrome, Edge, Firefox), hiển thị mượt mà ở 60fps, tương tác kéo thả xe và chạy mạng nơ-ron trơn tru.
- [x] Không có bất kỳ cảnh báo lỗi bộ nhớ hay rò rỉ tensor trong quá trình huấn luyện và đánh giá.

---

## 10. PHÂN CÔNG NHIỆM VỤ CHI TIẾT CHO CÁC AGENT TIẾP THEO

| Giai đoạn | Agent phụ trách | Nhiệm vụ cụ thể | Sản phẩm bàn giao |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Core Physics & Env Agent** | Hiện thực hóa `core/physics_engine.py`, `core/roadster_env.py`, `core/reward_engine.py`, `core/types.py` và viết toàn bộ unit tests trong `tests/`. | Môi trường mô phỏng vật lý 2D 120Hz chạy ổn định, vượt qua 100% test vật lý. |
| **Phase 2** | **Agent & Algorithmic Agent** | Hiện thực hóa `agents/networks.py`, `agents/dqn_agent.py`, `agents/replay_buffer.py`, `agents/teacher_controller.py`, và `training/losses.py`. | Các class mạng nơ-ron, Double DQN, bộ nhớ đệm và Heuristic Teacher tạo được 500 bài học mẫu an toàn. |
| **Phase 3** | **Training & Evaluation Agent** | Hiện thực hóa `training/curriculum_manager.py`, `training/train_pipeline.py`, `evaluation/benchmark_runner.py`, và `scripts/export_to_web.py`. | Huấn luyện mạng thành công, đạt $\ge 80/100$ ca test an toàn, xuất ra `trained_weights.json`. |
| **Phase 4** | **Web UI & Visualization Agent** | Xây dựng giao diện web Canvas hoàn chỉnh trong `web/` (`index.html`, `physics.js`, `neural_net.js`, `visualizer.js`, `app.js`), tái hiện chuẩn xác file mẫu. | Ứng dụng web chạy độc lập, trực quan hóa Q-values và xe bay đẹp mắt, sẵn sàng trình chiếu demo. |

---
**Tài liệu được ban hành bởi LEAD PROJECT ARCHITECT - Đồ án Học Tăng Cường K5.**  
*Mọi module triển khai tiếp theo bắt buộc đối chiếu với tài liệu này trước khi commit mã nguồn.*
