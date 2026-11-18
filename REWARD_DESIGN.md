# THIẾT KẾ HỆ THỐNG PHẦN THƯỞNG (REWARD DESIGN SPECIFICATION)
## DỰ ÁN: XE ĐÁP ĐẤT ROADSTER 4 VÒI PHUN (REINFORCEMENT LEARNING)
### TÁC GIẢ: REINFORCEMENT LEARNING REWARD DESIGN SPECIALIST

---

## 1. TỔNG QUAN & NGUYÊN TẮC ƯU TIÊN (REWARD HIERARCHY)

Trong bài toán điều khiển tiếp đất hạ cánh cho xe bay Tesla Roadster bằng 4 vòi phun khí nén, hàm phần thưởng (Reward Function) đóng vai trò là "la bàn đạo đức" dẫn dắt Agent hội tụ về hành vi mong muốn.

Thiết kế phần thưởng tuân thủ nghiêm ngặt **Trật tự Ưu tiên 7 cấp bậc**:

$$\Large \textbf{SAFE LANDING} \gg \textbf{CORRECT POSITION} \gg \textbf{LOW IMPACT VELOCITY} \gg \textbf{LOW ANGULAR VELOCITY} \gg \textbf{LOW ANGLE} \gg \textbf{LOW FUEL} \gg \textbf{SHORT TIME}$$

```
========================================================================================
                          BẢNG PHÂN CẤP TRỌNG SỐ TOÁN HỌC
========================================================================================
1. SAFE LANDING (Tiếp đất an toàn)            : Delta = +300 (+200 thưởng / -100 va chạm)
2. CORRECT POSITION (Vùng đáp mục tiêu)       : +30 (Chỉ thưởng khi đã đáp an toàn)
3. LOW IMPACT VELOCITY (Vận tốc tiếp xúc nhỏ) : -2 * (vx^2 + vy^2) tại va chạm (0 .. -2.5)
4. LOW ANGLE (Góc thân xe thăng bằng ngang)   : -10 * (1 - cos(theta)) tại va chạm
5. LOW ANGULAR VELOCITY (Vận tốc góc nhỏ)     : Ràng buộc an toàn |omega| < 0.2 rad/s
6. LOW FUEL (Tiết kiệm khí nén)               : -0.2 / % tiêu hao (-0.0 .. -20.0)
7. SHORT TIME (Thời gian bay ngắn)            : -1.0 / giây (-0.1 mỗi bước 0.1s)
========================================================================================
```

---

## 2. PHÂN TÁCH TƯỜNG MINH: ORIGINAL REWARD VS SHAPED REWARD

Hệ thống phân lập tuyệt đối hai kênh phần thưởng độc lập:

```
                            +---------------------------------+
                            |       QUÁ TRÌNH MÔ PHỎNG        |
                            |  s_t  --[ a_t ]-->  s_{t+1}     |
                            +---------------------------------+
                                             |
                     +-----------------------+-----------------------+
                     |                                               |
                     v                                               v
       +----------------------------+                 +----------------------------+
       |   ORIGINAL REWARD (r_base) |                 |    SHAPED REWARD (r_learn) |
       |     (Phần thưởng gốc)      |                 |   (Phần thưởng huấn luyện) |
       +----------------------------+                 +----------------------------+
       | - Không định hình (No PBRS)|                 | - Tích hợp PBRS F(s, s')   |
       | - Không co giãn (No scale) |                 | - Co giãn scale = 0.01     |
       | - Dùng cho EVALUATION &    |                 | - Dùng cho Q-LEARNING      |
       |   BENCHMARK 100 EPISODES   |                 |   TRAINING & LOSS UPDATE   |
       +----------------------------+                 +----------------------------+
```

---

## 3. ĐẶC TẢ CHI TIẾT ORIGINAL REWARD ($r_{base}$)

Original Reward là thước đo khách quan đánh giá năng lực thực thi nhiệm vụ của Agent. Không chứa bất kỳ hệ số nhân tạo hay tín hiệu định hình nào:

$$r_{base} = r_{terminal} + r_{zone} + r_{impact\_vel} + r_{impact\_angle} + r_{gas} + r_{time}$$

### 3.1. Thành phần kết thúc lượt bay ($r_{terminal}$)
- **Tiếp đất an toàn (Safe Landing)**: $+200.0$
  Thỏa mãn đồng thời cả 4 điều kiện tại thời điểm chạm đất: $|v_y| \le 1.0\text{ m/s}$, $|v_x| \le 0.5\text{ m/s}$, $|\theta| \le 5^\circ$, $|\omega| \le 0.2\text{ rad/s}$.
- **Va chạm thất bại (Crash)**: $-100.0$
  Vi phạm ít nhất 1 trong 4 điều kiện an toàn khi chạm đất. Hệ số phạt luôn bằng $-\frac{1}{2}$ mức thưởng an toàn (theo chuẩn thiết kế `Roadster_4_Voi_DQN.docx`).
- **Bay vượt biên (Out of Bounds)**: $-250.0$
  Khi trọng tâm xe bay ra ngoài phạm vi $x \notin [-20, 20]\text{ m}$ hoặc $y > 25\text{ m}$.
- **Hết thời gian (Timeout)**: $-500.0$
  Khi thời gian bay chạm mốc tối đa $20.0\text{ s}$ (200 bước quyết định) mà chưa thể tiếp đất. Mức phạt cực nặng này triệt tiêu hoàn toàn xu hướng bay lòng vòng né tránh tiếp đất.

### 3.2. Thành phần thưởng bãi đáp mục tiêu ($r_{zone}$)
- **Thưởng bãi đáp màu xanh**: $+30.0$
  Được cộng thêm khi cả hai bánh xe tiếp đất nằm trọn trong vùng mục tiêu:
  $$-2.0\text{ m} \le x_{wheel\_rear} \le +2.0\text{ m} \quad \text{và} \quad -2.0\text{ m} \le x_{wheel\_front} \le +2.0\text{ m}$$
  *(Lưu ý: Chỉ được cộng khi trạng thái kết thúc là Safe Landing; nếu Crash trong vùng xanh thì KHÔNG được cộng).*

### 3.3. Hình phạt va chạm mặt đất ($r_{impact\_vel}, r_{impact\_angle}$)
> [!IMPORTANT]
> Hai khoản phạt này **CHỈ ĐƯỢC TÍNH DUY NHẤT TẠI THỜI ĐIỂM CHẠM ĐẤT** ($y_{lowest} \le 0$). Khi xe đang bay trên không, giá trị này bằng $0.0$, cho phép xe tự do cơ động và nhào lộn mà không bị phạt vận tốc hay góc nghiêng giữa chừng!

1. **Hình phạt động năng va chạm**:
   $$r_{impact\_vel} = -w_{v} \left( v_x^2 + v_y^2 \right), \quad \text{với } w_v = 2.0$$
   - Khi đáp cực êm ($v_x = 0.1, v_y = -0.3$): phạt $-2.0 \times (0.01 + 0.09) = -0.20$.
   - Khi va chạm mạnh ($v_x = 1.0, v_y = -3.0$): phạt $-2.0 \times (1.0 + 9.0) = -20.0$.
2. **Hình phạt góc nghiêng va chạm**:
   $$r_{impact\_angle} = -w_{\theta} \left( 1 - \cos\theta_{contact} \right), \quad \text{với } w_\theta = 10.0$$
   - Khi xe thăng bằng hoàn hảo ($\theta = 0$): $1 - \cos(0) = 0 \implies$ phạt $0.0$.
   - Khi xe nghiêng trong ngưỡng an toàn ($\theta = 4^\circ = 0.0698\text{ rad}$): phạt $-10 \times (1 - 0.9975) = -0.024$.
   - Khi xe lộn ngược cắm nóc xuống đất ($\theta = 180^\circ$): phạt $-10 \times (1 - (-1)) = -20.0$.

### 3.4. Chi phí vận hành từng bước ($r_{gas}, r_{time}$)
Tính liên tục tại mỗi bước quyết định $\Delta t = 0.1\text{ s}$:
1. **Chi phí tiêu hao khí nén**:
   $$r_{gas} = -w_{gas} \times \Delta\text{fuel}, \quad \text{với } w_{gas} = 0.2 / \% \text{ khí nén}$$
   Nếu dùng $1.0\%$ khí trong một bước, phạt $-0.2$. Tiết kiệm được khí nén giúp tăng tổng điểm.
2. **Chi phí thời gian trôi qua**:
   $$r_{time} = -w_{time} \times \Delta t, \quad \text{với } w_{time} = 1.0 / \text{giây} \implies -0.1 \text{ mỗi step } 0.1\text{s}$$
   Thúc đẩy Agent hạ cánh khẩn trương, thỏa mãn tiêu chí $< 5.0\text{ s}$ của đề thi.

---

## 4. ĐỊNH LÝ POTENTIAL-BASED REWARD SHAPING (PBRS) & CHỨNG MINH TOÁN HỌC

Trong môi trường không gian liên tục 16 hành động, phần thưởng tiếp đất an toàn $+200$ là cực kỳ thưa thớt (sparse reward). Thăm dò ngẫu nhiên $\epsilon$-greedy có xác suất hạ cánh an toàn $< 10^{-6}$, dẫn đến Agent không nhận được tín hiệu học.

Để giải quyết vấn đề này mà **không làm sai lệch chính sách tối ưu gốc**, ta áp dụng định lý kinh điển của **Andrew Ng, Harada & Russell (ICML 1999)**:

### 4.1. Công thức Shaping PBRS
$$F(s, s') = \gamma \Phi(s') - \Phi(s)$$

Với $\gamma = 0.995$ là hệ số chiết khấu tương lai. Hàm thế năng $\Phi(s)$ được thiết kế dựa trên khoảng cách Euclidean tới bãi đáp và độ lớn vận tốc:

$$\Phi(s) = -10.0 \sqrt{x^2 + \max(0, h - 0.72)^2} - 4.0 \sqrt{v_x^2 + v_y^2}$$

### 4.2. Điều kiện biên bắt buộc của PBRS
Tại mọi trạng thái kết thúc (Terminal states: Safe Landing, Crash, Timeout, OOB):

$$\Phi(s_{terminal}) \equiv 0.0$$

> [!NOTE]
> **Tại sao không đưa góc nghiêng $\theta$ vào $\Phi(s)$?**
> Nếu đưa góc $\theta$ vào hàm thế năng, mạng nơ-ron sẽ bị bẫy cục bộ (local minima): nó sẽ cố gắng giữ xe luôn nằm ngang ngay từ trên cao và không dám nghiêng thân để tạo lực đẩy ngang hoặc thực hiện kỹ năng nhào lộn xoay vòng (flip trick). Do đó, $\theta$ được giữ tự do trên không và chỉ bị kiểm tra nghiêm ngặt tại thời điểm tiếp đất.

### 4.3. Chứng minh toán học: Bảo toàn chính sách tối ưu (Policy Invariance)
Xét một quỹ đạo bất kỳ từ trạng thái ban đầu $s_0$ đến trạng thái kết thúc $s_T$:

$$\sum_{t=0}^T \gamma^t F(s_t, s_{t+1}) = \sum_{t=0}^T \gamma^t \left( \gamma \Phi(s_{t+1}) - \Phi(s_t) \right)$$

Khai triển chuỗi tổng:

$$\begin{aligned}
&= \left( \gamma \Phi(s_1) - \Phi(s_0) \right) + \gamma \left( \gamma \Phi(s_2) - \Phi(s_1) \right) + \dots + \gamma^T \left( \gamma \Phi(s_{T+1}) - \Phi(s_T) \right) \\
&= \gamma^{T+1} \Phi(s_{T+1}) - \Phi(s_0)
\end{aligned}$$

Vì $s_{T+1}$ là trạng thái kết thúc, $\Phi(s_{T+1}) = 0$, ta thu được:

$$\sum_{t=0}^T \gamma^t F(s_t, s_{t+1}) = -\Phi(s_0)$$

Hàm giá trị tối ưu của MDP đã định hình $\mathcal{M}'$:

$$Q_{\mathcal{M}'}^*(s, a) = Q_{\mathcal{M}}^*(s, a) - \Phi(s)$$

Vì $\Phi(s)$ độc lập với hành động $a$:

$$\arg\max_{a \in \mathcal{A}} Q_{\mathcal{M}'}^*(s, a) = \arg\max_{a \in \mathcal{A}} \left[ Q_{\mathcal{M}}^*(s, a) - \Phi(s) \right] = \arg\max_{a \in \mathcal{A}} Q_{\mathcal{M}}^*(s, a)$$

$$\implies \pi_{\mathcal{M}'}^* \equiv \pi_{\mathcal{M}}^* \quad (\textbf{Chính sách tối ưu không hề bị thay đổi!})$$

---

## 5. CƠ CHẾ PHÒNG CHỐNG GIAN LẬN PHẦN THƯỞNG (ANTI-REWARD-HACKING)

### 5.1. Bẫy "Treo lơ lửng trên không" (The Mid-Air Hovering Trap)
Nếu một kỹ sư thiết kế phần thưởng ngây thơ bằng cách cộng điểm trực tiếp khi xe gần đất:

$$r_{naive} = +\frac{10}{\text{khoảng cách}}$$

Agent sẽ phát hiện một lỗ hổng (exploit): **Bay tới cách mặt đất 10 cm, kích hoạt van giữ xe treo lơ lửng tại chỗ để thu thập điểm thưởng $+100$ mỗi giây vĩnh viễn!** Nếu xe đáp đất, episode sẽ dừng lại và mất nguồn thu điểm. Do đó, Agent sẽ từ chối tiếp đất!

### 5.2. Cách PBRS của chúng ta triệt tiêu hoàn toàn hành vi Hovering
Xét một Agent cố tình treo lơ lửng tại cùng một vị trí $s \to s$:

$$F(s, s) = \gamma \Phi(s) - \Phi(s) = (\gamma - 1) \Phi(s)$$

Vì $\gamma = 0.995 < 1$ và $\Phi(s) < 0$, giá trị $(\gamma - 1) \Phi(s)$ là một số dương cực nhỏ. Nhưng để treo xe lơ lửng ngược trọng lực $11,772\text{ N}$, Agent bắt buộc phải mở 2 vòi nâng $V_1 + V_2$:
- Tiêu hao khí nén: $\Delta\text{fuel} \approx 2.0 \% / \text{s} \implies r_{gas} = -0.4 / \text{s}$.
- Chi phí thời gian: $r_{time} = -1.0 / \text{s}$.
- Tổng chi phí mỗi giây: $-1.4 / \text{s} \ll (\gamma - 1)\Phi(s)$.

Do đó, **mỗi giây treo lơ lửng trên không đều khiến tổng điểm bị trừ liên tục**. Cách duy nhất để đạt điểm dương cực lớn là hoàn thành hạ cánh an toàn trước khi cạn khí và trước khi hết giờ!

---

## 6. THIẾT KẾ SHAPED REWARD DÙNG CHO DEEP Q-LEARNING ($r_{learn}$)

Nếu đưa trực tiếp $r_{base}$ (dao động từ $-500$ đến $+230$) vào mạng nơ-ron Deep Q-Learning:
- Hàm mất mát $\mathcal{L} = (y - Q)^2$ sẽ lên tới hàng trăm nghìn $(\delta^2 \approx 250,000)$, gây bùng nổ Gradient (Gradient Explosion) và phá vỡ trọng số mạng.
- Nếu dùng phương pháp của Atari: $r = \text{sign}(r) \in \{-1, 0, +1\}$, sự chênh lệch mang tính sống còn giữa va chạm ($-100$) và hết giờ ($-500$) sẽ bị san phẳng hoàn toàn.

**Giải pháp tối ưu chuẩn hóa**:

$$r_{learn} = 0.01 \times \left( r_{base} + F(s, s') \right)$$

Hệ số $0.01$ đưa phần thưởng huấn luyện về khoảng ổn định $[-5.0, +2.5]$, giúp Gradient của Huber Loss luôn nằm trong vùng tối ưu trơn tru mà không làm mất tương quan tỷ lệ giữa các mục tiêu.

---

## 7. BÁO CÁO KIỂM THỬ ĐƠN VỊ (UNIT TEST VERIFICATION)

Bộ kiểm thử [**`tests/test_reward.py`**](file:///F:/Programming%20Study/Reforcement%20Learning/%C4%90%E1%BB%93%20%C3%A1n%20H%E1%BB%8Dc%20t%C4%83ng%20c%C6%B0%E1%BB%9Dng/tests/test_reward.py) kiểm tra độc lập từng thành phần và đã vượt qua $100\%$ các bài test:

```bash
$ python -m pytest tests/test_reward.py -v
============================= test session starts =============================
tests/test_reward.py::test_component_safe_landing PASSED                 [  7%]
tests/test_reward.py::test_component_target_zone PASSED                  [ 15%]
tests/test_reward.py::test_component_crash_penalty PASSED                [ 23%]
tests/test_reward.py::test_component_timeout_penalty PASSED              [ 30%]
tests/test_reward.py::test_component_out_of_bounds_penalty PASSED        [ 38%]
tests/test_reward.py::test_component_ground_velocity_penalty PASSED      [ 46%]
tests/test_reward.py::test_component_ground_angle_penalty PASSED         [ 53%]
tests/test_reward.py::test_component_fuel_and_time_costs PASSED          [ 61%]
tests/test_reward.py::test_priority_hierarchy PASSED                     [ 69%]
tests/test_reward.py::test_potential_function_and_zero_terminal PASSED   [ 76%]
tests/test_reward.py::test_anti_hovering_reward_hacking PASSED           [ 84%]
tests/test_reward.py::test_telescoping_sum_invariance PASSED             [ 92%]
tests/test_reward.py::test_strict_reward_separation PASSED               [100%]
============================== 13 passed in 0.12s ==============================
```

---

## 8. NGUYÊN TẮC ÁP DỤNG TRONG THỰC TẾ (BENCHMARK PROTOCOL)

1. **Khi Huấn luyện (Training Mode)**:
   - Sử dụng `reward_bd.r_learn` làm reward $r_t$ trong Replay Buffer và tính toán TD Target $y = r + \gamma \max Q$.
2. **Khi Đánh giá độc lập (Evaluation / Benchmark 100 Episodes)**:
   - **BẮT BUỘC** chỉ sử dụng `info["raw_reward"]` ($r_{base}$) để tính toán điểm số tổng kết.
   - Tuyệt đối không dùng shaped reward để báo cáo kết quả trước hội đồng giảng viên.
