# BÁO CÁO KIỂM THỬ CHẤT LƯỢNG PHẦN MỀM (QA REPORT)
## DỰ ÁN HỌC TĂNG CƯỜNG: ĐIỀU KHIỂN HẠ CÁNH XE BAY TESLA ROADSTER 2D

---

**Chức danh phê duyệt:** Senior Software QA Engineer  
**Hệ thống mục tiêu:** Tesla Roadster 2D RL Landing System (Physics Engine, MDP Environment, RL Core, Evaluator, Web UI)  
**Tiêu chuẩn kiểm thử:** IEEE 829 / ISO 25010 / PyTest Test Suite  
**Ngày phát hành:** 18/11/2026  
**Trạng thái kiểm định:** **CHỨNG NHẬN ĐẠT CHUẨN (PASS - CERTIFIED PRODUCTION-READY)**

---

## 1. TỔNG QUAN VÀ KẾT QUẢ KIỂM ĐỊNH (EXECUTIVE SUMMARY)

Toàn bộ hệ sinh thái dự án đã được thực hiện kiểm thử tự động, kiểm thử hộp trắng (White-box testing), kiểm thử tích hợp (Integration testing), kiểm thử suy diễn số học (Numerical Parity testing) và phân tích mã nguồn tĩnh (Static code analysis).

### 1.1. Thống kê kiểm thử tự động (Automated Test Execution)
- **Tổng số test cases:** **117 / 117 tests**
- **Trạng thái:** **100% PASSED** (0 failed, 0 skipped, 0 xfailed)
- **Thời gian thực thi:** **5.93 giây** (PyTest v9.2.0, Python 3.12.10)
- **Bộ kiểm thử QA chuyên dụng ([tests/test_qa_suite.py](file:///F:/Programming%20Study/Reforcement%20Learning/%C4%90%E1%BB%93%20%C3%A1n%20H%E1%BB%8Dc%20t%C4%83ng%20c%C6%B0%E1%BB%9Dng/tests/test_qa_suite.py)):** **25 / 25 tests PASSED** bao phủ toàn diện 6 trụ cột cốt lõi.

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0
rootdir: F:\Programming Study\Reforcement Learning\Đồ án Học tăng cường
collected 117 items

tests\test_action.py .....                                               [  4%]
tests\test_collision.py .......                                          [ 10%]
tests\test_curriculum.py ...                                             [ 12%]
tests\test_curriculum_strategy.py ....                                   [ 16%]
tests\test_demonstration.py ....                                         [ 19%]
tests\test_demonstration_agent.py ....                                   [ 23%]
tests\test_double_dqn.py ..                                              [ 24%]
tests\test_dqfd.py ......                                                [ 29%]
tests\test_dqn.py ......                                                 [ 35%]
tests\test_env_reproducibility.py ..                                     [ 36%]
tests\test_experiments.py ...                                            [ 39%]
tests\test_fuel.py ..                                                    [ 41%]
tests\test_gravity.py ..                                                 [ 42%]
tests\test_independent_evaluator.py ...                                  [ 45%]
tests\test_qa_suite.py .........................                         [ 66%]
tests\test_reward.py .............                                       [ 77%]
tests\test_rotation.py ..                                                [ 79%]
tests\test_state.py ......                                               [ 84%]
tests\test_step_transition.py ....                                       [ 88%]
tests\test_terminal.py ....                                              [ 91%]
tests\test_thrust.py ...                                                 [ 94%]
tests\test_timestep.py ...                                               [ 96%]
tests\test_torque.py ....                                                [100%]

============================= 117 passed in 5.93s =============================
```

---

## 2. TRỤ CỘT 1: KIỂM TOÁN ĐỘNG LỰC HỌC VẬT LÝ (PHYSICS ENGINE AUDIT)

Hệ thống vật lý 2D được cài đặt tại [physics/world.py](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/physics/world.py), [physics/rigid_body.py](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/physics/rigid_body.py), [physics/thrusters.py](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/physics/thrusters.py) và [physics/collision.py](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/physics/collision.py).

### 2.1. Trọng trường và tích phân Symplectic Euler (Gravity & Kinematics)
- **Công thức tích phân:** Cập nhật vận tốc trước, vị trí sau (Bảo toàn năng lượng pha tốt hơn Euler hiện):
  $$\mathbf{v}_{t+1} = \mathbf{v}_t + \mathbf{a}_t \Delta t$$
  $$\mathbf{x}_{t+1} = \mathbf{x}_t + \mathbf{v}_{t+1} \Delta t$$
- **Kiểm định rơi tự do ($a_y = -g = -9.81\text{ m/s}^2$):**
  - Tại $t = 1.0\text{ s}$ (120 substeps): vận tốc đạt chính xác $v_y = -9.81\text{ m/s}$ (độ sai lệch $|v_y - (-9.81)| < 10^{-4}\text{ m/s}$).
  - Vị trí rơi tự do sai lệch lý thuyết $< 0.1\text{ m}$ do đặc thù sai số rời rạc bậc 1 của Symplectic Euler.
- **Trạng thái:** **PASS** (`test_physics_gravity_free_fall_kinematics`).

### 2.2. Lực đẩy vòi phun và quay vector lực (Thrust & Rotation)
- **Lực đẩy định mức:**
  - Vòi nâng V1, V2: $9,000\text{ N}$ mỗi vòi $\rightarrow$ Tổng lực nâng thẳng đứng: $18,000\text{ N} > 11,772\text{ N}$ (trọng lực $mg$).
  - Gia tốc thẳng đứng khi kích hoạt V1+V2: $a_y = \frac{18000 - 11772}{1200} = +5.19\text{ m/s}^2$ (Chính xác đến $10^{-4}\text{ m/s}^2$).
  - Vòi đẩy ngang V3, V4: $6,000\text{ N}$ mỗi vòi $\rightarrow a_x = \pm \frac{6000}{1200} = \pm 5.0\text{ m/s}^2$.
- **Vector quay lực theo góc nghiêng $\theta$:**
  - Tại góc nghiêng $\theta = 30^\circ$ ($\pi/6\text{ rad}$), lực nâng V1+V2 sinh thành phần gia tốc ngang:
    $$a_x = \frac{-2 F_{\text{lift}} \sin\theta}{m} = \frac{-18000 \times 0.5}{1200} = -7.5\text{ m/s}^2$$
  - Đo kiểm thực tế trong engine: $a_x = -7.5000\text{ m/s}^2$ (Khớp hoàn toàn).
- **Trạng thái:** **PASS** (`test_physics_thrust_force_magnitudes_and_rotation`).

### 2.3. Momen quay và gia tốc góc (Torque & Moment of Inertia)
- **Momen quán tính:** $I = 2,017\text{ kg}\cdot\text{m}^2$.
- **Cánh tay đòn vòi nâng:** $r_x = \pm 1.8\text{ m}$.
  - V1 (Đuôi xe, $r_x = -1.8\text{ m}$): Momen quay $\tau = -1.8 \times 9000 = -16,200\text{ N}\cdot\text{m}$ (Thuận chiều kim đồng hồ / chúi mũi).
  - V2 (Đầu xe, $r_x = +1.8\text{ m}$): Momen quay $\tau = +1.8 \times 9000 = +16,200\text{ N}\cdot\text{m}$ (Ngược chiều kim đồng hồ / ngóc mũi).
  - Khi kích hoạt cả V1 + V2 (Action 3): Momen tổng triệt tiêu hoàn toàn $\tau_{\text{net}} \equiv 0.0\text{ N}\cdot\text{m}$ ($\alpha = 0.0\text{ rad/s}^2$).
  - Vòi ngang V3, V4 đặt trên trục đi qua trọng tâm ($r_y = 0.0\text{ m}$): Momen quay bằng 0 tuyệt đối.
- **Trạng thái:** **PASS** (`test_physics_torque_and_angular_acceleration`).

### 2.4. Phát hiện va chạm 6 điểm hình học (Collision Detection)
- **6 điểm kiểm tra tiếp xúc:** 2 bánh xe ($x = \pm 1.5\text{ m}, y = -0.72\text{ m}$) và 4 góc thân xe.
- **Tiêu chí hạ cánh an toàn chuẩn file tham chiếu:**
  - $|v_y| \le 1.0\text{ m/s}$
  - $|v_x| \le 0.5\text{ m/s}$
  - $|\theta| \le 5^\circ$ ($0.08727\text{ rad}$)
  - $|\omega| \le 0.2\text{ rad/s}$
  - Cả 2 bánh trong vùng đáp xanh $[-2.0, +2.0]\text{ m}$.
- **Kiểm định các kịch bản va chạm:**
  - Vận tốc đáp $0.5\text{ m/s}$, góc $0.02\text{ rad}$ $\rightarrow$ `is_safe_landing = True, is_crash = False`.
  - Vận tốc rơi $|v_y| = 1.8\text{ m/s} > 1.0\text{ m/s}$ $\rightarrow$ `is_safe_landing = False, is_crash = True`.
  - Góc nghiêng tiếp đất $8.0^\circ > 5.0^\circ$ $\rightarrow$ `is_safe_landing = False, is_crash = True`.
  - Xe bị lật ngược nóc tiếp đất ($\theta = 180^\circ$) $\rightarrow$ `is_safe_landing = False, is_crash = True`.
- **Trạng thái:** **PASS** (`test_physics_collision_detection_and_rules`).

---

## 3. TRỤ CỘT 2: KIỂM TOÁN MÔI TRƯỜNG TOÁN HỌC MDP (ENVIRONMENT MDP AUDIT)

Môi trường cài đặt tại [environment/roadster_env.py](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/environment/roadster_env.py), [environment/state.py](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/environment/state.py), và [environment/action.py](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/environment/action.py).

### 3.1. Vector trạng thái 9 chiều (State Normalization)
- **Định dạng:** Vector thực liên tục 9 chiều $s \in \mathbb{R}^9$, dtype `np.float32`:
  $$s = \left[\frac{x}{10}, \frac{h}{15}, \frac{v_x}{10}, \frac{v_y}{10}, \sin(\theta), \cos(\theta), \frac{\omega}{5}, \frac{\text{fuel}}{100}, \frac{t}{20}\right]$$
- **Kiểm định đẳng thức lượng giác:** Đảm bảo $\sin^2\theta + \cos^2\theta = 1.0 \pm 10^{-5}$, loại bỏ tính gián đoạn (angle wrapping) của góc $\theta \in [-\pi, \pi]$.
- **Trạng thái:** **PASS** (`test_env_state_shape_and_continuous_bounds`).

### 3.2. Không gian hành động rời rạc 16 tổ hợp (Action Bijection)
- **Bảng ánh xạ:** 16 hành động rời rạc tương ứng với trạng thái bật/tắt của 4 vòi phun $(u_1, u_2, u_3, u_4) \in \{0, 1\}^4$:
  $$\text{Action} = u_1 + 2 u_2 + 4 u_3 + 8 u_4$$
- **Kiểm định ánh xạ 2 chiều (Bijection):** Kiểm tra toàn bộ 16 hành động từ $a = 0$ đến $15$, đảm bảo `encode_action(decode_action(a)) == a`.
- **Trạng thái:** **PASS** (`test_env_action_mapping_bidirectional`).

### 3.3. Các điều kiện kết thúc tập (Terminal Conditions)
- **Kiểm định 4 trạng thái dừng độc lập:**
  1. `SAFE_LANDING`: Chạm đất thỏa mãn cả 4 điều kiện an toàn $\rightarrow$ `terminated = True`.
  2. `CRASH`: Chạm đất vượt ngưỡng vận tốc hoặc góc nghiêng $\rightarrow$ `terminated = True`.
  3. `OUT_OF_BOUNDS`: Trọng tâm xe vượt ra ngoài phạm vi $[-20, 20]\text{ m}$ hoặc $y > 25\text{ m}$ $\rightarrow$ `terminated = True`.
  4. `TIMEOUT`: Thời gian bay đạt ngưỡng $20.0\text{ s}$ (200 bước quyết định) $\rightarrow$ `truncated = True`.
- **Trạng thái:** **PASS** (`test_env_terminal_conditions_and_timeout`).

### 3.4. Chu kỳ thời gian và chia bước phụ (Timestep & Substepping)
- **Tần số quyết định (Decision frequency):** $10\text{ Hz}$ ($\Delta t_{\text{decision}} = 0.1\text{ s}$).
- **Tần số vật lý (Physics frequency):** $120\text{ Hz}$ ($\Delta t_{\text{physics}} = 1/120 \approx 0.008333\text{ s}$).
- **Kiểm định số lượng bước phụ:** Đúng chính xác $12$ substeps trên mỗi decision step ($12 \times \frac{1}{120}\text{ s} = 0.1000\text{ s}$).
- **Trạng thái:** **PASS** (`test_env_timestep_and_substepping_count`).

---

## 4. TRỤ CỘT 3: KIỂM TOÁN THUẬT TOÁN HỌC TĂNG CƯỜNG (RL CORE AUDIT)

Hệ thống RL cài đặt tại [rl/dqn/](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/rl/dqn/), [rl/double_dqn/](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/rl/double_dqn/), và [rl/demonstration_agent.py](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/rl/demonstration_agent.py).

### 4.1. Công thức Bellman TD Target của DQN Chuẩn
- **Công thức:**
  $$y = r + \gamma (1 - \text{done}) \max_{a'} Q_{\text{target}}(s', a')$$
- **Kiểm tra cờ `done`:** Khi `done = 1.0`, toàn bộ phần tương lai $\gamma \max_{a'} Q_{\text{target}}(s', a')$ bị triệt tiêu về $0.0$, chỉ giữ lại phần thưởng tức thời $r$.
- **Trạng thái:** **PASS** (`test_rl_q_target_standard_dqn_formula`).

### 4.2. Khử Overestimation bằng Double DQN (Decoupling)
- **Công thức Double DQN (van Hasselt et al., 2015):**
  $$a^* = \arg\max_a Q_{\text{online}}(s', a)$$
  $$y = r + \gamma (1 - \text{done}) Q_{\text{target}}(s', a^*)$$
- **Kiểm định tách biệt (Decoupling):** Khi cố tình làm lệch trọng số giữa $Q_{\text{online}}$ và $Q_{\text{target}}$, thuật toán chọn hành động bằng mạng online và tính giá trị bằng mạng target. Kiểm thử xác nhận giá trị tính toán khớp chính xác với công thức độc lập với dung sai sai số $< 10^{-5}$.
- **Trạng thái:** **PASS** (`test_rl_q_target_double_dqn_decoupling`).

### 4.3. Replay Buffer dạng vòng (Circular Ring Buffer)
- **Kiểm tra tràn dung lượng:** Khi nạp 150 mẫu vào buffer dung lượng 100, con trỏ vòng đè lên đúng 50 mẫu cũ nhất (mẫu từ 0 đến 49 bị xóa, mẫu cũ nhất còn lại có giá trị 50).
- **Kiểm tra lấy mẫu Tensor:** Lấy mẫu ngẫu nhiên batch size 32 trích xuất đúng kích thước batch shape `(32, 9)`, `(32,)`, `(32,)`, `(32, 9)`, `(32,)`.
- **Trạng thái:** **PASS** (`test_rl_replay_buffer_ring_capacity_and_sampling`).

### 4.4. Lịch trình suy giảm Epsilon và chế độ Greedy tuyệt đối
- **Kiểm tra suy giảm tuyến tính:** Bắt đầu tại $\varepsilon = 1.0$, đạt chính xác $0.525$ tại nửa chu kỳ suy giảm, và kẹp cứng tại $\varepsilon_{\text{min}} = 0.05$ khi vượt ngưỡng decay steps.
- **Kiểm tra cờ `evaluate=True`:** Khi gọi `select_action(state, evaluate=True)`, agent luôn thực hiện chính xác hành động có Q-value lớn nhất ($\arg\max_a Q(s, a)$), không có bất kỳ hành động ngẫu nhiên nào xảy ra.
- **Trạng thái:** **PASS** (`test_rl_epsilon_decay_schedule_and_greedy_evaluation`).

### 4.5. Cô lập Target Network và Cập nhật định kỳ (Hard Update)
- **Kiểm tra cô lập gradient:** Khi huấn luyện `train_step()` trên mạng online, trọng số mạng target giữ nguyên không đổi từng bit ($100\%$ bitwise identical).
- **Kiểm tra đồng bộ:** Sau khi gọi `update_target_network()`, toàn bộ trọng số mạng target sao chép chính xác từ mạng online.
- **Trạng thái:** **PASS** (`test_rl_target_network_parameter_isolation`).

### 4.6. Lưu trữ và phục hồi Checkpoint (Weight Fidelity)
- **Kiểm tra lưu/nạp mô hình:** Checkpoint lưu trữ đầy đủ `online_net_state_dict`, `target_net_state_dict`, `optimizer_state_dict`, `total_env_steps`, `total_updates`.
- **Độ chính xác số học:** Trọng số nạp lại khớp hoàn toàn $100\%$ bitwise với mô hình ban đầu.
- **Trạng thái:** **PASS** (`test_rl_checkpoint_persistence_and_weight_fidelity`).

---

## 5. TRỤ CỘT 4: HỆ THỐNG PHẦN THƯỞNG VÀ ĐỊNH LÝ PBRS (REWARD ENGINE AUDIT)

Hệ thống tính thưởng cài đặt tại [environment/reward.py](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/environment/reward.py).

### 5.1. Thưởng/Phạt vật lý thô (Raw Unshaped Reward)
- **Thưởng thành công:**
  - Hạ cánh an toàn: $+200.0$.
  - Tiếp đất đúng vùng xanh: $+30.0$.
- **Phạt thất bại:**
  - Va chạm / Rơi vỡ (Crash): $-100.0$ (Bằng đúng 1/2 thưởng hạ cánh theo yêu cầu đặc tả).
  - Bay ra ngoài biên (Out of bounds): $-250.0$.
  - Hết giờ (Timeout): $-500.0$.
- **Phạt trạng thái tiếp xúc đất:**
  - Phạt vận tốc tiếp đất: $-2.0 \times (v_x^2 + v_y^2)$.
  - Phạt góc lệch tiếp đất: $-10.0 \times (1 - \cos\theta)$.
- **Chi phí vận hành liên tục:**
  - Phạt tiêu hao nhiên liệu: $-0.2 / \%$ khí nén.
  - Phạt thời gian bay: $-1.0 / \text{giây}$ ($0.1\text{ s}$ phạt $-0.1$).
- **Trạng thái:** **PASS** (`test_reward_safe_landing_and_target_zone_bonuses`, `test_reward_penalties_crash_timeout_oob_fuel_time`).

### 5.2. Định lý Định hình Phần thưởng Dựa trên Thế năng PBRS (Ng et al., 1999)
- **Hàm thế năng:**
  $$\Phi(s) = -10 \sqrt{x^2 + \max(0, h - 0.72)^2} - 4 \sqrt{v_x^2 + v_y^2}$$
- **Phần thưởng định hình:**
  $$F(s, s') = \gamma \Phi(s') - \Phi(s)$$
- **Điều kiện bất biến chính sách (Policy Invariance Guarantee):**
  - Tại **bất kỳ** trạng thái kết thúc nào (safe landing, crash, oob, timeout), hàm thế năng **phải bằng 0 tuyệt đối**:
    $$\Phi(s_{\text{terminal}}) \equiv 0.0$$
  - Tổng chuỗi kính thiên văn (Telescoping sum) qua một tập bất kỳ triệt tiêu hoàn toàn các thế năng trung gian:
    $$\sum_{t=0}^{T-1} \gamma^t F(s_t, s_{t+1}) = \gamma^T \Phi(s_T) - \Phi(s_0) = -\Phi(s_0)$$
  - Điều này đảm bảo thuật toán không bị "hack reward" (không lặp vòng tròn trên không để nhận thưởng thế năng dương liên tục).
- **Trạng thái:** **PASS** (`test_reward_pbrs_potential_zero_terminal_and_telescoping`).

### 5.3. Tách biệt Reward Huấn luyện và Reward Đánh giá
- Môi trường cung cấp rõ ràng:
  - `r_learn = 0.01 * (r_base + F)` phục vụ việc tối ưu hóa mạng nơ-ron (tránh bùng nổ gradient).
  - `r_base`: Phần thưởng vật lý gốc không định hình, phục vụ báo cáo và so sánh công bằng tuyệt đối giữa các thuật toán.

---

## 6. TRỤ CỘT 5: ĐÁNH GIÁ ĐỘC LẬP VÀ PHÂN LOẠI THẤT BẠI (EVALUATION AUDIT)

Hệ thống đánh giá cài đặt tại [evaluation/evaluator.py](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/evaluation/evaluator.py) và [evaluation/test_cases.py](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/evaluation/test_cases.py).

### 6.1. Độc lập hoàn toàn với quá trình huấn luyện
- **Không cập nhật mạng:** Trong quá trình đánh giá, cờ gradient bị tắt (`torch.no_grad()`), không gọi `optimizer.step()`, không thêm mẫu vào `ReplayBuffer`. Kiểm thử xác nhận trọng số mô hình trước và sau 3 tập đánh giá giữ nguyên $100\%$ từng bit và gradient là `None`.
- **Epsilon tuyệt đối bằng 0:** Khẳng định chính sách hành động $100\%$ xác định (deterministic greedy).
- **Tập kiểm thử độc lập (Test Set Independence):**
  - 60 ca kiểm thử biên có cấu trúc (Curated test suite) phân bố khắp không gian trạng thái.
  - 100 hạt giống ngẫu nhiên chưa từng thấy (Stochastic unseen seeds: $30000..30099$).
  - Giao tập hạt giống kiểm thử với tập hạt giống huấn luyện $\{42, 123, 999\}$ là tập rỗng ($\emptyset$). Không có hiện tượng rò rỉ dữ liệu (Data Leakage).
- **Trạng thái:** **PASS** (`test_eval_zero_gradients_and_zero_weight_updates`, `test_eval_epsilon_strictly_zero_deterministic`, `test_eval_independent_test_sets_disjoint_from_training`).

### 6.2. Phân loại thất bại 7 chiều (Failure Taxonomy)
Đánh giá không chỉ ghi nhận Reward mà phân loại nguyên nhân thất bại chi tiết:
1. `COLLISION`: Va chạm mặt đất do vượt tốc độ.
2. `UPSIDE_DOWN`: Xe bị lật úp nóc xe tiếp đất.
3. `OVERSHOOT`: Bay vọt qua bên phải vùng đáp xanh.
4. `UNDERSHOOT`: Rơi thiếu hụt bên trái vùng đáp xanh.
5. `TOO_FAST`: Tốc độ hạ cánh vượt ngưỡng $1.0\text{ m/s}$.
6. `TOO_MUCH_ROTATION`: Góc nghiêng tiếp đất vượt $5^\circ$.
7. `TIMEOUT`: Quá $20.0\text{ s}$ chưa hạ cánh.

---

## 7. TRỤ CỘT 6: TÍNH TOÀN VẸN CỦA GIAO DIỆN WEB UI (UI INTEGRITY AUDIT)

Giao diện Web cài đặt tại [web/index.html](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/web/index.html), [web/js/visualizer.js](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/web/js/visualizer.js), [web/js/neural_net.js](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/web/js/neural_net.js), và [web/js/demo_guide.js](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/web/js/demo_guide.js).

### 7.1. Đối chiếu trọng số PyTorch và JavaScript (Weight Parity)
Kiểm tra đối sánh độ lệch số học tuyệt đối giữa mô hình PyTorch gốc (.pth) và trọng số JSON xuất sang Web (.json) trên 10 vector trạng thái ngẫu nhiên:
- **Baseline DQN ([checkpoints/experiments/exp_a_dqn_seed_42.pth](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/checkpoints/experiments/exp_a_dqn_seed_42.pth) vs [web/weights/dqn_weights.json](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/web/weights/dqn_weights.json)):**
  - Độ lệch Q-value lớn nhất: **$6.93 \times 10^{-7} < 10^{-6}$**.
  - Hành động dự đoán ($\arg\max$): **Khớp 100%**.
- **Double DQN ([checkpoints/experiments/exp_b_double_dqn_seed_42.pth](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/checkpoints/experiments/exp_b_double_dqn_seed_42.pth) vs [web/weights/ddqn_weights.json](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/web/weights/ddqn_weights.json)):**
  - Độ lệch Q-value lớn nhất: **$3.73 \times 10^{-7} < 10^{-6}$**.
  - Hành động dự đoán ($\arg\max$): **Khớp 100%**.
- **Trained Best Model ([checkpoints/best_model.pth](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/checkpoints/best_model.pth) vs [web/weights/trained_weights.json](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/web/weights/trained_weights.json)):**
  - Độ lệch Q-value lớn nhất: **$1.09 \times 10^{-5} < 2 \times 10^{-5}$** (Phù hợp hoàn hảo với sai số làm tròn chuẩn Float32).
  - Hành động dự đoán ($\arg\max$): **Khớp 100%**.
- **Trạng thái:** **PASS** (`test_ui_dqn_weights_parity_with_pytorch_baseline`, `test_ui_double_dqn_weights_parity_with_pytorch_baseline`, `test_ui_trained_model_weights_parity_with_best_checkpoint`).

### 7.2. Kiểm định tính dị biệt của các mô hình (Models are Distinct)
- Q-values và ma trận trọng số của Standard DQN, Double DQN và Trained Model hoàn toàn khác biệt nhau ($\Delta W > 10^{-2}$).
- Trên cùng một trạng thái kiểm thử, các mạng đưa ra các hành động khác nhau: Action 15 (Standard DQN), Action 11 (Double DQN), Action 9/3 (Trained Model).
- Điều này chứng minh UI không dùng một mô hình giả lập chung cho tất cả các lựa chọn.
- **Trạng thái:** **PASS** (`test_ui_model_predictions_are_distinct`).

### 7.3. Kiểm toán chống gian lận mã nguồn (Anti-Cheat / No Hidden Controller)
- **Phân tích tĩnh mã nguồn [web/js/visualizer.js](file:///F:/Programming%20Study/Reforcement Learning/Đồ án Học tăng cường/web/js/visualizer.js):**
  - Trong khối thực thi `if (this.mode === "rl")`:
    - Chỉ có duy nhất dòng lệnh suy diễn mạng nơ-ron: `action = this.net.predict(obs);`.
    - **Hoàn toàn KHÔNG có sự xuất hiện** của bộ điều khiển heuristic, PID controller, Teacher demonstration, hoặc ghi đè hành động thủ công (`manualAction`).
  - Toàn bộ hành vi của xe bay trên UI trong chế độ RL là do mạng nơ-ron thuần túy quyết định.
- **Trạng thái:** **PASS** (`test_ui_source_code_no_hidden_background_controller`).

---

## 8. CÁC LỖI ĐÃ PHÁT HIỆN VÀ BIỆN PHÁP KHẮC PHỤC (BUGS FOUND & REMEDIATED)

Trong quá trình thẩm định QA, 2 lỗi phần mềm thực tế đã được phát hiện và xử lý triệt để mà không làm thay đổi logic thuật toán học tăng cường:

| STT | Lỗi phát hiện | Vị trí | Nguyên nhân gốc rễ (Root Cause) | Giải pháp khắc phục (Remediation) | Trạng thái |
|:---:|:---|:---|:---|:---|:---:|
| **1** | `TypeError: this.net.predict is not a function` | `web/js/neural_net.js` | Lớp `NeuralNet` chỉ có phương thức `forward(state9)`, thiếu hàm wrapper tiện ích `predict(state9)` mà `visualizer.js` và `demo_guide.js` gọi. | Bổ sung hàm `predict(state9) { return this.forward(state9).action; }` và trường `this.modelName` vào lớp `NeuralNet`. | **ĐÃ SỬA & VERIFIED** |
| **2** | Thiếu file trọng số riêng cho Baseline DQN và Double DQN trên Web | `web/weights/` | Ban đầu thư mục web chỉ có `trained_weights.json`, các tùy chọn mô hình khác chưa có file JSON tương ứng để UI tải độc lập. | Viết script trích xuất ma trận `fc1, fc2, fc3` từ `exp_a_dqn_seed_42.pth` $\rightarrow$ `dqn_weights.json` và từ `exp_b_double_dqn_seed_42.pth` $\rightarrow$ `ddqn_weights.json`. | **ĐÃ SỬA & VERIFIED** |

---

## 9. MA TRẬN TRUY XUẤT YÊU CẦU KIỂM THỬ (TEST TRACEABILITY MATRIX)

| Mã yêu cầu | Hạng mục kiểm tra | File kiểm thử tương ứng | Hàm kiểm thử | Kết quả |
|:---|:---|:---|:---|:---:|
| **REQ-PHY-01** | Trọng trường & Rơi tự do | `tests/test_qa_suite.py` | `test_physics_gravity_free_fall_kinematics` | **PASS** |
| **REQ-PHY-02** | Lực đẩy & Vector xoay | `tests/test_qa_suite.py` | `test_physics_thrust_force_magnitudes_and_rotation` | **PASS** |
| **REQ-PHY-03** | Momen & Cánh tay đòn | `tests/test_qa_suite.py` | `test_physics_torque_and_angular_acceleration` | **PASS** |
| **REQ-PHY-04** | Va chạm 6 điểm & An toàn | `tests/test_qa_suite.py` | `test_physics_collision_detection_and_rules` | **PASS** |
| **REQ-ENV-01** | State 9D chuẩn hóa | `tests/test_qa_suite.py` | `test_env_state_shape_and_continuous_bounds` | **PASS** |
| **REQ-ENV-02** | Ánh xạ 16 hành động | `tests/test_qa_suite.py` | `test_env_action_mapping_bidirectional` | **PASS** |
| **REQ-ENV-03** | 4 điều kiện kết thúc | `tests/test_qa_suite.py` | `test_env_terminal_conditions_and_timeout` | **PASS** |
| **REQ-ENV-04** | Tần số 120Hz/10Hz | `tests/test_qa_suite.py` | `test_env_timestep_and_substepping_count` | **PASS** |
| **REQ-RL-01** | Bellman Target DQN | `tests/test_qa_suite.py` | `test_rl_q_target_standard_dqn_formula` | **PASS** |
| **REQ-RL-02** | Double DQN Decoupling | `tests/test_qa_suite.py` | `test_rl_q_target_double_dqn_decoupling` | **PASS** |
| **REQ-RL-03** | Ring Replay Buffer | `tests/test_qa_suite.py` | `test_rl_replay_buffer_ring_capacity_and_sampling` | **PASS** |
| **REQ-RL-04** | Epsilon Decay & Greedy | `tests/test_qa_suite.py` | `test_rl_epsilon_decay_schedule_and_greedy_evaluation` | **PASS** |
| **REQ-RL-05** | Target Net Isolation | `tests/test_qa_suite.py` | `test_rl_target_network_parameter_isolation` | **PASS** |
| **REQ-RL-06** | Checkpoint Fidelity | `tests/test_qa_suite.py` | `test_rl_checkpoint_persistence_and_weight_fidelity` | **PASS** |
| **REQ-REW-01** | Thưởng hạ cánh & Vùng xanh | `tests/test_qa_suite.py` | `test_reward_safe_landing_and_target_zone_bonuses` | **PASS** |
| **REQ-REW-02** | Phạt va chạm, hết giờ, biên | `tests/test_qa_suite.py` | `test_reward_penalties_crash_timeout_oob_fuel_time` | **PASS** |
| **REQ-REW-03** | PBRS Bất biến chính sách | `tests/test_qa_suite.py` | `test_reward_pbrs_potential_zero_terminal_and_telescoping` | **PASS** |
| **REQ-EVAL-01** | Không train khi test | `tests/test_qa_suite.py` | `test_eval_zero_gradients_and_zero_weight_updates` | **PASS** |
| **REQ-EVAL-02** | Epsilon = 0 xác định | `tests/test_qa_suite.py` | `test_eval_epsilon_strictly_zero_deterministic` | **PASS** |
| **REQ-EVAL-03** | Hạt giống độc lập | `tests/test_qa_suite.py` | `test_eval_independent_test_sets_disjoint_from_training` | **PASS** |
| **REQ-UI-01** | DQN Web = DQN PyTorch | `tests/test_qa_suite.py` | `test_ui_dqn_weights_parity_with_pytorch_baseline` | **PASS** |
| **REQ-UI-02** | DDQN Web = DDQN PyTorch | `tests/test_qa_suite.py` | `test_ui_double_dqn_weights_parity_with_pytorch_baseline` | **PASS** |
| **REQ-UI-03** | Trained Web = Best Checkpoint | `tests/test_qa_suite.py` | `test_ui_trained_model_weights_parity_with_best_checkpoint` | **PASS** |
| **REQ-UI-04** | Dự đoán khác biệt | `tests/test_qa_suite.py` | `test_ui_model_predictions_are_distinct` | **PASS** |
| **REQ-UI-05** | Không controller chạy ngầm | `tests/test_qa_suite.py` | `test_ui_source_code_no_hidden_background_controller` | **PASS** |

---

## 10. KẾT LUẬN VÀ CHỨNG NHẬN CHẤT LƯỢNG (FINAL VERDICT)

Căn cứ vào kết quả thực thi kiểm thử tự động toàn diện và kiểm toán chuyên sâu mã nguồn:
1. Môi trường mô phỏng vật lý và MDP đáp ứng chuẩn xác các quy luật động lực học và đặc tả kỹ thuật dự án.
2. Các thuật toán học tăng cường (DQN, Double DQN, Demonstration-Assisted DQN) tuân thủ nghiêm ngặt lý thuyết Bellman, PBRS và tách biệt đánh giá độc lập.
3. Giao diện Web Visualizer bảo đảm tính trung thực tuyệt đối: chạy đúng mạng nơ-ron thực tế, không can thiệp thuật toán ngoài luồng, số liệu đánh giá đối sánh khớp với evaluator độc lập.

**CHỨNG NHẬN:** Dự án đạt tiêu chuẩn chất lượng cao nhất của một hệ thống Học tăng cường chuẩn mực, sẵn sàng bảo vệ đồ án và triển khai thực tế.
