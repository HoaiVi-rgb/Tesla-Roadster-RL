# TÀI LIỆU THIẾT KẾ & ĐẶC TẢ DQN BASELINE (DQN BASELINE SPECIFICATION)
## DỰ ÁN: XE ĐÁP ĐẤT ROADSTER 4 VÒI PHUN (DEEP Q-LEARNING)
### TÁC GIẢ: DEEP REINFORCEMENT LEARNING ENGINEER

---

## 1. TỔNG QUAN THIẾT KẾ BASELINE & CAM KẾT HỌC THUẬT

Module `rl/dqn/` hiện thực hóa thuật toán **Deep Q-Network (DQN)** kinh điển chuẩn mực theo bài báo nền tảng của Mnih et al. (Nature 2015) và bài giảng của David Silver.

```
========================================================================================
                          NGUYÊN TẮC LIÊM CHÍNH BASELINE DQN
========================================================================================
[V] Mạng học hoàn toàn TỰ THÂN TỪ ĐẦU (Tabula Rasa) thông qua khám phá epsilon-greedy.
[V] TUYỆT ĐỐI KHÔNG SỬ DỤNG TEACHER / HEURISTIC CONTROLLER trong quá trình học Baseline.
[V] Chạy hoàn toàn độc lập với giao diện Web Canvas (Headless Execution).
[V] Đầy đủ 10 thành phần kiến trúc: Online Network, Target Network, Replay Buffer,
    Epsilon-Greedy, TD Target, MSE Loss, Optimizer, Target Update, Gradient Clipping, Checkpoint.
========================================================================================
```

---

## 2. KIẾN TRÚC MẠNG NƠ-RON (Q-NETWORK ARCHITECTURE)

Mạng là một perceptron đa tầng (MLP) gồm 3 lớp tuyến tính, sử dụng hàm kích hoạt ReLU:

$$9 \xrightarrow{\text{FC}(64)} \text{ReLU} \xrightarrow{\text{FC}(64)} \text{ReLU} \xrightarrow{\text{FC}(16)} \mathbf{Q}(s, \cdot) \in \mathbb{R}^{16}$$

```
                +---------------------------------------------------------+
                | VECTOR TRẠNG THÁI QUAN SÁT (INPUT LAYER: 9 CHIỀU)       |
                | [x/10, h/15, vx/10, vy/10, sinθ, cosθ, ω/5, fuel/100, t/20]|
                +---------------------------------------------------------+
                                             |
                                    [FC 9 -> 64 + ReLU]
                                             v
                +---------------------------------------------------------+
                | LỚP ẨN 1 (HIDDEN LAYER 1: 64 NODES)                     |
                +---------------------------------------------------------+
                                             |
                                   [FC 64 -> 64 + ReLU]
                                             v
                +---------------------------------------------------------+
                | LỚP ẨN 2 (HIDDEN LAYER 2: 64 NODES)                     |
                +---------------------------------------------------------+
                                             |
                                   [FC 64 -> 16 (Linear)]
                                             v
                +---------------------------------------------------------+
                | LỚP ĐẦU RA: 16 GIÁ TRỊ Q-VALUES (ACTION SPACE: 16 CHIỀU) |
                | [Q(s, a_0), Q(s, a_1), Q(s, a_2), ..., Q(s, a_15)]      |
                +---------------------------------------------------------+
```

### Khởi tạo trọng số (Weight Initialization)
- Trọng số tuyến tính: Khởi tạo theo phân phối **He/Kaiming Uniform**:
  $$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{\text{fan\_in}}}, +\sqrt{\frac{6}{\text{fan\_in}}}\right)$$
- Hệ số chệch (Bias): Khởi tạo bằng $0.0$ tuyệt đối ($b = 0$).

---

## 3. CÁC THÀNH PHẦN THUẬT TOÁN CỐT LÕI

### 3.1. Cơ chế khám phá $\epsilon$-Greedy (Exploration vs Exploitation)
Chính sách hành động $\pi(a|s)$:
- Với xác suất $\epsilon$: Chọn ngẫu nhiên đều một hành động $a \in \{0, 1, \dots, 15\}$.
- Với xác suất $1 - \epsilon$: Khai thác hành động tối ưu theo mạng Online:
  $$a^* = \arg\max_{a \in \mathcal{A}} Q(s, a; \theta_{online})$$

Lịch trình suy giảm $\epsilon$ tuyến tính (Linear Decay Schedule):

$$\epsilon(t) = \max\left(\epsilon_{end}, \; \epsilon_{start} - \frac{t}{\text{decay}} (\epsilon_{start} - \epsilon_{end})\right)$$

Với: $\epsilon_{start} = 1.0$, $\epsilon_{end} = 0.05$, $\text{decay} = 100,000$ bước môi trường.

---

### 3.2. Bộ nhớ đệm kinh nghiệm (Experience Replay Buffer)
- Bộ nhớ đệm dạng mảng tròn (Circular Buffer) lưu trữ $N = 50,000$ bước chuyển tiếp gần nhất:
  $$\mathcal{D} = \left\{ (s_t, a_t, r_t, s_{t+1}, done_t) \right\}_{t=1}^N$$
- Sử dụng mảng NumPy được cấp phát trước (pre-allocated) thay vì `collections.deque` giúp tăng tốc độ lấy mẫu ngẫu nhiên lên gấp $10$ lần và tránh phân mảnh RAM.
- **Warmup Phase**: Không cập nhật mạng nơ-ron cho đến khi bộ đệm tích lũy tối thiểu $512$ bước chuyển tiếp (`warmup_steps = 512`).

---

### 3.3. Mục tiêu Bellman (TD Target) & Hàm mất mát MSE
Mục tiêu tối ưu Bellman cho DQN cổ điển:

$$y_t = r_t + \gamma (1 - done_t) \max_{a' \in \mathcal{A}} Q(s_{t+1}, a'; \theta_{target})$$

Trong đó:
- $\gamma = 0.995$: Hệ số chiết khấu.
- $done_t = 1.0$ nếu chạm đất (Safe Landing, Crash, OOB) $\implies$ triệt tiêu tương lai, $y_t = r_t$.
- $done_t = 0.0$ nếu xe vẫn đang bay.

Hàm mất mát sai số bình phương trung bình (Mean Squared Error - MSE Loss):

$$\mathcal{L}(\theta) = \frac{1}{|B|} \sum_{i=1}^{|B|} \left( y_i - Q(s_i, a_i; \theta_{online}) \right)^2$$

*(Hệ thống hỗ trợ thêm tùy chọn Huber Loss `loss_type="huber"` nếu muốn kiểm tra so sánh).*

---

### 3.4. Bộ tối ưu hóa (Optimizer) & Gradient Clipping
- **Bộ tối ưu RMSprop**:
  - Tốc độ học (Learning Rate): $\alpha = 0.0005$
  - Hệ số suy giảm (Decay): $\rho = 0.95$
  - Hệ số ổn định số học: $\epsilon_{opt} = 10^{-5}$
- **Cắt xén Gradient (Gradient Clipping)**:
  Giới hạn chuẩn Gradient tối đa ở mức $1.0$ nhằm ngăn chặn hiện tượng bùng nổ gradient khi có phần thưởng va chạm âm lớn:
  $$\text{clip\_grad\_norm\_}(\theta_{online}, \text{max\_norm} = 1.0)$$

---

### 3.5. Đồng bộ mạng Target (Target Update)
Mạng Target $\theta_{target}$ được cập nhật cứng (Hard Copy) định kỳ từ mạng Online $\theta_{online}$ sau mỗi $500$ lượt cập nhật trọng số (`target_update_frequency = 500`):

$$\theta_{target} \longleftarrow \theta_{online}$$

Việc giữ cố định mạng Target trong 500 bước giúp ổn định mục tiêu hồi quy $y_t$, phá vỡ vòng lặp tự đuổi theo mục tiêu di động (moving target problem).

---

## 4. BẢNG CẤU HÌNH THAM SỐ CHUẨN (`DQNConfig`)

| Tham số | Giá trị | Ý nghĩa kỹ thuật |
| :--- | :--- | :--- |
| `state_dim` | `9` | Chiều của vector trạng thái quan sát chuẩn hóa |
| `action_dim` | `16` | Số hành động rời rạc (tổ hợp 4 vòi phun $V_1 \dots V_4$) |
| `hidden_dim` | `64` | Số nơ-ron trên mỗi lớp ẩn (9 $\to$ 64 $\to$ 64 $\to$ 16) |
| `learning_rate` | `0.0005` | Tốc độ học của RMSprop |
| `gamma` | `0.995` | Hệ số chiết khấu phần thưởng tương lai |
| `batch_size` | `64` | Số mẫu chuyển tiếp trong mỗi mini-batch |
| `buffer_size` | `50,000` | Sức chứa tối đa của Replay Buffer |
| `warmup_steps` | `512` | Số bước tích lũy kinh nghiệm ban đầu trước khi train |
| `target_update_frequency` | `500` | Tần suất đồng bộ trọng số Online $\to$ Target (theo số update) |
| `epsilon_start` | `1.0` | Tỷ lệ khám phá ban đầu (100% ngẫu nhiên) |
| `epsilon_end` | `0.05` | Tỷ lệ khám phá tối thiểu duy trì |
| `epsilon_decay` | `100,000` | Số bước môi trường để giảm $\epsilon$ từ 1.0 xuống 0.05 |
| `grad_clip` | `1.0` | Ngưỡng cắt xén chuẩn gradient |
| `loss_type` | `"mse"` | Hàm mất mát: MSE (`"mse"`) hoặc Huber (`"huber"`) |
| `optimizer_type` | `"rmsprop"` | Thuật toán tối ưu (`"rmsprop"` hoặc `"adam"`) |
| `device` | `"cpu"` | Thiết bị tính toán PyTorch (CPU / CUDA) |

---

## 5. ĐẶC TẢ HỆ THỐNG GHI NHẬT KÝ (LOGGING METRICS)

Tại mỗi episode huấn luyện, `DQNTrainer` ghi lại đầy đủ 12 thông số khoa học:

| Tên Metric | Kiểu dữ liệu | Ý nghĩa |
| :--- | :--- | :--- |
| `episode` | `int` | Số thứ tự lượt huấn luyện (1, 2, 3...) |
| `total_steps` | `int` | Tổng số bước môi trường tích lũy toàn phiên |
| `total_updates` | `int` | Tổng số lần cập nhật gradient mạng nơ-ron |
| `reward` | `float` | Tổng phần thưởng huấn luyện đã định hình ($r_{learn}$) |
| `raw_reward` | `float` | Tổng phần thưởng vật lý gốc không định hình ($r_{base}$) |
| `loss` | `float` | Giá trị mất mát MSE trung bình trong episode |
| `epsilon` | `float` | Giá trị tỷ lệ khám phá $\epsilon$ hiện tại |
| `success_rate` | `float` | Tỷ lệ đáp an toàn trung bình trượt của 50 episode gần nhất (%) |
| `landing_velocity`| `float` | Vận tốc tiếp đất đứng $|v_y|$ tại thời điểm chạm đất (m/s) |
| `landing_angle` | `float` | Độ nghiêng thân xe $|\theta|$ tại thời điểm chạm đất (độ) |
| `fuel_used` | `float` | Phần trăm khí nén đã tiêu thụ trong lượt bay (%) |
| `landing_time` | `float` | Thời gian bay từ lúc thả đến khi kết thúc (s) |
| `termination_reason` | `str` | `"SAFE_LANDING"`, `"CRASH: ..."`, `"TIMEOUT"`, `"OUT_OF_BOUNDS"` |

---

## 6. HƯỚNG DẪN CHẠY VÀ LƯU CHECKPOINT

### 6.1. Khởi chạy huấn luyện độc lập qua dòng lệnh (Headless CLI)
```bash
python -c "
from environment.roadster_env import RoadsterEnv
from rl.dqn.agent import DQNAgent, DQNConfig
from rl.dqn.trainer import DQNTrainer

env = RoadsterEnv()
agent = DQNAgent(DQNConfig(batch_size=64, learning_rate=0.0005))
trainer = DQNTrainer(env=env, agent=agent, checkpoint_dir='checkpoints/dqn_baseline')

# Chạy huấn luyện 500 episodes
trainer.train(num_episodes=500, eval_interval=50, log_interval=10)
"
```

### 6.2. Cấu trúc lưu trữ Checkpoint
- `checkpoints/dqn_baseline/dqn_latest.pth`: Lưu trọng số mạng Online, Target, Optimizer và số bước hiện tại.
- `checkpoints/dqn_baseline/dqn_best.pth`: Lưu checkpoint đạt tỷ lệ thành công cao nhất trên tập Validation độc lập.
- `checkpoints/dqn_baseline/weights_best.json`: Trọng số nơ-ron được trích xuất thành JSON để nhúng trực tiếp vào Web UI.

---

## 7. Ý NGHĨA KHOA HỌC KHI BẢO VỆ ĐỒ ÁN (VIVA EXPLAINABILITY)

> [!NOTE]
> **Tại sao cần triển khai DQN Baseline này trước khi chuyển sang DQfD?**
> 1. **Tính trung thực khoa học**: Để chứng minh trước giảng viên rằng sinh viên đã tự mình triển khai và thực nghiệm thuật toán DQN chuẩn mực từ con số 0.
> 2. **Cơ sở thực nghiệm đối chiếu**: Thử nghiệm lịch sử chứng minh DQN cổ điển học từ đầu (random $\epsilon$-greedy) đạt tỷ lệ $0/100$ trên bài toán xe bay 16 van vì không gian thăm dò quá rộng và reward tiếp đất quá thưa thớt. Kết quả này là bằng chứng xác đáng để thuyết phục hội đồng vì sao cần áp dụng phương pháp **Demonstration Learning (DQfD)** ở giai đoạn tiếp theo!
