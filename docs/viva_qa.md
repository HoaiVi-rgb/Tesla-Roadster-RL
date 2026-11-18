# BỘ CÂU HỎI VÀ ĐÁP ÁN BẢO VỆ ĐỒ ÁN (VIVA DEFENSE GUIDE)
## Đề tài: "XE ĐÁP ĐẤT LẤY CẢM HƯNG TỪ XE BAY ROADSTER TESLA"
### Môn học: Học Tăng Cường (Reinforcement Learning - K5 KHDL & TTNT)

---

## MỤC LỤC
1. [Bản chất bài toán Reinforcement Learning vs Heuristic/Control](#câu-1)
2. [Tại sao DQN thuần (Tabula Rasa) thất bại 0/100 ca test?](#câu-2)
3. [Cơ chế Guided Double DQN và Supervised Large Margin Loss](#câu-3)
4. [Double DQN giải quyết triệt để Overestimation Bias như thế nào?](#câu-4)
5. [Cơ sở toán học của Potential-Based Reward Shaping (PBRS)](#câu-5)
6. [Mô hình Động lực học vật rắn 2D và Chuyển động quay](#câu-6)
7. [Tại sao lại có hiện tượng rung lắc/lộn nhào (Spinning) khi đáp?](#câu-7)
8. [Chiến lược Curriculum Learning 4 cấp độ](#câu-8)
9. [Cách triển khai suy luận Neural Network thuần trên Web Canvas](#câu-9)
10. [Ý nghĩa và cách thực hiện bài tập lộn nhào (Acrobatic Flip Bonus)](#câu-10)

---

<a name="câu-1"></a>
### CÂU 1: Giảng viên hỏi: "Dự án này thực sự là Reinforcement Learning hay các bạn dùng Controller kinh điển giả lập rồi gọi là DQN?"

**Trả lời chuẩn chỉnh:**
Thưa Thầy/Cô, đồ án được xây dựng với cam kết **Liêm chính Khoa học tuyệt đối**:
1. **Phân tách hoàn toàn giữa Thu thập dữ liệu và Huấn luyện/Suy luận:**
   - Teacher Controller (bộ điều khiển phản hồi) **chỉ được dùng duy nhất** trong giai đoạn ban đầu để tạo ra 500 tập quỹ đạo mẫu (`demonstrations.jsonl`). Sau khi tạo xong file này, Teacher Controller bị ngắt kết nối hoàn toàn khỏi hệ sinh thái RL.
2. **Quá trình huấn luyện hoàn toàn dựa trên Hàm Giá trị Hành động $Q(s, a)$:**
   - Mạng Q-network được huấn luyện thông qua phương trình Bellman Optimality kết hợp Supervised Large Margin Loss.
   - Không có bất kỳ logic điều khiển heuristic nào can thiệp vào tham số mạng.
3. **Giai đoạn Đánh giá (Evaluation) và Triển khai (Web Demo):**
   - Hành động điều khiển xe $100\%$ là do mạng nơ-ron sâu quyết định:
     $$a^* = \arg\max_{a \in \{0, \dots, 15\}} Q(s, a; \theta)$$
   - Ngay cả trên Web Canvas, mạng nơ-ron được thực thi bằng JavaScript thuần nhân ma trận $W_1, b_1, W_2, b_2, W_3, b_3$ với trọng số được huấn luyện từ PyTorch xuất ra file JSON. Không có một dòng code `if (h < 1.0) v_y = 0` hay heuristic nào ép xe đáp an toàn.

---

<a name="câu-2"></a>
### CÂU 2: Giảng viên hỏi: "Tại sao DQN thuần (Tabula Rasa) học từ đầu lại thất bại hoàn toàn (0/100 ca hạ cánh an toàn)?"

**Trả lời chuẩn chỉnh:**
Thưa Thầy/Cô, hiện tượng này bắt nguồn từ bản chất toán học của bài toán:
1. **Bùng nổ tổ hợp trong không gian rời rạc:**
   - 4 vòi phun tạo ra $16$ tổ hợp hành động. Một episode 4 giây có 40 bước quyết định. Không gian tìm kiếm quỹ đạo có độ phức tạp lên tới $16^{40} \approx 10^{48}$ nhánh.
2. **Tín hiệu phần thưởng cực kỳ thưa thớt (Sparse Reward):**
   - Phần thưởng hạ cánh an toàn $+200$ chỉ xuất hiện ở trạng thái cuối cùng khi thỏa mãn đồng thời 4 điều kiện khắt khe: $|v_y| < 1.0\text{ m/s}$, $|v_x| < 0.5\text{ m/s}$, $|\theta| < 5^\circ$, $|\omega| < 0.2\text{ rad/s}$.
3. **Thảm họa va chạm không thể đảo ngược (Irreversible Catastrophe):**
   - Xe bắt đầu ở độ cao $4 - 9\text{ m}$. Với gia tốc trọng trường $g = 9.8\text{ m/s}^2$, nếu trong $0.5 - 1.0\text{ s}$ đầu tiên agent khám phá ngẫu nhiên ($\epsilon$-greedy) mà không bật vòi nâng, xe sẽ rơi tự do và va chạm mặt đất với vận tốc $> 10\text{ m/s}$, lập tức kết thúc episode. Xác suất để $\epsilon$-greedy ngẫu nhiên chạm đất an toàn là $< 10^{-6}$.
   - Do đó, thuật toán DQN thuần không bao giờ nhận được mẫu chuyển tiếp thành công nào vào Replay Buffer để học. Đây là lý do bài báo **Hester et al. (DeepMind, 2017)** đề xuất kỹ thuật **DQfD** (Deep Q-learning from Demonstrations).

---

<a name="câu-3"></a>
### CÂU 3: Giảng viên hỏi: "Supervised Large Margin Classification Loss trong DQfD hoạt động như thế nào? Tại sao không dùng Cross-Entropy hay MSE trực tiếp?"

**Trả lời chuẩn chỉnh:**
1. **Công thức toán học:**
   $$\mathcal{L}_E(\theta) = \max_{a \in \mathcal{A}} \left[ Q(s, a; \theta) + \ell(a_E, a) \right] - Q(s, a_E; \theta)$$
   Trong đó:
   $$\ell(a_E, a) = \begin{cases} 0 & \text{nếu } a = a_E \\ m & \text{nếu } a \ne a_E \quad (m = 0.2) \end{cases}$$
2. **Tại sao không dùng Cross-Entropy hay MSE?**
   - Trong Imitation Learning (Behavioral Cloning), Cross-Entropy ép xác suất của hành động chuyên gia $P(a_E) \to 1$ và các hành động khác về $0$. Điều này phá hủy hoàn toàn hàm Q-value của Học tăng cường!
   - Trong RL, ta cần hàm $Q(s, a)$ thỏa mãn phương trình Bellman:
     $$Q(s, a) = r + \gamma \max_{a'} Q(s', a')$$
   - **Supervised Large Margin Loss** không ép $Q(s, a)$ của các hành động khác về 0. Nó chỉ yêu cầu $Q(s, a_E)$ của chuyên gia phải **lớn hơn các hành động còn lại một biên độ tối thiểu $m$**:
     $$Q(s, a_E) \ge Q(s, a) + m \quad \forall a \ne a_E$$
   - Nhờ đó, hàm Q vừa bảo tồn giá trị kỳ vọng phần thưởng tương lai của Bellman, vừa bảo đảm chính sách tham lam $\pi(s) = \arg\max_a Q(s, a)$ luôn ưu tiên hành động tối ưu của chuyên gia.

---

<a name="câu-4"></a>
### CÂU 4: Giảng viên hỏi: "Double DQN giải quyết hiện tượng Overestimation Bias như thế nào? Chứng minh bằng toán học."

**Trả lời chuẩn chỉnh:**
1. **Hiện tượng thiên lệch tối đa hóa trong Standard DQN:**
   - Trong DQN chuẩn, mục tiêu TD là:
     $$y_t = r_t + \gamma \max_{a'} Q(s', a'; \theta^-)$$
   - Vì toán tử $\max$ là hàm lồi, theo **Bất đẳng thức Jensen**:
     $$\mathbb{E}\left[ \max_a X_a \right] \ge \max_a \mathbb{E}\left[ X_a \right]$$
   - Khi mạng Q-network có nhiễu xấp xỉ $\epsilon_a \sim \mathcal{N}(0, \sigma^2)$, toán tử $\max$ luôn chọn phần tử có nhiễu dương lớn nhất, dẫn đến Q-value bị thổi phồng có hệ thống (overestimation bias), gây mất ổn định huấn luyện.
2. **Cơ chế tách rời của Double DQN (van Hasselt et al., 2016):**
   - Double DQN tách biệt 2 bước:
     - **Bước 1: Chọn hành động (Action Selection)** dùng Online Network:
       $$a^* = \arg\max_{a'} Q(s', a'; \theta_{online})$$
     - **Bước 2: Ước lượng giá trị (Action Evaluation)** dùng Target Network:
       $$y_t = r_t + \gamma Q(s', a^*; \theta_{target})$$
   - Nhờ đó, ngay cả khi Online Network bị nhiễu làm chọn sai $a^*$, Target Network độc lập sẽ đánh giá giá trị không thiên lệch, loại bỏ hoàn toàn hiện tượng overestimation.

---

<a name="câu-5"></a>
### CÂU 5: Giảng viên hỏi: "Reward Shaping có làm thay đổi chính sách tối ưu không? Chứng minh tại sao hệ thống giữ nguyên Policy."

**Trả lời chuẩn chỉnh:**
1. **Định lý Bất biến Chính sách (Ng, Harada & Russell, 1999):**
   - Một hàm reward shaping $F(s, a, s')$ bảo đảm không làm thay đổi thứ tự ưu tiên của mọi chính sách khi và chỉ khi nó có dạng hiệu thế:
     $$F(s, s') = \gamma \Phi(s') - \Phi(s)$$
2. **Chứng minh qua chuỗi lồng nhau (Telescoping Sum):**
   - Tổng phần thưởng tích lũy qua một episode từ $t=0$ đến $T$:
     $$\sum_{t=0}^T \gamma^t \left[ r_t + F(s_t, s_{t+1}) \right] = \sum_{t=0}^T \gamma^t r_t + \sum_{t=0}^T \gamma^t \left( \gamma \Phi(s_{t+1}) - \Phi(s_t) \right)$$
     $$= \sum_{t=0}^T \gamma^t r_t + \left[ \gamma \Phi(s_1) - \Phi(s_0) + \gamma^2 \Phi(s_2) - \gamma \Phi(s_1) + \dots + \gamma^{T+1} \Phi(s_{T+1}) - \gamma^T \Phi(s_T) \right]$$
     $$= \sum_{t=0}^T \gamma^t r_t - \Phi(s_0) + \gamma^{T+1} \Phi(s_{T+1})$$
   - Khi trạng thái kết thúc có $\Phi(s_{terminal}) = 0$, phần thưởng định hình chỉ làm dịch chuyển hàm mục tiêu một lượng cố định $-\Phi(s_0)$ hoàn toàn độc lập với hành động của agent:
     $$Q^*(s, a) = Q_{base}^*(s, a) - \Phi(s)$$
     $$\arg\max_a Q^*(s, a) = \arg\max_a Q_{base}^*(s, a)$$
   - Do đó, chính sách tối ưu $\pi^*(s)$ được bảo toàn $100\%$, ngăn chặn hoàn toàn hiện tượng Reward Hacking (ví dụ như bay lơ lửng để cày điểm).

---

<a name="câu-6"></a>
### CÂU 6: Giảng viên hỏi: "Mô hình Vật lý 2D tính toán lực và mô-men quay ra sao? Vòi phun tác động thế nào?"

**Trả lời chuẩn chỉnh:**
1. **Thông số vật lý:**
   - Khối lượng $m = 1200\text{ kg}$, Kích thước $L = 4.4\text{ m}, H = 0.9\text{ m}$.
   - Mô-men quán tính: $I_{zz} = \frac{1}{12} m (L^2 + H^2) = \frac{1200}{12} (4.4^2 + 0.9^2) = 2017.0\text{ kg}\cdot\text{m}^2$.
2. **Bố trí 4 vòi phun:**
   - $V_1$ (Nâng đuôi): tọa độ cục bộ $(-1.8, -0.45)$, hướng lực $+y_{local}$. Lực $9000\text{ N}$.
   - $V_2$ (Nâng đầu): tọa độ cục bộ $(+1.8, -0.45)$, hướng lực $+y_{local}$. Lực $9000\text{ N}$.
   - $V_3$ (Đẩy phải): tọa độ cục bộ $(0.0, 0.0)$, hướng lực $+x_{local}$. Lực $6000\text{ N}$.
   - $V_4$ (Đẩy trái): tọa độ cục bộ $(0.0, 0.0)$, hướng lực $-x_{local}$. Lực $6000\text{ N}$.
3. **Mô-men quay tác động lên xe:**
   $$\tau = r \times F$$
   - Khi bật riêng $V_1$: $\tau_1 = (-1.8) \times (+9000) = -16,200\text{ N}\cdot\text{m}$ (quay theo chiều kim đồng hồ, chúi đầu xuống).
     - Gia tốc góc: $\alpha_1 = \frac{\tau_1}{I_{zz}} = \frac{-16200}{2017} = -8.03\text{ rad/s}^2$.
   - Khi bật riêng $V_2$: $\tau_2 = (+1.8) \times (+9000) = +16,200\text{ N}\cdot\text{m}$ (ngóc đầu lên, $\alpha_2 = +8.03\text{ rad/s}^2$).
   - Khi bật cả $V_1$ và $V_2$: $\tau = 0$, tổng lực nâng đạt $18,000\text{ N} > m \cdot g = 11,760\text{ N}$, xe tăng tốc thẳng đứng với gia tốc:
     $$a_y = \frac{18000 - 11760}{1200} = +5.2\text{ m/s}^2$$

---

<a name="câu-7"></a>
### CÂU 7: Giảng viên hỏi: "Tại sao trong bước điều khiển 0.1s lại xảy ra hiện tượng chattering/rung lắc góc nghiêng nếu không xử lý cẩn thận?"

**Trả lời chuẩn chỉnh:**
- Mỗi bước quyết định của mạng nơ-ron kéo dài $\Delta t_{decision} = 0.1\text{ s}$.
- Với gia tốc góc $\alpha = 8.03\text{ rad/s}^2$, nếu mạng giữ hành động bật một vòi nâng trong $0.1\text{ s}$, độ biến thiên vận tốc góc đạt:
  $$\Delta \omega = \alpha \cdot \Delta t = 8.03 \times 0.1 \approx 0.803\text{ rad/s}$$
- Trong khi đó, tiêu chuẩn hạ cánh an toàn đòi hỏi $|\omega| < 0.2\text{ rad/s}$.
- Do đó, nếu ở sát mặt đất mà mạng nơ-ron thực hiện điều khiển bật/tắt luân phiên (bang-bang chattering), mỗi bước sẽ tạo ra một cú giật góc $\Delta \omega = 0.8\text{ rad/s} > 0.2\text{ rad/s}$, ngay lập tức vi phạm tiêu chí an toàn và bị xử thua.
- Mạng đã học được chính sách tối ưu: Khi tiếp cận mặt đất ($h < 1.2\text{ m}$), khóa cân bằng góc sớm ($|\theta| < 3^\circ$), kích hoạt đồng thời cả 2 vòi $V_1$ và $V_2$ để tạo lực hãm thẳng đứng thuần túy mà không sinh mô-men quay ($\tau = 0$).

---

<a name="câu-8"></a>
### CÂU 8: Giảng viên hỏi: "Curriculum Learning được thiết kế như thế nào để đảm bảo tính tổng quát hóa?"

**Trả lời chuẩn chỉnh:**
Chương trình học gồm 3 cấp độ chuẩn + 1 cấp độ nhào lộn mở rộng:
- **Level 1 (Căn bản):** Độ cao thấp $h \in [1.2, 3.0]\text{ m}$, góc lệch nhỏ $|\theta| \le 3^\circ$, sai lệch vị trí $|x| \le 0.2\text{ m}$. Agent học phản xạ hãm tốc độ rơi $v_y$ và tiếp đất êm.
- **Level 2 (Trung cấp):** Độ cao vừa $h \in [2.0, 5.0]\text{ m}$, góc nghiêng $|\theta| \le 8^\circ$, lệch ngang $|x| \le 0.5\text{ m}$. Agent học kết hợp điều khiển thăng bằng góc $\theta$ và vị trí $x$.
- **Level 3 (Thách thức toàn diện):** Độ cao cực đại $h \in [4.0, 9.0]\text{ m}$, góc nghiêng lớn $|\theta| \le 15^\circ$, lệch ngang tối đa $|x| \le 2.0\text{ m}$, vận tốc ban đầu đa dạng.
- **Điều kiện thăng cấp tự động:** Chỉ khi tỷ lệ thành công của cửa sổ 50 episode gần nhất đạt $\ge 80\%$, Curriculum Manager mới nâng cấp độ, tránh hiện tượng Catastrophic Forgetting (quên thảm họa).

---

<a name="câu-9"></a>
### CÂU 9: Giảng viên hỏi: "Web Visualizer chạy như thế nào? Có cần server Python chạy ngầm không?"

**Trả lời chuẩn chỉnh:**
- Giao diện trực quan hóa là **100% Standalone Web Client** (HTML5 + CSS3 + Pure JavaScript). **Không cần cài đặt, không cần backend Python hay server trung gian!**
- **Cơ chế suy luận:**
  1. Trọng số mạng PyTorch sau khi huấn luyện được xuất ra file JSON chuẩn hóa `trained_weights.json`.
  2. File JavaScript `neural_net.js` đọc trực tiếp các ma trận trọng số và bias:
     - $z_1 = \text{ReLU}(W_1 s + b_1) \quad (64 \times 9)$
     - $z_2 = \text{ReLU}(W_2 z_1 + b_2) \quad (64 \times 64)$
     - $q = W_3 z_2 + b_3 \quad (16 \times 64)$
     - $a = \arg\max(q)$
  3. Engine vật lý 2D trong `physics.js` mô phỏng chính xác tích phân Symplectic Euler 120Hz với đúng thông số khối lượng, mô-men quán tính, lực vòi phun và va chạm mặt đất như phiên bản Python.
  4. Canvas 2D vẽ đồ họa xe, vòi phun rực lửa, véc-tơ lực, vùng mục tiêu và sơ đồ mạng nơ-ron 64 nơ-ron hoạt động trực quan theo thời gian thực.

---

<a name="câu-10"></a>
### CÂU 10: Giảng viên hỏi: "Điểm thưởng Acrobatic Flip (lộn nhào trên không $360^\circ$) được thực hiện và kiểm chứng như thế nào?"

**Trả lời chuẩn chỉnh:**
- **Quy tắc chấm điểm:** Hoàn thành lộn vòng trên không ($n \times 360^\circ$) và sau đó hạ cánh an toàn: $+1$ điểm/vòng, tối đa $+2$ điểm thưởng.
- **Giải pháp kỹ thuật:**
  1. Môi trường theo dõi góc quay tích lũy liên tục $\theta_{accumulated} = \int \omega \, dt$ để đếm chính xác số vòng quay $n = \lfloor |\theta_{accumulated}| / 2\pi \rfloor$.
  2. Ở độ cao lớn ($h \ge 7.5\text{ m}$), xe kích hoạt mô-men quay cực đại bằng một vòi nâng ($V_1$ hoặc $V_2$) để đạt tốc độ quay $\omega \approx 4 - 6\text{ rad/s}$.
  3. Hoàn thành đủ $360^\circ$ trong khoảng $1.5\text{ s}$, sau đó vòi đối nghịch kích hoạt để triệt tiêu hoàn toàn vận tốc góc $\omega \to 0$, đưa thân xe về phương nằm ngang $|\theta| < 3^\circ$.
  4. Cuối cùng, 2 vòi cùng phụt hãm tốc rơi để xe chạm đất ở vận tốc $|v_y| < 0.6\text{ m/s}$ trong thời gian $< 4.5\text{ s}$, giành trọn vẹn điểm thưởng tối đa mà không bị trừ điểm thời gian.
