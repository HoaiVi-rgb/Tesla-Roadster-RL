# KỊCH BẢN THUYẾT TRÌNH VÀ DEMO ĐỒ ÁN HỌC TĂNG CƯỜNG (3–5 PHÚT)
## XE BAY TESLA ROADSTER HẠ CÁNH TỰ ĐỘNG BẰNG DEEP REINFORCEMENT LEARNING

---

> [!IMPORTANT]
> **CAM KẾT TÍNH TRUNG THỰC KHOA HỌC (ZERO FAKE METRICS):**
> Toàn bộ các chỉ số báo cáo trong kịch bản và giao diện trực quan đều được trích xuất nguyên bản từ hệ thống đánh giá thực nghiệm độc lập [evaluation_executive_summary.json](file:///F:/Programming%20Study/Reforcement%20Learning/%C4%90%E1%BB%93%20%C3%A1n%20H%E1%BB%8Dc%20t%C4%83ng%20c%C6%B0%E1%BB%9Dng/evaluation/reports/evaluation_executive_summary.json) trên 100 tập hạt giống ngẫu nhiên chưa từng xuất hiện (`Seeds 30000..30099`) với điều kiện $\epsilon = 0.0$, không cập nhật mạng, không bổ sung buffer.

---

## 1. TỔNG QUAN VÀ PHÂN BỔ THỜI GIAN (3–5 PHÚT)

| Mốc thời gian | Bước thực hiện | Trọng tâm trình diễn | Mục tiêu học thuật |
| :--- | :--- | :--- | :--- |
| **0:00 – 0:30** | **STEP 1: Rơi tự do** | Thả rơi tự do không phản lực ($a = 0$) | Chứng minh trọng lực $g = 9.81\text{ m/s}^2$ và sự phá hủy nếu không điều khiển |
| **0:30 – 1:15** | **STEP 2: Lái thủ công** | Mời người dùng bấm phím `W/A/S/D/Q/E` | Cho thấy tính phi tuyến, quán tính lớn và sự bất khả thi của phản xạ người |
| **1:15 – 2:00** | **STEP 3: Chạy DQN** | Chạy mô hình Standard DQN baseline | Giải thích tại sao Tabula Rasa thất bại ($16^{40}$ states, ảo tưởng Q-value) |
| **2:00 – 2:45** | **STEP 4: Trained Model** | Chạy mô hình Demonstration-Assisted DDQN | Trình diễn chuỗi phản xạ chuẩn xác: cân bằng góc, trôi ngang, retro-burn flare |
| **2:45 – 3:30** | **STEP 5: Minh bạch hóa** | Soi 5 panel: State, Action, Reward, Q-values | Giải đáp 5 câu hỏi cốt lõi: Agent nhìn thấy gì, chọn gì, tại sao, reward, Q |
| **3:30 – 4:15** | **STEP 6: Batch Test** | Bấm chạy 100 tập test Monte Carlo | Chứng minh tính tổng quát hóa không ăn may qua thanh tiến trình thời gian thực |
| **4:15 – 4:45** | **STEP 7: 5 Chỉ số vàng** | Chiếu 5 KPI Cards rực rỡ | Công bố 5 chỉ số vàng thực nghiệm: **92% Safe Landing**, **92% Target Zone** |
| **4:45 – 5:15** | **STEP 8: Đối sánh** | Bảng Head-to-Head & Phân loại lỗi | Đối chiếu 3 thuật toán và phân tích nguyên nhân triệt tiêu 100% lỗi bay lệch biên |

---

## 2. NỔI BẬT 5 CHỈ SỐ VÀNG CỐT LÕI (THE 5 GOLDEN METRICS)

Giao diện Web UI và kịch bản làm nổi bật tuyệt đối 5 chỉ số vàng được quy định trong đề bài đồ án:

```
+--------------------------------------------------------------------------------------------------------+
|                                  5 CHỈ SỐ VÀNG ĐÁNH GIÁ THỰC NGHIỆM ĐỘC LẬP                            |
+------------------------------------+-----------------------------------+-------------------------------+
| 1. SAFE LANDING: 92.0%             | 2. LANDING POSITION: 92.0%        | 3. LANDING TIME: 4.34 s       |
|    |vy| = 0.56 m/s (< 1.0 m/s)     |    |x| <= 0.8m (Vùng mục tiêu)    |    0.0% Timeout (< 20.0s)     |
|    |vx| = 0.10 m/s (< 0.5 m/s)     |    Sai số trung bình: 0.035 m     |    Hạ cánh thần tốc, ổn định  |
+------------------------------------+-----------------------------------+-------------------------------+
| 4. ROTATION ATTITUDE: 2.15°        | 5. SUCCESS RATE: 92.0%            | BONUS: FUEL USED: 7.54%       |
|    100% trong hạn mức 5.0°         |    Vượt xa ngưỡng đồ án (80.0%)   |    Bảo toàn 92.46% nhiên liệu |
|    |omega| = 0.08 rad/s (< 0.2)    |    Đánh giá trên 100 test seeds   |    Khí nén lạnh tiết kiệm     |
+------------------------------------+-----------------------------------+-------------------------------+
```

---

## 3. CHI TIẾT 8 BƯỚC THUYẾT TRÌNH VÀ DEMO

### STEP 1: CHO XE RƠI TỰ DO (FREE FALL)
* **Thời lượng:** `0:00 – 0:30` (30 giây)
* **Thao tác trên UI:**
  - Nhấp chọn **Step 1: Rơi Tự Do** trên thanh điều khiển hoặc bấm phím tắt `[1]`.
  - Hệ thống tự động đặt trạng thái $y = 8.5\text{ m}, x = 0\text{ m}$, toàn bộ 4 van phản lực bị khóa ở mức $0$ (Coast).
* **Lời thoại thuyết trình mẫu:**
  > *"Kính thưa Thầy/Cô trong Hội đồng và các bạn, đây là đồ án nghiên cứu hạ cánh tự động cho xe bay lấy cảm hứng từ Tesla Roadster 2D bằng Deep Reinforcement Learning. Để bắt đầu, ở Bước 1, chúng em cho xe rơi hoàn toàn tự do không bật van đẩy. Dưới gia tốc trọng trường $g = 9.81\text{ m/s}^2$, thân xe nặng 1,200 kg rơi thẳng đứng và đập xuống mặt đất với vận tốc lên tới hơn $12.5\text{ m/s}$. Ngưỡng chịu lực an toàn theo đặc tả kỹ thuật là $|v_y| < 1.0\text{ m/s}$. Rơi tự do gây nổ vụn thân xe, va chạm phá hủy với điểm phạt cực nặng -100 điểm. Điều này đặt ra bài toán bắt buộc phải có một chính sách điều khiển phản lực cực kỳ chính xác và kịp thời."*
* **Trọng tâm học thuật:**
  - Thiết lập baseline động lực học: không có lực nâng, hàm phạt va chạm $r_{crash} = -100$ được kích hoạt.

---

### STEP 2: THỬ THÁCH ĐIỀU KHIỂN THỦ CÔNG (MANUAL PILOT)
* **Thời lượng:** `0:30 – 1:15` (45 giây)
* **Thao tác trên UI:**
  - Nhấp chọn **Step 2: Lái Thủ Công**.
  - Bấm các phím `W` (Nâng thẳng), `A` (Đẩy trái), `D` (Đẩy phải), `Q` (Nâng đuôi / Chúi mũi), `E` (Nâng đầu / Ngửa mũi).
  - Mời một thành viên hội đồng hoặc sinh viên thử lái giữ thăng bằng xe trong 10 giây.
* **Lời thoại thuyết trình mẫu:**
  > *"Ở Bước 2, chúng em kính mời Thầy/Cô hoặc các bạn thử trực tiếp lái chiếc xe này bằng bàn phím. Các bạn sẽ thấy việc điều khiển bằng tay là cực kỳ gian nan. Thân xe dài 4.4 mét có mô-men quán tính lên tới $2,017\text{ kg}\cdot\text{m}^2$. Mỗi van phản lực nâng tạo lực đẩy $9,000\text{ N}$ với cánh tay đòn $1.8\text{ m}$, sinh ra mô-men xoắn khổng lồ $\pm 16,200\text{ N}\cdot\text{m}$. Chỉ cần một tích tắc chậm trễ $100\text{ ms}$, xe sẽ lập tức bị lộn nhào hoặc trôi dạt với gia tốc góc hơn $8\text{ rad/s}^2$ rồi rơi vỡ vụn. Hệ thống vật lý phi tuyến bậc cao này đòi hỏi một thuật toán có tần số ra quyết định $10\text{ Hz}$ và khả năng phản xạ bù trừ mô-men xoắn tức thời."*
* **Trọng tâm học thuật:**
  - Thể hiện sự khắc nghiệt của bài toán con lắc ngược đa chiều (inverted pendulum flight) với lực đẩy xung rời rạc 16 trạng thái.

---

### STEP 3: CHẠY MÔ HÌNH DQN BASELINE (TABULA RASA)
* **Thời lượng:** `1:15 – 2:00` (45 giây)
* **Thao tác trên UI:**
  - Nhấp chọn **Step 3: Chạy DQN**.
  - Hệ thống tự động nạp mạng `dqn_baseline_weights.json` (`window.DQN_BASELINE_WEIGHTS`).
  - Reset xe ở Level 3 (độ cao 4–9m, góc nghiêng ngẫu nhiên).
  - Quan sát xe kích hoạt van hỗn loạn, không hãm được tốc độ và bay văng ra ngoài biên (`Out of Bounds`).
* **Lời thoại thuyết trình mẫu:**
  > *"Bước 3, chúng em chạy mô hình Standard DQN nguyên bản được huấn luyện từ con số 0 (Tabula Rasa). Kết quả thực nghiệm cho thấy DQN baseline đạt tỷ lệ thành công đúng 0.0% trên toàn bộ 100 tập kiểm định. Tại sao lại như vậy? Thứ nhất, không gian tìm kiếm là 16 hành động rời rạc trong 200 bước quyết định, tạo ra $16^{40} \approx 10^{48}$ quỹ đạo khả dĩ trong khi phần thưởng hạ cánh an toàn (+200 điểm) lại cực kỳ thưa thớt (sparse reward). Thứ hai, toán tử $\max_a Q(s', a)$ trong Standard DQN bị ảnh hưởng nặng nề bởi bất đẳng thức Jensen, gây hiện tượng thổi phồng giá trị ảo (Overestimation Bias). Mạng nơ-ron ảo tưởng rằng việc trôi dạt ra biên có giá trị $Q \approx +11.5$, dẫn tới 100% các lần bay đều bị văng khỏi biên (93% Overshoot, 7% Undershoot)."*
* **Trọng tâm học thuật:**
  - Giải thích Overestimation Bias: $\mathbb{E}[\max X] \ge \max \mathbb{E}[X]$.
  - Minh chứng cho sự thất bại của thám hiểm ngẫu nhiên $\epsilon$-greedy trong không gian thưa thớt.

---

### STEP 4: CHẠY MÔ HÌNH ĐÃ HUẤN LUYỆN (TRAINED MODEL)
* **Thời lượng:** `2:00 – 2:45` (45 giây)
* **Thao tác trên UI:**
  - Nhấp chọn **Step 4: Trained Model**.
  - Hệ thống nạp trọng số tối ưu `trained_weights.json` (Demonstration-Assisted Double DQN).
  - Reset thử 2–3 lần ở Level 3 với các độ cao và góc nghiêng khác nhau.
  - Quan sát xe tự động: (1) Cân bằng góc nghiêng về $0^\circ$ trong $< 1.2\text{s}$, (2) Trôi ngang về tâm $x = 0$, (3) Bật đồng thời $V_1+V_2$ hãm xung hạ cánh nhẹ nhàng như nhung.
* **Lời thoại thuyết trình mẫu:**
  > *"Tại Bước 4, chúng em chuyển sang mô hình hoàn chỉnh của đồ án: Demonstration-Assisted Double DQN. Thầy/Cô có thể thấy sự khác biệt một trời một vực! Ngay khi xuất phát ở độ cao 8 mét và thân xe đang bị nghiêng lệch tới $15^\circ$, Agent lập tức kích hoạt van phản xạ vi sai để triệt tiêu góc nghiêng về dưới $2^\circ$ chỉ trong vòng 1.2 giây. Sau đó, xe kích hoạt van đẩy ngang đưa vị trí về sát tâm bãi đáp $x \approx 0\text{ m}$. Khi cách mặt đất khoảng 1.5 mét, mạng kích hoạt xung lực hãm retro-burn đồng thời cả van 1 và van 2 ($18,000\text{ N} > m \cdot g = 11,772\text{ N}$) để hãm vận tốc rơi từ $-6\text{ m/s}$ xuống còn $-0.56\text{ m/s}$. Xe chạm đất êm ái, đạt trọn vẹn điểm thưởng an toàn +200 và thưởng trúng tâm +30 điểm!"*
* **Trọng tâm học thuật:**
  - Trình diễn 3 pha bay kinh điển của kỹ thuật hàng không vũ trụ:
    1. *Attitude Recovery Phase:* Triệt tiêu $\theta$ và $\omega$.
    2. *Glideslope Guidance Phase:* Dẫn đường $x \to 0$.
    3. *Terminal Touchdown Flare:* Hãm xung tiếp đất $|v_y| < 1.0\text{ m/s}$.

---

### STEP 5: PHÂN TÍCH MINH BẠCH (STATE / ACTION / REWARD / Q-VALUES)
* **Thời lượng:** `2:45 – 3:30` (45 giây)
* **Thao tác trên UI:**
  - Nhấp chọn **Step 5: Minh Bạch Hóa**.
  - Giảm tốc độ mô phỏng hoặc bấm Pause để chỉ vào 5 panel bên tay phải:
    - Panel 1: State 9D
    - Panel 2: Action 16 tổ hợp van
    - Panel 3: Reward phân rã ($r_{base}$, $\Delta\Phi$, cost)
    - Panel 4: Biểu đồ cột 16 giá trị Q-values
    - Panel 5: Lời giải thích tự nhiên của Policy
* **Lời thoại thuyết trình mẫu:**
  > *"Ở Bước 5, hệ thống mở rộng khung Explainability giúp hội đồng nhìn thấy chính xác nội tại của Agent qua 5 câu hỏi then chốt:
  > 1. **Agent nhìn thấy gì?** Vector trạng thái chuẩn hóa 9 chiều $[x/10, h/15, v_x/10, v_y/10, \sin\theta, \cos\theta, \omega/5, \text{fuel}/100, t/20]$, giải quyết triệt để tính gián đoạn tuần hoàn góc xoay bằng mã hóa lượng giác.
  > 2. **Agent chọn hành động nào?** Trong 16 tổ hợp van, Agent đang chọn Action 3 (bật van 1 và van 2), tạo lực nâng $18,000\text{ N}$ với mô-men xoắn bằng 0.
  > 3. **Tại sao chọn hành động này?** Nhờ Supervised Large Margin Loss ($\lambda = 5.0, m = 0.2$), hành động tối ưu $a^*$ có giá trị $Q$ vượt trội hơn các hành động sai lầm một khoảng lề an toàn $\Delta Q \ge 0.2$.
  > 4. **Reward là bao nhiêu?** Hàm thưởng định hình thế năng bảo toàn PBRS $r_{learn} = 0.01[r_{base} + \gamma\Phi(s') - \Phi(s)]$ giúp dẫn đường học tập mà không làm biến đổi nghiệm tối ưu.
  > 5. **Q-value thay đổi thế nào?** Mạng Double DQN tách rời khâu chọn hành động và định giá, triệt tiêu ảo tưởng phần thưởng, đưa giá trị $Q$ hội tụ chuẩn xác về kỳ vọng thực tế $+1.97$."*
* **Trọng tâm học thuật:**
  - Trả lời trọn vẹn 5 câu hỏi explainability của đề bài.

---

### STEP 6: CHẠY KIỂM THỬ HÀNG LOẠT (BATCH BENCHMARK)
* **Thời lượng:** `3:30 – 4:15` (45 giây)
* **Thao tác trên UI:**
  - Chuyển sang Tab 4: **Benchmark & Đối Sánh**.
  - Nhấp nút **▶ Chạy 100 Tập (Full Benchmark)**.
  - Quan sát thanh tiến trình chạy từ $0\% \to 100\%$ trong vòng 3–5 giây với các bộ đếm số lần Safe Landing, Crash, Out of Bounds nhảy liên tục thời gian thực.
* **Lời thoại thuyết trình mẫu:**
  > *"Để chứng minh mô hình không phải chỉ ăn may trong một vài tình huống khởi tạo thuận lợi, tại Bước 6, chúng em kích hoạt Trình chạy kiểm thử hàng loạt Monte Carlo gồm đúng 100 tập thử nghiệm độc lập (Seeds 30000..30099). Hệ thống tích hợp sẵn nhân tính toán vật lý 120Hz thuần JavaScript chạy song song với mạng nơ-ron inference. Thầy/Cô có thể thấy thanh tiến trình đang quét qua từng hạt giống ngẫu nhiên: số ca Safe Landing tăng đều đặn, số ca trôi biên là 0, và tỷ lệ thành công hội tụ ổn định ở mức 92%."*
* **Trọng tâm học thuật:**
  - Tính tái lập (reproducibility) và phương pháp đánh giá Monte Carlo không thiên lệch.

---

### STEP 7: BÁO CÁO 5 CHỈ SỐ VÀNG THỰC NGHIỆM ĐỘC LẬP
* **Thời lượng:** `4:15 – 4:45` (30 giây)
* **Thao tác trên UI:**
  - Cuộn đến phần **5 Chỉ Số Vàng Cốt Lõi** trên Tab Benchmark.
  - Chỉ vào 5 KPI Card đang phát sáng viền vàng/xanh:
* **Lời thoại thuyết trình mẫu:**
  > *"Tại Bước 7, chúng em trân trọng công bố 5 chỉ số vàng thực nghiệm được xác thực khách quan bởi bộ đánh giá độc lập:
  > - **Chỉ số 1 (SAFE LANDING):** Đạt **92.0%** (92/100 tập đạt chuẩn tuyệt đối, $|v_y| = 0.561\text{ m/s} < 1.0$, $|v_x| = 0.104\text{ m/s} < 0.5$).
  > - **Chỉ số 2 (LANDING POSITION):** Đạt **92.0%** rơi trúng vùng mục tiêu tâm bãi đáp, sai số vị trí tiếp đất trung bình chỉ **3.5 cm** ($0.035\text{ m}$).
  > - **Chỉ số 3 (LANDING TIME):** Thời gian hạ cánh trung bình chỉ **4.34 giây**, tỷ lệ Timeout quá giờ là **0.0%**.
  > - **Chỉ số 4 (ROTATION ATTITUDE):** Góc lệch thân xe trung bình chỉ **2.15°**, **100% các lần đáp đều nằm trong giới hạn an toàn 5.0°**.
  > - **Chỉ số 5 (SUCCESS RATE):** Đạt **92.0%**, vượt xa yêu cầu đồ án (80.0%).
  > Đồng thời mức tiêu hao nhiên liệu trung bình chỉ **7.54%**, bảo toàn hơn 92% khí nén lạnh."*
* **Trọng tâm học thuật:**
  - Xác nhận tính chính xác tuyệt đối không giả mạo số liệu (Zero Fake Metrics).

---

### STEP 8: SO SÁNH ĐỐI ĐẦU THUẬT TOÁN & PHÂN LOẠI LỖI
* **Thời lượng:** `4:45 – 5:15` (30 giây)
* **Thao tác trên UI:**
  - Cuộn xuống **Bảng Đối Sánh 3 Mô Hình** và **Ma Trận Phân Loại Lỗi Thất Bại**.
* **Lời thoại thuyết trình mẫu:**
  > *"Cuối cùng, ở Bước 8, chúng em đặt 3 thuật toán lên bàn cân đối đầu trực tiếp:
  > - **Standard DQN:** 0% thành công, 100% văng khỏi biên do không gian thám hiểm quá rộng và Overestimation bias.
  > - **Double DQN:** Đã giảm được một phần ảo tưởng Q-value nhưng vẫn đạt 0% thành công và 95% văng biên vì bài toán thưa thớt phần thưởng cần mồi khởi động.
  > - **Demonstration-Assisted Double DQN:** Nhờ tích hợp dữ liệu chuyên gia và hàm phạt lề lớn (Large Margin Loss), mô hình học được phản xạ điều khiển ngay từ giai đoạn tiền huấn luyện, triệt tiêu 100% lỗi bay lệch biên, giảm lỗi lật xe từ 59% về 0.0%, đạt tỷ lệ thành công 92.0%.
  > Đây là minh chứng rõ ràng cho sức mạnh của việc kết hợp học tăng cường hiện đại với dữ liệu chuyên gia. Em xin chân thành cảm ơn Thầy/Cô đã chú ý lắng nghe và xin sẵn sàng nhận câu hỏi phản biện!"*
* **Trọng tâm học thuật:**
  - Bảng tổng kết Ablation Study và phân loại lỗi thất bại (Failure Taxonomy).

---

## 4. HỎI ĐÁP PHẢN BIỆN DỰ PHÒNG (DEFENSE Q&A GUIDE)

Dưới đây là 6 câu hỏi hóc búa nhất mà hội đồng chấm đồ án thường đặt ra, kèm câu trả lời chuẩn xác về mặt toán học và kỹ thuật:

### Q1: Tại sao Standard DQN đạt 0.0% trong khi Demonstration-Assisted Double DQN đạt tới 92.0%?
* **Trả lời:**
  1. *Không gian trạng thái - hành động khổng lồ:* Với 16 hành động và 200 bước quyết định, số lượng nhánh tìm kiếm là $16^{40} \approx 1.2 \times 10^{48}$. Thám hiểm ngẫu nhiên $\epsilon$-greedy trong không gian liên tục 9D gần như có xác suất bằng 0 để vô tình tìm thấy chuỗi bật van hãm chuẩn xác.
  2. *Thưa thớt phần thưởng (Sparse Reward):* Chỉ khi tiếp đất an toàn mới nhận +200 điểm, còn giữa không trung chỉ nhận chi phí nhiên liệu và thời gian. Tabula rasa DQN không nhận được tín hiệu gradient có ý nghĩa trước khi hết ngân sách huấn luyện.
  3. *Mồi khởi động từ Demonstrations:* Thuật toán DQfD đưa 20,906 transitions từ bộ điều khiển chuyên gia vào replay buffer và áp dụng Supervised Large Margin Loss. Nhờ đó, ngay từ epoch 0, mạng nơ-ron đã nắm bắt được cấu trúc điều khiển cơ bản, sau đó Double DQN tiếp tục tinh chỉnh để đạt 92% tối ưu.

### Q2: Việc thêm hàm thưởng định hình (PBRS) có làm thay đổi chính sách tối ưu của bài toán không?
* **Trả lời:**
  - **Hoàn toàn không!** Theo định lý nổi tiếng của *Ng, Harada & Russell (1999) - "Policy invariance under reward shaping"*, hàm định hình có dạng thế năng:
    $$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$
  - Khi tính tổng chiết khấu dọc theo toàn bộ quỹ đạo từ $t=0$ đến $T$:
    $$\sum_{t=0}^T \gamma^t F(s_t, a_t, s_{t+1}) = \gamma \Phi(s_1) - \Phi(s_0) + \gamma^2 \Phi(s_2) - \gamma \Phi(s_1) + \dots = \gamma^T \Phi(s_T) - \Phi(s_0)$$
  - Đây là chuỗi lồng nhau (telescoping sum). Vì $\Phi(s_{terminal}) = 0$, tổng phần thưởng định hình chỉ phụ thuộc vào trạng thái khởi đầu $\Phi(s_0)$, hoàn toàn độc lập với chuỗi hành động được chọn. Do đó, **bộ chính sách tối ưu $\pi^*$ được bảo toàn nguyên vẹn**, không bị reward hacking.

### Q3: Double DQN giải quyết hiện tượng Overestimation Bias như thế nào về mặt giải tích?
* **Trả lời:**
  - Trong Standard DQN, mục tiêu TD là:
    $$y_t^{DQN} = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)$$
    Toán tử $\max$ áp dụng trên cùng một bộ trọng số $\theta^-$ vừa để chọn hành động vừa để lượng giá. Do nhiễu ước lượng ngẫu nhiên, theo bất đẳng thức Jensen:
    $$\mathbb{E}[\max_a Q(s, a)] \ge \max_a \mathbb{E}[Q(s, a)]$$
  - Double DQN (*van Hasselt et al., 2015*) tách rời hai nhiệm vụ này:
    $$a^* = \arg\max_a Q(s_{t+1}, a; \theta_t) \quad \text{(Mạng Online chọn hành động)}$$
    $$y_t^{DDQN} = r_t + \gamma Q(s_{t+1}, a^*; \theta_t^-) \quad \text{(Mạng Target lượng giá trị)}$$
  - Vì hai mạng có trọng số khác nhau, sai số ước lượng của mạng online không tương quan với sai số của mạng target, triệt tiêu hoàn toàn độ lệch dương (positive maximization bias).

### Q4: Large Margin Loss trong DQfD hoạt động ra sao?
* **Trả lời:**
  - Hàm mất mát giám sát lề lớn (*Hester et al., 2018*) được định nghĩa:
    $$J_E(Q) = \max_{a \in A} \left[ Q(s, a) + l(a_E, a) \right] - Q(s, a_E)$$
    trong đó hàm lề $l(a_E, a) = 0$ nếu $a = a_E$, và $l(a_E, a) = m = 0.2$ nếu $a \neq a_E$.
  - Cơ chế này ép giá trị $Q$ của hành động chuyên gia $a_E$ phải lớn hơn giá trị $Q$ của mọi hành động khác ít nhất một khoảng lề $m$:
    $$Q(s, a_E) \ge Q(s, a) + 0.2, \quad \forall a \neq a_E$$
  - Nếu mạng dự đoán hành động khác có $Q$ cao hơn hoặc bằng hành động chuyên gia, gradient sẽ phạt rất nặng, giúp mạng học được phản xạ chuẩn mực ngay cả ở những vùng trạng thái thưa thớt dữ liệu.

### Q5: Tại sao không dùng bộ điều khiển cổ điển (PID / LQR) cho toàn bộ bài toán thay vì RL?
* **Trả lời:**
  1. *Ràng buộc phi tuyến và lực đẩy rời rạc:* 4 van khí nén chỉ có 2 trạng thái Bật/Tắt ($0$ hoặc $1$), không thể điều biến liên tục tuyến tính như giả định của LQR hay PID.
  2. *Động học ghép kênh (Coupled Dynamics):* Lực đẩy nâng $V_1, V_2$ vừa sinh lực tịnh tiến theo trục $y$, vừa sinh mô-men xoắn theo góc $\theta$. Khi thân xe nghiêng một góc $\theta$, lực nâng bị xoay thành lực đẩy ngang $F_y \cos\theta + F_x \sin\theta$.
  3. *Khả năng chịu đựng nhiễu lớn:* RL học được trường chính sách phi tuyến toàn cục, có khả năng phục hồi từ góc nghiêng cực lớn $\pm 20^\circ$ và vận tốc rơi bất kỳ, trong khi PID/LQR chỉ ổn định tiệm cận quanh điểm cân bằng nhỏ.

### Q6: Làm thế nào để ngăn chặn hành vi Agent học bay lơ lửng (Hovering Reward Hacking)?
* **Trả lời:**
  - Trong tài liệu thiết kế hàm thưởng [REWARD_DESIGN.md](file:///F:/Programming%20Study/Reforcement%20Learning/%C4%90%E1%BB%93%20%C3%A1n%20H%E1%BB%8Dc%20t%C4%83ng%20c%C6%B0%E1%BB%9Dng/REWARD_DESIGN.md), chúng em áp đặt hai ràng buộc vật lý nghiêm ngặt:
    1. *Phạt thời gian:* $-1.0\text{ điểm / giây}$ liên tục trong suốt chuyến bay.
    2. *Phạt tiêu hao nhiên liệu:* $-0.2\text{ điểm / \% khí nén}$.
  - Nếu Agent bay lượn tại chỗ, mỗi giây sẽ bị trừ điểm thời gian và mất thêm $2.0\%$ nhiên liệu (phạt thêm $-0.4$ điểm). Trong 10 giây bay lượn, Agent sẽ mất $-14$ điểm. Chi phí này lớn hơn rất nhiều so với phần thưởng thế năng trung gian, buộc Agent phải chọn quỹ đạo hạ cánh thẳng xuống bãi đáp trong thời gian ngắn nhất (trung bình chỉ $4.34\text{ s}$).

---

## 5. HƯỚNG DẪN MỞ GIAO DIỆN VÀ VẬN HÀNH THỰC TẾ

1. **Khởi chạy giao diện trực quan:**
   - Mở tệp [index.html](file:///F:/Programming%20Study/Reforcement%20Learning/%C4%90%E1%BB%93%20%C3%A1n%20H%E1%BB%8Dc%20t%C4%83ng%20c%C6%B0%E1%BB%9Dng/web/index.html) bằng bất kỳ trình duyệt hiện đại nào (Chrome, Edge, Firefox).
   - Không yêu cầu cài đặt máy chủ Node.js hay Python backend, toàn bộ mô hình chạy thuần túy bằng JavaScript.
2. **Sử dụng thanh Presenter Toolbar:**
   - Bấm nút **Bước Tiếp Theo ▶** để chuyển tuần tự qua 8 bước.
   - Nhìn vào hộp **Teleprompter** màu xanh để đọc lời thoại mẫu tương ứng với từng bước.
   - Nếu muốn diễn tập tự động, bấm nút **▶ Tự Động Trình Diễn (3-5 Phút)**.
