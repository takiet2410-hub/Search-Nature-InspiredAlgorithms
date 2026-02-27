# GIẢI THÍCH CHI TIẾT: Thuật Toán, Bài Toán & Phương Pháp So Sánh

---

## PHẦN 1: GIẢI NGHĨA CÁC BÀI TOÁN (PROBLEMS)

### 1.1 Bài Toán Tối Ưu Liên Tục (Continuous Optimization)

Mục tiêu chung: **Tìm vector x = (x₁, x₂, ..., xₙ) sao cho f(x) nhỏ nhất**, trong một miền tìm kiếm liên tục.

---

#### 🔵 Sphere Function — Bài toán ĐƠN GIẢN nhất

```
f(x) = Σ xᵢ²
```

- **Bounds**: [-5.12, 5.12]
- **Min**: f(0, 0, ..., 0) = 0
- **Đặc điểm**: Unimodal (chỉ có 1 đáy duy nhất), lồi (convex), trơn
- **Ý nghĩa**: Như một cái **bát úp ngược** — chỉ có đúng 1 điểm thấp nhất ở tâm. Thuật toán nào cũng dễ tìm được lời giải tối ưu. Dùng làm **baseline test** — nếu thuật toán KHÔNG giải được Sphere thì chắc chắn có bug.

```
  ╲        ╱
   ╲      ╱
    ╲    ╱
     ╲  ╱
      ╲╱  ← min tại (0,0)
```

---

#### 🔴 Rastrigin Function — Bài toán nhiều BẪY LỪNG

```
f(x) = 10d + Σ [xᵢ² - 10·cos(2π·xᵢ)]
```

- **Bounds**: [-5.12, 5.12]
- **Min**: f(0, 0, ..., 0) = 0
- **Đặc điểm**: Highly multimodal — có **RẤT NHIỀU cực tiểu cục bộ** phân bố đều
- **Ý nghĩa**: Như một **mặt sóng gợn đều** — có hàng trăm "hố" nhỏ, nhưng chỉ có 1 hố sâu nhất ở tâm. Thuật toán greedy (Hill Climbing) rất dễ **rơi vào bẫy** tại các cực tiểu cục bộ. Test khả năng **exploration** (khám phá) của thuật toán.

```
  ∿∿∿∿∿∿∿∿∿∿
  ∿  ∿  ∿  ∿
  ∿∿∿╲∿∿╱∿∿∿
       ╲╱  ← global min, bao quanh bởi hàng trăm local min
```

---

#### 🟡 Rosenbrock Function — Bài toán "THUNG LŨNG HẸP"

```
f(x) = Σ [100·(xᵢ₊₁ - xᵢ²)² + (1 - xᵢ)²]
```

- **Bounds**: [-5, 10]
- **Min**: f(1, 1, ..., 1) = 0
- **Đặc điểm**: Unimodal nhưng **cực kỳ khó hội tụ** — có hình dạng thung lũng cong hẹp (banana shape)
- **Ý nghĩa**: Tìm được vùng thung lũng thì dễ, nhưng **đi đến đáy thung lũng** thì rất chậm vì hướng gradient rất phẳng dọc theo thung lũng. Test khả năng **hội tụ chính xác** (fine-tuning / exploitation).

```
      ___________
     /            ╲
    /   valley     ╲
   /    ~~~~~~~~    ╲
  /     ↓ min at    ╲
  ╲    (1,1,...,1)  ╱
   ╲_______________╱
```

---

#### 🟢 Griewank Function — Bài toán "CỰC TIỂU PHÂN BỐ ĐỀU"

```
f(x) = 1 + Σ(xᵢ²/4000) - Π cos(xᵢ/√i)
```

- **Bounds**: [-600, 600]
- **Min**: f(0, 0, ..., 0) = 0
- **Đặc điểm**: Multimodal nhưng các cực tiểu cục bộ **nông dần** khi ra xa tâm. Miền tìm kiếm RẤT LỚN ([-600, 600]).
- **Ý nghĩa**: Tương tự Rastrigin nhưng có **tương tác giữa các chiều** (thông qua tích cosine). Ở chiều THẤP thì nhiều bẫy, ở chiều CAO thì gần giống Sphere. Test **scalability** — thuật toán xử lý không gian lớn ra sao.

---

#### 🟣 Ackley Function — Bài toán "NHIỀU ĐỈNH NÚI"

```
f(x) = -20·exp(-0.2·√(Σxᵢ²/d)) - exp(Σcos(2πxᵢ)/d) + 20 + e
```

- **Bounds**: [-32, 32]
- **Min**: f(0, 0, ..., 0) = 0
- **Đặc điểm**: Gần như phẳng ở xa tâm nhưng có **1 lỗ sâu đột ngột** tại gốc. Nhiều local optima nông.
- **Ý nghĩa**: Thuật toán cần **exploration mạnh** để tìm được vùng gần gốc, sau đó cần **exploitation tốt** để lặn xuống lỗ sâu. Test cân bằng **exploration vs exploitation**.

---

### 1.2 Bài Toán Rời Rạc (Discrete Optimization)

---

#### 🗺️ Traveling Salesman Problem (TSP) — Bài toán NGƯỜI BÁN HÀNG

**Đề bài**: Cho N thành phố, tìm lộ trình ngắn nhất đi qua tất cả thành phố đúng 1 lần rồi quay về.

- **Không gian tìm kiếm**: N! hoán vị → tăng cực nhanh (10 thành phố = 3,628,800 trạng thái)
- **Tại sao khó**: NP-hard — không có thuật toán chính xác chạy trong thời gian đa thức
- **Ứng dụng thực tế**: Logistics, routing, chip design

> Trong project: BFS/DFS/A* tìm lời giải **chính xác** (nhưng chậm exponential), GA/ACO/SA tìm lời giải **xấp xỉ** (nhanh hơn nhiều).

---

#### 🎒 Knapsack Problem (KP) — Bài toán CHIẾC BA LÔ

**Đề bài**: Cho N vật phẩm, mỗi vật có trọng lượng wᵢ và giá trị vᵢ. Ba lô chứa được tối đa W kg. Chọn tập hợp vật phẩm sao cho **tổng giá trị lớn nhất** mà không vượt quá sức chứa.

- **Không gian tìm kiếm**: 2ᴺ tổ hợp (mỗi vật chọn hoặc không chọn)
- **Tại sao khó**: NP-hard, DP giải chính xác nhưng O(N×W) — chậm khi W lớn
- **Ứng dụng**: Phân bổ ngân sách, cắt vật liệu, portfolio optimization

> Trong project: DP tìm lời giải **chính xác**, GA (binary-encoded) tìm lời giải **xấp xỉ**.

---

#### 🎨 Graph Coloring (GC) — Bài toán TÔ MÀU ĐỒ THỊ

**Đề bài**: Cho đồ thị G(V, E), tô màu các đỉnh sao cho **hai đỉnh kề nhau không cùng màu**, dùng ít màu nhất có thể (chromatic number).

- **Không gian tìm kiếm**: kᴺ (N đỉnh × k màu)
- **Tại sao khó**: NP-complete (ngay cả xác định chromatic number)
- **Ứng dụng**: Lập lịch thi, phân kênh tần số, register allocation trong compiler

> Trong project: Greedy (Welsh-Powell) tô theo bậc giảm dần, GA (integer-encoded) tối thiểu xung đột.

---

#### 🛤️ Shortest Path — Bài toán ĐƯỜNG ĐI NGẮN NHẤT

**Đề bài**: Tìm đường đi ngắn nhất giữa 2 đỉnh trên đồ thị có trọng số.

- **Giải bằng**: BFS (unweighted), DFS (tìm 1 đường bất kỳ), A* (heuristic-guided), UCS (optimal)
- **Trong project**: Được bao phủ bởi TSP — các thuật toán BFS/DFS/A* chính là giải Shortest Path trên không gian trạng thái.

---

---

## PHẦN 2: CÁCH CÁC THUẬT TOÁN HOẠT ĐỘNG

### 2.1 Thuật Toán Tìm Kiếm Cổ Điển (Classical Search)

---

#### BFS (Breadth-First Search) — Tìm kiếm theo CHIỀU RỘNG

```
Hàng đợi (Queue): FIFO
1. Đưa trạng thái ban đầu vào hàng đợi
2. Lấy phần tử ĐẦU hàng đợi ra mở rộng
3. Thêm tất cả con vào CUỐI hàng đợi
4. Lặp lại cho đến khi tìm thấy đích
```

- **Đặc điểm**: Duyệt TỪNG TẦNG — tìm tất cả đường dài 1, rồi dài 2, rồi dài 3...
- **Đảm bảo**: Tìm được lời giải **ngắn nhất** (theo số bước)
- **Nhược điểm**: Tốn RAM cực kỳ — O(bᵈ) bộ nhớ với b = branching factor, d = depth

---

#### DFS (Depth-First Search) — Tìm kiếm theo CHIỀU SÂU

```
Ngăn xếp (Stack): LIFO
1. Đưa trạng thái ban đầu vào stack
2. Lấy phần tử TRÊN CÙNG stack ra mở rộng
3. Thêm tất cả con vào ĐẦU stack
4. Lặp lại
```

- **Đặc điểm**: Đào SÂU hết 1 nhánh trước, rồi quay lại (backtrack) thử nhánh khác
- **Ưu điểm**: Tiết kiệm bộ nhớ — O(b×d)
- **Nhược điểm**: Không đảm bảo tìm đường ngắn nhất, có thể lặp vô tận

---

#### A* Search — Tìm kiếm CÓ HƯỚNG DẪN

```
f(n) = g(n) + h(n)
  g(n) = chi phí thực tế từ gốc → n
  h(n) = ước lượng chi phí từ n → đích (heuristic)

1. Dùng Priority Queue sắp theo f(n)
2. Luôn mở rộng node có f(n) NHỎ NHẤT
3. Nếu h(n) admissible (không bao giờ ước lượng quá) → tìm optimal
```

- **Đặc điểm**: Kết hợp UCS (tối ưu) + Greedy (nhanh) = **vừa nhanh vừa tối ưu**
- **Heuristic trong TSP**: h(n) = (số thành phố chưa thăm) × (cạnh ngắn nhất)
- **Nhược điểm**: Vẫn tốn RAM nhiều, chỉ hiệu quả với heuristic tốt

---

#### Hill Climbing — Leo đồi (Local Search)

```
1. Bắt đầu tại điểm ngẫu nhiên x
2. Tạo hàng xóm x' = x + nhiễu nhỏ
3. Nếu f(x') < f(x) → di chuyển đến x'
4. Nếu không → giữ nguyên
5. Lặp lại
```

- **Đặc điểm**: **Greedy thuần túy** — chỉ chấp nhận cải thiện, không bao giờ đi xuống
- **Ưu điểm**: Đơn giản, nhanh, hội tụ tốt trên bài toán unimodal (Sphere)
- **Nhược điểm**: **Bị kẹt hoàn toàn** tại local optima — trên Rastrigin, HC thường dừng ở bẫy đầu tiên gặp được

---

### 2.2 Thuật Toán Tiến Hóa (Evolution-Based)

---

#### 🧬 Genetic Algorithm (GA) — Giải thuật DI TRUYỀN

**Lấy cảm hứng**: Tiến hóa Darwin — chọn lọc tự nhiên, lai ghép, đột biến.

```
1. KHỞI TẠO: Tạo quần thể P cá thể ngẫu nhiên
2. LẶP mỗi thế hệ:
   a. ĐÁNH GIÁ: Tính fitness mỗi cá thể
   b. CHỌN LỌC: Tournament — chọn K ngẫu nhiên, giữ tốt nhất
   c. LAI GHÉP (Crossover): 
      - TSP: Order Crossover (OX) — giữ đoạn giữa cha, điền thứ tự mẹ
      - KP: Uniform Crossover — mỗi gene 50% từ cha/mẹ
   d. ĐỘT BIẾN (Mutation):
      - TSP: Swap 2 thành phố
      - KP: Flip bit
   e. ELITISM: Giữ lại top cá thể tốt nhất
3. TRẢ VỀ cá thể tốt nhất
```

**GA giải TSP**: Mỗi chromosome = hoán vị [3, 1, 4, 0, 2] = thứ tự đi thăm thành phố.
**GA giải KP**: Mỗi chromosome = binary [1, 0, 1, 1, 0] = chọn/không chọn vật phẩm.
**GA giải GC**: Mỗi chromosome = integer [0, 2, 1, 0, 3] = màu gán cho đỉnh.

---

#### 🔀 Differential Evolution (DE) — Tiến hóa VI SAI

**Lấy cảm hứng**: Tiến hóa nhưng dùng **vector sai phân** thay vì crossover sinh học.

```
Với mỗi cá thể xᵢ trong quần thể:
1. MUTATION: Chọn 3 cá thể r1, r2, r3 khác nhau
   v = x_r1 + F × (x_r2 - x_r3)           ← vector đột biến
   
2. CROSSOVER: Với mỗi chiều j:
   u_j = v_j  nếu rand() < CR              ← lấy từ mutant
   u_j = x_j  nếu rand() >= CR             ← giữ nguyên
   
3. SELECTION: Tham lam
   Nếu f(u) ≤ f(x) → thay x bằng u
   Ngược lại → giữ x
```

- **F** (mutation factor): Điều khiển bước nhảy. F lớn → khám phá mạnh. F nhỏ → khai thác chính xác.
- **CR** (crossover rate): Xác suất lấy gene từ mutant. CR cao → thay đổi nhiều chiều cùng lúc.
- **Ưu điểm**: Ít tham số, hiệu quả cao trên bài liên tục
- **Nhược điểm**: Không áp dụng trực tiếp cho bài rời rạc (TSP)

---

### 2.3 Thuật Toán Vật Lý (Physics-Based)

---

#### 🌡️ Simulated Annealing (SA) — Luyện kim MÔ PHỎNG

**Lấy cảm hứng**: Quá trình ủ kim loại — nung nóng rồi hạ nhiệt từ từ.

```
1. Khởi tạo: x ngẫu nhiên, nhiệt độ T = T_init (cao)
2. LẶP:
   a. Tạo hàng xóm x'
   b. Δ = f(x') - f(x)
   c. Nếu Δ < 0 → LUÔN chấp nhận (cải thiện)
   d. Nếu Δ ≥ 0 → chấp nhận với xác suất p = exp(-Δ/T)
      ← Khi T CAO: p ≈ 1 → chấp nhận cả lời giải tệ hơn (EXPLORATION)
      ← Khi T THẤP: p ≈ 0 → gần như chỉ chấp nhận cải thiện (EXPLOITATION)
   e. Hạ nhiệt: T = T × cooling_rate
3. Dừng khi T < T_min
```

**Khác biệt với Hill Climbing**: HC KHÔNG BAO GIỜ chấp nhận lời giải tệ hơn → bị kẹt. SA CÓ THỂ nhảy qua "đồi" nhờ nhiệt độ cao → thoát local optima.

**SA giải TSP**: Hàng xóm = 2-opt swap (đảo ngược một đoạn trong tour).
**SA giải Continuous**: Hàng xóm = x + Gaussian noise (bước nhỏ dần khi T giảm).

---

### 2.4 Thuật Toán Sinh Học / Bầy Đàn (Biology / Swarm-Based)

---

#### 🐜 Ant Colony Optimization (ACO) — Tối ưu ĐÀN KIẾN

**Lấy cảm hứng**: Kiến thật tìm đường ngắn nhất bằng **pheromone** (dấu mùi).

```
1. Khởi tạo ma trận pheromone τ[i][j] đều nhau
2. LẶP mỗi vòng:
   a. Mỗi con kiến xây lộ trình:
      - Tại thành phố i, chọn thành phố j tiếp theo theo xác suất:
        P(i→j) = [τ(i,j)^α × η(i,j)^β] / Σ
        ← τ = pheromone (kinh nghiệm tích lũy)
        ← η = 1/distance (thông tin heuristic)
        ← α = trọng số pheromone, β = trọng số heuristic
   b. BAY HƠI: τ = τ × (1 - ρ)
   c. THÊM PHEROMONE: Kiến đi đường ngắn → rải nhiều pheromone hơn
      ← deposit = Q / cost
3. Đường tốt nhất tích lũy nhiều pheromone → kiến sau ưu tiên đi
```

- **Ưu điểm**: Tốt cho bài rời rạc, tổ hợp (TSP, scheduling)
- **Nhược điểm**: Nhiều tham số (α, β, ρ, Q), chậm hội tụ

---

#### 🐦 Particle Swarm Optimization (PSO) — Tối ưu BẦY ĐÀN

**Lấy cảm hứng**: Đàn chim tìm thức ăn — mỗi con bay theo 2 hướng: **kinh nghiệm bản thân** (pbest) và **kinh nghiệm đàn** (gbest).

```
Mỗi hạt i có: vị trí xᵢ, vận tốc vᵢ, kỷ lục cá nhân pbestᵢ

Cập nhật mỗi vòng lặp:
  vᵢ = w·vᵢ + c1·r1·(pbestᵢ - xᵢ) + c2·r2·(gbest - xᵢ)
        ↑         ↑                      ↑
     quán tính   kéo về pbest         kéo về gbest
     (exploration) (exploitation)     (exploitation)
     
  xᵢ = xᵢ + vᵢ
```

- **w** (inertia): w lớn → bay thẳng dài (explore). w nhỏ → phanh lại (exploit).
- **c1** (cognitive): Mức độ tin vào kinh nghiệm bản thân
- **c2** (social): Mức độ tin vào kinh nghiệm đàn
- **Ưu điểm**: Rất đơn giản, hội tụ nhanh
- **Nhược điểm**: Dễ premature convergence (đàn đổ xô về 1 điểm local optima)

---

#### 🐝 Artificial Bee Colony (ABC) — Tối ưu ĐÀN ONG

**Lấy cảm hứng**: Ong mật tìm nguồn hoa qua 3 pha.

```
3 PHA mỗi vòng lặp:

PHA 1 — EMPLOYED BEE (Ong thợ):
  Mỗi ong khai thác 1 nguồn hoa, thử tìm hàng xóm tốt hơn
  v_ij = x_ij + φ × (x_ij - x_kj)
  ← φ ngẫu nhiên trong [-1, 1], k = nguồn khác, j = chiều ngẫu nhiên
  Nếu tốt hơn → thay thế. Nếu không → tăng counter thất bại.

PHA 2 — ONLOOKER BEE (Ong quan sát):
  Đợi ở tổ, chọn nguồn hoa theo ĐỘ TỐT (Roulette Wheel)
  p_i = fitness_i / Σ fitness     ← nguồn tốt → nhiều ong đến hơn
  Tìm hàng xóm giống Employed Bee

PHA 3 — SCOUT BEE (Ong trinh sát):
  Nếu 1 nguồn bị bỏ quá nhiều lần (counter > limit) → BỎ
  Tạo nguồn hoa MỚI ngẫu nhiên → EXPLORATION MẠNH
```

- **Ưu điểm**: Cân bằng exploration/exploitation tự nhiên nhờ 3 pha
- **Nhược điểm**: Hội tụ chậm hơn PSO, nhiều hàm eval

---

#### 🔥 Firefly Algorithm (FA) — Thuật toán ĐOM ĐÓM

**Lấy cảm hứng**: Đom đóm bay về phía con **sáng hơn** (lời giải tốt hơn).

```
Mỗi đom đóm i, so sánh với mọi đom đóm j:
  Nếu j SÁNG HƠN (f(j) < f(i)):
    β(r) = β₀ · exp(-γ · r²)    ← Lực hấp dẫn giảm theo khoảng cách
    xᵢ = xᵢ + β·(xⱼ - xᵢ) + α·(rand - 0.5)·scale
              ↑                   ↑
         kéo về phía j      nhiễu ngẫu nhiên

  Nếu không có ai sáng hơn → random walk
  
  α giảm dần: α = α × decay     ← exploration → exploitation
```

- **γ** (absorption): γ lớn → chỉ nhìn gần (LOCAL search). γ nhỏ → nhìn xa (GLOBAL search).
- **Ưu điểm**: Exploration tự nhiên (mỗi con bay về 1 đích khác), tốt cho multimodal
- **Nhược điểm**: O(N²) mỗi vòng lặp (so sánh từng cặp), chậm

---

#### 🐣 Cuckoo Search (CS) — Tìm kiếm CHIM CU CU

**Lấy cảm hứng**: Chim cu cu ĐẺ NHỜ — đặt trứng vào tổ chim khác. Levy flight mô phỏng hành vi tìm kiếm trong tự nhiên.

```
1. Mỗi tổ = 1 lời giải
2. Tạo lời giải mới bằng LEVY FLIGHT:
   x_new = x + α × L(β)
   
   L(β) = u / |v|^(1/β)    ← Levy distribution
   ← Có bước ĐI NGẮN thường xuyên + bước NHẢY DÀI thỉnh thoảng
   ← Rất hiệu quả cho exploration: nhảy xa vượt qua local optima
   
3. Replace ngẫu nhiên: Nếu trứng mới tốt hơn tổ j → thay thế
4. ABANDON: pa% tổ tệ nhất bị bỏ → tạo tổ mới ngẫu nhiên
```

- **α** (step size): Điều khiển độ lớn bước Levy
- **pa** (abandonment): Tỉ lệ tổ bị bỏ mỗi vòng → exploration
- **Ưu điểm**: Levy flight = **exploration cực mạnh**, ít tham số
- **Nhược điểm**: Có thể premature converge nếu pa quá thấp

---

### 2.5 Thuật Toán Hành Vi Con Người (Human Behavior-Based)

---

#### 👨‍🏫 TLBO (Teaching-Learning-Based Optimization) — Tối ưu DẠY-HỌC

**Lấy cảm hứng**: Lớp học — Giáo viên dạy học sinh, học sinh trao đổi với nhau.

**ĐẶC BIỆT: KHÔNG CÓ THAM SỐ ĐIỀU CHỈNH** (parameter-free) — chỉ cần pop_size.

```
PHA 1 — TEACHER PHASE (Giáo viên dạy):
  Teacher = cá thể TỐT NHẤT trong quần thể
  Mean = trung bình quần thể
  T_F = random(1 hoặc 2)          ← Teaching Factor
  
  Với mỗi học sinh xᵢ:
    x_new = xᵢ + r × (Teacher - T_F × Mean)
                       ↑              ↑
                  kéo về phía     đẩy xa khỏi
                  người giỏi nhất  trung bình lớp
                  
  Nếu x_new tốt hơn xᵢ → thay thế (greedy)

PHA 2 — LEARNER PHASE (Học sinh trao đổi):
  Với mỗi học sinh xᵢ, chọn ngẫu nhiên 1 bạn xⱼ:
    Nếu xᵢ giỏi hơn xⱼ:
      x_new = xᵢ + r × (xᵢ - xⱼ)     ← đi xa khỏi bạn yếu hơn
    Nếu xⱼ giỏi hơn xᵢ:
      x_new = xᵢ + r × (xⱼ - xᵢ)     ← đi về phía bạn giỏi hơn
      
  Nếu x_new tốt hơn → thay thế
```

**Tại sao parameter-free?**
- Không có F, CR (như DE), không có w, c1, c2 (như PSO)
- Chỉ cần chọn pop_size → đơn giản, ít phải tuning
- T_F tự động random → tự điều chỉnh exploration/exploitation

**Ưu điểm**:
- **Không cần tinh chỉnh tham số** — lợi thế lớn trong thực tế
- Teacher Phase = exploitation (học theo người giỏi nhất)
- Learner Phase = exploration (tương tác đa dạng)

**Nhược điểm**:
- Hội tụ có thể chậm hơn DE/PSO khi tham số được tinh chỉnh tốt
- Teacher = best → nếu best là local optima thì cả lớp bị kéo theo

---

---

## PHẦN 3: PHƯƠNG PHÁP SO SÁNH (COMPARISON METHODOLOGY)

### 3.1 Các chỉ số so sánh (Metrics)

| Metric | Ý nghĩa | Cách đo |
|---|---|---|
| **Convergence Speed** | Thuật toán hội tụ NHANH hay CHẬM? | Biểu đồ fitness theo thế hệ (Mean ± Std qua 30 runs) |
| **Solution Quality** | Lời giải TỐT đến mức nào? | Mean, Std, Best, Worst qua 30 runs → Boxplot |
| **Scalability** | Chịu được bài toán LỚN không? | Thời gian chạy vs kích thước (N thành phố / D chiều) |
| **Robustness** | Kết quả CÓ ỔN ĐỊNH không? | Std nhỏ = ổn định. Std lớn = kết quả dao động |
| **Parameter Sensitivity** | Nhạy với tham số KHÔNG? | Heatmap: thay đổi params → fitness thay đổi bao nhiêu? |

---

### 3.2 Biểu đồ hội tụ (Convergence Plot)

```
Fitness ↑
   100 |  ╲ HC (bị kẹt sớm)
       |   ╲___________________
       |    ╲
    50 |     ╲ DE (hội tụ dần)
       |      ╲
       |       ╲
     0 |________╲______________ → Generations
       0    25    50    75   100
```

- **Đường giảm nhanh rồi phẳng sớm**: Thuật toán hội tụ nhanh nhưng bị kẹt (HC)
- **Đường giảm đều**: Thuật toán khai thác dần dần (DE, PSO)
- **Band rộng** (Mean ± Std): Kết quả dao động giữa các runs → không ổn định

---

### 3.3 Boxplot so sánh chất lượng

```
       ┌───┐
   ╌╌╌╌│   │╌╌╌╌         ← whiskers
       │   │    
       ├───┤   ← median  
       │ ■ │   ← mean    
       │   │              
       └───┘              
      Alg A       Alg B
```

- **Hộp ngắn**: Kết quả tập trung → ổn định
- **Hộp dài**: Kết quả phân tán → không ổn định
- **Median thấp**: Thuật toán thường cho lời giải tốt
- **Outliers (dots)**: Trường hợp đặc biệt xấu/tốt

---

### 3.4 T-Test thống kê

Dùng **Welch's T-test** (two-sample) để kiểm tra: "Sự khác biệt giữa 2 thuật toán có **ý nghĩa thống kê** hay chỉ do ngẫu nhiên?"

```
t = (mean₁ - mean₂) / √(var₁/n₁ + var₂/n₂)
```

- **|t| > 2**: Sự khác biệt có ý nghĩa thống kê (p < 0.05)
- **|t| < 2**: Không đủ bằng chứng — hai thuật toán tương đương
- **t < 0**: Thuật toán 1 tốt hơn (mean nhỏ hơn = fitness thấp hơn)
- **t > 0**: Thuật toán 2 tốt hơn

---

### 3.5 Heatmap độ nhạy tham số

```
         CR=0.1   CR=0.5   CR=0.9
F=0.3  │ 45.2  │  12.3  │   8.1  │   ← Fitness trung bình
F=0.5  │ 32.1  │   5.6  │   3.2  │
F=0.9  │ 28.4  │   4.1  │   2.8  │   ← MÀU XANH ĐẬM = tốt nhất
```

- **Ô đồng màu**: Thuật toán KHÔNG nhạy với tham số → robust
- **Ô thay đổi mạnh**: Thuật toán RẤT NHẠY → cần tuning cẩn thận
- **Kết luận bổ ích**: "TLBO parameter-free nên không cần heatmap, luôn ổn định"

---

### 3.6 3D Trajectory (Quỹ đạo tìm kiếm)

Vẽ đường đi của nghiệm tốt nhất trên bề mặt hàm mục tiêu 2D:

- **HC**: Đường thẳng → kẹt tại local optima đầu tiên
- **DE**: Nhảy khắp nơi → exploration mạnh
- **CS**: Bước ngắn + nhảy dài (Levy flight) → thoát bẫy hiệu quả
- **PSO**: Bay theo đàn → nhanh nhưng dễ tụ sớm
- **SA**: Lúc đầu nhảy xa (T cao), sau co dần → cân bằng tốt

---

### 3.7 Tại sao chạy 30 lần?

> Các thuật toán metaheuristic đều **ngẫu nhiên** — mỗi lần chạy cho kết quả khác nhau. Chạy 30 lần để:
> 1. Ước lượng **mean & std** đáng tin cậy
> 2. Áp dụng **Central Limit Theorem** — trung bình 30 mẫu xấp xỉ phân phối chuẩn
> 3. Thực hiện **T-test** để kết luận có ý nghĩa thống kê

---

### 3.8 Fair Comparison — Công bằng số lần gọi hàm

Một vấn đề quan trọng: thuật toán population-based (DE, PSO, GA) gọi hàm mục tiêu `pop_size × generations` lần, còn single-solution (HC, SA) chỉ gọi `max_iter` lần.

**Cách giải quyết trong project**:
- HC/SA: `max_iter = pop_size × generations` → tổng số function evaluations BẰNG NHAU
- History được downsample để cùng trục hoành trên convergence plot

---

## PHẦN 4: TỔNG KẾT SO SÁNH

| Thuật toán | Loại | Exploration | Exploitation | Tham số | Bài toán tốt nhất |
|---|---|---|---|---|---|
| **BFS/DFS/A*** | Exact | N/A | N/A | 0 | Nhỏ (N ≤ 10) |
| **HC** | Local | ❌ Yếu | ✅ Mạnh | 1 (step_size) | Unimodal |
| **SA** | Physics | ✅ Tốt (T cao) | ✅ Tốt (T thấp) | 3 | Multimodal |
| **GA** | Evolution | ✅ Tốt | ✅ Tốt | 3 | Discrete (TSP, KP) |
| **DE** | Evolution | ✅ Mạnh | ✅ Mạnh | 2 (F, CR) | Continuous |
| **PSO** | Swarm | ⚠️ Trung bình | ✅ Nhanh | 3 (w, c1, c2) | Continuous |
| **ABC** | Swarm | ✅ Tốt (Scout) | ✅ Tốt | 2 | Large-scale |
| **FA** | Swarm | ✅ Mạnh (pairwise) | ⚠️ Chậm | 3 (α, β₀, γ) | Multimodal |
| **CS** | Swarm | ✅ Rất mạnh (Levy) | ⚠️ Trung bình | 3 (α, β, pa) | Exploration-heavy |
| **ACO** | Swarm | ✅ Tốt | ✅ Tốt (pheromone) | 4 | Discrete (TSP) |
| **TLBO** | Human | ✅ Tốt | ✅ Tốt | 0 ★ | All (parameter-free) |
