import numpy as np
from collections import deque
import heapq

# ==========================================
# PHẦN 1: SEARCH CHO TSP
# ==========================================
class TSPGraphSearch:
    """Các thuật toán tìm kiếm trên đồ thị cho bài toán TSP."""
    
    def __init__(self, num_cities, dist_matrix):
        self.num_cities = num_cities
        self.dist_matrix = dist_matrix

    def bfs(self):
        """BFS — duyệt theo tầng, dùng Queue (FIFO). Optimal nhưng chậm O(N!)."""
        queue = deque([([0], 0.0)])
        best_cost = float('inf')
        
        while queue:
            path, cost = queue.popleft()
            if cost >= best_cost: continue

            if len(path) == self.num_cities:
                total = cost + self.dist_matrix[path[-1]][0]
                if total < best_cost: best_cost = total
                continue

            last = path[-1]
            for next_city in range(self.num_cities):
                if next_city not in path:
                    queue.append((path + [next_city], cost + self.dist_matrix[last][next_city]))
        return best_cost

    def dfs(self):
        """DFS — duyệt theo chiều sâu, dùng Stack (LIFO). Không đảm bảo optimal."""
        stack = [([0], 0.0)]
        best_cost = float('inf')
        
        while stack:
            path, cost = stack.pop()
            if len(path) == self.num_cities:
                total = cost + self.dist_matrix[path[-1]][0]
                if total < best_cost: best_cost = total
                continue

            last = path[-1]
            for next_city in range(self.num_cities - 1, 0, -1):
                if next_city not in path:
                    stack.append((path + [next_city], cost + self.dist_matrix[last][next_city]))
        return best_cost

    def a_star(self):
        """A* — f(n) = g(n) + h(n). Heuristic: h = (TP còn lại) × (cạnh ngắn nhất)."""
        min_edge = np.min(self.dist_matrix[self.dist_matrix > 0])
        pq = [(0, 0, [0])]  # (f, g, path)
        best_cost = float('inf')

        while pq:
            f, g, path = heapq.heappop(pq)
            if g >= best_cost: continue

            if len(path) == self.num_cities:
                total = g + self.dist_matrix[path[-1]][0]
                if total < best_cost: best_cost = total
                continue

            last = path[-1]
            h_rem = (self.num_cities - len(path)) * min_edge
            for next_city in range(self.num_cities):
                if next_city not in path:
                    new_g = g + self.dist_matrix[last][next_city]
                    heapq.heappush(pq, (new_g + h_rem, new_g, path + [next_city]))
        return best_cost

    def ucs(self):
        """UCS — ưu tiên đường chi phí thấp nhất. Optimal (giống A* khi h=0)."""
        pq = [(0.0, [0])]
        best_cost = float('inf')

        while pq:
            cost, path = heapq.heappop(pq)
            if cost >= best_cost: continue

            if len(path) == self.num_cities:
                total = cost + self.dist_matrix[path[-1]][0]
                if total < best_cost: best_cost = total
                continue

            last = path[-1]
            for next_city in range(self.num_cities):
                if next_city not in path:
                    new_cost = cost + self.dist_matrix[last][next_city]
                    heapq.heappush(pq, (new_cost, path + [next_city]))
        return best_cost

    def greedy_best_first(self):
        """Greedy BFS — luôn chọn TP gần nhất (Nearest Neighbor). Nhanh O(N²), không optimal."""
        best_cost = float('inf')

        for start in range(self.num_cities):
            visited = {start}
            cost = 0
            current = start
            
            while len(visited) < self.num_cities:
                min_dist = float('inf')
                next_city = -1
                for j in range(self.num_cities):
                    if j not in visited and self.dist_matrix[current][j] < min_dist:
                        min_dist = self.dist_matrix[current][j]
                        next_city = j
                visited.add(next_city)
                cost += min_dist
                current = next_city

            cost += self.dist_matrix[current][start]
            if cost < best_cost:
                best_cost = cost

        return best_cost

# ==========================================
# PHẦN 2: SEARCH CHO CONTINUOUS OPTIMIZATION
# ==========================================
class ContinuousLocalSearch:
    """Hill Climbing + Simulated Annealing cho tối ưu hàm liên tục."""
    
    def __init__(self, step_size=0.1, max_iter=1000):
        self.step_size = step_size
        self.max_iter = max_iter

    def hill_climbing(self, func, bounds):
        """HC — chỉ chấp nhận lời giải tốt hơn (greedy). Dễ kẹt local optima."""
        bounds = np.array(bounds)
        dim = len(bounds)
        curr_pos = bounds[:, 0] + np.random.rand(dim) * (bounds[:, 1] - bounds[:, 0])
        curr_score = func(curr_pos)
        
        history = [curr_score]
        path = [curr_pos.copy()]

        for _ in range(self.max_iter):
            neighbor = curr_pos + np.random.normal(0, self.step_size, dim)
            neighbor = np.clip(neighbor, bounds[:, 0], bounds[:, 1])
            n_score = func(neighbor)
            
            if n_score < curr_score:
                curr_pos = neighbor
                curr_score = n_score
            
            history.append(curr_score)
            path.append(curr_pos.copy())
            
        return curr_pos, curr_score, history, path

    def simulated_annealing(self, func, bounds, T_init=100.0, T_min=1e-8, cooling_rate=0.995):
        """SA — chấp nhận lời giải tệ hơn với xác suất exp(-Δ/T). Thoát được local optima."""
        bounds = np.array(bounds)
        dim = len(bounds)
        
        curr_pos = bounds[:, 0] + np.random.rand(dim) * (bounds[:, 1] - bounds[:, 0])
        curr_score = func(curr_pos)
        best_pos = curr_pos.copy()
        best_score = curr_score
        
        history = [best_score]
        path = [best_pos.copy()]
        
        T = T_init
        iteration = 0
        
        while T > T_min and iteration < self.max_iter:
            # Step size tỉ lệ với nhiệt độ: ban đầu bước lớn, sau bước nhỏ
            adaptive_step = self.step_size * (T / T_init)
            neighbor = curr_pos + np.random.normal(0, adaptive_step, dim)
            neighbor = np.clip(neighbor, bounds[:, 0], bounds[:, 1])
            n_score = func(neighbor)
            
            delta = n_score - curr_score
            
            # Chấp nhận nếu tốt hơn HOẶC theo xác suất Boltzmann
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                curr_pos = neighbor
                curr_score = n_score
                
                if curr_score < best_score:
                    best_score = curr_score
                    best_pos = curr_pos.copy()
            
            history.append(best_score)
            path.append(best_pos.copy())
            
            T *= cooling_rate
            iteration += 1
        
        return best_pos, best_score, history, path
