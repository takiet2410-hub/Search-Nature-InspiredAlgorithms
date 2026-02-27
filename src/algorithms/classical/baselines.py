import numpy as np
from collections import deque
import heapq

# ==========================================
# PHẦN 1: SEARCH CHO TSP (BFS, DFS, A*)
# ==========================================
class TSPGraphSearch:
    def __init__(self, num_cities, dist_matrix):
        self.num_cities = num_cities
        self.dist_matrix = dist_matrix

    def bfs(self):
        """Breadth-First Search cho TSP"""
        queue = deque([([0], 0.0)]) # (path, cost)
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
        """Depth-First Search cho TSP"""
        stack = [([0], 0.0)]
        best_cost = float('inf')
        
        while stack:
            path, cost = stack.pop()
            if len(path) == self.num_cities:
                total = cost + self.dist_matrix[path[-1]][0]
                if total < best_cost: best_cost = total
                continue

            last = path[-1]
            # Duyệt ngược để thứ tự pop ra là xuôi
            for next_city in range(self.num_cities - 1, 0, -1):
                if next_city not in path:
                    stack.append((path + [next_city], cost + self.dist_matrix[last][next_city]))
        return best_cost

    def a_star(self):
        """A* Search cho TSP"""
        min_edge = np.min(self.dist_matrix[self.dist_matrix > 0])
        pq = [(0, 0, [0])] # (F, G, path)
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
    def hill_climbing(self, max_iter=5000):
        """Hill Climbing (Local Search) cho TSP dùng 2-opt Swap"""
        # Khởi tạo lộ trình ngẫu nhiên
        curr_route = list(np.random.permutation(self.num_cities))
        
        def calc_cost(route):
            cost = 0
            for i in range(self.num_cities - 1):
                cost += self.dist_matrix[route[i]][route[i+1]]
            cost += self.dist_matrix[route[-1]][route[0]]
            return cost
            
        curr_cost = calc_cost(curr_route)
        best_route = curr_route.copy()
        best_cost = curr_cost
        history = [best_cost]

        for _ in range(max_iter):
            # Chọn 2 điểm cắt ngẫu nhiên (2-opt swap)
            i, j = sorted(np.random.choice(self.num_cities, 2, replace=False))
            neighbor_route = curr_route.copy()
            neighbor_route[i:j+1] = reversed(neighbor_route[i:j+1])
            
            neighbor_cost = calc_cost(neighbor_route)
            
            if neighbor_cost <= curr_cost: # Greedy
                curr_route = neighbor_route
                curr_cost = neighbor_cost
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_route = curr_route.copy()
                    
            history.append(best_cost)
            
        return best_route, best_cost, history
# ==========================================
# PHẦN 2: SEARCH CHO CONTINUOUS (Hill Climbing)
# ==========================================
class ContinuousLocalSearch:
    def __init__(self, step_size=0.1, max_iter=1000):
        self.step_size = step_size
        self.max_iter = max_iter

    def hill_climbing(self, func, bounds):
        """Hill Climbing cho hàm số liên tục"""
        bounds = np.array(bounds)
        dim = len(bounds)
        # Random khởi tạo
        curr_pos = bounds[:, 0] + np.random.rand(dim) * (bounds[:, 1] - bounds[:, 0])
        curr_score = func(curr_pos)
        
        history = [curr_score]
        path = [curr_pos.copy()] # <--- MỚI: Lưu điểm xuất phát

        for _ in range(self.max_iter):
            # Sinh hàng xóm (Gaussian noise)
            neighbor = curr_pos + np.random.normal(0, self.step_size, dim)
            neighbor = np.clip(neighbor, bounds[:, 0], bounds[:, 1])
            n_score = func(neighbor)
            
            # Chỉ nhận nếu tốt hơn (Leo đồi - Greedy)
            if n_score < curr_score:
                curr_pos = neighbor
                curr_score = n_score
            
            history.append(curr_score)
            path.append(curr_pos.copy()) # <--- MỚI: Lưu toạ độ mới vào đường đi
            
        # Trả về 4 giá trị: Vị trí cuối, Điểm số cuối, Lịch sử điểm số, Lịch sử đường đi
        return curr_pos, curr_score, history, path