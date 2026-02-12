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

        for _ in range(self.max_iter):
            # Sinh hàng xóm (Gaussian noise)
            neighbor = curr_pos + np.random.normal(0, self.step_size, dim)
            neighbor = np.clip(neighbor, bounds[:, 0], bounds[:, 1])
            n_score = func(neighbor)
            
            # Chỉ nhận nếu tốt hơn (Leo đồi)
            if n_score < curr_score:
                curr_pos = neighbor
                curr_score = n_score
            
            history.append(curr_score)
        return curr_pos, curr_score, history