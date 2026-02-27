import numpy as np

class AntColonyOptimizationGC:
    """ACO chuyên biệt cho bài toán Graph Coloring."""
    
    def __init__(self, adj_matrix, num_colors, num_ants=30, alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=10.0):
        self.adj_matrix = adj_matrix
        self.num_colors = num_colors
        self.num_ants = num_ants
        self.num_nodes = len(adj_matrix)
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        
        # Pheromone được rải trên cặp (Đỉnh, Màu)
        self.pheromone = np.ones((self.num_nodes, self.num_colors)) * 0.1
        
    def _count_conflicts(self, coloring):
        conflicts = 0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.adj_matrix[i][j] == 1 and coloring[i] == coloring[j]:
                    conflicts += 1
        return conflicts
        
    def _construct_solution(self):
        coloring = np.full(self.num_nodes, -1, dtype=int)
        # Kiến đi qua từng đỉnh ngẫu nhiên
        nodes_order = np.random.permutation(self.num_nodes)
        
        for node in nodes_order:
            probs = np.zeros(self.num_colors)
            for c in range(self.num_colors):
                # Heuristic: Số hàng xóm đã dùng màu này (càng ít càng tốt)
                conflicts = sum(1 for neighbor in range(self.num_nodes) 
                              if self.adj_matrix[node][neighbor] == 1 and coloring[neighbor] == c)
                eta = 1.0 / (1.0 + conflicts)
                tau = self.pheromone[node][c]
                probs[c] = (tau ** self.alpha) * (eta ** self.beta)
                
            if np.sum(probs) == 0:
                probs = np.ones(self.num_colors)
            probs /= np.sum(probs)
            
            chosen_color = np.random.choice(self.num_colors, p=probs)
            coloring[node] = chosen_color
            
        return coloring
        
    def solve(self, iterations=100):
        best_coloring = None
        best_conflicts = float('inf')
        history = []
        
        for _ in range(iterations):
            all_colorings = []
            all_conflicts = []
            
            for _ in range(self.num_ants):
                col = self._construct_solution()
                conf = self._count_conflicts(col)
                all_colorings.append(col)
                all_conflicts.append(conf)
                
                if conf < best_conflicts:
                    best_conflicts = conf
                    best_coloring = col.copy()
                    
            # Cập nhật pheromone
            self.pheromone *= (1 - self.evaporation_rate)
            for col, conf in zip(all_colorings, all_conflicts):
                deposit = self.Q / (1.0 + conf) # Xung đột ít -> Thưởng nhiều
                for node in range(self.num_nodes):
                    self.pheromone[node][col[node]] += deposit
                    
            history.append(best_conflicts)
            if best_conflicts == 0: # Dừng sớm nếu tối ưu
                history.extend([0] * (iterations - len(history)))
                break
                
        return best_coloring, best_conflicts, history