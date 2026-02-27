import numpy as np

class AntColonyOptimizationSP:
    """ACO chuyên biệt cho bài toán Shortest Path (Đường đi có độ dài biến đổi)."""
    
    def __init__(self, adj_matrix, start, goal, num_ants=30, alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=100.0):
        self.adj_matrix = adj_matrix
        self.start = start
        self.goal = goal
        self.num_ants = num_ants
        self.num_nodes = len(adj_matrix)
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        
        self.pheromone = np.ones((self.num_nodes, self.num_nodes)) * 0.1
        
    def _construct_path(self):
        path = [self.start]
        visited = {self.start}
        current = self.start
        explored = 1
        
        while current != self.goal:
            # Lấy danh sách hàng xóm hợp lệ
            neighbors = [v for v in range(self.num_nodes) 
                         if self.adj_matrix[current][v] > 0 and v not in visited]
            
            if not neighbors: # Ngõ cụt kiến chết
                return None, float('inf'), explored
                
            probs = np.zeros(len(neighbors))
            for i, v in enumerate(neighbors):
                tau = self.pheromone[current][v]
                eta = 1.0 / self.adj_matrix[current][v]
                probs[i] = (tau ** self.alpha) * (eta ** self.beta)
                
            if np.sum(probs) == 0:
                probs = np.ones(len(neighbors))
            probs /= np.sum(probs)
            
            next_node = np.random.choice(neighbors, p=probs)
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            explored += 1
            
        # Tính cost
        cost = 0
        for i in range(len(path) - 1):
            cost += self.adj_matrix[path[i]][path[i+1]]
        return path, cost, explored
        
    def solve(self, iterations=50):
        best_path = None
        best_cost = float('inf')
        total_explored = 0
        history = [] 
        for _ in range(iterations):
            valid_paths = []
            valid_costs = []
            
            for _ in range(self.num_ants):
                path, cost, exp = self._construct_path()
                total_explored += exp
                if path is not None:
                    valid_paths.append(path)
                    valid_costs.append(cost)
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path
                history.append(best_cost)
            # Cập nhật pheromone
            self.pheromone *= (1 - self.evaporation_rate)
            for path, cost in zip(valid_paths, valid_costs):
                deposit = self.Q / cost
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    self.pheromone[u][v] += deposit
                    self.pheromone[v][u] += deposit
                    
        return best_path, best_cost, total_explored // (iterations * self.num_ants), history # Trung bình explored