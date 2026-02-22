import numpy as np

class AntColonyOptimizationTSP:
    """ACO cho bài toán TSP — mô phỏng hành vi đàn kiến tìm đường ngắn nhất."""
    
    def __init__(self, num_cities, dist_matrix, num_ants=30, alpha=1.0, beta=3.0, 
                 evaporation_rate=0.5, Q=100.0):
        self.num_cities = num_cities
        self.dist_matrix = dist_matrix
        self.num_ants = num_ants
        self.alpha = alpha              # Trọng số pheromone
        self.beta = beta                # Trọng số heuristic (1/distance)
        self.evaporation_rate = evaporation_rate
        self.Q = Q

        # Ma trận pheromone — ban đầu đều nhau
        self.pheromone = np.ones((num_cities, num_cities)) * 0.1

        # Ma trận heuristic: η(i,j) = 1/d(i,j)
        self.heuristic = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j and dist_matrix[i][j] > 0:
                    self.heuristic[i][j] = 1.0 / dist_matrix[i][j]

    def _select_next_city(self, current_city, visited):
        """Chọn thành phố tiếp theo — P(i→j) = τ^α × η^β / Σ (Roulette Wheel)"""
        pheromone = self.pheromone[current_city]
        heuristic = self.heuristic[current_city]

        probabilities = np.zeros(self.num_cities)
        for j in range(self.num_cities):
            if j not in visited:
                probabilities[j] = (pheromone[j] ** self.alpha) * (heuristic[j] ** self.beta)

        total = np.sum(probabilities)
        if total == 0:
            unvisited = [j for j in range(self.num_cities) if j not in visited]
            return np.random.choice(unvisited)
        
        probabilities /= total
        return np.random.choice(self.num_cities, p=probabilities)

    def _construct_route(self):
        """Một con kiến xây dựng lộ trình hoàn chỉnh qua tất cả thành phố."""
        start = np.random.randint(0, self.num_cities)
        route = [start]
        visited = set(route)

        for _ in range(self.num_cities - 1):
            next_city = self._select_next_city(route[-1], visited)
            route.append(next_city)
            visited.add(next_city)

        return route

    def _calculate_route_cost(self, route):
        """Tính tổng quãng đường (khép kín — quay về thành phố đầu)."""
        cost = 0
        for i in range(len(route) - 1):
            cost += self.dist_matrix[route[i]][route[i + 1]]
        cost += self.dist_matrix[route[-1]][route[0]]
        return cost

    def _update_pheromone(self, all_routes, all_costs):
        """Bay hơi pheromone cũ + thêm pheromone mới (kiến đi ngắn → deposit nhiều)."""
        self.pheromone *= (1 - self.evaporation_rate)

        for route, cost in zip(all_routes, all_costs):
            deposit = self.Q / cost
            for i in range(len(route) - 1):
                self.pheromone[route[i]][route[i + 1]] += deposit
                self.pheromone[route[i + 1]][route[i]] += deposit
            self.pheromone[route[-1]][route[0]] += deposit
            self.pheromone[route[0]][route[-1]] += deposit

    def solve(self, iterations=100):
        """
        Chạy ACO.
        Returns: (best_route, best_cost, history)
        """
        best_route = None
        best_cost = float('inf')
        history = []

        for it in range(iterations):
            all_routes = []
            all_costs = []

            for _ in range(self.num_ants):
                route = self._construct_route()
                cost = self._calculate_route_cost(route)
                all_routes.append(route)
                all_costs.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_route = route.copy()

            self._update_pheromone(all_routes, all_costs)
            history.append(best_cost)

        return best_route, best_cost, history
