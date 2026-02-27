import numpy as np


class GeneticAlgorithmKnapsack:
    """
    GA nhị phân cho bài toán 0/1 Knapsack.

    Mỗi cá thể là mảng nhị phân có chiều dài = số vật phẩm.
    Gene = 1  → chọn vật phẩm, Gene = 0 → không chọn.

    Fitness = tổng giá trị nếu tổng trọng lượng ≤ capacity,
              ngược lại sửa nghiệm bằng cách loại vật phẩm nặng nhất (repair).

    Returns: (best_individual, best_value, history)
    """

    def __init__(self, weights, values, capacity,
                 pop_size=50, mutation_rate=0.05, elitism_rate=0.1):
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.num_items = len(weights)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elitism_count = max(1, int(pop_size * elitism_rate))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _random_individual(self):
        """Tạo cá thể nhị phân ngẫu nhiên."""
        return np.random.randint(0, 2, size=self.num_items)

    def _repair(self, individual):
        """Sửa nghiệm vi phạm: loại vật phẩm có value/weight thấp nhất."""
        ind = individual.copy()
        while np.dot(ind, self.weights) > self.capacity:
            selected = np.where(ind == 1)[0]
            if len(selected) == 0:
                break
            # Tỉ lệ value/weight — loại thấp nhất
            ratios = self.values[selected] / (self.weights[selected] + 1e-12)
            worst = selected[np.argmin(ratios)]
            ind[worst] = 0
        return ind

    def _fitness(self, individual):
        """Tính tổng giá trị (sau repair)."""
        ind = self._repair(individual)
        return float(np.dot(ind, self.values)), ind

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------
    def _tournament_selection(self, population, fitnesses, k=3):
        idxs = np.random.randint(0, len(population), size=k)
        best = idxs[np.argmax(fitnesses[idxs])]
        return population[best].copy()

    def _uniform_crossover(self, p1, p2):
        mask = np.random.randint(0, 2, size=self.num_items)
        child = np.where(mask, p1, p2)
        return child

    def _mutate(self, individual):
        for i in range(self.num_items):
            if np.random.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def solve(self, generations=100):
        # Khởi tạo quần thể
        population = np.array([self._random_individual() for _ in range(self.pop_size)])

        # Evaluate + repair
        fitnesses = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            fitnesses[i], population[i] = self._fitness(population[i])

        best_idx = np.argmax(fitnesses)
        best_ind = population[best_idx].copy()
        best_val = fitnesses[best_idx]
        history = [best_val]

        for gen in range(generations):
            # Sắp xếp giảm dần theo fitness
            order = np.argsort(-fitnesses)
            population = population[order]
            fitnesses = fitnesses[order]

            # Elitism
            new_pop = [population[i].copy() for i in range(self.elitism_count)]

            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)
                child = self._uniform_crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)

            population = np.array(new_pop)

            # Evaluate + repair
            for i in range(self.pop_size):
                fitnesses[i], population[i] = self._fitness(population[i])

            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_val:
                best_val = fitnesses[gen_best_idx]
                best_ind = population[gen_best_idx].copy()

            history.append(best_val)

        return best_ind, best_val, history
