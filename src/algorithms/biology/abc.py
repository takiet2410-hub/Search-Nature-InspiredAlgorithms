import numpy as np

class ArtificialBeeColony:
    """ABC cho tối ưu hàm liên tục — mô phỏng 3 pha: Employed, Onlooker, Scout."""
    
    def __init__(self, func, bounds, colony_size=30, limit=None):
        self.func = func
        self.bounds = np.array(bounds)
        self.colony_size = colony_size
        self.num_employed = colony_size // 2
        self.dim = len(bounds)
        self.limit = limit if limit is not None else self.num_employed * self.dim

    def _initialize_food_source(self):
        """Tạo nguồn thức ăn ngẫu nhiên trong bounds."""
        min_b, max_b = self.bounds[:, 0], self.bounds[:, 1]
        return min_b + np.random.rand(self.dim) * (max_b - min_b)

    def _calculate_fitness(self, obj_value):
        """Chuyển objective (nhỏ=tốt) → fitness (lớn=tốt) cho Roulette Wheel."""
        if obj_value >= 0:
            return 1.0 / (1.0 + obj_value)
        else:
            return 1.0 + abs(obj_value)

    def _search_neighbor(self, food_sources, current_idx):
        """Tìm lân cận: v_ij = x_ij + φ*(x_ij - x_kj), thay đổi 1 chiều ngẫu nhiên."""
        current = food_sources[current_idx].copy()
        d = np.random.randint(0, self.dim)
        
        candidates = list(range(self.num_employed))
        candidates.remove(current_idx)
        k = np.random.choice(candidates)
        
        phi = np.random.uniform(-1, 1)
        new_source = current.copy()
        new_source[d] = current[d] + phi * (current[d] - food_sources[k][d])
        new_source = np.clip(new_source, self.bounds[:, 0], self.bounds[:, 1])
        
        return new_source

    def optimize(self, iterations=100):
        """
        Chạy ABC.
        Returns: (best_pos, best_score, history, trajectory)
        """
        # Khởi tạo nguồn thức ăn
        food_sources = np.array([self._initialize_food_source() for _ in range(self.num_employed)])
        obj_values = np.array([self.func(fs) for fs in food_sources])
        fitness_values = np.array([self._calculate_fitness(ov) for ov in obj_values])
        trial_counters = np.zeros(self.num_employed)

        best_idx = np.argmin(obj_values)
        best_pos = food_sources[best_idx].copy()
        best_score = obj_values[best_idx]

        history = [best_score]
        trajectory = [best_pos.copy()]

        for it in range(iterations):
            # Pha 1: Employed Bees — tìm kiếm lân cận
            for i in range(self.num_employed):
                new_source = self._search_neighbor(food_sources, i)
                new_obj = self.func(new_source)
                new_fit = self._calculate_fitness(new_obj)

                if new_fit > fitness_values[i]:
                    food_sources[i] = new_source
                    obj_values[i] = new_obj
                    fitness_values[i] = new_fit
                    trial_counters[i] = 0
                else:
                    trial_counters[i] += 1

            # Pha 2: Onlooker Bees — chọn nguồn tốt theo xác suất
            total_fitness = np.sum(fitness_values)
            if total_fitness > 0:
                probabilities = fitness_values / total_fitness
            else:
                probabilities = np.ones(self.num_employed) / self.num_employed

            for _ in range(self.num_employed):
                selected = np.random.choice(self.num_employed, p=probabilities)
                new_source = self._search_neighbor(food_sources, selected)
                new_obj = self.func(new_source)
                new_fit = self._calculate_fitness(new_obj)

                if new_fit > fitness_values[selected]:
                    food_sources[selected] = new_source
                    obj_values[selected] = new_obj
                    fitness_values[selected] = new_fit
                    trial_counters[selected] = 0
                else:
                    trial_counters[selected] += 1

            # Pha 3: Scout Bees — bỏ nguồn kém, khám phá ngẫu nhiên
            for i in range(self.num_employed):
                if trial_counters[i] > self.limit:
                    food_sources[i] = self._initialize_food_source()
                    obj_values[i] = self.func(food_sources[i])
                    fitness_values[i] = self._calculate_fitness(obj_values[i])
                    trial_counters[i] = 0

            # Cập nhật global best
            current_best_idx = np.argmin(obj_values)
            if obj_values[current_best_idx] < best_score:
                best_score = obj_values[current_best_idx]
                best_pos = food_sources[current_best_idx].copy()

            history.append(best_score)
            trajectory.append(best_pos.copy())

        return best_pos, best_score, history, trajectory
