import numpy as np
import random

class GeneticAlgorithmTSP:
    def __init__(self, num_cities, dist_matrix, pop_size=50, mutation_rate=0.1, elitism_rate=0.1):
        self.num_cities = num_cities
        self.dist_matrix = dist_matrix
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elitism_count = int(pop_size * elitism_rate) # Số lượng giữ lại (Elitism)

    def create_individual(self):
        """Tạo 1 cá thể là hoán vị ngẫu nhiên các thành phố"""
        return np.random.permutation(self.num_cities)

    def calculate_distance(self, individual):
        """Tính tổng quãng đường (Fitness nghịch đảo của cái này)"""
        total_dist = 0
        for i in range(self.num_cities - 1):
            u, v = individual[i], individual[i+1]
            total_dist += self.dist_matrix[u][v]
        # Cộng đoạn quay về điểm đầu
        total_dist += self.dist_matrix[individual[-1]][individual[0]]
        return total_dist

    def selection_tournament(self, population, scores, k=3):
        """Chọn lọc giải đấu"""
        # Chọn ngẫu nhiên k chỉ số
        selection_ix = np.random.randint(len(population), size=k)
        # Lấy cá thể có score (distance) nhỏ nhất trong nhóm k người đó
        best_ix = selection_ix[0]
        for ix in selection_ix[1:]:
            if scores[ix] < scores[best_ix]:
                best_ix = ix
        return population[best_ix]

    def crossover_ox(self, parent1, parent2):
        """Lai ghép Order Crossover (OX) - Quan trọng cho TSP"""
        size = self.num_cities
        p1, p2 = parent1.copy(), parent2.copy()
        
        # 1. Chọn 2 điểm cắt ngẫu nhiên
        cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
        
        # 2. Tạo con điền sẵn -1
        child = np.full(size, -1)
        
        # 3. Copy đoạn giữa của Parent 1 vào Con
        child[cxpoint1:cxpoint2+1] = p1[cxpoint1:cxpoint2+1]
        
        # 4. Điền các số còn thiếu theo thứ tự của Parent 2
        p2_pos = (cxpoint2 + 1) % size
        child_pos = (cxpoint2 + 1) % size
        
        while -1 in child:
            gene = p2[p2_pos]
            if gene not in child:
                child[child_pos] = gene
                child_pos = (child_pos + 1) % size
            p2_pos = (p2_pos + 1) % size
            
        return child

    def mutate_swap(self, individual):
        """Đột biến: Hoán đổi vị trí 2 thành phố"""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(self.num_cities), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def solve(self, generations=100):
        """Hàm chạy chính"""
        # 1. Khởi tạo quần thể
        population = [self.create_individual() for _ in range(self.pop_size)]
        history = [] # Lưu lại quãng đường tốt nhất mỗi thế hệ để vẽ biểu đồ

        for gen in range(generations):
            # Tính quãng đường cho cả quần thể
            scores = [self.calculate_distance(ind) for ind in population]
            
            # Lưu kết quả tốt nhất hiện tại
            best_score = min(scores)
            history.append(best_score)
            
            # Sắp xếp quần thể theo quãng đường tăng dần (ngắn nhất lên đầu)
            sorted_indices = np.argsort(scores)
            population = [population[i] for i in sorted_indices]
            
            # --- ELITISM: Giữ lại những người giỏi nhất ---
            new_population = population[:self.elitism_count]
            
            # --- Tạo thế hệ mới ---
            while len(new_population) < self.pop_size:
                # Selection
                p1 = self.selection_tournament(population, scores)
                p2 = self.selection_tournament(population, scores)
                
                # Crossover
                child = self.crossover_ox(p1, p2)
                
                # Mutation
                child = self.mutate_swap(child)
                
                new_population.append(child)
            
            population = new_population
            
        # Trả về lộ trình tốt nhất và lịch sử
        final_scores = [self.calculate_distance(ind) for ind in population]
        best_idx = np.argmin(final_scores)
        return population[best_idx], final_scores[best_idx], history