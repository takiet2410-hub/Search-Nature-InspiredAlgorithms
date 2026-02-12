import numpy as np

class DifferentialEvolution:
    def __init__(self, func, bounds, pop_size=50, mutation_factor=0.8, crossover_rate=0.7):
        self.func = func             
        self.bounds = np.array(bounds) 
        self.pop_size = pop_size
        self.F = mutation_factor     
        self.CR = crossover_rate     
        self.dim = len(bounds)       

    def optimize(self, generations=100):
        min_b, max_b = self.bounds[:, 0], self.bounds[:, 1]
        population = min_b + np.random.rand(self.pop_size, self.dim) * (max_b - min_b)
        
        history = [] 
        trajectory = [] # <--- MỚI: Lưu toạ độ best mỗi gen

        # Tính fitness ban đầu
        fitness_scores = np.array([self.func(ind) for ind in population])
        best_idx = np.argmin(fitness_scores)
        history.append(fitness_scores[best_idx])
        trajectory.append(population[best_idx].copy())

        for gen in range(generations):
            new_population = []
            new_fitness = []
            
            for i in range(self.pop_size):
                target_vector = population[i]
                
                # Mutation
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                
                x_r1, x_r2, x_r3 = population[r1], population[r2], population[r3]
                mutant_vector = x_r1 + self.F * (x_r2 - x_r3)
                mutant_vector = np.clip(mutant_vector, min_b, max_b)
                
                # Crossover
                trial_vector = np.copy(target_vector)
                cross_points = np.random.rand(self.dim) < self.CR
                k = np.random.randint(0, self.dim)
                cross_points[k] = True
                trial_vector[cross_points] = mutant_vector[cross_points]
                
                # Selection
                f_target = fitness_scores[i]
                f_trial = self.func(trial_vector)
                
                if f_trial <= f_target:
                    new_population.append(trial_vector)
                    new_fitness.append(f_trial)
                else:
                    new_population.append(target_vector)
                    new_fitness.append(f_target)
            
            population = np.array(new_population)
            fitness_scores = np.array(new_fitness)
            
            # Ghi lại kết quả tốt nhất
            best_gen_idx = np.argmin(fitness_scores)
            history.append(fitness_scores[best_gen_idx])
            trajectory.append(population[best_gen_idx].copy()) # <--- Lưu toạ độ
            
        best_idx = np.argmin(fitness_scores)
        # Trả về thêm trajectory
        return population[best_idx], fitness_scores[best_idx], history, trajectory