import numpy as np

class ParticleSwarmOptimization:
    """PSO cho tối ưu hàm liên tục — mô phỏng hành vi đàn chim tìm thức ăn."""
    
    def __init__(self, func, bounds, num_particles=30, w=0.7, c1=1.5, c2=1.5):
        self.func = func
        self.bounds = np.array(bounds)
        self.num_particles = num_particles
        self.w = w      # Inertia weight
        self.c1 = c1    # Cognitive coefficient (hướng về pbest)
        self.c2 = c2    # Social coefficient (hướng về gbest)
        self.dim = len(bounds)

    def optimize(self, iterations=100):
        """
        Chạy PSO.
        Returns: (gbest_pos, gbest_score, history, trajectory)
        """
        min_b, max_b = self.bounds[:, 0], self.bounds[:, 1]

        # Khởi tạo vị trí + vận tốc ngẫu nhiên
        positions = min_b + np.random.rand(self.num_particles, self.dim) * (max_b - min_b)
        v_max = (max_b - min_b) * 0.2
        velocities = np.random.uniform(-v_max, v_max, (self.num_particles, self.dim))

        fitness = np.array([self.func(p) for p in positions])

        # Personal best + Global best
        pbest_pos = positions.copy()
        pbest_score = fitness.copy()
        gbest_idx = np.argmin(fitness)
        gbest_pos = positions[gbest_idx].copy()
        gbest_score = fitness[gbest_idx]

        history = [gbest_score]
        trajectory = [gbest_pos.copy()]

        for it in range(iterations):
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
                cognitive = self.c1 * r1 * (pbest_pos[i] - positions[i])
                social = self.c2 * r2 * (gbest_pos - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive + social
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # x = x + v
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], min_b, max_b)

                new_fitness = self.func(positions[i])
                fitness[i] = new_fitness

                if new_fitness < pbest_score[i]:
                    pbest_score[i] = new_fitness
                    pbest_pos[i] = positions[i].copy()

                if new_fitness < gbest_score:
                    gbest_score = new_fitness
                    gbest_pos = positions[i].copy()

            history.append(gbest_score)
            trajectory.append(gbest_pos.copy())

        return gbest_pos, gbest_score, history, trajectory
