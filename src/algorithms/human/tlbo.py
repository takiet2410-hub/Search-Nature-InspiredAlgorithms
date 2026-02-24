import numpy as np

class TLBO:
    """
    Teaching-Learning-Based Optimization (TLBO) for continuous optimization.

    A parameter-free population-based algorithm with two phases per iteration:
      - Teacher Phase : The best solution (teacher) pulls the population mean
                        toward the global optimum.
      - Learner Phase : Each learner randomly interacts with a peer and moves
                        toward the better one.

    Return format is aligned with DE / CS / HC in this workspace:
        (best_position, best_score, history, trajectory)
    """

    def __init__(self, func, bounds, pop_size: int = 50):
        """
        Args:
            func     : Objective function to *minimise*.
            bounds   : List of [min, max] pairs, one per dimension.
            pop_size : Number of learners in the class (≥ 2).
        """
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.pop_size = max(2, pop_size)
        self.dim = len(bounds)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        return np.array([self.func(ind) for ind in population])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def optimize(self, iterations: int = 100):
        """
        Run TLBO for the given number of iterations.

        Args:
            iterations : Number of teaching-learning cycles.

        Returns:
            best_position : ndarray – coordinates of the best solution found.
            best_score    : float   – objective value at best_position.
            history       : list[float] – best score recorded after every iteration
                            (length = iterations + 1, index 0 is the initial best).
            trajectory    : list[ndarray] – best_position snapshot at each history
                            point (same length as history).
        """
        min_b, max_b = self.bounds[:, 0], self.bounds[:, 1]

        # ── Initialise population ──────────────────────────────────────
        population = min_b + np.random.rand(self.pop_size, self.dim) * (max_b - min_b)
        fitness = self._evaluate(population)

        best_idx = int(np.argmin(fitness))
        best_pos = population[best_idx].copy()
        best_score = float(fitness[best_idx])

        history: list = [best_score]
        trajectory: list = [best_pos.copy()]

        # ── Main loop ─────────────────────────────────────────────────
        for _ in range(iterations):

            # ── TEACHER PHASE ─────────────────────────────────────────
            teacher_idx = int(np.argmin(fitness))
            teacher = population[teacher_idx]
            mean = population.mean(axis=0)

            # Teaching factor T_F ∈ {1, 2}  (randomly chosen each iteration)
            T_F = np.random.randint(1, 3)  # 1 or 2

            # r  is a uniform random vector in [0, 1]^dim
            r = np.random.rand(self.dim)

            # candidate_i = x_i + r * (teacher - T_F * mean)
            diff = teacher - T_F * mean
            new_population = self._clip(population + r * diff)
            new_fitness = self._evaluate(new_population)

            # Greedy selection: keep whichever is better
            improved = new_fitness < fitness
            population[improved] = new_population[improved]
            fitness[improved] = new_fitness[improved]

            # ── LEARNER PHASE ─────────────────────────────────────────
            indices = np.arange(self.pop_size)
            for i in range(self.pop_size):
                # Pick a random peer different from i
                j = np.random.choice(indices[indices != i])

                r = np.random.rand(self.dim)

                if fitness[i] < fitness[j]:
                    # Learner i is better → move away from j
                    candidate = self._clip(population[i] + r * (population[i] - population[j]))
                else:
                    # Learner j is better → move toward j
                    candidate = self._clip(population[i] + r * (population[j] - population[i]))

                f_candidate = self.func(candidate)
                if f_candidate < fitness[i]:
                    population[i] = candidate
                    fitness[i] = f_candidate

            # ── Track best ────────────────────────────────────────────
            cur_best_idx = int(np.argmin(fitness))
            if fitness[cur_best_idx] < best_score:
                best_score = float(fitness[cur_best_idx])
                best_pos = population[cur_best_idx].copy()

            history.append(best_score)
            trajectory.append(best_pos.copy())

        return best_pos, best_score, history, trajectory