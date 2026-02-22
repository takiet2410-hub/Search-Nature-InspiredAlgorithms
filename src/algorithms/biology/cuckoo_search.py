import numpy as np
from math import gamma, sin, pi


class CuckooSearch:
    """
    Cuckoo Search for continuous optimization.
    Return format is aligned with DE/HC in this workspace:
    (best_position, best_score, history, trajectory)
    """
    def __init__(self, func, bounds, n_nests=50, pa=0.25, alpha=0.01, beta=1.5):
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.n_nests = n_nests
        self.pa = pa                  # Abandonment probability
        self.alpha = alpha            # Lévy step scale
        self.beta = beta              # Lévy exponent
        self.dim = len(bounds)

    def _levy_flight(self, size):
        # Mantegna algorithm
        sigma_u = (
            gamma(1 + self.beta) * sin(pi * self.beta / 2) /
            (gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))
        ) ** (1 / self.beta)

        u = np.random.normal(0, sigma_u, size=size)
        v = np.random.normal(0, 1, size=size)
        step = u / (np.abs(v) ** (1 / self.beta) + 1e-12)
        return step

    def optimize(self, iterations=100):
        min_b, max_b = self.bounds[:, 0], self.bounds[:, 1]

        # Initialize nests
        nests = min_b + np.random.rand(self.n_nests, self.dim) * (max_b - min_b)
        fitness = np.array([self.func(n) for n in nests])

        best_idx = np.argmin(fitness)
        best_nest = nests[best_idx].copy()
        best_score = fitness[best_idx]

        history = [best_score]
        trajectory = [best_nest.copy()]

        for _ in range(iterations):
            # Generate new solutions by Lévy flights
            for i in range(self.n_nests):
                step = self._levy_flight(self.dim)
                candidate = nests[i] + self.alpha * step
                candidate = np.clip(candidate, min_b, max_b)
                f_candidate = self.func(candidate)

                # Randomly chosen nest replacement
                j = np.random.randint(0, self.n_nests)
                if f_candidate < fitness[j]:
                    nests[j] = candidate
                    fitness[j] = f_candidate

            # Abandon worst nests
            n_abandon = max(1, int(self.pa * self.n_nests))
            worst_idx = np.argsort(fitness)[-n_abandon:]
            new_nests = min_b + np.random.rand(n_abandon, self.dim) * (max_b - min_b)
            nests[worst_idx] = new_nests
            fitness[worst_idx] = np.array([self.func(n) for n in new_nests])

            # Track best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_score:
                best_score = fitness[best_idx]
                best_nest = nests[best_idx].copy()

            history.append(best_score)
            trajectory.append(best_nest.copy())

        return best_nest, best_score, history, trajectory