import numpy as np


class SimulatedAnnealing:
    """
    Simulated Annealing (SA) — Physics-based metaheuristic inspired by
    the thermodynamic annealing process in metallurgy.

    Supports two modes:
      - Continuous optimisation  : optimize()   → (best_pos, best_score, history, trajectory)
      - Discrete / TSP           : solve_tsp()  → (best_route, best_cost, history)

    Both return formats are aligned with the rest of this workspace so the
    runners in main1.py can register SA with @register_continuous and
    @register_tsp without any adapter code.

    Cooling schedule
    ----------------
    T(k) = T_init * cooling_rate^k   (geometric / exponential decay)

    Acceptance criterion (Metropolis)
    ----------------------------------
    Accept worse solution with probability  p = exp(-Δf / T)
    This allows escaping local optima early in the search when T is high,
    and converges to a greedy selector as T → 0.
    """

    # ------------------------------------------------------------------ #
    #  Constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        T_init: float = 1000.0,
        T_min: float = 1e-3,
        cooling_rate: float = 0.995,
        max_iter: int = 10_000,
    ):
        """
        Args:
            T_init      : Initial temperature  (controls early exploration).
            T_min       : Stop when temperature falls below this value.
            cooling_rate: Multiplicative decay factor ∈ (0, 1).
                          Larger  (e.g. 0.999) → slow cooling → thorough search.
                          Smaller (e.g. 0.90)  → fast cooling → quick convergence.
            max_iter    : Hard upper bound on the number of iterations.
        """
        self.T_init = T_init
        self.T_min = T_min
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #
    def _accept(self, delta: float, T: float) -> bool:
        """
        Metropolis acceptance criterion.
        Always accept improvements (delta < 0).
        Accept deteriorations with probability exp(-delta / T).
        """
        if delta < 0:
            return True
        return np.random.rand() < np.exp(-delta / T)

    # ------------------------------------------------------------------ #
    #  Mode 1 – Continuous optimisation                                   #
    # ------------------------------------------------------------------ #
    def optimize(self, func, bounds):
        """
        Run SA on a continuous objective function.

        Args:
            func   : Callable f(x) → float to *minimise*.
            bounds : List of [min, max] pairs, one per dimension.

        Returns:
            best_pos   : ndarray  – best solution found.
            best_score : float    – objective value at best_pos.
            history    : list[float] – best score after every accepted step
                         (length ≤ max_iter + 1).
            trajectory : list[ndarray] – best_pos snapshot parallel to history.
        """
        bounds = np.array(bounds, dtype=float)
        dim = len(bounds)
        min_b, max_b = bounds[:, 0], bounds[:, 1]

        # ── Initialise ──────────────────────────────────────────────────
        curr_pos = min_b + np.random.rand(dim) * (max_b - min_b)
        curr_score = func(curr_pos)

        best_pos = curr_pos.copy()
        best_score = curr_score

        history: list = [best_score]
        trajectory: list = [best_pos.copy()]

        T = self.T_init

        # ── Main loop ───────────────────────────────────────────────────
        for _ in range(self.max_iter):
            if T < self.T_min:
                break

            # Neighbour generation — Gaussian perturbation scaled to temperature
            # Step size shrinks as T cools → fine-grained search near the end
            step_scale = (max_b - min_b) * (T / self.T_init) * 0.1
            neighbor = curr_pos + np.random.normal(0, step_scale, dim)
            neighbor = np.clip(neighbor, min_b, max_b)

            neighbor_score = func(neighbor)
            delta = neighbor_score - curr_score

            # ── Metropolis accept ────────────────────────────────────────
            if self._accept(delta, T):
                curr_pos = neighbor
                curr_score = neighbor_score

                # Update global best
                if curr_score < best_score:
                    best_score = curr_score
                    best_pos = curr_pos.copy()

            history.append(best_score)
            trajectory.append(best_pos.copy())

            # ── Cool down ────────────────────────────────────────────────
            T *= self.cooling_rate

        return best_pos, best_score, history, trajectory

    # ------------------------------------------------------------------ #
    #  Mode 2 – Discrete / TSP optimisation                              #
    # ------------------------------------------------------------------ #
    def solve_tsp(self, num_cities: int, dist_matrix: np.ndarray):
        """
        Run SA on the Travelling Salesman Problem.

        Neighbourhood operator: random 2-opt swap (reverse a sub-tour segment).

        Args:
            num_cities  : Number of cities.
            dist_matrix : (n × n) symmetric distance matrix.

        Returns:
            best_route : list[int] – best permutation found.
            best_cost  : float    – total tour length of best_route.
            history    : list[float] – best cost recorded every iteration.
        """
        # ── Initialise with a random tour ────────────────────────────────
        curr_route = list(np.random.permutation(num_cities))
        curr_cost = self._tour_cost(curr_route, dist_matrix)

        best_route = curr_route.copy()
        best_cost = curr_cost

        history: list = [best_cost]

        T = self.T_init

        # ── Main loop ────────────────────────────────────────────────────
        for _ in range(self.max_iter):
            if T < self.T_min:
                break

            # ── 2-opt neighbour ──────────────────────────────────────────
            neighbor_route = self._two_opt_swap(curr_route)
            neighbor_cost = self._tour_cost(neighbor_route, dist_matrix)
            delta = neighbor_cost - curr_cost

            # ── Metropolis accept ────────────────────────────────────────
            if self._accept(delta, T):
                curr_route = neighbor_route
                curr_cost = neighbor_cost

                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_route = curr_route.copy()

            history.append(best_cost)

            # ── Cool down ────────────────────────────────────────────────
            T *= self.cooling_rate

        return best_route, best_cost, history

    # ------------------------------------------------------------------ #
    #  TSP helpers                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tour_cost(route: list, dist_matrix: np.ndarray) -> float:
        """Total length of a closed tour."""
        cost = 0.0
        n = len(route)
        for i in range(n - 1):
            cost += float(dist_matrix[route[i]][route[i + 1]])
        cost += float(dist_matrix[route[-1]][route[0]])   # return to start
        return cost

    @staticmethod
    def _two_opt_swap(route: list) -> list:
        """
        Reverse a random sub-segment of the tour.
        This is the classic 2-opt neighbourhood for TSP.

        Example:  [0,1,2,3,4,5] with i=2, j=4
              →  [0,1,4,3,2,5]
        """
        n = len(route)
        new_route = route.copy()
        i, j = sorted(np.random.choice(n, 2, replace=False))
        new_route[i:j + 1] = reversed(new_route[i:j + 1])
        return new_route