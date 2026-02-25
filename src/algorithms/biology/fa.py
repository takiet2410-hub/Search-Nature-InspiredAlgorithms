import numpy as np


class FireflyAlgorithm:
    """
    Firefly Algorithm (FA) for continuous optimization.
    Inspired by the bioluminescent flashing behavior of fireflies.

    Core mechanics:
      - Attractiveness : β(r) = β0 * exp(-γ * r²)
        A firefly moves toward a brighter (better) one; attraction fades with distance.
      - Randomization  : When no brighter neighbor exists, the firefly moves randomly
        with step size scaled by α.
      - Light intensity: I ∝ 1 / (1 + f(x))  — lower objective = brighter firefly.

    Return format is aligned with DE / CS / PSO / ABC / SA / TLBO in this workspace:
        (best_position, best_score, history, trajectory)
    """

    # ------------------------------------------------------------------ #
    #  Constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        func,
        bounds,
        n_fireflies: int = 30,
        alpha: float = 0.5,
        beta0: float = 1.0,
        gamma: float = 1.0,
        alpha_decay: float = 0.97,
    ):
        """
        Args:
            func        : Objective function f(x) → float to *minimise*.
            bounds      : List of [min, max] pairs, one per dimension.
            n_fireflies : Population size (number of fireflies).
            alpha       : Randomisation step size ∈ (0, 1].
                          Controls exploration width; decays each iteration.
            beta0       : Maximum attractiveness at distance r = 0.
            gamma       : Light absorption coefficient γ > 0.
                          Large γ  → attractiveness drops fast (local search).
                          Small γ  → attractiveness stays high (global search).
            alpha_decay : Multiplicative decay for alpha each iteration ∈ (0, 1].
                          Gradually shifts from exploration to exploitation.
        """
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.n_fireflies = n_fireflies
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.alpha_decay = alpha_decay
        self.dim = len(bounds)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #
    def _clip(self, x: np.ndarray) -> np.ndarray:
        """Enforce boundary constraints."""
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def _light_intensity(self, obj_value: float) -> float:
        """
        Convert objective value → light intensity.
        Monotonically decreasing in obj_value so minimisation ≡ maximising brightness.
        I = 1 / (1 + f(x))  [always positive, avoids division-by-zero]
        """
        return 1.0 / (1.0 + obj_value)

    def _attractiveness(self, r_squared: float) -> float:
        """
        β(r) = β0 * exp(−γ * r²)
        r_squared : squared Euclidean distance between two fireflies.
        """
        return self.beta0 * np.exp(-self.gamma * r_squared)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def optimize(self, iterations: int = 100):
        """
        Run the Firefly Algorithm.

        Args:
            iterations : Number of generations (full population sweeps).

        Returns:
            best_pos   : ndarray     – coordinates of the best solution found.
            best_score : float       – objective value at best_pos.
            history    : list[float] – best score after every iteration
                         (length = iterations + 1; index 0 = initial best).
            trajectory : list[ndarray] – best_pos snapshot parallel to history.
        """
        min_b, max_b = self.bounds[:, 0], self.bounds[:, 1]
        scale = max_b - min_b          # per-dimension range (used for random step)

        # ── Initialise population ─────────────────────────────────────
        fireflies = min_b + np.random.rand(self.n_fireflies, self.dim) * scale
        obj_values = np.array([self.func(f) for f in fireflies])
        intensities = np.array([self._light_intensity(v) for v in obj_values])

        best_idx = int(np.argmin(obj_values))
        best_pos = fireflies[best_idx].copy()
        best_score = float(obj_values[best_idx])

        history: list = [best_score]
        trajectory: list = [best_pos.copy()]

        alpha = self.alpha              # local copy — will be decayed

        # ── Main loop ─────────────────────────────────────────────────
        for _ in range(iterations):

            # ── Pairwise movement ─────────────────────────────────────
            # For every firefly i, check all fireflies j.
            # If j is brighter, i moves toward j.
            for i in range(self.n_fireflies):
                moved = False

                for j in range(self.n_fireflies):
                    if intensities[j] <= intensities[i]:
                        continue                          # j is not brighter → skip

                    # Squared Euclidean distance
                    diff = fireflies[j] - fireflies[i]
                    r_sq = float(np.dot(diff, diff))

                    # Attraction-based step toward j
                    beta = self._attractiveness(r_sq)

                    # Random perturbation (uniform, zero-mean: rand - 0.5 ∈ [-0.5, 0.5])
                    rand_step = alpha * scale * (np.random.rand(self.dim) - 0.5)

                    # Position update:
                    # x_i = x_i + β*(x_j - x_i) + α*(rand - 0.5)*scale
                    new_pos = self._clip(
                        fireflies[i] + beta * diff + rand_step
                    )

                    new_obj = self.func(new_pos)
                    new_int = self._light_intensity(new_obj)

                    # Accept unconditionally (FA always moves toward brighter)
                    fireflies[i] = new_pos
                    obj_values[i] = new_obj
                    intensities[i] = new_int
                    moved = True

                # ── Random walk if no brighter neighbor found ─────────
                if not moved:
                    rand_step = alpha * scale * (np.random.rand(self.dim) - 0.5)
                    new_pos = self._clip(fireflies[i] + rand_step)
                    new_obj = self.func(new_pos)
                    new_int = self._light_intensity(new_obj)

                    # Greedy accept for random walk (avoid degrading the best)
                    if new_obj < obj_values[i]:
                        fireflies[i] = new_pos
                        obj_values[i] = new_obj
                        intensities[i] = new_int

            # ── Update global best ────────────────────────────────────
            cur_best_idx = int(np.argmin(obj_values))
            if obj_values[cur_best_idx] < best_score:
                best_score = float(obj_values[cur_best_idx])
                best_pos = fireflies[cur_best_idx].copy()

            history.append(best_score)
            trajectory.append(best_pos.copy())

            # ── Decay randomisation step (exploration → exploitation) ─
            alpha *= self.alpha_decay

        return best_pos, best_score, history, trajectory