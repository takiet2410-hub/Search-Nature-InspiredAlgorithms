"""
Bộ giải bài toán Knapsack bằng các thuật toán khác nhau.

Chiến lược chung cho thuật toán liên tục (PSO, DE, ABC, CS, FA, TLBO):
  - Chạy trong không gian liên tục [0, 1]^n
  - Chuyển đổi: x_i > 0.5 → chọn vật phẩm (1), ngược lại (0)
  - Repair nếu vượt quá capacity
  - Fitness = -total_value (minimize → maximize value)

SA giải trực tiếp trên binary vector bằng bit-flip neighbor.
"""

import numpy as np


# ============================================================
# HELPER: Chuyển đổi liên tục → nhị phân + repair + fitness
# ============================================================

def _continuous_to_binary(x):
    """Sigmoid threshold: x > 0.5 → 1, else 0."""
    return (np.array(x) > 0.5).astype(int)


def _repair(individual, weights, values, capacity):
    """Sửa nghiệm vi phạm: loại vật phẩm có value/weight thấp nhất."""
    ind = individual.copy()
    while np.dot(ind, weights) > capacity:
        selected = np.where(ind == 1)[0]
        if len(selected) == 0:
            break
        ratios = values[selected] / (weights[selected] + 1e-12)
        worst = selected[np.argmin(ratios)]
        ind[worst] = 0
    return ind


def _kp_fitness_for_minimize(x, weights, values, capacity):
    """
    Hàm fitness để MINIMIZE (vì các thuật toán liên tục minimize).
    Trả về -value (minimize -value = maximize value).
    """
    binary = _continuous_to_binary(x)
    binary = _repair(binary, weights, values, capacity)
    return -float(np.dot(binary, values))


# ============================================================
# SA cho Knapsack — bit-flip neighbor
# ============================================================

def sa_knapsack(weights, values, capacity, max_iter=5000,
                T_init=100.0, T_min=0.01, cooling_rate=0.995):
    """
    Simulated Annealing giải Knapsack bằng bit-flip neighbor.

    Returns: (best_individual, best_value, history)
    """
    n = len(weights)
    # Khởi tạo ngẫu nhiên
    current = np.random.randint(0, 2, size=n)
    current = _repair(current, weights, values, capacity)
    current_val = float(np.dot(current, values))

    best = current.copy()
    best_val = current_val
    history = [best_val]

    T = T_init
    for _ in range(max_iter):
        # Neighbor: flip 1 bit ngẫu nhiên
        neighbor = current.copy()
        idx = np.random.randint(0, n)
        neighbor[idx] = 1 - neighbor[idx]
        neighbor = _repair(neighbor, weights, values, capacity)
        neighbor_val = float(np.dot(neighbor, values))

        delta = neighbor_val - current_val  # Maximize → delta > 0 is good
        if delta > 0 or np.random.rand() < np.exp(delta / (T + 1e-12)):
            current = neighbor
            current_val = neighbor_val

        if current_val > best_val:
            best_val = current_val
            best = current.copy()

        history.append(best_val)
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best, best_val, history


# ============================================================
# PSO cho Knapsack — binary PSO (sigmoid transfer)
# ============================================================

def pso_knapsack(weights, values, capacity, num_particles=30,
                 iterations=100, w=0.7, c1=1.5, c2=1.5):
    """
    Binary PSO cho Knapsack: chạy trong [0,1]^n, sigmoid → binary.

    Returns: (best_individual, best_value, history)
    """
    n = len(weights)
    # Khởi tạo trong [0, 1]
    positions = np.random.rand(num_particles, n)
    velocities = np.random.randn(num_particles, n) * 0.1

    pbest_pos = positions.copy()
    pbest_val = np.full(num_particles, -np.inf)
    gbest_pos = None
    gbest_val = -np.inf

    for i in range(num_particles):
        binary = _continuous_to_binary(positions[i])
        binary = _repair(binary, weights, values, capacity)
        val = float(np.dot(binary, values))
        pbest_val[i] = val
        if val > gbest_val:
            gbest_val = val
            gbest_pos = positions[i].copy()

    history = [gbest_val]

    for _ in range(iterations):
        for i in range(num_particles):
            r1, r2 = np.random.rand(n), np.random.rand(n)
            velocities[i] = (w * velocities[i]
                             + c1 * r1 * (pbest_pos[i] - positions[i])
                             + c2 * r2 * (gbest_pos - positions[i]))
            # Sigmoid clamping
            velocities[i] = np.clip(velocities[i], -4, 4)
            # Update position via sigmoid
            sigmoid = 1.0 / (1.0 + np.exp(-velocities[i]))
            positions[i] = (np.random.rand(n) < sigmoid).astype(float)

            binary = _continuous_to_binary(positions[i])
            binary = _repair(binary, weights, values, capacity)
            val = float(np.dot(binary, values))

            if val > pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = positions[i].copy()
            if val > gbest_val:
                gbest_val = val
                gbest_pos = positions[i].copy()

        history.append(gbest_val)

    best_binary = _continuous_to_binary(gbest_pos)
    best_binary = _repair(best_binary, weights, values, capacity)
    return best_binary, gbest_val, history


# ============================================================
# Generic Continuous-to-KP adapter
# (cho DE, ABC, CS, FA, TLBO)
# ============================================================

def _run_continuous_for_kp(AlgorithmClass, weights, values, capacity,
                           alg_kwargs, run_kwargs):
    """
    Chạy thuật toán liên tục trong [0,1]^n rồi chuyển sang binary KP.

    Args:
        AlgorithmClass: Class thuật toán (DE, ABC, CS, FA, TLBO)
        weights, values, capacity: Dữ liệu KP
        alg_kwargs: Tham số khởi tạo của thuật toán
        run_kwargs: Tham số chạy (generations/iterations)

    Returns: (best_individual, best_value, history)
    """
    n = len(weights)
    bounds = [[0, 1]] * n

    # Hàm mục tiêu: minimize -value
    def obj(x):
        return _kp_fitness_for_minimize(x, weights, values, capacity)

    alg = AlgorithmClass(obj, bounds, **alg_kwargs)

    # Xử lý method name khác nhau
    if hasattr(alg, 'optimize'):
        result = alg.optimize(**run_kwargs)
    elif hasattr(alg, 'solve'):
        result = alg.solve(**run_kwargs)
    else:
        raise ValueError(f"Unknown method for {AlgorithmClass}")

    best_x = result[0]
    best_binary = _continuous_to_binary(best_x)
    best_binary = _repair(best_binary, weights, values, capacity)
    best_val = float(np.dot(best_binary, values))

    # Convert history (negative → positive)
    history = [-h for h in result[2]]
    return best_binary, best_val, history


# --- Các wrapper cụ thể ---

def de_knapsack(weights, values, capacity, pop_size=30, generations=100):
    from src.algorithms.evolution.differential_evolution import DifferentialEvolution
    return _run_continuous_for_kp(
        DifferentialEvolution, weights, values, capacity,
        alg_kwargs={'pop_size': pop_size},
        run_kwargs={'generations': generations}
    )

def abc_knapsack(weights, values, capacity, colony_size=30, iterations=100):
    from src.algorithms.biology.abc import ArtificialBeeColony
    return _run_continuous_for_kp(
        ArtificialBeeColony, weights, values, capacity,
        alg_kwargs={'colony_size': colony_size},
        run_kwargs={'iterations': iterations}
    )

def cs_knapsack(weights, values, capacity, n_nests=30, iterations=100):
    from src.algorithms.biology.cuckoo_search import CuckooSearch
    return _run_continuous_for_kp(
        CuckooSearch, weights, values, capacity,
        alg_kwargs={'n_nests': n_nests, 'pa': 0.25, 'alpha': 0.01},
        run_kwargs={'iterations': iterations}
    )

def fa_knapsack(weights, values, capacity, n_fireflies=30, iterations=100):
    from src.algorithms.biology.fa import FireflyAlgorithm
    return _run_continuous_for_kp(
        FireflyAlgorithm, weights, values, capacity,
        alg_kwargs={'n_fireflies': n_fireflies},
        run_kwargs={'iterations': iterations}
    )

def pso_continuous_knapsack(weights, values, capacity, num_particles=30, iterations=100):
    """PSO liên tục → binary (alternative to binary PSO above)."""
    from src.algorithms.biology.pso import ParticleSwarmOptimization
    return _run_continuous_for_kp(
        ParticleSwarmOptimization, weights, values, capacity,
        alg_kwargs={'num_particles': num_particles},
        run_kwargs={'iterations': iterations}
    )

def tlbo_knapsack(weights, values, capacity, pop_size=30, iterations=100):
    from src.algorithms.human.tlbo import TLBO
    return _run_continuous_for_kp(
        TLBO, weights, values, capacity,
        alg_kwargs={'pop_size': pop_size},
        run_kwargs={'iterations': iterations}
    )
