import numpy as np
import os
import csv
import time
from src.utils import problems, visualization

# Import Algorithms
# Bio
from src.algorithms.biology.abc import ArtificialBeeColony
from src.algorithms.biology.aco import AntColonyOptimizationTSP
from src.algorithms.biology.cuckoo_search import CuckooSearch
from src.algorithms.biology.fa import FireflyAlgorithm
from src.algorithms.biology.pso import ParticleSwarmOptimization
# Classical
from src.algorithms.classical.baselines import TSPGraphSearch, ContinuousLocalSearch
from src.algorithms.physics.sa import SimulatedAnnealing
# Evo
from src.algorithms.evolution.genetic_algorithm import GeneticAlgorithmTSP
from src.algorithms.evolution.differential_evolution import DifferentialEvolution
from src.algorithms.evolution.ga_knapsack import GeneticAlgorithmKnapsack
# Human
from src.algorithms.human.tlbo import TLBO
# Discrete problems
from src.problems.discrete.graph_coloring import (
    generate_random_graph, count_conflicts,
    ga_graph_coloring, greedy_graph_coloring, sa_graph_coloring,
)
# Knapsack solvers (all algorithms)
from src.algorithms.knapsack_solvers import (
    sa_knapsack, pso_knapsack, de_knapsack,
    abc_knapsack, cs_knapsack, fa_knapsack, tlbo_knapsack,
)
# Shortest Path
from src.problems.discrete.shortest_path import (
    generate_weighted_graph, dijkstra, a_star_shortest,
    bfs_shortest, dfs_shortest,
)

# --- THÊM VÀO PHẦN IMPORT ---
# Import ACO mới
from src.algorithms.biology.aco_graph_coloring import AntColonyOptimizationGC
from src.algorithms.biology.aco_shortest_path import AntColonyOptimizationSP

# Import HC, DFS cho Graph Coloring
from src.problems.discrete.graph_coloring import hc_graph_coloring, dfs_graph_coloring

# Import HC, DFS cho Knapsack
from src.algorithms.knapsack_solvers import hc_knapsack, dfs_knapsack

# =============================================================================
# HELPERS
# =============================================================================
def manual_onesample_ttest(sample, population_mean):
    n = len(sample)
    mean = np.mean(sample)
    std = np.std(sample, ddof=1)
    se = std / np.sqrt(n)
    if se == 0: return 0.0
    return (mean - population_mean) / se

def manual_twosample_ttest(sample1, sample2):
    n1, n2 = len(sample1), len(sample2)
    m1, m2 = np.mean(sample1), np.mean(sample2)
    v1, v2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    se = np.sqrt(v1/n1 + v2/n2)
    if se == 0: return 0.0
    return (m1 - m2) / se


# =============================================================================
# CSV EXPORT HELPERS
# =============================================================================
CSV_DIR = 'results/csv'

def _ensure_csv_dir(subdir=''):
    path = os.path.join(CSV_DIR, subdir) if subdir else CSV_DIR
    os.makedirs(path, exist_ok=True)
    return path

def save_scores_csv(all_scores, filename, subdir=''):
    """
    Lưu điểm số mỗi lần chạy của các thuật toán ra CSV.
    Columns: Run, Alg1, Alg2, ...
    """
    path = _ensure_csv_dir(subdir)
    filepath = os.path.join(path, filename)
    names = list(all_scores.keys())
    n_runs = max(len(v) for v in all_scores.values())

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Run'] + names)
        for i in range(n_runs):
            row = [i + 1]
            for name in names:
                row.append(f"{all_scores[name][i]:.6f}" if i < len(all_scores[name]) else '')
            writer.writerow(row)

        # Summary row
        writer.writerow([])
        writer.writerow(['STATS'] + names)
        writer.writerow(['Mean'] + [f"{np.mean(all_scores[n]):.6f}" for n in names])
        writer.writerow(['Std']  + [f"{np.std(all_scores[n]):.6f}" for n in names])
        writer.writerow(['Best'] + [f"{np.min(all_scores[n]):.6f}" for n in names])
        writer.writerow(['Worst']+ [f"{np.max(all_scores[n]):.6f}" for n in names])

    print(f"[CSV] Saved: {filepath}")

def save_scalability_csv(x_values, times_dict, filename, xlabel, subdir=''):
    """Lưu thời gian chạy vs kích thước ra CSV."""
    path = _ensure_csv_dir(subdir)
    filepath = os.path.join(path, filename)
    names = list(times_dict.keys())

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([xlabel] + names)
        for i, x in enumerate(x_values):
            row = [x]
            for name in names:
                row.append(f"{times_dict[name][i]:.6f}" if i < len(times_dict[name]) else '')
            writer.writerow(row)

    print(f"[CSV] Saved: {filepath}")

def save_sensitivity_csv(matrix, x_labels, y_labels, filename, xlabel, ylabel, subdir=''):
    """Lưu kết quả sensitivity analysis ra CSV."""
    path = _ensure_csv_dir(subdir)
    filepath = os.path.join(path, filename)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([f"{ylabel} \\ {xlabel}"] + [str(x) for x in x_labels])
        for i, y in enumerate(y_labels):
            row = [str(y)] + [f"{matrix[i, j]:.6f}" for j in range(len(x_labels))]
            writer.writerow(row)

    print(f"[CSV] Saved: {filepath}")

def save_convergence_csv(all_histories, filename, subdir=''):
    """
    Lưu lịch sử hội tụ (mean per generation) ra CSV.
    Columns: Generation, Alg1_Mean, Alg1_Std, Alg2_Mean, Alg2_Std, ...
    """
    path = _ensure_csv_dir(subdir)
    filepath = os.path.join(path, filename)
    names = list(all_histories.keys())

    # Tính max length chung
    min_lens = {}
    for name in names:
        if all_histories[name]:
            min_lens[name] = min(len(h) for h in all_histories[name])
        else:
            min_lens[name] = 0
    max_gen = max(min_lens.values()) if min_lens else 0

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['Generation']
        for name in names:
            header += [f"{name}_Mean", f"{name}_Std"]
        writer.writerow(header)

        for g in range(max_gen):
            row = [g]
            for name in names:
                if g < min_lens.get(name, 0):
                    vals = [h[g] for h in all_histories[name]]
                    row += [f"{np.mean(vals):.6f}", f"{np.std(vals):.6f}"]
                else:
                    row += ['', '']
            writer.writerow(row)

    print(f"[CSV] Saved: {filepath}")



# =============================================================================
# ALGORITHM REGISTRY
# Each entry returns (score, history, path) given (func, bounds, generations)
# =============================================================================
CONTINUOUS_ALGORITHMS = {}
DISCRETE_ALGORITHMS = {}

def register_continuous(name):
    """Decorator to register a continuous optimization algorithm runner."""
    def decorator(fn):
        CONTINUOUS_ALGORITHMS[name] = fn
        return fn
    return decorator

def register_tsp(name):
    """Decorator to register a TSP algorithm runner."""
    def decorator(fn):
        DISCRETE_ALGORITHMS[name] = fn
        return fn
    return decorator


# --- Register Continuous Algorithms ---
# Each runner signature: (func, bounds, generations, pop_size) -> (score, history, path)

@register_continuous("DE")
def run_de(func, bounds, generations, pop_size):
    de = DifferentialEvolution(func, bounds, pop_size=pop_size)
    _, score, history, path = de.optimize(generations=generations)
    return score, history, path

@register_continuous("HC")
def run_hc(func, bounds, generations, pop_size):
    hc = ContinuousLocalSearch(step_size=0.5, max_iter=generations * pop_size)
    _, score, history, path = hc.hill_climbing(func, bounds)
    sampled_history = history[::pop_size][:generations]
    return score, sampled_history, path

@register_continuous("CS")
def run_cs(func, bounds, generations, pop_size):
    cs = CuckooSearch(func, bounds, n_nests=pop_size, pa=0.25, alpha=0.01, beta=1.5)
    _, score, history, path = cs.optimize(iterations=generations)
    return score, history, path

@register_continuous("TLBO")
def run_tlbo(func, bounds, generations, pop_size):
    tlbo = TLBO(func, bounds, pop_size=pop_size)
    _, score, history, path = tlbo.optimize(iterations=generations)
    return score, history, path

@register_continuous("PSO")
def run_pso(func, bounds, generations, pop_size):
    pso = ParticleSwarmOptimization(func, bounds, num_particles=pop_size)
    _, score, history, path = pso.optimize(iterations=generations)
    return score, history, path

@register_continuous("ABC")
def run_abc(func, bounds, generations, pop_size):
    abc = ArtificialBeeColony(func, bounds, colony_size=pop_size)
    _, score, history, path = abc.optimize(iterations=generations)
    return score, history, path

@register_continuous("FA")
def run_fa(func, bounds, generations, pop_size):
    fa = FireflyAlgorithm(
        func, bounds,
        n_fireflies=pop_size,
        alpha=0.5, beta0=1.0, gamma=1.0, alpha_decay=0.97,
    )
    _, score, history, path = fa.optimize(iterations=generations)
    return score, history, path

@register_continuous("SA")
def run_sa_continuous(func, bounds, generations, pop_size):
    sa = SimulatedAnnealing(
        T_init=1000.0, T_min=1e-3, cooling_rate=0.995,
        max_iter=generations * pop_size,
    )
    _, score, history, path = sa.optimize(func, bounds)
    step = max(1, len(history) // generations)
    sampled_history = history[::step][:generations]
    return score, sampled_history, path


# --- Register TSP Algorithms ---
# Each runner signature: (n, dist, generations, pop_size) -> (score, history, route)

@register_tsp("GA")
def run_ga_tsp(n, dist, generations, pop_size):
    ga = GeneticAlgorithmTSP(n, dist, pop_size=pop_size)
    route, cost, history = ga.solve(generations=generations)
    return cost, history, route

@register_tsp("HC")
def run_hc_tsp(n, dist, generations, pop_size):
    solver = TSPGraphSearch(n, dist)
    # Generations * pop_size để công bằng số lần đánh giá hàm mục tiêu
    route, cost, history = solver.hill_climbing(max_iter=generations * pop_size)
    step = max(1, len(history) // generations)
    sampled_history = history[::step][:generations]
    return cost, sampled_history, route

@register_tsp("ACO")
def run_aco_tsp(n, dist, generations, pop_size):
    aco = AntColonyOptimizationTSP(
        n, dist, num_ants=pop_size,
        alpha=1.0, beta=3.0, evaporation_rate=0.5, Q=100.0
    )
    route, cost, history = aco.solve(iterations=generations)
    return cost, history, route

@register_tsp("SA")
def run_sa_tsp(n, dist, generations, pop_size):
    sa = SimulatedAnnealing(
        T_init=1000.0, T_min=1e-3, cooling_rate=0.995,
        max_iter=generations * pop_size,
    )
    route, cost, history = sa.solve_tsp(n, dist)
    step = max(1, len(history) // generations)
    sampled_history = history[::step][:generations]
    return cost, sampled_history, route


# =============================================================================
# PROBLEM REGISTRY
# =============================================================================
CONTINUOUS_PROBLEMS = {
    "Sphere": {
        "func": problems.sphere_function,
        "bounds": [[-5.12, 5.12]] * 10,
        "generations": 50,
    },
    "Rastrigin": {
        "func": problems.rastrigin_function,
        "bounds": [[-5.12, 5.12]] * 10,
        "generations": 100,
    },
    "Rosenbrock": {
        "func": problems.rosenbrock_function,
        "bounds": [[-5, 10]] * 10,
        "generations": 100,
    },
    "Griewank": {
        "func": problems.griewank_function,
        "bounds": [[-600, 600]] * 10,
        "generations": 100,
    },
    "Ackley": {
        "func": problems.ackley_function,
        "bounds": [[-32, 32]] * 10,
        "generations": 100,
    },
}


# =============================================================================
# GENERIC CONTINUOUS COMPARISON ENGINE
# =============================================================================
class ContinuousComparison:
    """
    Runs any subset of registered algorithms on any subset of registered problems.
    Add algorithms via @register_continuous. Add problems to CONTINUOUS_PROBLEMS.
    """

    def __init__(self, algorithm_names=None, problem_names=None, runs=30, pop_size=50):
        self.algorithm_names = algorithm_names or list(CONTINUOUS_ALGORITHMS.keys())
        self.problem_names = problem_names or list(CONTINUOUS_PROBLEMS.keys())
        self.runs = runs
        self.pop_size = pop_size

    def run_all(self):
        print("\n" + "#"*60)
        print(f"CONTINUOUS COMPARISON: {self.algorithm_names}")
        print(f"PROBLEMS: {self.problem_names}")
        print("#"*60)

        for prob_name in self.problem_names:
            self._run_on_problem(prob_name)

        self._scalability_dimensions()
        self._sensitivity_continuous()

    def _run_on_problem(self, prob_name):
        print(f"\n>>> PROBLEM: {prob_name} <<<")
        config = CONTINUOUS_PROBLEMS[prob_name]
        func = config["func"]
        bounds = config["bounds"]
        generations = config["generations"]

        all_scores = {name: [] for name in self.algorithm_names}
        all_histories = {name: [] for name in self.algorithm_names}

        for _ in range(self.runs):
            for name in self.algorithm_names:
                runner = CONTINUOUS_ALGORITHMS[name]
                score, history, _ = runner(func, bounds, generations, self.pop_size)
                all_scores[name].append(score)
                all_histories[name].append(history)

        tag = f"cont_{prob_name.lower()}_{'_vs_'.join(self.algorithm_names)}"
        
        # --- LƯU VÀO FOLDER TÊN HÀM ---
        folder_path = f"continuous/{prob_name.lower()}"

        # Convergence plot
        visualization.plot_robustness_convergence(
            all_histories,
            f"Convergence on {prob_name}: {' vs '.join(self.algorithm_names)}",
            f"{folder_path}/{tag}_convergence"
        )

        # Boxplot
        visualization.plot_boxplot_comparison(
            all_scores,
            f"Quality on {prob_name}: {' vs '.join(self.algorithm_names)}",
            f"{folder_path}/{tag}_boxplot"
        )

        # --- CSV EXPORT ---
        save_scores_csv(all_scores, f"{tag}_scores.csv", subdir=folder_path)
        save_convergence_csv(all_histories, f"{tag}_convergence.csv", subdir=folder_path)

        # Stats
        names = self.algorithm_names
        for name in names:
            print(f"  [{name}] Mean={np.mean(all_scores[name]):.4f}, Std={np.std(all_scores[name]):.4f}")

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                t = manual_twosample_ttest(all_scores[names[i]], all_scores[names[j]])
                print(f"  T-Test ({names[i]} vs {names[j]}): t={t:.4f}")

    def _scalability_dimensions(self):
        print("\n[SCALABILITY] Time vs Dimensions on Rastrigin")
        dims = [2, 5, 10, 20]
        times = {name: [] for name in self.algorithm_names}

        for d in dims:
            bounds = [[-5.12, 5.12]] * d
            func = problems.rastrigin_function
            for name in self.algorithm_names:
                s = time.time()
                CONTINUOUS_ALGORITHMS[name](func, bounds, generations=50, pop_size=30)
                times[name].append(time.time() - s)

        tag = '_vs_'.join(self.algorithm_names)
        folder_path = "continuous/rastrigin" # Gom chung vào rastrigin
        visualization.plot_scalability_lines(
            dims, times,
            f"Scalability: {' vs '.join(self.algorithm_names)} on Rastrigin (Time)",
            f"{folder_path}/cont_scalability_time_{tag}",
            "Dimensions (D)", "Execution Time (s)"
        )
        save_scalability_csv(dims, times, f"cont_scalability_{tag}.csv",
                             'Dimensions', subdir=folder_path)

    # ------------------------------------------------------------------
    # SENSITIVITY ANALYSES — All Algorithms
    # ------------------------------------------------------------------
    def _sensitivity_continuous(self):
        bounds = [[-5.12, 5.12]] * 10
        func = problems.rastrigin_function

        # --- DE sensitivity (F vs CR) ---
        if "DE" in self.algorithm_names:
            print("\n[SENSITIVITY] DE on Rastrigin (F vs CR)")
            F_vals = [0.3, 0.5, 0.9]
            CR_vals = [0.1, 0.5, 0.9]
            results = np.zeros((len(F_vals), len(CR_vals)))
            for i, f in enumerate(F_vals):
                for j, cr in enumerate(CR_vals):
                    scores = []
                    for _ in range(5):
                        de = DifferentialEvolution(func, bounds, pop_size=30,
                                                   mutation_factor=f, crossover_rate=cr)
                        _, s, _, _ = de.optimize(generations=50)
                        scores.append(s)
                    results[i, j] = np.mean(scores)
            visualization.plot_parameter_sensitivity(
                results, CR_vals, F_vals,
                "DE Sensitivity Analysis on Rastrigin",
                "continuous/sensitivity/cont_sensitivity_de",
                "Crossover Rate (CR)", "Mutation Factor (F)"
            )
            save_sensitivity_csv(results, CR_vals, F_vals,
                                 'sensitivity_de.csv', 'CR', 'F',
                                 subdir='continuous/sensitivity')

        # --- CS sensitivity (alpha vs pa) ---
        if "CS" in self.algorithm_names:
            print("\n[SENSITIVITY] CS on Rastrigin (alpha vs pa)")
            alpha_vals = [0.005, 0.01, 0.05]
            pa_vals = [0.1, 0.25, 0.4]
            cs_results = np.zeros((len(alpha_vals), len(pa_vals)))
            for i, alpha in enumerate(alpha_vals):
                for j, pa in enumerate(pa_vals):
                    scores = []
                    for _ in range(5):
                        cs = CuckooSearch(func, bounds, n_nests=30, pa=pa,
                                          alpha=alpha, beta=1.5)
                        _, s, _, _ = cs.optimize(iterations=50)
                        scores.append(s)
                    cs_results[i, j] = np.mean(scores)
            visualization.plot_parameter_sensitivity(
                cs_results, pa_vals, alpha_vals,
                "Cuckoo Search Sensitivity on Rastrigin",
                "continuous/sensitivity/cont_sensitivity_cs",
                "Abandonment Probability (pa)", "Step Size (alpha)"
            )
            save_sensitivity_csv(cs_results, pa_vals, alpha_vals,
                                 'sensitivity_cs.csv', 'pa', 'alpha',
                                 subdir='continuous/sensitivity')

        # --- PSO sensitivity (w vs c1=c2) ---
        if "PSO" in self.algorithm_names:
            print("\n[SENSITIVITY] PSO on Rastrigin (w vs c)")
            w_vals = [0.4, 0.7, 0.9]
            c_vals = [1.0, 1.5, 2.0]
            pso_results = np.zeros((len(w_vals), len(c_vals)))
            for i, w in enumerate(w_vals):
                for j, c in enumerate(c_vals):
                    scores = []
                    for _ in range(5):
                        pso = ParticleSwarmOptimization(func, bounds,
                                                        num_particles=30, w=w, c1=c, c2=c)
                        _, s, _, _ = pso.optimize(iterations=50)
                        scores.append(s)
                    pso_results[i, j] = np.mean(scores)
            visualization.plot_parameter_sensitivity(
                pso_results, c_vals, w_vals,
                "PSO Sensitivity on Rastrigin",
                "continuous/sensitivity/cont_sensitivity_pso",
                "Cognitive/Social Coeff (c1=c2)", "Inertia Weight (w)"
            )
            save_sensitivity_csv(pso_results, c_vals, w_vals,
                                 'sensitivity_pso.csv', 'c', 'w',
                                 subdir='continuous/sensitivity')

        # --- ABC sensitivity (colony_size vs limit) ---
        if "ABC" in self.algorithm_names:
            print("\n[SENSITIVITY] ABC on Rastrigin (colony_size vs limit)")
            colony_vals = [20, 40, 60]
            limit_vals = [50, 100, 200]
            abc_results = np.zeros((len(colony_vals), len(limit_vals)))
            for i, cs_val in enumerate(colony_vals):
                for j, lim in enumerate(limit_vals):
                    scores = []
                    for _ in range(5):
                        abc = ArtificialBeeColony(func, bounds,
                                                   colony_size=cs_val, limit=lim)
                        _, s, _, _ = abc.optimize(iterations=50)
                        scores.append(s)
                    abc_results[i, j] = np.mean(scores)
            visualization.plot_parameter_sensitivity(
                abc_results, limit_vals, colony_vals,
                "ABC Sensitivity on Rastrigin",
                "continuous/sensitivity/cont_sensitivity_abc",
                "Abandon Limit", "Colony Size"
            )
            save_sensitivity_csv(abc_results, limit_vals, colony_vals,
                                 'sensitivity_abc.csv', 'limit', 'colony_size',
                                 subdir='continuous/sensitivity')

        # --- FA sensitivity (alpha vs gamma) ---
        if "FA" in self.algorithm_names:
            print("\n[SENSITIVITY] FA on Rastrigin (alpha vs gamma)")
            fa_alpha_vals = [0.1, 0.5, 1.0]
            fa_gamma_vals = [0.5, 1.0, 2.0]
            fa_results = np.zeros((len(fa_alpha_vals), len(fa_gamma_vals)))
            for i, a in enumerate(fa_alpha_vals):
                for j, g in enumerate(fa_gamma_vals):
                    scores = []
                    for _ in range(5):
                        fa = FireflyAlgorithm(func, bounds, n_fireflies=30,
                                              alpha=a, gamma=g)
                        _, s, _, _ = fa.optimize(iterations=50)
                        scores.append(s)
                    fa_results[i, j] = np.mean(scores)
            visualization.plot_parameter_sensitivity(
                fa_results, fa_gamma_vals, fa_alpha_vals,
                "FA Sensitivity on Rastrigin",
                "continuous/sensitivity/cont_sensitivity_fa",
                "Absorption Coeff (gamma)", "Randomization (alpha)"
            )
            save_sensitivity_csv(fa_results, fa_gamma_vals, fa_alpha_vals,
                                 'sensitivity_fa.csv', 'gamma', 'alpha',
                                 subdir='continuous/sensitivity')

        # --- SA sensitivity (T_init vs cooling_rate) ---
        if "SA" in self.algorithm_names:
            print("\n[SENSITIVITY] SA on Rastrigin (T_init vs cooling_rate)")
            T_vals = [100, 1000, 5000]
            cool_vals = [0.990, 0.995, 0.999]
            sa_results = np.zeros((len(T_vals), len(cool_vals)))
            for i, t_init in enumerate(T_vals):
                for j, cr in enumerate(cool_vals):
                    scores = []
                    for _ in range(5):
                        sa = SimulatedAnnealing(T_init=t_init, cooling_rate=cr,
                                                max_iter=2500)
                        _, s, _, _ = sa.optimize(func, bounds)
                        scores.append(s)
                    sa_results[i, j] = np.mean(scores)
            visualization.plot_parameter_sensitivity(
                sa_results, cool_vals, T_vals,
                "SA Sensitivity on Rastrigin",
                "continuous/sensitivity/cont_sensitivity_sa",
                "Cooling Rate", "Initial Temperature (T₀)"
            )
            save_sensitivity_csv(sa_results, cool_vals, T_vals,
                                 'sensitivity_sa.csv', 'cooling_rate', 'T_init',
                                 subdir='continuous/sensitivity')

        # --- TLBO sensitivity (pop_size — single param, heatmap 1×N) ---
        if "TLBO" in self.algorithm_names:
            print("\n[SENSITIVITY] TLBO on Rastrigin (pop_size)")
            ps_vals = [10, 20, 50, 100]
            tlbo_results = np.zeros((1, len(ps_vals)))
            for j, ps in enumerate(ps_vals):
                scores = []
                for _ in range(5):
                    tlbo = TLBO(func, bounds, pop_size=ps)
                    _, s, _, _ = tlbo.optimize(iterations=50)
                    scores.append(s)
                tlbo_results[0, j] = np.mean(scores)
            visualization.plot_parameter_sensitivity(
                tlbo_results, ps_vals, ["TLBO"],
                "TLBO Sensitivity on Rastrigin (Pop Size)",
                "continuous/sensitivity/cont_sensitivity_tlbo",
                "Population Size", ""
            )
            save_sensitivity_csv(tlbo_results, ps_vals, ['TLBO'],
                                 'sensitivity_tlbo.csv', 'pop_size', '',
                                 subdir='continuous/sensitivity')


# =============================================================================
# GENERIC DISCRETE (TSP) COMPARISON ENGINE
# =============================================================================
class DiscreteComparison:
    """
    Runs any subset of registered TSP heuristics vs exact solvers on configurable sizes.
    """

    def __init__(self, algorithm_names=None, sizes=None, runs=30, pop_size=50):
        self.algorithm_names = algorithm_names or list(DISCRETE_ALGORITHMS.keys())
        self.sizes = sizes or [8, 9, 10]
        self.runs = runs
        self.pop_size = pop_size

    def run_all(self):
        print("\n" + "#"*60)
        print(f"DISCRETE COMPARISON: {self.algorithm_names} | Sizes: {self.sizes}")
        print("#"*60)

        self._scalability()
        self._quality_vs_optimal()
        self._sensitivity()

    def _scalability(self):
        print("\n[1] SCALABILITY")
        exact_times = {'BFS': [], 'DFS': [], 'A* (Exact)': []}
        heuristic_times = {name: [] for name in self.algorithm_names}

        for n in self.sizes:
            print(f"  -> N={n}")
            cities = problems.generate_cities(n, seed=42)
            dist = problems.calculate_distance_matrix(cities)
            solver = TSPGraphSearch(n, dist)

            s = time.time(); solver.bfs(); exact_times['BFS'].append(time.time() - s)
            s = time.time(); solver.dfs(); exact_times['DFS'].append(time.time() - s)
            s = time.time(); solver.a_star(); exact_times['A* (Exact)'].append(time.time() - s)

            for name in self.algorithm_names:
                run_times = []
                for _ in range(5):
                    s = time.time()
                    DISCRETE_ALGORITHMS[name](n, dist, generations=50, pop_size=self.pop_size)
                    run_times.append(time.time() - s)
                heuristic_times[name].append(np.mean(run_times))

        all_times = {**exact_times, **heuristic_times}
        
        # --- ĐỊNH NGHĨA FOLDER CHUNG CHO TSP ---
        folder = "discrete/tsp" 
        tag = '_'.join(self.algorithm_names)

        # CHỈ GỌI MỘT LẦN VỚI ĐƯỜNG DẪN ĐÃ ĐƯỢC GOM NHÓM
        visualization.plot_scalability_lines(
            self.sizes, all_times,
            f"Discrete Scalability: {' vs '.join(all_times.keys())}",
            f"{folder}/tsp_scalability_{tag}", # Đã xóa 'scalability/' ở giữa
            "Number of Cities", "Execution Time (s)"
        )
        
        # LƯU CSV VÀO CÙNG THƯ MỤC
        save_scalability_csv(self.sizes, all_times, f"tsp_scalability_{tag}.csv",
                             'Cities', subdir=folder)

    def _quality_vs_optimal(self):
        print("\n[2] QUALITY vs OPTIMAL")
        n = self.sizes[-1]
        cities = problems.generate_cities(n, seed=100)
        dist = problems.calculate_distance_matrix(cities)

        solver = TSPGraphSearch(n, dist)
        optimal_cost = solver.a_star()
        print(f"  Optimal (A*): {optimal_cost:.2f}")

        all_scores = {name: [] for name in self.algorithm_names}
        all_histories = {name: [] for name in self.algorithm_names}
        best_routes = {name: None for name in self.algorithm_names}
        best_costs = {name: float('inf') for name in self.algorithm_names}

        for _ in range(self.runs):
            for name in self.algorithm_names:
                cost, history, route = DISCRETE_ALGORITHMS[name](
                    n, dist, generations=100, pop_size=self.pop_size)
                all_scores[name].append(cost)
                all_histories[name].append(history)
                if cost < best_costs[name]:
                    best_costs[name] = cost
                    best_routes[name] = route

        scores_for_plot = {**all_scores, 'Exact (Fixed)': [optimal_cost] * self.runs}

        folder = "discrete/tsp" # Gom vào folder tsp
        visualization.plot_robustness_convergence(
            all_histories,
            f"Discrete Convergence: {' vs '.join(self.algorithm_names)}",
            f"{folder}/tsp_convergence_{'_'.join(self.algorithm_names)}"
        )
        visualization.plot_boxplot_comparison(
            scores_for_plot,
            f"Discrete Quality: {' vs '.join(self.algorithm_names)} vs Exact",
            f"{folder}/tsp_quality_{'_'.join(self.algorithm_names)}",
            ylabel="Path Cost"
        )

        # --- CSV EXPORT ---
        tag = '_'.join(self.algorithm_names)
        save_scores_csv(scores_for_plot, f"tsp_quality_{tag}.csv", subdir=folder)
        save_convergence_csv(all_histories, f"tsp_convergence_{tag}.csv", subdir=folder)

        for name in self.algorithm_names:
            mean = np.mean(all_scores[name])
            t = manual_onesample_ttest(all_scores[name], optimal_cost)
            print(f"  [{name}] Mean={mean:.2f}, Gap={((mean-optimal_cost)/optimal_cost*100):.2f}%, t={t:.4f}")
            if best_routes[name] is not None:
                visualization.plot_tsp_route(
                    cities, best_routes[name],
                    f"{name} Best Route (Cost {best_costs[name]:.2f})",
                    f"{folder}/tsp_best_route_{name.lower()}"
                )

    def _sensitivity(self):
        print("\n[3] SENSITIVITY ANALYSIS (TSP)")
        n = self.sizes[-1]
        cities = problems.generate_cities(n, seed=99)
        dist = problems.calculate_distance_matrix(cities)
        folder = "discrete/tsp"

        # --- 1. GA Sensitivity (Pop Size vs Mutation Rate) ---
        if "GA" in self.algorithm_names:
            print("  Running GA Sensitivity...")
            mut_rates, pop_sizes = [0.01, 0.1, 0.5], [20, 50, 100]
            results = np.zeros((len(mut_rates), len(pop_sizes)))
            for i, mr in enumerate(mut_rates):
                for j, ps in enumerate(pop_sizes):
                    costs = [GeneticAlgorithmTSP(n, dist, pop_size=ps, mutation_rate=mr).solve(generations=50)[1] for _ in range(5)]
                    results[i, j] = np.mean(costs)
            visualization.plot_parameter_sensitivity(results, pop_sizes, mut_rates, "GA TSP Sensitivity", f"{folder}/tsp_sens_ga", "Pop Size", "Mut Rate")

        # --- 2. ACO Sensitivity (Alpha vs Beta) ---
        if "ACO" in self.algorithm_names:
            print("  Running ACO Sensitivity...")
            alphas, betas = [0.5, 1.0, 2.0], [1.0, 3.0, 5.0]
            results = np.zeros((len(alphas), len(betas)))
            for i, a in enumerate(alphas):
                for j, b in enumerate(betas):
                    costs = [AntColonyOptimizationTSP(n, dist, alpha=a, beta=b).solve(iterations=50)[1] for _ in range(3)]
                    results[i, j] = np.mean(costs)
            visualization.plot_parameter_sensitivity(results, betas, alphas, "ACO TSP Sensitivity", f"{folder}/tsp_sens_aco", "Beta", "Alpha")

        # --- 3. SA Sensitivity (T_init vs Cooling Rate) ---
        if "SA" in self.algorithm_names:
            print("  Running SA Sensitivity...")
            temps, cools = [100, 1000, 5000], [0.95, 0.99, 0.999]
            results = np.zeros((len(temps), len(cools)))
            for i, t in enumerate(temps):
                for j, c in enumerate(cools):
                    costs = [SimulatedAnnealing(T_init=t, cooling_rate=c).solve_tsp(n, dist)[1] for _ in range(3)]
                    results[i, j] = np.mean(costs)
            visualization.plot_parameter_sensitivity(results, cools, temps, "SA TSP Sensitivity", f"{folder}/tsp_sens_sa", "Cooling Rate", "T_init")


# =============================================================================
# KNAPSACK COMPARISON
# =============================================================================
class KnapsackComparison:
    """
    So sánh TẤT CẢ thuật toán trên bài toán Knapsack.
    Algorithms: DFS, HC, GA, SA, PSO, DE, ABC, CS, FA, TLBO
    """

    def __init__(self, num_items_list=None, runs=30, pop_size=50):
        self.num_items_list = num_items_list or [10, 15, 20, 30]
        self.runs = runs
        self.pop_size = pop_size

        # Register all KP solvers
        self.KP_ALGORITHMS = {
            'DFS':  lambda w, v, c, ps, g: dfs_knapsack(w, v, c), # DFS vét cạn (Optimal)
            'HC':   lambda w, v, c, ps, g: hc_knapsack(w, v, c, max_iter=ps*g),
            'GA':   lambda w, v, c, ps, g: GeneticAlgorithmKnapsack(w, v, c, pop_size=ps).solve(generations=g),
            'SA':   lambda w, v, c, ps, g: sa_knapsack(w, v, c, max_iter=ps*g),
            'PSO':  lambda w, v, c, ps, g: pso_knapsack(w, v, c, num_particles=ps, iterations=g),
            'DE':   lambda w, v, c, ps, g: de_knapsack(w, v, c, pop_size=ps, generations=g),
            'ABC':  lambda w, v, c, ps, g: abc_knapsack(w, v, c, colony_size=ps, iterations=g),
            'CS':   lambda w, v, c, ps, g: cs_knapsack(w, v, c, n_nests=ps, iterations=g),
            'FA':   lambda w, v, c, ps, g: fa_knapsack(w, v, c, n_fireflies=ps, iterations=g),
            'TLBO': lambda w, v, c, ps, g: tlbo_knapsack(w, v, c, pop_size=ps, iterations=g),
        }

    def run_all(self):
        print("\n" + "#"*60)
        print("KNAPSACK PROBLEM — ALL ALGORITHMS")
        print("#"*60)

        self._quality_comparison()
        self._scalability()
        self._sensitivity()

    def _quality_comparison(self):
        print("\n[1] QUALITY COMPARISON — All Algorithms")
        n_items = 20
        weights, values, capacity = problems.generate_knapsack_problem(n_items, seed=42)

        all_values = {}    # name → [val per run]
        all_histories = {} # name → [history per run]

        for alg_name, runner in self.KP_ALGORITHMS.items():
            print(f"  Running {alg_name}...", end=' ')
            vals = []
            hists = []
            
            # DFS là deterministic (chạy bao nhiêu lần cũng ra 1 kết quả), nên chỉ chạy 1 lần cho lẹ
            run_count = 1 if alg_name == 'DFS' else self.runs
            
            for _ in range(run_count):
                _, val, hist = runner(weights, values, capacity, self.pop_size, 100)
                vals.append(val)
                hists.append(hist)
                
            if alg_name == 'DFS':
                # Nhân bản kết quả DFS ra cho đủ số lượng để vẽ boxplot không bị lỗi
                vals = vals * self.runs
                hists = hists * self.runs
                
            all_values[alg_name] = vals
            all_histories[alg_name] = hists
            print(f"Mean={np.mean(vals):.2f}, Std={np.std(vals):.2f}")

        # Lấy kết quả của DFS làm chuẩn (Optimal) để tính sai số (Gap)
        optimal = np.max(all_values['DFS'])
        print(f"  -> DFS Optimal Value (Baseline): {optimal}")

        # Convergence
        visualization.plot_robustness_convergence(
            all_histories,
            "Knapsack Convergence — All Algorithms",
            "discrete/knapsack/kp_convergence_all"
        )

        # Boxplot
        visualization.plot_boxplot_comparison(
            all_values,
            "Knapsack Quality — All Algorithms",
            "discrete/knapsack/kp_quality_all_boxplot",
            ylabel="Total Value"
        )

        # CSV
        save_scores_csv(all_values, 'kp_quality_all.csv', subdir='discrete/knapsack')
        save_convergence_csv(all_histories, 'kp_convergence_all.csv', subdir='discrete/knapsack')

        # Stats
        for name in self.KP_ALGORITHMS:
            mean = np.mean(all_values[name])
            gap = (optimal - mean) / optimal * 100 if optimal > 0 else 0
            t = manual_onesample_ttest(all_values[name], optimal)
            print(f"  [{name}] Mean={mean:.2f}, Gap to DFS={gap:.2f}%, t={t:.4f}")

    def _scalability(self):
        print("\n[2] SCALABILITY (Time vs Num Items)")
        times = {name: [] for name in self.KP_ALGORITHMS.keys()}

        for n_items in self.num_items_list:
            w, v, c = problems.generate_knapsack_problem(n_items, seed=42)

            # Chạy toàn bộ Heuristics (Tất cả các thuật toán)
            for name, runner in self.KP_ALGORITHMS.items():
                s = time.time()
                for _ in range(3): # Chạy 3 lần lấy trung bình cho lẹ
                    runner(w, v, c, self.pop_size, 50)
                times[name].append((time.time() - s) / 3)

        visualization.plot_scalability_lines(
            self.num_items_list, times,
            "Knapsack Scalability: All Algorithms",
            "discrete/knapsack/kp_scalability_all",
            "Number of Items", "Execution Time (s)"
        )
        save_scalability_csv(self.num_items_list, times, 'kp_scalability_all.csv', 'Num_Items', subdir='discrete/knapsack')

    def _sensitivity(self):
        print("\n[3] SENSITIVITY ANALYSIS (Knapsack) - FULL ALGORITHMS")
        w, v, c = problems.generate_knapsack_problem(20, seed=42)
        folder = "discrete/knapsack"
        import src.algorithms.knapsack_solvers as ks

        if 'GA' in self.KP_ALGORITHMS:
            print("  -> GA Sensitivity...")
            mr_list, ps_list = [0.01, 0.1, 0.2], [20, 50, 100]
            res = np.zeros((len(mr_list), len(ps_list)))
            for i, mr in enumerate(mr_list):
                for j, ps in enumerate(ps_list):
                    res[i,j] = np.mean([GeneticAlgorithmKnapsack(w,v,c,pop_size=ps, mutation_rate=mr).solve(generations=30)[1] for _ in range(3)])
            visualization.plot_parameter_sensitivity(res, ps_list, mr_list, "GA KP Sensitivity", f"{folder}/kp_sens_ga", "Pop Size", "Mut Rate")

        if 'SA' in self.KP_ALGORITHMS:
            print("  -> SA Sensitivity...")
            t_list, c_list = [500, 1000, 5000], [0.9, 0.95, 0.99]
            res = np.zeros((len(t_list), len(c_list)))
            for i, t in enumerate(t_list):
                for j, cl in enumerate(c_list):
                    res[i,j] = np.mean([sa_knapsack(w,v,c, max_iter=1000, T_init=t, cooling_rate=cl)[1] for _ in range(3)])
            visualization.plot_parameter_sensitivity(res, c_list, t_list, "SA KP Sensitivity", f"{folder}/kp_sens_sa", "Cooling Rate", "T_init")

        if 'DE' in self.KP_ALGORITHMS:
            print("  -> DE Sensitivity...")
            f_list, cr_list = [0.3, 0.5, 0.9], [0.1, 0.5, 0.9]
            res = np.zeros((len(f_list), len(cr_list)))
            for i, f_val in enumerate(f_list):
                for j, cr in enumerate(cr_list):
                    res[i,j] = np.mean([ks._run_continuous_for_kp(DifferentialEvolution, w, v, c, {'pop_size':30, 'mutation_factor':f_val, 'crossover_rate':cr}, {'generations':30})[1] for _ in range(3)])
            visualization.plot_parameter_sensitivity(res, cr_list, f_list, "DE KP Sensitivity", f"{folder}/kp_sens_de", "Crossover Rate", "Mutation Factor")

        if 'PSO' in self.KP_ALGORITHMS:
            print("  -> PSO Sensitivity...")
            w_list, c_list = [0.4, 0.7, 0.9], [1.0, 1.5, 2.0]
            res = np.zeros((len(w_list), len(c_list)))
            for i, w_val in enumerate(w_list):
                for j, c_val in enumerate(c_list):
                    res[i,j] = np.mean([ks._run_continuous_for_kp(ParticleSwarmOptimization, w, v, c, {'num_particles':30, 'w':w_val, 'c1':c_val, 'c2':c_val}, {'iterations':30})[1] for _ in range(3)])
            visualization.plot_parameter_sensitivity(res, c_list, w_list, "PSO KP Sensitivity", f"{folder}/kp_sens_pso", "C1=C2", "Inertia Weight")

        if 'ABC' in self.KP_ALGORITHMS:
            print("  -> ABC Sensitivity...")
            lim_list, col_list = [50, 100, 200], [20, 40, 60]
            res = np.zeros((len(lim_list), len(col_list)))
            for i, lim in enumerate(lim_list):
                for j, col in enumerate(col_list):
                    res[i,j] = np.mean([ks._run_continuous_for_kp(ArtificialBeeColony, w, v, c, {'colony_size':col, 'limit':lim}, {'iterations':30})[1] for _ in range(3)])
            visualization.plot_parameter_sensitivity(res, col_list, lim_list, "ABC KP Sensitivity", f"{folder}/kp_sens_abc", "Colony Size", "Limit")

        if 'FA' in self.KP_ALGORITHMS:
            print("  -> FA Sensitivity...")
            a_list, g_list = [0.1, 0.5, 1.0], [0.5, 1.0, 2.0]
            res = np.zeros((len(a_list), len(g_list)))
            for i, a_val in enumerate(a_list):
                for j, g_val in enumerate(g_list):
                    res[i,j] = np.mean([ks._run_continuous_for_kp(FireflyAlgorithm, w, v, c, {'n_fireflies':30, 'alpha':a_val, 'gamma':g_val}, {'iterations':30})[1] for _ in range(3)])
            visualization.plot_parameter_sensitivity(res, g_list, a_list, "FA KP Sensitivity", f"{folder}/kp_sens_fa", "Gamma", "Alpha")

        if 'CS' in self.KP_ALGORITHMS:
            print("  -> CS Sensitivity...")
            a_list, pa_list = [0.005, 0.01, 0.05], [0.1, 0.25, 0.4]
            res = np.zeros((len(a_list), len(pa_list)))
            for i, a_val in enumerate(a_list):
                for j, pa_val in enumerate(pa_list):
                    res[i,j] = np.mean([ks._run_continuous_for_kp(CuckooSearch, w, v, c, {'n_nests':30, 'alpha':a_val, 'pa':pa_val}, {'iterations':30})[1] for _ in range(3)])
            visualization.plot_parameter_sensitivity(res, pa_list, a_list, "CS KP Sensitivity", f"{folder}/kp_sens_cs", "Pa", "Alpha")

        if 'TLBO' in self.KP_ALGORITHMS:
            print("  -> TLBO Sensitivity...")
            ps_list = [10, 20, 50, 100]
            res = np.zeros((1, len(ps_list)))
            for j, ps in enumerate(ps_list):
                res[0,j] = np.mean([ks._run_continuous_for_kp(TLBO, w, v, c, {'pop_size':ps}, {'iterations':30})[1] for _ in range(3)])
            visualization.plot_parameter_sensitivity(res, ps_list, ["TLBO"], "TLBO KP Sensitivity", f"{folder}/kp_sens_tlbo", "Pop Size", "")


# =============================================================================
# GRAPH COLORING COMPARISON
# =============================================================================
class GraphColoringComparison:
    """
    So sánh các thuật toán GA, SA, ACO, HC, DFS trên bài toán Graph Coloring.
    """

    def __init__(self, num_nodes_list=None, edge_prob=0.5, runs=30):
        self.num_nodes_list = num_nodes_list or [10, 15, 20, 30]
        self.edge_prob = edge_prob
        self.runs = runs

    def run_all(self):
        print("\n" + "#"*60)
        print("GRAPH COLORING — FULL ALGORITHMS")
        print("#"*60)
        self._quality_comparison()
        self._scalability()
        self._sensitivity()

    def _quality_comparison(self):
        print("\n[1] QUALITY COMPARISON - GC (GA, SA, ACO, HC, DFS)")
        n = 20
        adj, edges = generate_random_graph(n, self.edge_prob, seed=42)

        # Greedy baseline chỉ dùng để lấy số màu làm chuẩn (không đưa vào so sánh)
        _, num_colors = greedy_graph_coloring(adj)
        print(f"  Graph has {n} nodes. Using {num_colors} colors as target.")

        all_conflicts = {'GA': [], 'SA': [], 'ACO': [], 'HC': [], 'DFS': []}
        all_histories = {'GA': [], 'SA': [], 'ACO': [], 'HC': [], 'DFS': []}

        for _ in range(self.runs):
            # 1. GA
            _, c, h = ga_graph_coloring(adj, num_colors=num_colors, pop_size=30, generations=100)
            all_conflicts['GA'].append(c); all_histories['GA'].append(h)
            
            # 2. SA
            _, c, h = sa_graph_coloring(adj, num_colors=num_colors, max_iter=3000)
            all_conflicts['SA'].append(c); all_histories['SA'].append(h)
            
            # 3. ACO
            aco = AntColonyOptimizationGC(adj, num_colors, num_ants=30)
            _, c, h = aco.solve(iterations=100)
            all_conflicts['ACO'].append(c); all_histories['ACO'].append(h)
            
            # 4. HC
            _, c, h = hc_graph_coloring(adj, num_colors=num_colors, max_iter=3000)
            all_conflicts['HC'].append(c); all_histories['HC'].append(h)
            
            # 5. DFS
            _, c, h = dfs_graph_coloring(adj, num_colors=num_colors)
            all_conflicts['DFS'].append(c); all_histories['DFS'].append(h)

        # Plot & CSV
        folder = "discrete/graph_coloring"
        visualization.plot_robustness_convergence(all_histories, f"GC Convergence", f"{folder}/gc_convergence")
        visualization.plot_boxplot_comparison(all_conflicts, f"GC Conflicts Comparison", f"{folder}/gc_boxplot", ylabel="Conflicts")
        save_scores_csv(all_conflicts, 'gc_quality.csv', subdir=folder)
        save_convergence_csv(all_histories, 'gc_convergence.csv', subdir=folder)

        for alg in all_conflicts.keys():
            print(f"  [{alg}] Mean Conflicts: {np.mean(all_conflicts[alg]):.2f}")

    def _scalability(self):
        print("\n[2] SCALABILITY")
        # ĐÃ XÓA GREEDY KHỎI DICTIONARY NÀY
        times = {'GA': [], 'SA': [], 'ACO': [], 'HC': [], 'DFS': []}

        for n in self.num_nodes_list:
            adj, _ = generate_random_graph(n, self.edge_prob, seed=42)
            
            # Chỉ chạy Greedy ngầm 1 lần để mượn số màu (nc) làm target
            _, nc = greedy_graph_coloring(adj)

            s = time.time(); [ga_graph_coloring(adj, num_colors=nc, pop_size=30, generations=50) for _ in range(3)]; times['GA'].append((time.time()-s)/3)
            s = time.time(); [sa_graph_coloring(adj, num_colors=nc, max_iter=1500) for _ in range(3)]; times['SA'].append((time.time()-s)/3)
            s = time.time(); [AntColonyOptimizationGC(adj, nc, num_ants=20).solve(iterations=50) for _ in range(3)]; times['ACO'].append((time.time()-s)/3)
            s = time.time(); [hc_graph_coloring(adj, num_colors=nc, max_iter=1500) for _ in range(3)]; times['HC'].append((time.time()-s)/3)
            
            # DFS chạy vét cạn nên rất lâu, chỉ đo 1 lần cho biểu đồ
            s = time.time(); dfs_graph_coloring(adj, num_colors=nc); times['DFS'].append(time.time()-s)

        visualization.plot_scalability_lines(
            self.num_nodes_list, times,
            "Graph Coloring Scalability: All Solvers",
            "discrete/graph_coloring/gc_scalability_all",
            "Number of Nodes", "Execution Time (s)"
        )
        save_scalability_csv(self.num_nodes_list, times, 'gc_scalability.csv', 'Nodes', subdir='discrete/graph_coloring')

    def _sensitivity(self):
        print("\n[3] SENSITIVITY ANALYSIS (Graph Coloring)")
        n = 20
        adj, _ = generate_random_graph(n, self.edge_prob, seed=42)
        _, nc = greedy_graph_coloring(adj)
        folder = "discrete/graph_coloring"

        print("  -> GA Sensitivity...")
        mr_list, ps_list = [0.01, 0.1, 0.3], [20, 50, 100]
        res = np.zeros((len(mr_list), len(ps_list)))
        for i, mr in enumerate(mr_list):
            for j, ps in enumerate(ps_list):
                res[i,j] = np.mean([ga_graph_coloring(adj, nc, pop_size=ps, generations=30, mutation_rate=mr)[1] for _ in range(3)])
        visualization.plot_parameter_sensitivity(res, ps_list, mr_list, "GA GC Sensitivity", f"{folder}/gc_sens_ga", "Pop Size", "Mut Rate")

        print("  -> SA Sensitivity...")
        t_list, c_list = [10, 100, 500], [0.9, 0.95, 0.99]
        res = np.zeros((len(t_list), len(c_list)))
        for i, t in enumerate(t_list):
            for j, c in enumerate(c_list):
                res[i,j] = np.mean([sa_graph_coloring(adj, nc, max_iter=1000, T_init=t, cooling_rate=c)[1] for _ in range(3)])
        visualization.plot_parameter_sensitivity(res, c_list, t_list, "SA GC Sensitivity", f"{folder}/gc_sens_sa", "Cooling Rate", "T_init")

        print("  -> ACO Sensitivity...")
        a_list, b_list = [0.5, 1.0, 2.0], [1.0, 2.0, 5.0]
        res = np.zeros((len(a_list), len(b_list)))
        for i, a in enumerate(a_list):
            for j, b in enumerate(b_list):
                res[i,j] = np.mean([AntColonyOptimizationGC(adj, nc, num_ants=20, alpha=a, beta=b).solve(30)[1] for _ in range(3)])
        visualization.plot_parameter_sensitivity(res, b_list, a_list, "ACO GC Sensitivity", f"{folder}/gc_sens_aco", "Beta", "Alpha")


# =============================================================================
# SHORTEST PATH COMPARISON
# =============================================================================
class ShortestPathComparison:
    """
    So sánh A*, BFS, DFS và ACO trên bài toán Shortest Path.
    A* = heuristic-guided. BFS = fewest edges. DFS = any path. ACO = meta-heuristic.
    """

    def __init__(self, num_nodes_list=None, edge_prob=0.4, runs=30):
        self.num_nodes_list = num_nodes_list or [20, 50, 100, 200]
        self.edge_prob = edge_prob
        self.runs = runs

    def run_all(self):
        print("\n" + "#"*60)
        print("SHORTEST PATH — FULL ALGORITHMS")
        print("#"*60)
        self._quality_comparison()
        self._scalability()
        self._sensitivity()

    def _quality_comparison(self):
        print("\n[1] QUALITY COMPARISON & CONVERGENCE")
        n = 50
        folder = "discrete/shortest_path"

        # ==========================================
        # PHẦN A: BOXPLOT (Test trên 30 đồ thị khác nhau)
        # ==========================================
        all_costs = {'A*': [], 'BFS': [], 'DFS': [], 'ACO': []}
        all_explored = {'A*': [], 'BFS': [], 'DFS': [], 'ACO': []}

        for run in range(self.runs):
            adj_r, pos_r = generate_weighted_graph(n, self.edge_prob, seed=run*7+1)
            s, g = 0, n - 1

            _, c, e = a_star_shortest(adj_r, pos_r, s, g)
            all_costs['A*'].append(c); all_explored['A*'].append(e)

            _, c, e = bfs_shortest(adj_r, s, g)
            all_costs['BFS'].append(c); all_explored['BFS'].append(e)

            _, c, e = dfs_shortest(adj_r, s, g)
            all_costs['DFS'].append(c); all_explored['DFS'].append(e)

            # Lấy vị trí [1] là cost, [2] là explored
            aco = AntColonyOptimizationSP(adj_r, s, g, num_ants=20)
            res = aco.solve(iterations=50)
            all_costs['ACO'].append(res[1])
            all_explored['ACO'].append(res[2])

        visualization.plot_boxplot_comparison(all_costs, f"SP Path Cost", f"{folder}/sp_cost_boxplot", ylabel="Cost")
        visualization.plot_boxplot_comparison({k: [float(v) for v in vals] for k, vals in all_explored.items()}, f"SP Nodes Explored", f"{folder}/sp_explored_boxplot", ylabel="Nodes")
        save_scores_csv(all_costs, 'sp_cost.csv', subdir=folder)

        # ==========================================
        # PHẦN B: CONVERGENCE (Test nhiều lần trên 1 đồ thị duy nhất)
        # ==========================================
        adj_fixed, pos_fixed = generate_weighted_graph(n, self.edge_prob, seed=999)
        s_f, g_f = 0, n - 1

        # Lấy cost cố định của các baselines
        _, c_astar, _ = a_star_shortest(adj_fixed, pos_fixed, s_f, g_f)
        _, c_bfs, _ = bfs_shortest(adj_fixed, s_f, g_f)
        _, c_dfs, _ = dfs_shortest(adj_fixed, s_f, g_f)

        # Thêm biến lưu số vòng lặp của ACO
        aco_iters = 1000 # Hoặc 50, 100 tùy bạn thiết lập ở aco.solve()

        # Baselines vẽ đường thẳng dài bằng với số vòng lặp của ACO
        all_histories = {
            'A*': [[c_astar] * aco_iters] * 5,
            'BFS': [[c_bfs] * aco_iters] * 5,
            'DFS': [[c_dfs] * aco_iters] * 5,
            'ACO': []
        }

        # Chạy ACO 5 lần để lấy lịch sử tiến hóa
        for _ in range(5):
            aco = AntColonyOptimizationSP(adj_fixed, s_f, g_f, num_ants=20)
            res = aco.solve(iterations=aco_iters) # Dùng biến ở đây
            h = res[3] if len(res) > 3 else [res[1]] * aco_iters 
            all_histories['ACO'].append(h)

        visualization.plot_robustness_convergence(all_histories, "Shortest Path Convergence: ACO vs Baselines", f"{folder}/sp_convergence_all")
        save_convergence_csv(all_histories, 'sp_convergence_all.csv', subdir=folder)

    def _scalability(self):
        print("\n[2] SCALABILITY")
        # Đã xóa Dijkstra
        times = {'A*': [], 'BFS': [], 'DFS': [], 'ACO': []}

        for n in self.num_nodes_list:
            adj, pos = generate_weighted_graph(n, self.edge_prob, seed=42)
            s, g = 0, n - 1

            for name, fn in [('A*', lambda: a_star_shortest(adj, pos, s, g)),
                             ('BFS', lambda: bfs_shortest(adj, s, g)),
                             ('DFS', lambda: dfs_shortest(adj, s, g)),
                             ('ACO', lambda: AntColonyOptimizationSP(adj, s, g, num_ants=20).solve(20))]:
                t0 = time.time()
                for _ in range(3): fn()
                times[name].append((time.time() - t0) / 3)

        visualization.plot_scalability_lines(
            self.num_nodes_list, times,
            "Shortest Path Scalability (All Solvers)",
            "discrete/shortest_path/sp_scalability_all",
            "Number of Nodes", "Execution Time (s)"
        )
        save_scalability_csv(self.num_nodes_list, times, 'sp_scalability_all.csv', 'Nodes', subdir='discrete/shortest_path')

    def _sensitivity(self):
        print("\n[3] SENSITIVITY ANALYSIS (Shortest Path)")
        adj, _ = generate_weighted_graph(30, self.edge_prob, seed=42)
        s, g = 0, 29
        folder = "discrete/shortest_path"

        print("  -> ACO Sensitivity...")
        a_list, b_list = [0.5, 1.0, 2.0], [1.0, 3.0, 5.0]
        res = np.zeros((len(a_list), len(b_list)))
        for i, a in enumerate(a_list):
            for j, b in enumerate(b_list):
                # Lưu cost của đường đi
                res[i,j] = np.mean([AntColonyOptimizationSP(adj, s, g, num_ants=20, alpha=a, beta=b).solve(20)[1] for _ in range(3)])
        visualization.plot_parameter_sensitivity(res, b_list, a_list, "ACO SP Sensitivity (Cost)", f"{folder}/sp_sens_aco", "Beta", "Alpha")

# EXPLORATION / 3D VISUALIZATION (standalone utility)
# =============================================================================
def run_exploration_visualization(algorithm_names=None, problem_name="Rastrigin"):
    """Plot 3D landscape trajectories for 2D functions."""
    algorithm_names = algorithm_names or list(CONTINUOUS_ALGORITHMS.keys())

    # Chọn hàm và bounds 2D phù hợp
    func_map = {
        "Rastrigin":  (problems.rastrigin_function,  [[-5.12, 5.12], [-5.12, 5.12]]),
        "Ackley":     (problems.ackley_function,     [[-5, 5], [-5, 5]]),
        "Sphere":     (problems.sphere_function,     [[-5.12, 5.12], [-5.12, 5.12]]),
        "Rosenbrock": (problems.rosenbrock_function, [[-2, 2], [-1, 3]]),
        "Griewank":   (problems.griewank_function,   [[-10, 10], [-10, 10]]),
    }

    func, bounds_2d = func_map.get(problem_name,
                                    (problems.rastrigin_function, [[-5.12, 5.12], [-5.12, 5.12]]))

    paths = {}
    for name in algorithm_names:
        if name == "HC":
            hc = ContinuousLocalSearch(step_size=0.2, max_iter=100)
            _, _, _, path = hc.hill_climbing(func, bounds_2d)
            paths["HC (Exploitation)"] = path
        elif name == "DE":
            de = DifferentialEvolution(func, bounds_2d, pop_size=20)
            _, _, _, path = de.optimize(generations=20)
            paths["DE (Exploration)"] = path
        elif name == "CS":
            cs = CuckooSearch(func, bounds_2d, n_nests=20, pa=0.25, alpha=0.02)
            _, _, _, path = cs.optimize(iterations=20)
            paths["CS (Levy Flights)"] = path
        elif name == "TLBO":
            tlbo = TLBO(func, bounds_2d, pop_size=20)
            _, _, _, path = tlbo.optimize(iterations=20)
            paths["TLBO (Teacher-Learner)"] = path
        elif name == "PSO":
            pso = ParticleSwarmOptimization(func, bounds_2d, num_particles=20)
            _, _, _, path = pso.optimize(iterations=20)
            paths["PSO (Swarm)"] = path
        elif name == "ABC":
            abc = ArtificialBeeColony(func, bounds_2d, colony_size=20)
            _, _, _, path = abc.optimize(iterations=20)
            paths["ABC (Bee Colony)"] = path
        elif name == "SA":
            sa = SimulatedAnnealing(T_init=1000.0, cooling_rate=0.98, max_iter=500)
            _, _, _, path = sa.optimize(func, bounds_2d)
            paths["SA (Annealing)"] = path
        elif name == "FA":
            fa = FireflyAlgorithm(func, bounds_2d, n_fireflies=20, gamma=0.5)
            _, _, _, path = fa.optimize(iterations=20)
            paths["FA (Firefly)"] = path

    tag = '_vs_'.join(algorithm_names)
    folder_path = f"continuous/{problem_name.lower()}" # Lưu vào folder tên hàm
    visualization.plot_3d_landscape_path(
        func, bounds_2d, paths,
        f"Trajectory on {problem_name}: {' vs '.join(algorithm_names)}",
        f"{folder_path}/cont_3d_{tag}"
    )


# =============================================================================
# ENTRY POINT — Compose your experiments here
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists('results/figures'):
        os.makedirs('results/figures')

# ================================================================
    # PART 1: DISCRETE EXPERIMENTS
    # ================================================================

    # --- TSP ---
    DiscreteComparison(algorithm_names=["GA", "ACO", "SA", "HC"], sizes=[8, 9, 10]).run_all()

    # --- Knapsack Problem ---
    KnapsackComparison(num_items_list=[10, 15, 20, 30]).run_all()

    # --- Graph Coloring ---
    GraphColoringComparison(num_nodes_list=[10, 15, 20]).run_all()

    # --- Shortest Path ---
    ShortestPathComparison(num_nodes_list=[20, 50, 100]).run_all()

    """
    # ================================================================
    # PART 2: CONTINUOUS EXPERIMENTS — Pairwise & Group Comparisons
    # ================================================================

    # --- DE vs HC on Rastrigin (baseline comparison) ---
    ContinuousComparison(
        algorithm_names=["DE", "HC"],
        problem_names=["Rastrigin"]
    ).run_all()

    # --- CS vs HC on Rastrigin ---
    ContinuousComparison(
        algorithm_names=["CS", "HC"],
        problem_names=["Rastrigin"]
    ).run_all()

    # --- SA vs HC (Physics baseline) ---
    ContinuousComparison(
        algorithm_names=["SA", "HC"],
        problem_names=["Sphere", "Rastrigin"]
    ).run_all()

    # --- DE vs TLBO ---
    ContinuousComparison(
        algorithm_names=["DE", "TLBO"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()

    # --- CS vs TLBO ---
    ContinuousComparison(
        algorithm_names=["CS", "TLBO"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()

    # --- DE vs PSO ---
    ContinuousComparison(
        algorithm_names=["DE", "PSO"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()

    # --- DE vs ABC ---
    ContinuousComparison(
        algorithm_names=["DE", "ABC"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()

    # --- PSO vs ABC vs CS (biology swarm showdown) ---
    ContinuousComparison(
        algorithm_names=["PSO", "ABC", "CS"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()

    # --- SA vs population methods ---
    ContinuousComparison(
        algorithm_names=["SA", "DE", "PSO"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()

    # --- FA pairwise ---
    ContinuousComparison(
        algorithm_names=["FA", "CS"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()

    ContinuousComparison(
        algorithm_names=["FA", "PSO"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()

    # --- FA vs ABC vs CS (biology trio) ---
    ContinuousComparison(
        algorithm_names=["FA", "ABC", "CS"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()

    # --- FA vs SA ---
    ContinuousComparison(
        algorithm_names=["FA", "SA"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()

    # ================================================================
    # PART 3: GRAND COMPARISON — ALL algorithms × ALL 5 functions
    # ================================================================
    ContinuousComparison(
        algorithm_names=["DE", "CS", "HC", "TLBO", "PSO", "ABC", "FA", "SA"],
        problem_names=["Sphere", "Rastrigin", "Rosenbrock", "Griewank", "Ackley"],
        runs=30
    ).run_all()

    # ================================================================
    # PART 4: 3D TRAJECTORY VISUALIZATIONS
    # ================================================================

    # --- Rastrigin trajectories ---
    run_exploration_visualization(algorithm_names=["DE", "HC"], problem_name="Rastrigin")
    run_exploration_visualization(algorithm_names=["CS", "HC"], problem_name="Rastrigin")
    run_exploration_visualization(algorithm_names=["DE", "TLBO"], problem_name="Rastrigin")
    run_exploration_visualization(algorithm_names=["PSO", "HC"], problem_name="Rastrigin")
    run_exploration_visualization(algorithm_names=["ABC", "HC"], problem_name="Rastrigin")
    run_exploration_visualization(algorithm_names=["SA", "HC", "DE"], problem_name="Rastrigin")
    run_exploration_visualization(algorithm_names=["DE", "PSO", "ABC"], problem_name="Rastrigin")
    run_exploration_visualization(algorithm_names=["FA", "CS", "HC"], problem_name="Rastrigin")
    run_exploration_visualization(algorithm_names=["FA", "PSO", "DE"], problem_name="Rastrigin")

    # --- Ackley trajectories (visually interesting) ---
    run_exploration_visualization(algorithm_names=["DE", "HC"], problem_name="Ackley")
    run_exploration_visualization(algorithm_names=["PSO", "CS", "FA"], problem_name="Ackley")
    run_exploration_visualization(algorithm_names=["SA", "TLBO"], problem_name="Ackley")

    # --- Rosenbrock trajectories (narrow valley) ---
    run_exploration_visualization(algorithm_names=["DE", "HC"], problem_name="Rosenbrock")
    run_exploration_visualization(algorithm_names=["PSO", "TLBO"], problem_name="Rosenbrock")
    """
    print("\n" + "="*60)
    print("HOÀN THÀNH. KIỂM TRA FOLDER RESULTS/FIGURES.")
    print("="*60)