import numpy as np
import os
import time
from src.utils import problems, visualization

# Import Algorithms
from src.algorithms.evolution.genetic_algorithm import GeneticAlgorithmTSP
from src.algorithms.evolution.differential_evolution import DifferentialEvolution
from src.algorithms.biology.cuckoo_search import CuckooSearch
from src.algorithms.biology.pso import ParticleSwarmOptimization
from src.algorithms.biology.abc import ArtificialBeeColony
from src.algorithms.biology.aco import AntColonyOptimizationTSP
from src.algorithms.classical.baselines import TSPGraphSearch, ContinuousLocalSearch
from src.algorithms.human.tlbo import TLBO

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
    # colony_size covers both employed + onlooker bees; pop_size maps naturally
    abc = ArtificialBeeColony(func, bounds, colony_size=pop_size)
    _, score, history, path = abc.optimize(iterations=generations)
    return score, history, path


# --- Register TSP Algorithms ---
# Each runner signature: (n, dist, generations, pop_size) -> (score, history, route)

@register_tsp("GA")
def run_ga_tsp(n, dist, generations, pop_size):
    ga = GeneticAlgorithmTSP(n, dist, pop_size=pop_size)
    route, cost, history = ga.solve(generations=generations)
    return cost, history, route

@register_tsp("ACO")
def run_aco_tsp(n, dist, generations, pop_size):
    # num_ants maps to pop_size; other params are sensible defaults
    aco = AntColonyOptimizationTSP(
        n, dist,
        num_ants=pop_size,
        alpha=1.0,
        beta=3.0,
        evaporation_rate=0.5,
        Q=100.0
    )
    route, cost, history = aco.solve(iterations=generations)
    return cost, history, route


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
    # Add new problems here without touching any other code:
    # "Ackley": {
    #     "func": problems_evo.ackley_function,
    #     "bounds": [[-32, 32]] * 10,
    #     "generations": 100,
    # },
}

DISCRETE_PROBLEMS = {
    "TSP": {
        "sizes": [8, 9, 10],
        "generate_fn": problems.generate_cities,
        "dist_fn": problems.calculate_distance_matrix,
    },
}

# =============================================================================
# GENERIC COMPARISON ENGINE
# =============================================================================
class ContinuousComparison:
    """
    Runs any subset of registered algorithms on any subset of registered problems.
    Add algorithms via @register_continuous. Add problems to CONTINUOUS_PROBLEMS.
    """

    def __init__(self, algorithm_names=None, problem_names=None, runs=30, pop_size=50):
        """
        Args:
            algorithm_names: List of algorithm names to compare. None = all registered.
            problem_names:   List of problem names to test on. None = all registered.
            runs:            Number of repetitions for statistical analysis.
            pop_size:        Population size (used by population-based algorithms).
        """
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

        # Convergence plot
        visualization.plot_robustness_convergence(
            all_histories,
            f"Convergence on {prob_name}: {' vs '.join(self.algorithm_names)}",
            f"continuous/convergence/{tag}_convergence"
        )

        # Boxplot
        visualization.plot_boxplot_comparison(
            all_scores,
            f"Quality on {prob_name}: {' vs '.join(self.algorithm_names)}",
            f"continuous/quality/{tag}_boxplot"
        )

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
        visualization.plot_scalability_lines(
            dims, times,
            f"Scalability: {' vs '.join(self.algorithm_names)} on Rastrigin (Time)",
            f"continuous/scalability/cont_scalability_time_{tag}",
            "Dimensions (D)", "Execution Time (s)"
        )

    def _sensitivity_continuous(self):
        bounds = [[-5.12, 5.12]] * 10
        func = problems.rastrigin_function

        if "DE" in self.algorithm_names:
            print("\n[SENSITIVITY] DE on Rastrigin (F vs CR)")
            F_vals = [0.3, 0.5, 0.9]
            CR_vals = [0.1, 0.5, 0.9]
            results = np.zeros((len(F_vals), len(CR_vals)))
            for i, f in enumerate(F_vals):
                for j, cr in enumerate(CR_vals):
                    scores = []
                    for _ in range(5):
                        de = DifferentialEvolution(func, bounds, pop_size=30, mutation_factor=f, crossover_rate=cr)
                        _, s, _, _ = de.optimize(generations=50)
                        scores.append(s)
                    results[i, j] = np.mean(scores)
            visualization.plot_parameter_sensitivity(
                results, CR_vals, F_vals,
                "DE Sensitivity Analysis on Rastrigin Function",
                "continuous/sensitivity/cont_sensitivity",
                "Crossover Rate (CR)", "Mutation Factor (F)"
            )

        if "CS" in self.algorithm_names:
            print("\n[SENSITIVITY] CS on Rastrigin (alpha vs pa)")
            alpha_vals = [0.005, 0.01, 0.05]
            pa_vals = [0.1, 0.25, 0.4]
            cs_results = np.zeros((len(alpha_vals), len(pa_vals)))
            for i, alpha in enumerate(alpha_vals):
                for j, pa in enumerate(pa_vals):
                    scores = []
                    for _ in range(5):
                        cs = CuckooSearch(func, bounds, n_nests=30, pa=pa, alpha=alpha, beta=1.5)
                        _, s, _, _ = cs.optimize(iterations=50)
                        scores.append(s)
                    cs_results[i, j] = np.mean(scores)
            visualization.plot_parameter_sensitivity(
                cs_results, pa_vals, alpha_vals,
                "Cuckoo Search Sensitivity on Rastrigin Function", "continuous/sensitivity/cont_sensitivity_cs",
                "Abandonment Probability (pa)", "Step Size (alpha)"
            )


class DiscreteComparison:
    """
    Runs any subset of registered TSP heuristics vs exact solvers on configurable city sizes.
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
        visualization.plot_scalability_lines(
            self.sizes, all_times,
            f"Discrete Scalability: {' vs '.join(all_times.keys())}",
            f"discrete/scalability/tsp_scalability_{'_'.join(self.algorithm_names)}",
            "Number of Cities", "Execution Time (s)"
        )

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
                cost, history, route = DISCRETE_ALGORITHMS[name](n, dist, generations=100, pop_size=self.pop_size)
                all_scores[name].append(cost)
                all_histories[name].append(history)
                if cost < best_costs[name]:
                    best_costs[name] = cost
                    best_routes[name] = route

        scores_for_plot = {**all_scores, 'Exact (Fixed)': [optimal_cost] * self.runs}

        visualization.plot_robustness_convergence(
            all_histories,
            f"Discrete Convergence: {' vs '.join(self.algorithm_names)}",
            f"discrete/convergence/tsp_convergence_{'_'.join(self.algorithm_names)}"
        )
        visualization.plot_boxplot_comparison(
            scores_for_plot,
            f"Discrete Quality: {' vs '.join(self.algorithm_names)} vs Exact",
            f"discrete/quality/tsp_quality_{'_'.join(self.algorithm_names)}",
            ylabel="Path Cost"
        )

        for name in self.algorithm_names:
            mean = np.mean(all_scores[name])
            t = manual_onesample_ttest(all_scores[name], optimal_cost)
            print(f"  [{name}] Mean={mean:.2f}, Gap={((mean-optimal_cost)/optimal_cost*100):.2f}%, t={t:.4f}")
            if best_routes[name] is not None:
                visualization.plot_tsp_route(
                    cities, best_routes[name],
                    f"{name} Best Route (Cost {best_costs[name]:.2f})",
                    f"discrete/routes/tsp_best_route_{name.lower()}"
                )

    def _sensitivity(self):
        print("\n[3] SENSITIVITY (GA)")
        if "GA" not in self.algorithm_names:
            return
        mut_rates = [0.01, 0.1, 0.2, 0.5]
        pop_sizes = [20, 50, 100]
        results = np.zeros((len(mut_rates), len(pop_sizes)))
        n = self.sizes[-1]
        cities = problems.generate_cities(n, seed=99)
        dist = problems.calculate_distance_matrix(cities)

        for i, mr in enumerate(mut_rates):
            for j, ps in enumerate(pop_sizes):
                costs = []
                for _ in range(5):
                    ga = GeneticAlgorithmTSP(n, dist, pop_size=ps, mutation_rate=mr)
                    _, c, _ = ga.solve(generations=50)
                    costs.append(c)
                results[i, j] = np.mean(costs)

        tag = '_'.join(self.algorithm_names)
        visualization.plot_parameter_sensitivity(
            results, pop_sizes, mut_rates,
            "GA Sensitivity Analysis",
            f"discrete/sensitivity/tsp_sensitivity_{tag}",
            "Population Size", "Mutation Rate"
        )


# =============================================================================
# EXPLORATION / 3D VISUALIZATION (standalone utility)
# =============================================================================
def run_exploration_visualization(algorithm_names=None, problem_name="Rastrigin"):
    """Plot 3D landscape trajectories"""
    algorithm_names = algorithm_names or list(CONTINUOUS_ALGORITHMS.keys())
    bounds_2d = [[-5.12, 5.12], [-5.12, 5.12]]
    func = problems.rastrigin_function

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

    tag = '_vs_'.join(algorithm_names)
    visualization.plot_3d_landscape_path(
        func, bounds_2d, paths,
        f"Trajectory Comparison: {' vs '.join(algorithm_names)}",
        f"continuous/3d/cont_3d_{tag}_rastrigin"
    )


# =============================================================================
# ENTRY POINT — Compose your experiments here
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists('results/figures'):
        os.makedirs('results/figures')

    # --- Discrete Experiments ---
    # GA only (exact methods are the baseline inside DiscreteComparison)
    DiscreteComparison(algorithm_names=["GA"], sizes=[8, 9, 10]).run_all()

    # GA + ACO head-to-head
    DiscreteComparison(algorithm_names=["GA", "ACO"], sizes=[8, 9, 10]).run_all()

    # --- Continuous Experiments ---
    # All algorithms on all problems:
    #ContinuousComparison().run_all()

    # Custom:
    ContinuousComparison(algorithm_names=["DE", "HC"], problem_names=["Rastrigin"]).run_all()
    ContinuousComparison(algorithm_names=["CS", "HC"], problem_names=["Rastrigin"]).run_all()
    ContinuousComparison(
        algorithm_names=["DE", "TLBO"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()
    ContinuousComparison(
        algorithm_names=["CS", "TLBO"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()

    # New pairwise: PSO and ABC vs existing algorithms
    ContinuousComparison(
        algorithm_names=["DE", "PSO"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()
    ContinuousComparison(
        algorithm_names=["DE", "ABC"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()
    ContinuousComparison(
        algorithm_names=["PSO", "ABC", "CS"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()

    # Full 7-way comparison across all registered continuous algorithms
    ContinuousComparison(
        algorithm_names=["DE", "CS", "HC", "TLBO", "PSO", "ABC"],
        problem_names=["Sphere", "Rastrigin"],
        runs=30
    ).run_all()

    # --- 3D Trajectory Visualizations ---
    run_exploration_visualization(algorithm_names=["DE", "HC"], problem_name="Rastrigin")
    run_exploration_visualization(algorithm_names=["CS", "HC"], problem_name="Rastrigin")
    run_exploration_visualization(algorithm_names=["DE", "TLBO"], problem_name="Rastrigin")
    run_exploration_visualization(algorithm_names=["PSO", "HC"], problem_name="Rastrigin")
    run_exploration_visualization(algorithm_names=["ABC", "HC"], problem_name="Rastrigin")
    # Full 6-way trajectory (only 3 colours in plot_3d_landscape_path — pick key 3)
    run_exploration_visualization(
        algorithm_names=["DE", "PSO", "ABC"],
        problem_name="Rastrigin"
    )

    print("\n" + "="*60)
    print("HOÀN THÀNH. KIỂM TRA FOLDER RESULTS/FIGURES.")
    print("="*60)