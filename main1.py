import numpy as np
import os
import time
from src.utils import problems_evo, visualization_evo

# Import Algorithms
from src.algorithms.evolution.genetic_algorithm import GeneticAlgorithmTSP
from src.algorithms.evolution.differential_evolution import DifferentialEvolution
from src.algorithms.biology.cuckoo_search import CuckooSearch
from src.algorithms.classical.baselines_evo import TSPGraphSearch, ContinuousLocalSearch

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
TSP_ALGORITHMS = {}

def register_continuous(name):
    """Decorator to register a continuous optimization algorithm runner."""
    def decorator(fn):
        CONTINUOUS_ALGORITHMS[name] = fn
        return fn
    return decorator

def register_tsp(name):
    """Decorator to register a TSP algorithm runner."""
    def decorator(fn):
        TSP_ALGORITHMS[name] = fn
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

# To add a new algorithm, just add a new block like this:
# @register_continuous("PSO")
# def run_pso(func, bounds, generations, pop_size):
#     pso = ParticleSwarmOptimization(func, bounds, n_particles=pop_size)
#     _, score, history, path = pso.optimize(iterations=generations)
#     return score, history, path


# --- Register TSP Algorithms ---
# Each runner signature: (n, dist, generations, pop_size) -> (score, history, route)

@register_tsp("GA")
def run_ga_tsp(n, dist, generations, pop_size):
    ga = GeneticAlgorithmTSP(n, dist, pop_size=pop_size)
    route, cost, history = ga.solve(generations=generations)
    return cost, history, route

# To add a new TSP algorithm:
# @register_tsp("ACO")
# def run_aco_tsp(n, dist, generations, pop_size):
#     aco = AntColonyOptimization(n, dist, n_ants=pop_size)
#     route, cost, history = aco.solve(iterations=generations)
#     return cost, history, route


# =============================================================================
# PROBLEM REGISTRY
# =============================================================================
CONTINUOUS_PROBLEMS = {
    "Sphere": {
        "func": problems_evo.sphere_function,
        "bounds": [[-5.12, 5.12]] * 10,
        "generations": 50,
    },
    "Rastrigin": {
        "func": problems_evo.rastrigin_function,
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
        visualization_evo.plot_robustness_convergence(
            all_histories,
            f"Convergence on {prob_name}: {' vs '.join(self.algorithm_names)}",
            f"{tag}_convergence"
        )

        # Boxplot
        visualization_evo.plot_boxplot_comparison(
            all_scores,
            f"Quality on {prob_name}: {' vs '.join(self.algorithm_names)}",
            f"{tag}_boxplot"
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
        """Replicates ContinuousExperiment.metric_scalability_dimensions from main.py"""
        print("\n[SCALABILITY] Time vs Dimensions on Rastrigin")
        dims = [2, 5, 10, 20]
        times = {name: [] for name in self.algorithm_names}

        for d in dims:
            bounds = [[-5.12, 5.12]] * d
            func = problems_evo.rastrigin_function
            for name in self.algorithm_names:
                s = time.time()
                # Use fixed pop_size=30 and fixed HC budget to match main.py behavior
                if name == "HC":
                    hc = ContinuousLocalSearch(step_size=0.5, max_iter=1500)
                    hc.hill_climbing(func, bounds)
                else:
                    CONTINUOUS_ALGORITHMS[name](func, bounds, generations=50, pop_size=30)
                times[name].append(time.time() - s)

        visualization_evo.plot_scalability_lines(
            dims, times,
            f"Scalability: {' vs '.join(self.algorithm_names)} on Rastrigin (Time)",
            # Match exact filename from main.py
            "cont_scalability_time",
            "Dimensions (D)", "Execution Time (s)"
        )

    def _sensitivity_continuous(self):
        """Replicates metric_sensitivity for DE and CS from main.py"""
        bounds = [[-5.12, 5.12]] * 10
        func = problems_evo.rastrigin_function

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
            visualization_evo.plot_parameter_sensitivity(
                results, CR_vals, F_vals,
                "DE Sensitivity Analysis on Rastrigin Function",
                "cont_sensitivity",   # ← match main.py filename (not cont_sensitivity_de)
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
            visualization_evo.plot_parameter_sensitivity(
                cs_results, pa_vals, alpha_vals,
                "Cuckoo Search Sensitivity on Rastrigin Function", "cont_sensitivity_cs",
                "Abandonment Probability (pa)", "Step Size (alpha)"
            )


class TSPComparison:
    """
    Runs any subset of registered TSP heuristics vs exact solvers on configurable city sizes.
    """

    def __init__(self, algorithm_names=None, sizes=None, runs=30, pop_size=50):
        self.algorithm_names = algorithm_names or list(TSP_ALGORITHMS.keys())
        self.sizes = sizes or [8, 9, 10]
        self.runs = runs
        self.pop_size = pop_size

    def run_all(self):
        print("\n" + "#"*60)
        print(f"TSP COMPARISON: {self.algorithm_names} | Sizes: {self.sizes}")
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
            cities = problems_evo.generate_cities(n, seed=42)
            dist = problems_evo.calculate_distance_matrix(cities)
            solver = TSPGraphSearch(n, dist)

            s = time.time(); solver.bfs(); exact_times['BFS'].append(time.time() - s)
            s = time.time(); solver.dfs(); exact_times['DFS'].append(time.time() - s)
            s = time.time(); solver.a_star(); exact_times['A* (Exact)'].append(time.time() - s)

            for name in self.algorithm_names:
                run_times = []
                for _ in range(5):
                    s = time.time()
                    TSP_ALGORITHMS[name](n, dist, generations=50, pop_size=self.pop_size)
                    run_times.append(time.time() - s)
                heuristic_times[name].append(np.mean(run_times))

        all_times = {**exact_times, **heuristic_times}
        visualization_evo.plot_scalability_lines(
            self.sizes, all_times,
            f"TSP Scalability: {' vs '.join(all_times.keys())}",
            f"tsp_scalability_{'_'.join(self.algorithm_names)}",
            "Number of Cities", "Execution Time (s)"
        )

    def _quality_vs_optimal(self):
        print("\n[2] QUALITY vs OPTIMAL")
        n = self.sizes[-1]
        cities = problems_evo.generate_cities(n, seed=100)
        dist = problems_evo.calculate_distance_matrix(cities)

        solver = TSPGraphSearch(n, dist)
        optimal_cost = solver.a_star()
        print(f"  Optimal (A*): {optimal_cost:.2f}")

        all_scores = {name: [] for name in self.algorithm_names}
        all_histories = {name: [] for name in self.algorithm_names}
        best_routes = {name: None for name in self.algorithm_names}
        best_costs = {name: float('inf') for name in self.algorithm_names}

        for _ in range(self.runs):
            for name in self.algorithm_names:
                cost, history, route = TSP_ALGORITHMS[name](n, dist, generations=100, pop_size=self.pop_size)
                all_scores[name].append(cost)
                all_histories[name].append(history)
                if cost < best_costs[name]:
                    best_costs[name] = cost
                    best_routes[name] = route

        scores_for_plot = {**all_scores, 'Exact (Fixed)': [optimal_cost] * self.runs}

        visualization_evo.plot_robustness_convergence(
            all_histories,
            f"TSP Convergence: {' vs '.join(self.algorithm_names)}",
            f"tsp_convergence_{'_'.join(self.algorithm_names)}"
        )
        visualization_evo.plot_boxplot_comparison(
            scores_for_plot,
            f"TSP Quality: {' vs '.join(self.algorithm_names)} vs Exact",
            f"tsp_quality_{'_'.join(self.algorithm_names)}",
            ylabel="Path Cost"
        )

        for name in self.algorithm_names:
            mean = np.mean(all_scores[name])
            t = manual_onesample_ttest(all_scores[name], optimal_cost)
            print(f"  [{name}] Mean={mean:.2f}, Gap={((mean-optimal_cost)/optimal_cost*100):.2f}%, t={t:.4f}")
            if best_routes[name] is not None:
                visualization_evo.plot_tsp_route(
                    cities, best_routes[name],
                    f"{name} Best Route (Cost {best_costs[name]:.2f})",
                    f"tsp_best_route_{name.lower()}"
                )

    def _sensitivity(self):
        print("\n[3] SENSITIVITY (GA)")
        if "GA" not in self.algorithm_names:
            return
        mut_rates = [0.01, 0.1, 0.2, 0.5]
        pop_sizes = [20, 50, 100]
        results = np.zeros((len(mut_rates), len(pop_sizes)))
        n = self.sizes[-1]
        cities = problems_evo.generate_cities(n, seed=99)
        dist = problems_evo.calculate_distance_matrix(cities)

        for i, mr in enumerate(mut_rates):
            for j, ps in enumerate(pop_sizes):
                costs = []
                for _ in range(5):
                    ga = GeneticAlgorithmTSP(n, dist, pop_size=ps, mutation_rate=mr)
                    _, c, _ = ga.solve(generations=50)
                    costs.append(c)
                results[i, j] = np.mean(costs)

        visualization_evo.plot_parameter_sensitivity(
            results, pop_sizes, mut_rates, "GA Sensitivity Analysis", "tsp_sensitivity",
            "Population Size", "Mutation Rate"
        )


# =============================================================================
# EXPLORATION / 3D VISUALIZATION (standalone utility)
# =============================================================================
def run_exploration_visualization(algorithm_names=None, problem_name="Rastrigin"):
    """Plot 3D landscape trajectories"""
    algorithm_names = algorithm_names or list(CONTINUOUS_ALGORITHMS.keys())
    bounds_2d = [[-5.12, 5.12], [-5.12, 5.12]]
    func = problems_evo.rastrigin_function

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

    visualization_evo.plot_3d_landscape_path(
        func, bounds_2d, paths,
        "Trajectory Comparison: Rastrigin",
        "cont_3d_rastrigin_comparison"
    )


# =============================================================================
# ENTRY POINT — Compose your experiments here
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists('results/figures'):
        os.makedirs('results/figures')

    # --- TSP Experiments ---
    # Compare only GA (default). To add ACO: register it above, then:
    # TSPComparison(algorithm_names=["GA", "ACO"]).run_all()
    TSPComparison(algorithm_names=["GA"], sizes=[8, 9, 10]).run_all()

    # --- Continuous Experiments ---
    # All algorithms on all problems:
    ContinuousComparison().run_all()

    # Custom: only DE vs CS on Rastrigin:
    # ContinuousComparison(algorithm_names=["DE", "CS"], problem_names=["Rastrigin"]).run_all()

    # 3D visualization
    run_exploration_visualization(algorithm_names=["DE", "HC", "CS"], problem_name="Rastrigin")

    # Scalability & Sensitivity (still available as standalone)
    cont = ContinuousComparison(algorithm_names=["DE", "HC", "CS"], problem_names=["Rastrigin"])
    # cont.run_all() already covers these; call individually if needed

    print("\n" + "="*60)
    print("HOÀN THÀNH. KIỂM TRA FOLDER RESULTS/FIGURES.")
    print("="*60)