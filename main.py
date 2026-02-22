import numpy as np
import os
import time
from src.utils import problems_bio, visualization_bio

# Import Algorithms
from src.algorithms.biology.aco import AntColonyOptimizationTSP
from src.algorithms.biology.pso import ParticleSwarmOptimization
from src.algorithms.biology.abc import ArtificialBeeColony
from src.algorithms.classical.baselines_bio import TSPGraphSearch, ContinuousLocalSearch

# --- HÀM THỐNG KÊ (NUMPY ONLY) ---
def manual_onesample_ttest(sample, population_mean):
    """Kiểm định T-test 1 mẫu (So sánh ACO với kết quả Optimal cố định)"""
    n = len(sample)
    mean = np.mean(sample)
    std = np.std(sample, ddof=1)
    se = std / np.sqrt(n)
    if se == 0: return 0.0
    t_stat = (mean - population_mean) / se
    return t_stat

def manual_twosample_ttest(sample1, sample2):
    """Kiểm định Welch's T-test (So sánh PSO vs ABC vs HC vs SA)"""
    n1, n2 = len(sample1), len(sample2)
    m1, m2 = np.mean(sample1), np.mean(sample2)
    v1, v2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    se = np.sqrt(v1/n1 + v2/n2)
    if se == 0: return 0.0
    return (m1 - m2) / se

# ============================================================================
class TSPExperiment:
    """
    EXPERIMENT 1: DISCRETE TSP
    So sánh ACO (Biology-based) với BFS, DFS, UCS, Greedy, A* (Classical Search)
    """
    def run_all(self):
        print("\n" + "#"*60)
        print("EXPERIMENT 1: DISCRETE TSP (ACO vs Classical Search)")
        print("#"*60)

        self.metric_scalability_full_comparison()
        self.metric_robustness_quality()
        self.metric_sensitivity()

    def metric_scalability_full_comparison(self):
        print("\n[1] SCALABILITY & COMPLEXITY (Time vs Size)")
        print("Mục tiêu: So sánh thời gian chạy của 6 thuật toán: BFS, DFS, UCS, Greedy, A*, ACO")

        # LƯU Ý: BFS/DFS/UCS có độ phức tạp O(N!).
        # Với N >= 11, Python sẽ chạy rất lâu hoặc tràn RAM.
        sizes = [8, 9, 10]

        times = {
            'BFS': [],
            'DFS': [],
            'UCS': [],
            'Greedy BFS': [],
            'A* (Exact)': [],
            'ACO (Approx)': []
        }

        for n in sizes:
            print(f"-> Running Size N={n}...")
            cities = problems_bio.generate_cities(n, seed=42)
            dist = problems_bio.calculate_distance_matrix(cities)
            solver = TSPGraphSearch(n, dist)

            # 1. BFS
            s = time.time()
            solver.bfs()
            times['BFS'].append(time.time() - s)

            # 2. DFS
            s = time.time()
            solver.dfs()
            times['DFS'].append(time.time() - s)

            # 3. UCS
            s = time.time()
            solver.ucs()
            times['UCS'].append(time.time() - s)

            # 4. Greedy Best-First Search
            s = time.time()
            solver.greedy_best_first()
            times['Greedy BFS'].append(time.time() - s)

            # 5. A*
            s = time.time()
            solver.a_star()
            times['A* (Exact)'].append(time.time() - s)

            # 6. ACO (Avg 5 runs)
            aco_times = []
            for _ in range(5):
                s = time.time()
                aco = AntColonyOptimizationTSP(n, dist, num_ants=20)
                aco.solve(iterations=50)
                aco_times.append(time.time() - s)
            times['ACO (Approx)'].append(np.mean(aco_times))

            print(f"   Time: BFS={times['BFS'][-1]:.3f}s, DFS={times['DFS'][-1]:.3f}s, "
                  f"UCS={times['UCS'][-1]:.3f}s, Greedy={times['Greedy BFS'][-1]:.4f}s, "
                  f"A*={times['A* (Exact)'][-1]:.3f}s, ACO={times['ACO (Approx)'][-1]:.3f}s")

        # Vẽ biểu đồ
        visualization_bio.plot_scalability_lines(
            sizes, times,
            "TSP Scalability: Full Comparison (6 Algorithms)", 
            "tsp_scalability_full",
            "Number of Cities", "Execution Time (s)"
        )

    def metric_robustness_quality(self):
        print("\n[2] ROBUSTNESS & CONVERGENCE (ACO Focus)")
        # Với N=10, so sánh ACO với Optimal (A*)
        n = 10
        cities = problems_bio.generate_cities(n, seed=100)
        dist = problems_bio.calculate_distance_matrix(cities)

        # Ground Truth (A*)
        solver = TSPGraphSearch(n, dist)
        optimal_cost = solver.a_star()
        ucs_cost = solver.ucs()
        greedy_cost = solver.greedy_best_first()
        print(f"  Optimal Cost (A*): {optimal_cost:.2f}")
        print(f"  UCS Cost: {ucs_cost:.2f}")
        print(f"  Greedy BFS Cost: {greedy_cost:.2f}")

        # ACO Runs (30 lần)
        aco_costs = []
        aco_histories = []
        best_route = None
        min_aco_cost = float('inf')

        for i in range(30):
            aco = AntColonyOptimizationTSP(n, dist, num_ants=30)
            route, cost, hist = aco.solve(iterations=100)
            aco_costs.append(cost)
            aco_histories.append(hist)
            if cost < min_aco_cost:
                min_aco_cost = cost
                best_route = route

        # Visualization 1: Convergence ACO
        visualization_bio.plot_robustness_convergence(
            {'ACO Convergence': aco_histories}, 
            "ACO Convergence Speed (over 30 runs)", 
            "tsp_convergence_aco_only"
        )

        # Visualization 2: Boxplot so sánh chất lượng
        visualization_bio.plot_boxplot_comparison(
            {
                'ACO (30 runs)': aco_costs, 
                'Exact (A*/UCS)': [optimal_cost]*30,
                'Greedy BFS': [greedy_cost]*30
            },
            "Solution Quality: ACO vs Classical Methods", 
            "tsp_quality_boxplot", 
            ylabel="Path Cost"
        )

        # Visualization 3: Best Route
        visualization_bio.plot_tsp_route(cities, best_route, 
            f"ACO Best Route (Cost {min_aco_cost:.2f})", "tsp_best_route")

        # Stats
        mean_aco = np.mean(aco_costs)
        t_stat = manual_onesample_ttest(aco_costs, optimal_cost)
        print(f"  ACO Stats: Mean={mean_aco:.2f}, Gap={(mean_aco-optimal_cost)/optimal_cost*100:.2f}%")
        print(f"  T-test (ACO vs Optimal): t={t_stat:.4f}")

    def metric_sensitivity(self):
        print("\n[3] PARAMETER SENSITIVITY (ACO)")
        alpha_vals = [0.5, 1.0, 2.0]
        beta_vals = [1.0, 3.0, 5.0]
        results = np.zeros((len(alpha_vals), len(beta_vals)))

        n = 10
        cities = problems_bio.generate_cities(n, seed=99)
        dist = problems_bio.calculate_distance_matrix(cities)

        for i, alpha in enumerate(alpha_vals):
            for j, beta in enumerate(beta_vals):
                costs = []
                for _ in range(5):
                    aco = AntColonyOptimizationTSP(n, dist, num_ants=20, 
                                                    alpha=alpha, beta=beta)
                    _, c, _ = aco.solve(iterations=50)
                    costs.append(c)
                results[i, j] = np.mean(costs)

        visualization_bio.plot_parameter_sensitivity(
            results, beta_vals, alpha_vals, 
            "ACO Sensitivity Analysis", "tsp_sensitivity",
            "Beta (Heuristic Weight)", "Alpha (Pheromone Weight)"
        )

# ============================================================================
class ContinuousExperiment:
    """
    EXPERIMENT 2: CONTINUOUS OPTIMIZATION
    So sánh PSO, ABC (Biology-based) với Hill Climbing + Simulated Annealing (Classical)
    Test trên 5 hàm: Sphere, Rastrigin, Rosenbrock, Ackley, Griewank
    """
    def run_all(self):
        print("\n" + "#"*60)
        print("EXPERIMENT 2: CONTINUOUS (PSO vs ABC vs HC vs SA)")
        print("#"*60)

        # --- 5 Benchmark Functions ---
        # 1. Sphere (Unimodal - Dễ)
        self.run_comparison_on_function(
            func_name="Sphere",
            func=problems_bio.sphere_function,
            bounds=[[-5.12, 5.12]] * 10,
            iterations=50
        )

        # 2. Rastrigin (Multimodal - Khó)
        self.run_comparison_on_function(
            func_name="Rastrigin",
            func=problems_bio.rastrigin_function,
            bounds=[[-5.12, 5.12]] * 10,
            iterations=100
        )

        # 3. Rosenbrock (Narrow Valley - Khó hội tụ)
        self.run_comparison_on_function(
            func_name="Rosenbrock",
            func=problems_bio.rosenbrock_function,
            bounds=[[-5, 10]] * 10,
            iterations=100
        )

        # 4. Ackley (Many Local Optima)
        self.run_comparison_on_function(
            func_name="Ackley",
            func=problems_bio.ackley_function,
            bounds=[[-5, 5]] * 10,
            iterations=100
        )

        # 5. Griewank (Regularly Distributed Minima)
        self.run_comparison_on_function(
            func_name="Griewank",
            func=problems_bio.griewank_function,
            bounds=[[-600, 600]] * 10,
            iterations=100
        )

        # Phân tích chuyên sâu trên Rastrigin (hàm khó nhất)
        self.metric_exploration_visualization()
        self.metric_scalability_dimensions()
        self.metric_sensitivity()

    def run_comparison_on_function(self, func_name, func, bounds, iterations):
        """Hàm khung sườn chạy so sánh Robustness cho bất kỳ hàm nào"""
        print(f"\n>>> RUNNING COMPARISON ON: {func_name} Function <<<")
        runs = 30

        hist_pso_all, hist_abc_all, hist_hc_all, hist_sa_all = [], [], [], []
        scores_pso, scores_abc, scores_hc, scores_sa = [], [], [], []

        # Quy đổi tương đối: 1 iteration PSO/ABC (30 particles) ~= 30 lần lặp HC/SA
        num_particles = 30
        hc_iter = iterations * num_particles

        for i in range(runs):
            # PSO
            pso = ParticleSwarmOptimization(func, bounds, num_particles=num_particles)
            _, s_pso, h_pso, _ = pso.optimize(iterations=iterations)
            scores_pso.append(s_pso)
            hist_pso_all.append(h_pso)

            # ABC
            abc = ArtificialBeeColony(func, bounds, colony_size=num_particles*2)
            _, s_abc, h_abc, _ = abc.optimize(iterations=iterations)
            scores_abc.append(s_abc)
            hist_abc_all.append(h_abc)

            # HC (Hill Climbing)
            hc = ContinuousLocalSearch(step_size=0.5, max_iter=hc_iter)
            _, s_hc, h_hc, _ = hc.hill_climbing(func, bounds)
            scores_hc.append(s_hc)

            # SA (Simulated Annealing)
            sa = ContinuousLocalSearch(step_size=0.5, max_iter=hc_iter)
            _, s_sa, h_sa, _ = sa.simulated_annealing(func, bounds)
            scores_sa.append(s_sa)

            # Sample HC/SA history để vẽ biểu đồ khớp trục X với PSO/ABC
            sampled_hist_hc = h_hc[::num_particles][:iterations]
            hist_hc_all.append(sampled_hist_hc)

            sampled_hist_sa = h_sa[::num_particles][:iterations]
            hist_sa_all.append(sampled_hist_sa)

        # --- VẼ BIỂU ĐỒ ---
        # 1. Convergence (4 thuật toán)
        visualization_bio.plot_robustness_convergence(
            {'PSO': hist_pso_all, 'ABC': hist_abc_all, 'HC': hist_hc_all, 'SA': hist_sa_all}, 
            f"Convergence: PSO vs ABC vs HC vs SA on {func_name}", 
            f"cont_{func_name.lower()}_convergence"
        )

        # 2. Quality (Boxplot 4 thuật toán)
        visualization_bio.plot_boxplot_comparison(
            {'PSO': scores_pso, 'ABC': scores_abc, 'HC': scores_hc, 'SA': scores_sa}, 
            f"Quality Distribution on {func_name}", 
            f"cont_{func_name.lower()}_boxplot"
        )

        # Stats
        print(f"  [Stats {func_name}] PSO Mean: {np.mean(scores_pso):.4f} | "
              f"ABC Mean: {np.mean(scores_abc):.4f} | "
              f"HC Mean: {np.mean(scores_hc):.4f} | SA Mean: {np.mean(scores_sa):.4f}")
        t_pso_abc = manual_twosample_ttest(scores_pso, scores_abc)
        t_pso_hc = manual_twosample_ttest(scores_pso, scores_hc)
        t_pso_sa = manual_twosample_ttest(scores_pso, scores_sa)
        t_sa_hc = manual_twosample_ttest(scores_sa, scores_hc)
        print(f"  T-Test (PSO vs ABC): t={t_pso_abc:.4f}")
        print(f"  T-Test (PSO vs HC): t={t_pso_hc:.4f}")
        print(f"  T-Test (PSO vs SA): t={t_pso_sa:.4f}")
        print(f"  T-Test (SA vs HC): t={t_sa_hc:.4f}")

    def metric_exploration_visualization(self):
        print("\n[3] EXPLORATION vs EXPLOITATION (3D Visualization)")
        bounds = [[-5.12, 5.12]] * 2
        func = problems_bio.rastrigin_function

        # PSO
        pso = ParticleSwarmOptimization(func, bounds, num_particles=20)
        _, _, _, path_pso = pso.optimize(iterations=20)

        # ABC
        abc = ArtificialBeeColony(func, bounds, colony_size=40)
        _, _, _, path_abc = abc.optimize(iterations=20)

        # SA
        sa = ContinuousLocalSearch(step_size=0.5, max_iter=400)
        _, _, _, path_sa = sa.simulated_annealing(func, bounds)

        # HC
        hc = ContinuousLocalSearch(step_size=0.2, max_iter=100)
        _, _, _, path_hc = hc.hill_climbing(func, bounds)

        # Vẽ
        visualization_bio.plot_3d_landscape_path(
            func, bounds,
            {'PSO': path_pso, 'ABC': path_abc, 'SA': path_sa},
            "Trajectory: PSO vs ABC vs SA on Rastrigin", "cont_3d_rastrigin_pso_abc_sa"
        )

        visualization_bio.plot_3d_landscape_path(
            func, bounds,
            {'SA (Exploration)': path_sa, 'HC (Exploitation)': path_hc},
            "SA vs HC: Exploration vs Exploitation", "cont_3d_rastrigin_sa_vs_hc"
        )

    def metric_scalability_dimensions(self):
        print("\n[4] SCALABILITY (Time vs Dimensions)")
        dims = [2, 5, 10, 20]
        times = {'PSO': [], 'ABC': [], 'SA': [], 'HC': []}

        for d in dims:
            bounds = [[-5.12, 5.12]] * d
            func = problems_bio.rastrigin_function

            # PSO
            s = time.time()
            pso = ParticleSwarmOptimization(func, bounds, num_particles=30)
            pso.optimize(iterations=50)
            times['PSO'].append(time.time() - s)

            # ABC
            s = time.time()
            abc = ArtificialBeeColony(func, bounds, colony_size=60)
            abc.optimize(iterations=50)
            times['ABC'].append(time.time() - s)

            # SA
            s = time.time()
            sa = ContinuousLocalSearch(step_size=0.5, max_iter=1500)
            sa.simulated_annealing(func, bounds)
            times['SA'].append(time.time() - s)

            # HC
            s = time.time()
            hc = ContinuousLocalSearch(max_iter=1500)
            hc.hill_climbing(func, bounds)
            times['HC'].append(time.time() - s)

        visualization_bio.plot_scalability_lines(
            dims, times, 
            "Scalability: PSO vs ABC vs SA vs HC on Rastrigin (Time)", "cont_scalability_time",
            "Dimensions (D)", "Execution Time (s)"
        )

    def metric_sensitivity(self):
        print("\n[5] PARAMETER SENSITIVITY (PSO on Rastrigin)")
        w_vals = [0.3, 0.5, 0.7, 0.9]
        c_vals = [0.5, 1.0, 1.5, 2.0]
        results = np.zeros((len(w_vals), len(c_vals)))
        bounds = [[-5.12, 5.12]] * 10

        for i, w in enumerate(w_vals):
            for j, c in enumerate(c_vals):
                scores = []
                for _ in range(5):
                    pso = ParticleSwarmOptimization(
                        problems_bio.rastrigin_function, bounds, 
                        num_particles=30, w=w, c1=c, c2=c
                    )
                    _, s, _, _ = pso.optimize(iterations=50)
                    scores.append(s)
                results[i, j] = np.mean(scores)

        visualization_bio.plot_parameter_sensitivity(
            results, c_vals, w_vals, 
            "PSO Sensitivity Analysis on Rastrigin", "cont_sensitivity",
            "Cognitive/Social Coefficient (c1=c2)", "Inertia Weight (w)"
        )

if __name__ == "__main__":
    if not os.path.exists('results/figures'):
        os.makedirs('results/figures')

    # Chạy các thực nghiệm
    tsp_exp = TSPExperiment()
    tsp_exp.run_all()

    cont_exp = ContinuousExperiment()
    cont_exp.run_all()

    print("\n" + "="*60)
    print("HOÀN THÀNH. KIỂM TRA FOLDER RESULTS/FIGURES.")
    print("="*60)
