import numpy as np
import os
import time
from src.utils import problems_evo, visualization_evo

# Import Algorithms
from src.algorithms.evolution.genetic_algorithm import GeneticAlgorithmTSP
from src.algorithms.evolution.differential_evolution import DifferentialEvolution
from src.algorithms.classical.baselines_evo import TSPGraphSearch, ContinuousLocalSearch

# --- HÀM THỐNG KÊ (NUMPY ONLY) ---
def manual_onesample_ttest(sample, population_mean):
    """Kiểm định T-test 1 mẫu (So sánh GA với kết quả A* cố định)"""
    n = len(sample)
    mean = np.mean(sample)
    std = np.std(sample, ddof=1)
    se = std / np.sqrt(n)
    if se == 0: return 0.0
    t_stat = (mean - population_mean) / se
    return t_stat

def manual_twosample_ttest(sample1, sample2):
    """Kiểm định Welch's T-test (So sánh DE vs HC)"""
    n1, n2 = len(sample1), len(sample2)
    m1, m2 = np.mean(sample1), np.mean(sample2)
    v1, v2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    se = np.sqrt(v1/n1 + v2/n2)
    if se == 0: return 0.0
    return (m1 - m2) / se

# ============================================================================
# EXPERIMENT 1: DISCRETE OPTIMIZATION (TSP)
# So sánh: GA vs Exact Methods (A*)
# ============================================================================
class TSPExperiment:
    def run_all(self):
        print("\n" + "#"*60)
        print("EXPERIMENT 1: DISCRETE TSP (GA vs A*)")
        print("#"*60)
        
        self.metric_scalability()
        self.metric_robustness_quality()
        self.metric_sensitivity()
        
    def metric_scalability(self):
        print("\n[1] SCALABILITY & COMPLEXITY (Time vs Size)")
        sizes = [8, 9, 10, 11]
        times_ga = []
        times_astar = []
        
        for n in sizes:
            cities = problems_evo.generate_cities(n, seed=42)
            dist = problems_evo.calculate_distance_matrix(cities)
            
            # A*
            s = time.time()
            TSPGraphSearch(n, dist).a_star()
            times_astar.append(time.time() - s)
            
            # GA (Avg 5 runs)
            ga_times = []
            for _ in range(5):
                s = time.time()
                ga = GeneticAlgorithmTSP(n, dist, pop_size=50)
                ga.solve(generations=50)
                ga_times.append(time.time() - s)
            times_ga.append(np.mean(ga_times))
            
            print(f"  Size {n}: A*={times_astar[-1]:.4f}s, GA={times_ga[-1]:.4f}s")
            
        visualization_evo.plot_scalability_lines(
            sizes, {'A* (Exact)': times_astar, 'GA (Approx)': times_ga},
            "TSP Scalability: Time Complexity", "tsp_scalability",
            "Number of Cities", "Time (s)"
        )

    def metric_robustness_quality(self):
        print("\n[2] ROBUSTNESS, QUALITY & STATS (N=10)")
        n = 10
        cities = problems_evo.generate_cities(n, seed=100)
        dist = problems_evo.calculate_distance_matrix(cities)
        
        # Ground Truth (A*)
        solver = TSPGraphSearch(n, dist)
        optimal_cost = solver.a_star()
        print(f"  Optimal Cost (A*): {optimal_cost:.2f}")
        
        # GA Runs (30 lần)
        ga_costs = []
        ga_histories = []
        best_route = None
        min_ga_cost = float('inf')
        
        for i in range(30):
            ga = GeneticAlgorithmTSP(n, dist, pop_size=50)
            route, cost, hist = ga.solve(generations=80)
            ga_costs.append(cost)
            ga_histories.append(hist)
            if cost < min_ga_cost:
                min_ga_cost = cost
                best_route = route
        
        # Visualization
        visualization_evo.plot_robustness_convergence(
            {'GA': ga_histories}, "GA Convergence vs Iterations", "tsp_convergence_robust"
        )
        visualization_evo.plot_boxplot_comparison(
            {'GA Costs': ga_costs, 'Optimal (A*)': [optimal_cost]*30},
            "TSP Quality Distribution", "tsp_boxplot", ylabel="Path Cost"
        )
        visualization_evo.plot_tsp_route(cities, best_route, f"GA Best Route (Cost {min_ga_cost:.2f})", "tsp_best_route")

        # Stats & T-Test
        mean_ga = np.mean(ga_costs)
        std_ga = np.std(ga_costs)
        t_stat = manual_onesample_ttest(ga_costs, optimal_cost)
        
        print(f"  GA Stats: Mean={mean_ga:.2f}, Std={std_ga:.2f}, Best={min_ga_cost:.2f}")
        print(f"  Gap to Optimal: {((mean_ga - optimal_cost)/optimal_cost)*100:.2f}%")
        print(f"  One-sample T-test (H0: GA Mean == Optimal): t={t_stat:.4f}")
        if t_stat > 2.0: print("  -> GA significantly differs from Optimal (Expected for heuristic)")

    def metric_sensitivity(self):
        print("\n[3] PARAMETER SENSITIVITY (GA)")
        # Phân tích Mutation Rate vs Pop Size
        mut_rates = [0.01, 0.05, 0.1, 0.2]
        pop_sizes = [20, 40, 60, 80]
        results = np.zeros((len(mut_rates), len(pop_sizes)))
        
        n = 10
        cities = problems_evo.generate_cities(n, seed=99)
        dist = problems_evo.calculate_distance_matrix(cities)
        
        for i, mr in enumerate(mut_rates):
            for j, ps in enumerate(pop_sizes):
                costs = []
                for _ in range(5): # Chạy 5 lần lấy trung bình
                    ga = GeneticAlgorithmTSP(n, dist, pop_size=ps, mutation_rate=mr)
                    _, c, _ = ga.solve(generations=50)
                    costs.append(c)
                results[i, j] = np.mean(costs)
                
        visualization_evo.plot_parameter_sensitivity(
            results, pop_sizes, mut_rates, "GA Sensitivity (Mean Cost)", "tsp_sensitivity",
            "Population Size", "Mutation Rate"
        )


# ============================================================================
# EXPERIMENT 2: CONTINUOUS OPTIMIZATION (Rastrigin)
# So sánh: DE vs Hill Climbing (HC)
# ============================================================================
class ContinuousExperiment:
    def run_all(self):
        print("\n" + "#"*60)
        print("EXPERIMENT 2: CONTINUOUS (DE vs HC)")
        print("#"*60)
        
        self.metric_exploration_exploitation()
        self.metric_robustness_stats()
        self.metric_scalability_dimensions()
        self.metric_sensitivity()

    def metric_exploration_exploitation(self):
        print("\n[1] EXPLORATION vs EXPLOITATION (3D Visualization)")
        # Chạy trên 2D Rastrigin để vẽ hình
        bounds = [[-5.12, 5.12]] * 2 
        func = problems_evo.rastrigin_function
        
        # HC (Exploitation - kẹt local)
        hc = ContinuousLocalSearch(step_size=0.5, max_iter=200)
        _, _, hist_hc = hc.hill_climbing(func, bounds)
        # Tạo path giả lập cho HC (vì code HC cũ trả về history score chứ ko phải path, 
        # Cần sửa HC một chút nếu muốn path chính xác, ở đây ta giả lập điểm đầu/cuối để demo)
        # Tốt nhất: Sửa HC trả về path. Nhưng để nhanh, ta dùng DE path.
        
        # DE (Exploration)
        de = DifferentialEvolution(func, bounds, pop_size=20)
        _, _, _, path_de = de.optimize(generations=30)
        
        # Vẽ 3D
        visualization_evo.plot_3d_landscape_path(
            func, bounds, {'DE': path_de}, 
            "Exploration (DE) on Rastrigin", "continuous_3d_exploration"
        )
        print("  -> Saved 3D landscape with DE trajectory.")

    def metric_robustness_stats(self):
        print("\n[2] ROBUSTNESS & STATS (Dim=10)")
        bounds = [[-5.12, 5.12]] * 10
        runs = 30
        scores_de, scores_hc = [], []
        hist_de_all, hist_hc_all = [], []
        
        for i in range(runs):
            # DE
            de = DifferentialEvolution(problems_evo.rastrigin_function, bounds, pop_size=50)
            _, s_de, h_de, _ = de.optimize(generations=100)
            scores_de.append(s_de)
            hist_de_all.append(h_de)
            
            # HC
            hc = ContinuousLocalSearch(step_size=0.1, max_iter=5000)
            _, s_hc, h_hc = hc.hill_climbing(problems_evo.rastrigin_function, bounds)
            scores_hc.append(s_hc)
            hist_hc_all.append(h_hc[::50][:100]) # Sample cho khớp length
            
        # Viz
        visualization_evo.plot_robustness_convergence(
            {'DE': hist_de_all, 'HC': hist_hc_all}, "Convergence Speed (DE vs HC)", "cont_convergence"
        )
        visualization_evo.plot_boxplot_comparison(
            {'DE': scores_de, 'HC': scores_hc}, "Quality Distribution (Rastrigin)", "cont_boxplot"
        )
        
        # Stats
        t_stat = manual_twosample_ttest(scores_de, scores_hc)
        print(f"  DE Mean: {np.mean(scores_de):.4f} (Std: {np.std(scores_de):.4f})")
        print(f"  HC Mean: {np.mean(scores_hc):.4f} (Std: {np.std(scores_hc):.4f})")
        print(f"  T-Test (DE vs HC): t={t_stat:.4f}")
        if abs(t_stat) > 2.0: print("  -> Significant difference detected.")

    def metric_scalability_dimensions(self):
        print("\n[3] SCALABILITY (Performance with Problem Size/Dimensions)")
        dims = [2, 5, 10, 20]
        time_de, time_hc = [], []
        fit_de, fit_hc = [], []
        
        for d in dims:
            bounds = [[-5.12, 5.12]] * d
            func = problems_evo.rastrigin_function
            
            # DE
            s = time.time()
            de = DifferentialEvolution(func, bounds, pop_size=30)
            _, fit, _, _ = de.optimize(generations=50)
            time_de.append(time.time() - s)
            fit_de.append(fit)
            
            # HC
            s = time.time()
            hc = ContinuousLocalSearch(max_iter=1500) # Giảm iter cho nhanh
            _, fit, _ = hc.hill_climbing(func, bounds)
            time_hc.append(time.time() - s)
            fit_hc.append(fit)
            
            print(f"  Dim {d}: DE Time={time_de[-1]:.3f}s, HC Time={time_hc[-1]:.3f}s")

        visualization_evo.plot_scalability_lines(
            dims, {'DE': time_de, 'HC': time_hc}, "Scalability: Time vs Dimension", "cont_scalability_time",
            "Dimensions", "Time (s)"
        )
        visualization_evo.plot_scalability_lines(
            dims, {'DE': fit_de, 'HC': fit_hc}, "Scalability: Quality vs Dimension", "cont_scalability_quality",
            "Dimensions", "Best Fitness"
        )

    def metric_sensitivity(self):
        print("\n[4] PARAMETER SENSITIVITY (DE)")
        F_vals = [0.3, 0.5, 0.8]
        CR_vals = [0.2, 0.5, 0.9]
        results = np.zeros((len(F_vals), len(CR_vals)))
        bounds = [[-5.12, 5.12]] * 10
        
        for i, f in enumerate(F_vals):
            for j, cr in enumerate(CR_vals):
                scores = []
                for _ in range(5):
                    de = DifferentialEvolution(problems_evo.rastrigin_function, bounds, pop_size=30, mutation_factor=f, crossover_rate=cr)
                    _, s, _, _ = de.optimize(generations=50)
                    scores.append(s)
                results[i, j] = np.mean(scores)
                
        visualization_evo.plot_parameter_sensitivity(
            results, CR_vals, F_vals, "DE Sensitivity (Mean Fitness)", "cont_sensitivity",
            "Crossover Rate (CR)", "Mutation Factor (F)"
        )

if __name__ == "__main__":
    if not os.path.exists('results/figures'):
        os.makedirs('results/figures')

    # Chạy toàn bộ thực nghiệm
    tsp_exp = TSPExperiment()
    tsp_exp.run_all()
    
    cont_exp = ContinuousExperiment()
    cont_exp.run_all()

    print("\n" + "="*60)
    print("HOÀN THÀNH TOÀN BỘ PROJECT! KIỂM TRA FOLDER RESULTS/FIGURES.")
    print("="*60)