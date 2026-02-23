import numpy as np
import os
import time
from src.utils import problems, visualization

# Import Algorithms
from src.algorithms.evolution.genetic_algorithm import GeneticAlgorithmTSP
from src.algorithms.evolution.differential_evolution import DifferentialEvolution
from src.algorithms.biology.cuckoo_search import CuckooSearch
from src.algorithms.classical.baselines_evo import TSPGraphSearch, ContinuousLocalSearch

# --- HÀM THỐNG KÊ (NUMPY ONLY) ---
def manual_onesample_ttest(sample, population_mean):
    """Kiểm định T-test 1 mẫu (So sánh GA với kết quả Optimal cố định)"""
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
# So sánh: GA vs Exact Methods (BFS, DFS, A*)
# ============================================================================
class TSPExperiment:
    def run_all(self):
        print("\n" + "#"*60)
        print("EXPERIMENT 1: DISCRETE TSP (GA vs BFS/DFS/A*)")
        print("#"*60)
        
        self.metric_scalability_full_comparison()
        self.metric_robustness_quality()
        self.metric_sensitivity()
        
    def metric_scalability_full_comparison(self):
        print("\n[1] SCALABILITY & COMPLEXITY (Time vs Size)")
        print("Mục tiêu: So sánh thời gian chạy của 4 thuật toán: BFS, DFS, A*, GA")
        
        # LƯU Ý: BFS/DFS có độ phức tạp O(N!). 
        # Với N=11, 11! = 39 triệu trạng thái, Python sẽ chạy rất lâu hoặc tràn RAM.
        # Nên ta chỉ test sizes nhỏ: 8, 9, 10.
        sizes = [8, 9, 10] 
        
        times = {
            'BFS': [],
            'DFS': [],
            'A* (Exact)': [],
            'GA (Approx)': []
        }
        
        for n in sizes:
            print(f"-> Running Size N={n}...")
            cities = problems.generate_cities(n, seed=42)
            dist = problems.calculate_distance_matrix(cities)
            solver = TSPGraphSearch(n, dist)
            
            # 1. BFS
            s = time.time()
            solver.bfs()
            times['BFS'].append(time.time() - s)
            
            # 2. DFS
            s = time.time()
            solver.dfs()
            times['DFS'].append(time.time() - s)
            
            # 3. A*
            s = time.time()
            solver.a_star()
            times['A* (Exact)'].append(time.time() - s)
            
            # 4. GA (Avg 5 runs)
            ga_times = []
            for _ in range(5):
                s = time.time()
                ga = GeneticAlgorithmTSP(n, dist, pop_size=50)
                ga.solve(generations=50) # Chạy số thế hệ vừa phải để đo tốc độ
                ga_times.append(time.time() - s)
            times['GA (Approx)'].append(np.mean(ga_times))
            
            print(f"   Time: BFS={times['BFS'][-1]:.3f}s, DFS={times['DFS'][-1]:.3f}s, A*={times['A* (Exact)'][-1]:.3f}s, GA={times['GA (Approx)'][-1]:.3f}s")
            
        # Vẽ biểu đồ 4 đường
        visualization.plot_scalability_lines(
            sizes, times,
            "TSP Scalability: Full Comparison (BFS, DFS, A*, GA)", 
            "tsp_scalability_full",
            "Number of Cities", "Execution Time (s)"
        )

    def metric_robustness_quality(self):
        print("\n[2] ROBUSTNESS & CONVERGENCE (GA Focus)")
        # Với N=10, so sánh GA với Optimal (A*)
        n = 10
        cities = problems.generate_cities(n, seed=100)
        dist = problems.calculate_distance_matrix(cities)
        
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
            route, cost, hist = ga.solve(generations=100)
            ga_costs.append(cost)
            ga_histories.append(hist)
            if cost < min_ga_cost:
                min_ga_cost = cost
                best_route = route
        
        # Visualization 1: Convergence của GA (Yêu cầu giữ lại)
        visualization.plot_robustness_convergence(
            {'GA Convergence': ga_histories}, 
            "GA Convergence Speed (over 30 runs)", 
            "tsp_convergence_ga_only"
        )
        
        # Visualization 2: Boxplot so sánh chất lượng (GA vs Optimal Line)
        # Vì BFS, DFS, A* đều ra cùng 1 số Optimal, ta gom chung là "Exact Methods"
        visualization.plot_boxplot_comparison(
            {'GA (30 runs)': ga_costs, 'Exact Methods (Fixed)': [optimal_cost]*30},
            "Solution Quality: GA vs Exact Methods", 
            "tsp_quality_boxplot", 
            ylabel="Path Cost"
        )
        
        # Visualization 3: Best Route
        visualization.plot_tsp_route(cities, best_route, f"GA Best Route (Cost {min_ga_cost:.2f})", "tsp_best_route")

        # Stats
        mean_ga = np.mean(ga_costs)
        t_stat = manual_onesample_ttest(ga_costs, optimal_cost)
        print(f"  GA Stats: Mean={mean_ga:.2f}, Gap={(mean_ga-optimal_cost)/optimal_cost*100:.2f}%")
        print(f"  T-test (GA vs Optimal): t={t_stat:.4f}")

    def metric_sensitivity(self):
        print("\n[3] PARAMETER SENSITIVITY (GA)")
        mut_rates = [0.01, 0.1, 0.2, 0.5]
        pop_sizes = [20, 50, 100]
        results = np.zeros((len(mut_rates), len(pop_sizes)))
        
        n = 10
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
                
        visualization.plot_parameter_sensitivity(
            results, pop_sizes, mut_rates, "GA Sensitivity Analysis", "tsp_sensitivity",
            "Population Size", "Mutation Rate"
        )


# ============================================================================
# EXPERIMENT 2: CONTINUOUS OPTIMIZATION (Rastrigin)
# So sánh: DE vs Hill Climbing (HC) - (Đóng vai trò như A* Baseline)
# ============================================================================
# ============================================================================
# EXPERIMENT 2: CONTINUOUS OPTIMIZATION (Sphere & Rastrigin)
# So sánh: DE vs Hill Climbing (HC)
# ============================================================================
class ContinuousExperiment:
    def run_all(self):
        print("\n" + "#"*60)
        print("EXPERIMENT 2: CONTINUOUS (DE vs Hill Climbing)")
        print("#"*60)
        
        # 1. Chạy so sánh trên hàm Sphere (Bài toán DỄ - Unimodal)
        self.run_comparison_on_function(
            func_name="Sphere",
            func=problems.sphere_function,
            bounds=[[-5.12, 5.12]] * 10, # 10 chiều
            generations=50
        )
        
        # 2. Chạy so sánh trên hàm Rastrigin (Bài toán KHÓ - Multimodal)
        self.run_comparison_on_function(
            func_name="Rastrigin",
            func=problems.rastrigin_function,
            bounds=[[-5.12, 5.12]] * 10, # 10 chiều
            generations=100
        )

        # 3. Các phân tích chuyên sâu khác (Chỉ làm trên Rastrigin vì nó phức tạp hơn)
        self.metric_exploration_visualization()
        self.metric_scalability_dimensions() # Thử thách Scalability trên Rastrigin
        self.metric_sensitivity()

    def run_comparison_on_function(self, func_name, func, bounds, generations):
        """Hàm khung sườn để chạy so sánh Robustness cho bất kỳ hàm nào"""
        print(f"\n>>> RUNNING COMPARISON ON: {func_name} Function <<<")
        runs = 30
        
        hist_de_all, hist_hc_all, hist_cs_all = [], [], []
        scores_de, scores_hc, scores_cs = [], [], []
        
        # Quy đổi tương đối: 1 generation của DE (50 cá thể) ~= 50 lần lặp của HC
        # Để công bằng về số lần gọi hàm (Function Evaluations)
        pop_size = 50
        hc_iter = generations * pop_size 
        
        for i in range(runs):
            # DE
            de = DifferentialEvolution(func, bounds, pop_size=pop_size)
            _, s_de, h_de, _ = de.optimize(generations=generations)
            scores_de.append(s_de)
            hist_de_all.append(h_de)
            
            # HC
            hc = ContinuousLocalSearch(step_size=0.5, max_iter=hc_iter)
            _, s_hc, h_hc, _ = hc.hill_climbing(func, bounds) # Nhớ: HC trả về 4 giá trị (đã sửa ở bước trước)
            scores_hc.append(s_hc)
            
            # Sample HC history để vẽ biểu đồ cho khớp trục hoành với DE
            # Lấy mẫu cứ mỗi `pop_size` lần lặp thì lấy 1 điểm
            sampled_hist_hc = h_hc[::pop_size][:generations]
            hist_hc_all.append(sampled_hist_hc)

            # CS
            cs = CuckooSearch(func, bounds, n_nests=50, pa=0.25, alpha=0.01, beta=1.5)
            _, s_cs, h_cs, _ = cs.optimize(iterations=generations)
            scores_cs.append(s_cs)
            hist_cs_all.append(h_cs)
            
        # --- VẼ BIỂU ĐỒ ---
        # 1. Convergence (Tốc độ hội tụ)
        visualization.plot_robustness_convergence(
            {'DE': hist_de_all, 'HC': hist_hc_all, 'CS': hist_cs_all}, 
            f"Convergence: DE vs HC on {func_name}", 
            f"cont_{func_name.lower()}_convergence"
        )
        
        # 2. Quality (Boxplot)
        visualization.plot_boxplot_comparison(
            {'DE': scores_de, 'HC': scores_hc, 'CS': scores_cs}, 
            f"Quality Distribution on {func_name}", 
            f"cont_{func_name.lower()}_boxplot"
        )
        
        # Stats Output
        print(
            f"  [Stats {func_name}] "
            f"DE Mean: {np.mean(scores_de):.4f} | "
            f"HC Mean: {np.mean(scores_hc):.4f} | "
            f"CS Mean: {np.mean(scores_cs):.4f}"
        )
        t_stat = manual_twosample_ttest(scores_de, scores_hc)
        print(f"  T-Test (DE vs HC): t={t_stat:.4f}")
        print(f"  T-Test (CS vs HC): t={manual_twosample_ttest(scores_cs, scores_hc):.4f}")
        print(f"  T-Test (DE vs CS): t={manual_twosample_ttest(scores_de, scores_cs):.4f}")

    def metric_exploration_visualization(self):
        print("\n[3] EXPLORATION vs EXPLOITATION (3D Visualization)")
        # Chỉ vẽ Rastrigin vì Sphere quá đơn giản (cái bát)
        bounds = [[-5.12, 5.12]] * 2 
        func = problems.rastrigin_function
        
        # HC
        hc = ContinuousLocalSearch(step_size=0.2, max_iter=100) 
        _, _, _, path_hc = hc.hill_climbing(func, bounds)
        
        # DE
        de = DifferentialEvolution(func, bounds, pop_size=20)
        _, _, _, path_de = de.optimize(generations=20)
        
        cs = CuckooSearch(func, bounds, n_nests=20, pa=0.25, alpha=0.02)
        _, _, _, path_cs = cs.optimize(iterations=20)

        # Vẽ
        visualization.plot_3d_landscape_path(
            func, bounds, 
            {
                'DE (Exploration)': path_de,
                'HC (Exploitation)': path_hc,
                'CS (Levy Flights)': path_cs
            },
            "Trajectory Comparison: Rastrigin", "cont_3d_rastrigin_comparison"
        )

    def metric_scalability_dimensions(self):
        print("\n[4] SCALABILITY (Time vs Dimensions)")
        # Vẫn giữ nguyên logic cũ (chạy trên Rastrigin để test độ khó)
        dims = [2, 5, 10, 20]
        times = {'DE': [], 'HC': [], 'CS': []}
        
        for d in dims:
            bounds = [[-5.12, 5.12]] * d
            func = problems.rastrigin_function
            
            # DE
            s = time.time()
            de = DifferentialEvolution(func, bounds, pop_size=30)
            de.optimize(generations=50)
            times['DE'].append(time.time() - s)
            
            # HC
            s = time.time()
            hc = ContinuousLocalSearch(max_iter=1500)
            hc.hill_climbing(func, bounds)
            times['HC'].append(time.time() - s)

            # CS
            s = time.time()
            cs = CuckooSearch(func, bounds, n_nests=30, pa=0.25, alpha=0.01)
            cs.optimize(iterations=50)
            times['CS'].append(time.time() - s)
            
        visualization.plot_scalability_lines(
            dims, times, 
            "Scalability: DE vs HC vs CS on Rastrigin Function (Time)", "cont_scalability_time",
            "Dimensions (D)", "Execution Time (s)"
        )

    def metric_sensitivity(self):
        print("\n[5] PARAMETER SENSITIVITY (DE on Rastrigin)")
        # Logic cũ
        F_vals = [0.3, 0.5, 0.9]
        CR_vals = [0.1, 0.5, 0.9]
        results = np.zeros((len(F_vals), len(CR_vals)))
        bounds = [[-5.12, 5.12]] * 10
        
        for i, f in enumerate(F_vals):
            for j, cr in enumerate(CR_vals):
                scores = []
                for _ in range(5):
                    de = DifferentialEvolution(problems.rastrigin_function, bounds, pop_size=30, mutation_factor=f, crossover_rate=cr)
                    _, s, _, _ = de.optimize(generations=50)
                    scores.append(s)
                results[i, j] = np.mean(scores)
                
        visualization.plot_parameter_sensitivity(
            results, CR_vals, F_vals, "DE Sensitivity Analysis on Rastrigin Function", "cont_sensitivity",
            "Crossover Rate (CR)", "Mutation Factor (F)"
        )

        # --- CS sensitivity ---
        alpha_vals = [0.005, 0.01, 0.05]
        pa_vals = [0.1, 0.25, 0.4]
        cs_results = np.zeros((len(alpha_vals), len(pa_vals)))

        for i, alpha in enumerate(alpha_vals):
            for j, pa in enumerate(pa_vals):
                scores = []
                for _ in range(5):
                    cs = CuckooSearch(
                        problems.rastrigin_function, bounds,
                        n_nests=30, pa=pa, alpha=alpha, beta=1.5
                    )
                    _, s, _, _ = cs.optimize(iterations=50)
                    scores.append(s)
                cs_results[i, j] = np.mean(scores)

        visualization.plot_parameter_sensitivity(
            cs_results, pa_vals, alpha_vals,
            "Cuckoo Search Sensitivity on Rastrigin Function",
            "cont_sensitivity_cs",
            "Abandonment Probability (pa)", "Step Size (alpha)"
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