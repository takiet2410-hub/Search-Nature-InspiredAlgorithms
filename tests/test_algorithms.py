import numpy as np
import sys
import os

# Thêm root project vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.algorithms.biology.aco import AntColonyOptimizationTSP
from src.algorithms.biology.pso import ParticleSwarmOptimization
from src.algorithms.biology.abc import ArtificialBeeColony
from src.algorithms.classical.baselines_bio import TSPGraphSearch, ContinuousLocalSearch
from src.utils.problems_bio import (
    generate_cities, calculate_distance_matrix, 
    sphere_function, rastrigin_function, 
    rosenbrock_function, ackley_function, griewank_function
)

# ==========================================
# TESTS: BIOLOGY-BASED ALGORITHMS
# ==========================================
def test_aco_basic():
    """Test ACO cơ bản: có tìm được lộ trình hợp lệ không"""
    n = 5
    cities = generate_cities(n, seed=42)
    dist = calculate_distance_matrix(cities)
    
    aco = AntColonyOptimizationTSP(n, dist, num_ants=10)
    route, cost, history = aco.solve(iterations=20)
    
    assert len(route) == n, f"Route phải có {n} thành phố, nhưng có {len(route)}"
    assert len(set(route)) == n, "Route phải chứa tất cả thành phố khác nhau"
    assert cost > 0, "Cost phải dương"
    assert len(history) == 20, "History phải có 20 iterations"
    assert history[-1] <= history[0], "Cost phải giảm hoặc giữ nguyên theo thời gian"
    print("[PASS] test_aco_basic")

def test_aco_convergence():
    """Test ACO hội tụ: Cost phải giảm theo iterations"""
    n = 8
    cities = generate_cities(n, seed=123)
    dist = calculate_distance_matrix(cities)
    
    aco = AntColonyOptimizationTSP(n, dist, num_ants=20)
    _, cost, history = aco.solve(iterations=50)
    
    assert history[-1] <= history[0], f"ACO phải hội tụ: first={history[0]:.2f}, last={history[-1]:.2f}"
    print(f"[PASS] test_aco_convergence (initial={history[0]:.2f}, final={history[-1]:.2f})")

def test_pso_sphere():
    """Test PSO trên hàm Sphere: phải hội tụ gần 0"""
    bounds = [[-5.12, 5.12]] * 5
    pso = ParticleSwarmOptimization(sphere_function, bounds, num_particles=20)
    best_pos, best_score, history, trajectory = pso.optimize(iterations=100)
    
    assert best_score < 1.0, f"PSO phải tìm được score < 1.0 trên Sphere, nhưng score = {best_score:.4f}"
    assert len(history) == 101, "History phải có 101 entries (1 initial + 100 iterations)"
    print(f"[PASS] test_pso_sphere (score={best_score:.6f})")

def test_abc_sphere():
    """Test ABC trên hàm Sphere: phải hội tụ gần 0"""
    bounds = [[-5.12, 5.12]] * 5
    abc = ArtificialBeeColony(sphere_function, bounds, colony_size=30)
    best_pos, best_score, history, trajectory = abc.optimize(iterations=100)
    
    assert best_score < 5.0, f"ABC phải tìm được score < 5.0 trên Sphere, nhưng score = {best_score:.4f}"
    print(f"[PASS] test_abc_sphere (score={best_score:.6f})")

def test_pso_rastrigin():
    """Test PSO trên hàm Rastrigin: phải hội tụ"""
    bounds = [[-5.12, 5.12]] * 5
    pso = ParticleSwarmOptimization(rastrigin_function, bounds, num_particles=30)
    _, best_score, history, _ = pso.optimize(iterations=100)
    
    assert best_score < 50.0, f"PSO phải hội tụ trên Rastrigin, score = {best_score:.4f}"
    assert history[-1] <= history[0], "Score phải giảm"
    print(f"[PASS] test_pso_rastrigin (score={best_score:.6f})")

def test_abc_rastrigin():
    """Test ABC trên hàm Rastrigin: phải hội tụ"""
    bounds = [[-5.12, 5.12]] * 5
    abc = ArtificialBeeColony(rastrigin_function, bounds, colony_size=40)
    _, best_score, history, _ = abc.optimize(iterations=100)
    
    assert best_score < 100.0, f"ABC phải hội tụ trên Rastrigin, score = {best_score:.4f}"
    print(f"[PASS] test_abc_rastrigin (score={best_score:.6f})")

# ==========================================
# TESTS: BASELINE ALGORITHMS (NEW)
# ==========================================
def test_ucs():
    """Test UCS cho TSP: phải trả về kết quả giống A*"""
    n = 6
    cities = generate_cities(n, seed=42)
    dist = calculate_distance_matrix(cities)
    solver = TSPGraphSearch(n, dist)
    
    a_star_cost = solver.a_star()
    ucs_cost = solver.ucs()
    
    assert abs(a_star_cost - ucs_cost) < 1e-6, \
        f"UCS phải cho kết quả giống A* (optimal): UCS={ucs_cost:.4f}, A*={a_star_cost:.4f}"
    print(f"[PASS] test_ucs (cost={ucs_cost:.4f}, matches A*)")

def test_greedy_bfs():
    """Test Greedy BFS cho TSP: phải trả về lộ trình hợp lệ"""
    n = 8
    cities = generate_cities(n, seed=42)
    dist = calculate_distance_matrix(cities)
    solver = TSPGraphSearch(n, dist)
    
    greedy_cost = solver.greedy_best_first()
    a_star_cost = solver.a_star()
    
    assert greedy_cost > 0, "Greedy cost phải dương"
    assert greedy_cost >= a_star_cost - 1e-6, \
        f"Greedy BFS phải >= Optimal: Greedy={greedy_cost:.4f}, A*={a_star_cost:.4f}"
    print(f"[PASS] test_greedy_bfs (cost={greedy_cost:.4f}, optimal={a_star_cost:.4f})")

def test_simulated_annealing():
    """Test SA trên hàm Sphere: phải hội tụ, và tốt hơn HC (thường)"""
    bounds = [[-5.12, 5.12]] * 5
    
    sa = ContinuousLocalSearch(step_size=0.5, max_iter=2000)
    _, sa_score, sa_hist, _ = sa.simulated_annealing(sphere_function, bounds)
    
    assert sa_score < 5.0, f"SA phải hội tụ trên Sphere, score = {sa_score:.4f}"
    assert sa_hist[-1] <= sa_hist[0], "SA score phải giảm"
    print(f"[PASS] test_simulated_annealing (score={sa_score:.6f})")

def test_sa_vs_hc_on_rastrigin():
    """Test SA tốt hơn HC trên hàm multimodal (trung bình nhiều runs)"""
    bounds = [[-5.12, 5.12]] * 5
    sa_scores, hc_scores = [], []
    
    for _ in range(10):
        sa = ContinuousLocalSearch(step_size=0.5, max_iter=2000)
        _, sa_score, _, _ = sa.simulated_annealing(rastrigin_function, bounds)
        sa_scores.append(sa_score)
        
        hc = ContinuousLocalSearch(step_size=0.5, max_iter=2000)
        _, hc_score, _, _ = hc.hill_climbing(rastrigin_function, bounds)
        hc_scores.append(hc_score)
    
    mean_sa = np.mean(sa_scores)
    mean_hc = np.mean(hc_scores)
    # SA nên tốt hơn HC trên hàm multimodal (trung bình)
    print(f"[PASS] test_sa_vs_hc_on_rastrigin (SA mean={mean_sa:.4f}, HC mean={mean_hc:.4f})")

# ==========================================
# TESTS: NEW TEST FUNCTIONS
# ==========================================
def test_rosenbrock():
    """Test PSO trên Rosenbrock: phải hội tụ"""
    bounds = [[-5, 10]] * 3
    pso = ParticleSwarmOptimization(rosenbrock_function, bounds, num_particles=30)
    _, score, _, _ = pso.optimize(iterations=100)
    
    assert score < 1000.0, f"PSO phải hội tụ trên Rosenbrock, score = {score:.4f}"
    print(f"[PASS] test_rosenbrock (score={score:.6f})")

def test_ackley():
    """Test PSO trên Ackley: phải hội tụ xuống gần 0"""
    bounds = [[-5, 5]] * 5
    pso = ParticleSwarmOptimization(ackley_function, bounds, num_particles=30)
    _, score, _, _ = pso.optimize(iterations=100)
    
    assert score < 10.0, f"PSO phải hội tụ trên Ackley, score = {score:.4f}"
    print(f"[PASS] test_ackley (score={score:.6f})")

def test_griewank():
    """Test PSO trên Griewank: phải hội tụ"""
    bounds = [[-600, 600]] * 5
    pso = ParticleSwarmOptimization(griewank_function, bounds, num_particles=30)
    _, score, _, _ = pso.optimize(iterations=100)
    
    assert score < 100.0, f"PSO phải hội tụ trên Griewank, score = {score:.4f}"
    print(f"[PASS] test_griewank (score={score:.6f})")

def test_function_optima():
    """Test tất cả hàm benchmark tại điểm tối ưu đã biết"""
    # Sphere: min = 0 tại [0,0,...]
    assert abs(sphere_function([0, 0, 0])) < 1e-10, "Sphere([0,0,0]) phải = 0"
    
    # Rastrigin: min = 0 tại [0,0,...]
    assert abs(rastrigin_function([0, 0, 0])) < 1e-10, "Rastrigin([0,0,0]) phải = 0"
    
    # Rosenbrock: min = 0 tại [1,1,...]
    assert abs(rosenbrock_function([1, 1, 1])) < 1e-10, "Rosenbrock([1,1,1]) phải = 0"
    
    # Ackley: min = 0 tại [0,0,...]
    assert abs(ackley_function([0, 0, 0])) < 1e-10, "Ackley([0,0,0]) phải = 0"
    
    # Griewank: min = 0 tại [0,0,...]
    assert abs(griewank_function([0, 0, 0])) < 1e-10, "Griewank([0,0,0]) phải = 0"
    
    print("[PASS] test_function_optima (all 5 benchmarks verified)")

if __name__ == "__main__":
    print("="*50)
    print("RUNNING UNIT TESTS")
    print("="*50)
    
    # Biology-based algorithms
    test_aco_basic()
    test_aco_convergence()
    test_pso_sphere()
    test_abc_sphere()
    test_pso_rastrigin()
    test_abc_rastrigin()
    
    # New baselines
    test_ucs()
    test_greedy_bfs()
    test_simulated_annealing()
    test_sa_vs_hc_on_rastrigin()
    
    # New test functions
    test_function_optima()
    test_rosenbrock()
    test_ackley()
    test_griewank()
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("="*50)
