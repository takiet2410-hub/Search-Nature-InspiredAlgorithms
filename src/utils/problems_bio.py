import numpy as np

# --- 1. CHO BÀI TOÁN TSP ---
def generate_cities(num_cities, map_size=100, seed=42):
    """Tạo tọa độ (x, y) ngẫu nhiên cho các thành phố."""
    np.random.seed(seed) 
    return np.random.rand(num_cities, 2) * map_size

def calculate_distance_matrix(cities):
    """Tính ma trận khoảng cách Euclid giữa tất cả cặp thành phố."""
    num_cities = len(cities)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])
    return dist_matrix

# --- 2. CHO BÀI TOÁN CONTINUOUS OPTIMIZATION ---
def sphere_function(x):
    """Sphere: f(x) = Σx² — Unimodal, min tại [0,...] = 0"""
    x = np.array(x)
    return np.sum(x**2)

def rastrigin_function(x):
    """Rastrigin: f(x) = 10d + Σ(x²-10cos(2πx)) — Multimodal, min tại [0,...] = 0"""
    x = np.array(x)
    A = 10
    d = len(x)
    return A * d + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock_function(x):
    """Rosenbrock: f(x) = Σ[100(x_{i+1}-x_i²)²+(1-x_i)²] — Narrow valley, min tại [1,...] = 0"""
    x = np.array(x)
    total = 0
    for i in range(len(x) - 1):
        total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return total

def ackley_function(x):
    """Ackley: Many local optima, min tại [0,...] = 0"""
    x = np.array(x)
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e

def griewank_function(x):
    """Griewank: Regular distributed minima, min tại [0,...] = 0"""
    x = np.array(x)
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_part - prod_part + 1
