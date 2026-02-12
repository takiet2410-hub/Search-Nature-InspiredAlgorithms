import numpy as np

# --- 1. CHO BÀI TOÁN GA (TSP) ---
def generate_cities(num_cities, map_size=100, seed=42):
    """Tạo tọa độ (x, y) ngẫu nhiên cho các thành phố"""
    np.random.seed(seed) 
    return np.random.rand(num_cities, 2) * map_size

def calculate_distance_matrix(cities):
    """Tính ma trận khoảng cách giữa tất cả cặp thành phố"""
    num_cities = len(cities)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])
    return dist_matrix

# --- 2. CHO BÀI TOÁN DE (Continuous Optimization) ---
def sphere_function(x):
    """Hàm Sphere: Min tại [0,0,...] với giá trị 0"""
    x = np.array(x)  # <--- THÊM DÒNG NÀY (Chuyển list thành numpy array)
    return np.sum(x**2)

def rastrigin_function(x):
    """Hàm Rastrigin: Min tại [0,0,...] với giá trị 0"""
    x = np.array(x)  # <--- THÊM DÒNG NÀY (Quan trọng)
    A = 10
    d = len(x)
    return A * d + np.sum(x**2 - A * np.cos(2 * np.pi * x))