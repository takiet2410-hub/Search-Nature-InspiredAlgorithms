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

def rosenbrock_function(x):
    """Hàm Rosenbrock (Narrow Valley): Min tại [1,1,...,1] với giá trị 0
    f(x) = Σ [100*(x_{i+1} - x_i²)² + (1 - x_i)²]
    Bounds: [-5, 10]
    """
    x = np.array(x)
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def griewank_function(x):
    """Hàm Griewank (Regularly Distributed Minima): Min tại [0,0,...,0] với giá trị 0
    f(x) = 1 + Σ(x_i²/4000) - Π cos(x_i/√i)
    Bounds: [-600, 600]
    """
    x = np.array(x)
    d = len(x)
    sum_part = np.sum(x**2) / 4000.0
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
    return 1.0 + sum_part - prod_part

def ackley_function(x):
    """Hàm Ackley (Many Local Optima): Min tại [0,0,...,0] với giá trị 0
    f(x) = -20*exp(-0.2*√(Σx_i²/d)) - exp(Σcos(2πx_i)/d) + 20 + e
    Bounds: [-32, 32]
    """
    x = np.array(x)
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + 20.0 + np.e

# --- 3. CHO BÀI TOÁN KP (Knapsack Problem) ---
def generate_knapsack_problem(num_items, max_weight=50, max_value=100, capacity_ratio=0.5, seed=42):
    """
    Tạo dữ liệu ngẫu nhiên cho bài toán Knapsack
    
    Args:
        num_items     : Số lượng vật phẩm
        max_weight    : Trọng lượng tối đa của mỗi vật phẩm
        max_value     : Giá trị tối đa của mỗi vật phẩm
        capacity_ratio: Tỉ lệ sức chứa so với tổng trọng lượng (0 < ratio < 1)
        seed          : Hạt giống ngẫu nhiên
    
    Returns:
        weights  : Mảng trọng lượng các vật phẩm
        values   : Mảng giá trị các vật phẩm
        capacity : Sức chứa tối đa của túi
    """
    np.random.seed(seed)
    weights = np.random.randint(1, max_weight + 1, size=num_items)
    values  = np.random.randint(1, max_value + 1, size=num_items)
    capacity = int(np.sum(weights) * capacity_ratio)
    return weights, values, capacity

def knapsack_fitness(individual, weights, values, capacity):
    """
    Tính fitness cho một cá thể trong bài toán Knapsack (dùng cho GA/DE)
    
    Args:
        individual: Mảng nhị phân (0/1) - 1 nghĩa là chọn vật phẩm đó
        weights   : Mảng trọng lượng các vật phẩm
        values    : Mảng giá trị các vật phẩm
        capacity  : Sức chứa tối đa của túi
    
    Returns:
        fitness: Tổng giá trị nếu hợp lệ, 0 nếu vượt quá sức chứa
    """
    individual = np.array(individual, dtype=int)
    total_weight = np.dot(individual, weights)
    total_value  = np.dot(individual, values)

    if total_weight > capacity:
        return 0  # Phạt nghiệm vi phạm ràng buộc
    return total_value

def knapsack_repair(individual, weights, capacity):
    """
    Sửa nghiệm vi phạm bằng cách loại bỏ vật phẩm có tỉ lệ value/weight thấp nhất
    
    Args:
        individual: Mảng nhị phân (0/1)
        weights   : Mảng trọng lượng
        capacity  : Sức chứa tối đa
    
    Returns:
        individual: Mảng nhị phân đã được sửa
    """
    individual = np.array(individual, dtype=int)
    selected   = np.where(individual == 1)[0]

    while np.dot(individual, weights) > capacity and len(selected) > 0:
        # Loại bỏ vật phẩm có trọng lượng lớn nhất trong số đã chọn
        heaviest = selected[np.argmax(weights[selected])]
        individual[heaviest] = 0
        selected = np.where(individual == 1)[0]

    return individual