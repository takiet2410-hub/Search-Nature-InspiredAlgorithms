import numpy as np


# ============================================================
# GRAPH COLORING PROBLEM
# Mục tiêu: Tô màu đồ thị sao cho hai đỉnh kề nhau
# không cùng màu — tối thiểu hóa số xung đột (conflicts).
# ============================================================


def generate_random_graph(num_nodes, edge_prob=0.5, seed=42):
    """
    Tạo đồ thị ngẫu nhiên (Erdős-Rényi model).

    Args:
        num_nodes : Số đỉnh
        edge_prob : Xác suất có cạnh giữa 2 đỉnh
        seed      : Hạt giống ngẫu nhiên

    Returns:
        adj_matrix : Ma trận kề (symmetric, 0-1)
        edges      : Danh sách cạnh [(u, v), ...]
    """
    np.random.seed(seed)
    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append((i, j))
    return adj, edges


def count_conflicts(coloring, adj_matrix):
    """
    Đếm số cạnh có 2 đầu cùng màu (xung đột).

    Args:
        coloring  : Mảng int, coloring[i] = màu của đỉnh i
        adj_matrix: Ma trận kề

    Returns:
        conflicts : Số xung đột
    """
    n = len(coloring)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] == 1 and coloring[i] == coloring[j]:
                conflicts += 1
    return conflicts


# ============================================================
# GA-Based Graph Coloring Solver
# ============================================================

def ga_graph_coloring(adj_matrix, num_colors=3, pop_size=50, generations=100,
                      mutation_rate=0.1, elitism_rate=0.1):
    """
    Giải bài toán tô màu đồ thị bằng GA (Integer-encoded).

    Mỗi cá thể = mảng int, gene[i] ∈ {0, 1, ..., num_colors-1}.
    Fitness = -conflicts (maximize → tối thiểu xung đột).

    Args:
        adj_matrix    : Ma trận kề
        num_colors    : Số màu cho phép
        pop_size      : Kích thước quần thể
        generations   : Số thế hệ
        mutation_rate : Tỉ lệ đột biến
        elitism_rate  : Tỉ lệ giữ lại

    Returns:
        best_coloring : Mảng màu tốt nhất
        best_conflicts: Số xung đột tốt nhất
        history       : Lịch sử best conflicts mỗi thế hệ
    """
    num_nodes = len(adj_matrix)
    elitism_count = max(1, int(pop_size * elitism_rate))

    # --- Khởi tạo ---
    population = [np.random.randint(0, num_colors, size=num_nodes) for _ in range(pop_size)]
    conflicts = [count_conflicts(ind, adj_matrix) for ind in population]

    best_idx = int(np.argmin(conflicts))
    best_coloring = population[best_idx].copy()
    best_conflicts = conflicts[best_idx]
    history = [best_conflicts]

    for gen in range(generations):
        # Sắp xếp tăng dần theo conflicts (ít xung đột = tốt)
        order = np.argsort(conflicts)
        population = [population[i] for i in order]
        conflicts = [conflicts[i] for i in order]

        # Elitism
        new_pop = [population[i].copy() for i in range(elitism_count)]

        while len(new_pop) < pop_size:
            # Tournament selection (k=3)
            idxs = np.random.randint(0, pop_size, size=3)
            p1_idx = idxs[np.argmin([conflicts[k] for k in idxs])]
            idxs = np.random.randint(0, pop_size, size=3)
            p2_idx = idxs[np.argmin([conflicts[k] for k in idxs])]

            p1, p2 = population[p1_idx], population[p2_idx]

            # Uniform crossover
            mask = np.random.randint(0, 2, size=num_nodes)
            child = np.where(mask, p1, p2)

            # Mutation: đổi ngẫu nhiên màu
            for i in range(num_nodes):
                if np.random.rand() < mutation_rate:
                    child[i] = np.random.randint(0, num_colors)

            new_pop.append(child)

        population = new_pop
        conflicts = [count_conflicts(ind, adj_matrix) for ind in population]

        gen_best_idx = int(np.argmin(conflicts))
        if conflicts[gen_best_idx] < best_conflicts:
            best_conflicts = conflicts[gen_best_idx]
            best_coloring = population[gen_best_idx].copy()

        history.append(best_conflicts)

    return best_coloring, best_conflicts, history


# ============================================================
# Greedy Baseline
# ============================================================

def greedy_graph_coloring(adj_matrix):
    """
    Tô màu đồ thị bằng giải thuật Greedy (Welsh-Powell).

    Sắp xếp đỉnh theo bậc giảm dần, gán màu nhỏ nhất có thể.

    Returns:
        coloring   : Mảng màu
        num_colors : Số màu đã dùng
    """
    n = len(adj_matrix)
    degrees = np.sum(adj_matrix, axis=1)
    order = np.argsort(-degrees)  # Bậc cao trước

    coloring = np.full(n, -1, dtype=int)

    for node in order:
        # Tìm các màu đã dùng bởi hàng xóm
        neighbor_colors = set()
        for j in range(n):
            if adj_matrix[node][j] == 1 and coloring[j] >= 0:
                neighbor_colors.add(coloring[j])

        # Gán màu nhỏ nhất chưa dùng
        color = 0
        while color in neighbor_colors:
            color += 1
        coloring[node] = color

    num_colors = int(np.max(coloring)) + 1
    return coloring, num_colors


# ============================================================
# SA-Based Graph Coloring Solver
# ============================================================

def sa_graph_coloring(adj_matrix, num_colors=3, max_iter=5000,
                      T_init=100.0, T_min=0.01, cooling_rate=0.995):
    """
    Giải bài toán tô màu đồ thị bằng Simulated Annealing.

    Neighbor: đổi màu 1 đỉnh ngẫu nhiên.
    Fitness: số xung đột (minimize).

    Returns:
        best_coloring  : Mảng màu tốt nhất
        best_conflicts : Số xung đột tốt nhất
        history        : Lịch sử best conflicts
    """
    n = len(adj_matrix)
    # Khởi tạo ngẫu nhiên
    current = np.random.randint(0, num_colors, size=n)
    current_conf = count_conflicts(current, adj_matrix)

    best = current.copy()
    best_conf = current_conf
    history = [best_conf]

    T = T_init
    for _ in range(max_iter):
        # Neighbor: đổi 1 đỉnh sang màu khác
        neighbor = current.copy()
        node = np.random.randint(0, n)
        old_color = neighbor[node]
        new_color = np.random.randint(0, num_colors - 1)
        if new_color >= old_color:
            new_color += 1
        neighbor[node] = new_color

        neighbor_conf = count_conflicts(neighbor, adj_matrix)
        delta = neighbor_conf - current_conf  # minimize → delta < 0 is good

        if delta < 0 or np.random.rand() < np.exp(-delta / (T + 1e-12)):
            current = neighbor
            current_conf = neighbor_conf

        if current_conf < best_conf:
            best_conf = current_conf
            best = current.copy()

        history.append(best_conf)
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best, best_conf, history
