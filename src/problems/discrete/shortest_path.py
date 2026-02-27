"""
Bài toán Shortest Path trên đồ thị có trọng số.

Gồm:
  - Tạo đồ thị ngẫu nhiên có trọng số
  - Dijkstra (UCS — optimal, O((V+E)logV))
  - A* (heuristic — optimal nếu heuristic admissible)
  - BFS (tìm đường ít cạnh nhất — NOT optimal cho weighted graph)
  - DFS (tìm 1 đường bất kỳ — KHÔNG optimal)
"""

import numpy as np
import heapq
from collections import deque


# ============================================================
# GRAPH GENERATION
# ============================================================

def generate_weighted_graph(num_nodes, edge_prob=0.4, max_weight=20, seed=42):
    """
    Tạo đồ thị vô hướng có trọng số (adjacency matrix).

    Args:
        num_nodes  : Số đỉnh
        edge_prob  : Xác suất có cạnh giữa 2 đỉnh
        max_weight : Trọng số tối đa
        seed       : Hạt giống

    Returns:
        adj : Ma trận kề (0 = không có cạnh, >0 = trọng số)
        positions : Tọa độ 2D của mỗi đỉnh (dùng cho A* heuristic & vẽ)
    """
    np.random.seed(seed)
    adj = np.zeros((num_nodes, num_nodes))
    positions = np.random.rand(num_nodes, 2) * 100  # Tọa độ [0, 100]

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                w = np.random.randint(1, max_weight + 1)
                adj[i][j] = w
                adj[j][i] = w

    # Đảm bảo đồ thị liên thông (thêm cạnh nếu cần)
    visited = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for j in range(num_nodes):
            if adj[node][j] > 0 and j not in visited:
                stack.append(j)

    # Nối các đỉnh chưa đến được
    for node in range(num_nodes):
        if node not in visited:
            # Nối với 1 đỉnh đã thăm
            target = np.random.choice(list(visited))
            w = np.random.randint(1, max_weight + 1)
            adj[node][target] = w
            adj[target][node] = w
            visited.add(node)

    return adj, positions


def _get_neighbors(adj, node):
    """Lấy danh sách hàng xóm có trọng số."""
    n = len(adj)
    neighbors = []
    for j in range(n):
        if adj[node][j] > 0:
            neighbors.append((j, adj[node][j]))
    return neighbors


# ============================================================
# DIJKSTRA (Uniform Cost Search — Optimal)
# ============================================================

def dijkstra(adj, start, goal):
    """
    Thuật toán Dijkstra tìm đường ngắn nhất.

    Args:
        adj   : Ma trận kề có trọng số
        start : Đỉnh bắt đầu
        goal  : Đỉnh đích

    Returns:
        path     : Danh sách đỉnh trên đường đi
        cost     : Chi phí đường đi
        explored : Số đỉnh đã khám phá
    """
    n = len(adj)
    dist = np.full(n, np.inf)
    dist[start] = 0
    prev = np.full(n, -1, dtype=int)
    pq = [(0, start)]  # (cost, node)
    visited = set()
    explored = 0

    while pq:
        cost, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        explored += 1

        if u == goal:
            break

        for v, w in _get_neighbors(adj, u):
            if v not in visited and cost + w < dist[v]:
                dist[v] = cost + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))

    # Reconstruct path
    path = []
    if dist[goal] < np.inf:
        node = goal
        while node != -1:
            path.append(node)
            node = prev[node]
        path.reverse()

    return path, dist[goal], explored


# ============================================================
# A* (Heuristic Search — Optimal if admissible)
# ============================================================

def a_star_shortest(adj, positions, start, goal):
    """
    A* tìm đường ngắn nhất dùng Euclidean distance làm heuristic.

    Args:
        adj       : Ma trận kề có trọng số
        positions : Tọa độ 2D của mỗi đỉnh
        start     : Đỉnh bắt đầu
        goal      : Đỉnh đích

    Returns:
        path     : Danh sách đỉnh
        cost     : Chi phí thực tế
        explored : Số đỉnh đã khám phá
    """
    n = len(adj)

    def heuristic(u, v):
        return np.sqrt(np.sum((positions[u] - positions[v]) ** 2))

    dist = np.full(n, np.inf)
    dist[start] = 0
    prev = np.full(n, -1, dtype=int)
    pq = [(heuristic(start, goal), 0, start)]  # (f, g, node)
    visited = set()
    explored = 0

    while pq:
        f, g, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        explored += 1

        if u == goal:
            break

        for v, w in _get_neighbors(adj, u):
            new_g = g + w
            if v not in visited and new_g < dist[v]:
                dist[v] = new_g
                prev[v] = u
                f_v = new_g + heuristic(v, goal)
                heapq.heappush(pq, (f_v, new_g, v))

    # Reconstruct path
    path = []
    if dist[goal] < np.inf:
        node = goal
        while node != -1:
            path.append(node)
            node = prev[node]
        path.reverse()

    return path, dist[goal], explored


# ============================================================
# BFS (Unweighted — tìm đường ít CẠNH nhất)
# ============================================================

def bfs_shortest(adj, start, goal):
    """
    BFS tìm đường có ít cạnh nhất (KHÔNG optimal cho weighted graph).

    Returns:
        path     : Danh sách đỉnh
        cost     : Tổng trọng số thực tế trên path
        explored : Số đỉnh đã khám phá
    """
    n = len(adj)
    prev = np.full(n, -1, dtype=int)
    visited = {start}
    queue = deque([start])
    explored = 0

    while queue:
        u = queue.popleft()
        explored += 1

        if u == goal:
            break

        for v, _ in _get_neighbors(adj, u):
            if v not in visited:
                visited.add(v)
                prev[v] = u
                queue.append(v)

    # Reconstruct
    path = []
    node = goal
    while node != -1:
        path.append(node)
        node = prev[node]
    path.reverse()

    # Tính chi phí thực tế
    cost = 0
    for i in range(len(path) - 1):
        cost += adj[path[i]][path[i + 1]]

    return path, cost, explored


# ============================================================
# DFS (Depth-First — tìm 1 đường BẤT KỲ)
# ============================================================

def dfs_shortest(adj, start, goal):
    """
    DFS tìm 1 đường bất kỳ (KHÔNG optimal).

    Returns:
        path     : Danh sách đỉnh
        cost     : Tổng trọng số thực tế trên path
        explored : Số đỉnh đã khám phá
    """
    n = len(adj)
    prev = np.full(n, -1, dtype=int)
    visited = set()
    stack = [start]
    explored = 0
    found = False

    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        explored += 1

        if u == goal:
            found = True
            break

        for v, _ in _get_neighbors(adj, u):
            if v not in visited:
                prev[v] = u
                stack.append(v)

    # Reconstruct
    path = []
    if found:
        node = goal
        while node != -1:
            path.append(node)
            node = prev[node]
        path.reverse()

    # Tính chi phí thực tế
    cost = 0
    for i in range(len(path) - 1):
        cost += adj[path[i]][path[i + 1]]

    return path, cost, explored
