import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns  # Thư viện vẽ biểu đồ thống kê đẹp
import os

# Cấu hình style cho đẹp
sns.set(style="whitegrid")

# Tạo folder lưu kết quả nếu chưa có
if not os.path.exists('results/figures'):
    os.makedirs('results/figures')

# ==========================================
# 1. CÁC HÀM CƠ BẢN (TSP & CONVERGENCE)
# ==========================================
def plot_convergence(history, title, filename):
    """Vẽ biểu đồ hội tụ đơn giản"""
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Best Fitness', color='b', linewidth=2)
    plt.title(f'Convergence Plot - {title}', fontsize=14)
    plt.xlabel('Generations', fontsize=12)
    plt.ylabel('Fitness (Cost)', fontsize=12)
    plt.legend()
    plt.savefig(f'results/figures/{filename}.png', dpi=150)
    plt.close()
    print(f"[INFO] Saved Convergence: {filename}")

def plot_tsp_route(cities, route, title, filename):
    """Vẽ lộ trình TSP"""
    plt.figure(figsize=(8, 8))
    plt.scatter(cities[:, 0], cities[:, 1], c='red', marker='o', s=50, label='Cities')
    
    # Tạo danh sách toạ độ theo thứ tự route
    x_coords = cities[route, 0]
    y_coords = cities[route, 1]
    
    # Khép kín vòng
    x_coords = np.append(x_coords, x_coords[0])
    y_coords = np.append(y_coords, y_coords[0])
    
    plt.plot(x_coords, y_coords, c='blue', linestyle='-', linewidth=1.5, alpha=0.7, label='Route')
    plt.title(title, fontsize=14)
    plt.legend()
    plt.savefig(f'results/figures/{filename}.png', dpi=150)
    plt.close()
    print(f"[INFO] Saved TSP Route: {filename}")

# ==========================================
# 2. HÀM NÂNG CAO: ROBUSTNESS & BOXPLOT
# ==========================================
def plot_robustness_convergence(results_dict, title, filename):
    """Vẽ đường Mean ± Std"""
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(results_dict))
    
    for i, (label, histories) in enumerate(results_dict.items()):
        min_len = min(len(h) for h in histories)
        histories = [h[:min_len] for h in histories]
        data = np.array(histories)
        
        mean_fitness = np.mean(data, axis=0)
        std_fitness = np.std(data, axis=0)
        generations = range(len(mean_fitness))
        
        plt.plot(generations, mean_fitness, label=label, color=colors[i], linewidth=2)
        plt.fill_between(generations, mean_fitness - std_fitness, mean_fitness + std_fitness, 
                         color=colors[i], alpha=0.2)

    plt.title(f'Robustness Analysis - {title}', fontsize=14)
    plt.xlabel('Generations', fontsize=12)
    plt.ylabel('Fitness (Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'results/figures/{filename}.png', dpi=150)
    plt.close()
    print(f"[INFO] Saved Robustness Plot: {filename}")

def plot_boxplot_comparison(data_dict, title, filename, ylabel='Fitness'):
    """Vẽ Boxplot so sánh phân phối"""
    plt.figure(figsize=(8, 6))
    labels = []
    values = []
    for alg_name, scores in data_dict.items():
        labels.extend([alg_name] * len(scores))
        values.extend(scores)
        
    sns.boxplot(x=labels, y=values, palette="Set2")
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.savefig(f'results/figures/{filename}.png', dpi=150)
    plt.close()
    print(f"[INFO] Saved Boxplot: {filename}")

def plot_scalability_lines(x_values, y_dict, title, filename, xlabel, ylabel):
    """Vẽ biểu đồ đường so sánh Scalability (Time hoặc Fitness theo Size)"""
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D']
    
    for i, (alg_name, values) in enumerate(y_dict.items()):
        plt.plot(x_values, values, marker=markers[i % len(markers)], label=alg_name, linewidth=2)
        
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/figures/{filename}.png', dpi=150)
    plt.close()
    print(f"[INFO] Saved Scalability Line Plot: {filename}")

def plot_parameter_sensitivity(matrix, x_labels, y_labels, title, filename, xlabel, ylabel):
    """Vẽ Heatmap độ nhạy tham số"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=x_labels, yticklabels=y_labels, 
                annot=True, fmt=".4f", cmap="viridis", cbar_kws={'label': 'Mean Fitness/Cost'})
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.savefig(f'results/figures/{filename}.png', dpi=150)
    plt.close()
    print(f"[INFO] Saved Heatmap: {filename}")

def plot_3d_landscape_path(func, bounds, trajectories_dict, title, filename, resolution=50):
    """
    Vẽ 3D Surface + Trajectories của nhiều thuật toán
    trajectories_dict: {'DE': [[x1,y1], [x2,y2]...], 'HC': [...]}
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Vẽ mặt phẳng
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]]) # Hàm 2 chiều

    surf = ax.plot_surface(X, Y, Z, cmap='gray', edgecolor='none', alpha=0.3)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Vẽ đường đi
    colors = ['red', 'blue', 'green']
    for i, (alg_name, path) in enumerate(trajectories_dict.items()):
        if len(path) > 0:
            path = np.array(path)
            px, py = path[:, 0], path[:, 1]
            pz = np.array([func(p) for p in path])
            
            # Start/End
            ax.scatter(px[0], py[0], pz[0], marker='x', s=50, label=f'{alg_name} Start')
            ax.scatter(px[-1], py[-1], pz[-1], marker='o', s=50, label=f'{alg_name} End')
            ax.plot(px, py, pz, color=colors[i % len(colors)], linewidth=2, label=alg_name)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')
    ax.legend()
    plt.savefig(f'results/figures/{filename}.png', dpi=150)
    plt.close()
    print(f"[INFO] Saved 3D Trajectory: {filename}")