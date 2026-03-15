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

def _save_figure(filename):
    full_path = f'results/figures/{filename}.png'
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    plt.savefig(full_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {full_path}")

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
    _save_figure(filename)
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
    _save_figure(filename)
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
    _save_figure(filename)
    plt.close()
    print(f"[INFO] Saved Robustness Plot: {filename}")

def plot_boxplot_comparison(data_dict, title, filename, ylabel='Fitness'):
    """Vẽ Boxplot so sánh phân phối - Đã tinh chỉnh"""
    plt.figure(figsize=(8, 6))
    labels = []
    values = []
    for alg_name, scores in data_dict.items():
        labels.extend([alg_name] * len(scores))
        values.extend(scores)
        
    # Thêm showmeans để thấy điểm trung bình ngay cả khi hộp bị phẳng
    # Thêm notch=True nếu bạn muốn so sánh độ tin cậy giữa các trung vị
    sns.boxplot(x=labels, y=values, palette="Set2", showmeans=True, 
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
    
    # Nếu muốn thấy rõ từng điểm dữ liệu chạy (30 runs), hãy dùng thêm stripplot
    sns.stripplot(x=labels, y=values, color="orange", alpha=0.5, jitter=True)
    
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    _save_figure(filename)
    plt.close()

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
    _save_figure(filename)
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
    _save_figure(filename)
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
    _save_figure(filename)
    plt.close()
    print(f"[INFO] Saved 3D Trajectory: {filename}")


# ==========================================
# 4. ANIMATED 3D LANDSCAPE (Rotating GIF)
# ==========================================
from matplotlib.animation import FuncAnimation
import matplotlib.cm as mcm
import matplotlib.colors as mcolors

def plot_3d_landscape_animated(func, bounds, title, filename,
                                resolution=50, cmap='viridis',
                                fps=30):
    """
    Tạo ảnh GIF xoay 360° bề mặt 3D của hàm benchmark.
    Bề mặt (landscape) xoay tại chỗ, trục toạ độ đứng yên.

    Args:
        func       : Hàm mục tiêu (nhận vector 2D)
        bounds     : [[x_min, x_max], [y_min, y_max]]
        title      : Tiêu đề đồ thị
        filename   : Đường dẫn lưu (không kèm phần mở rộng)
        resolution : Số điểm lưới mỗi trục (mặc định 100)
        cmap       : Bảng màu matplotlib (mặc định 'viridis')
        fps        : Số khung hình mỗi giây (mặc định 30)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- Tạo bề mặt (tính 1 lần, dùng lại Z cho mọi frame) ---
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    # --- Tâm xoay = tâm miền ---
    cx = (bounds[0][0] + bounds[0][1]) / 2.0
    cy = (bounds[1][0] + bounds[1][1]) / 2.0
    Xc = X - cx  # Toạ độ đã dời về gốc
    Yc = Y - cy

    # --- Giới hạn trục cố định (đủ chứa bề mặt khi xoay bất kỳ góc) ---
    max_r = np.sqrt(Xc**2 + Yc**2).max()
    xlim = (cx - max_r, cx + max_r)
    ylim = (cy - max_r, cy + max_r)
    zlim = (Z.min(), Z.max())

    # --- Colorbar cố định (tạo 1 lần qua ScalarMappable) ---
    norm = mcolors.Normalize(vmin=Z.min(), vmax=Z.max())
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)

    # --- Góc nhìn cố định ---
    fixed_elev, fixed_azim = 30, 45

    def _setup_axes():
        """Thiết lập lại trục sau khi xoá (ax.cla)."""
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.grid(False)
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.set_zlabel('Fitness', fontsize=11)
        ax.view_init(elev=fixed_elev, azim=fixed_azim)

    def update(frame):
        ax.cla()
        # Xoay toạ độ (X, Y) quanh tâm miền — Z giữ nguyên
        theta = np.radians(frame)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        X_rot = Xc * cos_t - Yc * sin_t + cx
        Y_rot = Xc * sin_t + Yc * cos_t + cy

        ax.plot_surface(X_rot, Y_rot, Z, cmap=cmap, alpha=0.85,
                        edgecolor='none', vmin=Z.min(), vmax=Z.max())
        _setup_axes()
        return []

    # --- Vẽ frame đầu tiên ---
    update(0)

    # --- Tạo animation 360 khung hình (1° mỗi khung) ---
    anim = FuncAnimation(fig, update, frames=range(360),
                         interval=1000 // fps, blit=False)

    # --- Lưu GIF ---
    full_path = f'results/figures/{filename}.gif'
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    anim.save(full_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"[INFO] Saved animated 3D landscape: {full_path}")