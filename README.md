# Biology-based Algorithms (ACO, PSO, ABC)

Đồ án 1 - Search & Nature-Inspired Algorithms  
Phần: **Biology-based Feature** (Ant Colony, Particle Swarm, Artificial Bee Colony)

## Cấu trúc project

```
BiologyFeature/
├── main.py                          # Script chạy tất cả thực nghiệm
├── requirements.txt                 # Dependencies
├── results/figures/                  # Biểu đồ kết quả (auto-generated)
├── src/
│   ├── algorithms/
│   │   ├── biology/
│   │   │   ├── aco.py               # Ant Colony Optimization (TSP)
│   │   │   ├── pso.py               # Particle Swarm Optimization (Continuous)
│   │   │   └── abc.py               # Artificial Bee Colony (Continuous)
│   │   └── classical/
│   │       └── baselines_bio.py     # BFS, DFS, A*, Hill Climbing (Baseline)
│   └── utils/
│       ├── problems_bio.py          # Định nghĩa bài toán (TSP cities, Sphere, Rastrigin)
│       └── visualization_bio.py     # Vẽ biểu đồ (Convergence, Boxplot, Heatmap, 3D)
└── tests/
    └── test_algorithms.py           # Unit tests
```

## Thuật toán

| Thuật toán | Bài toán | Mô tả |
|-----------|---------|-------|
| **ACO** | TSP (Discrete) | Mô phỏng đàn kiến, dùng pheromone + heuristic |
| **PSO** | Continuous Opt. | Mô phỏng đàn chim, velocity = inertia + cognitive + social |
| **ABC** | Continuous Opt. | Mô phỏng đàn ong, 3 pha: Employed/Onlooker/Scout |

## Cách chạy

```bash
# Cài dependencies
pip install -r requirements.txt

# Chạy unit tests
python tests/test_algorithms.py

# Chạy full thực nghiệm
python main.py
```

## Thực nghiệm

### Experiment 1: TSP (ACO vs BFS/DFS/A*)
- **Scalability**: Thời gian chạy vs số thành phố
- **Robustness**: ACO 30 runs vs Optimal (A*)
- **Sensitivity**: Alpha, Beta parameters

### Experiment 2: Continuous (PSO vs ABC vs HC)
- **Convergence**: So sánh tốc độ hội tụ trên Sphere & Rastrigin
- **Quality**: Boxplot phân phối kết quả
- **3D Trajectory**: Đường đi trên mặt phẳng Rastrigin
- **Scalability**: Thời gian vs số chiều
- **Sensitivity**: PSO params (w, c1, c2)
