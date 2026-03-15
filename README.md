# 🔬 Search & Nature-Inspired Algorithms — Benchmark & Visualization

A comprehensive Python framework for **benchmarking, comparing, and visualizing** nature-inspired and classical search algorithms across both **continuous** and **discrete** optimization problems.

> Run a single command to generate quality comparisons, convergence curves, scalability charts, sensitivity heatmaps, 3D trajectory plots, and animated landscape GIFs — all with CSV exports for further analysis.

---

## ✨ Features

- **11 optimization algorithms** spanning 5 paradigms (biology, evolution, physics, human-based, classical)
- **9 problem domains** — 5 continuous benchmark functions + 4 discrete combinatorial problems
- **Automated experiment pipeline** — quality, convergence, scalability, and sensitivity analysis in one run
- **Rich visualizations** — boxplots, convergence ribbons, scalability lines, parameter heatmaps, 3D surfaces, and animated rotating GIFs
- **YAML-driven configuration** — tweak every hyperparameter from a single `config.yaml`
- **CSV export** — raw scores, convergence histories, and timing data saved alongside figures

---

## 📂 Project Structure

```
.
├── main.py                          # Entry point — orchestrates all experiments
├── config.yaml                      # Centralized hyperparameters & experiment settings
├── requirements.txt                 # Python dependencies
├── src/
│   ├── algorithms/
│   │   ├── biology/                 # Swarm & bio-inspired algorithms
│   │   │   ├── abc.py               #   Artificial Bee Colony (ABC)
│   │   │   ├── aco.py               #   Ant Colony Optimization — TSP
│   │   │   ├── aco_graph_coloring.py#   ACO — Graph Coloring
│   │   │   ├── aco_shortest_path.py #   ACO — Shortest Path
│   │   │   ├── cuckoo_search.py     #   Cuckoo Search (CS)
│   │   │   ├── fa.py                #   Firefly Algorithm (FA)
│   │   │   └── pso.py               #   Particle Swarm Optimization (PSO)
│   │   ├── evolution/               # Evolutionary algorithms
│   │   │   ├── differential_evolution.py  # Differential Evolution (DE)
│   │   │   ├── genetic_algorithm.py       # GA — TSP
│   │   │   └── ga_knapsack.py             # GA — Knapsack
│   │   ├── physics/                 # Physics-inspired algorithms
│   │   │   └── sa.py                #   Simulated Annealing (SA)
│   │   ├── human/                   # Human-behavior-inspired algorithms
│   │   │   └── tlbo.py              #   Teaching-Learning-Based Optimization (TLBO)
│   │   ├── classical/               # Traditional search baselines
│   │   │   └── baselines.py         #   Hill Climbing (HC), BFS, DFS, A* for TSP
│   │   └── knapsack_solvers.py      # Unified Knapsack adapters for all algorithms
│   ├── problems/
│   │   ├── continuous/              # (Benchmark functions defined in utils/problems.py)
│   │   └── discrete/
│   │       ├── graph_coloring.py    #   Graph Coloring (GA, SA, ACO, HC, DFS)
│   │       └── shortest_path.py     #   Shortest Path (A*, BFS, DFS, ACO)
│   └── utils/
│       ├── problems.py              # Benchmark functions & problem generators
│       └── visualization.py         # All plotting: convergence, boxplots, 3D, GIFs
├── results/
│   ├── figures/                     # Generated PNG plots & GIF animations
│   │   ├── continuous/{function}/   #   Per-function figures (sphere, rastrigin, …)
│   │   └── discrete/{problem}/      #   Per-problem figures (tsp, knapsack, …)
│   └── csv/                         # Exported CSV data
│       ├── continuous/{function}/
│       └── discrete/{problem}/
└── tests/                           # Test directory
```

---

## 🧠 Algorithms

### Biology / Swarm-Inspired

| Abbreviation | Algorithm | Key Idea |
|:---:|---|---|
| **PSO** | Particle Swarm Optimization | Swarm of particles guided by personal & global bests |
| **ABC** | Artificial Bee Colony | Employed, onlooker, and scout bee phases |
| **FA** | Firefly Algorithm | Light-attraction-based movement with randomization decay |
| **CS** | Cuckoo Search | Lévy flights + nest abandonment |
| **ACO** | Ant Colony Optimization | Pheromone-based path construction (TSP, Graph Coloring, Shortest Path) |

### Evolutionary

| Abbreviation | Algorithm | Key Idea |
|:---:|---|---|
| **DE** | Differential Evolution | Mutation, crossover, and selection on real-valued vectors |
| **GA** | Genetic Algorithm | Crossover, mutation, and elitism (TSP & Knapsack variants) |

### Physics-Inspired

| Abbreviation | Algorithm | Key Idea |
|:---:|---|---|
| **SA** | Simulated Annealing | Probabilistic acceptance with geometric cooling schedule |

### Human-Behavior-Inspired

| Abbreviation | Algorithm | Key Idea |
|:---:|---|---|
| **TLBO** | Teaching-Learning-Based Optimization | Parameter-free teacher & learner phases |

### Classical Baselines

| Abbreviation | Algorithm | Key Idea |
|:---:|---|---|
| **HC** | Hill Climbing | Greedy neighbor search (continuous & discrete) |
| **BFS** | Breadth-First Search | Exhaustive level-order search |
| **DFS** | Depth-First Search | Exhaustive depth-order / backtracking search |
| **A*** | A-Star Search | Heuristic-guided optimal search |

---

## 📊 Optimization Problems

### Continuous Benchmark Functions

| Function | Global Minimum | Default Bounds | Characteristics |
|---|:---:|:---:|---|
| **Sphere** | f(0) = 0 | [−5.12, 5.12] | Unimodal, smooth, convex |
| **Rastrigin** | f(0) = 0 | [−5.12, 5.12] | Highly multimodal, regular local minima |
| **Rosenbrock** | f(1) = 0 | [−5, 10] | Narrow curved valley ("banana") |
| **Griewank** | f(0) = 0 | [−600, 600] | Regularly distributed local minima |
| **Ackley** | f(0) = 0 | [−32, 32] | Nearly flat outer region, steep center |

### Discrete Combinatorial Problems

| Problem | Objective | Algorithms Used |
|---|---|---|
| **Traveling Salesman (TSP)** | Minimize tour cost | GA, ACO, SA, HC + BFS/DFS/A* baselines |
| **0/1 Knapsack** | Maximize total value under capacity constraint | DFS, HC, GA, SA, PSO, DE, ABC, CS, FA, TLBO |
| **Graph Coloring** | Minimize edge conflicts with k colors | GA, SA, ACO, HC, DFS |
| **Shortest Path** | Minimize weighted path cost | A*, BFS, DFS, ACO |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/takiet2410-hub/Search-Nature-InspiredAlgorithms.git
cd Search-Nature-InspiredAlgorithms

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical computing & array operations |
| `matplotlib` | Plotting & 3D visualizations |
| `seaborn` | Statistical plot styling (boxplots, heatmaps) |
| `pyyaml` | YAML configuration parsing |

---

## ▶️ Usage

### Run All Experiments

```bash
python main.py
```

This executes the full experiment pipeline in order:

1. **Discrete Experiments** — TSP → Knapsack → Graph Coloring → Shortest Path
2. **Continuous Experiments** — All 8 algorithms × 5 functions (quality, convergence, scalability, sensitivity)
3. **3D Trajectory Plots** — Search paths overlaid on benchmark function surfaces
4. **Animated 3D Landscapes** — 360° rotating surface GIFs for each benchmark function

> ⏱️ A complete run with default settings (30 runs × 8 algorithms × 5 functions + discrete problems) may take **several hours**. Adjust `experiment.runs` in `config.yaml` to reduce runtime.

---

## ⚙️ Configuration

All experiment parameters are centralized in **`config.yaml`**:

```yaml
experiment:
  runs: 30          # Independent runs per algorithm (for statistical robustness)
  pop_size: 50      # Default population size

continuous:
  problems:
    Sphere:
      dim: 10
      bound: [-5.12, 5.12]
      generations: 50
    # ... Rastrigin, Rosenbrock, Griewank, Ackley

algorithms:
  de:
    mutation_factor: 0.8
    crossover_rate: 0.7
  pso:
    w: 0.7
    c1: 1.5
    c2: 1.5
  # ... abc, fa, cs, sa, hc, ga_tsp, aco_tsp, tlbo
```

Key sections:

| Section | Controls |
|---|---|
| `experiment` | Number of runs, default population size |
| `continuous.problems` | Dimension, bounds, and generations per benchmark function |
| `continuous.scalability` | Dimension sweep settings for scalability tests |
| `algorithms.*` | Per-algorithm hyperparameters |
| `tsp` | City counts for TSP experiments |
| `knapsack` | Item counts and generation settings |
| `graph_coloring` | Node counts, edge probability, per-solver settings |
| `shortest_path` | Node counts, edge probability, ACO settings |

---

## 📈 Outputs

All results are saved under `results/`:

### Figures (`results/figures/`)

| Type | Description | Format |
|---|---|---|
| **Boxplot** | Distribution of solution quality across 30 runs | PNG |
| **Convergence** | Mean ± std fitness over generations (log scale) | PNG |
| **Scalability** | Execution time vs. problem size | PNG |
| **Sensitivity Heatmap** | Mean fitness across parameter grid | PNG |
| **3D Trajectory** | Algorithm search paths on function surface | PNG |
| **Animated 3D Landscape** | 360° rotating surface visualization | GIF |
| **TSP Route** | Best route visualization for each algorithm | PNG |

### CSV Data (`results/csv/`)

| Type | Columns |
|---|---|
| **Scores** | `Run, Alg1, Alg2, …` + summary stats (Mean, Std, Best, Worst) |
| **Convergence** | `Generation, Alg1_Mean, Alg1_Std, …` |
| **Scalability** | `Size, Alg1_Time, Alg2_Time, …` |
| **Sensitivity** | Parameter grid × mean fitness matrix |

---

## 🏗️ Architecture

The framework uses a **registry pattern** for extensibility:

```python
# Register a new continuous algorithm
@register_continuous("MY_ALG")
def run_my_alg(func, bounds, generations, pop_size):
    alg = MyAlgorithm(func, bounds, pop_size=pop_size)
    _, score, history, path = alg.optimize(generations=generations)
    return score, history, path
```

### Comparison Engines

| Class | Domain | Analyses |
|---|---|---|
| `ContinuousComparison` | Benchmark functions | Quality, convergence, scalability, sensitivity |
| `DiscreteComparison` | TSP | Scalability, quality vs. optimal, sensitivity |
| `KnapsackComparison` | 0/1 Knapsack | Quality, scalability, sensitivity |
| `GraphColoringComparison` | Graph Coloring | Quality, scalability, sensitivity |
| `ShortestPathComparison` | Shortest Path | Quality, convergence, scalability, sensitivity |

Each engine automatically runs **multiple analysis criteria** and exports both figures and CSV data into organized subdirectories.

---

## 📄 License

This project is provided for educational and research purposes.

---

## 🙏 Acknowledgments

- Benchmark functions from the standard optimization test function literature
- Algorithm implementations inspired by original research papers and textbook descriptions
