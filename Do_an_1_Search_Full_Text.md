
VNUHCM-UNIVERSITY OF SCIENCE  
FACULTY OF INFORMATION TECHNOLOGY  

Team Project | Fundamentals of Artificial Intelligence  

==================================================

TEAM PROJECT 01 – SEARCH & NATURE-INSPIRED ALGORITHMS  
CSC14003 – FUNDAMENTALS OF ARTIFICIAL INTELLIGENCE  

--------------------------------------------------

## 1. PROJECT OVERVIEW

This project focuses on implementing, analyzing, and comparing classical graph search algorithms and nature-inspired optimization algorithms using only NumPy and basic Python for implementing data structures and optimization process from scratch.

The project is divided into two major algorithm families:

1. Classical Search Algorithms on Graphs  
2. Nature-Inspired Algorithms, including:
   1. Evolution-based algorithms  
   2. Physics-based algorithms  
   3. Biology-based algorithms  
   4. Human behavior-based algorithms  

Through this project, students will understand how different search paradigms explore state spaces and optimization landscapes, ranging from deterministic graph traversal to stochastic population-based optimization.

--------------------------------------------------

## 2. LEARNING OBJECTIVES

By completing this project, students will:

- Understand the theoretical foundations of graph search and metaheuristic optimization.
- Distinguish between exact search algorithms and approximate/metaheuristic methods.
- Implement classical and nature-inspired algorithms from scratch.
- Analyze convergence behavior, exploration–exploitation trade-offs, and scalability.
- Apply different algorithm families to both discrete and continuous problems.
- Develop skills in visualization, experimentation, and scientific reporting.
- Collaborate effectively in a team-based AI project.

--------------------------------------------------

## 3. ALGORITHM TAXONOMY

### 3.1 Classical Search Algorithms on Graphs

**Category – Algorithms**

Uninformed Search:
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Uniform Cost Search (UCS)

Informed Search:
- Greedy Best-First Search
- A* Search

Local Search:
- Hill Climbing (Steepest Ascent)
- Simulated Annealing

--------------------------------------------------

### 3.2 Nature-Inspired Algorithms

### 3.2.1 Evolution-Based Algorithms

Inspired by Darwinian evolution: selection, crossover, mutation.

Algorithm – Inspiration – Applications

- Genetic Algorithm (GA): Population evolution via crossover & mutation – Feature selection, scheduling
- Differential Evolution (DE): Vector differences for mutation – Continuous optimization
- Evolution Strategies (ES): Self-adaptive mutation – Engineering optimization

--------------------------------------------------

### 3.2.2 Physics-Based Algorithms

Inspired by physical laws and phenomena.

Algorithm – Inspiration – Applications

- Simulated Annealing (SA): Thermodynamic cooling – Combinatorial optimization
- Gravitational Search Algorithm (GSA): Newtonian gravity – Continuous optimization
- Harmony Search (HS): Musical harmony principles – Parameter tuning

--------------------------------------------------

### 3.2.3 Biology-Based Algorithms

Inspired by collective intelligence in biological systems.

Algorithm – Strengths – Weaknesses – Best Applications

Ant Colony Optimization (ACO) [1]:
- Strengths: Effective for combinatorial problems and handles complex discrete spaces well
- Weaknesses: Computationally intensive and requires fine-tuning
- Applications: Routing problems, scheduling, and resource allocation

Particle Swarm Optimization (PSO) [2]:
- Strengths: Good for continuous optimization, simple and easy to implement
- Weaknesses: Can converge to local optima and is less effective for discrete problems
- Applications: Hyperparameter tuning, engineering design, financial modeling

Artificial Bee Colony (ABC) [3]:
- Strengths: Adaptable to large, dynamic problems and balanced exploration and exploitation
- Weaknesses: Computationally intensive and requires careful parameter tuning
- Applications: Telecommunications, large-scale optimization, high-dimensional spaces

Firefly Algorithm (FA) [4]:
- Strengths: Excels in multimodal optimization and has strong global search ability
- Weaknesses: Sensitive to parameter settings and slower convergence
- Applications: Image processing, engineering design, multimodal optimization

Cuckoo Search (CS) [5]:
- Strengths: Efficient for solving optimization problems and has strong exploration capabilities
- Weaknesses: May converge prematurely and performance depends on tuning
- Applications: Scheduling, feature selection, engineering applications

--------------------------------------------------

### 3.2.4 Human Behavior-Based Algorithms

Inspired by social and cognitive behaviors.

Algorithm – Inspiration – Applications / Notes

- Teaching–Learning-Based Optimization (TLBO): Classroom learning – Parameter-free
- Social Force Optimization (SFO): Crowd dynamics – Continuous domains
- Cultural Algorithm (CA): Cultural evolution – Parameter tuning

--------------------------------------------------

## 4. IMPLEMENTATION REQUIREMENTS

Students are required to work in group for performing this project.

You have to implement:

Evolution-based:
- Genetic Algorithm (GA)
- Differential Evolution (DE)

Physics-based:
- Simulated Annealing (SA)

Biology-based:
- ACO, PSO, ABC, FA, CS

Human-based:
- TLBO (optional, bonus)

Then, compare against at least four traditional search algorithms:
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- A* Search
- Hill Climbing
- Simulated Annealing

Implementation requirements:

- All algorithms must be implemented using NumPy only (no scikit-learn, scipy.optimize, or other high-level libraries)
- Code should be modular, well-documented, and follow Python best practices
- Each algorithm should have configurable parameters (population size, iterations, etc.)
- Implementations should handle both continuous and discrete optimization problems

--------------------------------------------------

## 5. EXPERIMENTS & VISUALIZATION

Students are required to create visualizations that demonstrate:

- Convergence ability
- Comparative performance
- Parameter sensitivity analysis
- (Advanced) 3D surface plots to show the objective function landscape

Recommended libraries:
- Matplotlib
- Seaborn

Recommended metrics:

- Convergence speed
- Best / average solution quality
- Computational complexity (time and space)
- Robustness (mean ± std over multiple runs)
- Scalability (performance with problem size)
- Exploration vs exploitation behavior

--------------------------------------------------

## 6. TEST PROBLEMS

Continuous Optimization:
- Sphere function (convex, unimodal)
- Rastrigin function (highly multimodal)
- Rosenbrock function (narrow valley)
- Griewank function (regularly distributed minima)
- Ackley function (many local optima)

Discrete Optimization:
- Traveling Salesman Problem (TSP) with constraints time and cost
- Knapsack Problem (KP)
- Graph Coloring (GC)
- Shortest Path

--------------------------------------------------

## 7. REPORT REQUIREMENTS

The report must fully give the following sections:

- Member information (student ID, full name, ...)
- Work assignment table with completion rate
- Self-evaluation
- Detailed algorithm descriptions by level
- Test cases and experimental results
- Well-formatted PDF output
- References (APA format)
- Language usage: Vietnamese or English

Recommended report structure:

1. Chapter 1 – Introduction & Algorithm Taxonomy  
2. Chapter 2 – Classical Graph Search Algorithms  
3. Chapter 3 – Local Search & Physics-Based Algorithms  
4. Chapter 4 – Evolution-Based Algorithms  
5. Chapter 5 – Swarm & Biology-Based Algorithms  
6. Chapter 6 – Human-Inspired Algorithms  
7. Chapter 7 – Discussion & Insights  
8. Chapter 8 – Conclusion & Future Work  

--------------------------------------------------

## 8. SUBMISSION

- Report, source code, and test cases in a compressed .zip file named `<Group_ID>.zip`
- Demo videos uploaded to YouTube or Google Drive (public URLs included)
- If file size > 25MB, prioritize compressing report and source code

--------------------------------------------------

## 9. ASSESSMENT

1. Technical Report – 40%
   - Algorithm descriptions with mathematical formulations
   - Implementation details
   - Experimental methodology
   - Results analysis
   - Comparative discussion
   - Minimum 25 pages

2. Source Code – 40%
   - Python implementation of swarm algorithms
   - Traditional search algorithms
   - README with setup instructions (uploaded on GitHub)

3. Demo – 20%
   - Video demo ≥ 5 minutes

--------------------------------------------------