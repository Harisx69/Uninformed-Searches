# AI Pathfinder — Uninformed Search Visualizer

An interactive Python visualization tool that demonstrates how classical uninformed search algorithms explore a grid world to find a path from a start node to a target node.

This project is designed for understanding algorithm behavior step-by-step through real-time animation.

Features

Interactive grid-based environment

Visual comparison of multiple uninformed search algorithms

Step-by-step animation of:

Frontier expansion

Node exploration

Final shortest (or discovered) path

Performance statistics:

Nodes explored

Path length

Time elapsed

Adjustable animation speed

Reset and re-run functionality

# Implemented Algorithms

1.Breadth-First Search (BFS)
Explores level by level. Guarantees shortest path in unweighted graphs.

2.Depth-First Search (DFS)
Explores deep before backtracking. Not guaranteed optimal.

3.Uniform Cost Search (UCS)
Expands the lowest-cost node first. Optimal with varying step costs (diagonal vs straight moves supported).

4.Depth-Limited Search (DLS)
DFS with a fixed depth limit.

5.Iterative Deepening DFS (IDDFS)
Repeated DFS with increasing depth. Combines DFS memory efficiency with BFS optimality (in unweighted cases).

6.Bidirectional Search
Simultaneously searches from start and goal until meeting in the middle.

# Grid Environment

Fixed grid size: 18 × 22

Predefined wall layout

Start and goal positions are predefined

Supports diagonal and straight movement

Movement cost:

Straight: 1.0

Diagonal: 1.414

# Visualization Legend

Start Node

Target Node

Frontier Nodes

Explored Nodes

Final Path

Walls

# Requirements

Make sure you have Python 3 installed.

Install dependencies:

pip install numpy matplotlib
How to Run

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Run the script:

python3 pathfinder.py

The visualization window will open.

# Controls

Run – Start algorithm visualization

Reset – Clear the grid and reset stats

Speed – Cycle between Slow / Normal / Fast / Turbo

Radio Buttons – Switch between algorithms

Educational Purpose

This project is ideal for:

Artificial Intelligence coursework

Understanding uninformed search strategies

Comparing algorithm efficiency

Demonstrating search behavior visually

Possible Extensions

Add heuristic-based algorithms (A*, Greedy Best-First Search)

Allow user-drawn walls

Custom start and goal selection

Dynamic grid resizing

Heuristic visualization overlay
