# PROJECT REPORT

## Problem Description

Combinatorial optimization problem where a thief must visit a set of cities on a graph to collect gold, starting and returning to a depot (city 0). The challenge lies in the fact that carrying gold increases the travel cost according to the formula:


Cost: $d + (d \cdot \alpha \cdot w)^\beta$ with $\alpha \ge 0$ and $\beta \ge 0$


where:
- `distance` is the edge length between two cities
- `weight` is the current gold being carried
- `alpha` and `beta` are problem-specific parameters that control how heavily the weight penalty affects the total cost

The thief can return to the depot at any point to unload gold (resetting the weight to zero) before continuing the journey. The objective is to collect all available gold while minimizing the total travel cost.

## Project structure

- `src/`: Solver implementations and core project code.
	- `Solver.py`: Main Genetic Algorithm solver (`Solver`) and adaptive-split logic.
	- `utils.py`: Utility functions used throughout the codebase.
- `tests/`: Unit and integration tests (uses `unittest`). Key file: `tests/test_solver.py`.
- `experiments.ipynb`: Jupyter notebook for experiments, visualizations and ad-hoc runs.
- `Problem.py`: Problem model and input parsing helpers (defines the problem instance API).
- `base_requirements.txt`: List of Python dependencies required for development and testing.
- `s345905.py`: Auxiliary script (student/assignment related) and quick-run utilities.
- `all_configurations_results.csv`: Aggregated results from parameter sweeps and experiments. 


### Collaboration
In order to produce this solution I share ideas with 3 collegues: Davide Carletto (s339425),  Michele Carena (349483), Alessandro Benvenuti (343748). 
