# PROJECT REPORT

## Problem Description

Combinatorial optimization problem where a thief must visit a set of cities on a graph to collect gold, starting and returning to a depot (city 0). The challenge lies in the fact that carrying gold increases the travel cost according to the formula:


Cost: $d + (d \cdot \alpha \cdot w)^\beta$ with $\alpha \ge 0$ and $\beta \ge 0$


where:
- `distance` is the edge length between two cities
- `weight` is the current gold being carried
- `alpha` and `beta` are problem-specific parameters that control how heavily the weight penalty affects the total cost

The thief can return to the depot at any point to unload gold (resetting the weight to zero) before continuing the journey. The objective is to collect all available gold while minimizing the total travel cost.

## Solution Approach

The solution is based on a **Genetic Algorithm (GA)** with specialized operators designed for this routing problem with weight constraints.

### Core Components

1. **Genotype Encoding**  
   Each individual is represented as a sequence of `(city, gold)` tuples, where the route may include multiple returns to the depot (0, 0) to unload collected gold.

2. **Greedy Decoder (`_evaluate_and_segment`)**  
   Given a  (random) city target order, this decoder greedily decides at each step whether to:
   - Go directly to the next city
   - Return to the depot to unload, then proceed to the next city  
   
   It chooses the option with lower incremental cost.

3. **Initialization**  
   The initial population combines:
   - **Greedy individuals**: Random permutations decoded with the greedy strategy
   - **Improved Baseline**: Starting from the baseline, adiacent tours are merged if it's favorable

4. **Genetic Operators**

   - **Mutation**: Two strategies based on problem parameters:
     - **Tour merging**: Attempts to merge two consecutive tours into one if cost-effective
     - **Tour re-decoding**: Splits the route into segments and re-applies the greedy decoder to each segment for local optimization
     The choice between these two strategies is **dynamically tuned** based on problem parameters:
      - Low `beta` (distance-dominant): favors tour merging
      - High `beta` (weight-dominant): favors tour splitting and re-decoding   
      - If `beta` ≈ 1: favors tour merging for `alpha` < 1

   The threshold is computed dynamically at each mutation to balance exploration and exploitation.
    
   - **Crossover**: Combines two parents by taking tours from the first parent to cover approximately half the cities, then completes gold collection using tours from the second parent (re-optimized)

5. **Selection**  
   Tournament selection with elitism: only the best individuals survive to the next generation.
   The individual are sorted in base of their cost and only the first :population size are kept.



### Key Algorithmic Features

- **Multiple Cycle Strategy**: When weight penalties are severe, tours are split into multiple lighter trips collecting fractional gold amounts per visit (`_multiple_cycle`)
- **Improved Baseline**: Starts from a nearest-neighbor baseline and iteratively merges adjacent tours if beneficial (`_improved_baseline_individual`)

### Parameters

Parameters are dynamically adjusted based on `beta` and `n_cities` to balance solution quality and computation time:

- When `beta` is large and tours are split into many light trips, computation time grows significantly with the number of cities.
- For such cases, smaller populations are used to remain tractable.
- When `beta` is small, larger populations provide better exploration.


**Configurable parameters:**
- `pop_size`: Population size
- `generations`: Number of GA iterations
- `offprint`: Number of offspring generated per generation

Optionally, a boolean flag can be passed to `solution()` to enable **stagnation control**, which stops the GA early if the best cost does not improve for more than 10 consecutive generations.

The algorithm balances exploration (through mutation/crossover diversity) and exploitation (through greedy decoding and local optimization) to find high-quality solutions across different problem configurations. 


### Collaboration
In order to produce this solution I share ideas with 3 collegues: Davide Carletto (s339425),  Michele Carena (349483), Alessandro Benvenuti (343748)