# project-work

### Repository Setup

1. Create a Git repository named project-work.
2. Inside the repository, include:
    - A Python script named Problem.py which generates the problem through the class constructor and the baseline solution. 
    - A Python file named s<student_id>.py that contains a function named solution(p:Problem) which receives as input an instance of the class Problem which generates the problem.
    - A folder named src/ containing all additional code required to run your solution.
    - A TXT file named base_requirements.txt containing the basic python libraries that you need to run the code to generate the problem.


### Main File Requirements (s<student_id>.py)

1. Import the class responsible from Problem.py for generating the problem in your code.
2. Implement a method called solution() to place in s<student_id>.py that returns the optimal path in the following format: 
```python
[(c1, g1), (c2, g2), …, (cN, gN), (0, 0)]
```
where:
- c1, …, cN represent the sequence of cities visited.
- g1, …, gN represent the corresponding gold collected at each city.


### Rules
1. The thief must start and finish at (0, 0).
2. Returning to (0, 0) during the route is allowed to unload collected gold before continuing.
3. Don't forget to change the name of the file s123456.py provided as an example ;).

### Notes
- It is not necessary to push the report.pdf or log.pdf in this repo.
- It is mandatory to upload it in "materiale" section of "portale della didattica" at least 168 hours before the exam call.
- For well commented codes, I can't ensure a higher mark but they would be very welcome.
- In case you face any issue or you have any doubt text me at the email giuseppe.esposito@polito.it and professor Squillero giovanni.squillero@polito.it.

# PROJECT REPORT

## Problem Description

The **Traveling Thief Problem** is a combinatorial optimization problem where a thief must visit a set of cities on a graph to collect gold, starting and returning to a depot (city 0). The challenge lies in the fact that carrying gold increases the travel cost according to the formula:

```
cost(edge) = distance + (alpha × distance × weight)^beta
```

where:
- `distance` is the edge length between two cities
- `weight` is the current gold being carried
- `alpha` and `beta` are problem-specific parameters that control how heavily the weight penalty affects the total cost

The thief can return to the depot at any point to unload gold (resetting the weight to zero) before continuing the journey. The objective is to collect all available gold while minimizing the total travel cost.

## Solution Approach

The solution is based on a **Genetic Algorithm (GA)** with specialized operators designed for this routing problem with weight constraints.

### Core Components

1. **Chromosome Encoding**  
   Each individual is represented as a sequence of `(city, gold)` tuples, where the route may include multiple returns to the depot (0, 0) to unload collected gold.

2. **Greedy Decoder (`_evaluate_and_segment`)**  
   Given a city visiting order, this decoder greedily decides at each step whether to:
   - Go directly to the next city
   - Return to the depot to unload, then proceed to the next city  
   
   It chooses the option with lower incremental cost.

3. **Initialization**  
   The initial population combines:
   - **Greedy individuals**: Random permutations decoded with the greedy strategy
   - **Nearest-neighbor individual**: Cities ordered by proximity, then optimally segmented using dynamic programming
   - For high `beta` values (>1), the `_multiple_cycle` function further splits tours into multiple lighter trips, where each trip starts and ends at the depot while collecting only a small fraction of gold per iteration.

4. **Genetic Operators**

   - **Mutation**: Two strategies based on problem parameters:
     - **Tour merging**: Attempts to merge two consecutive tours into one if cost-effective
     - **Tour re-decoding**: Splits the route into segments and re-applies the greedy decoder to each segment for local optimization
   
   The choice between these two strategies is **dynamically tuned** based on problem parameters:
   - Low `beta` (distance-dominant): favors tour merging
   - High `beta` (weight-dominant): favors tour splitting and re-decoding   
   - If `beta` ≈ 1: favors tour merging for `alpha` < 1

   The threshold is computed dynamically at each mutation to balance exploration and exploitation:
   
   ```python
   beta_factor = 0.5 * (1.0 - self.prob.beta)
   alpha_influence = 0.2 * (0.05 - self.prob.alpha) if 0.9 <= self.prob.beta <= 1.1 else 0
   threshold = 0.5 + beta_factor + alpha_influence
   threshold = max(0.1, min(0.9, threshold))
   ```
    
   - **Crossover**: Combines two parents by taking tours from the first parent to cover approximately half the cities, then completes gold collection using tours from the second parent (re-optimized)

5. **Selection**  
   Tournament selection with elitism: only the best individuals survive to the next generation.
   The individual are sorted in base of their cost and only the first :population size are kept.



### Key Algorithmic Features

- **Dynamic Programming Split**: The `_dynamic_programming_split` function uses DP to find the optimal segmentation of a given city sequence
- **Multiple Cycle Strategy**: When weight penalties are severe, tours are split into multiple lighter trips collecting fractional gold amounts per visit (`_multiple_cycle`)
- **Improved Baseline**: Starts from a nearest-neighbor baseline and iteratively merges adjacent tours if beneficial (`_improved_baseline_individual`)

### Parameters

Parameters are dynamically adjusted based on `beta` and `n_cities` to balance solution quality and computation time:

- When `beta` is large and tours are split into many light trips, computation time grows significantly with the number of cities.
- For such cases, smaller populations are used to remain tractable.
- When `beta` is small, larger populations provide better exploration.

```python
if beta >= 2:
    population_size = min(max(1, n_cities // beta), 200)
    generations = population_size * 2
    offprint = int(population_size * 0.2)
else:
    population_size = min(n_cities, 200)
    generations = population_size * 2
    offprint = int(population_size * 0.5)
```


**Configurable parameters:**
- `pop_size`: Population size
- `generations`: Number of GA iterations
- `offprint`: Number of offspring generated per generation

Optionally, a boolean flag can be passed to `solution()` to enable **stagnation control**, which stops the GA early if the best cost does not improve for more than 10 consecutive generations.

The algorithm balances exploration (through mutation/crossover diversity) and exploitation (through greedy decoding and local optimization) to find high-quality solutions across different problem configurations. 


