# SRC

This folder contains the final solver implementation used by the project.

## `GA_solver_2.py`

`GA_solver_2.py` implements a genetic algorithm tailored to the gold-collection problem on a weighted graph.
The solver works on a graph where city `0` is the depot and every other relevant city may contain some gold.
Travel cost depends both on distance and on the amount of gold currently carried.

## Problem model

For each traversed edge with distance `d` and carried gold `w`, the incremental cost is:

$$
 d + (\alpha \cdot d \cdot w)^\beta
$$

where:
- `d` is the edge distance
- `w` is the current carried gold
- `alpha` controls the penalty strength
- `beta` controls how aggressively the penalty grows

The thief must collect all gold and can return to the depot at any time to unload.

## Representation

### Genotype
A genotype is a list of genes, where each gene is one tour:

```python
[[(1, 10)], [(2, 20), (3, 30)]]
```

Each tuple is `(city, gold_collected_at_city)`.
A gene represents a route that starts from the depot, visits the listed cities in order, and returns to the depot.
The genotype only stores the cities where gold is collected, not the implicit intermediate nodes of shortest paths.

### Phenotype
A phenotype is the expanded route with all implicit shortest-path nodes included:

```python
[(1, 10), (4, 0), (0, 0), (4, 0), (2, 20), (3, 30), (1, 0), (4, 0), (0, 0)]
```

The phenotype is useful for feasibility checks and for cost validation on the real graph.

## Cost computation

The solver uses two related cost evaluators:

- `compute_cost_genotype()` evaluates a full genotype and matches the semantics used by the GA.
- `compute_cost_phenotype()` evaluates the fully expanded route.

The first city of the first gene is treated specially because the final solution does not need to count an explicit depot-to-first-city departure in the same way as the internal gene cost model.

## Greedy decoder

`evaluate_and_segment()` takes a chromosome, which is a permutation of relevant cities, and greedily decides whether to:

- go directly to the next target city
- detour through the depot to unload first

At each step it compares the two incremental costs and keeps the cheaper option.
The function returns:

- the decoded genotype
- the total cost of the route

## Local gene operators

### `optimize_gene()`
Tries all permutations of a single gene and keeps the best ordering.
This is useful because the cost depends on the order of visited cities, not only on the set of cities.

### `merge_genes()`
Combines two genes into one.
If the same city appears in both genes, the gold amounts are summed and the city appears only once.
The merged gene is then optimized with `optimize_gene()`.

### `split_gene()`
Splits a gene into two parts at a given index.
Each part is optimized independently and their costs are recomputed.
This is used by mutation and helps the GA explore alternative decompositions of a tour.

## Mutation strategy

The current mutation operator has two main modes:

- **Split mutation** with probability `<= 0.6`
  - chooses a gene with at least two cities
  - splits it into two genes around the middle
  - re-evaluates the genotype cost

- **Merge mutation** with probability `> 0.6`
  - selects two random genes
  - merges them into one gene
  - re-evaluates the genotype cost

This design balances exploration and exploitation:

- splitting increases the number of tours and can reduce weight penalties
- merging reduces fragmentation and can improve route compactness

## Multiple-cycle improvement

When `beta > 1`, the solver can call `_multiple_cycle()` to split tours into several lighter trips if this is cheaper.
This is useful when weight penalties dominate travel cost.
The method duplicates a gene into multiple lighter versions, redistributing the gold across the new trips.

## Population and GA loop

The GA pipeline is:

1. generate an initial population using random permutations and greedy decoding
2. optionally apply the multiple-cycle improvement
3. sort individuals by cost
4. repeatedly apply mutation or crossover
5. keep the best individuals through elitism

## Public API

The solver exposes the following main methods:

- `generate_initial_population()`
- `mutation()`
- `crossover()`
- `run_ga_logic()`
- `solution()`

## Notes

- `crossover()` is currently a placeholder in this version of the solver and returns the first parent unchanged.
- The code includes feasibility checks for both genotype and phenotype representations.
- The README in the project root contains a higher-level description of the overall project and problem statement.

## Example usage

```python
from Problem import Problem
from src.GA_solver_2 import GA_Solver

problem = Problem(num_cities=12, alpha=1.0, beta=1.5, density=0.7, seed=7)
solver = GA_Solver(problem, pop_size=20, generations=30, offprint=10)

phenotype, cost = solver.solution(fast=True)
print(cost)
print(phenotype)
```


# Cose da fare: 
Cambiare GA_solver a Solver e basta 
Scrivere il Readme bene spiegando le diverse soluzioni 
Pulire la classe dalle funzioni non utilizzate o vecchie 

NOTA BENE: nel caso beta > 1 uso solo i full_path che partono da zero così c'è troppo overhead.

Rirannare tutte le compinazioni 
