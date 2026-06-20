# SRC

This folder contains the solver implementation used in the project.
The main entry point is [Solver.py](Solver.py), which defines the `Solver` class.
[utils.py](utils.py) : 

## Overview

The solver works on a weighted graph where node `0` is the depot and every other relevant node may contain gold.
Travel cost depends both on distance and on the amount of gold currently carried.

For each traversed edge with distance $d$ and carried gold $w$, the incremental cost is:

$$
d + (\alpha \cdot d \cdot w)^\beta
$$

where:
- `d` is the edge distance
- `w` is the current carried gold
- `alpha` controls the penalty strength
- `beta` controls how aggressively the penalty grows

The solver must collect all gold and can return to the depot whenever unloading is convenient.

## Main idea

The implementation uses three different strategies depending on the parameters:

1. `alpha == 0` or `beta == 0`
   The cost is effectively linear in distance, so the solver builds a single tour that visits all gold cities and then returns to the depot. The tour is optimized with `optimize_gene_suboptimal()`.

2. `beta > 1.0`
   Carrying gold becomes expensive, so it is often better to split the collection of a city into multiple trips.
   In this case `solution()` uses `generate_adaptive_split()`, which:
   - estimates the best number of trips `K` for each city with `_binary_search_K()`
   - refines the return path with `_refine_trip_with_weighted_path()`
   - repeats the chosen city trip `K` times with equal gold shares

   The outgoing path depot -> city is taken from the precomputed shortest paths, while the return path city -> depot is recomputed with the weighted cost.

3. `0 < beta <= 1`
   The distance component is usually more important, so the solver runs the genetic algorithm pipeline.
   The GA pipeline is:
    1. generate an initial population using random permutations and greedy decoding
    2. sort individuals by cost
    3. repeatedly apply mutation or crossover
    4. keep the best individuals through elitism

## Representations

### Genotype

A genotype is a list of genes, where each gene is one tour:

```python
[[(1, 10)], [(2, 20), (3, 30)]]
```

Each tuple is `(city, gold_collected_at_city)`.
A gene stores only the cities where gold is collected; the implicit intermediate nodes of the shortest path are not stored explicitly.
Those paths are precomputed with Dijkstra when the solver is initialized.

### Phenotype

A phenotype is the expanded route with all implicit shortest-path nodes included:

```python
[(1, 10), (4, 0), (0, 0), (4, 0), (2, 20), (3, 30), (1, 0), (4, 0), (0, 0)]
```

This representation is useful for feasibility checks and for validating the cost on the actual graph.

## Cost evaluation

The solver exposes two related cost functions:

- `compute_cost_genotype()` evaluates a full genotype and matches the semantics used internally by the solver.
- `compute_cost_phenotype()` evaluates the fully expanded route.

The first city of the first gene is treated specially because the final solution does not need to count an explicit depot-to-first-city departure in the same way as the internal gene cost model.


# GA Algorithm methods:

## Greedy decoder

`evaluate_and_segment()` takes a chromosome, which is a permutation of relevant cities, and greedily decides whether to:

- go directly to the next target city
- detour through the depot to unload first

At each step it compares the two incremental costs and keeps the cheaper option.
It returns:

- the decoded genotype
- the total cost of the route

## Local operators

### `optimize_gene()`

Tries all permutations of a gene and keeps the best ordering.
This is useful because the cost depends on the order of visited cities, not only on the set of cities.

### `optimize_gene_suboptimal()`

Uses a farthest-insertion heuristic for larger genes.
It is the practical version used when exhaustive permutation search would be too expensive.

### `merge_genes()`

Combines two genes into one.
If the same city appears in both genes, the gold amounts are summed and the city appears only once.
The merged gene is then optimized with `optimize_gene()`.

### `split_gene()`

Splits a gene into two parts at a given index.
Each part is optimized independently and their costs are recomputed.
This is used by mutation and helps the GA explore different tour decompositions.

## Mutation and crossover

The mutation operator has two modes, with the split/merge threshold computed from the problem parameters:

- Split mutation
  - chooses a gene with at least two cities
  - splits it into two genes around the middle
  - re-evaluates the genotype cost

- Merge mutation
  - selects two random genes
  - merges them into one gene
  - re-evaluates the genotype cost

This balances exploration and exploitation:

- splitting increases the number of tours and can reduce weight penalties
- merging reduces fragmentation and can improve route compactness

`crossover()` builds an order-based offspring chromosome, decodes it with `evaluate_and_segment()`.


## Notes

- `solution()` selects the strategy automatically from `alpha` and `beta`.
- The code includes feasibility checks for both genotype and phenotype representations.
- The root README contains the broader project and problem statement.

## Example usage

```python
from Problem import Problem
from src.Solver import Solver

problem = Problem(num_cities=12, alpha=1.0, beta=1.5, density=0.7, seed=7)
solver = Solver(problem)

phenotype, cost = solver.solution(fast=True)
print(cost)
print(phenotype)
```



