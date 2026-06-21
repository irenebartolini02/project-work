
import numpy as np
import networkx as nx
import Problem
from typing import List, Tuple

def compute_ga_params(n_cities, beta, alpha):
    """Dynamically compute GA parameters based on problem complexity."""
    
    if n_cities <= 50:
            pop_size = n_cities
            generations = 100
            offprint = int(pop_size * 0.6)
    
    elif n_cities <= 100:
        if beta >= 2:
            pop_size = min(20, n_cities)
            generations = 50
            offprint = int(pop_size * 0.25)
        else:
            pop_size = min(100, n_cities)
            generations = 80 if beta >= 2 else 100
            offprint = int(pop_size * (0.3 if beta >= 2 else 0.5))
    
    elif n_cities <= 200:
        if beta >= 2:
            pop_size = min(25, n_cities // 5)
            generations = 40
            offprint = max(3, int(pop_size * 0.2))
        else:
            pop_size = min(100, n_cities // 2)
            generations = 60 if beta >= 2 else 80
            offprint = int(pop_size * (0.25 if beta >= 2 else 0.4))
    
    else:  # n_cities > 200 (e.g., 1000)
        if beta >= 2 :
            # High beta (>2): VERY expensive initialization, minimal population
            pop_size = 3
            generations = 10
            offprint = 2
            # tanto restituiamo greedy
        else:
            # Lower beta: can afford more exploration
            pop_size = min(30, int(100 / np.sqrt(n_cities / 100)))
            generations = int(40 - 5 * beta)
            offprint = max(8, int(pop_size * 0.3))
    
    # Safety bounds
    pop_size = max(3, min(pop_size, 150))
    generations = max(10, min(generations, 200))
    offprint = max(2, int(offprint))
    #print(f"GA Params: pop_size={pop_size}, generations={generations}, offprint={offprint}")
    return pop_size, generations, offprint


def check_feasibility_without_start_depot(
     problem: Problem,
     solution: List[Tuple[int, float]],
) -> bool:
     """
     Checks if a solution is feasible:
     1. Each step must be between adjacent cities
     2. All gold from all cities must be collected (at least once)

     :param problem: Problem instance
     :param solution: List of (city, gold_picked)
     :return: True if feasible, False otherwise
     """
     graph = problem.graph
     gold_at = nx.get_node_attributes(graph, "gold")

     # Track collected gold per city
     gold_collected = {}
     prev_city = 0  # Start from depot

     current_weight = 0
     i=0

     for city, gold in solution:
         # Check adjacency
         if not graph.has_edge(prev_city, city):
             print(f"❌ ADIACENCY Feasibility failed: no edge between {prev_city} and {city} i={i}")
             print(f"Path segment: {prev_city} -> {city}")
             print( solution)
             return False

         # Track collected gold
         if gold > 0:
             gold_collected[city] = gold_collected.get(city, 0.0) + gold

         # Update current weight
         current_weight += gold
         if city == 0:
             current_weight = 0

         prev_city = city

     # Verify all gold was collected
     for city in graph.nodes():
         if city == 0:  # Depot has no gold
             continue
         expected_gold = gold_at.get(city, 0.0)
         collected_gold = gold_collected.get(city, 0.0)

         if abs(expected_gold - collected_gold) > 1e-4:  # Float tolerance
             print(f"❌ Feasibility failed: city {city} i={i} has{expected_gold:.2f} gold, collected {collected_gold:.2f}")
             return False
         i += 1

     return True


def compute_cost(problem, path):
    if not path:
        return 0.0
    alpha = problem.alpha
    beta  = problem.beta
    total = 0.0
    start, current_gold = path[0]
    g = problem.graph
    for city, gold in path[1:]:
        if not g.has_edge(start, city):
            return float("inf")
        d = g[start][city]['dist']
        total += d + (alpha * d * current_gold) ** beta
        current_gold = 0.0 if city == 0 else current_gold + gold
        start = city
    return total