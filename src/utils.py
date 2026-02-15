
import numpy as np

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


def check_path(problem, path):
    """Check if a solution is valid and compute its cost."""
    total_cost = 0
    current_weight = 0
    is_correct=True
    total_gold= sum(problem.graph.nodes[ n ]['gold'] for n in problem.graph.nodes if n !=0 )

    for i in range(len(path)-1):
        u, v = path[i][0], path[i+1][0]
        d= problem.graph[u][v]['dist']
        total_cost += d + (problem.alpha * d * current_weight) ** problem.beta
        # Aggiorna il peso se si raccoglie oro
        if v != 0:  # Non raccogliere oro al deposito
            current_weight += path[i+1][1]
            total_gold -= path[i+1][1]
                
        else:
            current_weight = 0  # Svuota il peso al deposito
    is_correct = abs(total_gold) < 1e-6  # Tolleranza per errori float
    
    return is_correct, total_cost

