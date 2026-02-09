
import numpy as np

def compute_ga_params(n_cities, beta, alpha):
    """Dynamically compute GA parameters based on problem complexity."""
    
    if n_cities <= 50:
        pop_size = min(50, n_cities)
        generations = 100
        offprint = int(pop_size * 0.6)
    
    elif n_cities <= 100:
        if beta > 2:
            pop_size = min(30, n_cities)
            generations = 50
            offprint = int(pop_size * 0.25)
        else:
            pop_size = min(100, n_cities)
            generations = 80 if beta >= 2 else 100
            offprint = int(pop_size * (0.3 if beta >= 2 else 0.5))
    
    elif n_cities <= 200:
        if beta > 2:
            pop_size = min(25, n_cities // 5)
            generations = 40
            offprint = max(3, int(pop_size * 0.2))
        else:
            pop_size = min(100, n_cities // 2)
            generations = 60 if beta >= 2 else 80
            offprint = int(pop_size * (0.25 if beta >= 2 else 0.4))
    
    else:  # n_cities > 200 (e.g., 1000)
        if beta > 2:
            # High beta (>2): VERY expensive initialization, minimal population
            base_pop = max(10, int(40 - 5 * (beta - 2)))  # Decreases with beta
            pop_size = max(10, int(base_pop / (1 + 0.05 * alpha)))
            generations = max(20, int(40 - 5 * beta))
            offprint = max(2, int(pop_size * 0.1))
        else:
            # Lower beta: can afford more exploration
            base_pop = min(80, int(200 / np.sqrt(n_cities / 100)))
            pop_size = min(base_pop + 20, int(80 - 10 * beta))
            generations = int(80 - 5 * beta)
            offprint = max(8, int(pop_size * 0.3))
    
    # Safety bounds
    pop_size = max(10, min(pop_size, 150))
    generations = max(20, min(generations, 200))
    offprint = max(2, int(offprint))
    
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

