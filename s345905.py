from Problem import Problem
from src.GA_solver import GA_Solver
from src.utils import compute_ga_params

def solution(p:Problem):

    n_cities = p.graph.number_of_nodes()
    population_size, generations, offprint = compute_ga_params(n_cities, p.beta, p.alpha)

    solver = GA_Solver(p, pop_size=population_size, generations=generations, offprint=offprint)
    # set fast to True for a quicker solution, it enable starvation control in GA
    best_path , best_cost = solver.solution(fast=False)

    return best_path[1:]  # Escludiamo il deposito (0) all'inizio come specificato dal professor Guseppe Esposito su Telegram