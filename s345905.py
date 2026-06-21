from Problem import Problem
from src.Solver import Solver


def solution(p:Problem):
    
    solver = Solver(p)
    # set fast to True for a quicker solution, it enable starvation control in GA
    best_path , best_cost = solver.solution(fast=False)

    return best_path[1:]  # Non si deve partire dal deposito (0) come specificato dal professor Guseppe Esposito su Telegram