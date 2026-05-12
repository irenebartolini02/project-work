from Problem import Problem
from src.GA_solver_2 import GA_Solver

p = Problem(num_cities=12, alpha=1.0, beta=1.5, density=0.7, seed=7)
solver = GA_Solver(p)
chromosome = solver.cities_to_visit[:]
genotype, reported_cost = solver.evaluate_and_segment(chromosome)
computed_cost = solver.compute_cost_genotype(genotype)
print('chromosome:', chromosome)
print('genotype:', genotype)
print('reported_cost:', reported_cost)
print('computed_cost:', computed_cost)

# Print per-gene compute breakdown
for i, gene in enumerate(genotype):
    cost = solver.compute_cost_genotype([gene])
    print(f'gene {i} cost by compute_cost_genotype: {cost}')

# Also compute phenotype and phenotype cost
phenotype = solver.genotype_to_phenotype(genotype)
print('phenotype length:', len(phenotype))
print('phenotype snippet:', phenotype[:20])
phen_cost = solver.compute_cost_phenotype(phenotype)
print('phenotype_cost:', phen_cost)

# Replicate evaluate_and_segment step-by-step to print contributions
def replicate_eval(chromosome, solver):
    current_node = chromosome[0]
    current_gold = solver.graph.nodes[current_node].get('gold', 0)
    total_cost = 0
    route = []
    if current_gold > 0:
        route.append((current_node, current_gold))

    print('\nReplicating evaluate_and_segment:')
    for next_target in chromosome[1:]:
        print('\nFrom', current_node, 'gold', current_gold, 'to', next_target)
        path_direct = solver.full_paths[current_node][next_target]
        cost_direct = 0
        tmp_node = current_node
        for c in path_direct[1:]:
            d = solver.graph[tmp_node][c]['dist']
            add = d + (solver.alpha * d * current_gold) ** solver.beta
            cost_direct += add
            print(' direct edge', tmp_node, '->', c, 'd', d, 'add', add)
            tmp_node = c

        path_to_depot = solver.full_paths[tmp_node][0]
        distance_from_depot = solver.dist_matrix[0][next_target]
        cost_unload = 0
        tmp_node2 = tmp_node
        for c in path_to_depot[1:]:
            d = solver.graph[tmp_node2][c]['dist']
            add = d + (solver.alpha * d * current_gold) ** solver.beta
            cost_unload += add
            print(' to depot edge', tmp_node2, '->', c, 'd', d, 'add', add)
            tmp_node2 = c
        cost_unload += distance_from_depot
        print(' cost_direct', cost_direct, 'cost_unload', cost_unload)

        if current_gold > 0 and cost_unload < cost_direct:
            print('Unload chosen')
            total_cost += cost_unload
            route = [(next_target, solver.graph.nodes[next_target].get('gold', 0))]
            current_gold = solver.graph.nodes[next_target].get('gold', 0)
        else:
            print('Continue chosen')
            total_cost += cost_direct
            g = solver.graph.nodes[next_target].get('gold', 0)
            route.append((next_target, g))
            current_gold += g
        current_node = next_target

    # return home
    path_home = solver.full_paths[current_node][0]
    for c in path_home[1:]:
        d = solver.graph[current_node][c]['dist']
        add = d + (solver.alpha * d * current_gold) ** solver.beta
        total_cost += add
        print(' home edge', current_node, '->', c, 'd', d, 'add', add)
        current_node = c

    print('Replicated total_cost', total_cost)

replicate_eval(chromosome, solver)

print('\nPer-edge phenotype contributions:')
start = phenotype[0][0]
current_gold = phenotype[0][1]
sum_check = 0
for city, gold in phenotype[1:]:
    d = solver.graph[start][city]['dist']
    add = d + (solver.alpha * d * current_gold) ** solver.beta
    print(f'{start}->{city} d={d:.6f} gold={current_gold:.6f} add={add:.6f}')
    sum_check += add
    if city == 0:
        current_gold = 0
    else:
        current_gold += gold
    start = city

print('Phenotype per-edge sum:', sum_check)
