from importlib.resources import path
import numpy as np
import networkx as nx
import random

class GA_Solver:

    def __init__(self, problem, pop_size=50, generations=100, offprint=20):
        """Initialize the GA solver with problem data and GA parameters."""
        self.prob = problem
        self.graph = problem.graph
        self.cities_to_visit = [n for n in self.graph.nodes if n != 0]
        self.all_paths = dict(nx.all_pairs_dijkstra_path(self.graph, weight='dist'))
        self.dist_matrix = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='dist'))
        self.pop_size = pop_size
        self.generations = generations
        self.offprint = offprint

    def _calc_path_cost_constat_weight(self, path : list[int], weight: float) -> float:
        """Calculate cost of a path with constant weight."""
        c = 0
        current_w = weight
        for u, v in zip(path, path[1:]):
            d = self.dist_matrix[u][v]
            c += d + (self.prob.alpha * d * current_w) ** self.prob.beta
        return c

    
    def _path_cost(self, path: list[tuple[int, float]]) -> float:
        """Calculate cost of a path with format [(city, weight), ...]."""
        total_cost = 0
        current_weight = 0
        for i in range(len(path)-1):
            u, v = path[i][0], path[i+1][0]
            d = self.graph[u][v]['dist']
            total_cost += d + (self.prob.alpha * d * current_weight) ** self.prob.beta
            if v==0:
                current_weight = 0
            else:
                current_weight += path[i+1][1]
        return total_cost  
    
    def _greedy_initialization(self)-> tuple[list[tuple[int, float]], float]:
        """Build an initial individual using a greedy strategy."""
        ind = list(self.cities_to_visit)
        random.shuffle(ind)
        chromo, cost= self._evaluate_and_segment(ind)
        if self.prob.beta >=1:
            chromo, cost= self._multiple_cycle(chromo)

        return chromo, cost
    
    def _multiple_cycle (self, chromo: list[tuple[int, float]])-> tuple[list[tuple[int, float]], float]:
        """Split tours into multiple lighter trips when beneficial."""
        tours = []
        current_tour_cities = []
        
        for n, w in chromo:
            if n == 0:
                if current_tour_cities:
                    tours.append(current_tour_cities)
                    current_tour_cities = []
            else:
                current_tour_cities.append((n,w))
        
        trips=[]

        for t in tours:
          closed_tour= [(0,0)]+ t+ [(0,0)]
          cost= self._path_cost( closed_tour)
          set_t= [ (c,w) for c,w in t if c!=0 and w!=0]
          
          best_trip=closed_tour

          for i in range(2, int (min(set_t, key=lambda x: x[1])[1] ) ):
              trip=[(0,0)]
              for j in range(i):
                r= [ 0 for _ in t]
                if j== i-1:
                    r= [ w % i for c, w in t]
                trip.extend( [ (c, w//i + r) for (c, w), r in zip(t, r) ]) 
                trip.extend([(0,0)])
              cost_trip= self._path_cost( trip)       
              if cost_trip < cost and cost_trip>0:
                  cost= cost_trip
                  best_trip= trip
              else:
                  break
          
          trips.extend(best_trip[:-1])  
        
        trips.append((0,0))  
        total_cost= self._path_cost( trips) 
        return trips, total_cost



    def _evaluate_and_segment(self, chromosome: list[int]) -> tuple[list[tuple[int, float]], float]:
        """Greedy decoder: decides whether to return to the depot to unload."""
        route = []
        total_cost = 0
        current_node = 0
        current_weight = 0

        route.append((0, 0)) 
        for next_target in chromosome:
            path_direct = self.all_paths[current_node][next_target]
            cost_direct = self._calc_path_cost_constat_weight(path_direct, current_weight)
            
            path_to_depot = self.all_paths[current_node][0]
            path_from_depot = self.all_paths[0][next_target]
            cost_unload = (self._calc_path_cost_constat_weight(path_to_depot, current_weight) + 
                           self._calc_path_cost_constat_weight(path_from_depot, 0))
            
            if current_weight > 0 and cost_unload < cost_direct:
                for v in path_to_depot[1:]:
                    route.append((v, 0))
                current_weight = 0
                for v in path_from_depot[1:]:
                    gold = self.graph.nodes[v].get('gold', 0) if v == next_target else 0
                    route.append((v, gold))
                    current_weight += gold
                total_cost += cost_unload
            else:
                for v in path_direct[1:]:
                    gold = self.graph.nodes[v].get('gold', 0) if v == next_target else 0
                    route.append((v, gold))
                    current_weight += gold
                total_cost += cost_direct
            current_node = next_target

        path_home = self.all_paths[current_node][0]
        total_cost += self._calc_path_cost_constat_weight(path_home, current_weight)
        for node in path_home[1:]:
            route.append((node, 0))
            
        return route, total_cost
    
    def _create_baseline_individual(self) -> tuple[list[tuple[int, float]], float]:
        """Create a baseline individual: return to depot after each target city."""
        """This function is used only for testing purposes, as the baseline solution is computed by the Problem class. - Not used in final solution """
        path= []
        for target_city in self.cities_to_visit:
            if self.graph.nodes[target_city]['gold'] == 0:
                continue
            andata= self.all_paths[0][target_city]
            path.extend([(c, 0) for c in andata[:-1]])
            w= self.graph.nodes[target_city]['gold']
            path.append((target_city, w))
            ritorno= self.all_paths[target_city][0]
            path.extend([(c, 0) for c in ritorno[1:-1]])
            
        path.extend([(0, 0)])
        cost= self._path_cost(path)   
        return path, cost
    
    def _create_nearest_neighbor_individual(self):
        """Create an individual using nearest-neighbor ordering and split decoding."""
        unvisited = set(self.cities_to_visit)
        current_node = 0
        sequence = []
        
        while unvisited:
            next_city = min(unvisited, key=lambda city: self.dist_matrix[current_node][city])
            sequence.append(next_city)
            unvisited.remove(next_city)
            current_node = next_city
            
        return self._dynamic_programming_split(sequence)
    

    def _dynamic_programming_split(self, chromosome: list[int]) -> tuple[list[tuple[int, float]], float]:
        """Dynamic programming split to minimize tour cost for a visit order."""
        n = len(chromosome)
        V = [float('inf')] * (n + 1)
        V[0] = 0
        p = [0] * (n + 1)

        nodes = [0] + chromosome 

        for i in range(1, n + 1):
            current_weight = 0
            current_dist_cost = 0
            
            for j in range(i - 1, -1, -1):
                segment_cost = self._calc_path_cost_constat_weight(self.all_paths[0][nodes[j+1]], 0)
                
                w = 0
                for k in range(j + 1, i):
                    w += self.graph.nodes[nodes[k]]['gold']
                    path = self.all_paths[nodes[k]][nodes[k+1]]
                    segment_cost += self._calc_path_cost_constat_weight(path, w)
                
                w += self.graph.nodes[nodes[i]]['gold']
                segment_cost += self._calc_path_cost_constat_weight(self.all_paths[nodes[i]][0], w)

                if V[j] + segment_cost < V[i]:
                    V[i] = V[j] + segment_cost
                    p[i] = j
                
                # if w > 10000: # early break 
                #     break

        final_route = [(0, 0)]
        curr = n
        segments = []
        while curr > 0:
            segments.append((p[curr], curr))
            curr = p[curr]
        segments.reverse()

        for start, end in segments:
            path_to_first = self.all_paths[0][nodes[start+1]]
            for v in path_to_first[1:]:
                gold = self.graph.nodes[v]['gold'] if v == nodes[start+1] else 0
                final_route.append((v, gold))
            
            for k in range(start + 1, end):
                path = self.all_paths[nodes[k]][nodes[k+1]]
                for v in path[1:]:
                    gold = self.graph.nodes[v]['gold'] if v == nodes[k+1] else 0
                    final_route.append((v, gold))
            
            path_home = self.all_paths[nodes[end]][0]
            for v in path_home[1:]:
                final_route.append((v, 0))

        return final_route, V[n]
    
    
    def _optimize_tour(self, tour: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """Re-decode a single tour to improve its visit order."""
        target_city = [(c, w) for c, w in tour if c != 0 and w > 0]
        
        city_to_gold = {c: self.graph.nodes[c]['gold'] for c, w in target_city}
        
        optimize_tour, _ = self._evaluate_and_segment([c for c, w in target_city])
        
        new_tour = []
        for c, w in optimize_tour:
            real_w = city_to_gold.get(c, 0)
            if real_w:
                new_tour.append((c, real_w))
            else:
                new_tour.append((c, 0))
        
        return new_tour
    
    
    def _merge_two_tours(self, tour_A, tour_B) -> tuple[list[tuple[int, float]], float]:
        """Build a merged tour visiting the union of two tours."""
        """Not used in the current solution """

        city_gold_map = {}
        for c, w in tour_A + tour_B:
            if w > 0:
                city_gold_map[c] = w
        
        if not city_gold_map:
            joined_tours= [(0,0)]+ tour_A+ [(0,0)]+ tour_B + [(0,0)]
            return joined_tours, self._path_cost(joined_tours)
        
        target_cities = list(city_gold_map.keys())
        
        merged_tour = []
        
        far_city = max(target_cities, key=lambda c: self.dist_matrix[0][c])
        
        current_node = far_city
        
        merged_tour.extend(self.all_paths[0][current_node][:-1])
        target_cities.remove(current_node)
         
        while target_cities:
            next_node = min(target_cities, key=lambda c: self.dist_matrix[current_node][c])
            merged_tour.extend(self.all_paths[current_node][next_node][:-1])
            target_cities.remove(next_node)
            current_node = next_node
        
        merged_tour.extend(self.all_paths[current_node][0])

        merged_tour_with_weights = [ ]
        for c in merged_tour:
            w= city_gold_map[c] if c in city_gold_map else 0
            merged_tour_with_weights.append((c,w))
        
        cost_merged = self._path_cost( merged_tour_with_weights)   
        
        return merged_tour_with_weights, cost_merged

    def _improved_baseline_individual(self):
        """Construct a baseline solution and iteratively merge tours if beneficial."""
        """Starting from the baseline we connect different tours, it is not used in the current solution because nearest neighbor performs better"""

        current_city=0
        cities_to_visit= self.cities_to_visit
        target_cities=[]
        while cities_to_visit:
            next_city = min(cities_to_visit, key=lambda c: self.dist_matrix[current_city][c])
            target_cities.append(next_city)
            cities_to_visit.remove(next_city)
            current_city = next_city
        
        path= []
        tours=[]
        for target_city in target_cities:
            tour=[]
            if self.graph.nodes[target_city]['gold'] == 0:
                continue
            andata= self.all_paths[0][target_city]
            tour.extend([(c, 0) for c in andata[:-1]])

            w= self.graph.nodes[target_city]['gold']
            tour.append((target_city, w))

            ritorno= self.all_paths[target_city][0]
            tour.extend([(c, 0) for c in ritorno[1:-1]])
            
            path.extend(tour)
            tours.append(tour[1:])
        
        path.extend([(0, 0)])
        cost= self._path_cost(path)  

        diff=1

        while diff > 0 :
            diff=0
            copy_tours= tours.copy()

            i=0
            while i <len(copy_tours)-1: 
                a=i
                b=i+1
                tour_A= copy_tours[a]
                tour_B= copy_tours[b]
                separate_cost= self._path_cost([(0,0)] +tour_A+ [(0,0)])+ self._path_cost([(0,0)]+tour_B+ [(0,0)])
                merged_tour, merged_cost= self._merge_two_tours(tour_A, tour_B)
                if merged_cost< separate_cost:
                    copy_tours[a]= merged_tour[1:-1]
                    copy_tours.pop(b)
                else: 
                    i+=1
                    
                tours= copy_tours
                new_route = [(0, 0)]
                for tour in tours:
                    new_route.extend(tour)
                    new_route.append((0, 0))

                new_cost= self._path_cost(new_route)
                diff= cost - new_cost
                cost= new_cost
                
        path = [(0, 0)]
        for tour in tours:
            path.extend(tour)
            path.append((0, 0))
        
        return path, self._path_cost(path)

    #------------------------------ GENETIC ALGORITHM OPERATORS----------------------------------#
    
    def _mutate_merge_tours(self, route: list[tuple[int, float]]) -> tuple[list[tuple[int, float]], float]:
        """Try merging two consecutive tours and keep the better solution."""
        tours = []
        current_tour = []
        for n, w in route:
            if n == 0:
                if current_tour:
                    tours.append(current_tour)
                    current_tour = []
            else:
    
                current_tour.append((n,w))
        
        if len(tours) < 2:
            return route, self._path_cost(route)

        idx = random.randint(0, len(tours) - 2)
        tour_a = tours[idx]
        tour_b = tours[idx+1]
        
        cost_separate = self._path_cost([(0, 0)] +  tour_a + [(0, 0)]) + \
                        self._path_cost([(0, 0)] + tour_b + [(0, 0)])
        
        city_gold_map = {}
        for c, w in tour_a + tour_b:
            if w > 0:
                city_gold_map[c] = w
        
        if not city_gold_map:
            return route, self._path_cost(route)
        
        target_cities = list(city_gold_map.keys())
        
        merged_tour = []
        
        far_city = max(target_cities, key=lambda c: self.dist_matrix[0][c])
        
        current_node = far_city
        
        merged_tour.extend(self.all_paths[0][current_node][:-1])
        target_cities.remove(current_node)
        
        while target_cities:
            next_node = min(target_cities, key=lambda c: self.dist_matrix[current_node][c])
            merged_tour.extend(self.all_paths[current_node][next_node][:-1])
            target_cities.remove(next_node)
            current_node = next_node
        
        merged_tour.extend(self.all_paths[current_node][0])
        
        merged_tour_with_weights = [ ]
        for c in merged_tour:
            w= city_gold_map[c] if c in city_gold_map else 0
            merged_tour_with_weights.append((c,w))
        cost_merged = self._path_cost( merged_tour_with_weights)   
        
        if cost_merged < cost_separate:
            tours[idx] = merged_tour_with_weights[1:-1]
            tours.pop(idx + 1)

        final_route = [(0, 0)]
        for tour in tours:
            final_route.extend(tour)
            final_route.append((0, 0))
        return final_route, self._path_cost(final_route)


    def _mutate(self, ind: list[tuple[int, float]] ) -> tuple[list[tuple[int, float]], float]:
        """Mutation with two options: merge tours or re-decode tours locally."""
        beta_factor = 0.5 * (1.0 - self.prob.beta)
        
        alpha_influence = 0.2 * (0.05 - self.prob.alpha) if 0.9 <= self.prob.beta <= 1.1 else 0
        
        threshold = 0.5 + beta_factor + alpha_influence
        threshold = max(0.1, min(0.9, threshold))

        ratio = random.random()

        if ratio < threshold:
            chromo, cost= self._mutate_merge_tours(ind)
            return chromo, cost
        
        else:
            tours = []
            current_tour_cities = []
                
            for n, w in ind:
                if n == 0:
                    if current_tour_cities:
                        tours.append(current_tour_cities)
                        current_tour_cities = []
                else:
                    current_tour_cities.append(n)
                        
            if current_tour_cities:
                tours.append(current_tour_cities)

            mutated_ind = []
            mutated_ind.append((0, 0))
            golden_city= set()
            for t in tours:
                t = [city for city in t if city not in golden_city]
                t_set = set(t)
                if len(t_set) > 0:
                    new_segment_route, _ = self._evaluate_and_segment(t_set)
                    mutated_ind.extend(new_segment_route[1:])
                    
                    golden_city.update(t_set)

            if self.prob.beta >= 1.0:
                mutated_ind, cost= self._multiple_cycle(mutated_ind)
            else:
                cost= self._path_cost(mutated_ind)
                
            return mutated_ind, cost

    
    def _crossover(self, parent1: list[tuple[int, float]], parent2: list[tuple[int, float]]) -> tuple[list[tuple[int, float]], float]:
        """Crossover: take tours from parent1 to cover first k cities, then from parent2."""
        
        k= len(self.cities_to_visit) // 2
        tours_p1= []
        collect_gold_in_p1 = {} 
        tour=[]
        
        for c, w in parent1[1:]: 
            if c==0:
                if tour:
                    tours_p1.extend([(0,0)]+tour)
                    tour=[]
                if len(collect_gold_in_p1) >= k and all(v == 0 for v in collect_gold_in_p1.values()):
                    break
            else:
                if c not in collect_gold_in_p1 and w>0 and len(collect_gold_in_p1)<k:
                    collect_gold_in_p1[c] = self.graph.nodes[c]['gold']-w 
                    tour.append((c,w)) 
                else:
                    if c in collect_gold_in_p1:
                        collect_gold_in_p1[c] = max(0, collect_gold_in_p1[c] - w)
                        tour.append((c,w))
                    else:
                        tour.append((c,0))
        tours_p2= []
        collect_gold_in_p2 = {c: self.graph.nodes[c]['gold'] 
                          for c in self.graph.nodes 
                          if c not in collect_gold_in_p1 and c != 0}
        tour=[]
        
        for c, w in parent2[1:]:
            if c==0:
                if tour:
                    tour = self._optimize_tour(tour) 
                    tours_p2.extend(tour[:-1])
                    tour=[]
                if all(v == 0 for v in collect_gold_in_p2.values()):
                    break
            else:
                if c in collect_gold_in_p2 and w>0 :
                    collect_gold_in_p2[c] = max(0, collect_gold_in_p2[c] - w)
                    tour.append((c,w))
                else:
                    tour.append((c,0))
        
        offspring = tours_p1 + tours_p2 + [(0,0)]  # aggiungo il ritorno finale al deposito
        cost = self._path_cost(offspring)

        return offspring, cost
    
        
    
    def _run_ga_logic(self, fast: bool = False )-> list[tuple[int, float]]:
        """Run the GA loop and return the best chromosome found."""
        population = [self._greedy_initialization() for _ in range(self.pop_size-1)]
        population.append(self._create_nearest_neighbor_individual())  
        population = sorted(population, key=lambda x: x[1])
        best_chromo = population[0][0]

        best_cost = population[0][1]
        stagnation_counter = 0

        for gen in range(self.generations):
            next_gen = []
            for _ in range(self.offprint//2):
                parents = []
                for _ in range(2):
                    candidates = random.sample(population, 2)
                    parents.append(min(candidates, key=lambda x: x[1])[0])

                ratio = random.random()
                if ratio < 0.8:
                    offspring1=self._mutate(parents[0])
                    offspring2=self._mutate(parents[1])
                else:
                    offspring1 = self._crossover(parents[0], parents[1])
                    offspring2 = self._crossover(parents[1], parents[0])
                
                next_gen.extend([offspring1, offspring2])

            
            next_gen = [(ind, cost) for ind, cost in sorted(next_gen, key=lambda x: x[1])]  
            population = next_gen[:self.pop_size]  # mantieni solo i migliori
        
            if population[0][1] < best_cost:
                best_chromo, best_cost = population[0]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            if stagnation_counter >= 10 and fast:
                break

        return best_chromo
        

    def solution(self, fast: bool = False):
        """Return the best chromosome and its cost."""
        best_chromo = self._run_ga_logic(fast=fast)
        return best_chromo, self._path_cost(best_chromo)