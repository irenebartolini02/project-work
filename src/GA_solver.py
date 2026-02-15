from importlib.resources import path
from icecream import List, Tuple
import numpy as np
import networkx as nx
import random

class GA_Solver:

    def __init__(self, problem, pop_size=50, generations=100, offprint=20):
        """Initialize the GA solver with problem data and GA parameters."""
        self.prob = problem
        self.graph = problem.graph
        self.cities_to_visit = [n for n in self.graph.nodes if n != 0]
        # Inizializza i dizionari
        self.dist_matrix = {}
        self.all_paths = {}

        # nx.all_pairs_dijkstra restituisce un generatore di tuple: (sorgente, (distanze, percorsi))
        # for source, (distances, paths) in nx.all_pairs_dijkstra(self.graph, weight='dist'):
        #     self.dist_matrix[source] = distances
        #     self.all_paths[source] = paths

        # Lista dei nodi di interesse: deposito + città con oro
        relevant_nodes = [n for n in self.graph.nodes if n == 0 or self.graph.nodes[n].get('gold', 0) > 0]

        self.dist_matrix = {}
        self.all_paths = {}

        for source in relevant_nodes:
            # Calcola i percorsi solo partendo dai nodi rilevanti
            lengths, paths = nx.single_source_dijkstra(self.graph, source, weight='dist')
            
            # Filtra i risultati: tieni solo i percorsi verso gli altri nodi rilevanti
            self.dist_matrix[source] = {target: lengths[target] for target in relevant_nodes if target in lengths}
            self.all_paths[source] = {target: paths[target] for target in relevant_nodes if target in paths}
        
        # Ga parameters
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

            # controllo se esiste l'arco u, v
            if not self.graph.has_edge(u, v):
                print(f"Warning: No edge between {u} and {v}. Returning inf cost.")
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
        if self.prob.beta >=2:
            chromo, cost= self._multiple_cycle(chromo)
        is_valid= self.check_feasibility(chromo)
        if not is_valid:
            print("Warning: Greedy initialization produced an invalid solution.")
        #print(f"Greedy initialization | Cost: {cost:.2f} | Valid: {is_valid}")
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
          
          if not set_t:  # Tour vuoto
            continue
          
          best_factor=1
          min_gold= min(set_t, key=lambda x: x[1])[1]
          # Invece di testare TUTTI i valori per essere più efficienti:
          #for i in [2, 3, 5, int(min_gold//2), int(min_gold)]:  # Subset strategico
          for i in range(2, int (min_gold)+1 ):
              single_trip=[(0,0)]
              # calcola il consto di approssimato di 1 trip: i* costo_singolo_tour(prendendo w//i oro)
              single_trip.extend( [ (c, w//i ) for (c, w) in t ]) 
              single_trip.extend([(0,0)])

              cost_single_trip= self._path_cost( single_trip)
              approx_cost= cost_single_trip*i

              if cost_single_trip*i < cost and self._path_cost( single_trip)>0:
                  cost= approx_cost
                  best_factor=i
                  continue 
              else:
                  break # Dato che i cresce, se non migliora non migliorerà più
          
          # costruisco i trip effettivi con w//best_factor e r= w%best_factor
          trip= [(0,0)]            
          for j in range(best_factor):
            r= [ 0 for _ in t]
            if j== best_factor-1:
                r= [ w % best_factor for c, w in t]
            trip.extend( [ (c, w//best_factor + r) for (c, w), r in zip(t, r) ]) 
            trip.extend([(0,0)])
            
          trips.extend(trip[:-1])  
        
        trips.append((0,0))  # aggiungo il ritorno finale al deposito
        
        total_cost= self._path_cost( trips) 
        return trips, total_cost
    

    def _multiple_cycle_old (self, chromo: list[tuple[int, float]])-> tuple[list[tuple[int, float]], float]:
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
        is_valid= self.check_feasibility(path)
        if not is_valid:
            print(f"Warning: Baseline individual is not feasible! Cost: {cost:.2f}")
         
        return path, cost
    

    
    def _optimize_tour(self, tour: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """Re-decode a single tour to improve its visit order."""
        target_city = [(c, w) for c, w in tour if c != 0 and w > 0]
        
        target_city_dict = {}
        for c, w in target_city:
            if c in target_city_dict:
                target_city_dict[c] += w
            else:
                target_city_dict[c] = w

        target_city = [c for c, w in target_city_dict.items()]

        city_to_gold = {c: w for c, w in target_city_dict.items()}
        
        optimize_tour, _ = self._evaluate_and_segment(target_city)
        
        visited = set()
        new_tour = []
        for c, w in optimize_tour:
            real_w = city_to_gold.get(c, 0)
            if real_w and c not in visited:
                new_tour.append((c, real_w))
                visited.add(c)
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
        visited_targets = set()

        for c in merged_tour:
            if c in city_gold_map and c not in visited_targets:
                w = city_gold_map[c]
                visited_targets.add(c)
            else:
                w = 0
            merged_tour_with_weights.append((c,w))
        
        cost_merged = self._path_cost( merged_tour_with_weights)   
        
        return merged_tour_with_weights, cost_merged

    def _improved_baseline_individual(self):
        """Construct a baseline solution and iteratively merge tours if beneficial."""
        
        # 1. Generiamo l'ordine di visita iniziale (Nearest Neighbor semplice)
        cities_to_visit = list(self.cities_to_visit)
        current_city = 0
        ordered_targets = []
        while cities_to_visit:
            next_city = min(cities_to_visit, key=lambda c: self.dist_matrix[current_city][c])
            ordered_targets.append(next_city)
            cities_to_visit.remove(next_city)
            current_city = next_city

        # 2. Creiamo i tour iniziali: ogni città è un tour Deposito -> Target -> Deposito
        # Memorizziamo i tour COMPLETI (incluso lo zero iniziale e finale)
        tours = []
        for target in ordered_targets:
            gold = self.graph.nodes[target]['gold']
            if gold <= 0: continue
            
            # Costruiamo il tour usando i cammini precalcolati
            tour = []
            path_to = self.all_paths[0][target]
            path_back = self.all_paths[target][0]
            
            # Andata (tutti pesi 0)
            tour.extend([(c, 0) for c in path_to[:-1]])
            # Target (raccoglie oro)
            tour.append((target, gold))
            # Ritorno (tutti pesi 0)
            tour.extend([(c, 0) for c in path_back[1:]]) # Include lo 0 finale
            
            tours.append(tour)

        # 3. Iterative Merge
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(tours) - 1:
                tour_a = tours[i]
                tour_b = tours[i+1]
                
                # Calcoliamo i costi separati
                cost_a = self._path_cost(tour_a)
                cost_b = self._path_cost(tour_b)
                
                # Proviamo il merge
                # NOTA: Assicurati che _merge_two_tours accetti tour che iniziano/finiscono con (0,0)
                merged_tour, merged_cost = self._merge_two_tours(tour_a, tour_b)
                
                if merged_cost < (cost_a + cost_b):
                    tours[i] = merged_tour
                    tours.pop(i + 1)
                    changed = True
                    # Non incrementiamo i per ricontrollare il nuovo tour con il suo prossimo vicino
                else:
                    i += 1

        # 4. Assemble finale
        final_path = []
        for i, tour in enumerate(tours):
            if i == 0:
                final_path.extend(tour)
            else:
                # Evitiamo di duplicare lo zero: se il tour precedente finisce con 0 
                # e questo inizia con 0, saltiamo il primo elemento di questo tour
                final_path.extend(tour[1:])

        # Verifica finale
        is_valid = self.check_feasibility(final_path)
        final_cost = self._path_cost(final_path)
        
        if not is_valid:
            print(f"Warning: Improved baseline invalid! Cost: {final_cost}")
        return final_path, final_cost

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
                # Se la stessa città è presente in entrambi i tour, sommiamo l'oro totale
                if c in city_gold_map:
                    city_gold_map[c] += w
                else:
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
        
        merged_tour_with_weights = []
        visited_cities = set()

        # Assicuriamoci che l'oro venga assegnato solo alla PRIMA occorrenza 
        # della città target nel nuovo percorso TSP
        for c in merged_tour:
            if c in city_gold_map and c not in visited_cities:
                w = city_gold_map[c]
                visited_cities.add(c)
            else:
                w = 0
            merged_tour_with_weights.append((c, w)) 
            
        cost_merged = self._path_cost(merged_tour_with_weights)

        if cost_merged < cost_separate:
            tours[idx] = merged_tour_with_weights[1:-1]
            tours.pop(idx + 1)

        final_route = [(0, 0)]
        for tour in tours:
            final_route.extend(tour)
            final_route.append((0, 0))
        if not self.check_feasibility(final_route):
            print(f"Warning: Merged tour is not feasible | Cost separate: {cost_separate:.2f} | Cost merged: {cost_merged:.2f}")
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

            if self.prob.beta > 1.0:
                mutated_ind, cost= self._multiple_cycle(mutated_ind)
            else:
                cost= self._path_cost(mutated_ind)
            # is_valid= self.check_feasibility(mutated_ind)
            # if not is_valid:
            #     print(f"Warning: Mutation produced an invalid solution | New cost: {cost:.2f} | Valid: {is_valid}")
            return mutated_ind, cost

    
    def _crossover(self, parent1: list[tuple[int, float]], parent2: list[tuple[int, float]]) -> tuple[list[tuple[int, float]], float]:
        # 1. Definizione RIGIDA dei set di città
        # Prendiamo le prime k città uniche che appaiono in Parent 1 (che hanno oro)
        p1_assigned = set()
        for c, w in parent1:
            if c != 0 and self.graph.nodes[c]['gold'] > 0:
                p1_assigned.add(c)
                if len(p1_assigned) >= len(self.cities_to_visit) // 2:
                    break
        
        # 2. Registri per l'oro (per non superare il totale del nodo)
        gold_remaining = {c: self.graph.nodes[c]['gold'] for c in self.cities_to_visit}
        
        final_offspring = []
        
        # --- PARTE 1: TOUR DA PARENT 1 ---
        # Prendiamo solo i tour di P1 che servono le "sue" città
        current_tour = []
        for c, w in parent1:
            if c == 0:
                if current_tour:
                    # Controlliamo se questo tour ha raccolto oro assegnato a P1
                    # Se sì, lo aggiungiamo tutto
                    final_offspring.append((0, 0))
                    for tc, tw in current_tour:
                        if tc in p1_assigned:
                            amount = min(tw, gold_remaining[tc])
                            final_offspring.append((tc, amount))
                            gold_remaining[tc] -= amount
                        else:
                            final_offspring.append((tc, 0)) # Passa ma non raccoglie
                    current_tour = []
            else:
                current_tour.append((c, w))

        # --- PARTE 2: TOUR DA PARENT 2 ---
        # Prendiamo l'oro rimanente dalle città NON assegnate a P1
        current_tour = []
        for c, w in parent2:
            if c == 0:
                if current_tour:
                    # Usiamo la tua funzione optimize per pulire il tour di P2
                    # ma solo per l'oro che p1 non ha toccato
                    temp_tour = []
                    has_useful_gold = False
                    for tc, tw in current_tour:
                        if tc not in p1_assigned and gold_remaining[tc] > 0:
                            amount = min(tw, gold_remaining[tc])
                            temp_tour.append((tc, amount))
                            gold_remaining[tc] -= amount
                            has_useful_gold = True
                        else:
                            temp_tour.append((tc, 0))
                    
                    if has_useful_gold:
                        opt = self._optimize_tour([(0,0)] + temp_tour + [(0,0)])
                        if opt:
                            # Evitiamo doppi zeri
                            if final_offspring and final_offspring[-1] == (0,0):
                                final_offspring.extend(opt[1:])
                            else:
                                final_offspring.extend(opt)
                    current_tour = []
            else:
                current_tour.append((c, w))

        if final_offspring[-1] != (0,0):
            final_offspring.append((0,0))
        # is_valid= self.check_feasibility(final_offspring)
        # if not is_valid:
        #     print(f"Warning: Crossover produced an invalid solution | New cost: {self._path_cost(final_offspring):.2f} | Valid: {is_valid}")
        
        return final_offspring, self._path_cost(final_offspring)
            
    
    def _run_ga_logic(self, fast: bool = False )-> list[tuple[int, float]]:
        """Run the GA loop and return the best chromosome found."""
        
        # Per problemi grandi e beta>=2 , uso solo la greedy perchè multiple_cycle è molto costosa
        if self.prob.beta >= 2.0 and len(self.cities_to_visit) > 1000:
            solution, _ = self._greedy_initialization()
            return solution
        population = []

        if self.prob.beta > 0.5 and len(self.cities_to_visit) >  100:
            # se beta < 0.5 partire dalla baseline è sconveniente
            greedy=1
        else:
            greedy=0
       
        population = [self._greedy_initialization() for _ in range(self.pop_size-greedy)]
        # se beta < 1 troppo costosa
        if greedy: 
            population.append(self._improved_baseline_individual())  # baseline individual

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
    
        
    def check_feasibility(
        self,
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
        problem = self.prob
        graph = problem.graph
        gold_at = nx.get_node_attributes(graph, "gold")
        
        # Track collected gold per city
        gold_collected = {}
        prev_city = 0  # Start from depot
        
        current_weight = 0
        i=0
        
        for city, gold in solution[1:]:
            # Check adjacency
            if not graph.has_edge(prev_city, city):
                print(f"❌ Feasibility failed: no edge between {prev_city} and {city} i={i}")
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
                print(f"❌ Feasibility failed: city {city} i={i} has {expected_gold:.2f} gold, collected {collected_gold:.2f}")
                return False
            i += 1
        
        return True