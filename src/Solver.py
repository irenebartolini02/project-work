from collections import defaultdict, deque
import random
from matplotlib.pylab import beta
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from src.utils import compute_ga_params

class Solver:

    def __init__(self, problem):
        self.prob = problem
        self.graph = problem.graph
        self.alpha = problem.alpha
        self.beta  = problem.beta

        # ── Relevant nodes (depot + gold cities), sorted ──────────────────
        self.relevant_nodes = sorted(
            n for n in self.graph.nodes
            if n == 0 or self.graph.nodes[n].get('gold', 0) > 0
        )
        self.node_to_idx    = {node: i for i, node in enumerate(self.relevant_nodes)}
        self.cities_to_visit = [n for n in self.relevant_nodes if n != 0]
        n_rel = len(self.relevant_nodes)

        # ── NumPy distance matrix (O(1) index lookup) ─────────────────────
        self.dist_matrix = np.zeros((n_rel, n_rel), dtype=np.float64)
        self.full_paths  = [[None] * n_rel for _ in range(n_rel)]
        self.node_gold   = {n: self.graph.nodes[n].get('gold', 0)
                            for n in self.graph.nodes}

        for i, source in enumerate(self.relevant_nodes):
            lengths, paths = nx.single_source_dijkstra(
                self.graph, source, weight='dist'
            )
            for j, target in enumerate(self.relevant_nodes):
                if target in lengths:
                    self.dist_matrix[i, j] = lengths[target]
                    self.full_paths[i][j]   = paths[target]

        # ── Flat edge-distance dict: (u,v) -> float  ──────────────────────
        # Eliminates repeated graph[u][v]['dist'] attribute lookups in hot loops.
        self._edge_dist: dict[tuple, float] = {}
        for u, v, data in self.graph.edges(data=True):
            d = data.get('dist', 0.0)
            self._edge_dist[(u, v)] = d
            self._edge_dist[(v, u)] = d

        # ── Pre-computed per-gene dist arrays ─────────────────────────────
        # For every (source_idx, target_idx) pair we cache the list of edge
        # distances along the shortest path. This avoids re-walking paths
        # and repeated dict lookups during the hot inner loop.
        self._path_dists: list[list] = [[None] * n_rel for _ in range(n_rel)]
        ed = self._edge_dist
        for i in range(n_rel):
            for j in range(n_rel):
                path = self.full_paths[i][j]
                if path is not None and len(path) >= 2:
                    self._path_dists[i][j] = [
                        ed[(path[k], path[k+1])] for k in range(len(path)-1)
                    ]
                else:
                    self._path_dists[i][j] = []

        # ── Local copies of hot scalars ────────────────────────────────────
        self._alpha = float(self.alpha)
        self._beta  = float(self.beta)


        beta_factor = 0.5 * (1.0 - self.prob.beta)
        alpha_influence = 0.2 * (0.05 - self.prob.alpha) if 0.9 <= self.prob.beta <= 1.1 else 0
        threshold = 0.5 + beta_factor + alpha_influence
        
        threshold = max(0.1, min(0.9, threshold))
        self.mutation_threshold = threshold

    # ──────────────────────────────────────────────────────────────────────
    #  CORE COST — pure Python, no per-call np.array allocation
    # ──────────────────────────────────────────────────────────────────────

    def _gene_cost(self, gene: list) -> float:
        """
        Cost of one gene (round-trip from/to depot).
        Uses pre-computed per-edge distance lists; pure Python arithmetic
        is faster than np.array construction for the typical gene length.
        """
        alpha   = self._alpha
        beta    = self._beta
        pd      = self._path_dists
        n2i     = self.node_to_idx
        gold    = 0.0
        total   = 0.0
        prev    = 0   # depot

        for city, gold_amount in gene:
            ci  = n2i[city]
            for d in pd[n2i[prev]][ci]:
                total += d + (alpha * d * gold) ** beta
            gold += gold_amount
            prev  = city

        # Return leg
        for d in pd[n2i[prev]][0]:
            total += d + (alpha * d * gold) ** beta

        return total

    def compute_cost_genotype(self, genotype: list) -> float:
        """
        Total cost of a genotype.
        Subtracts the free depot→first_city leg (matches original semantics).
        """
        if not genotype:
            return 0.0

        gc    = self._gene_cost
        total = 0.0

        for gene in genotype:
            total += gc(gene)

        first_city = genotype[0][0][0]
        start_cost = float(self.dist_matrix[0, self.node_to_idx[first_city]])
        return total - start_cost

    # ──────────────────────────────────────────────────────────────────────
    #  FEASIBILITY
    # ──────────────────────────────────────────────────────────────────────

    def check_feasibility_genotype(self, genotype: list) -> bool:
        gold_collected: dict[int, float] = {}
        for gene in genotype:
            start = 0
            for city, gold in gene:
                if self.full_paths[start][city] is None:
                    print(f"[FAIL] no path {start} -> {city}")
                    return False
                if gold > 0:
                    gold_collected[city] = gold_collected.get(city, 0.0) + gold
                start = city

        gold_at = nx.get_node_attributes(self.graph, "gold")
        for city in self.graph.nodes():
            if city == 0:
                continue
            if abs(gold_at.get(city, 0.0) - gold_collected.get(city, 0.0)) > 1e-4:
                print(f"[FAIL] city {city}: expected {gold_at.get(city,0):.2f}, "
                      f"got {gold_collected.get(city,0):.2f}")
                return False
        return True

    def check_feasibility_phenotype(self, phenotype: list) -> bool:
        if not phenotype:
            return False
        gold_at          = nx.get_node_attributes(self.graph, "gold")
        gold_collected: dict[int, float] = {}
        start, init_gold = phenotype[0]
        if init_gold > 0:
            gold_collected[start] = init_gold

        for city, gold in phenotype[1:]:
            if not self.graph.has_edge(start, city):
                print(f"[FAIL] no edge {start} -> {city}")
                return False
            if gold > 0:
                gold_collected[city] = gold_collected.get(city, 0.0) + gold
            start = city

        for city in self.graph.nodes():
            if city == 0:
                continue
            if abs(gold_at.get(city, 0.0) - gold_collected.get(city, 0.0)) > 1e-4:
                print(f"[FAIL] city {city}: expected {gold_at.get(city,0):.2f}, "
                      f"got {gold_collected.get(city,0):.2f}")
                return False
        return True

    # ──────────────────────────────────────────────────────────────────────
    #  REPRESENTATION CONVERSION
    # ──────────────────────────────────────────────────────────────────────

    def genotype_to_phenotype(self, genotype: list) -> list:
        phenotype = []
        if not genotype:
            return phenotype

        n2i = self.node_to_idx
        fp  = self.full_paths

        for gene in genotype:
            start = 0
            for city, gold in gene:
                for c in fp[n2i[start]][n2i[city]][1:-1]:
                    phenotype.append((c, 0))
                phenotype.append((city, gold))
                start = city
            for c in fp[n2i[start]][0][1:]:
                phenotype.append((c, 0))

        first_city = genotype[0][0]          # (city, gold) tuple
        idx = phenotype.index(first_city)
        return phenotype[idx:]

    def compute_cost_phenotype(self, phenotype: list) -> float:
        if not phenotype:
            return 0.0
        alpha = self._alpha
        beta  = self._beta
        total = 0.0
        start, current_gold = phenotype[0]
        g = self.graph
        for city, gold in phenotype[1:]:
            if not g.has_edge(start, city):
                return float("inf")
            d = g[start][city]['dist']
            total += d + (alpha * d * current_gold) ** beta
            current_gold = 0.0 if city == 0 else current_gold + gold
            start = city
        return total

    # ──────────────────────────────────────────────────────────────────────
    #  GREEDY DECODER
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_and_segment(self, chromosome: list) -> tuple:
        """
        Greedy decoder: for each city decides whether detour via depot is cheaper.
        Uses pre-computed _path_dists to avoid repeated dict lookups.
        """
        chromosome = [c for c in chromosome if c != 0]
        if not chromosome:
            return [], 0.0

        pd    = self._path_dists
        n2i   = self.node_to_idx
        ng    = self.node_gold
        dm    = self.dist_matrix
        alpha = self._alpha
        beta  = self._beta

        def _walk_cost(path_ds: list, carry: float) -> float:
            t = 0.0
            for d in path_ds:
                t += d + (alpha * d * carry) ** beta
            return t

        current_node  = chromosome[0]
        ni_cur        = n2i[current_node]
        current_gold  = ng.get(current_node, 0.0)
        route         = [(current_node, current_gold)]
        genotype      = []
        total_cost    = 0.0

        for next_target in chromosome[1:]:
            ni_next = n2i[next_target]

            # Direct cost
            cost_direct = _walk_cost(pd[ni_cur][ni_next], current_gold)

            # Unload cost (weighted leg home + free dist to next)
            if current_gold > 0:
                cost_unload = (_walk_cost(pd[ni_cur][0], current_gold)
                               + float(dm[0, ni_next]))
                do_unload = cost_unload < cost_direct
            else:
                cost_unload = 0.0
                do_unload   = False

            if do_unload:
                genotype.append(route)
                g = ng.get(next_target, 0.0)
                current_gold = g
                route = [(next_target, g)]
                total_cost += cost_unload
            else:
                g = ng.get(next_target, 0.0)
                route.append((next_target, g))
                total_cost += cost_direct
                current_gold += g

            current_node = next_target
            ni_cur       = ni_next

        # Final return leg
        total_cost += _walk_cost(pd[ni_cur][0], current_gold)
        genotype.append(route)
        return genotype, total_cost

    
    # ──────────────────────────────────────────────────────────────────────
    #  GENE OPTIMIZER 
    # ──────────────────────────────────────────────────────────────────────

    # NOTE: testing every permutation is too expensive.
    def optimize_gene_optimal(self, gene: list) -> tuple:
        """
        Optimize a single gene by trying all permutations of its cities.
        Returns the best gene and its cost.
        """
        from itertools import permutations

        best_gene = gene
        gc= self._gene_cost
        best_cost = gc(gene)

        cities = [c for c, _ in gene]
        golds  = [g for _, g in gene]

        for perm in permutations(range(len(gene))):
            permuted_gene = [(cities[i], golds[i]) for i in perm]
            cost = gc(permuted_gene)
            if cost < best_cost:
                best_cost = cost
                best_gene = permuted_gene

        return best_gene, best_cost

    def optimize_gene_suboptimal(self, gene):
        """
        Farthest-insertion heuristic for TSP applied to a single gene.
        It builds a loop starting from the farthest cities.
        It is good at avoiding crossings without heavy computation.
        """
        if not gene:
            return [], 0.0

        if len(gene) < 2:
            return list(gene), self._gene_cost(gene)

        # Preserve the exact city->gold assignment and only reorder cities.
        gold_map = {city: gold for city, gold in gene}
        cities = list(gold_map.keys())
        unvisited = set(cities)

        # Start from the farthest city from the depot.
        first_city = max(unvisited, key=lambda city: self.dist_matrix[0][self.node_to_idx[city]])
        tour = [first_city]
        unvisited.remove(first_city)

        while unvisited:
            # Pick the city farthest from the current partial tour.
            next_city = max(
                unvisited,
                key=lambda city: min(
                    self.dist_matrix[self.node_to_idx[city]][self.node_to_idx[tour_city]]
                    for tour_city in tour
                ),
            )

            # Insert it in the position that minimizes the actual gene cost.
            best_cost = float('inf')
            best_tour = None

            for insert_pos in range(len(tour) + 1):
                candidate_order = tour[:insert_pos] + [next_city] + tour[insert_pos:]
                candidate_gene = [(city, gold_map[city]) for city in candidate_order]
                candidate_cost = self._gene_cost(candidate_gene)
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_tour = candidate_order

            tour = best_tour
            unvisited.remove(next_city)

        optimized_gene = [(city, gold_map[city]) for city in tour]
        return optimized_gene, self._gene_cost(optimized_gene)
    
    
    
    def optimize_gene(self, gene):
        """
        Wrapper for gene optimization. Chooses between optimal and suboptimal based on gene length.
        """
        if len(gene) <= 6:  # Threshold can be tuned based on performance tests
            return self.optimize_gene_optimal(gene)
        else:
            return self.optimize_gene_suboptimal(gene)
        
    def merge_genes(self, gene1: list, gene2: list) -> tuple:
        """
        Merge two genes into one by concatenating their cities and summing gold.
        Returns the merged gene and its cost.
        """
        # If one or more cities appear in both genes, sum the gold and keep only one occurrence.
        g1_cities = [ c for c, g in gene1 ]
        g2_cities = [ c for c, g in gene2 ]

        intersection_cities= set(g1_cities) & set(g2_cities)
        if intersection_cities:
            merged_dict = {}
            for c, g in gene1 + gene2:
                merged_dict[c] = merged_dict.get(c, 0) + g
            merged_gene = list(merged_dict.items())
        else:
            merged_gene = gene1 + gene2
        
        
        merged_gene, cost = self.optimize_gene(merged_gene) # Optimize the merged gene
        
        return merged_gene, cost
    
    def split_gene(self, gene: list, split_index: int) -> tuple:
        """
        Split a gene into two at the given index.
        Returns the two new genes and their costs.
        """
        gc= self._gene_cost
        og= self.optimize_gene
        gene1 = gene[:split_index]
        gene2 = gene[split_index:]

        gene1, cost1 = og(gene1) # Optimize gene1
        gene2, cost2 = og(gene2) # Optimize gene2

        cost1 = gc(gene1)
        cost2 = gc(gene2)

        return (gene1, cost1), (gene2, cost2)


    # ──────────────────────────────────────────────────────────────────────
    #  INITIAL POPULATION
    # ──────────────────────────────────────────────────────────────────────
    
    # not used 
    def _shortest_path(self) -> list:
        """
        When the cost depends only on the distance and not on the gold carried, we can use the precomputed shortest paths.
        """
        cities = [city for city in self.cities_to_visit if self.node_gold.get(city, 0.0) > 0]
        # Order the cities by increasing distance from the previous one.

        start= 0
        next_city= min(cities, key=lambda city: self.dist_matrix[self.node_to_idx[start]][self.node_to_idx[city]])

        cities.remove(next_city)
        genotype= [(next_city, self.node_gold[next_city])]

        while cities:
            current_city = next_city
            next_city = min(cities, key=lambda c: self.dist_matrix[current_city][c])
            genotype.append((next_city, self.node_gold[next_city]))
            cities.remove(next_city)

        return [genotype], self._gene_cost(genotype)

    # not used 
    def _greedy_solution(self) -> list:
        """
        Neirest Neighbour greedy solution based on distance from previous city."""
        if not self.cities_to_visit:
            return []

        unvisited = set(self.cities_to_visit)
        tour = []
        current = 0  # start at depot

        while unvisited:
            next_city = min(
                unvisited,
                key=lambda city: self.dist_matrix[self.node_to_idx[current]][self.node_to_idx[city]]
            )
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        return tour
    
    
    def generate_initial_population(self, pop_size: int) -> list:
        population = []
        ctv    = self.cities_to_visit
        eas    = self.evaluate_and_segment
    

        greedy = 0
        if self.prob.beta >= 0.5:
            # If beta < 0.5, starting from the baseline is not convenient.
            greedy+=1      
            greedy_gen, greedy_cost= self._improved_baseline_individual()
            population.append((greedy_gen, greedy_cost))
            
        for _ in range(pop_size - greedy):
            chromosome = ctv[:]
            np.random.shuffle(chromosome)
            genotype, cost = eas(chromosome)
            population.append((genotype, cost))

        return population

    def _improved_baseline_individual(self):
        """Construct a baseline solution and iteratively merge tours when beneficial."""
        # 1. Build the initial visit order (simple nearest-neighbor).
        cities_to_visit = list(self.cities_to_visit)
        # To save on the outbound leg of the first trip, start from the city farthest from the depot, then use normal NN.
        current_city = max(cities_to_visit, key=lambda c: self.dist_matrix[0][self.node_to_idx[c]])
        cities_to_visit.remove(current_city)
        ordered_targets = [current_city]
        while cities_to_visit:
            next_city = min(cities_to_visit, key=lambda c: self.dist_matrix[current_city][c])
            ordered_targets.append(next_city)
            cities_to_visit.remove(next_city)
            current_city = next_city

        # 2. Create the initial tours: each city is a Depot -> Target -> Depot trip.
        # Store the COMPLETE tours (including the initial and final zero).
        genotype = []
    
        for target in ordered_targets:
            gold = self.graph.nodes[target]['gold']
            if gold <= 0: continue
            genotype.append([(target, gold)])
            


        # 3. Iterative merge.
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(genotype) - 1:
                gene_a = genotype[i]
                gene_b = genotype[i+1]
                
                # Compute the separate costs.
                cost_a = self._gene_cost(gene_a)
                cost_b = self._gene_cost(gene_b)
                
                # Try the merge.
                # NOTE: Make sure _merge_two_tours accepts tours that start/end with (0,0).
                merged_gene, merged_cost = self.merge_genes(gene_a, gene_b)
                
                if merged_cost < (cost_a + cost_b):
                    genotype[i] = merged_gene
                    genotype.pop(i + 1)
                    changed = True
                    # Do not increment i so we can re-check the new tour against its next neighbor.
                else:
                    i += 1
        
        return genotype, self.compute_cost_genotype(genotype)
    

#### ──────────────────────────────────────────────────────────────────────
    #  SPLIT ADAPTIVE (beta > 1.0)
#### ______________________________________________________________________

    def _refine_trip_with_weighted_path(self, city, K_fixed, gold_per_visit):
        """
        Recompute the return path (city -> depot) using weighted Dijkstra
        on the actual carried gold. One call per city.
        Args: 
        - city: the city to start from
        - K_fixed: the fixed number of visits (trips) for this city
        - gold_per_visit: the amount of gold carried on each visit (total_gold / K_fixed)
        Returns:
        - length (cost): the total return length (city -> depot) considering the weight
        - path: the actual return path from city to depot optimized for the weight
        """
        def weight_func(u, v, data):
            d = data['dist']
            return d + (self.alpha * d * gold_per_visit) ** self.beta

        length, path = nx.single_source_dijkstra(
            self.graph, city, target=0, weight=weight_func
        )
        return length, path
    
    def _adaptive_split_with_refinement(self, city, total_gold, max_search=1000):
        """
        Adaptive optimal split with refinement using a weighted path for return trips.
        It returns the phenotype gene for depot -> city -> depot
        Args:
        - city: the city target (depot -> city -> depot)   
        - total_gold: the total gold in that city
        - max_search: the maximum number of trips to consider in binary search
        Returns:
        - best_K: the optimal number of trips (K) found
        - best_cost: the total cost for the best_K trips
        - best_path: the optimized path for the return trip (city -> depot) considering the gold weight     
        """
        # Phase 1: cheap binary search with a fixed geometric path (the current one).
        K_fixed = self._binary_search_K(city, total_gold, max_search)  # already implemented

        # Phase 2: refinement with a weighted path, in a small neighborhood of K_fixed.
        candidates = {max(1, int(K_fixed * 0.5)), K_fixed, int(K_fixed * 1.5)}
        best_K, best_cost, best_path = None, float('inf'), None

        for K in candidates:
            gold_per_visit = total_gold / K
            length, path = self._refine_trip_with_weighted_path(city, K, gold_per_visit)
            total_trip_cost = K * (self.dist_matrix[0][city] + length)  # light outbound leg + weighted return
            if total_trip_cost < best_cost:
                best_cost, best_K, best_path = total_trip_cost, K, path
        
        return best_K, best_cost, best_path
    
    def _binary_search_K(self, city, total_gold, max_search=1000):
        if total_gold == 0:
            return 0
            
        # Helper to calculate the cost for K trips
        def get_total_strategy_cost(k):
            gold_per_visit = total_gold / k
            temp_trip = [(city, gold_per_visit)] # Single trip for this city with gold split into K parts
            single_trip_cost = self._gene_cost(temp_trip)
            return k * single_trip_cost, temp_trip

        # Binary Search for Optimal K
        low = 1
        high = max_search
        
        best_k = 1
        best_trip_obj = None
        min_cost = float('inf')

        while low < high:
            mid = (low + high) // 2
            
            # 1. Capture BOTH trip objects
            cost_mid, trip_mid = get_total_strategy_cost(mid)
            cost_next, trip_next = get_total_strategy_cost(mid + 1) # <--- Capture trip_next
            
            # 2. Update Best if 'mid' is better
            if cost_mid < min_cost:
                min_cost = cost_mid
                best_k = mid
                best_trip_obj = trip_mid
            
            # 3. Update Best if 'mid+1' is better
            if cost_next < min_cost: 
                min_cost = cost_next
                best_k = mid + 1
                best_trip_obj = trip_next
            
            # Binary Search Direction
            if cost_mid < cost_next:
                high = mid
            else:
                low = mid + 1
        
        # Final cleanup (check if 'low' is better than what we found during search)
        final_k = low
        final_trip=[(city, total_gold / final_k) ] 
        final_cost = self.compute_cost_genotype([final_trip])
        
        if final_cost < min_cost:
            winner_trip = final_trip
            winner_k = final_k
        else:
            if best_trip_obj is None:
                winner_trip = final_trip
                winner_k = final_k
            else:
                winner_trip = best_trip_obj
                winner_k = best_k

        return winner_k

    def _generate_solution_with_adaptive_split(self, max_search=1000):
        """
        Generate a solution using the adaptive split with refinement for each city.
        Returns the full phenotype and its cost.

        """
        phenotype = []
        
        for city in self.cities_to_visit:
            total_gold = self.graph.nodes[city]['gold']
            if total_gold == 0: continue

            # Outbound path
            initial_path = self.full_paths[self.node_to_idx[0]][self.node_to_idx[city]]  # depot -> city
            
            best_K, best_cost, best_path = self._adaptive_split_with_refinement(city, total_gold, max_search)
           # print(f"[DEBUG] City {city}: total_gold={total_gold}, best_K={best_K}, best_cost={best_cost:.2f}, best_return_path={best_path}")
            for _ in range(best_K):
                if phenotype:
                    # Connect the current depot to the current city without duplicating depot/city.
                    phenotype.extend((c, 0) for c in initial_path[1:-1])
                phenotype.append((city, total_gold / best_K))
                phenotype.extend([(c, 0) for c in best_path[1:]])  # Add the optimized return path, skipping the city itself.
                
            #print(f"[DEBUG] Phenotype after city {city}: {phenotype}")
        return phenotype, self.compute_cost_phenotype(phenotype)

   
    # ──────────────────────────────────────────────────────────────────────
    #  GA OPERATORS
    # ──────────────────────────────────────────────────────────────────────
    def merge_mutation(self, genotype):
        """Merge two random genes into one (if it doesn't exceed gold limits)."""
        # Merge
        if len(genotype) < 2:
            #print(f"[MUTATION] not enough genes to merge: {genotype}")
            return genotype, self.compute_cost_genotype(genotype)
        idx1, idx2 = random.sample(range(len(genotype)), 2)
        g1, g2 = genotype[idx1], genotype[idx2]
        merged_gene, merged_cost = self.merge_genes(g1, g2)
        new_genotype = [g for i, g in enumerate(genotype) if i not in (idx1, idx2)] + [merged_gene]
        new_cost = self.compute_cost_genotype(new_genotype)
        # validate = self.check_feasibility_genotype(new_genotype)
        # if not validate:
        #     print(f"[MUTATION] Invalid merge: {g1} + {g2} -> {merged_gene}")
        return new_genotype, new_cost
    
    def split_mutation (self, genotype):
        valid_indices = [i for i, g in enumerate(genotype) if len(g) >= 2]
        if not valid_indices:
            #print(f"[MUTATION] Baseline every gene is lenght 1:")
            return genotype, self.compute_cost_genotype(genotype)
        gene_idx = random.choice(valid_indices)
        gene = genotype[gene_idx]
        (g1, c1), (g2, c2) = self.split_gene(gene, len(gene) // 2)
        new_genotype = genotype[:gene_idx] + [g1, g2] + genotype[gene_idx+1:]
        new_cost = self.compute_cost_genotype(new_genotype)
        # validate = self.check_feasibility_genotype(new_genotype)
        # if not validate:
        #     print(f"[MUTATION] Invalid split: {gene} -> {g1} + {g2}")
        return new_genotype, new_cost
    
    def mutation(self, genotype: list) -> tuple:
        """
                Two modes:
                    < 0.8 -> split a random gene into two (if it has >= 2 cities)
                    >= 0.8 -> merge two random genes into one (if it does not exceed gold limits)
        """
        if not genotype:
            return genotype, self.compute_cost_genotype(genotype)
        
        # Dynamically decide whether to prefer split or merge based on beta.


        ratio = random.random()

        if ratio <= self.mutation_threshold:
            # Split.
            new_genotype, new_cost= self.split_mutation(genotype)
            return new_genotype, new_cost
        else:
            new_genotype, new_cost= self.merge_mutation(genotype)
            return new_genotype, new_cost


    def crossover(self, parent1: list, parent2: list) -> tuple:
        """
        Order crossover: take first n cities from parent1, rest from parent2,
        re-decode with evaluate_and_segment to guarantee feasibility.
        """
        cities_p1 = [city for gene in parent1 for city, _ in gene]
        cities_p2 = [city for gene in parent2 for city, _ in gene]

        n          = np.random.randint(1, max(2, len(self.cities_to_visit)))
        chromosome = []
        seen: set  = set()

        for city in cities_p1[:n]:
            if city not in seen:
                chromosome.append(city)
                seen.add(city)

        ctv_set = set(self.cities_to_visit)
        for city in cities_p2:
            if city not in seen and city in ctv_set:
                chromosome.append(city)
                seen.add(city)

        for city in self.cities_to_visit:
            if city not in seen:
                chromosome.append(city)
                seen.add(city)

        genotype, cost = self.evaluate_and_segment(chromosome)

        # validate = self.check_feasibility_genotype(genotype)
        # if not validate:
        #     print(f"[CROSSOVER] Invalid crossover: {parent1} + {parent2} -> {genotype}")

        return genotype, cost

        

    # ──────────────────────────────────────────────────────────────────────
    #  GA MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────

    def run_ga_logic(self, pop_size: int, generations: int, off_size: int, fast: bool = False) -> tuple:
        """
        GA loop with:
          - Population as (cost, genotype) tuples → sort on float, not list
          - Inline tournament selection (no lambda)
          - Elitism: best always survives
        """
        raw_pop = self.generate_initial_population( pop_size)
        # (cost, genotype) so sort key is just a float
        pop: list = [(c, g) for g, c in raw_pop]
        pop.sort(key=lambda x: x[0])

        max_generation=0

        best_cost, best_chromo = pop[0]
        stagnation = 0
        half_off   = off_size // 2
        _mutation  = self.mutation
        _crossover = self.crossover

        for _gen in range(generations):
            next_gen = []

            for _ in range(half_off):
                # Tournament selection (inline)
                c1a, c1b = random.sample(pop, 2)
                p1 = c1a[1] if c1a[0] <= c1b[0] else c1b[1]
                c2a, c2b = random.sample(pop, 2)
                p2 = c2a[1] if c2a[0] <= c2b[0] else c2b[1]

                if random.random() <= 0.8:
                    o1, cost1 = _mutation(p1)
                    o2, cost2 = _mutation(p2)
                else:
                    o1, cost1 = _crossover(p1, p2)
                    o2, cost2 = _crossover(p2, p1)

                next_gen.append((cost1, o1))
                next_gen.append((cost2, o2))

            # Elitism
            next_gen.append(pop[0])
            next_gen.sort(key=lambda x: x[0])
            pop = next_gen[: pop_size]

            if pop[0][0] < best_cost:
                best_cost, best_chromo = pop[0]
                max_generation=_gen
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= 10 and fast:
                break
        #print(f"Max found Generation {max_generation}/{generations} ")
        return best_chromo, best_cost

    # ──────────────────────────────────────────────────────────────────────
    #  PUBLIC API  (signature unchanged)
    # ──────────────────────────────────────────────────────────────────────

    def solution(self, fast: bool = True):
        if self.alpha == 0 or self.beta == 0:
            # In these two cases the cost is linear and depends only on distance.
            # It is convenient to perform a single tour (only one gene).
            tour = [(city, self.node_gold.get(city, 0.0)) for city in self.cities_to_visit if self.node_gold.get(city, 0.0) > 0]
            best_path, best_cost = self.optimize_gene(tour)
            phenotype= self.genotype_to_phenotype([best_path])
            return  phenotype, best_cost
        
        if self.beta > 1.0: 
            phenotype, best_cost = self._generate_solution_with_adaptive_split(max_search=1000)
            return  phenotype, best_cost
        
        n_cities = len(self.cities_to_visit)
        pop, gen, off= compute_ga_params(n_cities=n_cities, beta=self.beta, alpha=self.alpha)
        genotype, best_cost = self.run_ga_logic(pop, gen, off, fast=fast)
        phenotype = self.genotype_to_phenotype(genotype)
        return phenotype, best_cost
    

    ##____________________________________________________________________________________

        # NOT USED IN THE FINAL SOLUTION - left over from earlier implementations/ideas.
    ##____________________________________________________________________________________
    # import numpy as np
    # from scipy.spatial import KDTree
    
    # # Not used.
    # def _multiple_cycle(self, genotype: list, max_search=1000) -> tuple:
    #     """
    #     Split each gene into K trips, carrying a fraction of the gold for each
    #     city at every pass. This version handles both integer and FLOAT gold
    #     values correctly, which is necessary because input genes may already
    #     come from a previous split, for example from generate_adaptive_split
    #     or from a prior call to this same function.

    #     Fixes compared to the previous version:

    #     1. K_max is bounded by min_gold (fixes genotype explosion)
    #     Without this bound, a gene with already small gold, such as 1.0 after
    #     a previous split, would be fragmented again up to max_search=1000
    #     trips with gold < 1, causing the genotype to explode in size
    #     (observed: 9021 -> 28126 elements between two calls).
    #     Below 1 unit of gold per trip, further splitting no longer makes
    #     physical sense: the natural limit is min_gold_int = floor(min(gold
    #     in the gene)).

    #     2. Exact balanced distribution via telescoping sum
    #     The previous version used divmod(w, K) assuming integer w.
    #     With FLOAT w (for example 777.63), divmod produces a float remainder r
    #     (for example 2.63): the condition "j < r" with integer j counts the
    #     trips incorrectly and the fractional part of r is lost, breaking gold
    #     conservation (observed bug: "city 1 expected 777.63, got 800.00").
    #     The telescoping sum always guarantees an exact total, for both integer
    #     and float gold:
    #         cumulative_target += w / K
    #         part = cumulative_target - cumulative_assigned
    #     because it telescopes: the sum of all `part` values is mathematically
    #     identical to w, with no residual rounding error.
    #     """
    #     beta = self._beta
    #     gc   = self._gene_cost

    #     # ── Early exit: per beta <= 1 K=1 e' sempre ottimale ─────────────
    #     if beta <= 1.0:
    #         return genotype, self.compute_cost_genotype(genotype)

    #     def make_trips_balanced_exact(tour, K):
    #         """Distribute each city in the gene across K trips with an exact sum."""
    #         trips = [[] for _ in range(K)]
    #         for c, w in tour:
    #             cumulative_target = 0.0
    #             cumulative_assigned = 0.0
    #             for j in range(K):
    #                 cumulative_target += w / K
    #                 part = cumulative_target - cumulative_assigned
    #                 trips[j].append((c, part))
    #                 cumulative_assigned += part
    #         return trips

    #     new_genotype: list = []

    #     for tour in genotype:
    #         if not tour:
    #             continue

    #         # ── Physical limit: K cannot exceed the minimum gold in the gene ───
    #         # Below 1 unit of gold per trip, splitting no longer makes sense
    #         # (and for already fractional gold coming from previous calls, it
    #         # avoids endlessly fragmenting an already minimal trip).
    #         min_gold = min(w for _, w in tour)
    #         if min_gold <= 1.0:
    #             new_genotype.append(tour)
    #             continue

    #         K_max = min(max_search, int(min_gold))
    #         if K_max <= 1:
    #             new_genotype.append(tour)
    #             continue

    #         # ── f(K) with float gold: perfectly convex for beta > 1 ──
    #         def f(K: int) -> float:
    #             return K * gc([(c, w / K) for c, w in tour])

    #         # ── Binary search for the minimum on [1, K_max] ────────────────────
    #         low, high = 1, K_max
    #         while low < high:
    #             mid = (low + high) // 2
    #             if f(mid) <= f(mid + 1):
    #                 high = mid
    #             else:
    #                 low = mid + 1
    #         best_K = low

    #         # Check neighbors for robustness on plateaus.
    #         best_cost_f = f(best_K)
    #         for K_cand in (best_K - 1, best_K + 1):
    #             if 1 <= K_cand <= K_max:
    #                 c = f(K_cand)
    #                 if c < best_cost_f:
    #                     best_cost_f = c
    #                     best_K = K_cand

    #         # ── Build the actual K* trips with an exact sum ─────────────
    #         if best_K == 1:
    #             new_genotype.append(tour)
    #         else:
    #             new_genotype.extend(make_trips_balanced_exact(tour, best_K))

    #     return new_genotype, self.compute_cost_genotype(new_genotype)

    # # Not used.
    # def merge_all_possible(self, genotype, max_neighbors=5, k_search=50):
    #     """
    #     O(n log n) version per iteration, instead of O(n^2 * k_search).

    #     Bottlenecks fixed compared to the previous version:

    #     1. gene_sets.index(set_b)  ->  O(n) per chiamata, chiamata O(n*k) volte
    #     FIX: dizionario set_to_indices[frozenset] = deque di indici,
    #     lookup e pop in O(1).

    #     2. The break after a SINGLE merge forced a full restart of the while
    #     loop (rebuilding gene_sets, centroids, KDTree to apply one change).
    #     Fix: apply ALL merge candidates found in a single pass over the
    #     unique sets, then rebuild only if merges were actually performed.

    #     3. idxs_a/idxs_b were recomputed with an O(n) scan for each merge
    #     found. Fix: keep them as pre-indexed deques, consumed with popleft()
    #     in O(1).

    #     4. new_genotype was reconstructed by scanning the entire genotype at
    #     each accepted merge. Fix: build it once per pass, iterating over the
    #     collected "keep" indices.

    #     Complexity per pass: O(n log n + n_unique * k_search)
    #     instead of O(n^2 * k_search) in the worst case of the previous version.
    #     The number of while passes is typically O(log n) because each pass
    #     applies ALL possible merges at once, not one at a time.
    #     """
    #     from scipy.spatial import KDTree
    #     import numpy as np
    #     import networkx as nx

    #     if not hasattr(self, '_node_positions'):
    #         self._node_positions = nx.get_node_attributes(self.graph, 'pos')

    #     def get_gene_centroid(gene):
    #         coords = [self._node_positions[c] for c, _ in gene if c in self._node_positions]
    #         return np.mean(coords, axis=0) if coords else np.array([0.5, 0.5])

    #     genotype = list(genotype)  # copia di lavoro

    #     while True:
    #         n = len(genotype)
    #         if n <= 1:
    #             break

    #         # ── 1. Indicizzazione O(n): frozenset -> deque di indici ─────────
    #         gene_sets = [frozenset(c for c, _ in g) for g in genotype]
    #         set_to_indices: dict = defaultdict(deque)
    #         for idx, gs in enumerate(gene_sets):
    #             set_to_indices[gs].append(idx)

    #         unique_sets = list(set_to_indices.keys())
    #         if len(unique_sets) <= 1:
    #             break

    #         # One representative gene for each unique set (for centroid and cost).
    #         rep_gene = {gs: genotype[idxs[0]] for gs, idxs in set_to_indices.items()}

    #         # ── 2. Centroidi + KDTree SOLO sui set unici, UNA volta per passata ─
    #         centroids = np.array([get_gene_centroid(rep_gene[gs]) for gs in unique_sets])
    #         tree = KDTree(centroids)
    #         set_to_pos = {gs: i for i, gs in enumerate(unique_sets)}

    #         any_merge_this_pass = False
    #         keep_indices: list = []          # indici originali da mantenere intatti
    #         merged_genes: list = []          # nuovi geni prodotti da merge

    #         # ── 3. Scan only the UNIQUE sets (not all n genes!) ────
    #         for gs in unique_sets:
    #             idxs = set_to_indices[gs]
    #             if not idxs:
    #                 continue  # already fully consumed by a previous merge.

    #             pos_a = set_to_pos[gs]
    #             k_eff = min(k_search, len(unique_sets))
    #             _, neighbor_positions = tree.query(centroids[pos_a], k=k_eff)
    #             if np.isscalar(neighbor_positions):
    #                 neighbor_positions = [neighbor_positions]

    #             gene_a = rep_gene[gs]
    #             tested = 0
    #             merged_here = False

    #             for pos_b in neighbor_positions:
    #                 set_b = unique_sets[pos_b]
    #                 if set_b == gs or not gs.isdisjoint(set_b):
    #                     continue
    #                 idxs_b = set_to_indices[set_b]
    #                 if not idxs_b:
    #                     continue  # already consumed by a previous merge in this pass.

    #                 gene_b = rep_gene[set_b]
    #                 cost_a = self._gene_cost(gene_a)
    #                 cost_b = self._gene_cost(gene_b)
    #                 merged_gene, merged_cost = self.merge_genes(gene_a, gene_b)

    #                 if merged_cost <= cost_a + cost_b:
    #                     # ── Applica il merge a TUTTE le coppie disponibili ────
    #                     quanti = min(len(idxs), len(idxs_b))
    #                     for _ in range(quanti):
    #                         idxs.popleft()
    #                         idxs_b.popleft()
    #                         merged_genes.append(merged_gene)
    #                     any_merge_this_pass = True
    #                     merged_here = True
    #                     break  # passa al prossimo set unico

    #                 tested += 1
    #                 if tested >= max_neighbors:
    #                     break

    #             if not merged_here:
    #                 # Nessun merge trovato per questo set: le sue copie restano intatte
    #                 keep_indices.extend(idxs)
    #                 idxs.clear()

    #         # ── 4. Aggiungi le copie residue di set parzialmente fusi ─────────
    #         # (es. set_a aveva 5 copie, solo 3 fuse: le restanti 2 vanno mantenute)
    #         for gs in unique_sets:
    #             keep_indices.extend(set_to_indices[gs])

    #         # ── 5. Costruisci il nuovo genotype in un'unica passata O(n) ─────
    #         keep_indices.sort()
    #         new_genotype = [genotype[i] for i in keep_indices] + merged_genes
    #         genotype = new_genotype

    #         if not any_merge_this_pass:
    #             break

    #     return genotype, self.compute_cost_genotype(genotype)

    # # Not used.
    # def hill_climber_optimize(self, genotype, max_iterations=1000):
    #     """
    #     Apply mutation to the genotype to optimize it, keeping only improvements.
    #     """

    #     def merge_matching_copies(current_genotype, set_a, set_b):
    #         gene_sets = [frozenset(c for c, _ in g) for g in current_genotype]

    #         if set_a == set_b:
    #             matching_indices = [idx for idx, gs in enumerate(gene_sets) if gs == set_a]
    #             merged_for_idx = {}
    #             skip_indices = set()

    #             for left_idx, right_idx in zip(matching_indices[0::2], matching_indices[1::2]):
    #                 merged_gene, _ = self.merge_genes(current_genotype[left_idx], current_genotype[right_idx])
    #                 merged_for_idx[left_idx] = merged_gene
    #                 skip_indices.add(right_idx)

    #             if not merged_for_idx:
    #                 return current_genotype

    #             rebuilt = []
    #             for idx in range(len(current_genotype)):
    #                 if idx in merged_for_idx:
    #                     rebuilt.append(merged_for_idx[idx])
    #                 elif idx in skip_indices:
    #                     continue
    #                 else:
    #                     rebuilt.append(current_genotype[idx])

    #             return rebuilt

    #         idxs_a = [idx for idx, gs in enumerate(gene_sets) if gs == set_a]
    #         idxs_b = [idx for idx, gs in enumerate(gene_sets) if gs == set_b]

    #         quanti_merge = min(len(idxs_a), len(idxs_b))
    #         if quanti_merge == 0:
    #             return current_genotype

    #         merged_for_a = {}
    #         skip_indices = set()

    #         for k in range(quanti_merge):
    #             a_idx = idxs_a[k]
    #             b_idx = idxs_b[k]
    #             merged_gene, _ = self.merge_genes(current_genotype[a_idx], current_genotype[b_idx])
    #             merged_for_a[a_idx] = merged_gene
    #             skip_indices.add(b_idx)

    #         rebuilt = []
    #         for idx in range(len(current_genotype)):
    #             if idx in merged_for_a:
    #                 rebuilt.append(merged_for_a[idx])
    #             elif idx in skip_indices:
    #                 continue
    #             else:
    #                 rebuilt.append(current_genotype[idx])

    #         return rebuilt
        
    #     iterations = 0
    #     cost= self.compute_cost_genotype(genotype)
    #     while iterations < max_iterations:
            
    #         # Select two random genes and try to merge them; if the cost improves, accept the change.
    #         gene1, gene2 = random.sample(genotype, 2)
    #         #print(f"[HILL_CLIMBER] Iteration {iterations} - Testing merge between gene {gene1} and gene {gene2}")
    #         cost1 = self._gene_cost(gene1)
    #         cost2 = self._gene_cost(gene2)
    #         set1 = frozenset(c for c, _ in gene1)
    #         set2 = frozenset(c for c, _ in gene2)
    #         merged_gene, merged_cost = self.merge_genes(gene1, gene2)
    #         if merged_cost <= cost1+cost2:
    #             #print(f"[HILL_CLIMBER] Merged gene {gene1} and {gene2} into {merged_gene} with cost {merged_cost:.2f} (improvement from {cost1+cost2:.2f})")
    #             genotype.remove(gene1)
    #             genotype.remove(gene2)
    #             genotype.append(merged_gene)
    #             genotype[:] = merge_matching_copies(genotype, set1, set2)
    #             new_cost = self.compute_cost_genotype(genotype)
    #             cost = new_cost
    #             # In addition to merging the two selected genes, also try to merge all identical copies of gene1 and gene2 to maximize the improvement.
    #             # This is important when there are many identical copies of gene1 and gene2.

            
    #         iterations += 1
    #     return genotype, self.compute_cost_genotype(genotype)

    # # not used
    # def hill_climber_classic(self, genotype, max_iterations=1000):
    #     """
    #     Hill climber performs split-gene or merge-gene moves to improve the solution.
    #     """
    #     cost= self.compute_cost_genotype(genotype)
    #     for i in range(max_iterations):

    #         new_genotype, new_cost = self.mutation(genotype)
    #         if new_cost<cost:
    #             print(f"\n[DEBUG] find convenient mutation iteration {i}")
    #             genotype=new_genotype
    #             cost= new_cost

    #     return genotype, cost
