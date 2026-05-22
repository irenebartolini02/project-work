import random
from matplotlib.pylab import beta
import numpy as np
import networkx as nx


class GA_Solver:

    def __init__(self, problem, pop_size=50, generations=100, offprint=20):
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

        self.pop_size    = pop_size
        self.generations = generations
        self.offprint    = offprint

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
    #  MULTIPLE-CYCLE OPTIMISER
    # ──────────────────────────────────────────────────────────────────────

    def _multiple_cycle(self, genotype: list) -> tuple:
        """Split tours into multiple lighter trips when beneficial."""
        new_genotype = []
        gc = self._gene_cost   # local alias

        for tour in genotype:
            cost     = gc(tour)
            min_gold = min(w for _, w in tour)
            best_factor = 1

            for i in range(2, int(min_gold) + 1):
                single_trip = [(c, w // i) for c, w in tour]
                c_single    = gc(single_trip)
                approx_cost = c_single * i

                if approx_cost < cost:
                    cost        = approx_cost
                    best_factor = i
                else:
                    break   # monotone

            if best_factor == 1:
                new_genotype.append(tour)
            else:
                for j in range(best_factor):
                    remainders = [0] * len(tour)
                    if j == best_factor - 1:
                        remainders = [w % best_factor for _, w in tour]
                    new_genotype.append(
                        [(c, w // best_factor + r)
                         for (c, w), r in zip(tour, remainders)]
                    )

        return new_genotype, self.compute_cost_genotype(new_genotype)
    
    # ──────────────────────────────────────────────────────────────────────
    #  GENE OPTIMIZER 
    # ──────────────────────────────────────────────────────────────────────

    # NOTA: testare ogni permutazione è troppo COSTOSO
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
        FARTHES INSERTION heuristic for TSP applied to a single gene.
        Costruisce un loop partendo dalle città più lontane.
        Ottimo per evitare incroci senza calcoli pesanti.
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
        # se una o più città compaiono in entrambi i geni, somma l'oro e mantieni una sola occorrenza
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
    
    
    def generate_initial_population(self) -> list:
        population = []
        ctv    = self.cities_to_visit
        mc     = self._multiple_cycle
        eas    = self.evaluate_and_segment
        use_mc = self.beta > 1.0

        # first chromosome is the greedy solution (sorted Neirest Neighbour based on distance from depot)

        # chromosome = self._greedy_solution()
        # genotype, cost = eas(chromosome)
        if self.prob.beta >= 0.5:
            # se beta < 0.5 partire dalla baseline è sconveniente
            greedy=1
        else:
            greedy=0

        if greedy:
            
            greedy_gen, greedy_cost= self._improved_baseline_individual()
            # validate = self.check_feasibility_genotype(greedy_gen)
            # if not validate:
            #     print(f"[INIT] Invalid greedy solution: {greedy_gen}")
            population.append((greedy_gen, greedy_cost))
        for _ in range(self.pop_size - greedy):
            chromosome = ctv[:]
            np.random.shuffle(chromosome)
            genotype, cost = eas(chromosome)
            if use_mc:
                genotype, cost = mc(genotype)
            population.append((genotype, cost))
            # validate = self.check_feasibility_genotype(genotype)
            # if not validate:
            #     print(f"[INIT] Invalid random solution: {genotype}")

        return population

    def _improved_baseline_individual(self):
        """Construct a baseline solution and iteratively merge tours if beneficial."""
        # 1. Generiamo l'ordine di visita iniziale (Nearest Neighbor semplice)
        cities_to_visit = list(self.cities_to_visit)
        # per risparmiare sull'andata del primo viaggio possiamo partire dalla città più lontana dal deposito, poi NN normale
        current_city = max(cities_to_visit, key=lambda c: self.dist_matrix[0][self.node_to_idx[c]])
        cities_to_visit.remove(current_city)
        ordered_targets = [current_city]
        while cities_to_visit:
            next_city = min(cities_to_visit, key=lambda c: self.dist_matrix[current_city][c])
            ordered_targets.append(next_city)
            cities_to_visit.remove(next_city)
            current_city = next_city

        # 2. Creiamo i tour iniziali: ogni città è un tour Deposito -> Target -> Deposito
        # Memorizziamo i tour COMPLETI (incluso lo zero iniziale e finale)
        genotype = []
    
        for target in ordered_targets:
            gold = self.graph.nodes[target]['gold']
            if gold <= 0: continue
            genotype.append([(target, gold)])
            


        # 3. Iterative Merge
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(genotype) - 1:
                gene_a = genotype[i]
                gene_b = genotype[i+1]
                
                # Calcoliamo i costi separati
                cost_a = self._gene_cost(gene_a)
                cost_b = self._gene_cost(gene_b)
                
                # Proviamo il merge
                # NOTA: Assicurati che _merge_two_tours accetti tour che iniziano/finiscono con (0,0)
                merged_gene, merged_cost = self.merge_genes(gene_a, gene_b)
                
                if merged_cost < (cost_a + cost_b):
                    genotype[i] = merged_gene
                    genotype.pop(i + 1)
                    changed = True
                    # Non incrementiamo i per ricontrollare il nuovo tour con il suo prossimo vicino
                else:
                    i += 1
        
        return genotype, self.compute_cost_genotype(genotype)
    
    def _chunked_star_routes(self) -> list:
        """When beta >= 1.5, the cost explodes with the weight carried. The baseline already does star routes (base -> city -> base for each city), but it picks up all the gold in one trip. The idea here is to split each city's gold into k smaller chunks and make k trips instead of one. """
        genotype = []
        # sort targets by distance from depot (nearest first, to maximize the benefit of chunking)
        nearest_first = sorted(self.cities_to_visit, key=lambda c: self.dist_matrix[0][self.node_to_idx[c]])
        
        for c in nearest_first:
           
            dist_base = self.dist_matrix[0][self.node_to_idx[c]]
            total_gold = self.graph.nodes[c]['gold']
            if total_gold <= 1e-6: continue

            # dynamic value of k (portion of the gold we take) to handle "big N" scenarios (N=1000)
            ops_budget = 10000 
            limit_k = max(5, int(ops_budget / len(self.cities_to_visit)))
            
            # k is the number of travel we do for each city, the higher the better. However, for big N we have
            # to be careful and not extend too much this value because of computational cost
            start_k = int(np.ceil(total_gold))
            start_k = min(start_k, limit_k)
            start_k = max(1, start_k)

            best_k = start_k
            best_val = float('inf')
            
            # test some values for k
            low_k = max(1, start_k - 5)
            high_k = min(start_k + 5, limit_k + 5)
            
            # simulate the cost for different k 
            for k in range(low_k, high_k + 1):
                chunk = total_gold / k

                cost_out = dist_base + (dist_base * self.alpha * 0)**self.beta
                cost_ret = dist_base + (dist_base * self.alpha * chunk)**self.beta
                total = k * (cost_out + cost_ret)
                
                if total < best_val:
                    best_val = total
                    best_k = k
            
            portion = total_gold / best_k
            remaining = total_gold
            
            # Use pre-computed base_paths directly (avoids repeated shortest_path calls)
       
            for _ in range(best_k):
                if remaining <= 1e-6: break
                take = min(portion, remaining)
                # Outbound: base -> city (gold only at destination)
                genotype.append([(c, take)])
                remaining -= take
                
        return genotype, self.compute_cost_genotype(genotype)
    # ──────────────────────────────────────────────────────────────────────
    #  GA OPERATORS
    # ──────────────────────────────────────────────────────────────────────

    def mutation(self, genotype: list) -> tuple:
        """
        Two modes:
          < 0.8 → split a random gene into two (if it has ≥ 2 cities)
          ≥ 0.8 → merge two random genes into one (if it doesn't exceed gold limits)
        """
        if not genotype:
            return genotype, self.compute_cost_genotype(genotype)
        
        # scegliamo dinamicamente se preferire spli o merge in base a beta


        ratio = random.random()

        if ratio <= self.mutation_threshold:
            # Split
            # selezionare solo i geni che hanno almeno 2 città (escludendo il depot) per evitare split non validi
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
        else:
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
        if self._beta > 1.0:
            genotype, cost = self._multiple_cycle(genotype)

        # validate = self.check_feasibility_genotype(genotype)
        # if not validate:
        #     print(f"[CROSSOVER] Invalid crossover: {parent1} + {parent2} -> {genotype}")

        return genotype, cost

        

    # ──────────────────────────────────────────────────────────────────────
    #  GA MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────

    def run_ga_logic(self, fast: bool = False) -> tuple:
        """
        GA loop with:
          - Population as (cost, genotype) tuples → sort on float, not list
          - Inline tournament selection (no lambda)
          - Elitism: best always survives
        """
        raw_pop = self.generate_initial_population()
        # (cost, genotype) so sort key is just a float
        pop: list = [(c, g) for g, c in raw_pop]
        pop.sort(key=lambda x: x[0])

        max_generation=0

        best_cost, best_chromo = pop[0]
        stagnation = 0
        half_off   = self.offprint // 2
        _mutation  = self.mutation
        _crossover = self.crossover

        for _gen in range(self.generations):
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
            pop = next_gen[: self.pop_size]

            if pop[0][0] < best_cost:
                best_cost, best_chromo = pop[0]
                max_generation=_gen
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= 10 and fast:
                break
        print(f"Max found Generation {max_generation}/{self.generations} ")
        return best_chromo, best_cost

    # ──────────────────────────────────────────────────────────────────────
    #  PUBLIC API  (signature unchanged)
    # ──────────────────────────────────────────────────────────────────────

    def solution(self, fast: bool = True):
        if self.alpha == 0 or self.beta == 0:
            # in questi due casi il costo è lineare, si basa solo sulla distanza 
            # conviene fare un unico giro  (only one gene)
            tour = [(city, self.node_gold.get(city, 0.0)) for city in self.cities_to_visit if self.node_gold.get(city, 0.0) > 0]
            best_path, best_cost = self.optimize_gene_suboptimal(tour)
            phenotype= self.genotype_to_phenotype([best_path])
            return  phenotype, best_cost
        if self.beta >= 1.5: 
            # in questo caso non ha senso unire i tour il costo di trasportare oro è troppo grande,
            genotype, best_cost = self._chunked_star_routes()
            phenotype= self.genotype_to_phenotype(genotype)
            return  phenotype, best_cost

        genotype, best_cost = self.run_ga_logic(fast=fast)
        phenotype = self.genotype_to_phenotype(genotype)
        return phenotype, best_cost
