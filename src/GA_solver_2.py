from collections import defaultdict, deque
import random
from matplotlib.pylab import beta
import numpy as np
import networkx as nx
from scipy.spatial import KDTree


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

    # def _multiple_cycle(self, genotype: list) -> tuple:
    #     """Split tours into multiple lighter trips when beneficial."""
    #     new_genotype = []
    #     gc = self._gene_cost   # local alias

    #     for tour in genotype:
    #         cost     = gc(tour)
    #         min_gold = min(w for _, w in tour)
    #         best_factor = 1

    #         for i in range(2, int(min_gold) + 1):
    #             single_trip = [(c, w // i) for c, w in tour]
    #             c_single    = gc(single_trip)
    #             approx_cost = c_single * i

    #             if approx_cost < cost:
    #                 cost        = approx_cost
    #                 best_factor = i
    #             else:
    #                 break   # monotone

    #         if best_factor == 1:
    #             new_genotype.append(tour)
    #         else:
    #             for j in range(best_factor):
    #                 remainders = [0] * len(tour)
    #                 if j == best_factor - 1:
    #                     remainders = [w % best_factor for _, w in tour]
    #                 new_genotype.append(
    #                     [(c, w // best_factor + r)
    #                      for (c, w), r in zip(tour, remainders)]
    #                 )

    #     return new_genotype, self.compute_cost_genotype(new_genotype)
        

    def _multiple_cycle(self, genotype: list, max_search=1000) -> tuple:
        """
        Divide ogni gene in K trip portando una frazione di gold per citta'
        ad ogni passaggio. Versione corretta per gold sia intero che FLOAT
        (necessario perche' i geni in input possono provenire da una divisione
        precedente, es. da generate_adaptive_split o da una chiamata pregressa
        di questa stessa funzione).

        FIX rispetto alla versione precedente:

        1. K_max vincolato a min_gold (bug dell'esplosione del genotype)
        Senza questo vincolo, un gene con gold gia' piccolo (es. 1.0, frutto
        di una divisione precedente) veniva ulteriormente frammentato fino
        a max_search=1000 trip da gold<1, facendo esplodere la dimensione
        del genotype (osservato: 9021 -> 28126 elementi tra due chiamate).
        Sotto 1 unita' di oro per trip non ha senso fisico continuare a
        dividere: il limite naturale e' min_gold_int = floor(min(gold nel gene)).

        2. Distribuzione bilanciata ESATTA via somma telescopica
        La versione precedente usava divmod(w, K) assumendo w intero.
        Con w FLOAT (es. 777.63), divmod produce un resto r anch'esso float
        (es. 2.63): la condizione "j < r" con j intero conta male i trip e
        la parte frazionaria di r viene persa, rompendo la conservazione
        del gold (bug osservato: "city 1 expected 777.63, got 800.00").
        La somma telescopica garantisce SEMPRE somma esatta, sia per gold
        intero che float:
            cumulative_target += w / K
            part = cumulative_target - cumulative_assigned
        perche' e' una somma a telescopio: la somma di tutte le `part` e'
        matematicamente identica a w, senza arrotondamenti residui.
        """
        beta = self._beta
        gc   = self._gene_cost

        # ── Early exit: per beta <= 1 K=1 e' sempre ottimale ─────────────
        if beta <= 1.0:
            return genotype, self.compute_cost_genotype(genotype)

        def make_trips_balanced_exact(tour, K):
            """Distribuisce ogni citta' del gene in K trip con somma esatta."""
            trips = [[] for _ in range(K)]
            for c, w in tour:
                cumulative_target = 0.0
                cumulative_assigned = 0.0
                for j in range(K):
                    cumulative_target += w / K
                    part = cumulative_target - cumulative_assigned
                    trips[j].append((c, part))
                    cumulative_assigned += part
            return trips

        new_genotype: list = []

        for tour in genotype:
            if not tour:
                continue

            # ── Limite fisico: K non può superare il gold minimo nel gene ───
            # Sotto 1 unità di oro per trip la divisione non ha più senso
            # (e su gold già frazionario da chiamate precedenti, evita di
            # rifrantumare all'infinito un trip già minimale).
            min_gold = min(w for _, w in tour)
            if min_gold <= 1.0:
                new_genotype.append(tour)
                continue

            K_max = min(max_search, int(min_gold))
            if K_max <= 1:
                new_genotype.append(tour)
                continue

            # ── f(K) con gold float: perfettamente convessa per beta > 1 ──
            def f(K: int) -> float:
                return K * gc([(c, w / K) for c, w in tour])

            # ── Binary search del minimo su [1, K_max] ────────────────────
            low, high = 1, K_max
            while low < high:
                mid = (low + high) // 2
                if f(mid) <= f(mid + 1):
                    high = mid
                else:
                    low = mid + 1
            best_K = low

            # Controlla i vicini per robustezza su plateau
            best_cost_f = f(best_K)
            for K_cand in (best_K - 1, best_K + 1):
                if 1 <= K_cand <= K_max:
                    c = f(K_cand)
                    if c < best_cost_f:
                        best_cost_f = c
                        best_K = K_cand

            # ── Costruzione dei K* trip reali con somma esatta ─────────────
            if best_K == 1:
                new_genotype.append(tour)
            else:
                new_genotype.extend(make_trips_balanced_exact(tour, best_K))

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
    def _shortest_path(self) -> list:
        """
        When the cost depends only on the distance and not on the gold carried, we can use the precomputed shortest paths.
        """
        cities = [city for city in self.cities_to_visit if self.node_gold.get(city, 0.0) > 0]
        # ordinare le città in ordine crescente si distanza dalla precedente 

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
        greedy = 0
        if self.prob.beta >= 0.5:
            # se beta < 0.5 partire dalla baseline è sconveniente
            greedy+=1      
            greedy_gen, greedy_cost= self._improved_baseline_individual()
            population.append((greedy_gen, greedy_cost))
            if self.prob.beta > 1:
                greedy+=1
                chunked_gen, chunked_cost = self.generate_adaptive_split(max_search=1000)
                population.append((chunked_gen, chunked_cost))

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
    
    # def merge_all_possible(self, genotype):
    #     """
    #     Merge all the genes in the genotype iteratively until no further merge is beneficial.
    #     Checks all possible pairs (all-to-all), not just adjacent ones.
    #     """
    #     changed = True
    #     def gene_contains_gene(g1, g2):
    #         """Check if gene g1 contains any city from gene g2."""
    #         # se l'intersezione delle città è non vuota, consideriamo i geni come "condividenti" e non li uniamo (già ottimizzati)
    #         return any(c == city for c, _ in g1 for city, _ in g2)
    #     while changed:
    #         changed = False
    #         n = len(genotype)
            
    #         # Doppio ciclo per testare ogni possibile coppia (i, j) con i < j
    #         for i in range(n):
    #             for j in range(i + 1, n):
    #                 gene_a = genotype[i]
    #                 gene_b = genotype[j]
    #                 if gene_contains_gene(gene_a, gene_b):
    #                     continue  

    #                 cost_a = self._gene_cost(gene_a)
    #                 cost_b = self._gene_cost(gene_b)
                    
    #                 # Proviamo il merge
    #                 merged_gene, merged_cost = self.merge_genes(gene_a, gene_b)
                    
    #                 # Se il costo del gene unito è strettamente migliore della somma dei due separati
    #                 if merged_cost < (cost_a + cost_b):
    #                     # 1. Sostituiamo il gene i con quello unito
    #                     genotype[i] = merged_gene
    #                     # 2. Rimuoviamo il gene j (che era più avanti nella lista)
    #                     genotype.pop(j)
                        
    #                     # Segnaliamo il cambiamento per far ripartire il ciclo while principale
    #                     changed = True
    #                     break # Rompe il ciclo interno (j)
                        
    #             if changed:
    #                 break # Rompe il ciclo esterno (i) per ricominciare da capo con la nuova lista
                    
    #     return genotype, self.compute_cost_genotype(genotype)
    import numpy as np
    from scipy.spatial import KDTree


    # def merge_all_possible(self, genotype, k_search=50, max_neighbors=5):
    #     """
    #     Versione ultra-ottimizzata.
    #     Quando trova un merge conveniente tra due famiglie di geni, unisce 
    #     istantaneamente tutte le loro copie identiche per evitare calcoli ridondanti.
    #     """
        
    #     if not hasattr(self, '_node_positions'):
    #         self._node_positions = nx.get_node_attributes(self.graph, 'pos')

    #     def get_gene_centroid(gene):
    #         coords = []
    #         for city, _ in gene:
    #             if city in self._node_positions:
    #                 coords.append(self._node_positions[city])
    #         if not coords:
    #             return np.array([0.5, 0.5])
    #         return np.mean(coords, axis=0)

    #     changed = True
    #     while changed:
    #         changed = False
    #         n = len(genotype)
    #         if n <= 1:
    #             break

    #         # 1. Mappiamo ogni gene al suo frozenset
    #         gene_sets = [frozenset(c for c, _ in g) for g in genotype]

    #         # Estraiamo i geni UNICI
    #         unique_genes = []
    #         seen_sets = set()
    #         frozenset_to_unique_idx = dict()

    #         for idx, g_set in enumerate(gene_sets):
    #             if g_set not in seen_sets:
    #                 seen_sets.add(g_set)
    #                 unique_genes.append(genotype[idx])
    #                 frozenset_to_unique_idx[g_set] = len(unique_genes) - 1

    #         # 2. Calcola i centroidi solo per gli unici
    #         centroids = np.array([get_gene_centroid(g) for g in unique_genes])
    #         tree = KDTree(centroids)
            
    #         for i in range(n):
    #             # Siccome modifichiamo il genotipo dentro il ciclo, verifichiamo di non essere fuori indice
    #             if i >= len(genotype):
    #                 break
                    
    #             gene_a = genotype[i]
    #             set_a = gene_sets[i]
    #             unique_index = frozenset_to_unique_idx[set_a]
                
    #             k_search = min(k_search, len(unique_genes)) 
    #             distances, indices = tree.query(centroids[unique_index], k=k_search)
                
    #             if np.isscalar(indices):
    #                 indices = [indices]
                    
    #             tested_neighbors = 0
    #             for u_j in indices:
    #                 gene_b_candidate = unique_genes[u_j]
    #                 set_b = frozenset(c for c, _ in gene_b_candidate)
                    
    #                 if set_a == set_b: #or not set_a.isdisjoint(set_b):
    #                     continue 

    #                 if not set_a.isdisjoint(set_b):
    #                     continue

    #                 #print(f"[DEBUG] Testing merge between gene {i} (set {set_a}) and unique gene {u_j} (set {set_b})")

    #                 try:
    #                     j = gene_sets.index(set_b)
    #                 except ValueError:
    #                     continue

    #                 gene_b = genotype[j]
                    
    #                 cost_a = self._gene_cost(gene_a)
    #                 cost_b = self._gene_cost(gene_b)
                    
    #                 # Prova il merge
    #                 merged_gene, merged_cost = self.merge_genes(gene_a, gene_b)
                    
    #                 if merged_cost <= (cost_a + cost_b):
    #                     print(f"[MERGE] Merging gene {i} (set {set_a}) with gene {j} (set {set_b}) "
    #                           f"reduces cost from {cost_a + cost_b:.2f} to {merged_cost:.2f}")
    #                     # ── APPLICAZIONE DEL MERGE DI MASSA (corretto) ─────────────
    #                     # Invece di replicare lo stesso merged_gene per tutte le coppie
    #                     # dobbiamo fondere le copie effettive a coppie per preservare
    #                     # la quantità d'oro (ogni copia può avere gold diverso).
    #                     idxs_a = [idx for idx, gs in enumerate(gene_sets) if gs == set_a]
    #                     idxs_b = [idx for idx, gs in enumerate(gene_sets) if gs == set_b]

    #                     quanti_merge = min(len(idxs_a), len(idxs_b))

    #                     # Costruiamo i merged reali per ogni coppia (idxs_a[k], idxs_b[k])
    #                     merged_for_a = {}
    #                     skip_indices = set()
    #                     for k in range(quanti_merge):
    #                         a_idx = idxs_a[k]
    #                         b_idx = idxs_b[k]
    #                         m_gene, m_cost = self.merge_genes(genotype[a_idx], genotype[b_idx])
    #                         merged_for_a[a_idx] = m_gene
    #                         skip_indices.add(b_idx)

    #                     # Ricostruisci il nuovo genotipo rispettando l'ordine originale
    #                     nuovo_genotype = []
    #                     for idx in range(len(genotype)):
    #                         if idx in merged_for_a:
    #                             nuovo_genotype.append(merged_for_a[idx])
    #                         elif idx in skip_indices:
    #                             continue
    #                         else:
    #                             nuovo_genotype.append(genotype[idx])

    #                     genotype[:] = nuovo_genotype
    #                     changed = True
    #                     break
                    
    #                 tested_neighbors += 1
    #                 if tested_neighbors >= max_neighbors:
    #                     break
                
    #             if changed:
    #                 break 
                    
    #     return genotype, self.compute_cost_genotype(genotype)


    def merge_all_possible(self, genotype, max_neighbors=5, k_search=50):
        """
        Versione O(n log n) per iterazione, invece di O(n^2 * k_search).

        COLLI DI BOTTIGLIA RISOLTI rispetto alla versione precedente:

        1. gene_sets.index(set_b)  ->  O(n) per chiamata, chiamata O(n*k) volte
        FIX: dizionario set_to_indices[frozenset] = deque di indici,
        lookup e pop in O(1).

        2. Il break dopo un SOLO merge forzava un restart completo del while
        (ricostruzione di gene_sets, centroidi, KDTree per applicare un
        singolo cambiamento). FIX: si applicano TUTTI i merge trovabili
        in una singola passata sui set unici, poi si ricostruisce solo
        se sono stati fatti merge.

        3. idxs_a/idxs_b ricalcolati con una scansione O(n) per ogni merge
        trovato. FIX: mantenuti come deque pre-indicizzate, consumate con
        popleft() O(1).

        4. nuovo_genotype ricostruito scansionando l'intero genotype ad ogni
        merge accettato. FIX: costruito una sola volta per passata,
        scorrendo gli indici "da tenere" raccolti durante la scansione.

        Complessità per passata: O(n log n + n_unique * k_search)
        invece di O(n^2 * k_search) nel caso peggiore della versione precedente.
        Il numero di passate del while è tipicamente O(log n) perché ogni
        passata applica TUTTI i merge possibili contemporaneamente, non uno
        alla volta.
        """
        from scipy.spatial import KDTree
        import numpy as np
        import networkx as nx

        if not hasattr(self, '_node_positions'):
            self._node_positions = nx.get_node_attributes(self.graph, 'pos')

        def get_gene_centroid(gene):
            coords = [self._node_positions[c] for c, _ in gene if c in self._node_positions]
            return np.mean(coords, axis=0) if coords else np.array([0.5, 0.5])

        genotype = list(genotype)  # copia di lavoro

        while True:
            n = len(genotype)
            if n <= 1:
                break

            # ── 1. Indicizzazione O(n): frozenset -> deque di indici ─────────
            gene_sets = [frozenset(c for c, _ in g) for g in genotype]
            set_to_indices: dict = defaultdict(deque)
            for idx, gs in enumerate(gene_sets):
                set_to_indices[gs].append(idx)

            unique_sets = list(set_to_indices.keys())
            if len(unique_sets) <= 1:
                break

            # Un gene rappresentativo per ogni set unico (per centroide e costo)
            rep_gene = {gs: genotype[idxs[0]] for gs, idxs in set_to_indices.items()}

            # ── 2. Centroidi + KDTree SOLO sui set unici, UNA volta per passata ─
            centroids = np.array([get_gene_centroid(rep_gene[gs]) for gs in unique_sets])
            tree = KDTree(centroids)
            set_to_pos = {gs: i for i, gs in enumerate(unique_sets)}

            any_merge_this_pass = False
            keep_indices: list = []          # indici originali da mantenere intatti
            merged_genes: list = []          # nuovi geni prodotti da merge

            # ── 3. Scansione dei soli set UNICI (non di tutti gli n geni!) ────
            for gs in unique_sets:
                idxs = set_to_indices[gs]
                if not idxs:
                    continue  # già consumato completamente da un merge precedente

                pos_a = set_to_pos[gs]
                k_eff = min(k_search, len(unique_sets))
                _, neighbor_positions = tree.query(centroids[pos_a], k=k_eff)
                if np.isscalar(neighbor_positions):
                    neighbor_positions = [neighbor_positions]

                gene_a = rep_gene[gs]
                tested = 0
                merged_here = False

                for pos_b in neighbor_positions:
                    set_b = unique_sets[pos_b]
                    if set_b == gs or not gs.isdisjoint(set_b):
                        continue
                    idxs_b = set_to_indices[set_b]
                    if not idxs_b:
                        continue  # già consumato da un merge precedente in questa passata

                    gene_b = rep_gene[set_b]
                    cost_a = self._gene_cost(gene_a)
                    cost_b = self._gene_cost(gene_b)
                    merged_gene, merged_cost = self.merge_genes(gene_a, gene_b)

                    if merged_cost <= cost_a + cost_b:
                        # ── Applica il merge a TUTTE le coppie disponibili ────
                        quanti = min(len(idxs), len(idxs_b))
                        for _ in range(quanti):
                            idxs.popleft()
                            idxs_b.popleft()
                            merged_genes.append(merged_gene)
                        any_merge_this_pass = True
                        merged_here = True
                        break  # passa al prossimo set unico

                    tested += 1
                    if tested >= max_neighbors:
                        break

                if not merged_here:
                    # Nessun merge trovato per questo set: le sue copie restano intatte
                    keep_indices.extend(idxs)
                    idxs.clear()

            # ── 4. Aggiungi le copie residue di set parzialmente fusi ─────────
            # (es. set_a aveva 5 copie, solo 3 fuse: le restanti 2 vanno mantenute)
            for gs in unique_sets:
                keep_indices.extend(set_to_indices[gs])

            # ── 5. Costruisci il nuovo genotype in un'unica passata O(n) ─────
            keep_indices.sort()
            new_genotype = [genotype[i] for i in keep_indices] + merged_genes
            genotype = new_genotype

            if not any_merge_this_pass:
                break

        return genotype, self.compute_cost_genotype(genotype)

    
    def hill_climber_optimize(self, genotype, max_iterations=1000):
        """
        Applica mutation al genotipo per ottimizzarlo, mantiene solo i miglioramenti.
        """

        def merge_matching_copies(current_genotype, set_a, set_b):
            gene_sets = [frozenset(c for c, _ in g) for g in current_genotype]

            if set_a == set_b:
                matching_indices = [idx for idx, gs in enumerate(gene_sets) if gs == set_a]
                merged_for_idx = {}
                skip_indices = set()

                for left_idx, right_idx in zip(matching_indices[0::2], matching_indices[1::2]):
                    merged_gene, _ = self.merge_genes(current_genotype[left_idx], current_genotype[right_idx])
                    merged_for_idx[left_idx] = merged_gene
                    skip_indices.add(right_idx)

                if not merged_for_idx:
                    return current_genotype

                rebuilt = []
                for idx in range(len(current_genotype)):
                    if idx in merged_for_idx:
                        rebuilt.append(merged_for_idx[idx])
                    elif idx in skip_indices:
                        continue
                    else:
                        rebuilt.append(current_genotype[idx])

                return rebuilt

            idxs_a = [idx for idx, gs in enumerate(gene_sets) if gs == set_a]
            idxs_b = [idx for idx, gs in enumerate(gene_sets) if gs == set_b]

            quanti_merge = min(len(idxs_a), len(idxs_b))
            if quanti_merge == 0:
                return current_genotype

            merged_for_a = {}
            skip_indices = set()

            for k in range(quanti_merge):
                a_idx = idxs_a[k]
                b_idx = idxs_b[k]
                merged_gene, _ = self.merge_genes(current_genotype[a_idx], current_genotype[b_idx])
                merged_for_a[a_idx] = merged_gene
                skip_indices.add(b_idx)

            rebuilt = []
            for idx in range(len(current_genotype)):
                if idx in merged_for_a:
                    rebuilt.append(merged_for_a[idx])
                elif idx in skip_indices:
                    continue
                else:
                    rebuilt.append(current_genotype[idx])

            return rebuilt
        
        iterations = 0
        cost= self.compute_cost_genotype(genotype)
        while iterations < max_iterations:
            
            # seleziona due geni casuali e prova a unirli, se il costo migliora accetta la modifica
            gene1, gene2 = random.sample(genotype, 2)
            #print(f"[HILL_CLIMBER] Iteration {iterations} - Testing merge between gene {gene1} and gene {gene2}")
            cost1 = self._gene_cost(gene1)
            cost2 = self._gene_cost(gene2)
            set1 = frozenset(c for c, _ in gene1)
            set2 = frozenset(c for c, _ in gene2)
            merged_gene, merged_cost = self.merge_genes(gene1, gene2)
            if merged_cost <= cost1+cost2:
                #print(f"[HILL_CLIMBER] Merged gene {gene1} and {gene2} into {merged_gene} with cost {merged_cost:.2f} (improvement from {cost1+cost2:.2f})")
                genotype.remove(gene1)
                genotype.remove(gene2)
                genotype.append(merged_gene)
                genotype[:] = merge_matching_copies(genotype, set1, set2)
                new_cost = self.compute_cost_genotype(genotype)
                cost = new_cost
                # oltre ad unire i due geni selezionati cerchiamo anche di unire tutte le copie identiche di gene1 e gene2 per massimizzare il miglioramento
                # questo è importante perché se ci sono molte copie identiche di gene1 e gene2

            
            iterations += 1
        return genotype, self.compute_cost_genotype(genotype)

    def hill_climber_classic(self, genotype, max_iterations=1000):
        """
        Hill climber performs slit gene ore merge genes to improve solution
        """
        cost= self.compute_cost_genotype(genotype)
        for i in range(max_iterations):

            new_genotype, new_cost = self.mutation(genotype)
            if new_cost<cost:
                print(f"\n[DEBUG] find convenient mutation iteration {i}")
                genotype=new_genotype
                cost= new_cost

        return genotype, cost

#### ──────────────────────────────────────────────────────────────────────
    #  SPLIT ADAPTIVE (beta > 1.0)
#### ______________________________________________________________________

    def _refine_trip_with_weighted_path(self, city, K_fixed, gold_per_visit):
        """
        Ricalcola il path di ritorno (city -> depot) usando Dijkstra pesato
        sul gold effettivo trasportato. Una sola chiamata per città.
        Args: 
        - city: la città da cui partire
        - K_fixed: il numero di visite fisse (trip) per questa città
        - gold_per_visit: la quantità d'oro trasportata in ogni visita (total_gold / K_fixed) 
        Returns:
        - length (costo): la lunghezza totale del ritorno  (city -> depot) considerando il peso
        - path: il percorso di ritorno effettivo da city a depot ottimizzato per il peso 
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
        Adaptive Optimal Split with refinement using weighted path for return trips.
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
        # Fase 1: binary search economica con path geometrico fisso (quella attuale)
        K_fixed = self._binary_search_K(city, total_gold, max_search)  # già implementata

        # Fase 2: refinement con path pesato, su un piccolo intorno di K_fixed
        candidates = {max(1, int(K_fixed * 0.5)), K_fixed, int(K_fixed * 1.5)}
        best_K, best_cost, best_path = None, float('inf'), None

        for K in candidates:
            gold_per_visit = total_gold / K
            length, path = self._refine_trip_with_weighted_path(city, K, gold_per_visit)
            total_trip_cost = K * (self.dist_matrix[0][city] + length)  # andata leggera + ritorno pesato
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
        Returns the full fenotype and its cost.

        """
        phenotype = []
        
        for city in self.cities_to_visit:
            total_gold = self.graph.nodes[city]['gold']
            if total_gold == 0: continue

            # path di andata 
            initial_path = self.full_paths[self.node_to_idx[0]][self.node_to_idx[city]]  # depot -> city
            
            best_K, best_cost, best_path = self._adaptive_split_with_refinement(city, total_gold, max_search)
           # print(f"[DEBUG] City {city}: total_gold={total_gold}, best_K={best_K}, best_cost={best_cost:.2f}, best_return_path={best_path}")
            for _ in range(best_K):
                if phenotype:
                    # collega il depot corrente alla city corrente senza duplicare depot/city
                    phenotype.extend((c, 0) for c in initial_path[1:-1])
                phenotype.append((city, total_gold / best_K))
                phenotype.extend([ (c, 0) for c in best_path[1:]])  # aggiungi il percorso di ritorno ottimizzato, evitando di ripetere la città
                
            #print(f"[DEBUG] Phenotype after city {city}: {phenotype}")
        return phenotype, self.compute_cost_phenotype(phenotype)

    def generate_adaptive_split(self, max_search=50):
        """
        Adaptive Optimal Split (Binary Search Optimized)
        This function finds which one id the best number of trips (K) to split each city into, using binary search.
        """
        if self._beta <= 1.0:
            # Per beta <= 1 i trip singoli non sono vantaggiosi; usa l'euristica NN
            return self._improved_baseline_individual()
        best_trips = []
        
        for city in self.cities_to_visit:
            total_gold = self.graph.nodes[city]['gold']
            if total_gold == 0: continue
            
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

            # Apply the winner configuration
            # winner_trip was created with gold = total/winner_k
            # So repeating it winner_k times yields exactly total gold.
            if winner_k  >0:
                best_trips.extend([winner_trip.copy() for _ in range(winner_k)])
            
        return best_trips, self.compute_cost_genotype(best_trips)
    
    # def generate_adaptive_split_v2(self) -> tuple:
    #     """
    #     Generazione adattiva basata su costo marginale reale ed euristica di inserimento.
    #     Garantisce la feasibility al 100% gestendo accuratamente i residui float.
    #     """
    #     beta = self._beta
    #     alpha = self._alpha
    #     dm = self.dist_matrix
    #     n2i = self.node_to_idx
    #     ng = self.node_gold
    #     gc = self._gene_cost

    #     # Clona i valori esatti di partenza per essere sicuri di raccogliere TUTTO l'oro
    #     pool_cities = {c: float(ng.get(c, 0.0)) for c in self.cities_to_visit if ng.get(c, 0.0) > 0}
    #     final_genotype = []

    #     while pool_cities:
    #         current_trip = []
    #         current_node = 0  # Partiamo dal depot

    #         while pool_cities:
    #             ni_curr = n2i[current_node]
                
    #             # 1. Trova la città più vicina tra quelle rimaste nel pool
    #             next_city = min(pool_cities.keys(), key=lambda c: dm[ni_curr, n2i[c]])
    #             gold_available = pool_cities[next_city]

    #             # Se il viaggio corrente è vuoto, inseriamo la città di partenza
    #             if not current_trip:
    #                 quota = min(gold_available, 10.0)
    #                 # Protezione float: se prendiamo quasi tutto, prendiamo tutto
    #                 if abs(gold_available - quota) < 1e-4:
    #                     quota = gold_available
                    
    #                 current_trip.append((next_city, quota))
    #                 pool_cities[next_city] -= quota
    #                 if pool_cities[next_city] < 1e-4:
    #                     del pool_cities[next_city]
    #                 current_node = next_city
    #                 continue

    #             # 2. VALUTAZIONE COSTO MARGINALE CON OTTIMIZZAZIONE LOCALE
    #             # Calcoliamo il costo se dovessimo fare un viaggio isolato Depot -> Next -> Depot
    #             cost_dedicated_future = gc([(next_city, gold_available)])

    #             # Testiamo l'inserimento ottimizzato di una quota minima
    #             test_amount = min(gold_available, 1.0)
    #             candidate_trip = current_trip + [(next_city, test_amount)]
    #             # IMPORTANTE: ottimizziamo l'ordine prima di calcolare il costo, 
    #             # altrimenti l'ordine casuale fa esplodere la funzione di costo!
    #             candidate_trip, _ = self.optimize_gene(candidate_trip)
                
    #             cost_if_added = gc(candidate_trip)
    #             cost_if_closed = gc(current_trip)
    #             marginal_cost_increase = cost_if_added - cost_if_closed

    #             # Se l'inserimento costa più di un viaggio dedicato futuro, interrompiamo questo trip
    #             if marginal_cost_increase > cost_dedicated_future:
    #                 break
                
    #             # Se è conveniente, cerchiamo di caricare quanta più roba possibile
    #             step = max(1.0, gold_available / 5.0)
    #             gold_taken = 0.0
                
    #             while gold_taken < gold_available:
    #                 next_chunk = min(step, gold_available - gold_taken)
    #                 test_trip = current_trip + [(next_city, gold_taken + next_chunk)]
    #                 test_trip, _ = self.optimize_gene(test_trip)
                    
    #                 # Condizione di stop economico
    #                 if (gc(test_trip) - cost_if_closed) > gc([(next_city, gold_available)]):
    #                     break
                        
    #                 gold_taken += next_chunk
    #                 # Freno di sicurezza basato sulla combinazione Alpha-Peso
    #                 if (sum(g for _, g in current_trip) + gold_taken) * alpha > 2.0:
    #                     break

    #             # Se abbiamo preso dell'oro, aggiorniamo le strutture
    #             if gold_taken > 0:
    #                 # Protezione assoluta dai residui float
    #                 if abs(gold_available - gold_taken) < 1e-4:
    #                     gold_taken = gold_available

    #                 current_trip.append((next_city, gold_taken))
    #                 pool_cities[next_city] -= gold_taken
    #                 if pool_cities[next_city] < 1e-4:
    #                     del pool_cities[next_city]
    #                 current_node = next_city
    #             else:
    #                 # Se non è stato possibile prendere oro, interrompiamo il trip per evitare loop infiniti
    #                 break

    #         if current_trip:
    #             # Consolidamento e ottimizzazione finale del singolo trip
    #             optimized_trip, _ = self.optimize_gene(current_trip)
    #             # Risolviamo eventuali duplicati interni alla rotta aggregandoli (richiesto dal tuo solver)
    #             merged_dict = {}
    #             for c, g in optimized_trip:
    #                 merged_dict[c] = merged_dict.get(c, 0.0) + g
                
    #             final_trip, _ = self.optimize_gene(list(merged_dict.items()))
    #             final_genotype.append(final_trip)

    #     return final_genotype, self.compute_cost_genotype(final_genotype)


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
          < 0.8 → split a random gene into two (if it has ≥ 2 cities)
          ≥ 0.8 → merge two random genes into one (if it doesn't exceed gold limits)
        """
        if not genotype:
            return genotype, self.compute_cost_genotype(genotype)
        
        # scegliamo dinamicamente se preferire spli o merge in base a beta


        ratio = random.random()

        if ratio <= self.mutation_threshold:
            # Split
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
        
        if self.beta > 1.0: 
            # in questo caso non ha senso unire i tour il costo di trasportare oro è troppo grande,
            # print(f"[INFO] Using adaptive split for beta={self.beta} and pop_size={self.pop_size} NO GA will be run...")
            # genotype, best_cost = self.generate_adaptive_split(max_search=1000)
            # genotype, best_cost = self.hill_climber_optimize(genotype, max_iterations=100000)
            
            # phenotype= self.genotype_to_phenotype(genotype)
            phenotype, best_cost = self._generate_solution_with_adaptive_split(max_search=1000)
            return  phenotype, best_cost

        genotype, best_cost = self.run_ga_logic(fast=fast)
        phenotype = self.genotype_to_phenotype(genotype)
        return phenotype, best_cost
