
import random 

import numpy as np
import networkx as nx


class GA_Solver:
   
    def __init__(self, problem, pop_size=50, generations=100, offprint=20):
        self.prob = problem
        self.graph = problem.graph
        self.alpha = problem.alpha
        self.beta = problem.beta
        
        # Nodi rilevanti (deposito + città con oro)
        self.relevant_nodes = sorted([n for n in self.graph.nodes if n == 0 or self.graph.nodes[n].get('gold', 0) > 0])
        self.node_to_idx = {node: i for i, node in enumerate(self.relevant_nodes)}
        self.cities_to_visit = [n for n in self.relevant_nodes if n != 0]
        n_rel = len(self.relevant_nodes)

        # Matrici NumPy per distanze e oro (accesso ultra-rapido)
        self.dist_matrix = np.zeros((n_rel, n_rel))
        # Conserviamo i path reali (nodi del grafo originale) tra nodi rilevanti
        self.full_paths = [[None] * n_rel for _ in range(n_rel)]
        self.node_gold = {n: self.graph.nodes[n].get('gold', 0) for n in self.graph.nodes}

        for i, source in enumerate(self.relevant_nodes):
            lengths, paths = nx.single_source_dijkstra(self.graph, source, weight='dist')
            for j, target in enumerate(self.relevant_nodes):
                if target in lengths:
                    self.dist_matrix[i, j] = lengths[target]
                    self.full_paths[i][j] = paths[target]

        self.pop_size = pop_size
        self.generations = generations
        self.offprint = offprint

    def compute_cost_genotype(self, genotype):
        # Calcola il costo totale del percorso rappresentato dal genotipo (permutazione dei nodi rilevanti)
        # genotype sample: [[ (1,10)] , [(2,20), (3,30)] ...]
        total_cost=0
        
        for gene in genotype:
            start=0
            gene_cost=0
            gold=0
            for city, gold_amount in gene:
                # necessario calcolare il costo edge per edge per tenere conto del peso dinamico (oro raccolto)
                # per Beta > 1 non è equivalente usare la distanza totale del percorso implicito tra start e city, perchè il peso (oro) cambia ad ogni edge
                implicit_path= self.full_paths[start][city]
                
                for c in implicit_path[1:]: # percorro il path implicito tra start e city
                    d= self.graph[start][c]['dist']
                    gene_cost+= d + (d* self.alpha*gold )**self.beta
                    start=c
                
                gold+= gold_amount
            
            # ultimo percorso 
            last_path= self.full_paths[start][0]
            for c in last_path[1:]: # percorro il path implicito tra start e depot
                last_d= self.graph[start][c]['dist']
                gene_cost+= last_d + (last_d* self.alpha*gold )**self.beta
                start=c
            total_cost+=gene_cost 
        
        # tolgo la distanza tra depot e prima città con oro -> si parte dalla prima città con oro
        start_cost= self.dist_matrix[0][genotype[0][0][0]]
        total_cost-= start_cost
        return total_cost

    def check_feasibility_genotype(self, genotype):
        # Verifica che il genotipo rappresenti un percorso valido: esiste un percorso tra i nodi e viene raccolto tutto l'oro
        problem = self.prob
        graph = problem.graph
        gold_at = nx.get_node_attributes(graph, "gold")
        
        gold_collected = {}
        i = 0
        for gene in genotype:
            start=0
            for city, gold in gene:
                # check path existence
                if self.full_paths[start][city] is None:
                    print(f"[FAIL] Feasibility failed: no path between {start} and {city}")
                    print(f"Gene segment: {start} -> {city}")
                    return False
                
                # Track collected gold
                if gold > 0:
                    gold_collected[city] = gold_collected.get(city, 0.0) + gold
                start = city
             
        # Verify all gold was collected
        for city in graph.nodes():
            if city == 0:  # Depot has no gold
                continue
            expected_gold = gold_at.get(city, 0.0)
            collected_gold = gold_collected.get(city, 0.0)
            
            if abs(expected_gold - collected_gold) > 1e-4:  # Float tolerance
                print(f"[FAIL] Feasibility failed: city {city} i={i} has {expected_gold:.2f} gold, collected {collected_gold:.2f}")
                return False
            i += 1
        return True

    def genotype_to_phenotype(self, genotype):
        # Converte un genotipo (permutazione dei nodi rilevanti) in un fenotipo (percorso reale con nodi del grafo originale)
        phenotype= []
        if not genotype:
            return phenotype

        for gene in genotype:
            start=0
            for city, gold in gene:
                implicit_cities = [(c, 0) for c in self.full_paths[start][city][1:-1]]
                phenotype.extend(implicit_cities)
                phenotype.append((city, gold))
                start= city
            depot=0
            implicit_cities = [(c, 0) for c in self.full_paths[start][depot][1:]]
            phenotype.extend(implicit_cities)
        
        # tolgo le prime città implicite 
        first_city=genotype[0][0]
        index= phenotype.index(first_city)
        return phenotype[index:] # Return the segment starting from the first city
            

    def compute_cost_phenotype(self, phenotype):
        # Calcola il costo totale del percorso reale 
        # phenotype sample= [(1,10), (0,0), (1,0), (2,20), (1,0), (0,0)]
        total_cost = 0
        start= phenotype[0][0]
        current_gold= phenotype[0][1]
        for city, gold in phenotype[1:]:
            # controllo se esiste l'arco u, v
            if not self.graph.has_edge(start, city):
                print(f"Warning: No edge between {start} and {city}. Returning inf cost.")
                return float("inf")
            d = self.graph[start][city]['dist']
            total_cost += d + (self.prob.alpha * d * current_gold) ** self.prob.beta
            if city == 0:
                current_gold = 0
            else:
                current_gold += gold
            
            start = city
        
        return total_cost

    def check_feasibility_phenotype(self, phenotype):
        # Verifica che il phenotype sia valido
        problem = self.prob
        graph = problem.graph
        gold_at = nx.get_node_attributes(graph, "gold")

        if not phenotype:
            return False
        
        # Track collected gold per city
        gold_collected = {}

        start = phenotype[0][0]
        initial_gold = phenotype[0][1]
        
        # Track initial node's gold
        if initial_gold > 0:
            gold_collected[start] = initial_gold
        
        for city, gold in phenotype[1:]:
            # Check adjacency
            if not graph.has_edge(start, city):
                print(f"[FAIL] Feasibility failed: no edge between {start} and {city}")
                print(f"Path segment: {city} -> {city}")
                print(phenotype)
                return False
            
            # Track collected gold
            if gold > 0:
                gold_collected[city] = gold_collected.get(city, 0.0) + gold
                
            start = city
        
        # Verify all gold was collected
        for city in graph.nodes():
            if city == 0:  # Depot has no gold
                continue
            expected_gold = gold_at.get(city, 0.0)
            collected_gold = gold_collected.get(city, 0.0)
            
            if abs(expected_gold - collected_gold) > 1e-4:  # Float tolerance
                print(f"[FAIL] Feasibility failed: city {city} has {expected_gold:.2f} gold, collected {collected_gold:.2f}")
                return False
            
        return True
    
    
    def evaluate_and_segment(self, chromosome: list[int]) -> tuple[list[tuple[int, float]], float]:
        """Greedy decoder: decides whether to return to the depot to unload.
        
        Args:            
            chromosome: List of city indices to visit in order (excluding depot).
        Returns:            
            genotype. list of list of tuples (city, gold collected at city) representing the route with explicit unloads.
            total_cost: Total cost of the route.
        """
        genotype = []
        # se nel chromosome è presente il deposito, lo tolgo (non ha senso visitarlo più di una volta)
        chromosome = [c for c in chromosome if c != 0]
        if not chromosome:
            return genotype, 0.0
        current_node = chromosome[0]
        current_gold = self.graph.nodes[current_node].get('gold', 0)
        route = []
        
        total_cost = 0
        
        if current_gold > 0:
            route.append((current_node, current_gold))

        for next_target in chromosome[1:]:
            start_node = current_node

            # Calcola il costo diretto di andare da start_node a next_target.
            path_direct_distance = self.full_paths[start_node][next_target]
            cost_direct = 0
            traversal_node = start_node
            for c in path_direct_distance[1:]:  # percorro il path implicito tra start_node e next_target
                d = self.graph[traversal_node][c]['dist']
                cost_direct += d + (self.alpha * d * current_gold) ** self.beta
                traversal_node = c

            # Calcola il costo di andare al deposito, scaricare, e poi andare a next_target.
            # Il tratto deposito -> next_target viene percorso a peso nullo, quindi il costo
            # è solo la distanza geometrica complessiva.
            path_to_depot_path = self.full_paths[start_node][0]
            distance_from_depot = self.dist_matrix[0][next_target]
            cost_unload = 0
            traversal_node = start_node
            for c in path_to_depot_path[1:]:  # percorro il path implicito tra start_node e deposito
                d = self.graph[traversal_node][c]['dist']
                cost_unload += d + (self.alpha * d * current_gold) ** self.beta
                traversal_node = c

            # sommo il costo lineare di arrivare al next target senza oro (solo la distanza)
            cost_unload += distance_from_depot

            if current_gold > 0 and cost_unload < cost_direct:
                # Unload at depot before going to next_target
                genotype.append(route)
                current_gold = self.graph.nodes[next_target].get('gold', 0)
                route = [(next_target, current_gold)]
                total_cost += cost_unload
            else:
                g= self.graph.nodes[next_target].get('gold', 0)
                route.append((next_target, g))
                total_cost += cost_direct
                current_gold += g
            current_node = next_target

        path_home_distance = self.full_paths[current_node][0]
        
        for c in path_home_distance[1:]: # percorro il path implicito tra current_node e deposito
            d = self.graph[current_node][c]['dist']
            total_cost += d + (self.alpha * d * current_gold) ** self.beta
            current_node = c

        genotype.append(route)
            
        return genotype, total_cost
         

    def _multiple_cycle (self, genotype: list)-> tuple[list, float]:
        """Split tours into multiple lighter trips when beneficial."""
        new_genotype = []
        for tour in genotype:
            cost= self.compute_cost_genotype([tour])
          
            best_factor=1
            min_gold= min(tour, key=lambda x: x[1])[1]
            for i in range(2, int (min_gold)+1 ):
              single_trip=[]
              # calcola il consto di approssimato di 1 trip: i* costo_singolo_tour(prendendo w//i oro)
              single_trip.extend( [ (c, w//i ) for (c, w) in tour ]) 
              cost_single_trip= self.compute_cost_genotype([single_trip])
              approx_cost= cost_single_trip*i

              if cost_single_trip*i < cost :
                  cost= approx_cost
                  best_factor=i
                  continue 
              else:
                  break # Dato che i cresce, se non migliora non migliorerà più
          
            # costruisco il genotype con w//best_factor e r= w%best_factor      
            for j in range(best_factor):
                r= [ 0 for _ in tour]
                if j== best_factor-1:
                    r= [ w % best_factor for c, w in tour]
                new_genotype.append( [ (c, w//best_factor + r) for (c, w), r in zip(tour, r) ]) 
            
        return new_genotype, self.compute_cost_genotype( new_genotype) 
    

    def _optimize_tour(self, gene: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """Re-decode a single tour to improve its visit order."""
        ## orded the city from the farest to the closest to the depot, while keeping the same gold amounts
        ## check if the tour can be simplified by reordering cities (we don't want to repeat path segments if not necessary)
        pass  # Da implementare
    
    
    

# ------------ GENETIC ALGORITHM LOGIC ------------
    
    def generate_initial_population(self):
        # Genera una popolazione iniziale di genotipi (permutazioni dei nodi rilevanti)
        # Ogni genotipo rappresenta un possibile percorso di raccolta dell'oro

        population = []
        for _ in range(self.pop_size):
            chromosome = self.cities_to_visit[:]
            np.random.shuffle(chromosome)
            genotype, cost= self.evaluate_and_segment(chromosome)
            if self.beta > 1.0:
                genotype, cost= self._multiple_cycle(genotype)
            population.append((genotype, cost))

        return population
    

    def mutation(self, genotype):
        # Applica una mutazione al genotipo: ad esempio, scambia due nodi o inverte un segmento del percorso
        # genotype sample: [[ (1,10)] , [(2,20), (3,30)] ...]
        switch = np.random.rand()
        new_genotype = [list(gene) for gene in genotype]  # Deep copy of genotype
        
        if switch < 0.8:
            # Swap mutation: scambia due tuple tra geni diversi
            if len(new_genotype) >= 2:
                index_g1, index_g2 = np.random.choice(len(new_genotype), 2, replace=False)
                gene1 = new_genotype[index_g1]
                gene2 = new_genotype[index_g2]
                
                if len(gene1) > 0 and len(gene2) > 0:
                    idx1 = np.random.randint(len(gene1))
                    idx2 = np.random.randint(len(gene2))
                    
                    # Scambia gli elementi
                    gene1[idx1], gene2[idx2] = gene2[idx2], gene1[idx1]
                    
                    new_genotype[index_g1] = gene1
                    new_genotype[index_g2] = gene2
        else:
            # Inversion mutation: inverte un segmento all'interno di un gene
            if len(new_genotype) > 0:
                index_gene = np.random.randint(len(new_genotype))
                gene = new_genotype[index_gene]
                
                if len(gene) > 1:
                    # Inverti un sottosegmento casuale
                    start = np.random.randint(len(gene))
                    end = np.random.randint(start + 1, len(gene) + 1)
                    gene[start:end] = gene[start:end][::-1]
                    new_genotype[index_gene] = gene

        return new_genotype, self.compute_cost_genotype(new_genotype)
    
    # MIGLIORAMENTO 
    # al posto di mutation così potrei fare merge e split di geni, in modo da spostare interi segmenti di percorso da un gene all'altro, o creare nuovi geni (nuovi tour) o eliminarne alcuni (unire tour)
    # e ottimizzarli (reorder) con la funzione _optimize_tour

    def crossover(self, parent1, parent2):
        # Applica un crossover tra due genotipi per generare un nuovo genotipo (figlio)
        # Prende le prime n città da parent1 e le rimanenti da parent2, mantenendo l'ordine
        
        # Estrai le città da ogni parent in ordine di visita
        cities_p1 = [city for gene in parent1 for city, _ in gene]
        cities_p2 = [city for gene in parent2 for city, _ in gene]
        
        # Crossover: prendi n città da parent1 e il resto da parent2
        n = np.random.randint(1, max(2, len(self.cities_to_visit)))
        
        # Crea chromosome: prime n città da parent1, poi quelle mancanti da parent2
        chromosome = []
        cities_seen = set()
        
        # Aggiungi le prime n città da parent1
        for city in cities_p1[:n]:
            if city not in cities_seen:
                chromosome.append(city)
                cities_seen.add(city)
        
        # Aggiungi le città rimanenti da parent2
        for city in cities_p2:
            if city not in cities_seen and city in self.cities_to_visit:
                chromosome.append(city)
                cities_seen.add(city)
        
        # Se mancano ancora città, aggiungile
        for city in self.cities_to_visit:
            if city not in cities_seen:
                chromosome.append(city)
                cities_seen.add(city)
        
        genotype, cost = self.evaluate_and_segment(chromosome)
        if self.beta > 1.0:
            genotype, cost = self._multiple_cycle(genotype)
        return genotype, cost
            
        
    
    def run_ga_logic(self, fast: bool = False )-> list:
        # Logica dell'algoritmo genetico: selezione, crossover, mutazione, e valutazione della fitness
        # Restituisce il percorso ottimizzato (genotype) come lista di tuple (nodo, oro raccolto)
                # Per problemi grandi e beta>=2 , uso solo la greedy perchè multiple_cycle è molto costosa
        # if self.prob.beta >= 2.0 and len(self.cities_to_visit) > 1000:
        #     solution, _ = self._greedy_initialization()
        #     return solution
        # population = []


        population = self.generate_initial_population()
        population= [(ind, cost) for ind, cost in sorted(population, key=lambda x: x[1])]  # ordina per costo
        
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
                    offspring1, cost_1=self.mutation(parents[0])
                    offspring2, cost_2=self.mutation(parents[1])
                else:
                    offspring1, cost_1 = self.crossover(parents[0], parents[1])
                    offspring2, cost_2 = self.crossover(parents[1], parents[0])
                
                next_gen.extend([(offspring1, cost_1), (offspring2, cost_2)])

            
            next_gen = [(ind, cost) for ind, cost in sorted(next_gen, key=lambda x: x[1])]  
            population = next_gen[:self.pop_size]  # mantieni solo i migliori
        
            if population[0][1] < best_cost:
                best_chromo, best_cost = population[0]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            if stagnation_counter >= 10 and fast:
                break
        
        return best_chromo, best_cost

    def solution(self, fast=True):
        genotype, best_cost = self.run_ga_logic(fast=fast)
        # if not self.check_feasibility_genotype(genotype):
        #     print("[ERROR] Final genotype is not feasible!")
        # gen_cost = self.compute_cost_genotype(genotype)
        # if abs(gen_cost - best_cost) > 1e-4:
        #     print(f"[WARNING] Genotype cost {gen_cost:.2f} does not match recorded best cost {best_cost:.2f}")
        
        phenotype= self.genotype_to_phenotype(genotype)
        # if not self.check_feasibility_phenotype(phenotype):
        #     print("[ERROR] Final phenotype is not feasible!")
        # phen_cost = self.compute_cost_phenotype(phenotype)
        # print(f"Final genotype cost: {gen_cost:.2f} | Final phenotype cost: {phen_cost:.2f} | Recorded best cost: {best_cost:.2f}")
        return phenotype, best_cost 
    