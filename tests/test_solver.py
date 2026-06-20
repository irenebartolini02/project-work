import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Problem import Problem
from src.Solver import Solver


class FakeProblem:
    def __init__(self):
        graph = nx.Graph()
        graph.add_node(0, gold=0)
        graph.add_node(1, gold=10)
        graph.add_node(2, gold=20)
        graph.add_node(3, gold=0)
        graph.add_node(4, gold=0)

        graph.add_edge(0, 4, dist=1)
        graph.add_edge(4, 1, dist=1)
        graph.add_edge(1, 2, dist=1)
        graph.add_edge(2, 3, dist=1)
        graph.add_edge(3, 0, dist=1)

        self._graph = graph
        self._alpha = 1.0
        self._beta = 1.0

    @property
    def graph(self):
        return nx.Graph(self._graph)

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta


class FakeProblemBetaGreater1(FakeProblem):
    def __init__(self):
        super().__init__()
        self._beta = 2.0


class FakeProblemMergeAllPossible:
    def __init__(self):
        graph = nx.Graph()
        graph.add_node(0, gold=0.0, pos=(0.0, 0.0))
        graph.add_node(1, gold=0.3, pos=(1.0, 0.0))
        graph.add_node(2, gold=0.7, pos=(2.0, 0.0))

        graph.add_edge(0, 1, dist=1.0)
        graph.add_edge(1, 2, dist=1.0)
        graph.add_edge(2, 0, dist=1.0)

        self._graph = graph
        self._alpha = 1.0
        self._beta = 1.0

    @property
    def graph(self):
        return nx.Graph(self._graph)

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta


class GASolver2Tests(unittest.TestCase):
    def setUp(self):
        self.problem = FakeProblem()
        self.solver = Solver(self.problem)
        self.genotype = [[(1, 10), (2, 20)]]
        self.phenotype = [(1, 10), (2, 20), (3, 0), (0, 0)]

    def test_initialization_keeps_only_relevant_nodes(self):
        self.assertEqual(self.solver.relevant_nodes, [0, 1, 2])
        self.assertEqual(self.solver.cities_to_visit, [1, 2])
        self.assertEqual(self.solver.node_to_idx, {0: 0, 1: 1, 2: 2})
        self.assertEqual(self.solver.full_paths[0][1], [0, 4, 1])
        self.assertEqual(self.solver.full_paths[1][2], [1, 2])
        self.assertEqual(self.solver.full_paths[2][0], [2, 3, 0])

    def test_genotype_to_phenotype_expands_implicit_nodes(self):
        phenotype = self.solver.genotype_to_phenotype(self.genotype)
        self.assertEqual(phenotype, self.phenotype)

    def test_compute_cost_genotype_matches_phenotype_cost(self):
        genotype_cost = self.solver.compute_cost_genotype(self.genotype)
        phenotype_cost = self.solver.compute_cost_phenotype(self.phenotype)
        self.assertAlmostEqual(genotype_cost, 73.0)
        self.assertAlmostEqual(phenotype_cost, 73.0)
        self.assertAlmostEqual(genotype_cost, phenotype_cost)

    def test_feasibility_checks_accept_valid_solution(self):
        self.assertTrue(self.solver.check_feasibility_genotype(self.genotype))
        self.assertTrue(self.solver.check_feasibility_phenotype(self.phenotype))

    def test_feasibility_checks_reject_invalid_phenotype_adjacency(self):
        invalid_phenotype = [(1, 10), (3, 0), (0, 0)]
        self.assertFalse(self.solver.check_feasibility_phenotype(invalid_phenotype))

    def test_compute_cost_phenotype_returns_inf_for_missing_edge(self):
        invalid_phenotype = [(1, 10), (3, 0), (0, 0)]
        self.assertTrue(math.isinf(self.solver.compute_cost_phenotype(invalid_phenotype)))

    def test_evaluate_and_segment_returns_valid_genotype(self):
        genotype, cost = self.solver.evaluate_and_segment([1, 2])
        self.assertIsInstance(genotype, list)
        self.assertGreater(cost, 0)
        self.assertTrue(self.solver.check_feasibility_genotype(genotype))

    def test_evaluate_and_segment_cost_differs_by_start_cost(self):
        genotype, segmented_cost = self.solver.evaluate_and_segment([1, 2])
        compute_cost = self.solver.compute_cost_genotype(genotype)
        self.assertAlmostEqual(segmented_cost, compute_cost)

    def test_generate_initial_population_valid_and_cost_relation(self):
        np.random.seed(0)
        population = self.solver.generate_initial_population()
        self.assertEqual(len(population), self.solver.pop_size)
        for genotype, reported_cost in population:
            self.assertTrue(self.solver.check_feasibility_genotype(genotype))
            computed_cost = self.solver.compute_cost_genotype(genotype)
            self.assertAlmostEqual(reported_cost, computed_cost)

    def test_optimize_gene_reorders_cities_to_lower_cost(self):
        gene = [(2, 20), (1, 10)]

        optimized_gene, optimized_cost = self.solver.optimize_gene(gene)

        #self.assertEqual(optimized_gene, [(1, 10), (2, 20)])
        self.assertAlmostEqual(optimized_cost, self.solver._gene_cost(optimized_gene))
        self.assertLess(optimized_cost, self.solver._gene_cost(gene))

    def test_merge_genes_sums_overlap_gold(self):
        gene1 = [(1, 4)]
        gene2 = [(1, 6)]

        merged_gene, merged_cost = self.solver.merge_genes(gene1, gene2)

        self.assertEqual(merged_gene, [(1, 10)])
        self.assertAlmostEqual(merged_cost, self.solver._gene_cost(merged_gene))

    def test_merge_all_possible_preserves_gold_with_duplicate_city_sets(self):
        problem = FakeProblemMergeAllPossible()
        solver = GA_Solver(problem)

        genotype = [
            [(1, 0.1)],
            [(1, 0.2)],
            [(2, 0.3)],
            [(2, 0.4)],
        ]

        merged_genotype, merged_cost = solver.merge_all_possible(
            [list(gene) for gene in genotype],
            max_neighbors=5,
        )

        self.assertTrue(solver.check_feasibility_genotype(merged_genotype))
        self.assertAlmostEqual(merged_cost, solver.compute_cost_genotype(merged_genotype))

        collected = {1: 0.0, 2: 0.0}
        for gene in merged_genotype:
            for city, gold in gene:
                if city in collected:
                    collected[city] += gold

        self.assertAlmostEqual(collected[1], 0.3)
        self.assertAlmostEqual(collected[2], 0.7)

    def test_hill_climber_merges_duplicate_gene_families_after_successful_merge(self):
        problem = FakeProblem()
        solver = GA_Solver(problem)

        genotype = [
            [(1, 4)],
            [(1, 6)],
            [(2, 7)],
            [(2, 8)],
        ]

        def fake_merge_genes(gene1, gene2):
            return list(gene1) + list(gene2), 0.0

        with patch("src.GA_solver_2.random.sample", side_effect=lambda population, k: [population[0], population[2]]), \
             patch.object(solver, "merge_genes", side_effect=fake_merge_genes):
            optimized_genotype, optimized_cost = solver.hill_climber_optimize([list(gene) for gene in genotype], max_iterations=1)

        self.assertEqual(len(optimized_genotype), 2)
        self.assertTrue(all(len(gene) == 2 for gene in optimized_genotype))
        self.assertAlmostEqual(optimized_cost, solver.compute_cost_genotype(optimized_genotype))

    def test_split_gene_optimizes_first_segment_and_handles_empty_second(self):
        gene = [(2, 20), (1, 10)]

        (gene1, cost1), (gene2, cost2) = self.solver.split_gene(gene, 2)

        #self.assertEqual(gene1, [(1, 10), (2, 20)])
        self.assertAlmostEqual(cost1, self.solver._gene_cost(gene1))
        self.assertEqual(gene2, [])
        self.assertAlmostEqual(cost2, 0.0)


class GASolver2TestsBetaGreater1(unittest.TestCase):
    def setUp(self):
        self.problem = FakeProblemBetaGreater1()
        self.solver = GA_Solver(self.problem)
        self.genotype = [[(1, 10), (2, 20)]]

    def test_multiple_cycle_preserves_feasibility_and_gold(self):
        segmented_genotype, _ = self.solver._multiple_cycle(self.genotype)
        self.assertTrue(self.solver.check_feasibility_genotype(segmented_genotype))

        collected = {1: 0, 2: 0}
        for gene in segmented_genotype:
            for city, gold in gene:
                if city in collected:
                    collected[city] += gold

        self.assertAlmostEqual(collected[1], 10)
        self.assertAlmostEqual(collected[2], 20)

    def test_multiple_cycle_cost_matches_compute(self):
        segmented_genotype, reported_cost = self.solver._multiple_cycle(self.genotype)
        self.assertAlmostEqual(reported_cost, self.solver.compute_cost_genotype(segmented_genotype))


class GASolver2AdaptiveSplitTests(unittest.TestCase):
    def setUp(self):
        self.problem = FakeProblemBetaGreater1()
        self.solver = GA_Solver(self.problem)

    def test_generate_solution_with_adaptive_split_returns_feasible_phenotype(self):
        np.random.seed(5)

        phenotype, reported_cost = self.solver._generate_solution_with_adaptive_split(max_search=20)

        self.assertTrue(self.solver.check_feasibility_phenotype(phenotype))
        self.assertFalse(math.isinf(reported_cost))
        self.assertAlmostEqual(reported_cost, self.solver.compute_cost_phenotype(phenotype))

        gold_collected = {}
        for city, gold in phenotype:
            if city != 0:
                gold_collected[city] = gold_collected.get(city, 0.0) + gold

        for city in self.solver.cities_to_visit:
            expected_gold = self.solver.graph.nodes[city].get('gold', 0)
            self.assertAlmostEqual(gold_collected.get(city, 0.0), expected_gold)

    def test_generate_solution_with_adaptive_split_keeps_return_path_structure(self):
        def fake_refinement(city, total_gold, max_search):
            if city == 1:
                return 2, 0.0, [1, 4, 0]
            return 1, 0.0, [2, 3, 0]

        with patch.object(self.solver, "_adaptive_split_with_refinement", side_effect=fake_refinement):
            phenotype, reported_cost = self.solver._generate_solution_with_adaptive_split(max_search=20)

        self.assertTrue(self.solver.check_feasibility_phenotype(phenotype))
        self.assertAlmostEqual(reported_cost, self.solver.compute_cost_phenotype(phenotype))
        self.assertEqual(phenotype.count((1, self.solver.graph.nodes[1]['gold'] / 2)), 2)
        self.assertIn((4, 0), phenotype)
        self.assertIn((3, 0), phenotype)


class GASolver2ProblemIntegrationTests(unittest.TestCase):
    """Integration tests using the real Problem class."""

    def setUp(self):
        self.problem = Problem(
            num_cities=12,
            alpha=1.0,
            beta=1.5,
            density=0.7,
            seed=7,
        )
        self.solver = GA_Solver(self.problem, pop_size=12, generations=10, offprint=6)

    def test_problem_instance_builds_consistent_solver(self):
        self.assertGreaterEqual(len(self.solver.relevant_nodes), 2)
        self.assertEqual(self.solver.relevant_nodes[0], 0)
        self.assertEqual(self.solver.dist_matrix.shape[0], len(self.solver.relevant_nodes))
        self.assertEqual(self.solver.dist_matrix.shape[1], len(self.solver.relevant_nodes))

    def test_evaluate_and_segment_on_problem_returns_feasible_solution(self):
        chromosome = self.solver.cities_to_visit[:]
        genotype, reported_cost = self.solver.evaluate_and_segment(chromosome)
        self.assertTrue(self.solver.check_feasibility_genotype(genotype))

        computed_cost = self.solver.compute_cost_genotype(genotype)
        self.assertAlmostEqual(reported_cost, computed_cost)

    def test_generate_initial_population_on_problem_is_valid(self):
        np.random.seed(1)
        population = self.solver.generate_initial_population()
        self.assertEqual(len(population), self.solver.pop_size)

        for genotype, reported_cost in population:
            self.assertTrue(self.solver.check_feasibility_genotype(genotype))
            computed_cost = self.solver.compute_cost_genotype(genotype)
            self.assertAlmostEqual(reported_cost, computed_cost)

    def test_population_phenotypes_have_finite_cost(self):
        np.random.seed(2)
        population = self.solver.generate_initial_population()
        for genotype, _ in population:
            phenotype = self.solver.genotype_to_phenotype(genotype)
            ph_cost = self.solver.compute_cost_phenotype(phenotype)
            self.assertFalse(math.isinf(ph_cost))
            self.assertGreaterEqual(ph_cost, 0)

    def test_run_ga_logic_returns_feasible_solution_with_finite_cost(self):
        np.random.seed(3)
        genotype, cost = self.solver.run_ga_logic(fast=True)
        self.assertTrue(self.solver.check_feasibility_genotype(genotype))
        computed_cost = self.solver.compute_cost_genotype(genotype)
        self.assertAlmostEqual(cost, computed_cost)
        self.assertFalse(math.isinf(cost))
        self.assertGreaterEqual(cost, 0)
        cost_baseline= self.problem.baseline()
        self.assertLess(cost, cost_baseline)


class GASolver2MutationCrossoverTests(unittest.TestCase):
    """Test suite for mutation and crossover operators."""

    def setUp(self):
        self.problem = Problem(
            num_cities=10,
            alpha=1.0,
            beta=1.5,
            density=0.7,
            seed=42,
        )
        self.solver = GA_Solver(self.problem, pop_size=8, generations=5, offprint=4)
        np.random.seed(42)
        self.population = self.solver.generate_initial_population()

    def test_mutation_preserves_feasibility(self):
        """Test that mutation produces a feasible genotype."""
        parent_genotype, _ = self.population[0]
        mutated_genotype, mutated_cost = self.solver.mutation(parent_genotype)
        
        self.assertTrue(self.solver.check_feasibility_genotype(mutated_genotype))
        self.assertFalse(math.isinf(mutated_cost))
        self.assertGreater(mutated_cost, 0)

    def test_mutation_cost_consistency(self):
        """Test that mutation cost matches compute_cost_genotype."""
        parent_genotype, _ = self.population[0]
        mutated_genotype, reported_cost = self.solver.mutation(parent_genotype)
        
        computed_cost = self.solver.compute_cost_genotype(mutated_genotype)
        self.assertAlmostEqual(reported_cost, computed_cost)

    def test_mutation_preserves_all_gold(self):
        """Test that mutation preserves all gold collection."""
        parent_genotype, _ = self.population[0]
        mutated_genotype, _ = self.solver.mutation(parent_genotype)
        
        # Collect gold from mutated genotype
        gold_collected = {}
        for gene in mutated_genotype:
            for city, gold in gene:
                if city != 0:  # Exclude depot
                    gold_collected[city] = gold_collected.get(city, 0) + gold
        
        # Verify all gold from relevant cities is collected
        for city in self.solver.cities_to_visit:
            expected_gold = self.solver.graph.nodes[city].get('gold', 0)
            self.assertAlmostEqual(gold_collected.get(city, 0), expected_gold)

    def test_mutation_creates_diversity(self):
        """Test that mutation actually changes the genotype."""
        parent_genotype, _ = self.population[0]
        
        mutations = []
        for _ in range(10):
            mutated_genotype, _ = self.solver.mutation(parent_genotype)
            mutations.append(mutated_genotype)
        
        # At least some mutations should differ from parent
        different_count = sum(1 for m in mutations if str(m) != str(parent_genotype))
        self.assertGreater(different_count, 0)

    def test_crossover_produces_feasible_genotype(self):
        """Test that crossover produces a feasible genotype."""
        parent1_genotype, _ = self.population[0]
        parent2_genotype, _ = self.population[1]
        
        child_genotype, child_cost = self.solver.crossover(parent1_genotype, parent2_genotype)
        
        self.assertTrue(self.solver.check_feasibility_genotype(child_genotype))
        self.assertFalse(math.isinf(child_cost))
        self.assertGreater(child_cost, 0)

    def test_crossover_cost_consistency(self):
        """Test that crossover cost matches compute_cost_genotype."""
        parent1_genotype, _ = self.population[0]
        parent2_genotype, _ = self.population[1]
        
        child_genotype, reported_cost = self.solver.crossover(parent1_genotype, parent2_genotype)
        computed_cost = self.solver.compute_cost_genotype(child_genotype)
        
        self.assertAlmostEqual(reported_cost, computed_cost)

    def test_crossover_preserves_all_gold(self):
        """Test that crossover preserves all gold collection."""
        parent1_genotype, _ = self.population[0]
        parent2_genotype, _ = self.population[1]
        
        child_genotype, _ = self.solver.crossover(parent1_genotype, parent2_genotype)
        
        # Collect gold from child genotype
        gold_collected = {}
        for gene in child_genotype:
            for city, gold in gene:
                if city != 0:  # Exclude depot
                    gold_collected[city] = gold_collected.get(city, 0) + gold
        
        # Verify all gold from relevant cities is collected
        for city in self.solver.cities_to_visit:
            expected_gold = self.solver.graph.nodes[city].get('gold', 0)
            self.assertAlmostEqual(gold_collected.get(city, 0), expected_gold)

    def test_crossover_produces_offspring_from_both_parents(self):
        """Test that crossover inherits genetic material from both parents."""
        parent1_genotype, _ = self.population[0]
        parent2_genotype, _ = self.population[1]
        
        # Extract city sequences from parents
        cities_p1 = [city for gene in parent1_genotype for city, _ in gene]
        cities_p2 = [city for gene in parent2_genotype for city, _ in gene]
        
        child_genotype, _ = self.solver.crossover(parent1_genotype, parent2_genotype)
        cities_child = [city for gene in child_genotype for city, _ in gene]
        
        # Child should contain all cities from both parents
        self.assertEqual(set(cities_child), set(cities_p1))
        self.assertEqual(set(cities_child), set(cities_p2))

    def test_mutation_and_crossover_multiple_times(self):
        """Test that mutation and crossover work correctly when applied repeatedly."""
        genotype, cost = self.population[0]
        
        for _ in range(5):
            # Apply mutation
            genotype, cost = self.solver.mutation(genotype)
            self.assertTrue(self.solver.check_feasibility_genotype(genotype))
            
            # Apply crossover
            if len(self.population) > 1:
                other_parent, _ = self.population[1]
                genotype, cost = self.solver.crossover(genotype, other_parent)
                self.assertTrue(self.solver.check_feasibility_genotype(genotype))


class GASolver2SolutionTests(unittest.TestCase):
    """Test suite for the solution() method."""

    def setUp(self):
        self.problem = Problem(
            num_cities=10,
            alpha=1.0,
            beta=1.5,
            density=0.7,
            seed=42,
        )
        self.solver = GA_Solver(self.problem, pop_size=8, generations=5, offprint=4)

    def test_solution_returns_feasible_genotype(self):
        """Test that solution() returns a feasible genotype."""
        np.random.seed(0)
        phenotype, cost = self.solver.solution(fast=True)
        
        self.assertTrue(self.solver.check_feasibility_phenotype(phenotype))
        self.assertFalse(math.isinf(cost))
        self.assertGreater(cost, 0)

    def test_solution_cost_matches_compute_genotype(self):
        """Test that solution() cost matches compute_cost_genotype."""
        np.random.seed(1)
        phenotype, reported_cost = self.solver.solution(fast=True)
        
        computed_cost = self.solver.compute_cost_phenotype(phenotype)
        self.assertAlmostEqual(reported_cost, computed_cost)

    def test_solution_genotype_phenotype_costs_match(self):
        """Test that genotype and phenotype have the same cost."""
        np.random.seed(2)
        phenotype, genotype_cost = self.solver.solution(fast=True)
        
        phenotype_cost = self.solver.compute_cost_phenotype(phenotype)
        
        # Costs should be equal (or very close due to floating point)
        self.assertAlmostEqual(genotype_cost, phenotype_cost, places=5)

    def test_solution_collects_all_gold(self):
        """Test that solution collects all gold from all cities."""
        np.random.seed(3)
        phenotype, _ = self.solver.solution(fast=True)
        
        gold_collected = {}
        for city, gold in phenotype:
            if city != 0:  # Exclude depot
                gold_collected[city] = gold_collected.get(city, 0) + gold
        
        # Verify all gold is collected
        for city in self.solver.cities_to_visit:
            expected_gold = self.solver.graph.nodes[city].get('gold', 0)
            self.assertAlmostEqual(gold_collected.get(city, 0), expected_gold)

    def test_solution_cost_better_than_baseline(self):
        """Test that solution cost is better than or competitive with baseline."""
        np.random.seed(4)
        phenotype, solution_cost = self.solver.solution(fast=True)
        baseline_cost = self.problem.baseline()
        
        # Solution cost should be reasonable 
        self.assertLess(solution_cost, baseline_cost )

    def test_solution_multiple_runs_produce_results(self):
        """Test that solution() can be run multiple times successfully."""
        for seed in range(3):
            np.random.seed(seed)
            phenotype, cost = self.solver.solution(fast=True)
            
            self.assertTrue(self.solver.check_feasibility_phenotype(phenotype))
            self.assertFalse(math.isinf(cost))
            self.assertGreater(cost, 0)

    def test_solution_with_beta_greater_than_1(self):
        """Test solution with beta > 1 (multiple trips scenario)."""
        problem_beta_high = Problem(
            num_cities=8,
            alpha=1.0,
            beta=2.0,
            density=0.7,
            seed=99,
        )
        solver_beta_high = GA_Solver(problem_beta_high, pop_size=8, generations=3, offprint=4)
        
        np.random.seed(5)
        phenotype, cost = solver_beta_high.solution(fast=True)
        
        self.assertTrue(solver_beta_high.check_feasibility_phenotype(phenotype))
        self.assertFalse(math.isinf(cost))
        self.assertGreater(cost, 0)

    def test_solution_with_fast_flag_completes_quickly(self):
        """Test that solution with fast=True flag produces a result quickly."""
        import time
        
        np.random.seed(6)
        start_time = time.time()
        phenotype, cost = self.solver.solution(fast=True)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 10 seconds)
        self.assertLess(elapsed_time, 10.0)
        self.assertTrue(self.solver.check_feasibility_phenotype(phenotype))


if __name__ == "__main__":
    unittest.main()