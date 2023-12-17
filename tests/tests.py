import unittest
import numpy as np
from genetic_algorithm.genetic_algorithm import crossover, mutation, tournament_selection, ga_minimize


class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)  # for reproducible results

    def test_crossover(self):
        # Setup code for crossover test
        x1 = np.array([1, 2, 3, 4, 5])
        x2 = np.array([5, 4, 3, 2, 1])
        child = crossover(x1, x2, 0.5)
        # Assert statements to check crossover functionality
        self.assertEqual(child.shape, x1.shape)

    def test_mutation(self):
        # Setup code for mutation test
        x = np.array([1, 2, 3, 4, 5])
        bounds = np.array([-5*np.ones(x.shape), 5*np.ones(x.shape)])
        child = mutation(x, bounds, 0.5)
        # Assert statements to check mutation functionality
        self.assertEqual(child.shape, x.shape)

    def test_tournament_selection(self):
        # Setup code for tournament selection test
        pop = np.array([
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2],
            [3, 4, 5, 6, 7]
        ])
        fitness = np.array([1, 2, 3, 4, 5])
        parents = tournament_selection(pop, fitness, 1)
        # Assert statements to check tournament selection functionality
        self.assertEqual(parents.shape, (1, 5))
        # Asser that the parent is among the population
        self.assertTrue(np.any(np.all(parents == pop, axis=1)))

    def test_ga_minimize(self):
        # Setup code for ga_minimize test
        def f(x):
            return np.sum(x**2)
        bounds = np.array([-5*np.ones(2), 5*np.ones(2)])
        n_pop = 10
        n_gen = 10
        crossover_prob = 0.4
        mutation_prob = 0.1
        best, f_best, history = ga_minimize(f, bounds, n_pop, n_gen,
                                            crossover_prob, mutation_prob,
                                            verbose=False)
        # Assert statements to check ga_minimize functionality
        self.assertEqual(len(history), n_gen+1)
        self.assertEqual(history[-1], f_best)
        self.assertTrue(
            np.all(best >= bounds[0]) and np.all(best <= bounds[1]))
        self.assertTrue(f_best == np.min(history))
        self.assertTrue(best.shape == (2,))


if __name__ == '__main__':
    unittest.main()
