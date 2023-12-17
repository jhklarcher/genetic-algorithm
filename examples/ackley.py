import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import ga_minimize
from genetic_algorithm.example_functions import ackley

f = ackley
n_shape = 10
bounds = np.array([-5*np.ones(n_shape), 5*np.ones(n_shape)])
n_pop = 50
n_gen = 1000
crossover_prob = 0.3
mutation_prob = 0.1

best, f_best, history = ga_minimize(
    f, bounds, n_pop, n_gen, crossover_prob, mutation_prob, verbose=True
)
print(f'Best solution: {best} - {f_best}')

plt.plot(history)
plt.show()
