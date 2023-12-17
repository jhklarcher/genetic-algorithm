import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])


def rosenbrock(x):
    return sum(100.0*(x[i+1] - x[i]**2.0)**2.0 + (1 - x[i])**2.0 for i in range(len(x)-1))


def ackley(x, a=20, b=0.2, c=2*np.pi):
    n = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(xi**2 for xi in x) / n))
    cos_term = -np.exp(sum(np.cos(c * xi) for xi in x) / n)
    return sum_sq_term + cos_term + a + np.exp(1)


def crossover(x1, x2, crossover_prob):
    if np.random.rand() > crossover_prob:
        return x1
    mask = np.random.rand(x1.shape[0]) < 0.5
    child = np.where(mask, x2, x1)
    return child


def mutation(x, bounds, mutation_prob):
    mask = np.random.rand(x.shape[0]) < mutation_prob
    child = np.where(mask, np.random.uniform(bounds[0], bounds[1]), x)
    return child


def ga_minimize(f, bounds, n_pop, n_gen, crossover_prob, mutation_prob):
    pop = np.random.uniform(
        low=bounds[0], high=bounds[1], size=(n_pop, bounds.shape[1]))
    fitness = [f(x) for x in pop]
    sorted_idx = np.argsort(fitness)
    print(
        f'Best of generation: {pop[sorted_idx[0]]} - {fitness[sorted_idx[0]]}')
    history = [fitness[sorted_idx[0]]]
    for g in range(n_gen):
        # Crossover
        children = np.array([
            crossover(
                pop[np.random.choice(pop.shape[0])],
                pop[np.random.choice(pop.shape[0])],
                crossover_prob=crossover_prob
            ) for _ in range(n_pop)
        ])
        # Mutation
        children = np.array([mutation(c, bounds, mutation_prob)
                            for c in children])
        pop = np.concatenate([pop, children])
        fitness = np.apply_along_axis(f, 1, pop)
        sorted_idx = np.argsort(fitness)
        history.append(fitness[sorted_idx[0]])
        print(
            f'Best of generation: {pop[sorted_idx[0]]} - {fitness[sorted_idx[0]]}')
        pop = pop[sorted_idx[:n_pop]]
        best = pop[0]
        f_best = f(best)
    return best, f_best, history


f = ackley
n_shape = 10
bounds = np.array([-10*np.ones(n_shape), 10*np.ones(n_shape)])
n_pop = 50
n_gen = 1000
crossover_prob = 0.3
mutation_prob = 0.1

best, f_best, history = ga_minimize(
    f, bounds, n_pop, n_gen, crossover_prob, mutation_prob)
print(f'Best solution: {best} - {f_best}')

plt.plot(history)
