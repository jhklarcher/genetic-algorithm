import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

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


def tournament_selection(pop, fitness, n_parents):
    parents = np.empty((n_parents, pop.shape[1]))
    for i in range(n_parents):
        idx = np.random.choice(np.arange(pop.shape[0]), size=3, replace=False)
        parents[i] = pop[idx[np.argmin(fitness[idx])]]
    return parents


def ga_minimize(f, bounds, n_pop, n_gen,
                crossover_prob, mutation_prob,
                verbose = False):
    pop = np.random.uniform(
        low=bounds[0],
        high=bounds[1],
        size=(n_pop, bounds.shape[1])
    )
    fitness = np.apply_along_axis(f, 1, pop)
    sorted_idx = np.argsort(fitness)
    if verbose:
        print(
            f'Best of generation: {pop[sorted_idx[0]]} - {fitness[sorted_idx[0]]}'
        )
    history = [fitness[sorted_idx[0]]]
    for g in range(n_gen):
        # Crossover
        children = tournament_selection(pop, fitness, n_parents=n_pop)
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
        if verbose:
            print(
                f'Best of generation: {pop[sorted_idx[0]]} - {fitness[sorted_idx[0]]}'
            )
        pop = pop[sorted_idx[:n_pop]]
        best = pop[0]
        f_best = f(best)
    return best, f_best, history
