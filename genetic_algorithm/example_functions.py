import numpy as np

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
