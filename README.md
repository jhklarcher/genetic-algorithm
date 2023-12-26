# Genetic Algorithm Optimization

This project implements a Genetic Algorithm (GA) in Python, designed for solving optimization problems. The GA is capable of minimizing multi-dimensional functions through evolutionary techniques inspired by biological processes.

This is a personal project developed for educational purposes. For more mature and robust implementations of GAs, check out:

- [DEAP](https://github.com/DEAP/deap)
- [PyGAD](https://github.com/ahmedfgad/GeneticAlgorithmPython)
- PyEvolve

## Features

- **Crossover**: Combines pairs of individuals to produce offspring, ensuring diversity and exploration of the solution space.
- **Mutation**: Introduces random changes in the offspring, aiding in diversity and preventing premature convergence.
- **Tournament Selection**: Selects the best candidates as parents for the next generation, based on their fitness.
- **GA Minimize Function**: Orchestrates the GA process to minimize a given multi-dimensional function.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- (Optional) Matplotlib for any graphical representation of results

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/jhklarcher/genetic-algorithm.git
   ```
2. Navigate to the cloned directory.
3. Install the package:
   ```sh
   pip install .
   ```

### Usage

1. Import the GA function:
   ```python
   from genetic_algorithm import ga_minimize
   ```
2. Define your optimization problem as a function.
3. Call `ga_minimize` with your function and other GA parameters.

## Examples

See the [examples](examples) directory for some usage examples.

## License

Distributed under the MIT License. See `LICENSE` for more information.
