# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import time
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator
from bingo.symbolic_regression.explicit_regression import ExplicitRegression, ExplicitTrainingData

from bingo.evolutionary_algorithms.deterministic_crowding import DeterministicCrowdingEA
from bingo.evolutionary_optimizers.parallel_archipelago import ParallelArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.fitness_predictor_island import FitnessPredictorIsland
from bingo.local_optimizers.continuous_local_opt import ContinuousLocalOptimization
from bingo.stats.pareto_front import ParetoFront

POP_SIZE = 100
STACK_SIZE = 256
MAX_GENERATIONS = 20000
FITNESS_THRESHOLD = 1e-4
CHECK_FREQUENCY = 1000
MIN_GENERATIONS = 100
CROSSOVER_PROBABILITY = 0.6
MUTATION_PROBABILITY = 0.3

def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()

def print_pareto_front(hall_of_fame):
    print("  FITNESS    COMPLEXITY    EQUATION")

    for member in hall_of_fame:
        print('%.3e    ' % member.fitness, member.get_complexity(),
                '   f(X_0) =', member)

def plot_pareto_front(hall_of_fame):
    fitness_vals = []
    complexity_vals = []
    for member in hall_of_fame:
        fitness_vals.append(member.fitness)
        complexity_vals.append(member.get_complexity())
    plt.figure()
    plt.step(complexity_vals, fitness_vals, 'k', where='post')
    plt.plot(complexity_vals, fitness_vals, 'or')
    plt.xlabel('Complexity')
    plt.ylabel('Fitness')
    plt.savefig('pareto_front')

def execute_generational_steps():
    communicator = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()

    x = None
    y = None

    if rank == 0:

        df = pd.read_csv('data/combined_clean_data.csv')
        df = df.dropna()

        train, test = train_test_split(df, test_size = 0.2, random_state=42)

        columns = df.columns
        x = train.loc[:, ~columns.str.contains('Damage')]
        x = x.loc[:, x.columns != 'Time']
        x = x.loc[:, x.columns != 'Machine'].values

        y = train.loc[:, columns.str.contains('Damage')]
        y = y.iloc[:, 0].values.reshape((-1,1)) 

    x = MPI.COMM_WORLD.bcast(x, root=0)
    y = MPI.COMM_WORLD.bcast(y, root=0)

    training_data = ExplicitTrainingData(x, y)

    component_generator = ComponentGenerator(x.shape[1])
    component_generator.add_operator(2) # +
    component_generator.add_operator(3) # -
    component_generator.add_operator(4) # *
    component_generator.add_operator(5) # /
#    component_generator.add_operator(6) # sin
#    component_generator.add_operator(7) # cos
#    component_generator.add_operator(8) # exponential
#    component_generator.add_operator(10) # power
#    component_generator.add_operator(12) # sqrt

    crossover = AGraphCrossover(component_generator)
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator)

    fitness = ExplicitRegression(training_data=training_data, metric='mean squared error')
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')
    evaluator = Evaluation(local_opt_fitness)

    ea = DeterministicCrowdingEA(evaluator, crossover, mutation, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY)

    island = FitnessPredictorIsland(ea, agraph_generator, POP_SIZE, predictor_size_ratio=0.2)

    pareto_front = ParetoFront(secondary_key = lambda ag: ag.get_complexity(),
            similarity_function=agraph_similarity)

    archipelago = ParallelArchipelago(island, hall_of_fame=pareto_front)

    optim_result = archipelago.evolve_until_convergence(MAX_GENERATIONS, FITNESS_THRESHOLD,
            convergence_check_frequency=CHECK_FREQUENCY, min_generations=MIN_GENERATIONS,
            checkpoint_base_name='checkpoint', num_checkpoints=2)

    if optim_result.success:
        if rank == 0:
            print("best: ", archipelago.get_best_individual())

    if rank == 0:
        print(optim_result)
        print("Generation: ", archipelago.generational_age)
        print_pareto_front(pareto_front)
        plot_pareto_front(pareto_front)

def main():
    execute_generational_steps()

if __name__ == '__main__':
    main()
