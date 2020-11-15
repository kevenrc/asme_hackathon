import pandas as pd
import numpy as np
import sys

from bingo.evolutionary_optimizers.parallel_archipelago import load_parallel_archipelago_from_file



if __name__ == "__main__":

    checkpoint_num = sys.argv[1]
    pickle_file = "".join(['checkpoint_', str(checkpoint_num), '.pkl'])
    pickle = load_parallel_archipelago_from_file(pickle_file)

    for individual in pickle.hall_of_fame:

        equation = individual.get_console_string()
        print(individual.get_complexity(), ': ', equation)
        print('Fitness: ', individual.fitness)

