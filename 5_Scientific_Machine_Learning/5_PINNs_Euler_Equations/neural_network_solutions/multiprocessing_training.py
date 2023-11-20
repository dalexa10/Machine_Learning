import multiprocessing
import itertools
from forward_PINN import train_NN_Euler_equations

# Parameters to be tested in the PINN
domain = [[0, 1], [-1.5, 3.125]]
weights = [[1, 1], [0.1, 10]]
width = [30]
hidden = [9]
activation = ['tanh']  # relu
lr = [0.0005]


iter_ls = list(itertools.product(domain, weights, width, hidden, activation, lr))

cases_dict = {}

for i, iter in enumerate(iter_ls):
    cases_dict[i] = {'domain': iter[0],
                     'weights': iter[1],
                     'width': iter[2],
                     'hidden': iter[3],
                     'activation': iter[4],
                     'lr': iter[5],
                     'n_epochs': 50000,
                     'sampling_mode': 'uniform',
                     't_points': 1000,
                     'x_points': 1000,
                     'percent_int_points': 15,
                     'bc_points': 1000
                     }

# Multiprocessing parallelization of the training
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.map(train_NN_Euler_equations, cases_dict.values())