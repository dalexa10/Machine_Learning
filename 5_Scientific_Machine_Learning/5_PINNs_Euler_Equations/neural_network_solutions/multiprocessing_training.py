import multiprocessing
import itertools
from forward_PINN import train_NN_Euler_equations

# Parameters to be tested in the PINN
ext_domain = [0, 1]
weights = [0, 1]
width = [30]
hidden = [9]
activation = ['tanh', 'silu']  # relu
lr = [0.0005]

iter_ls = list(itertools.product(ext_domain, weights, width, hidden, activation, lr))

cases_dict = {}

for i, iter in enumerate(iter_ls):
    cases_dict[i] = {'ext_domain': iter[0],
                     'weights': iter[1],
                     'width': iter[2],
                     'hidden': iter[3],
                     'activation': iter[4],
                     'lr': iter[5],
                     'n_epochs': 30000}


# Multiprocessing parallelization of the training
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.map(train_NN_Euler_equations, cases_dict.values())