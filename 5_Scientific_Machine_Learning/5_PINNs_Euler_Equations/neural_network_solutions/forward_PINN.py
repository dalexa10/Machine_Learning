# Import needed modules
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import pickle
import time
def clustered_index(t_grid, x_grid, n_f_train):

    # Parameters of the line
    slope = 2/3
    intercept = -1/3

    # Calculate distance of each point in the meshgrid to the line
    dist_to_line = np.abs(t_grid - slope * x_grid - intercept)

    # Define a probability distribution with higher priority near the line
    probability = np.exp(-dist_to_line * 7)
    probability /= probability.sum()

    # Randomly sample points based on the probability distribution
    id_f = np.random.choice(np.arange(len(x_grid.flatten())), n_f_train, p=probability.flatten())

    return id_f


class ffnn(nn.Module):
    """
    Feed-forward neural network class
    """

    def __init__(self, nn_width, nn_hidden, activation_type='tanh'):
        super().__init__()
        self.first_layer = nn.Linear(2, nn_width)
        self.activation = self.set_activation_function(activation_type)

        layers = []
        for _ in range(nn_hidden):
            layers.append(nn.Linear(nn_width, nn_width))
        self.hidden_layers = nn.ModuleList(layers)

        self.last_layer = nn.Linear(nn_width, 3)

    def forward(self, tx):
        """ Input tx is a tensor of shape (n, 2) """
        u = self.activation(self.first_layer(tx))
        for hidden in self.hidden_layers:
            u = self.activation(hidden(u))
        u = self.last_layer(u)
        return u

    def set_activation_function(self, activation_type):
        """ Set the activation function for the hidden layers """
        if activation_type == 'tanh':
            activation = nn.Tanh()
        elif activation_type == 'relu':
            activation = nn.ReLU()
        elif activation_type == 'silu':
            activation = nn.SiLU()
        else:
            raise ValueError('Activation type not recognized')
        return activation

def set_initial_conditions(x):
    """ Set the initial conditions for rho, u, and p vectors
    according to the Sod shock tube problem. """

    N = len(x)

    # Initialization of vectors
    rho_ic = np.zeros((x.shape[0]))  # rho - initial condition
    u_ic = np.zeros((x.shape[0]))  # u - initial condition
    p_ic = np.zeros((x.shape[0]))  # p - initial condition

    # Set initial conditions of Sod problem
    for i in range(N):
        if x[i] <= 0.5:
            rho_ic[i] = 1.0
            p_ic[i] = 1.0
        else:
            rho_ic[i] = 0.125
            p_ic[i] = 0.1

    return rho_ic, u_ic, p_ic

def initial_condition_loss_function(model, tx, rho_ic, u_ic, p_ic):
    """
    Compute the loss for the initial conditions
    :param model: fnn instance
    :param tx:  tensor of shape (n, 2)
    :param rho_ic: tensor of shape (n, 1)
    :param u_ic: tensor of shape (n, 1)
    :param p_ic: tensor of shape (n, 1)
    :return:
        ic_loss (tensor): loss for the initial conditions
    """
    U_pred = model(tx)
    rho, u, p = U_pred[:, 0], U_pred[:, 1], U_pred[:, 2]

    # Losses for each state
    rho_loss = torch.mean((rho - rho_ic)**2)
    u_loss = torch.mean((u - u_ic)**2)
    p_loss = torch.mean((p - p_ic)**2)

    # Add up the losses for each state variable
    ic_loss = rho_loss + u_loss + p_loss

    return ic_loss

def pde_loss_function(model, tx):
    """
    Compute the loss for the PDE
    :param model: fnn instance
    :param tx: tensor of shape (n, 2)
    :return:
        pde_loss (tensor): loss for the PDE
    """
    U_pred = model(tx)
    rho, u, p = U_pred[:, 0], U_pred[:, 1], U_pred[:, 2]
    gamma = 1.4

    # Compute gradients with respec to time and space
    rho_grad = torch.autograd.grad(rho, tx, grad_outputs=torch.ones_like(rho), create_graph=True)[0]
    rho_t, rho_x = rho_grad[:, 0], rho_grad[:, 1]

    u_grad = torch.autograd.grad(u, tx, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t, u_x = u_grad[:, 0], u_grad[:, 1]

    p_grad = torch.autograd.grad(p, tx, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t, p_x = p_grad[:, 0], p_grad[:, 1]

    # Residual losses for each state (Enforcing characteristic form of Euler equations)
    rho_loss = rho_t + u * rho_x + rho * u_x
    u_loss = u_t + u * u_x + p_x / rho
    p_loss = p_t + gamma * p * u_x + u * p_x

    # Add up the losses for each state variable
    pde_loss = torch.mean(rho_loss**2) + torch.mean(u_loss**2) + torch.mean(p_loss**2)

    return pde_loss

def train_NN_Euler_equations(config_dict):

    # Start timer
    start_time = time.time()

    # Define time boundary (default)
    t_lb, t_ub = 0., 0.2

    # Define the space boundary
    x_lb, x_ub = config_dict['domain']
    if x_lb < 0:
        ext_domain_str = 1
    else:
        ext_domain_str = 0

    # Define weights for losses
    w_pde, w_ic = config_dict['weights']
    if w_pde != 1 or w_ic != 1:
        weights_str = 1
    else:
        weights_str = 0

    # Define the width and hidden layers
    width = config_dict['width']
    hidden = config_dict['hidden']

    # Define training parameters
    n_epochs = config_dict['n_epochs']
    lr = config_dict['lr']
    activation = config_dict['activation']

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model
    torch.manual_seed(23447)
    model = ffnn(nn_width=width, nn_hidden=hidden, activation_type=activation).to(device)

    # Sampling data
    nx = config_dict.get('x_points', 1000)
    nt = config_dict.get('t_points', 1000)
    n_i_train = config_dict.get('bc_points', nx)
    n_f_train = config_dict.get('percent_int_points', 15) * nx * nt // 100

    # Set the vectors
    x = np.linspace(x_lb, x_ub, nx)
    t = np.linspace(t_lb, t_ub, nt)
    t_grid, x_grid = np.meshgrid(t, x)
    T = t_grid.flatten()[:, None]
    X = x_grid.flatten()[:, None]  # Same as reshape

    # Initial condition (boundary points)
    id_ic = np.random.choice(nx, n_i_train, replace=False)  # Random sample numbering for IC
    x_ic = x_grid[id_ic, 0][:, None]
    t_ic = t_grid[id_ic, 0][:, None]
    rho_ic, u_ic, p_ic = set_initial_conditions(x_ic)
    tx_ic = np.hstack((t_ic, x_ic))

    # Convert to tensors
    tx_ic_train = torch.tensor(tx_ic, dtype=torch.float32).to(device)
    rho_ic_train = torch.tensor(rho_ic, dtype=torch.float32).to(device)
    u_ic_train = torch.tensor(u_ic, dtype=torch.float32).to(device)
    p_ic_train = torch.tensor(p_ic, dtype=torch.float32).to(device)

    # Internal points
    if config_dict['sampling_mode'] == 'uniform':
        # Random points
        id_f = np.random.choice(nx * nt, n_f_train, replace=False)
    else:
        id_f = clustered_index(t_grid, x_grid, n_f_train)

    x_int = X[id_f, 0][:, None]
    t_int = T[id_f, 0][:, None]
    tx_int = np.hstack((t_int, x_int))

    # Temporal plot to see distributions
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(x_ic, t_ic, c='r', label='IC points')
    ax.scatter(x_int, t_int, c='r', label='Internal points')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')
    ax.set_title(config_dict['sampling mode'])
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


    # Convert to tensors
    tx_int_train = torch.tensor(tx_int, requires_grad=True, dtype=torch.float32).to(device)

    # Set optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for i in range(n_epochs):

        loss_IC = initial_condition_loss_function(model, tx_ic_train, rho_ic_train, u_ic_train, p_ic_train)
        loss_PDE = pde_loss_function(model, tx_int_train)
        loss = w_ic * loss_IC + w_pde * loss_PDE

        opt.zero_grad()
        loss_history.append(loss.item())
        loss.backward()
        opt.step()

        if i % 100 == 0:
            print('Epoch: %d, LossIC: %.5f, LossPDE: %.5f, Loss: %5f' % (i, loss_IC.item(), loss_PDE.item(), loss.item()))

    # End time
    end_time = time.time()

    # Create sub folder with training_results
    subfolder_path = os.path.join(os.getcwd(), 'training_results')
    os.makedirs(subfolder_path, exist_ok=True)

    # Set the name of the subfolder for case
    case_name = ('case_' + 'ext_' + str(ext_domain_str) + '_w_' + str(weights_str) +
                 '_nn_' + '_w_' + str(width) + '_h_' + str(hidden) + '_act_' +
                 activation + '_lr_' + str(lr) + '_epo_' + str(n_epochs) +
                 '_samp_' + config_dict['sampling_mode'] + '_t_' + str(nt) +
                 '_x_' + str(nx) + '_per_' + str(config_dict['percent_int_points']) +
                 '_bc_' + str(n_i_train))

    case_path = os.path.join(subfolder_path, case_name)
    os.makedirs(case_path, exist_ok=True)

    # Store metadata of model training (dictionary) in json file
    metadata_file = os.path.join(case_path, 'case_metadata.txt')
    metadata_dict = config_dict.copy()
    metadata_dict['ext_domain_activated'] = ext_domain_str
    metadata_dict['weights_activated'] = weights_str
    metadata_dict['time'] = end_time - start_time

    # Store model metadata in json
    json_data = json.dumps(metadata_dict, indent=2)

    with open(metadata_file, 'w') as f:
        f.write(json_data)

    # Store model metadata in pickle
    with open(os.path.join(case_path, 'metadata_dict.pkl'), 'wb') as f:
        pickle.dump(metadata_dict, f)

    # Store loss history
    file_name_loss_hist = 'loss_history.pkl'

    with open(os.path.join(case_path, file_name_loss_hist), 'wb') as f:
        pickle.dump(loss_history, f)

    # Store model state
    torch.save(model.state_dict(), os.path.join(case_path, 'model.pth'))

    # Free CUDA memory
    torch.cuda.empty_cache()

    # Print elapsed time and model information
    print('-----------------------------------------')
    print('Model outcomes information:')
    print('-----------------------------------------')
    print('device: {}'.format(device))
    print('Total training time: {:.3f} seconds'.format(end_time - start_time))

    return None


if __name__ == '__main__':

    # --------------------------------------------------------
    #               Multiprocessing trainning
    # --------------------------------------------------------

    import multiprocessing

    cases_dict = {0: {'domain': [0, 1],   # [x_lb, x_ub]
                      'weights': [1, 1],
                      'width': 30,
                      'hidden': 7,
                      'n_epochs': 80000,
                      'lr': 0.0005,
                      'activation': 'tanh',
                      'sampling_mode': 'uniform',
                      't_points': 1000,
                      'x_points': 1000,
                      'percent_int_points': 11,
                      'bc_points': 1000},
                  1: {'domain': [0, 1],  # [x_lb, x_ub]
                      'weights': [1, 1],
                      'width': 30,
                      'hidden': 7,
                      'n_epochs': 100000,
                      'lr': 0.0005,
                      'activation': 'tanh',
                      'sampling_mode': 'uniform',
                      't_points': 1000,
                      'x_points': 1000,
                      'percent_int_points': 11,
                      'bc_points': 1000},
                  2: {'domain': [0, 1],  # [x_lb, x_ub]
                      'weights': [1, 1],
                      'width': 30,
                      'hidden': 7,
                      'n_epochs': 80000,
                      'lr': 0.0005,
                      'activation': 'tanh',
                      'sampling_mode': 'uniform',
                      't_points': 1000,
                      'x_points': 1000,
                      'percent_int_points': 20,
                      'bc_points': 1000},
                  3: {'domain': [0, 1],  # [x_lb, x_ub]
                      'weights': [1, 1],
                      'width': 30,
                      'hidden': 7,
                      'n_epochs': 80000,
                      'lr': 0.0005,
                      'activation': 'tanh',
                      'sampling_mode': 'uniform',
                      't_points': 1000,
                      'x_points': 1000,
                      'percent_int_points': 30,
                      'bc_points': 1000}
                  }

    # # Serialize the training
    # for case in cases_dict.values():
    #     train_NN_Euler_equations(case)

    # Parallelize the training
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(train_NN_Euler_equations, cases_dict.values())

    # --------------------------------------------------------
    #                Single case running
    # --------------------------------------------------------
    # case_dict = {'domain': [0, 1],   # [x_lb, x_ub]
    #              'weights': [1, 1],  # [wpde, wic]
    #              'width': 30,
    #              'hidden': 7,
    #              'activation': 'tanh',
    #              'lr': 0.0005,
    #              'n_epochs': 500,
    #              'sampling_mode': 'uniform',
    #              't_points': 1000,
    #              'x_points': 1000,
    #              'percent_int_points': 15,
    #              'bc_points': 1000}
    #
    # train_NN_Euler_equations(case_dict)
