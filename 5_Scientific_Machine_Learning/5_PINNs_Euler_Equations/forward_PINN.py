# Import needed modules
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import time

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

    # Unpack config_dict
    ext_domain = config_dict['ext_domain']
    weights = config_dict['weights']
    width = config_dict['width']
    hidden = config_dict['hidden']
    n_epochs = config_dict['n_epochs']
    lr = config_dict['lr']
    activation = config_dict['activation']

    t_lb, t_ub = 0., 0.2

    # Set domain bounds
    if ext_domain == 0:
        x_lb, x_ub = 0., 1.
        ext_domain_str = ''
    else:
        x_lb, x_ub = -1.5, 3.125
        ext_domain_str = '_ext'

    # Set weights (activated or not)
    if weights == 0:
        w_pde, w_ic = 1., 1.
        weights_str = ''
    else:
        w_pde, w_ic = 0.1, 10
        weights_str = '_w'

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model
    torch.manual_seed(23447)
    model = ffnn(nn_width=width, nn_hidden=hidden, ).to(device)

    # Sampling points
    nx = 1000
    nt = 1000
    n_i_train = 1000
    n_f_train = 11000

    # Set the vectors
    x = np.linspace(x_lb, x_ub, nx)
    t = np.linspace(t_lb, t_ub, nt)
    t_grid, x_grid = np.meshgrid(t, x)
    T = t_grid.flatten()[:, None]
    X = x_grid.flatten()[:, None]  # Same as reshape

    id_ic = np.random.choice(nx, n_i_train, replace=False)  # Random sample numbering for IC
    id_f = np.random.choice(nx * nt, n_f_train, replace=False)  # Random sample numbering for interior

    # Initial condition points
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
    x_int = X[id_f, 0][:, None]
    t_int = T[id_f, 0][:, None]
    tx_int = np.hstack((t_int, x_int))

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

    # Create subfolder with results
    subfolder_name = ('output' + ext_domain_str + weights_str + '_nn_' + str(width) + '_' + str(hidden)
                      + '_' + activation + '_lr_' + str(lr) + '_epochs_' + str(n_epochs))
    subfolder_path = os.path.join(os.getcwd(), 'results', subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Store loss history
    file_name_loss_hist = 'loss_history.pkl'

    with open(os.path.join(subfolder_path, file_name_loss_hist), 'wb') as f:
        pickle.dump(loss_history, f)

    # Store model state
    torch.save(model.state_dict(), os.path.join(subfolder_path, 'model.pth'))

    # End time
    end_time = time.time()

    # Print elapsed time and model information

    print('-----------------------------------------')
    print('Model outcomes information:')
    print('-----------------------------------------')
    print('device: {}'.format(device))
    print('Elapsed time: {:.3f} seconds'.format(end_time - start_time))

    return None


if __name__ == '__main__':
    import multiprocessing

    cases_dict = {0: {'ext_domain': 0,
                      'weights': 0,
                      'width': 30,
                      'hidden': 7,
                      'n_epochs': 20000,
                      'lr': 0.0005,
                      'activation': 'tanh'},
                  1: {'ext_domain': 1,
                      'weights': 0,
                      'width': 30,
                      'hidden': 7,
                      'n_epochs': 20000,
                      'lr': 0.0005,
                      'activation': 'tanh'},
                  2: {'ext_domain': 0,
                      'weights': 1,
                      'width': 30,
                      'hidden': 7,
                      'n_epochs': 20000,
                      'lr': 0.0005,
                      'activation': 'tanh'},
                  3: {'ext_domain': 1,
                      'weights': 1,
                      'width': 30,
                      'hidden': 7,
                      'n_epochs': 20000,
                      'lr': 0.0005,
                      'activation': 'tanh'}}

    # Parallelize the training
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(train_NN_Euler_equations, cases_dict.values())





