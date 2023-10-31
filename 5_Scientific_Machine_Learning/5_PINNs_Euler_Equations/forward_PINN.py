# Import needed modules
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

class ffnn(nn.Module):
    """
    Feed-forward neural network class
    """

    def __init__(self, nn_width, nn_hidden):
        super().__init__()
        self.first_layer = nn.Linear(2, nn_width)

        layers = []

        for _ in range(nn_hidden):
            layers.append(nn.Linear(nn_width, nn_width))
        self.hidden_layers = nn.ModuleList(layers)

        self.last_layer = nn.Linear(nn_width, 3)


    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        activation = nn.Tanh()
        u = activation(self.first_layer(xt))
        for hidden in self.hidden_layers:
            u = activation(hidden(u))
        u = self.last_layer(u)
        return u

def set_initial_conditions(x):
    """ Set the the initial conditions for rho, u, and p vectors
    according to the Sod shock tube problem. """

    # Initialize vectors
    rho_ic = np.zeros((x.shape[0], 1))
    u_ic = np.zeros((x.shape[0], 1))
    p_ic = np.zeros((x.shape[0], 1))

    # Set initial conditions
    for i in range(x.shape[0]):
        if x[i] <= 0.5:
            rho_ic[i] = 1.0
            p_ic[i] = 1.0
        else:
            rho_ic[i] = 0.125
            p_ic[i] = 0.1

    return rho_ic, u_ic, p_ic

def initial_condition_loss_function(model, x, t, rho_ic, u_ic, p_ic):
    U_pred = model(x, t)
    rho, u, p = U_pred[:, 0:1], U_pred[:, 1:2], U_pred[:, 2:3]

    # Losses for each state
    rho_loss = torch.mean((rho - rho_ic)**2)
    u_loss = torch.mean((u - u_ic)**2)
    p_loss = torch.mean((p - p_ic)**2)

    # Add up the losses for each state variable
    ic_loss = rho_loss + u_loss + p_loss

    return ic_loss


def pde_loss_function(model, x, t):
    U_pred = model(x, t)
    rho, u, p = U_pred[:, 0:1], U_pred[:, 1:2], U_pred[:, 2:3]
    gamma = 1.4

    # Compute gradient with respect to time
    rho_t = torch.autograd.grad(rho, t, grad_outputs=torch.ones_like(rho), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    # Compute gradient with respect to space
    rho_x = torch.autograd.grad(rho, x, grad_outputs=torch.ones_like(rho), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    # Losses for each state
    rho_loss = rho_t + u * rho_x + rho * u_x
    u_loss = u_t + u * u_x + p_x / rho
    p_loss = p_t + gamma * p * u_x + u * p_x

    # Add up the losses for each state variable
    pde_loss = torch.mean(rho_loss**2) + torch.mean(u_loss**2) + torch.mean(p_loss**2)

    return pde_loss



if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model
    torch.manual_seed(23447)
    model = ffnn(30, 7).to(device)
    lr = 0.0005
    n_epochs = 10000

    # Sampling points
    nx = 1000
    nt = 1000
    n_IC = 1000
    n_i_train = 1000
    n_f_train = 11000


    x = np.linspace(-1.5, 3.125, nx)
    t = np.linspace(0, 0.2, nt)
    x_grid, t_grid = np.meshgrid(x, t)
    X = x_grid.flatten()[:, None]  # Same as reshape
    T = t_grid.flatten()[:, None]


    id_ic = np.random.choice(nx, n_i_train, replace=False)  # Random sample numbering for IC
    id_f = np.random.choice(nx * nt, n_f_train, replace=False)  # Random sample numbering for interior

    x_ic = x_grid[id_ic, 0][:, None]
    t_ic = t_grid[id_ic, 0][:, None]

    rho_ic, u_ic, p_ic = set_initial_conditions(x_ic)
    xt_ic_train = np.hstack((x_ic, t_ic))

    x_int = X[id_f, 0][:, None]
    t_int = T[id_f, 0][:, None]
    xt_int_test = np.hstack((x_int, t_int))
    xt_train = np.hstack((X, T))

    # Convert to tensors
    xt_ic_train = torch.tensor(xt_ic_train, dtype=torch.float32).to(device)
    xt_int_test = torch.tensor(xt_int_test, requires_grad=True, dtype=torch.float32).to(device)
    xt_test = torch.tensor(xt_train, requires_grad=True, dtype=torch.float32).to(device)

    rho_ic_train = torch.tensor(rho_ic, dtype=torch.float32).to(device)
    u_ic_train = torch.tensor(u_ic, dtype=torch.float32).to(device)
    p_ic_train = torch.tensor(p_ic, dtype=torch.float32).to(device)

    # Set optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for i in range(n_epochs):
        loss_IC = initial_condition_loss_function(model, xt_ic_train[:, 1:2], xt_ic_train[:, 0:1],
                                                  rho_ic_train, u_ic_train, p_ic_train)
        loss_PDE = pde_loss_function(model, xt_int_test[:, 1:2], xt_int_test[:, 0:1])

        loss = 10 * loss_IC + 0.1 * loss_PDE

        opt.zero_grad()
        loss_history.append(loss.item())
        loss.backward()
        opt.step()

        if i % 100 == 0:
            print('Epoch: %d, Loss: %.3e' % (i, loss.item()))

#%%
    # Test model
    with torch.no_grad():
        # U_pred = model(xt_test[:, 1:2], xt_test[:, 0:1])
        # rho_pred = U_pred[:, 0:1]
        # u_pred = U_pred[:, 1:2]
        # p_pred = U_pred[:, 2:3]

        U_pred = model(xt_test[:, 1:2], xt_test[:, 0:1])
        rho_pred = U_pred[:, 0:1].reshape(x_grid.shape).detach().cpu().numpy()
        u_pred = U_pred[:, 1:2].reshape(x_grid.shape).detach().cpu().numpy()
        p_pred = U_pred[:, 2:3].reshape(x_grid.shape).detach().cpu().numpy()

#%%
    # Plot results for a specific time
    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    ax[0].plot(x, rho_pred[:, 0])
    ax[0].plot(x, rho_pred[:, -1])

    ax[1].plot(x, u_pred[:, 0])
    ax[1].plot(x, u_pred[:, -1])

    ax[2].plot(x, p_pred[:, 0])
    ax[2].plot(x, p_pred[:, -1])

    plt.tight_layout()
    plt.show()

#%%
    fig, ax = plt.subplots(3, 1, figsize=(15,5))

    heatmap0 = ax[0].imshow(rho_pred, cmap='Spectral', interpolation='nearest', aspect='auto', extent=[0, 0.2, 0, 1])
    fig.colorbar(heatmap0, ax=ax[0])
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$x$')
    ax[0].set_title(r'$\rho (x,t)$')

    heatmap1 = ax[1].imshow(u_pred, cmap='Spectral', interpolation='nearest', aspect='auto', extent=[0, 0.2, 0, 1])
    fig.colorbar(heatmap1, ax=ax[1])
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$x$')
    ax[1].set_title(r'$u(x,t)$')

    heatmap2 = ax[2].imshow(p_pred, cmap='Spectral', interpolation='nearest', aspect='auto', extent=[0, 0.2, 0, 1])
    fig.colorbar(heatmap1, ax=ax[2])
    ax[2].set_xlabel(r'$t$')
    ax[2].set_ylabel(r'$x$')
    ax[2].set_title(r'$p(x,t)$')



    plt.tight_layout()
    plt.show()




#%%
    # # Plot results
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # ax[0].scatter(xt_test[:, 1].cpu().detach().numpy(), rho_pred.cpu().detach().numpy(), label='Prediction')
    # ax[0].set_xlabel('x')
    # ax[0].set_ylabel('rho')
    #
    # ax[1].scatter(xt_test[:, 1].cpu().detach().numpy(), u_pred.cpu().detach().numpy(), label='Prediction')
    # ax[1].set_xlabel('x')
    # ax[1].set_ylabel('u')
    #
    # ax[2].scatter(xt_test[:, 1].cpu().detach().numpy(), p_pred.cpu().detach().numpy(), label='Prediction')
    # ax[2].set_xlabel('x')
    # ax[2].set_ylabel('p')
    #
    # plt.tight_layout()
    # plt.show()

    # Save the model
    # torch.save(model.state_dict(), 'model.pth')

