import os
import numpy as np
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import pickle
from neural_network_solutions import forward_PINN
from analytic_solutions import SodShockAnalytic


def generate_plot(data_dict):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    for k in data_dict.keys():
        ax[0, 0].plot(data_dict[k]['loss_history'], label=data_dict[k]['model_data_dict']['case_name'])
    ax[0, 0].set_xlabel('Epochs')
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].legend()

    ax[0, 1].plot(x_exact, rho_exact, 'k--', label='Analytic')
    for k in data_dict.keys():
        ax[0, 1].plot(x_test.numpy(), data_dict[k]['rho_pred'], label=data_dict[k]['model_data_dict']['case_name'])
    ax[0, 1].set_xlabel(r'$x$')
    ax[0, 1].set_ylabel(r'$\rho$')
    ax[0, 1].legend()

    ax[1, 0].plot(x_exact, u_exact, 'k--', label='Analytic')
    for k in data_dict.keys():
        ax[1, 0].plot(x_test.numpy(), data_dict[k]['u_pred'], label=data_dict[k]['model_data_dict']['case_name'])
    ax[1, 0].set_xlabel(r'$x$')
    ax[1, 0].set_ylabel(r'$u$')
    ax[1, 0].legend()

    ax[1, 1].plot(x_exact, p_exact, 'k--', label='Analytic')
    for k in data_dict.keys():
        ax[1, 1].plot(x_test.numpy(), data_dict[k]['p_pred'], label=data_dict[k]['model_data_dict']['case_name'])
    ax[1, 1].set_xlabel(r'$x$')
    ax[1, 1].set_ylabel(r'$p$')
    ax[1, 1].legend()

    return fig, ax
def create_animation(x_vec, U_exact, U_numerical, U_nn):

    matplotlib.rcParams['animation.embed_limit'] = 2**28

    # Figure initialization
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
    [ax[i].set_box_aspect(1) for i in range(3)]
    plt.tight_layout()

    ax[0].set_xlabel('x')
    ax[1].set_xlabel('x')
    ax[2].set_xlabel('x')

    ax[0].set_ylabel(r'$\rho$')
    ax[1].set_ylabel(r'$u$')
    ax[2].set_ylabel(r'$p$')

    # Exact solution
    rho_ex, u_ex, p_ex = [np.array(U_exact[i, :]) for i in range(3)]

    # Numerical solution
    rho_num, u_num, p_num = [np.array(U_numerical[i, :]) for i in range(3)]

    # Neural network solution
    rho_nn, u_nn, p_nn = [np.array(U_nn[i, :]) for i in range(3)]

    # Initial solutions for density
    line_rho_ex, = ax[0].plot(x_vec, rho_ex[0, :], c='black', lw=2, clip_on=False)
    line_rho_num, = ax[0].plot(x_vec, rho_num[0, :], c='red', lw=2, clip_on=False)
    line_rho_nn, = ax[0].plot(x_vec, rho_nn[0, :], c='blue', lw=2, clip_on=False)

    # Initial solutions for flow speed
    ax[1].set_ylim([- 0.1, np.max(u_ex) + 0.1])
    line_u_ex, = ax[1].plot(x_vec, u_ex[0, :], c='black', lw=2, clip_on=False)
    line_u_num, = ax[1].plot(x_vec, u_num[0, :], c='red', lw=2, clip_on=False)
    line_u_nn, = ax[1].plot(x_vec, u_nn[0, :], c='blue', lw=2, clip_on=False)

    # Initial solutions for pressure
    line_p_ex, = ax[2].plot(x_vec, p_ex[0, :], c='black', lw=2, clip_on=False)
    line_p_num, = ax[2].plot(x_vec, p_num[0, :], c='red', lw=2, clip_on=False)
    line_p_nn, = ax[2].plot(x_vec, p_nn[0, :], c='blue', lw=2, clip_on=False)

    def init():
        pass
    def time_stepper(n):
        # Density stepper
        line_rho_ex.set_data(x_vec, rho_ex[n, :])
        line_rho_num.set_data(x_vec, rho_num[n, :])
        line_rho_nn.set_data(x_vec, rho_nn[n, :])

        # Flow speed stepper
        line_u_ex.set_data(x_vec, u_ex[n, :])
        line_u_num.set_data(x_vec, u_num[n, :])
        line_u_nn.set_data(x_vec, u_nn[n, :])

        # Pressure stepper
        line_p_ex.set_data(x_vec, p_ex[n, :])
        line_p_num.set_data(x_vec, p_num[n, :])
        line_p_nn.set_data(x_vec, p_nn[n, :])

        return (line_rho_ex, line_rho_num, line_rho_nn,
                line_u_ex, line_u_num, line_u_nn,
                line_p_ex, line_p_num, line_p_nn)

    nt_an = rho_ex.shape[0]  # Set this value as the lowest size (in time) between the vectors you want to animate
    # In this case rho_an_2.shape[0] = 59 and rho_an.shape[0] = 60

    ani = animation.FuncAnimation(fig, time_stepper,
                                  frames=nt_an, interval=300,
                                  init_func=init)

    subfolder_path = os.path.join(os.getcwd(), 'animations')
    os.makedirs(subfolder_path, exist_ok=True)

    # Get the number of animations already existing
    n_animations = len([f for f in os.listdir(subfolder_path)
                        if os.path.isfile(os.path.join(subfolder_path, f))])

    filename = 'animation_' + str(n_animations) + '.gif'
    ani.save(filename, writer='PillowWriter', fps=30)

    return ani

if __name__ == '__main__':

    # Import outcomes from training

    results_folder = 'results/'
    subfolders = [f.path for f in os.scandir(results_folder) if f.is_dir()]

    models_dict = {}

    for i, subfolder in enumerate(subfolders):
        try:
            # Load model data
            with open(os.path.join(subfolder, 'data_dict.pkl'), 'rb') as f:
                model_data_dict = pickle.load(f)

            # Load history training
            with open(os.path.join(subfolder, 'loss_history.pkl'), 'rb') as f:
                loss_history = pickle.load(f)

            model = forward_PINN.ffnn(nn_width=model_data_dict['width'],
                                      nn_hidden=model_data_dict['hidden'],
                                      activation_type=model_data_dict['activation'])
            model.load_state_dict(torch.load(os.path.join(subfolder, 'model.pth')))
            model.eval()

            models_dict[i] = {'model': model,
                              'model_data_dict': model_data_dict,
                              'loss_history': loss_history}
        except FileNotFoundError:
            print('FileNotFoundError')
            continue

    # Compute analytic solutions
    x_exact = np.linspace(0, 1, 100)
    t_exact = 0.2
    left_bc = np.array([1., 0., 1.])
    right_bc = np.array([0.125, 0., 0.1])

    U_exact = SodShockAnalytic(left_bc[0], left_bc[1], left_bc[2], right_bc[0], right_bc[1], right_bc[2], x_exact, 50,
                               t_exact, 1.4)
    rho_exact, u_exact, p_exact = U_exact[0, :], U_exact[1, :], U_exact[2, :]

    # Set device
    device = torch.device('cpu')
    x_test = torch.linspace(0, 1, 100)[:, None].to(device)
    t_test = 0.2 * torch.ones_like(x_test).to(device)
    tx_test = np.hstack((t_test, x_test))
    tx_test = torch.tensor(tx_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        for k in models_dict.keys():
            model = models_dict[k]['model']
            U_pred = model(tx_test)
            rho_pred = U_pred[:, 0].detach().cpu().numpy()
            u_pred = U_pred[:, 1].detach().cpu().numpy()
            p_pred = U_pred[:, 2].detach().cpu().numpy()

            models_dict[k]['rho_pred'] = rho_pred
            models_dict[k]['u_pred'] = u_pred
            models_dict[k]['p_pred'] = p_pred

    # Compare NN architecture
    nn_dict = {}
    for k, v in models_dict.items():
        if (models_dict[k]['model_data_dict']['activation'] == 'tanh'
                and models_dict[k]['model_data_dict']['ext_domain'] == 0
                and models_dict[k]['model_data_dict']['weights'] == 0):
            nn_dict[k] = v

    fig, ax = generate_plot(nn_dict)
    plt.tight_layout()
    plt.show()

    # Compare activation functions
    act_dict = {}
    for k, v in models_dict.items():
        if (models_dict[k]['model_data_dict']['hidden'] == 7
                and models_dict[k]['model_data_dict']['width'] == 30
                and models_dict[k]['model_data_dict']['ext_domain'] == 0
                and models_dict[k]['model_data_dict']['weights'] == 0):
            act_dict[k] = v
    fig, ax = generate_plot(act_dict)
    plt.tight_layout()
    plt.show()

    # Compare weights and extension domain
    w_ext_dict = {}
    for k, v in models_dict.items():
        if (models_dict[k]['model_data_dict']['hidden'] == 7
                and models_dict[k]['model_data_dict']['width'] == 30
                and models_dict[k]['model_data_dict']['activation'] == 'tanh'):
            w_ext_dict[k] = v
    fig, ax = generate_plot(w_ext_dict)
    plt.tight_layout()
    plt.show()
