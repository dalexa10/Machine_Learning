import os
import numpy as np
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import pickle
from neural_network_solutions import forward_PINN


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

def load_all_trained_models(device):
    """ """

    results_path = 'neural_network_solutions/training_results/'
    cases_ls = [f.path for f in os.scandir(results_path) if f.is_dir()]

    models_dict = {}

    for i, case in enumerate(cases_ls):
        try:
            # Load model metadata
            with open(os.path.join(case, 'metadata_dict.pkl'), 'rb') as f:
                model_metadata = pickle.load(f)

            # Load training history
            with open(os.path.join(case, 'loss_history.pkl'), 'rb') as f:
                loss_history = pickle.load(f)

            # Load model
            model = forward_PINN.ffnn(nn_width=model_metadata['width'],
                                      nn_hidden=model_metadata['hidden'],
                                      activation_type=model_metadata['activation'])
            model.load_state_dict(torch.load(os.path.join(case, 'model.pth'),
                                             map_location=device))
            model.eval()

            models_dict[i] = {'model': model,
                              'model_metadata': model_metadata,
                              'loss_history': loss_history,
                              'case_name': case[42:]}  # 42 is the number of characters in the path 'neural_network_solutions/training_results/'
        except FileNotFoundError:
            print('FileNotFoundError')
            continue

    return models_dict

def load_single_trained_model(model_path, device):
    """ """
    # Load model data
    with open(os.path.join(model_path, 'metadata_dict.pkl'), 'rb') as f:
        model_metadata = pickle.load(f)

    # Load history training
    with open(os.path.join(model_path, 'loss_history.pkl'), 'rb') as f:
        loss_history = pickle.load(f)

    model = forward_PINN.ffnn(nn_width=model_metadata['width'],
                              nn_hidden=model_metadata['hidden'],
                              activation_type=model_metadata['activation'])
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'),
                                     map_location=torch.device(device)))
    model.eval()

    return model, model_metadata, loss_history


def eval_nn_model(models_dict, x_test, T, device):
    """ """
    # Set device
    device = torch.device(device)
    t_test = T * np.ones_like(x_test)

    # Concatenate time and space
    tx_test = np.hstack((t_test, x_test))
    tx_test = torch.tensor(tx_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        for k in models_dict.keys():
            model = models_dict[k]['model']
            U_pred = model(tx_test)
            rho_pred = U_pred[:, 0].detach().cpu().numpy()
            u_pred = U_pred[:, 1].detach().cpu().numpy()
            p_pred = U_pred[:, 2].detach().cpu().numpy()

            models_dict[k]['x'] = x_test
            models_dict[k]['rho_pred'] = rho_pred
            models_dict[k]['u_pred'] = u_pred
            models_dict[k]['p_pred'] = p_pred

    return models_dict

def generate_plot(models_dict):
    """    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # Plot loss history
    for k in models_dict.keys():
        ax[0, 0].plot(models_dict[k]['loss_history'], label=models_dict[k]['case_name'])
    ax[0, 0].set_xlabel('Epochs')
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].set_title('Loss history')

    # Plot density
    for k in models_dict.keys():
        ax[0, 1].plot(models_dict[k]['x'], models_dict[k]['rho_pred'], label=models_dict[k]['case_name'])
    ax[0, 1].set_xlabel(r'$x$')
    ax[0, 1].set_ylabel(r'$\rho$')
    ax[0, 1].set_title('Density')

    # Plot flow speed
    for k in models_dict.keys():
        ax[1, 0].plot(models_dict[k]['x'], models_dict[k]['u_pred'], label=models_dict[k]['case_name'])
    ax[1, 0].set_xlabel(r'$x$')
    ax[1, 0].set_ylabel(r'$u$')
    ax[1, 0].set_title('Flow speed')

    # Plot pressure
    for k in models_dict.keys():
        ax[1, 1].plot(models_dict[k]['x'], models_dict[k]['p_pred'], label=models_dict[k]['case_name'])
    ax[1, 1].set_xlabel(r'$x$')
    ax[1, 1].set_ylabel(r'$p$')
    ax[1, 1].set_title('Pressure')
    # ax[1, 1].legend()

    plt.tight_layout()

    return fig, ax

def generate_comparative_plot(x_nn, models_dict, x_exact, U_exact, x_num, U_num,
                              custom_title=None, custom_legend=None):
    """    """
    # Set legend lists
    if custom_legend is not None:
        legend_list = custom_legend
    else:
        legend_list = [models_dict[k]['case_name'] for k in models_dict.keys()]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # Plot loss history
    for k in models_dict.keys():
        ax[0, 0].plot(models_dict[k]['loss_history'])
    ax[0, 0].set_xlabel('Epochs')
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].set_title('Loss history')

    # Plot density
    ax[0, 1].plot(x_exact, U_exact[0, :], 'k--')
    ax[0, 1].plot(x_num, U_num[0, :], 'r--')
    for k in models_dict.keys():
        ax[0, 1].plot(x_nn, models_dict[k]['rho_pred'])
    ax[0, 1].set_xlabel(r'$x$')
    ax[0, 1].set_ylabel(r'$\rho$')
    ax[0, 1].set_title('Density')

    # Plot flow speed
    ax[1, 0].plot(x_exact, U_exact[1, :], 'k--')
    ax[1, 0].plot(x_num, U_num[1, :], 'r--')
    for k in models_dict.keys():
        ax[1, 0].plot(x_nn, models_dict[k]['u_pred'])
    ax[1, 0].set_xlabel(r'$x$')
    ax[1, 0].set_ylabel(r'$u$')
    ax[1, 0].set_title('Flow speed')

    # Plot pressure
    ax[1, 1].plot(x_exact, U_exact[2, :], 'k--', label='Analytic solution')
    ax[1, 1].plot(x_num, U_num[2, :], 'r--', label='Numerical solution')
    for idx, k in enumerate(models_dict.keys()):
        ax[1, 1].plot(x_nn, models_dict[k]['p_pred'], label=legend_list[idx])
    ax[1, 1].set_xlabel(r'$x$')
    ax[1, 1].set_ylabel(r'$p$')
    ax[1, 1].set_title('Pressure')
    ax[1, 1].legend()

    # Custom title
    if custom_title is not None:
        fig.suptitle(custom_title, fontsize=16)

    plt.tight_layout()

    return fig, ax

def create_animation(x_vec, U_exact, U_numerical, U_nn):

    matplotlib.rcParams['animation.embed_limit'] = 2**28

    # Figure initialization
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
    [ax[i].set_box_aspect(1) for i in range(3)]

    ax[0].set_xlabel('x')
    ax[1].set_xlabel('x')
    ax[2].set_xlabel('x')

    ax[0].set_ylabel(r'$\rho$')
    ax[1].set_ylabel(r'$u$')
    ax[2].set_ylabel(r'$p$')

    ax[0].set_title('Density')
    ax[1].set_title('Flow speed')
    ax[2].set_title('Pressure')

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
    line_p_ex, = ax[2].plot(x_vec, p_ex[0, :], c='black', lw=2, clip_on=False, label='Exact')
    line_p_num, = ax[2].plot(x_vec, p_num[0, :], c='red', lw=2, clip_on=False, label='WENO 5')
    line_p_nn, = ax[2].plot(x_vec, p_nn[0, :], c='blue', lw=2, clip_on=False, label='NN')
    ax[2].legend(loc='best')

    plt.tight_layout()

    subfolder_path_fig = os.path.join(os.getcwd(), 'animations/Figures')
    os.makedirs(subfolder_path_fig, exist_ok=True)

    # Get the number of animations already existing
    n_animations = len([f for f in os.listdir(subfolder_path_fig)
                        if os.path.isfile(os.path.join(subfolder_path_fig, f))])

    plt.savefig(os.path.join(subfolder_path_fig, 'animation_' + str(n_animations) + '.png'))

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

    nt_an = rho_ex.shape[0] # Number of time steps
    ani = animation.FuncAnimation(fig, time_stepper,
                                  frames=nt_an, interval=500,
                                  init_func=init)

    subfolder_path = os.path.join(os.getcwd(), 'animations')
    os.makedirs(subfolder_path, exist_ok=True)

    # Get the number of animations already existing
    n_animations = len([f for f in os.listdir(subfolder_path)
                        if os.path.isfile(os.path.join(subfolder_path, f))])

    filename = 'animation_' + str(n_animations) + '.gif'
    ani.save(os.path.join(subfolder_path, filename), writer='PillowWriter', fps=10)

    # mov_writer = animation.FFMpegWriter(fps=10, bitrate=1800)
    # ani.save(os.path.join(subfolder_path, 'animation_' + str(n_animations) + '.mov'), writer=mov_writer)

    return ani

if __name__ == '__main__':

    T = 0.25
    # Load all trained models
    trained_models_dict = load_all_trained_models(device='cpu')
    x_test = np.linspace(0, 1, 100).reshape(-1, 1)
    trained_models_dict = eval_nn_model(trained_models_dict, x_test, T, device='cpu')

    # Filter trained models by extension domain and weights with original NN architecture
    filt_dict_1 = {}
    for k, v in trained_models_dict.items():
        if (v['model_metadata']['activation'] == 'tanh' and v['model_metadata']['percent_int_points'] == 15 and
            v['model_metadata']['width'] == 30 and v['model_metadata']['hidden'] == 7 and
            v['model_metadata']['n_epochs'] == 50000):

            filt_dict_1[k] = v
    fig, ax = generate_plot(filt_dict_1)



    # # Compute analytic solutions
    # x_exact = np.linspace(0, 1, 100)
    # t_exact = 0.2
    # left_bc = np.array([1., 0., 1.])
    # right_bc = np.array([0.125, 0., 0.1])
    #
    # U_exact = SodShockAnalytic(left_bc[0], left_bc[1], left_bc[2], right_bc[0], right_bc[1], right_bc[2], x_exact, 50,
    #                            t_exact, 1.4)
    # rho_exact, u_exact, p_exact = U_exact[0, :], U_exact[1, :], U_exact[2, :]
    #
    # # Set device
    # device = torch.device('cpu')
    # x_test = torch.linspace(0, 1, 100)[:, None].to(device)
    # t_test = 0.2 * torch.ones_like(x_test).to(device)
    # tx_test = np.hstack((t_test, x_test))
    # tx_test = torch.tensor(tx_test, dtype=torch.float32).to(device)
    #
    # with torch.no_grad():
    #     for k in models_dict.keys():
    #         model = models_dict[k]['model']
    #         U_pred = model(tx_test)
    #         rho_pred = U_pred[:, 0].detach().cpu().numpy()
    #         u_pred = U_pred[:, 1].detach().cpu().numpy()
    #         p_pred = U_pred[:, 2].detach().cpu().numpy()
    #
    #         models_dict[k]['rho_pred'] = rho_pred
    #         models_dict[k]['u_pred'] = u_pred
    #         models_dict[k]['p_pred'] = p_pred
    #
    # # Compare NN architecture
    # nn_dict = {}
    # for k, v in models_dict.items():
    #     if (models_dict[k]['model_data_dict']['activation'] == 'tanh'
    #             and models_dict[k]['model_data_dict']['ext_domain'] == 0
    #             and models_dict[k]['model_data_dict']['weights'] == 0):
    #         nn_dict[k] = v
    #
    # fig, ax = generate_plot(nn_dict)
    # plt.tight_layout()
    # plt.show()
    #
    # # Compare activation functions
    # act_dict = {}
    # for k, v in models_dict.items():
    #     if (models_dict[k]['model_data_dict']['hidden'] == 7
    #             and models_dict[k]['model_data_dict']['width'] == 30
    #             and models_dict[k]['model_data_dict']['ext_domain'] == 0
    #             and models_dict[k]['model_data_dict']['weights'] == 0):
    #         act_dict[k] = v
    # fig, ax = generate_plot(act_dict)
    # plt.tight_layout()
    # plt.show()
    #
    # # Compare weights and extension domain
    # w_ext_dict = {}
    # for k, v in models_dict.items():
    #     if (models_dict[k]['model_data_dict']['hidden'] == 7
    #             and models_dict[k]['model_data_dict']['width'] == 30
    #             and models_dict[k]['model_data_dict']['activation'] == 'tanh'):
    #         w_ext_dict[k] = v
    # fig, ax = generate_plot(w_ext_dict)
    # plt.tight_layout()
    # plt.show()
