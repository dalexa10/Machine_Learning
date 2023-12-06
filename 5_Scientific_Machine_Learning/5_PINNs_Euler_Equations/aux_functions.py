import os
import numpy as np
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import pickle
from sklearn.metrics import mean_squared_error as MSE
from neural_network_solutions import forward_PINN


def load_single_trained_model(model_path, device):
    """ """

    model_dict = {}

    try:
        # Load model data
        with open(os.path.join(model_path, 'metadata_dict.pkl'), 'rb') as f:
            model_metadata = pickle.load(f)

        # Load total training loss history
        with open(os.path.join(model_path, 'loss_history.pkl'), 'rb') as f:
            loss_history = pickle.load(f)

        model = forward_PINN.ffnn(nn_width=model_metadata['width'],
                                  nn_hidden=model_metadata['hidden'],
                                  activation_type=model_metadata['activation']).to(device)
        model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location=device))
        model.eval()

        # Load IC  (if available)
        try:
            with open(os.path.join(model_path, 'loss_IC.pkl'), 'rb') as f:
                loss_history_ic = pickle.load(f)
        except FileNotFoundError:
            loss_history_ic = None

        # Load BC  (if available)
        try:
            with open(os.path.join(model_path, 'loss_PDE.pkl'), 'rb') as f:
                loss_history_PDE = pickle.load(f)
        except FileNotFoundError:
            loss_history_PDE = None

        model_dict[0] = {'model': model,
                        'model_metadata': model_metadata,
                        'loss_history': loss_history,
                        'loss_history_ic': loss_history_ic,
                        'loss_history_PDE': loss_history_PDE,
                        'case_name': model_path[42:]}

    except FileNotFoundError:
        print('FileNotFoundError')

    return model_dict


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
                               'loss_history_ic': None,
                               'loss_history_PDE': None,
                              'case_name': case[42:]}  # 42 is the number of characters in the path 'neural_network_solutions/training_results/'
        except FileNotFoundError:
            print('FileNotFoundError')
            continue

    return models_dict

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
                              custom_title=None, custom_legend=None, legend_loss=False):
    """    """
    # Set legend lists
    if custom_legend is not None:
        legend_list = custom_legend
    else:
        legend_list = ['Case_' + str(i) for i in models_dict.keys()]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # Plot loss history
    for k in models_dict.keys():
        ax[0, 0].plot(models_dict[k]['loss_history'], alpha=0.75, label='Total loss')

        if models_dict[k]['loss_history_ic'] is not None:
            ax[0, 0].plot(models_dict[k]['loss_history_ic'], alpha=0.9, label='Loss Initial Conditions')

        if models_dict[k]['loss_history_PDE'] is not None:
            ax[0, 0].plot(models_dict[k]['loss_history_PDE'], alpha=0.75, label='Loss Physical Laws')

    ax[0, 0].set_xlabel('Epochs', fontsize=14)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=12)
    ax[0, 0].ticklabel_format(style='sci', axis='x')
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_ylabel('Loss', fontsize=14)
    ax[0, 0].set_title('Loss history', fontsize=14)
    if legend_loss:
        ax[0, 0].legend(fontsize=12)

    # Plot density
    ax[0, 1].plot(x_exact, U_exact[0, :], 'k--')
    ax[0, 1].plot(x_num, U_num[0, :], 'r--')
    for k in models_dict.keys():
        ax[0, 1].plot(x_nn, models_dict[k]['rho_pred'])
    ax[0, 1].tick_params(axis='both', which='major', labelsize=12)
    ax[0, 1].set_xlabel(r'$x$', fontsize=14)
    ax[0, 1].set_ylabel(r'$\rho$', fontsize=14)
    ax[0, 1].set_title('Density', fontsize=14)

    # Plot flow speed
    ax[1, 0].plot(x_exact, U_exact[1, :], 'k--')
    ax[1, 0].plot(x_num, U_num[1, :], 'r--')
    for k in models_dict.keys():
        ax[1, 0].plot(x_nn, models_dict[k]['u_pred'])
    ax[1, 0].tick_params(axis='both', which='major', labelsize=12)
    ax[1, 0].set_xlabel(r'$x$', fontsize=14)
    ax[1, 0].set_ylabel(r'$u$', fontsize=14)
    ax[1, 0].set_title('Flow speed', fontsize=14)

    # Plot pressure
    ax[1, 1].plot(x_exact, U_exact[2, :], 'k--', label='Analytic solution')
    ax[1, 1].plot(x_num, U_num[2, :], 'r--', label='Numerical solution')
    for idx, k in enumerate(models_dict.keys()):
        ax[1, 1].plot(x_nn, models_dict[k]['p_pred'], label=legend_list[idx])
    ax[1, 1].tick_params(axis='both', which='major', labelsize=12)
    ax[1, 1].set_xlabel(r'$x$', fontsize=14)
    ax[1, 1].set_ylabel(r'$p$', fontsize=14)
    ax[1, 1].set_title('Pressure', fontsize=14)
    ax[1, 1].legend(fontsize=12)

    # Custom title
    if custom_title is not None:
        fig.suptitle(custom_title, fontsize=16)

    plt.tight_layout()

    return fig, ax

def generate_colormesh_plot(analytic_model, nn_model):
    """ """
    x = np.linspace(0, 1, 1000)
    t = np.linspace(0, 0.2, 1000)
    T, X = np.meshgrid(t, x)
    tx = np.hstack((T.reshape(-1, 1), X.reshape(-1, 1)))

    # Analytic solution
    U_exact = np.zeros((3, X.shape[0], X.shape[1]))

    for j in range(X.shape[1]):
        U_exact[:, :, j] = analytic_model(x, t[j])

    # Neural network solution
    tx = torch.tensor(tx, dtype=torch.float32)
    U_nn = nn_model(tx).detach().numpy()

    # Plot density
    fig, ax = plt.subplots(3, 3, figsize=(12, 8))

    pc_den_ex = ax[0, 0].pcolormesh(T, X, U_exact[0, :, :], cmap='jet')
    ax[0, 0].tick_params(axis='both', which='major', labelsize=12)
    ax[0, 0].set_xlabel('t', fontsize=14)
    ax[0, 0].set_ylabel('x', fontsize=14)
    ax[0, 0].set_title('Density (Analytic)', fontsize=14)
    fig.colorbar(pc_den_ex, ax=ax[0, 0])

    pc_den_nn = ax[1, 0].pcolormesh(T, X, U_nn[:, 0].reshape(X.shape), cmap='jet')
    ax[1, 0].tick_params(axis='both', which='major', labelsize=12)
    ax[1, 0].set_xlabel('t', fontsize=14)
    ax[1, 0].set_ylabel('x', fontsize=14)
    ax[1, 0].set_title('Density (NN)', fontsize=14)
    fig.colorbar(pc_den_nn, ax=ax[1, 0])

    pc_den_err = ax[2, 0].pcolormesh(T, X, np.abs(U_exact[0, :, :].reshape(X.shape) -
                                                  U_nn[:, 0].reshape(X.shape)) * 100 / U_exact[0, :, :].reshape(X.shape), cmap='jet')
    ax[2, 0].tick_params(axis='both', which='major', labelsize=12)
    ax[2, 0].set_xlabel('t', fontsize=14)
    ax[2, 0].set_ylabel('x', fontsize=14)
    ax[2, 0].set_title('Relative Error - Density', fontsize=14)
    fig.colorbar(pc_den_err, ax=ax[2, 0])

    # Plot flow speed
    pc_sp_ex = ax[0, 1].pcolormesh(T, X, U_exact[1, :, :].reshape(X.shape), cmap='jet')
    ax[0, 1].tick_params(axis='both', which='major', labelsize=12)
    ax[0, 1].set_xlabel('t', fontsize=14)
    ax[0, 1].set_ylabel('x', fontsize=14)
    ax[0, 1].set_title('Flow speed (Analytic)', fontsize=14)
    fig.colorbar(pc_sp_ex, ax=ax[0, 1])

    pc_sp_nn = ax[1, 1].pcolormesh(T, X, U_nn[:, 1].reshape(X.shape), cmap='jet')
    ax[1, 1].tick_params(axis='both', which='major', labelsize=12)
    ax[1, 1].set_xlabel('t', fontsize=14)
    ax[1, 1].set_ylabel('x', fontsize=14)
    ax[1, 1].set_title('Flow speed (NN)', fontsize=14)
    fig.colorbar(pc_sp_nn, ax=ax[1, 1])

    pc_sp_err = ax[2, 1].pcolormesh(T, X, np.abs(U_exact[1, :, :].reshape(X.shape) -
                                                  U_nn[:, 1].reshape(X.shape)) * 100 / (U_exact[1, :, :] +1e-12) .reshape(X.shape)
                                    , cmap='jet')
    ax[2, 1].tick_params(axis='both', which='major', labelsize=12)
    ax[2, 1].set_xlabel('t', fontsize=14)
    ax[2, 1].set_ylabel('x', fontsize=14)
    ax[2, 1].set_title('Relative Error - Speed', fontsize=14)
    fig.colorbar(pc_sp_err, ax=ax[2, 1])

    # Plot pressure
    pc_p_ex = ax[0, 2].pcolormesh(T, X, U_exact[2, :, :].reshape(X.shape), cmap='jet')
    ax[0, 2].tick_params(axis='both', which='major', labelsize=12)
    ax[0, 2].set_xlabel('t', fontsize=14)
    ax[0, 2].set_ylabel('x', fontsize=14)
    ax[0, 2].set_title('Pressure (Analytic)', fontsize=14)
    fig.colorbar(pc_p_ex, ax=ax[0, 2])

    pc_p_nn = ax[1, 2].pcolormesh(T, X, U_nn[:, 2].reshape(X.shape), cmap='jet')
    ax[1, 2].tick_params(axis='both', which='major', labelsize=12)
    ax[1, 2].set_xlabel('t', fontsize=14)
    ax[1, 2].set_ylabel('x', fontsize=14)
    ax[1, 2].set_title('Pressure (NN)', fontsize=14)
    fig.colorbar(pc_p_nn, ax=ax[1, 2])

    pc_p_err = ax[2, 2].pcolormesh(T, X, np.abs(U_exact[2, :, :].reshape(X.shape) -
                                                  U_nn[:, 2].reshape(X.shape)) * 100 / U_exact[2, :, :].reshape(X.shape), cmap='jet')
    ax[2, 2].tick_params(axis='both', which='major', labelsize=12)
    ax[2, 2].set_xlabel('t', fontsize=14)
    ax[2, 2].set_ylabel('x', fontsize=14)
    ax[2, 2].set_title('Relative Error - Pressure', fontsize=14)
    fig.colorbar(pc_p_err, ax=ax[2, 2])

    plt.tight_layout()

    return fig, ax

def compute_error_at_timesteps(U_exact, U_nn, U_num):
    """ """
    # Exact solution
    rho_ex, u_ex, p_ex = [np.array(U_exact[i, :]) for i in range(3)]

    # Numerical solution
    rho_num, u_num, p_num = [np.array(U_num[i, :]) for i in range(3)]

    # Neural network solution
    rho_nn, u_nn, p_nn = [np.array(U_nn[i, :]) for i in range(3)]

    # Density error
    rho_err_nn = [MSE(rho_ex[i, :], rho_nn[i, :]) for i in range(rho_ex.shape[0])]
    rho_err_num = [MSE(rho_ex[i, :], rho_num[i, :]) for i in range(rho_ex.shape[0])]

    # Flow speed error
    u_err_nn = [MSE(u_ex[i, :], u_nn[i, :]) for i in range(u_ex.shape[0])]
    u_err_num = [MSE(u_ex[i, :], u_num[i, :]) for i in range(u_ex.shape[0])]

    # Pressure error
    p_err_nn = [MSE(p_ex[i, :], p_nn[i, :]) for i in range(p_ex.shape[0])]
    p_err_num = [MSE(p_ex[i, :], p_num[i, :]) for i in range(p_ex.shape[0])]

    return [[rho_err_nn, u_err_nn, p_err_nn], [rho_err_num, u_err_num, p_err_num]]

def generate_error_plot(t_vec, error_data_list, custom_legend=None):

    if custom_legend is not None:
        legend_list = custom_legend
    else:
        legend_list = []

    fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))

    for i, error_data in enumerate(error_data_list):
        rho_err_nn, u_err_nn, p_err_nn = error_data[0]
        if i == 0:
            rho_err_num, u_err_num, p_err_num = error_data[1]
            ax[0].plot(t_vec, rho_err_num)
            ax[1].plot(t_vec, u_err_num)
            ax[2].plot(t_vec, p_err_num, label='Numerical')

        # Plot density error
        ax[0].plot(t_vec, rho_err_nn)

        # Plot flow speed error
        ax[1].plot(t_vec, u_err_nn)

        # Plot pressure error
        ax[2].plot(t_vec, p_err_nn, label='Case_' + str(i) + ' Neural network')

    ax[0].set_xlabel('Time step', fontsize=14)
    ax[0].set_ylabel('MSE', fontsize=14)
    ax[0].set_title('Density error', fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=12)

    ax[1].set_xlabel('Time step', fontsize=14)
    ax[1].set_ylabel('MSE', fontsize=14)
    ax[1].set_title('Flow speed error', fontsize=14)
    ax[1].tick_params(axis='both', which='major', labelsize=12)

    ax[2].set_xlabel('Time step', fontsize=14)
    ax[2].set_ylabel('MSE', fontsize=14)
    ax[2].set_title('Pressure error', fontsize=14)
    ax[2].tick_params(axis='both', which='major', labelsize=12)
    ax[2].legend(fontsize=12)

    plt.tight_layout()

    return fig, ax


def create_animation(x_vec, U_exact, U_numerical, U_nn):

    matplotlib.rcParams['animation.embed_limit'] = 2**28

    # Figure initialization
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
    [ax[i].set_box_aspect(1) for i in range(3)]

    ax[0].set_xlabel('x', fontsize=14)
    ax[1].set_xlabel('x', fontsize=14)
    ax[2].set_xlabel('x', fontsize=14)

    ax[0].set_ylabel(r'$\rho$', fontsize=14)
    ax[1].set_ylabel(r'$u$', fontsize=14)
    ax[2].set_ylabel(r'$p$', fontsize=14)

    ax[0].set_title('Density', fontsize=14)
    ax[1].set_title('Flow speed', fontsize=14)
    ax[2].set_title('Pressure', fontsize=14)

    ax[0].tick_params(axis='both', which='major', labelsize=12)
    ax[1].tick_params(axis='both', which='major', labelsize=12)
    ax[2].tick_params(axis='both', which='major', labelsize=12)

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


    # fig, ax = generate_comparative_plot(filt_dict_1)



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
